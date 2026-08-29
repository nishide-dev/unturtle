# Copyright 2025-present nishide-dev & the Unturtle team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FLM/FMLM sampling entries (#155 Stage 2/3) — Unturtle ADAPTATION.

Two SEPARATE loops mirroring the oracle line-for-line (the issue's headline
mutation target is FMLM dispatched through the FLM solver — structurally
impossible here because the loops live in different functions with
different model-call signatures):

- :func:`run_flm_request`  — `FLM.generate_samples` (algo.py:1054-1083):
  Euler ODE on the deterministic tau-linspace grid, one-time conditioning,
  final step jumps to the prediction, endpoint argmax;
- :func:`run_fmlm_request` — `FMLM.generate_samples` (algo.py:1516-1566):
  TWO-time flow-map composition with gamma churn, final step jumps to
  D_st, endpoint argmax.

Shared frozen semantics: NFE = steps (one denoiser forward per step);
executed grid recorded verbatim; global-RNG seeding scoped inside
torch.random.fork_rng (the #153 review's F5 lesson applied from day one);
CPU runs route the oracle's `use_jvp_attn=True` pure-torch attention path
(the flash kernels are CUDA-only — Stage-0 two-tier plan).
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from unturtle_flm.state_update import apply_state_update

# --- private profiling seam (#166 Stage 1) -----------------------------------
#
# DEFAULT OFF. `_OBSERVER` is None in every normal run, so the only cost on the
# production path is one `is not None` test per event boundary — no allocation,
# no state mutation, no branch whose RESULT differs.
#
# The observer receives an event NAME and a phase ("enter"/"exit") and nothing
# else: never `z`, never the model output. Handing it a tensor would invite a
# fingerprint, a `.item()` or a `.cpu()` inside the timed window, and any of
# those would change what is being measured.
#
# Why the terminal RNG state is captured INSIDE `fork_rng`: the sampling loops
# are wrapped in `torch.random.fork_rng`, which RESTORES the outer generator on
# exit. Verified — an extra `randn` drawn inside the fork leaves the outer state
# bit-identical, so an outer pre/post fingerprint cannot detect an observer that
# perturbs the internal stream. The check has to read the state before the fork
# closes or it proves nothing.
#: Observer state, scoped by EXECUTION CONTEXT. Independently established
#: contexts do not leak events, and nested installation restores via ContextVar
#: tokens.
#:
#: A plain module global did leak — MEASURED: with an observer installed on the
#: profiling thread, 26 events from an unrelated thread's ordinary
#: `run_fmlm_request` were captured.
#:
#: NOT claimed: that no child task can ever see the observer. A task created
#: INSIDE an observed context inherits the context, so it inherits the observer
#: — verified. That is standard ContextVar copy-on-spawn semantics, not a leak,
#: and it is irrelevant to the profiler, which spawns nothing and runs one
#: request at a time.
_OBSERVER_CONTEXT: ContextVar[Any] = ContextVar("_FMLM_OBSERVER", default=None)

#: Frozen event boundaries. `flow_map_forward` spans the double-time model call
#: AND the `.exp()` that follows it — the name alone would suggest only the
#: model body.
_FMLM_EVENTS = (
    "grid_init",
    "time_schedule",
    "flow_map_forward",
    "state_update",
    "endpoint_decode",
)


class _Scope:
    """Enter/exit notifier. Instantiated ONLY when an observer is installed."""

    __slots__ = ("_name", "_observer")

    def __init__(self, observer, name: str) -> None:
        self._observer = observer
        self._name = name

    def __enter__(self):
        self._observer(self._name, "enter")
        return self

    def __exit__(self, *_exc) -> bool:
        # Always fires, so a raising body cannot leave a scope unclosed and make
        # the next window absorb this one's time.
        self._observer(self._name, "exit")
        return False


class _Off:
    """Zero-work stand-in used when no observer is installed."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


_OFF = _Off()


def _scope(observer, name: str):
    """The hot path is a single `is not None` test; the observer is resolved
    ONCE per request at the function entry, not re-read per boundary."""
    return _Scope(observer, name) if observer is not None else _OFF


def _install_observer(callback):
    """Install an observer in THIS execution context. PRIVATE — profiling only.

    Returns the token needed to restore the previous value. Nesting restores as
    a stack via the token, so a nested install cannot clobber an outer one.
    """
    return _OBSERVER_CONTEXT.set(callback)


def _restore_observer(token) -> None:
    _OBSERVER_CONTEXT.reset(token)


#: Private diagnostic flags, read from a PRIVATE attribute on the request
#: object rather than from `request.kwargs` — the documented kwargs surface is
#: unchanged and `_common` never sees these.
_DIAGNOSTIC_ATTR = "_unturtle_profile_diagnostics"


def _diagnostics(request: Any) -> frozenset[str]:
    value = getattr(request, _DIAGNOSTIC_ATTR, None)
    return frozenset(value) if value else frozenset()


def _capture_terminal_rng(request: Any) -> bool:
    return "terminal_rng" in _diagnostics(request)


def _capture_final_latent(request: Any) -> bool:
    return "final_latent" in _diagnostics(request)


def _capture_rng_state(device) -> dict[str, Any]:
    """Terminal RNG state, read INSIDE `fork_rng` (see the note above)."""
    import torch

    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if device is not None and device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _common(model: Any, request: Any, *, default_steps: int) -> dict[str, Any]:
    kwargs = dict(getattr(request, "kwargs", None) or {})
    return {
        "steps": int(kwargs.get("steps", default_steps)),
        "num_samples": int(kwargs.get("num_samples", 1)),
        "seed": int(kwargs.get("seed", 1)),  # official eval default seed=1
        "gamma": float(kwargs.get("gamma", 0.0)),
    }


def _use_jvp_attn(model: Any) -> bool:
    import torch  # noqa: F401

    device = next(model.parameters()).device
    return device.type != "cuda"


def run_flm_request(model: Any, request: Any) -> dict[str, Any]:
    """Euler flow sampling — oracle FLM.generate_samples, line-cited."""
    import torch

    if not getattr(model, "is_flm_denoiser", False):
        raise ValueError(f"{type(model).__name__} is not a pack-loaded FLM denoiser")

    params = _common(model, request, default_steps=1024)
    num_steps = params["steps"]
    B = params["num_samples"]
    V = model.vocab_size
    L = model.num_tokens
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    use_jvp = _use_jvp_attn(model)

    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(params["seed"])
        # algo.py:1064-1065
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = torch.randn((B, L, V), device=device, dtype=dtype)

        with torch.no_grad():
            for i in range(num_steps):  # algo.py:1067-1081
                tau_t_curr = tau_vals[i]
                tau_t_next = tau_vals[i + 1]
                tau_t_in = tau_t_curr.expand(B)
                t_in = model._tau_to_t(tau_t_in)
                dt = model._tau_to_t(tau_t_next.expand(B)) - t_in
                x_1_pred = model(z, tau_t_in, use_jvp_attn=use_jvp)
                x_1_pred_probs = x_1_pred.exp()

                if i == num_steps - 1:
                    z = x_1_pred_probs
                    break

                v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)
                z = z + dt.view(-1, 1, 1) * v

        tokens = z.argmax(dim=-1)  # algo.py:1083

    return {
        "method": "flm",
        "tokens": tokens,
        "executed": {
            "solver": "euler",
            "steps_requested": num_steps,
            "steps_executed": num_steps,
            "nfe": num_steps,
            "tau_grid": [float(value) for value in tau_vals],
            "seed": params["seed"],
            "max_length": L,
            "use_jvp_attn": use_jvp,
        },
    }


def _reference_uses_compile() -> bool:
    """Whether the reference DIT was imported with compilation enabled.

    Read live rather than cached: the flag is a module attribute, and reading it
    at call time is what makes the compiled axis testable at all.
    """
    try:
        from unturtle_flm._reference import dit
    except ImportError:
        # Cannot show it is eager, so do not claim it is.
        return True
    return bool(getattr(dit, "USE_COMPILE", True))


def _execution_context(
    model: Any,
    steps: int,
    gamma: float,
    batch: int,
    length: int,
    vocab: int,
    device: Any,
) -> dict[str, Any]:
    """Describe the execution cell for the state-update scope gate.

    Reports what is true, never what would be convenient: an unresolvable axis
    is left as None so the guard rejects rather than assuming. The checkpoint is
    identified by repo_id AND revision, because the measurement was made against
    one pinned revision and a moved tag is a different model.
    """
    import torch

    checkpoint = getattr(model, "flm_checkpoint", None)
    if checkpoint is None:
        checkpoint_id = None
    else:
        checkpoint_id = f"{checkpoint.repo_id}@{checkpoint.revision}"

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        cuda_version = torch.version.cuda
    else:
        gpu_name = None
        cuda_version = None

    return {
        "gamma": gamma,
        "steps": steps,
        "batch": batch,
        "length": length,
        "vocab": vocab,
        "checkpoint": checkpoint_id,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        # Read from the reference module that actually decides it (DIT_USE_COMPILE
        # at import time). An earlier version read `model._unturtle_compiled`,
        # an attribute nothing ever assigns — so it was a constant False, and a
        # compiled model would have been labelled eager and admitted.
        # torch.compile may reassociate the very arithmetic whose bit identity
        # the fast path depends on, so this axis has to be real.
        "compiled": _reference_uses_compile(),
    }


def run_fmlm_request(model: Any, request: Any) -> dict[str, Any]:
    """Flow-map composition — oracle FMLM.generate_samples, line-cited.
    NEVER routes through the FLM Euler loop; every forward carries the
    (tau_curr, tau_tilde) double-time pair."""
    import torch

    if not getattr(model, "is_fmlm_flow_map", False):
        raise ValueError(
            f"{type(model).__name__} is not a pack-loaded FMLM flow map "
            "(double time conditioning required)"
        )

    params = _common(model, request, default_steps=1)
    num_steps = params["steps"]
    gamma = params["gamma"]
    B = params["num_samples"]
    V = model.vocab_size
    L = model.num_tokens
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    use_jvp = _use_jvp_attn(model)
    # Built ONCE per request, outside the loop: the state-update fast path is
    # admitted only inside the cell the Stage-2 outer-wall measurement actually
    # covers, and these are the axes that bound it. Anything unresolvable here
    # stays None, which the guard treats as "not shown to be in scope".
    execution_context = _execution_context(model, num_steps, gamma, B, L, V, device)
    flow_map_calls = 0
    # Resolved ONCE per request: a per-boundary lookup would put a ContextVar
    # read on the hot path for no isolation benefit.
    observer = _OBSERVER_CONTEXT.get()

    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(params["seed"])
        with _scope(observer, "grid_init"):
            # algo.py:1529-1532
            tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
            z = torch.randn((B, L, V), device=device, dtype=dtype)

        with torch.no_grad():
            for i in range(num_steps):  # algo.py:1534-1564
                with _scope(observer, "time_schedule"):
                    tau_curr = tau_vals[i]
                    tau_next = tau_vals[i + 1]

                    t_curr = model._tau_to_t(tau_curr.expand(B))
                    t_next = model._tau_to_t(tau_next.expand(B))
                    sigma_target = 1.0 - t_next

                    sigma_tilde = sigma_target * torch.sqrt(
                        torch.tensor(1.0 - gamma**2)
                    )
                    t_tilde = 1.0 - sigma_tilde
                    tau_tilde = model._t_to_tau(t_tilde)

                # Spans the model call AND the `.exp()`: the exponential is part
                # of producing D_st_pred, not a separate stage.
                with _scope(observer, "flow_map_forward"):
                    log_D_st_pred = model(
                        z, tau_curr.expand(B), tau_tilde, use_jvp_attn=use_jvp
                    )
                    flow_map_calls += 1
                    D_st_pred = log_D_st_pred.exp()

                # The final step exits BEFORE the state update, which is why
                # `state_update` has a structural count of 0 at steps=1.
                if i == num_steps - 1:
                    z = D_st_pred
                    break

                with _scope(observer, "state_update"):
                    weight_z = (1.0 - t_tilde.view(-1, 1, 1)) / (
                        1.0 - t_curr.view(-1, 1, 1)
                    )
                    weight_D = (t_tilde.view(-1, 1, 1) - t_curr.view(-1, 1, 1)) / (
                        1.0 - t_curr.view(-1, 1, 1)
                    )
                    if gamma > 0:
                        noise_std = gamma * sigma_target.view(-1, 1, 1)
                        mean_adjustment = sigma_tilde.view(
                            -1, 1, 1
                        ) - sigma_target.view(-1, 1, 1)
                        # `eps` is drawn HERE, not inside the helper, so the RNG
                        # stream advances identically whichever path runs.
                        eps = torch.randn_like(z)
                        z = apply_state_update(
                            z=z,
                            d_pred=D_st_pred,
                            weight_z=weight_z,
                            weight_d=weight_D,
                            mean_adjustment=mean_adjustment,
                            noise_std=noise_std,
                            eps=eps,
                            context=execution_context,
                        )
                    else:
                        # gamma == 0 is unspecialized: the fast path was only
                        # measured for the churn branch, and z_tilde is computed
                        # here rather than above so the gamma>0 path does not
                        # materialize a value it recomputes.
                        z = weight_z * z + weight_D * D_st_pred

        with _scope(observer, "endpoint_decode"):
            tokens = z.argmax(dim=-1)  # algo.py:1566

        # Read INSIDE the fork: `fork_rng` restores the outer generator on exit,
        # so a state read after the `with` block cannot see observer
        # perturbation of the internal stream.
        terminal_rng = (
            _capture_rng_state(device) if _capture_terminal_rng(request) else None
        )
        final_latent = z if _capture_final_latent(request) else None

    return {
        "method": "fmlm",
        "tokens": tokens,
        "executed": {
            "solver": "flow_map",
            "steps_requested": num_steps,
            "steps_executed": num_steps,
            "nfe": flow_map_calls,
            "gamma": gamma,
            "tau_grid": [float(value) for value in tau_vals],
            "seed": params["seed"],
            "max_length": L,
            "use_jvp_attn": use_jvp,
        },
        # Present only under the private diagnostic flags; absent in every
        # normal run, so the public result shape is unchanged.
        **({"_terminal_rng": terminal_rng} if terminal_rng is not None else {}),
        **({"_final_latent": final_latent} if final_latent is not None else {}),
    }


__all__ = ["run_flm_request", "run_fmlm_request"]
