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

from typing import Any


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
    flow_map_calls = 0

    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(params["seed"])
        # algo.py:1529-1532
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = torch.randn((B, L, V), device=device, dtype=dtype)

        with torch.no_grad():
            for i in range(num_steps):  # algo.py:1534-1564
                tau_curr = tau_vals[i]
                tau_next = tau_vals[i + 1]

                t_curr = model._tau_to_t(tau_curr.expand(B))
                t_next = model._tau_to_t(tau_next.expand(B))
                sigma_target = 1.0 - t_next

                sigma_tilde = sigma_target * torch.sqrt(torch.tensor(1.0 - gamma**2))
                t_tilde = 1.0 - sigma_tilde
                tau_tilde = model._t_to_tau(t_tilde)

                log_D_st_pred = model(
                    z, tau_curr.expand(B), tau_tilde, use_jvp_attn=use_jvp
                )
                flow_map_calls += 1
                D_st_pred = log_D_st_pred.exp()

                if i == num_steps - 1:
                    z = D_st_pred
                    break

                weight_z = (1.0 - t_tilde.view(-1, 1, 1)) / (
                    1.0 - t_curr.view(-1, 1, 1)
                )
                weight_D = (t_tilde.view(-1, 1, 1) - t_curr.view(-1, 1, 1)) / (
                    1.0 - t_curr.view(-1, 1, 1)
                )
                z_tilde = weight_z * z + weight_D * D_st_pred

                if gamma > 0:
                    noise_std = gamma * sigma_target.view(-1, 1, 1)
                    mean_adjustment = sigma_tilde.view(-1, 1, 1) - sigma_target.view(
                        -1, 1, 1
                    )
                    z = (
                        z_tilde
                        + mean_adjustment * D_st_pred
                        + noise_std * torch.randn_like(z)
                    )
                else:
                    z = z_tilde

        tokens = z.argmax(dim=-1)  # algo.py:1566

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
    }


__all__ = ["run_flm_request", "run_fmlm_request"]
