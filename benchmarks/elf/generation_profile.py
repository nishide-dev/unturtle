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

"""#166 Stage 1 — ELF generation profile (32-step and 64-step).

THE SOLE PRODUCER for `docs/artifacts/166-elf-generation-profile.json`.

NO PRODUCTION OR REFERENCE SOURCE IS EDITED. Instrumentation is
benchmark-local: this module patches module globals in
`unturtle_elf._reference.generation_utils` for the duration of one request and
restores them in `finally`.

The patch targets `generation_utils`, NOT `sampling_utils`. `generation_utils`
does `from ...sampling_utils import restore_cond, _ode_step, _sde_step`, so the
loop calls ALIASES bound at import time; patching `sampling_utils` would never
reach the call sites.

NESTED TAXONOMY. Wrapping the step functions alone would fuse the model forward
with the noise/restore/Euler update around it:

    solver_step_inclusive          audit parent, NOT counted in coverage
      └─ denoiser_forward          the rollout model call
    solver_state_update            = inclusive - denoiser_forward, PER STEP
    endpoint_projection            = _dlm_decode_batch (decoder head + argmax)

Coverage sums `denoiser_forward` + `solver_state_update` +
`endpoint_projection`. `solver_step_inclusive` is retained for audit only;
adding it would double-count its own children.

`solver_state_update` is a PER-STEP difference summed within a trial, never
`median(inclusive) - median(forward)` — medians of the two series can come from
different steps, so their difference is not an exclusive time.

`mask_after_eos` runs AFTER `_dlm_decode_batch` in the public entry point and is
therefore deliberately UNATTRIBUTED. It is not folded into
`endpoint_projection`; pretending it belongs there would misstate the boundary.

OVERHEAD IS DESCRIPTIVE, NOT ADJUDICATED (inherited from the #166 FMLM cell). At
TRIALS=3 the window reports a value and its per-trial range and cannot estimate
a noise floor, so no significance test is performed and nothing is invalidated
on the strength of one.

THE VERDICT IS THE INSTRUMENTATION-OFF OUTER WALL CLOCK. The ON pass only
attributes it.

Frozen configuration:

    solver sde, length 1024, cfg_scale 1.0, self_cond_cfg_scale 3.0,
    time_schedule logit_normal, batch in {1, 8, 32}
    32 steps -> sde_gamma 1.5
    64 steps -> sde_gamma 1.0

Usage:
    .venv/bin/python benchmarks/elf/generation_profile.py --device cuda:1 \
        --out docs/artifacts
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from types import MethodType
from typing import Any

#: Frozen measurement window. MODULE CONSTANTS, never CLI-settable.
TRIALS = 3
WARMUP = 2

#: Frozen cell configuration.
STEPS_CELLS = (32, 64)
BATCH_SIZES = (1, 8, 32)
SOLVER = "sde"
MAX_LENGTH = 1024
CFG_SCALE = 1.0
SELF_COND_CFG_SCALE = 3.0
TIME_SCHEDULE = "logit_normal"
SEED = 100
#: gamma is per-step-count in the frozen cells.
SDE_GAMMA = {32: 1.5, 64: 1.0}

EVENT_TAXONOMY = {
    "denoiser_forward": (
        "the rollout model call inside a solver step, attributed by an "
        "execution-phase-aware forward hook. Includes the SC-CFG token's own "
        "attention/prefix cost, which the frozen protocol does NOT decompose "
        "into a separate event. EXCLUDES the endpoint decoder forward."
    ),
    "solver_state_update": (
        "everything in a solver step that is not the rollout model call: "
        "noise draw, conditional restore and the Euler/SDE update. Computed as "
        "a PER-STEP difference (step_inclusive - its own child forward) summed "
        "within a trial, never as a difference of medians."
    ),
    "endpoint_projection": (
        "the whole of _dlm_decode_batch: the endpoint decoder forward and the "
        "argmax. Its model call is recorded here and NOT in denoiser_forward."
    ),
    "solver_step_inclusive": (
        "AUDIT PARENT, excluded from coverage. The full solver step including "
        "its child denoiser_forward; retained so the parent/child pairing can "
        "be checked, and because subtracting it from coverage would "
        "double-count its children."
    ),
}

#: Events that sum to coverage. The audit parent is excluded by construction.
COVERAGE_EVENTS = (
    "denoiser_forward",
    "solver_state_update",
    "endpoint_projection",
)


def expected_step_calls(steps: int) -> dict[str, int]:
    """`t_steps` has length steps+1, the loop runs len-2 times, and the final
    step is ALWAYS ODE — so an SDE cell is (steps-1) SDE plus 1 ODE."""
    return {"_sde_step": steps - 1, "_ode_step": 1}


def expected_forward_calls(steps: int) -> dict[str, int]:
    return {"rollout": steps, "endpoint": 1}


def expected_random_calls(steps: int) -> dict[str, int]:
    """Per callsite, not just the total: a future random op inside the model
    would otherwise be absorbed into a matching total."""
    return {
        # sample_timesteps for the logit-normal grid.
        "time_grid": 1,
        # The initial latent; the CUDA and CPU branches are exclusive.
        "initial_latent": 1,
        # One churn draw per SDE step; the final ODE step draws none.
        "sde_churn": steps - 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def require_supported_device(device: str) -> str:
    """Any single CUDA device, pinned as the CURRENT device: peak-stat calls
    default to it and the GPU name is read from it."""
    import torch

    if not device.startswith("cuda:"):
        raise SystemExit(
            f"--device {device!r} is not supported: a single CUDA device is required"
        )
    try:
        index = int(device.split(":")[1])
    except ValueError:
        raise SystemExit(
            f"--device {device!r} is not supported: expected cuda:<index>"
        ) from None
    if index < 0 or index >= torch.cuda.device_count():
        raise SystemExit(
            f"--device {device!r} does not exist: this host has "
            f"{torch.cuda.device_count()} CUDA device(s)"
        )
    torch.cuda.set_device(index)
    return device


def device_occupancy(device: str) -> dict[str, Any]:
    """Free/total memory and the CUDA processes resident on THIS device, from
    `nvidia-smi` compute-app data scoped by UUID. Process-name matching cannot
    tell whether a process holds GPU memory, nor on which device."""
    import torch

    index = int(device.split(":")[1])
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)

    def query(fields: str) -> list[str]:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in result.stdout.strip().splitlines() if line.strip()]

    uuid = None
    for row in query("gpu=index,uuid"):
        parts = [field.strip() for field in row.split(",")]
        if parts and parts[0] == str(index):
            uuid = parts[1]
            break
    if uuid is None:
        raise SystemExit(f"cannot resolve the UUID of {device} from nvidia-smi")

    processes = []
    for row in query("compute-apps=pid,used_memory,gpu_uuid"):
        pid, used_mib, row_uuid = (field.strip() for field in row.split(","))
        if row_uuid == uuid:
            processes.append({"pid": int(pid), "used_mib": int(used_mib)})
    return {
        "device": device,
        "uuid": uuid,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "compute_processes": processes,
        "foreign_process_count": len(
            [entry for entry in processes if entry["pid"] != os.getpid()]
        ),
        "source": "nvidia-smi compute-apps scoped by gpu_uuid",
    }


def require_idle_device(device: str) -> dict[str, Any]:
    """Refuse to measure on a device another CUDA process is using: latency,
    peak memory and any typed OOM would be attributable to another workload."""
    occupancy = device_occupancy(device)
    foreign = [
        entry for entry in occupancy["compute_processes"] if entry["pid"] != os.getpid()
    ]
    if foreign:
        raise SystemExit(
            f"{device} is shared: {len(foreign)} unrelated CUDA process(es) "
            f"hold GPU memory ({foreign}). Re-run on an exclusive device."
        )
    return occupancy


class Request:
    """The public request shape. Diagnostic flags live on a PRIVATE attribute so
    `kwargs` — the documented surface — carries only decoding options."""

    def __init__(self, *, steps: int, num_samples: int):
        self.kwargs = {
            "solver": SOLVER,
            "steps": steps,
            "num_samples": num_samples,
            "seed": SEED,
            "sde_gamma": SDE_GAMMA[steps],
            "cfg_scale": CFG_SCALE,
            "self_cond_cfg_scale": SELF_COND_CFG_SCALE,
            "time_schedule": TIME_SCHEDULE,
        }


class Recorder:
    """Collects nested spans and per-callsite counts for ONE request.

    `mode="count"` records call counts and parent/child pairing only.
    `mode="time"` additionally records CUDA event pairs. Neither mode hashes a
    tensor, calls `.item()`/`.cpu()`, clones, or reads RNG state — those belong
    in the preflight, never in a timed trial.
    """

    def __init__(self, torch_module, *, mode: str) -> None:
        self._torch = torch_module
        self.mode = mode
        #: One entry per solver step: {"inclusive": (start, end),
        #: "forwards": [(start, end), ...]} so the exclusive time is a PER-STEP
        #: difference and the pairing can be audited.
        self.steps: list[dict[str, Any]] = []
        self.endpoint: list[tuple[Any, Any]] = []
        self.step_calls: dict[str, int] = {}
        self.forward_calls: dict[str, int] = {"rollout": 0, "endpoint": 0}
        #: "rollout" while a solver step is open, "endpoint" inside decode.
        self.phase: str | None = None
        self._open_step_id: int | None = None
        self.per_step_exclusive: list[float] = []

    def _mark(self):
        if self.mode != "time":
            return None
        event = self._torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @contextlib.contextmanager
    def solver_step(self, name: str):
        self.step_calls[name] = self.step_calls.get(name, 0) + 1
        step_id = len(self.steps)
        record: dict[str, Any] = {
            "name": name,
            "step_id": step_id,
            "forwards": [],
            # The id of the step that was OPEN when each child was recorded, so
            # a child attributed to the wrong parent is detectable even when
            # every count is right.
            "forwards_step_id": [],
        }
        self.steps.append(record)
        previous_step, self._open_step_id = (
            getattr(self, "_open_step_id", None),
            step_id,
        )
        previous, self.phase = self.phase, "rollout"
        start = self._mark()
        try:
            yield
        finally:
            record["inclusive"] = (start, self._mark())
            self.phase = previous
            self._open_step_id = previous_step

    @contextlib.contextmanager
    def endpoint_projection(self):
        previous, self.phase = self.phase, "endpoint"
        start = self._mark()
        try:
            yield
        finally:
            self.endpoint.append((start, self._mark()))
            self.phase = previous

    @contextlib.contextmanager
    def model_forward(self):
        """Attributed by EXECUTION PHASE: the endpoint decoder's forward must
        not land in `denoiser_forward`."""
        phase = self.phase
        if phase == "rollout":
            self.forward_calls["rollout"] += 1
        elif phase == "endpoint":
            self.forward_calls["endpoint"] += 1
        start = self._mark()
        try:
            yield
        finally:
            end = self._mark()
            if phase == "rollout" and self.steps:
                open_id = getattr(self, "_open_step_id", None)
                target = self.steps[open_id] if open_id is not None else self.steps[-1]
                target["forwards"].append((start, end))
                target["forwards_step_id"].append(open_id)


class InstrumentationError(RuntimeError):
    """A benchmark-local instrumentation invariant was violated.

    Distinct from a measurement failure: the harness itself is in a state where
    attribution would be wrong, so the cell must be typed rather than the
    numbers trusted.
    """


@contextlib.contextmanager
def instrumented(model, recorder: Recorder | None):
    """Install the benchmark-local patches for ONE request and restore them.

    Patches module globals on `generation_utils` because the loop calls aliases
    bound there at import time. Restoration is unconditional: a leaked patch
    would instrument the next arm, and the OFF arm must be genuinely
    uninstrumented.
    """
    if recorder is None:
        yield
        return

    from unturtle_elf._reference import generation_utils as gu

    original_sde = gu._sde_step
    original_ode = gu._ode_step
    original_decode = gu._dlm_decode_batch

    def wrap_step(name, original):
        def wrapped(*args, **kwargs):
            with recorder.solver_step(name):
                return original(*args, **kwargs)

        return wrapped

    def wrapped_decode(*args, **kwargs):
        with recorder.endpoint_projection():
            return original_decode(*args, **kwargs)

    # `forward` is wrapped rather than hooked: a PRE-hook cannot close a span
    # and a post-hook fires after the call, so neither brackets the model call.
    #
    # The wrapper is INSTANCE-LOCAL. Assigning `type(model).forward` would
    # instrument every instance of the class for the duration of the request —
    # MEASURED: with a class-level patch installed for one model, calling an
    # unrelated instance of the same class recorded a rollout forward on this
    # recorder. `finally` restores after the fact but cannot prevent that
    # leakage while the patch is live. Same lesson as the #166 FMLM
    # module-global observer.
    if "forward" in model.__dict__:
        # A RuntimeError, not SystemExit: this is an invariant violation the
        # caller can type and record as a failed cell, whereas SystemExit
        # slips past `except Exception` and would kill the producer before it
        # could write an artifact for the other cells.
        raise InstrumentationError(
            "the model already carries an instance-level `forward` override; "
            "refusing to nest instrumentation, which would make attribution "
            "depend on install order"
        )
    original_bound_forward = model.forward

    def wrapped_forward(_self, *args, **kwargs):
        with recorder.model_forward():
            return original_bound_forward(*args, **kwargs)

    gu._sde_step = wrap_step("_sde_step", original_sde)
    gu._ode_step = wrap_step("_ode_step", original_ode)
    gu._dlm_decode_batch = wrapped_decode
    model.__dict__["forward"] = MethodType(wrapped_forward, model)
    try:
        yield
    finally:
        gu._sde_step = original_sde
        gu._ode_step = original_ode
        gu._dlm_decode_batch = original_decode
        # Delete rather than reassign: there was no instance override before,
        # so restoring one would leave a permanent shadow of the class
        # descriptor.
        model.__dict__.pop("forward", None)


def elapsed(pair) -> float:
    start, end = pair
    return start.elapsed_time(end) / 1000.0


def classify_random_call(stack) -> str:
    """Classify a `randn` callsite by its calling module AND function.

    Shape alone is insufficient: the initial latent and the SDE churn draw the
    SAME `[B, L, d_model]` shape, so a shape-keyed classifier would merge them.
    An unrecognised callsite is reported as `unknown` and FAILS the preflight
    rather than being ignored — a new random op inside the model would
    otherwise be absorbed into a matching total.
    """
    for frame in stack:
        module = frame.filename.replace("\\", "/")
        function = frame.function
        if module.endswith("_reference/sampling_utils.py"):
            if function == "sample_timesteps":
                return "time_grid"
            if function in ("_sde_step", "sde_step"):
                return "sde_churn"
        if module.endswith("unturtle_elf/sampler.py") and function in (
            "run_generation_request",
            "_generate",
        ):
            return "initial_latent"
    return "unknown"


def check_span_ordering(recorder: Recorder) -> list[str]:
    """Structural checks on the nested spans, independent of any timing value.

    The dangerous regression is not a wrong COUNT — it is subtracting a forward
    that belongs to a DIFFERENT step. Each parent must own exactly one child,
    recorded while that parent was the open step.
    """
    problems: list[str] = []
    for index, record in enumerate(recorder.steps):
        if "inclusive" not in record:
            problems.append(f"solver step {index}: parent span never closed")
            continue
        forwards = record["forwards"]
        if len(forwards) != 1:
            problems.append(
                f"solver step {index} ({record['name']}) has {len(forwards)} "
                "rollout forwards, expected exactly 1"
            )
            continue
        if record.get("step_id") != record["forwards_step_id"][0]:
            problems.append(
                f"solver step {index}: its child forward was recorded against "
                f"step id {record['forwards_step_id'][0]}, not {record.get('step_id')}"
            )
    return problems


def step_exclusive_seconds(recorder: Recorder) -> tuple[float, float, list[str]]:
    """Per-step exclusive time, summed within the trial.

    Returns (denoiser_seconds, state_update_seconds, problems); the per-step
    exclusive values are attached to the recorder for the artifact's audit
    fields. A step whose
    child forward count is not exactly one is a pairing failure: the counts can
    still add up while the parent/child correspondence is broken, and the
    exclusive time would then be attributed to the wrong event.
    """
    problems: list[str] = []
    denoiser = 0.0
    state_update = 0.0
    per_step: list[float] = []
    for index, record in enumerate(recorder.steps):
        forwards = record["forwards"]
        if len(forwards) != 1:
            problems.append(
                f"solver step {index} ({record['name']}) has {len(forwards)} "
                "rollout forwards, expected exactly 1"
            )
            continue
        inclusive = elapsed(record["inclusive"])
        child = elapsed(forwards[0])
        exclusive = inclusive - child
        if exclusive < 0:
            # On one stream a correctly nested parent cannot be shorter than
            # its own child, so a negative exclusive time is direct evidence of
            # broken pairing or ordering. NEVER clamped: clamping would hide it.
            problems.append(
                f"solver step {index} ({record['name']}): exclusive time "
                f"{exclusive:.6f}s is negative, so the parent span is shorter "
                "than the child it contains"
            )
            continue
        denoiser += child
        # PER-STEP difference. `median(inclusive) - median(forward)` would
        # subtract series whose medians can come from different steps.
        state_update += exclusive
        per_step.append(exclusive)
    recorder.per_step_exclusive = per_step
    return denoiser, state_update, problems


def stable_hash(tensor) -> dict[str, Any]:
    """A digest plus the shape/dtype/device that make it meaningful. A bare
    hash is not auditable, and the raw tensors are far too large to store."""
    import torch

    contiguous = tensor.detach().to("cpu").contiguous()
    payload = (
        contiguous.numpy().tobytes()
        if contiguous.dtype == torch.bool
        else contiguous.view(torch.uint8).numpy().tobytes()
    )
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": list(contiguous.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def rng_state_hashes(device) -> dict[str, Any]:
    import torch

    states = {"cpu": stable_hash(torch.get_rng_state())}
    if device is not None and str(device).startswith("cuda"):
        states["cuda"] = stable_hash(torch.cuda.get_rng_state(device))
    return states


@contextlib.contextmanager
def decode_diagnostics(model, sink: dict[str, Any], device):
    """Capture the endpoint boundary values around `_dlm_decode_batch`.

    Terminal RNG is read at decode EXIT, not entry: an endpoint forward that
    wrongly consumed randomness would be invisible to an entry-time reading.
    Everything after this point in the public entry (`mask_after_eos`) is pure
    tensor arithmetic with no random ops, so decode exit is the last moment the
    internal stream can move — verified against the reference source.

    DIAGNOSTIC ONLY. Hashing allocates and copies, so this never runs inside a
    timed trial.
    """
    from unturtle_elf._reference import generation_utils as gu

    original = gu._dlm_decode_batch

    def wrapped(z, *args, **kwargs):
        sink["final_latent"] = stable_hash(z)
        tokens = original(z, *args, **kwargs)
        sink["raw_endpoint_tokens"] = stable_hash(tokens)
        sink["terminal_rng"] = rng_state_hashes(device)
        return tokens

    gu._dlm_decode_batch = wrapped
    try:
        yield
    finally:
        gu._dlm_decode_batch = original


@contextlib.contextmanager
def random_call_spy(counts: dict[str, int]):
    """Count `torch.randn` by CALLSITE for exactly one request.

    The patch is process-global, so it wraps a single request and is restored in
    `finally`. It never runs during a timed trial.
    """
    import inspect as _inspect

    import torch

    original = torch.randn

    def counting(*args, **kwargs):
        kind = classify_random_call(_inspect.stack()[1:])
        counts[kind] = counts.get(kind, 0) + 1
        return original(*args, **kwargs)

    torch.randn = counting
    try:
        yield
    finally:
        torch.randn = original


def run_once(model, request, recorder: Recorder | None = None):
    """OFF and ON both go through the PUBLIC entry point; the only difference
    between the arms is whether instrumentation is installed."""
    with instrumented(model, recorder):
        return _public_request(model, request)


def non_interference_preflight(model, steps: int, batch: int, device) -> dict[str, Any]:
    """Sequential OFF/ON comparison on the REAL checkpoint.

    Both sides are recorded, not just the verdict, so the comparison is
    auditable after the fact.
    """
    import torch

    arms: dict[str, Any] = {}
    for arm in ("off", "on"):
        sink: dict[str, Any] = {}
        recorder = Recorder(torch, mode="count") if arm == "on" else None
        with decode_diagnostics(model, sink, device):
            result = run_once(model, Request(steps=steps, num_samples=batch), recorder)
        record = {
            "final_latent": sink["final_latent"],
            "raw_endpoint_tokens": sink["raw_endpoint_tokens"],
            "masked_public_tokens": stable_hash(result["tokens"]),
            "terminal_rng_cpu": sink["terminal_rng"]["cpu"],
            "executed_metadata": result.get("executed"),
        }
        if "cuda" in sink["terminal_rng"]:
            record["terminal_rng_cuda"] = sink["terminal_rng"]["cuda"]
        arms[arm] = record
        if recorder is not None:
            arms["on_structure"] = {
                "step_calls": dict(recorder.step_calls),
                "forward_calls": dict(recorder.forward_calls),
                "pairing_problems": check_span_ordering(recorder),
            }
        del result
    off, on = arms["off"], arms["on"]
    fields = [
        "final_latent",
        "raw_endpoint_tokens",
        "masked_public_tokens",
        "terminal_rng_cpu",
        "executed_metadata",
    ]
    if "terminal_rng_cuda" in off:
        fields.append("terminal_rng_cuda")
    matches = {field: off[field] == on[field] for field in fields}
    cleanup()
    return {
        "off": off,
        "on": on,
        "on_structure": arms["on_structure"],
        "matches": matches,
        "status": "ok" if all(matches.values()) else "observer_interference",
    }


def random_call_preflight(model, steps: int, batch: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    with random_call_spy(counts):
        run_once(model, Request(steps=steps, num_samples=batch))
    expected = expected_random_calls(steps)
    observed = {key: counts.get(key, 0) for key in expected}
    unknown = counts.get("unknown", 0)
    return {
        "observed": observed,
        "expected": expected,
        "unknown": unknown,
        "total_observed": sum(counts.values()),
        "total_expected": sum(expected.values()),
        # An unclassified callsite is a FAILURE, not a rounding detail: a new
        # random op inside the model would otherwise be absorbed into a
        # matching total.
        "matches": observed == expected and unknown == 0,
    }


def cleanup() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _run_off_trial(model, request, torch, time) -> dict[str, Any]:
    """One uninstrumented trial. Peak stats are reset PER TRIAL, before the
    clock starts."""
    # No instrumentation at all on this arm — not even an inert context.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    _public_request(model, request)
    torch.cuda.synchronize()
    return {
        "wall_seconds": time.perf_counter() - begin,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _run_on_trial(model, request, torch, time) -> dict[str, Any]:
    """One instrumented trial, in the frozen order:

        install -> synchronize -> start -> request -> synchronize -> stop
        -> restore -> collect

    Installing INSIDE the timed window would bill patch setup to the run, and
    synchronizing AFTER the clock stops would push the queue drain outside the
    wall — the defect the #166 FMLM cell hit, where the ON wall came out shorter
    than the event total it was supposed to contain.
    """
    recorder = Recorder(torch, mode="time")
    # `instrumented` is entered OUTSIDE the timed span: installing the patches
    # inside it would bill setup to the run.
    with instrumented(model, recorder):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        _public_request(model, request)
        torch.cuda.synchronize()
        wall = time.perf_counter() - begin
    # `collect` happens after restore and does NOT synchronize: the single
    # window-closing sync already ran inside the timed span.
    denoiser, state_update, problems = step_exclusive_seconds(recorder)
    problems = check_span_ordering(recorder) + problems
    endpoint = sum(elapsed(pair) for pair in recorder.endpoint)
    inclusive = sum(
        elapsed(record["inclusive"])
        for record in recorder.steps
        if "inclusive" in record
    )
    return {
        "wall_seconds": wall,
        "event_seconds": {
            "denoiser_forward": denoiser,
            "solver_state_update": state_update,
            "endpoint_projection": endpoint,
        },
        "audit_seconds": {"solver_step_inclusive": inclusive},
        "per_step_exclusive_seconds": recorder.per_step_exclusive,
        "step_calls": dict(recorder.step_calls),
        "forward_calls": dict(recorder.forward_calls),
        "problems": problems,
    }


def _public_request(model, request):
    """The ONE call site of the public entry point.

    Both arms route through this, so the OFF and ON paths differ only by
    whether `instrumented` installed anything — never by which function is
    called."""
    from unturtle_elf.sampler import run_generation_request

    return run_generation_request(model, request)


def warmup_arms(model, steps: int, batch: int) -> None:
    """Warm BOTH arms before any timing, in its own failure stage, so an OOM
    here is not recorded as a timed failure."""
    import torch

    request = Request(steps=steps, num_samples=batch)
    for _ in range(WARMUP):
        run_once(model, request)
        run_once(model, request, Recorder(torch, mode="time"))
    torch.cuda.synchronize()


def paired_trials(model, steps: int, batch: int) -> list[dict[str, Any]]:
    """PAIRED, INTERLEAVED trials with the order reversed each time:

        trial 0: OFF -> ON      trial 1: ON -> OFF      trial 2: OFF -> ON

    Running every OFF trial then every ON trial loads thermal state, clock
    drift and allocator growth onto whichever arm always runs second, in the
    same direction as the effect being measured.
    """
    import time

    import torch

    request = Request(steps=steps, num_samples=batch)
    trials: list[dict[str, Any]] = []
    for index in range(TRIALS):
        order = ("off", "on") if index % 2 == 0 else ("on", "off")
        measured: dict[str, dict[str, Any]] = {}
        for arm in order:
            if arm == "off":
                measured["off"] = _run_off_trial(model, request, torch, time)
            else:
                measured["on"] = _run_on_trial(model, request, torch, time)
        trials.append(
            {
                "trial": index,
                "order": list(order),
                "off_wall_seconds": measured["off"]["wall_seconds"],
                "on_wall_seconds": measured["on"]["wall_seconds"],
                "paired_overhead_seconds": (
                    measured["on"]["wall_seconds"] - measured["off"]["wall_seconds"]
                ),
                "peak_allocated_bytes": measured["off"]["peak_allocated_bytes"],
                "peak_reserved_bytes": measured["off"]["peak_reserved_bytes"],
                "event_seconds": measured["on"]["event_seconds"],
                "audit_seconds": measured["on"]["audit_seconds"],
                "per_step_exclusive_seconds": measured["on"][
                    "per_step_exclusive_seconds"
                ],
                "step_calls": measured["on"]["step_calls"],
                "forward_calls": measured["on"]["forward_calls"],
                "problems": measured["on"]["problems"],
            }
        )
    return trials


def overhead_estimate(trials: list[dict]) -> dict[str, Any]:
    """DESCRIPTIVE ONLY. Three trials are enough to report a value and its
    range, not to estimate a noise floor and declare a difference resolved, so
    no significance test is performed. `direction_consistent` is diagnostic and
    nothing gates on it: with a true overhead of zero, all three deltas land
    negative 12.5% of the time."""
    deltas = [trial["paired_overhead_seconds"] for trial in trials]
    off_walls = [trial["off_wall_seconds"] for trial in trials]
    return {
        "paired_delta_trials": deltas,
        "median_paired_delta": statistics.median(deltas) if deltas else 0.0,
        "off_wall_trials": off_walls,
        "off_trial_spread": (max(off_walls) - min(off_walls)) if off_walls else 0.0,
        "direction_consistent": bool(deltas)
        and (all(x > 0 for x in deltas) or all(x < 0 for x in deltas)),
        "resolvable": None,
        "resolution_status": "not_assessed",
        "resolution_reason": (
            "the frozen three-trial window is insufficient to estimate the noise floor"
        ),
        "basis": (
            "median of per-trial (on_wall - off_wall), from OFF/ON pairs run "
            "adjacently with the order reversed each trial. Descriptive; "
            "negative values are left as measured, neither clamped nor "
            "reinterpreted"
        ),
    }


def failure_record(
    *,
    stage: str,
    reason_code: str | None,
    timing_attempted: bool,
    status: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The typed failure disposition, built in ONE place. Every unmeasured
    field is null, never zero: a 0.0 latency reads as "measured, nothing
    there"."""
    return {
        "status": status
        or ("oom" if reason_code == "cuda_out_of_memory" else "measurement_invalid"),
        "failure_stage": stage,
        "reason_code": reason_code,
        "timing_attempted": timing_attempted,
        "latency": None,
        "events": None,
        "peak_memory": None,
        "attribution": None,
        "trials": None,
        **extra,
    }


def classify_failure(error: BaseException) -> str | None:
    """CUDA capacity only. A shape or device error is a different defect, and
    the host allocator also says "tried to allocate", so the word `cuda` is
    required."""
    import torch

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "cuda_out_of_memory"
    text = str(error).lower()
    if isinstance(error, RuntimeError) and "out of memory" in text and "cuda" in text:
        return "cuda_out_of_memory"
    return None


def gate_trial(steps: int, trial: dict) -> list[str]:
    """Per-trial structural gate on counts and pairing."""
    problems = list(trial.get("problems") or [])
    expected_steps = expected_step_calls(steps)
    if trial["step_calls"] != expected_steps:
        problems.append(f"step calls {trial['step_calls']}, expected {expected_steps}")
    expected_forwards = expected_forward_calls(steps)
    if trial["forward_calls"] != expected_forwards:
        problems.append(
            f"forward calls {trial['forward_calls']}, expected {expected_forwards}"
        )
    return problems


def profile_cell(model, steps: int, batch: int, device: str) -> dict[str, Any]:
    """One (steps, batch) cell.

    Failure staging distinguishes an instrumentation invariant violation from a
    measurement failure, and records `timing_attempted` from WHERE the exception
    arose rather than assuming it.
    """
    cell: dict[str, Any] = {
        "steps": steps,
        "batch_size": batch,
        "solver": SOLVER,
        "sde_gamma": SDE_GAMMA[steps],
        "cfg_scale": CFG_SCALE,
        "self_cond_cfg_scale": SELF_COND_CFG_SCALE,
        "time_schedule": TIME_SCHEDULE,
        "seed": SEED,
        "max_length": MAX_LENGTH,
    }
    stage = "preflight"
    try:
        cell["device_occupancy_before"] = require_idle_device(device)
        interference = non_interference_preflight(model, steps, batch, device)
        cell["non_interference"] = interference
        if interference["status"] != "ok":
            return cell | failure_record(
                stage="non_interference_preflight",
                reason_code="observer_interference",
                timing_attempted=False,
            )
        randoms = random_call_preflight(model, steps, batch)
        cell["random_calls"] = randoms
        if not randoms["matches"]:
            return cell | failure_record(
                stage="random_call_preflight",
                reason_code=(
                    "unclassified_random_callsite"
                    if randoms["unknown"]
                    else "random_call_count_mismatch"
                ),
                timing_attempted=False,
            )
        stage = "warmup"
        cleanup()
        warmup_arms(model, steps, batch)
        stage = "paired_trials"
        trials = paired_trials(model, steps, batch)
    except InstrumentationError as error:
        cleanup()
        return cell | failure_record(
            stage=stage,
            reason_code="instrumentation_structure_invalid",
            # True only once a trial clock has started.
            timing_attempted=stage == "paired_trials",
            status="profile_invalid",
            exception_class=type(error).__name__,
            exception_message=str(error)[:300],
        )
    except Exception as error:  # noqa: BLE001 - classified, then re-reported
        reason = classify_failure(error)
        cleanup()
        return cell | failure_record(
            stage=stage,
            reason_code=reason,
            timing_attempted=stage == "paired_trials",
            exception_class=type(error).__name__,
            exception_message=str(error)[:300],
        )

    problems: list[str] = []
    for index, trial in enumerate(trials):
        for problem in gate_trial(steps, trial):
            problems.append(f"trial[{index}]: {problem}")
    if problems:
        cleanup()
        return cell | failure_record(
            stage="paired_trials",
            reason_code="per_trial_structure_invalid",
            timing_attempted=True,
            status="profile_invalid",
            problems=problems,
        )
    return cell | assemble_cell(steps, trials)


def assemble_cell(steps: int, trials: list[dict]) -> dict[str, Any]:
    """Build the measured cell, gating coverage and residual on EVERY trial."""
    off_walls = [t["off_wall_seconds"] for t in trials]
    on_walls = [t["on_wall_seconds"] for t in trials]

    coverage_per_trial = [
        sum(t["event_seconds"].values()) / t["on_wall_seconds"]
        if t["on_wall_seconds"] > 0
        else float("inf")
        for t in trials
    ]
    residuals = [
        t["on_wall_seconds"] - sum(t["event_seconds"].values()) for t in trials
    ]

    # The ONLY validity conditions here are a non-positive wall, a NEGATIVE
    # residual, and a non-finite or negative event value. Coverage is reported
    # but never gated on: see `coverage_disposition`.
    trial_problems: list[str] = []
    for index, trial in enumerate(trials):
        if trial["on_wall_seconds"] <= 0 or trial["off_wall_seconds"] <= 0:
            trial_problems.append(f"trial[{index}]: non-positive wall clock")
        if residuals[index] < 0:
            trial_problems.append(
                f"trial[{index}]: residual {residuals[index]:.6f}s is negative"
            )
        for name, value in trial["event_seconds"].items():
            if not math.isfinite(value) or value < 0:
                trial_problems.append(f"trial[{index}]: {name} = {value!r}")
    if trial_problems:
        cleanup()
        return failure_record(
            stage="paired_trials",
            reason_code="negative_residual_or_invalid_event",
            timing_attempted=True,
            status="profile_invalid",
            problems=trial_problems
            + [
                f"coverage per trial: {coverage_per_trial}",
                f"residual per trial: {residuals}",
            ],
        )

    events = []
    for name in COVERAGE_EVENTS:
        per_trial = [t["event_seconds"][name] for t in trials]
        shares = [
            t["event_seconds"][name] / t["on_wall_seconds"]
            for t in trials
            if t["on_wall_seconds"] > 0
        ]
        events.append(
            {
                "name": name,
                "seconds": statistics.median(per_trial),
                "seconds_trials": per_trial,
                # Share PER TRIAL, then the median of the shares: dividing a
                # median event time by a median wall mixes different trials.
                "share_of_on_wall": statistics.median(shares) if shares else None,
                "description": EVENT_TAXONOMY[name],
            }
        )

    allocated = [t["peak_allocated_bytes"] for t in trials]
    reserved = [t["peak_reserved_bytes"] for t in trials]
    inclusive_trials = [t["audit_seconds"]["solver_step_inclusive"] for t in trials]
    cleanup()
    return {
        "status": "ok",
        "timing_attempted": True,
        "latency": {
            "verdict_seconds": statistics.median(off_walls),
            "verdict_basis": "instrumentation_off_outer_wall_clock",
            "off_wall_trials": off_walls,
            "on_wall_median": statistics.median(on_walls),
            "on_wall_trials": on_walls,
            "instrumentation_overhead": overhead_estimate(trials),
        },
        "trials": [
            {
                "trial": t["trial"],
                "order": t["order"],
                "off_wall_seconds": t["off_wall_seconds"],
                "on_wall_seconds": t["on_wall_seconds"],
                "paired_overhead_seconds": t["paired_overhead_seconds"],
                "step_calls": t["step_calls"],
                "forward_calls": t["forward_calls"],
            }
            for t in trials
        ],
        "peak_memory": {
            "allocated_bytes_trials": allocated,
            "reserved_bytes_trials": reserved,
            "max_allocated_bytes": max(allocated),
            "max_reserved_bytes": max(reserved),
            "basis": "instrumentation_off_trials",
        },
        "events": events,
        "audit": {
            # NOT in coverage: it contains its own children.
            "solver_step_inclusive_seconds_trials": inclusive_trials,
            "solver_state_update_seconds_trials": [
                t["event_seconds"]["solver_state_update"] for t in trials
            ],
            "per_step_exclusive_seconds": [
                t["per_step_exclusive_seconds"] for t in trials
            ],
            "note": (
                "retained so a later regression to median-of-medians is "
                "detectable from the artifact alone: the reported state update "
                "must equal the sum of that trial's per-step exclusive values"
            ),
        },
        "forward_accounting": {
            "rollout_forward_count": trials[0]["forward_calls"]["rollout"],
            "endpoint_forward_count": trials[0]["forward_calls"]["endpoint"],
            "extra_cfg_forward_count": 0,
            "total_top_level_model_calls": steps + 1,
            "sc_cfg_token_cost": "included_in_denoiser_forward",
        },
        "attribution": {
            "denominator": "per_trial_on_wall_seconds",
            "aggregation": "median_of_per_trial_ratios",
            # DESCRIPTIVE. Coverage never classifies a cell: `coverage > 1` is
            # exactly `residual < 0`, since both derive from the same attributed
            # sum and the same wall, so a coverage gate would be strictly
            # subsumed by the residual gate — verified, and an unreachable gate
            # is worse than none because it reads as an independent check.
            "coverage_ratio": statistics.median(coverage_per_trial),
            "coverage_ratio_trials": coverage_per_trial,
            "coverage_disposition": "descriptive_only",
            "unattributed_seconds": statistics.median(residuals),
            "unattributed_seconds_trials": residuals,
            "unattributed_note": (
                "mask_after_eos runs after _dlm_decode_batch in the public "
                "entry point, so it is inside the outer wall but outside every "
                "event span and is deliberately unattributed; fork_rng "
                "enter/exit, the time grid and result assembly are likewise "
                "unattributed rather than folded into an adjacent event"
            ),
        },
    }


def environment() -> dict[str, Any]:
    import torch

    def version(name: str) -> str | None:
        try:
            return __import__(name).__version__
        except Exception:  # pragma: no cover - reported as null
            return None

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device_index": torch.cuda.current_device(),
        "transformers": version("transformers"),
    }


def provenance(
    args: argparse.Namespace, occupancy_at_start: dict[str, Any]
) -> dict[str, Any]:
    from unturtle_elf.loader import DEFAULT_CHECKPOINT, DEFAULT_REVISION

    def git(*command: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *command], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover
            return None

    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    if head is None or dirty is None:
        raise SystemExit(
            "cannot establish provenance (git rev-parse / status failed); "
            "refusing to write an artifact whose measuring commit is unknown"
        )
    return {
        "head_sha": head,
        "worktree_clean": dirty == "",
        "worktree_dirty_paths": [line[3:] for line in dirty.splitlines()],
        "command": " ".join(sys.argv),
        "environment": environment(),
        # Captured in `main` before the model is loaded; the write-time reading
        # is separate and named for when it is taken.
        "device_occupancy_at_start": occupancy_at_start,
        "device_occupancy_at_artifact_write": device_occupancy(args.device),
        "exclusivity_contract": (
            "the run is refused if any unrelated CUDA process holds memory on "
            "the target device, checked before every cell via nvidia-smi "
            "compute-app data scoped by gpu_uuid"
        ),
        "instrumentation_contract": (
            "benchmark-local only: no production or reference source is "
            "edited. `generation_utils` module globals and the TARGET "
            "INSTANCE's `forward` are patched for one request and restored in "
            "`finally`. The forward wrapper is instance-local because a "
            "class-level patch instruments every instance of the class while "
            "it is live"
        ),
        "overhead_contract": (
            "instrumentation overhead is DESCRIPTIVE ONLY at TRIALS=3: no "
            "significance test is performed and no cell is invalidated on the "
            "basis of the overhead sign"
        ),
        "fixture": {
            "checkpoint": f"{DEFAULT_CHECKPOINT}@{DEFAULT_REVISION}",
            "solver": SOLVER,
            "max_length": MAX_LENGTH,
            "cfg_scale": CFG_SCALE,
            "self_cond_cfg_scale": SELF_COND_CFG_SCALE,
            "time_schedule": TIME_SCHEDULE,
            "seed": SEED,
            "steps_cells": list(STEPS_CELLS),
            "batch_sizes": list(BATCH_SIZES),
            "sde_gamma": {str(k): v for k, v in SDE_GAMMA.items()},
        },
        "frozen_constants": {"TRIALS": TRIALS, "WARMUP": WARMUP},
        "event_taxonomy": EVENT_TAXONOMY,
        "coverage_events": list(COVERAGE_EVENTS),
    }


def main() -> None:
    args = parse_args()
    device = require_supported_device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    occupancy_at_start = require_idle_device(device)

    from unturtle_elf.loader import load_elf_model

    model = load_elf_model(device=device).eval()

    cells = []
    for steps in STEPS_CELLS:
        for batch in BATCH_SIZES:
            cell = profile_cell(model, steps, batch, device)
            cells.append(cell)
            detail = (
                f" verdict={cell['latency']['verdict_seconds']:.4f}s"
                if cell.get("latency")
                else f" ({cell.get('reason_code')})"
            )
            print(f"steps={steps:3d} batch={batch:3d} -> {cell['status']}{detail}")

    payload = {"run": provenance(args, occupancy_at_start), "cells": cells}
    target = out / "166-elf-generation-profile.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(cells)} cells to {target}")


if __name__ == "__main__":
    main()
