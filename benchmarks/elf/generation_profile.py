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

#: How far summed event coverage may exceed 1.0 before a cell is invalid.
SHARE_TOLERANCE = 0.02

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
