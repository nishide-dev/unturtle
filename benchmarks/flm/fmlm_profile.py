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

"""#166 Stage 1 — FMLM generation profile (1-step and 32-step).

THE SOLE PRODUCER for `docs/artifacts/166-fmlm-profile.json`.

Execution scope, recorded in the artifact rather than assumed:

    request_concurrency: 1
    execution_mode: single_threaded_sequential
    observer_isolation: ContextVar
    observer_concurrency_contract: event isolation only
    rng_concurrency_contract: not deterministic across concurrent CPU-thread
                              requests

Those last two are separate facts. The ContextVar prevents observer EVENTS from
leaking into an unrelated execution. It does NOT make concurrent output
reproducible: `run_fmlm_request` seeds inside `torch.random.fork_rng`, which
forks the process-global CPU generator that threads share, so concurrent
requests interleave their draws. Measured with no observer installed anywhere.
That is the sampler's execution contract, not a defect of this profile, and it
is why everything here runs sequentially in one thread.

THE VERDICT IS THE INSTRUMENTATION-OFF OUTER WALL CLOCK. The ON pass only
attributes it; `covered_seconds` never substitutes for the verdict, and operation
shares are computed against the ON pass's own wall so an OFF measurement never
lands in a share denominator.

Diagnostics and timing are SEPARATE runs. `torch.get_rng_state()` and latent
hashing add allocation, copies and CPU work even when they advance no RNG, so
they belong only in the non-interference preflight — never inside a timed ON
trial, whose observer records CUDA events and nothing else.

Frozen configuration — the official FMLM cells (benchmarks/results/fmlm_owt_1,
fmlm_owt_32). gamma is 1.0 there, so the churn branch is the LIVE path:

    steps in {1, 32}   gamma 1.0   seed 100   max_length 1024   batch in {1,8,32}

Usage:
    .venv/bin/python benchmarks/flm/fmlm_profile.py --device cuda:0 \
        --out docs/artifacts
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

#: How far the summed event shares may exceed 1.0 before the cell is invalid.
#: Small positive slack only: CUDA event pairs are measured independently, so
#: rounding across five events can land marginally above the wall without
#: meaning the spans overlap.
SHARE_TOLERANCE = 0.02

#: Frozen measurement window. MODULE CONSTANTS, never CLI-settable: a verdict
#: that moves with a command-line flag is not a verdict. Three #166 gates failed
#: for exactly this reason.
TRIALS = 3
WARMUP = 2

#: Official FMLM cell configuration. Not overridable.
STEPS_CELLS = (1, 32)
GAMMA = 1.0
SEED = 100
MAX_LENGTH = 1024
BATCH_SIZES = (1, 8, 32)

#: Event taxonomy. The SPANS are part of the contract, not just the names —
#: moving `.exp()` or `randn_like` out of its scope leaves every count identical
#: while silently reattributing cost, so each description states its boundaries.
EVENT_TAXONOMY = {
    "grid_init": (
        "begins at the tau linspace; includes the initial randn draw of the "
        "[B, L, V] latent; ends once z exists. Does NOT include manual_seed or "
        "entry into fork_rng — those stay unattributed."
    ),
    "time_schedule": (
        "begins at tau_curr/tau_next indexing; includes _tau_to_t, "
        "sigma_target, sigma_tilde, t_tilde and the _t_to_tau inverse; ends "
        "once tau_tilde exists. NOT cheap scalar arithmetic despite appearances: "
        "_tau_to_t/_t_to_tau go through alpha_to_gamma/gamma_to_alpha, which "
        "run `lut(x.cpu().numpy())` — a device-to-host copy that SYNCHRONIZES "
        "the stream, a scipy CubicSpline evaluation on the host, then a copy "
        "back. Three such round trips per step (_tau_to_t twice, _t_to_tau "
        "once). Independently reproduced: 32 iterations of a busy stream cost "
        "~379 ms more with the round trip than without, against ~515 ms "
        "attributed to this event at steps=32 batch=1."
    ),
    "flow_map_forward": (
        "begins before model(...); includes the double-time model call, its "
        "return, and log_D_st_pred.exp(); ends after the exp result exists. "
        "The exponential is part of producing D_st_pred, not a later stage."
    ),
    "state_update": (
        "begins at the weight computation; includes z_tilde composition, the "
        "gamma churn arithmetic and torch.randn_like(z); ends after the new z "
        "exists. NOT reached on the final step."
    ),
    "endpoint_decode": ("the terminal argmax over the vocabulary axis."),
}


#: Per-trial expectations, gated on EVERY on-trial rather than on an aggregate:
#: an aggregate can match while individual trials are wrong.
def expected_counts(steps: int) -> dict[str, int]:
    return {
        "grid_init": 1,
        "time_schedule": steps,
        "flow_map_forward": steps,
        "state_update": max(steps - 1, 0),
        "endpoint_decode": 1,
    }


def expected_random_calls(steps: int) -> dict[str, int]:
    # gamma == 1.0, so every non-final step draws churn noise.
    return {"randn": 1, "randn_like": max(steps - 1, 0)}


def expected_event_order(steps: int) -> list[str]:
    order = ["grid_init"]
    for index in range(steps):
        order += ["time_schedule", "flow_map_forward"]
        if index != steps - 1:
            order.append("state_update")
    order.append("endpoint_decode")
    return order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def require_supported_device(device: str) -> None:
    """`cuda:0` only — CUDA events, peak stats and the GPU name all target the
    default device, so anything else is mis-recorded."""
    if device == "cuda:0":
        return
    raise SystemExit(
        f"--device {device!r} is not supported: this producer is cuda:0 only."
    )


class Request:
    """The public request shape. Diagnostic flags live on a PRIVATE attribute,
    so `kwargs` — the documented surface — carries only real decoding options."""

    def __init__(self, *, steps: int, num_samples: int, diagnostics=()):
        self.kwargs = {
            "steps": steps,
            "num_samples": num_samples,
            "seed": SEED,
            "gamma": GAMMA,
        }
        if diagnostics:
            self._unturtle_profile_diagnostics = frozenset(diagnostics)


class EventRecorder:
    """Counts and orders events. Used for GATING, not for timing."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.counts: dict[str, int] = {}
        self.depth = 0
        self.max_depth = 0
        self.unbalanced = False

    def __call__(self, name: str, phase: str) -> None:
        if phase == "enter":
            self.order.append(name)
            self.counts[name] = self.counts.get(name, 0) + 1
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
        else:
            self.depth -= 1
            if self.depth < 0:
                self.unbalanced = True


class CudaEventObserver:
    """Timed observer. Records CUDA event pairs and NOTHING else.

    No tensor fingerprints, no `.item()`, no `.cpu()`, no clones, no RNG
    capture: each of those would change what is being measured.

    Elapsed times are read after the ONE window-closing synchronize, which the
    CALLER performs inside its timed span so the wall clock contains the queue
    drain. `collect` itself never synchronizes and never blocks per event.
    """

    def __init__(self, torch_module) -> None:
        self._torch = torch_module
        self._open: list[tuple[str, Any, Any]] = []
        self._pairs: list[tuple[str, Any, Any]] = []
        self.calls: dict[str, int] = {}

    def __call__(self, name: str, phase: str) -> None:
        if phase == "enter":
            start = self._torch.cuda.Event(enable_timing=True)
            start.record()
            self._open.append((name, start, None))
            self.calls[name] = self.calls.get(name, 0) + 1
        else:
            open_name, start, _ = self._open.pop()
            end = self._torch.cuda.Event(enable_timing=True)
            end.record()
            self._pairs.append((open_name, start, end))

    def collect(self) -> dict[str, float]:
        if self._open:
            raise RuntimeError(
                f"{len(self._open)} scope(s) never closed; a partial window "
                "would attribute this run's time to the next one"
            )
        # No synchronize here: the caller performs the ONE window-closing
        # synchronize INSIDE its timed span, so the wall contains the drain. A
        # sync here would land outside the span and understate the wall again.
        seconds: dict[str, float] = {}
        for name, start, end in self._pairs:
            seconds[name] = seconds.get(name, 0.0) + start.elapsed_time(end) / 1000.0
        return seconds


def stable_hash(tensor) -> dict[str, Any]:
    """A hash plus the shape/dtype/device that make it meaningful.

    Storing the raw [B, 1024, V] latent in an artifact is not an option; a bare
    hash with no shape is not auditable.
    """
    import torch

    contiguous = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256(
        contiguous.view(torch.uint8).numpy().tobytes()
        if contiguous.dtype != torch.bool
        else contiguous.numpy().tobytes()
    ).hexdigest()
    return {
        "sha256": digest,
        "shape": list(contiguous.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def run_once(model, request, observer=None):
    """OFF and ON both go through the PUBLIC entry point.

    No duplicated loop, no private shortcut for the instrumented arm: the only
    difference between the arms is whether an observer is installed.
    """
    from unturtle_flm import sampler

    assert sampler._OBSERVER_CONTEXT.get() is None, (
        "an observer was already installed before this run: the seam leaked "
        "from an earlier run and this measurement would be contaminated"
    )
    token = sampler._install_observer(observer)
    try:
        return sampler.run_fmlm_request(model, request)
    finally:
        sampler._restore_observer(token)
        assert sampler._OBSERVER_CONTEXT.get() is None, "the seam leaked"


def non_interference_preflight(model, steps: int, batch: int) -> dict[str, Any]:
    """Sequential OFF/ON comparison on the REAL checkpoint.

    Deliberately separate from timing: the captures here allocate and copy, so
    including them in a timed trial would measure the diagnostics.
    """
    import torch

    diagnostics = ("terminal_rng", "final_latent")
    arms: dict[str, Any] = {}
    for arm, observer in (("off", None), ("on", EventRecorder())):
        request = Request(steps=steps, num_samples=batch, diagnostics=diagnostics)
        result = run_once(model, request, observer)
        record: dict[str, Any] = {
            "tokens": stable_hash(result["tokens"]),
            "final_latent": stable_hash(result["_final_latent"]),
            "terminal_rng_cpu": stable_hash(result["_terminal_rng"]["cpu"]),
            "executed": result["executed"],
        }
        if "cuda" in result["_terminal_rng"]:
            record["terminal_rng_cuda"] = stable_hash(result["_terminal_rng"]["cuda"])
        arms[arm] = record
        if isinstance(observer, EventRecorder):
            arms["on_events"] = {
                "counts": observer.counts,
                "order": observer.order,
                "balanced": not observer.unbalanced and observer.depth == 0,
            }
        del result
    off, on = arms["off"], arms["on"]
    matches = {
        "tokens": off["tokens"] == on["tokens"],
        "final_latent": off["final_latent"] == on["final_latent"],
        "terminal_rng_cpu": off["terminal_rng_cpu"] == on["terminal_rng_cpu"],
        "executed_metadata": off["executed"] == on["executed"],
    }
    if "terminal_rng_cuda" in off:
        matches["terminal_rng_cuda"] = (
            off["terminal_rng_cuda"] == on["terminal_rng_cuda"]
        )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        # BOTH sides are recorded, not only the verdict, so the comparison is
        # auditable after the fact.
        "off": off,
        "on": on,
        "on_events": arms["on_events"],
        "matches": matches,
        "status": "ok" if all(matches.values()) else "observer_interference",
    }


def random_call_preflight(model, steps: int, batch: int) -> dict[str, int]:
    """Count `randn` / `randn_like` for ONE request.

    The patch is process-global, so it is installed immediately before a single
    run and restored in `finally`. Never present during a timed trial.
    """
    import torch

    counts = {"randn": 0, "randn_like": 0}
    original_randn, original_like = torch.randn, torch.randn_like

    def counting_randn(*args, **kwargs):
        counts["randn"] += 1
        return original_randn(*args, **kwargs)

    def counting_like(*args, **kwargs):
        counts["randn_like"] += 1
        return original_like(*args, **kwargs)

    torch.randn, torch.randn_like = counting_randn, counting_like
    try:
        run_once(model, Request(steps=steps, num_samples=batch))
    finally:
        torch.randn, torch.randn_like = original_randn, original_like
    return counts


def timed_off(model, steps: int, batch: int) -> dict[str, Any]:
    """The VERDICT pass: outer wall clock with no instrumentation at all."""
    import time

    import torch

    request = Request(steps=steps, num_samples=batch)
    for _ in range(WARMUP):
        run_once(model, request)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    walls = []
    for _ in range(TRIALS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_once(model, request)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - start)
    return {
        "wall_seconds_median": statistics.median(walls),
        "wall_seconds_trials": walls,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def timed_on(model, steps: int, batch: int) -> dict[str, Any]:
    """The ATTRIBUTION pass. Shares are computed against THIS pass's wall."""
    import time

    import torch

    request = Request(steps=steps, num_samples=batch)
    for _ in range(WARMUP):
        run_once(model, request, CudaEventObserver(torch))

    trials = []
    for _ in range(TRIALS):
        observer = CudaEventObserver(torch)
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_once(model, request, observer)
        # Generation is ASYNC: `run_once` returns while kernels are still in
        # flight. Reading the wall here — before the queue drains — made the ON
        # wall SHORTER than the CUDA-event total it is supposed to contain,
        # producing negative instrumentation overhead in 3 of 5 cells and event
        # shares summing over 100%. The single window-closing synchronize must
        # happen INSIDE the timed span.
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        seconds = observer.collect()
        trials.append(
            {
                "wall_seconds": wall,
                "event_seconds": seconds,
                "calls": dict(observer.calls),
            }
        )
    return {"trials": trials}


def gate_trial(steps: int, calls: dict[str, int]) -> list[str]:
    """Per-trial event-count gate. Returns the problems, empty if clean."""
    problems = []
    expected = expected_counts(steps)
    for name, want in expected.items():
        got = calls.get(name, 0)
        if got != want:
            problems.append(f"{name}: {got} calls, expected {want}")
    for name in calls:
        if name not in expected:
            problems.append(f"{name}: not in the frozen taxonomy")
    return problems


def assemble_events(steps: int, trials: list[dict]) -> list[dict[str, Any]]:
    """Per-event rows. `state_update` at steps=1 is a STRUCTURAL zero.

    No artificial enter/exit is emitted for it — fabricating a boundary that
    does not exist would be worse than recording the zero. The row is supplied
    here, at assembly time, and flagged.
    """
    rows = []
    for name in EVENT_TAXONOMY:
        per_trial = [trial["event_seconds"].get(name, 0.0) for trial in trials]
        calls = [trial["calls"].get(name, 0) for trial in trials]
        structural = expected_counts(steps).get(name, 0) == 0
        row: dict[str, Any] = {
            "name": name,
            "calls": calls[0] if calls else 0,
            "seconds": statistics.median(per_trial) if per_trial else 0.0,
            "description": EVENT_TAXONOMY[name],
        }
        if structural:
            row |= {
                "calls": 0,
                "seconds": 0.0,
                "structural_zero": True,
                "reason": "final-step branch exits before state update",
            }
        else:
            # Share PER TRIAL, then the median of the shares. Dividing a median
            # event time by a median wall is the same median-of-sums error the
            # residual had one level up: the per-event medians can come from
            # different trials, so the shares summed OVER 100% in 3 of 5 cells.
            per_trial_shares = [
                trial["event_seconds"].get(name, 0.0) / trial["wall_seconds"]
                for trial in trials
                if trial["wall_seconds"] > 0
            ]
            row["share_of_on_wall"] = (
                statistics.median(per_trial_shares) if per_trial_shares else None
            )
        rows.append(row)
    return rows


def classify_failure(error: BaseException) -> str | None:
    """OOM only. A shape or device error is a different defect and must not be
    filed as a capacity limit (the #166 row-5 lesson)."""
    import torch

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "cuda_out_of_memory"
    text = str(error).lower()
    if isinstance(error, RuntimeError) and "out of memory" in text:
        return "cuda_out_of_memory"
    return None


def cleanup() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def failure_record(
    *,
    stage: str,
    reason_code: str | None,
    timing_attempted: bool,
    status: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """The typed failure disposition, built in ONE place.

    Every unmeasured field is null, never zero: a 0.0 latency or an empty event
    list reads as "measured, nothing there". `status` defaults to
    `measurement_invalid`; only a CLASSIFIED capacity failure is `oom`.
    """
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
        **extra,
    }


def overhead_estimate(off_walls: list[float], on_walls: list[float]) -> dict[str, Any]:
    """The ON-OFF difference, reported against its own noise floor.

    A signed point estimate is not publishable here: with TRIALS=3 the OFF
    trials alone spread by up to 16%, and the measured difference lands inside
    that, so its SIGN is an artifact of which trial happened to be the median.
    Reporting `seconds` without `resolvable` invites reading noise as a real
    instrumentation cost — or, worse, as instrumentation making the code faster.
    """
    off_median = statistics.median(off_walls)
    on_median = statistics.median(on_walls)
    difference = on_median - off_median
    off_spread = (max(off_walls) - min(off_walls)) if off_walls else 0.0
    return {
        "seconds": difference,
        "off_median_seconds": off_median,
        "on_median_seconds": on_median,
        "off_spread_seconds": off_spread,
        "resolvable": abs(difference) > off_spread,
        "basis": (
            "on_wall_median - off_wall_median, compared against the OFF trial "
            "spread; `resolvable` is False when the difference is smaller than "
            "the spread, in which case the sign carries no information"
        ),
    }


def gate_trials(steps: int, trials: list[dict]) -> list[str]:
    """Gate EVERY trial, not an aggregate: two trials can be individually wrong
    while their totals look right."""
    problems: list[str] = []
    if not trials:
        problems.append("no on-trials were recorded to gate")
    for index, trial in enumerate(trials):
        for problem in gate_trial(steps, trial["calls"]):
            problems.append(f"on_trial[{index}]: {problem}")
    return problems


def profile_cell(model, steps: int, batch: int) -> dict[str, Any]:
    """One (steps, batch) cell. An OOM types THIS cell and does not abort the
    producer: the latent is [B, 1024, V], so large batches are genuinely heavy
    and one capacity limit must not cost the other cells."""
    cell: dict[str, Any] = {
        "steps": steps,
        "batch_size": batch,
        "gamma": GAMMA,
        "seed": SEED,
        "max_length": MAX_LENGTH,
    }
    stage = "preflight"
    try:
        interference = non_interference_preflight(model, steps, batch)
        cell["non_interference"] = interference
        if interference["status"] != "ok":
            return cell | failure_record(
                stage="non_interference_preflight",
                reason_code="observer_interference",
                timing_attempted=False,
            )
        counts = random_call_preflight(model, steps, batch)
        cell["random_calls"] = {
            "observed": counts,
            "expected": expected_random_calls(steps),
            "matches": counts == expected_random_calls(steps),
        }
        if not cell["random_calls"]["matches"]:
            return cell | failure_record(
                stage="random_call_preflight",
                reason_code="random_call_count_mismatch",
                timing_attempted=False,
            )
        order = interference["on_events"]["order"]
        cell["event_order"] = {
            "observed_length": len(order),
            "matches_expected": order == expected_event_order(steps),
        }
        if not cell["event_order"]["matches_expected"]:
            return cell | failure_record(
                stage="event_order_preflight",
                reason_code="event_order_mismatch",
                timing_attempted=False,
            )

        stage = "warmup"
        cleanup()
        stage = "off_trial"
        off = timed_off(model, steps, batch)
        stage = "on_trial"
        on = timed_on(model, steps, batch)
    except Exception as error:  # noqa: BLE001 - classified, then re-reported
        reason = classify_failure(error)
        cleanup()
        return cell | failure_record(
            stage=stage,
            reason_code=reason,
            timing_attempted=stage in ("off_trial", "on_trial"),
            exception_class=type(error).__name__,
            exception_message=str(error)[:300],
        )

    problems = gate_trials(steps, on["trials"])
    if problems:
        return cell | failure_record(
            stage="on_trial",
            reason_code="per_trial_event_count_mismatch",
            timing_attempted=True,
            problems=problems,
        )

    on_walls = [trial["wall_seconds"] for trial in on["trials"]]
    events = assemble_events(steps, on["trials"])
    on_median = statistics.median(on_walls)

    # Over-coverage gate on the SHARES, independent of the residual gate: the
    # residual is computed from summed seconds, so it can look healthy while the
    # per-event shares still exceed the wall.
    # Per-trial coverage ratio, then the median. Summing per-event MEDIAN
    # shares does not preserve any trial's total — medians of different events
    # come from different trials — so that sum is not a coverage figure at all.
    coverage_per_trial = [
        sum(trial["event_seconds"].values()) / trial["wall_seconds"]
        for trial in on["trials"]
        if trial["wall_seconds"] > 0
    ]
    share_total = statistics.median(coverage_per_trial) if coverage_per_trial else 0.0
    if share_total > 1.0 + SHARE_TOLERANCE:
        cleanup()
        return cell | failure_record(
            stage="on_trial",
            reason_code="event_shares_exceed_wall",
            timing_attempted=True,
            status="profile_invalid",
            problems=[
                f"median per-trial event coverage is {share_total:.4f} of the ON wall",
                f"per-trial coverage: {coverage_per_trial}",
                "the event spans overlap or the wall excludes work the events cover",
            ],
        )

    # Residual PER TRIAL, then the median of residuals. Summing medians and
    # subtracting the median wall is the "residual as a difference of medians"
    # error: median-of-sums != sum-of-medians, so the two can cross and produce
    # a NEGATIVE unattributed time. Measured — three of five cells did exactly
    # that before this was fixed.
    attributed_per_trial = [
        sum(trial["event_seconds"].values()) for trial in on["trials"]
    ]
    residuals = [
        wall - attributed
        for wall, attributed in zip(on_walls, attributed_per_trial, strict=True)
    ]
    attributed = statistics.median(attributed_per_trial)
    unattributed = statistics.median(residuals)
    if unattributed < 0:
        # Over-coverage means the event spans overlap or the clock is wrong.
        # Never clamped to zero: a clamped residual hides a broken measurement.
        cleanup()
        return cell | failure_record(
            stage="on_trial",
            reason_code="negative_unattributed_seconds",
            timing_attempted=True,
            status="profile_invalid",
            problems=[
                f"trial residuals (seconds): {residuals}",
                "attributed CUDA-event time exceeds the wall clock, so the "
                "event spans overlap or are mis-recorded",
            ],
        )
    cleanup()
    return cell | {
        "status": "ok",
        "timing_attempted": True,
        # THE VERDICT: the instrumentation-OFF outer wall clock.
        "latency": {
            "verdict_seconds": off["wall_seconds_median"],
            "verdict_basis": "instrumentation_off_outer_wall_clock",
            "off_wall_trials": off["wall_seconds_trials"],
            "on_wall_median": on_median,
            "on_wall_trials": on_walls,
            # The ON-OFF difference is reported WITH the noise floor that
            # decides whether it means anything. At TRIALS=3 the OFF spread is
            # 1.4-16.2% while the difference is -1.4% to +10.1%, so the sign is
            # not resolvable and a bare signed number would read as a measured
            # slowdown or speedup. `resolvable` is False whenever the magnitude
            # sits inside the OFF spread.
            "instrumentation_overhead": overhead_estimate(
                off["wall_seconds_trials"], on_walls
            ),
        },
        "peak_memory": {
            "allocated_bytes": off["peak_memory_bytes"],
            "reserved_bytes": off["peak_reserved_bytes"],
            "basis": "instrumentation_off_pass",
        },
        "events": events,
        "attribution": {
            # Shares are ON-pass internal; the OFF wall never enters a
            # denominator.
            "denominator": "on_wall_median",
            "attributed_seconds": attributed,
            "unattributed_seconds": unattributed,
            "coverage_ratio": share_total,
            "coverage_basis": "median of per-trial (sum of event seconds / wall)",
            "coverage_note": (
                "the per-event `share_of_on_wall` values are per-trial medians "
                "and do NOT sum to this ratio: each event's median can come "
                "from a different trial, so their sum is not a coverage figure"
            ),
            "unattributed_basis": "median of per-trial (wall - attributed)",
            "unattributed_trials": residuals,
            "unattributed_note": (
                "fork_rng enter/exit, manual_seed, loop control, observer "
                "dispatch, result assembly and executed-metadata construction "
                "are deliberately left unattributed rather than folded into an "
                "adjacent event"
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
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "transformers": version("transformers"),
    }


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    from unturtle_flm.loader import FMLM_CHECKPOINT, FMLM_REVISION

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
        "execution_scope": {
            "request_concurrency": 1,
            "execution_mode": "single_threaded_sequential",
            "observer_isolation": "ContextVar",
            "observer_concurrency_contract": "event isolation only",
            "rng_concurrency_contract": (
                "not deterministic across concurrent CPU-thread requests: "
                "run_fmlm_request seeds inside torch.random.fork_rng, which "
                "forks the process-global CPU generator that threads share. "
                "Measured with no observer installed. This is the sampler's "
                "execution contract, not a defect of this profile."
            ),
        },
        "fixture": {
            "checkpoint": f"{FMLM_CHECKPOINT}@{FMLM_REVISION}",
            "gamma": GAMMA,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "steps_cells": list(STEPS_CELLS),
            "batch_sizes": list(BATCH_SIZES),
            "official_cells": [
                "benchmarks/results/fmlm_owt_1",
                "benchmarks/results/fmlm_owt_32",
            ],
        },
        "frozen_constants": {"TRIALS": TRIALS, "WARMUP": WARMUP},
        "gamma_semantics": {
            "gamma": 1.0,
            "sqrt_one_minus_gamma_squared": 0.0,
            "sigma_tilde": 0.0,
            "t_tilde": 1.0,
            "weight_z": 0.0,
            "weight_D": 1.0,
            "note": (
                "the z_tilde expression ALGEBRAICALLY reduces to D_st_pred at "
                "gamma=1. This is a property of the formula, not an asserted "
                "bit-identity of the computed tensors. The state update is NOT "
                "degenerate bookkeeping: it still performs (steps-1) "
                "randn_like draws and full-size [B, L, V] latent arithmetic, "
                "which is why it is measured rather than assumed free."
            ),
        },
        "event_taxonomy": EVENT_TAXONOMY,
    }


def main() -> None:
    args = parse_args()
    require_supported_device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from unturtle_flm.loader import load_fmlm_model

    model = load_fmlm_model(device=args.device).eval()

    cells = []
    # Deterministic order, so the artifact diff is stable across runs.
    for steps in STEPS_CELLS:
        for batch in BATCH_SIZES:
            cell = profile_cell(model, steps, batch)
            cells.append(cell)
            print(
                f"steps={steps:3d} batch={batch:3d} -> {cell['status']}"
                + (
                    f" verdict={cell['latency']['verdict_seconds']:.4f}s"
                    if cell.get("latency")
                    else f" ({cell.get('reason_code')})"
                )
            )

    payload = {"run": provenance(args), "cells": cells}
    target = out / "166-fmlm-profile.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {len(cells)} cells to {target}")


if __name__ == "__main__":
    main()
