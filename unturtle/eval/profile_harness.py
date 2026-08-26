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

"""#166 Stage-1 profiling harness — coverage arithmetic and cell envelope.

Implements the frozen protocol in ``docs/acceleration-profile-protocol.md``.
Deliberately NOT a general-purpose profiler API: this module owns the record
envelope and the coverage arithmetic, and the operation taxonomy stays
family-local in each benchmark.

The load-bearing rule is that **the verdict is the outer wall-clock with
instrumentation off**, and per-operation timings only explain where the time
went. The taxonomies are nested — hybrid's ``attention_path`` sits inside
``full_model_forward`` — so summing inclusive times across a parent and its
child double-counts, inflating coverage and driving the unattributed remainder
negative. That failure is worst exactly when the arithmetic looks best, so
coverage is computed only over mutually exclusive intervals and the invariants
below are enforced rather than assumed.
"""

from __future__ import annotations

import math
import numbers
import statistics
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "COVERAGE_TOLERANCE",
    "OperationEvent",
    "ProfileCell",
    "coverage_seconds",
    "profile_cell",
    "trial_statistics",
]

#: Slack allowed before over-coverage is called a bookkeeping error rather
#: than timer noise. Coverage legitimately lands a hair above the outer wall
#: time when a synchronize boundary falls between the two reads.
COVERAGE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class OperationEvent:
    """One measured operation.

    ``inclusive_seconds`` is for reading a single event's total cost and must
    never be summed across a parent and its child. Only ``coverage_eligible``
    events contribute to coverage, and a parent and its child are never both
    eligible — that is the invariant this type exists to make checkable.
    """

    name: str
    inclusive_seconds: float
    call_count: int
    parent: str | None = None
    coverage_eligible: bool = True
    exclusive_seconds: float | None = None

    def coverage_contribution(self) -> float:
        """What this event contributes to ``covered_seconds``.

        Exclusive time when it is known, inclusive otherwise: an eligible leaf
        has no children, so the two coincide.
        """
        if self.exclusive_seconds is not None:
            return self.exclusive_seconds
        return self.inclusive_seconds


def _validate_tree(events: list[OperationEvent]) -> dict[str, OperationEvent]:
    """Refuse an event set whose parent declarations cannot be trusted.

    Checked because coverage is computed FROM these declarations: a dangling
    parent, a duplicate name or a cycle would make the exclusivity check
    vacuous rather than wrong-looking.
    """
    by_name: dict[str, OperationEvent] = {}
    for event in events:
        if event.name in by_name:
            raise ValueError(
                f"duplicate event name {event.name!r}: parent references are "
                "resolved by name, so duplicates make the tree ambiguous"
            )
        by_name[event.name] = event
    for event in events:
        if event.parent is not None and event.parent not in by_name:
            raise ValueError(
                f"event {event.name!r} declares parent {event.parent!r}, which "
                "is not in this cell's events; coverage cannot be validated "
                "against a missing ancestor"
            )
    for event in events:
        seen = {event.name}
        cursor = event.parent
        while cursor is not None:
            if cursor in seen:
                raise ValueError(
                    f"parent cycle through {cursor!r}: the event tree must be "
                    "acyclic for ancestor checks to terminate"
                )
            seen.add(cursor)
            cursor = by_name[cursor].parent
    return by_name


def coverage_seconds(events: list[OperationEvent]) -> float:
    """Sum over mutually exclusive intervals only.

    Refuses an eligible event that has ANY eligible ancestor, not merely an
    eligible direct parent. The grandparent case is the one that bites:

        root    eligible
        └─ middle   not eligible
           └─ leaf  eligible

    `leaf`'s direct parent is ineligible, so a direct-parent check passes while
    `root` and `leaf` both contribute and the leaf's time is counted twice.
    """
    by_name = _validate_tree(events)
    for event in events:
        if not event.coverage_eligible:
            continue
        cursor = event.parent
        while cursor is not None:
            ancestor = by_name[cursor]
            if ancestor.coverage_eligible:
                raise ValueError(
                    f"event {event.name!r} and its ancestor {ancestor.name!r} "
                    "are both coverage_eligible: summing them would count the "
                    "same time twice. Exactly one level of any path may be "
                    "eligible."
                )
            cursor = ancestor.parent
    return sum(
        event.coverage_contribution() for event in events if event.coverage_eligible
    )


def trial_statistics(seconds: list[float]) -> dict[str, Any]:
    """Median with range — never a single trial (#166 measurement rules)."""
    if not seconds:
        raise ValueError("no trials recorded; an empty cell is not a measurement")
    return {
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "trials": len(seconds),
        "single_trial": len(seconds) == 1,
    }


@dataclass
class ProfileCell:
    """One (family, operation-set, batch, length, dtype) measurement."""

    family: str
    cell: str
    batch_size: int
    sequence_length: int
    dtype: str
    #: Per-trial wall times for BOTH arms. The verdict is derived from
    #: `wall_off_trials` rather than accepted as a scalar: a producer could
    #: otherwise submit three valid trials and inject an unrelated number as the
    #: verdict, and instrumentation overhead would be a replicated figure minus
    #: an arbitrary one.
    wall_off_trials: list[float] = field(default_factory=list)
    wall_on_trials: list[float] = field(default_factory=list)
    events: list[OperationEvent] = field(default_factory=list)
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    warmup_seconds: float | None = None
    hardware: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _validate_finite(cell: ProfileCell) -> list[str]:
    """Reject non-finite and negative measurements.

    Without this a `nan` wall time silently produces `status: ok`: every
    comparison against NaN is False, so the coverage and remainder checks pass
    vacuously. A `-inf` event time does the same to `covered_seconds`. These are
    measurement failures, and a harness must say so rather than publish them.
    """
    problems: list[str] = []
    for name, trials in (
        ("wall_off_trials", cell.wall_off_trials),
        ("wall_on_trials", cell.wall_on_trials),
    ):
        for index, value in enumerate(trials):
            if not math.isfinite(value):
                problems.append(f"{name}[{index}] is not finite ({value!r})")
            elif value < 0:
                problems.append(f"{name}[{index}] is negative ({value!r})")
    for event in cell.events:
        for field_name, value in (
            ("inclusive_seconds", event.inclusive_seconds),
            ("exclusive_seconds", event.exclusive_seconds),
        ):
            if value is None:
                continue
            if not math.isfinite(value):
                problems.append(
                    f"event {event.name!r} {field_name} is not finite ({value!r})"
                )
            elif value < 0:
                problems.append(
                    f"event {event.name!r} {field_name} is negative ({value!r})"
                )
        # A count, not a measurement: the annotation is not enforced at
        # runtime, and `nan < 0` is False, so a bare sign check lets NaN, inf
        # and fractional counts through as `ok`. `bool` is excluded because
        # `True` is an Integral that would silently read as one call.
        if isinstance(event.call_count, bool) or not isinstance(
            event.call_count, numbers.Integral
        ):
            problems.append(
                f"event {event.name!r} call_count must be a non-negative "
                f"integer, got {event.call_count!r} ({type(event.call_count).__name__})"
            )
        elif event.call_count < 0:
            problems.append(
                f"event {event.name!r} call_count is negative ({event.call_count})"
            )
        # Exclusive time feeds coverage directly, so an exclusive value above
        # its own inclusive total is over-coverage entering through a channel
        # the outer-wall check cannot see: it stays under the wall time while
        # inflating `covered_seconds`.
        if (
            event.exclusive_seconds is not None
            and math.isfinite(event.exclusive_seconds)
            and math.isfinite(event.inclusive_seconds)
            and event.exclusive_seconds > event.inclusive_seconds + COVERAGE_TOLERANCE
        ):
            problems.append(
                f"event {event.name!r} exclusive_seconds "
                f"({event.exclusive_seconds}) exceeds its inclusive_seconds "
                f"({event.inclusive_seconds}): exclusive time is a subset of "
                "inclusive time by definition"
            )
    return problems


def profile_cell(cell: ProfileCell) -> dict[str, Any]:
    """The record for one cell, with coverage typed rather than trusted.

    ``wall_seconds_instrumented_off`` is the verdict; the instrumented run
    exists to attribute it, and the difference between them is reported rather
    than hidden. A negative unattributed remainder is NEVER clamped to zero:
    clamping would conceal exactly the double-counting this arithmetic guards,
    so it types the cell ``profile_invalid`` instead.
    """
    status = "ok"
    reasons: list[str] = []

    finite_problems = _validate_finite(cell)
    if finite_problems:
        status = "measurement_invalid"
        reasons.extend(finite_problems)

    covered = coverage_seconds(cell.events)

    for name, trials in (
        ("wall_off_trials", cell.wall_off_trials),
        ("wall_on_trials", cell.wall_on_trials),
    ):
        if not trials:
            status = (
                "measurement_invalid"
                if status == "measurement_invalid"
                else "profile_invalid"
            )
            reasons.append(
                f"{name} is empty: the verdict and the instrumented time are "
                "derived from replicated trials, never accepted as scalars"
            )
        elif len(trials) == 1:
            status = (
                "measurement_invalid"
                if status == "measurement_invalid"
                else "profile_invalid"
            )
            reasons.append(
                f"{name} has a single trial: the protocol requires replicated, "
                "interleaved trials before any performance statement"
            )

    off_stats = trial_statistics(cell.wall_off_trials) if cell.wall_off_trials else None
    on_stats = trial_statistics(cell.wall_on_trials) if cell.wall_on_trials else None
    wall_off = off_stats["median_seconds"] if off_stats else float("nan")
    wall_on = on_stats["median_seconds"] if on_stats else float("nan")
    unattributed = wall_on - covered
    if on_stats is not None and covered > wall_on + COVERAGE_TOLERANCE:
        status = (
            "measurement_invalid"
            if status == "measurement_invalid"
            else "profile_invalid"
        )
        reasons.append(
            f"covered_seconds ({covered:.6f}) exceeds instrumented wall time "
            f"({wall_on:.6f}) beyond tolerance: the exclusivity declaration "
            "is wrong"
        )
    if on_stats is not None and unattributed < -COVERAGE_TOLERANCE:
        status = (
            "measurement_invalid"
            if status == "measurement_invalid"
            else "profile_invalid"
        )
        reasons.append(
            f"unattributed_seconds is negative ({unattributed:.6f}); reported "
            "as-is rather than clamped, because clamping hides the bookkeeping "
            "error that produced it"
        )
    return {
        "family": cell.family,
        "cell": cell.cell,
        "batch_size": cell.batch_size,
        "sequence_length": cell.sequence_length,
        "dtype": cell.dtype,
        "hardware": cell.hardware,
        # Both are trial medians, so the overhead is a replicated-vs-replicated
        # difference rather than a difference against an arbitrary scalar.
        "wall_seconds_instrumented_off": wall_off,
        "wall_seconds_instrumented_on": wall_on,
        "instrumentation_overhead_seconds": wall_on - wall_off,
        "wall_off_trials": off_stats,
        "wall_on_trials": on_stats,
        "covered_seconds": covered,
        "unattributed_seconds": unattributed,
        "verdict_source": "wall_seconds_instrumented_off",
        "warmup_seconds": cell.warmup_seconds,
        "peak_allocated_bytes": cell.peak_allocated_bytes,
        "peak_reserved_bytes": cell.peak_reserved_bytes,
        "operations": [
            {
                "name": event.name,
                "inclusive_seconds": event.inclusive_seconds,
                "exclusive_seconds": event.exclusive_seconds,
                "call_count": event.call_count,
                "parent": event.parent,
                "coverage_eligible": event.coverage_eligible,
            }
            for event in cell.events
        ],
        "status": status,
        "invalid_reasons": reasons,
        # Namespaced, NOT spread: a producer passing `extra={"status": "ok"}`
        # would otherwise overwrite a `profile_invalid` verdict the core just
        # computed, and zero out `covered_seconds` with it.
        "extra": cell.extra,
    }
