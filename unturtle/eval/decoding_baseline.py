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

"""#157 baseline — measurement discipline for the parallel-decoding frontier.

The Part-0 audit (`docs/parallel-decoding-reference-audit.md`) froze what a
baseline must produce *before* any number existed.  This module owns the
small shared pieces of that discipline so the producer script cannot drift
from the frozen protocol:

- the speed verdict is **wall-clock**; executed NFE is recorded as an
  explanatory variable and never as a denominator (per-forward efficiency can
  move opposite to wall-clock);
- commitment order comes from an **observed** commit trajectory, with the
  first commit kept separate from later revisions;
- `answer_before_reasoning_rate` is **unsupported**, not zero, on a fixture
  that declares no output spans — `dependency_slice`'s tasks are exactly that
  case;
- `threshold=None` is a **quota** policy (`floor(masked/steps)` with the
  remainder spread over the first steps), so multi-token steps are the norm
  and nothing is labelled one-token-per-step;
- the cache axis and the commit axis are separate fields, because a gain
  visible only on the diagonal is a commit gain wearing a cache label;
- `no_cache` is the **exact** reference path; only the two cache paths are
  approximate reuse.

Deliberately NOT here: any candidate implementation, threshold tuning, or a
universal cache/trace abstraction.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "CACHE_PATHS",
    "COMMIT_POLICIES",
    "answer_before_reasoning",
    "baseline_cell_key",
    "cache_path_class",
    "commit_order_metrics",
    "run_typed_cell",
    "speed_cell",
]

#: The cache axis of the frozen 2-D ablation grid, with the classification
#: the Part-0 audit established: `no_cache` recomputes a fresh full forward
#: every step and reuses no stale KV, so it is EXACT; the other two reuse
#: entries computed under an earlier masked context.
CACHE_PATHS = {
    "no_cache": "exact",
    "prefix_cache": "approximate_reuse",
    "dual_cache": "approximate_reuse",
}

#: The commit axis.  `quota` is `threshold=None`: `floor(masked/steps)` per
#: step with the remainder spread over the first steps — a step commits
#: SEVERAL tokens whenever masked > steps, so this is never
#: "one token per step".  A genuine one-token control would be a separate,
#: explicitly declared arm (Part-0 §4).
COMMIT_POLICIES = ("quota", "threshold")


def cache_path_class(cache_path: str) -> str:
    """`exact` or `approximate_reuse` for a cache path."""
    try:
        return CACHE_PATHS[cache_path]
    except KeyError:
        raise ValueError(
            f"unknown cache_path {cache_path!r}; the frozen paths are "
            f"{sorted(CACHE_PATHS)}"
        ) from None


def baseline_cell_key(*, cache_path: str, commit: str) -> dict[str, str]:
    """The two axes of a baseline cell, kept separate.

    Recording them as one label would make a diagonal-only gain
    indistinguishable from a cache gain — the confound the Part-0 audit's 2-D
    grid exists to prevent.
    """
    cache_path_class(cache_path)  # validates
    if commit not in COMMIT_POLICIES:
        raise ValueError(
            f"unknown commit policy {commit!r}; the frozen policies are "
            f"{list(COMMIT_POLICIES)} ('quota' is threshold=None, which "
            "commits several tokens per step — it is not one-token-per-step)"
        )
    return {"cache_path": cache_path, "commit": commit}


def speed_cell(
    *,
    wall_seconds: float,
    batch_size: int,
    executed_nfe: int | None,
    sequence_length: int,
) -> dict[str, Any]:
    """One timed cell: wall-clock verdict, NFE as an explanatory variable.

    The verdict metric is `samples_per_second` from measured wall time. NFE is
    carried for interpretation and is deliberately NOT used as a denominator:
    a path that needs more but cheaper forwards can be faster in seconds and
    worse per forward, so an NFE-normalized number can invert the verdict
    (#157 review B5).
    """
    if wall_seconds <= 0:
        raise ValueError(
            f"wall_seconds {wall_seconds} is not a measured duration; a cell "
            "with no elapsed time has no throughput"
        )
    if executed_nfe is None:
        raise ValueError(
            "executed_nfe is required: a requested step or token count is not "
            "evidence of what ran (#165)"
        )
    return {
        "wall_seconds": float(wall_seconds),
        "samples_per_second": batch_size / wall_seconds,
        "per_sample_latency": wall_seconds / batch_size,
        "batch_size": int(batch_size),
        "sequence_length": int(sequence_length),
        "executed_nfe": int(executed_nfe),
        "nfe_role": "explanatory",
    }


def commit_order_metrics(trajectory: list[Any], *, mask_id: int) -> dict[str, Any]:
    """Commitment order from an observed trajectory of committed states.

    ``trajectory`` is the sequence of committed-token snapshots, one per
    executed step plus the initial all-masked state.  A position's
    ``normalized_commit_step`` is the step at which it **first** stopped being
    ``mask_id``, divided by the executed step count; a later change to that
    position is a **revision**, counted separately.  The two answer different
    questions (when did the model decide, versus how much did it redo) and
    merging them would hide both.

    A position never committed is reported as ``None`` and counted, never
    imputed to the final step.
    """
    if len(trajectory) < 2:
        raise ValueError(
            "commit order needs at least two snapshots: the initial state and "
            "one step, otherwise no commit is observable"
        )
    import torch

    steps = len(trajectory) - 1
    first = trajectory[0]
    length = int(first.shape[-1])
    first_commit: list[int | None] = [None] * length
    per_step_counts: list[int] = []
    per_step_mean: list[float] = []
    per_step_std: list[float] = []
    revisions = 0

    previous = first
    for step_index in range(1, len(trajectory)):
        current = trajectory[step_index]
        newly: list[int] = []
        for position in range(length):
            was = previous[..., position]
            now = current[..., position]
            was_masked = bool((was == mask_id).all())
            now_masked = bool((now == mask_id).all())
            if was_masked and not now_masked:
                newly.append(position)
                if first_commit[position] is None:
                    first_commit[position] = step_index
            elif not was_masked and not now_masked and not bool((was == now).all()):
                revisions += 1
        per_step_counts.append(len(newly))
        if newly:
            positions = torch.tensor(newly, dtype=torch.float32)
            per_step_mean.append(float(positions.mean()))
            per_step_std.append(float(positions.std(unbiased=False)))
        else:
            per_step_mean.append(float("nan"))
            per_step_std.append(float("nan"))
        previous = current

    return {
        "steps_executed": steps,
        "normalized_commit_step": [
            (value / steps) if value is not None else None for value in first_commit
        ],
        "tokens_committed_per_step": per_step_counts,
        "committed_position_mean": per_step_mean,
        "committed_position_std": per_step_std,
        "revision_events": revisions,
        "uncommitted_positions": sum(1 for value in first_commit if value is None),
        # Never "one_token_per_step": under the quota policy a step commits
        # floor(masked/steps) tokens plus a remainder share (#157 review).
        "commit_policy_label": "observed",
    }


def answer_before_reasoning(
    *,
    normalized_commit_step: list[float | None],
    spans: dict[str, tuple[int, int]] | None,
) -> dict[str, Any]:
    """Whether the answer span committed before the reasoning span.

    Requires spans **declared by the task**, never inferred from generated
    text.  A fixture with no declared output spans — `dependency_slice`'s
    tasks, whose boundary is prompt-to-output rather than inside the output —
    returns `unsupported`, which #152 treats as data.  Returning 0 there would
    assert an ordering that was not measured.
    """
    if not spans:
        return {
            "status": "unsupported",
            "reason": "no task-declared output spans; the fixture provides no "
            "reasoning/answer boundary inside the output, so the ordering is "
            "not defined (never reported as 0)",
        }
    missing = [key for key in ("reasoning", "answer") if key not in spans]
    if missing:
        raise ValueError(
            f"spans must declare both 'reasoning' and 'answer'; missing {missing}"
        )

    summaries: dict[str, Any] = {}
    for name in ("reasoning", "answer"):
        start, end = spans[name]
        member_values = normalized_commit_step[start:end]
        committed = [value for value in member_values if value is not None]
        if not committed:
            return {
                "status": "excluded",
                "reason": f"the {name} span is empty or entirely uncommitted "
                f"({len(member_values)} positions, 0 committed); excluded with "
                "a reason rather than dropped",
            }
        summaries[name] = {
            "mean": sum(committed) / len(committed),
            "uncommitted": len(member_values) - len(committed),
        }
    return {
        "status": "ok",
        "reasoning_mean": summaries["reasoning"]["mean"],
        "answer_mean": summaries["answer"]["mean"],
        "reasoning_uncommitted": summaries["reasoning"]["uncommitted"],
        "answer_uncommitted": summaries["answer"]["uncommitted"],
        "answer_first": summaries["answer"]["mean"] < summaries["reasoning"]["mean"],
    }


def run_typed_cell(
    run: Callable[[int], Any],
    *,
    batch_size: int,
    unsupported: str | None = None,
) -> dict[str, Any]:
    """Run one cell, turning OOM and unsupported into typed data (#152).

    Any other exception propagates: a bug must not be recorded as a missing
    measurement.
    """
    import torch

    if unsupported is not None:
        return {"status": "unsupported", "reason": unsupported}
    try:
        value = run(batch_size)
    except torch.cuda.OutOfMemoryError as error:
        return {"status": "oom", "reason": str(error) or "CUDA out of memory"}
    return {"status": "ok", "value": value}
