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

"""Dependency-sensitive parallel-decoding slice (#152; ParallelBench-inspired,
arXiv:2510.04767).

A compact deterministic fixture, not the external benchmark: three task
kinds whose outputs carry strongly coupled token dependencies —

- ``copy``       output must equal the source verbatim;
- ``reverse``    output must be the exact reversal;
- ``kv_recall``  output must be the queried values in query order.

The property that earns this slice its place (pinned in tests): an output
that is UNIGRAM-PERFECT — exactly the right tokens, dependency-breaking
order — scores near zero here while distributional quality metrics see
nothing wrong.  That is the failure mode parallel/any-order decoding can
hide behind generic metrics.

For externally produced tasks (licensing/provenance), a JSONL adapter
(:func:`load_external_dependency_records`) consumes records instead of
embedding the benchmark.  Scoring reports measurements only; pass/fail
gates belong to each experiment.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DependencyTask",
    "all_numeric_runs",
    "answer_span",
    "extract_numeric_answer",
    "dependency_length_diagnostics",
    "dependency_tasks",
    "load_external_dependency_records",
    "score_dependency_outputs",
]

_KINDS = ("copy", "reverse", "kv_recall")


@dataclass(frozen=True)
class DependencyTask:
    """One coupled-dependency task: prompt for the model, source tokens the
    coupling is defined over, and the exact target token sequence."""

    name: str
    kind: str
    prompt: str
    source: tuple[str, ...]
    target: tuple[str, ...]


def _tokens(rng: random.Random, length: int) -> list[str]:
    return [str(rng.randrange(10, 100)) for _ in range(length)]


def dependency_tasks(
    *, n_per_kind: int = 8, seed: int = 0, length: int = 8
) -> tuple[DependencyTask, ...]:
    """The deterministic fixture: ``n_per_kind`` tasks of each kind, fully
    determined by ``seed`` (and pinned so — the slice is a protocol
    artifact, not a sampler)."""
    rng = random.Random(seed)
    tasks: list[DependencyTask] = []
    for kind in _KINDS:
        for index in range(n_per_kind):
            if kind == "copy":
                source = _tokens(rng, length)
                target = list(source)
                prompt = f"Repeat exactly: {' '.join(source)}"
            elif kind == "reverse":
                source = _tokens(rng, length)
                target = list(reversed(source))
                prompt = f"Reverse the sequence: {' '.join(source)}"
            else:  # kv_recall
                keys = [f"k{position}" for position in range(length)]
                values = _tokens(rng, length)
                pairs = " ".join(
                    f"{key}={value}" for key, value in zip(keys, values, strict=True)
                )
                query_order = list(range(length))
                rng.shuffle(query_order)
                source = values
                target = [values[position] for position in query_order]
                queried = " ".join(keys[position] for position in query_order)
                prompt = f"Given {pairs}, output the values of: {queried}"
            tasks.append(
                DependencyTask(
                    name=f"{kind}-{seed}-{index}",
                    kind=kind,
                    prompt=prompt,
                    source=tuple(source),
                    target=tuple(target),
                )
            )
    return tuple(tasks)


def answer_span(suffix: list[int], *, eos_id: int) -> list[int]:
    """The semantic answer is the pre-EOS span — nothing after it (#157 step 3).

    The generated canvas is fixed width, so a finished answer is followed by
    EOS and then whatever filled the rest. Decoding the whole canvas with
    ``skip_special_tokens=True`` deletes the EOS markers and splices those
    fragments into one answer; that is what made #157 step 2's quality column
    unreadable. Scoring therefore stops at the FIRST EOS. The full raw suffix
    is kept for diagnostics, never for scoring.
    """
    return suffix[: suffix.index(eos_id)] if eos_id in suffix else list(suffix)


# Every target value in this fixture is a two-digit integer in [10, 99]; the
# schema-aware parser below depends on that and must be revisited if
# `dependency_tasks` ever emits a different width.
_TARGET_DIGITS = 2

# A block is a digit run plus in-line separators only. A LINE BREAK ends a
# block: "11 22\n33 44 55" is two blocks, not one, so rule 4 can choose
# between them. Spaces and list punctuation stay inside a block.
_NUMERIC_BLOCK = re.compile(r"[0-9][0-9 \t,.\u3001\u3002;:/|-]*[0-9]|[0-9]")


def _parse_numeric_run(run: str) -> tuple[list[str], int]:
    """Split one digit run into two-digit values (#157 step 3, frozen rule 3).

    A run of even length is split left-to-right into two-digit values, so a
    concatenated answer like ``7155843774274627`` recovers the same values as
    a comma-separated one. An odd-length run cannot be a sequence of two-digit
    values: it is NOT discarded — it is counted as invalid, because dropping it
    would hide a malformed answer and flatter the arm that produced it.
    """
    if len(run) % _TARGET_DIGITS != 0:
        return [], 1
    return [
        run[index : index + _TARGET_DIGITS]
        for index in range(0, len(run), _TARGET_DIGITS)
    ], 0


def extract_numeric_answer(text: str) -> dict[str, Any]:
    """Task-schema-aware final-numeric-block extraction (#157 step 3 PRIMARY).

    Frozen before re-scoring, and deliberately NOT ``re.findall(r"\d+")``:
    that picks up prose digits (the ``4`` of ``k4``) and leaves concatenated
    two-digit runs undefined.

    The rule, applied to the pre-EOS answer span only:

    1. normalize with Unicode NFKC;
    2. enumerate numeric blocks separated by prose;
    3. parse each block under the task schema (two-digit values; even-length
       runs split; odd-length runs kept as invalid);
    4. choose the block with the MOST parsed items, later block wins a tie;
    5. never truncate or pad to the target length — surplus and shortfall are
       passed to the scorer as they are.

    Selecting the block that best matches the target, or cutting to the first
    eight values, would hide wrong and extra answers. This picks by count
    alone and never consults the target.
    """
    normalized = unicodedata.normalize("NFKC", text)
    best: list[str] = []
    best_invalid = 0
    found = False
    for match in _NUMERIC_BLOCK.finditer(normalized):
        values: list[str] = []
        invalid = 0
        for run in re.findall(r"[0-9]+", match.group(0)):
            parsed, bad = _parse_numeric_run(run)
            values.extend(parsed)
            invalid += bad
        found = True
        # rule 4: most parsed items wins; ">=" makes the LATER block win ties.
        if len(values) >= len(best):
            best, best_invalid = values, invalid
    return {
        "values": tuple(best),
        "invalid_runs": best_invalid,
        "status": "ok" if found else "no_numeric_block",
    }


def all_numeric_runs(text: str) -> tuple[str, ...]:
    """Broad SECONDARY extraction over the whole pre-EOS span (sensitivity).

    Reported beside the primary rule, never used for a verdict. When the two
    disagree on an arm's qualitative standing, the cell is typed
    ``extraction_sensitive / undecidable`` rather than resolved by preferring
    whichever parser reads better.
    """
    return tuple(re.findall(r"[0-9]+", unicodedata.normalize("NFKC", text)))


def dependency_length_diagnostics(
    suffixes: list[list[int]], *, eos_id: int, mask_id: int
) -> dict[str, Any]:
    """Generation-length reporting that cannot cancel itself out (#157 step 3).

    ``no_eos_fraction`` is a SEPARATE column and the first-EOS statistics cover
    only EOS-bearing rows. Imputing ``1024`` for a row that never stopped would
    make "filled the whole canvas" and "stopped late" the same number — and the
    #157 preflight found exactly the pattern that destroys: the maskgit arms
    fill the canvas on ``copy`` while stopping in single digits on ``reverse``
    and ``kv_recall``. A single mean averages that to something unremarkable.

    Returns ``None`` for the first-EOS statistics when NO row carried an EOS,
    rather than inventing a position.
    """
    if not suffixes:
        raise ValueError(
            "length diagnostics over zero rows — record the failed cell "
            "instead of a fabricated length"
        )
    positions = [row.index(eos_id) for row in suffixes if eos_id in row]
    specials = {eos_id, mask_id}
    return {
        "row_count": len(suffixes),
        "no_eos_fraction": sum(1 for row in suffixes if eos_id not in row)
        / len(suffixes),
        "first_eos_mean_over_eos_rows": (
            statistics.fmean(positions) if positions else None
        ),
        "first_eos_median_over_eos_rows": (
            statistics.median(positions) if positions else None
        ),
        "eos_bearing_rows": len(positions),
        "mean_non_special_tokens": statistics.fmean(
            [sum(1 for token in row if token not in specials) for row in suffixes]
        ),
        "residual_mask_total": sum(row.count(mask_id) for row in suffixes),
    }


def score_dependency_outputs(
    tasks: tuple[DependencyTask, ...] | list[DependencyTask],
    outputs: list[list[str]] | list[tuple[str, ...]],
) -> dict[str, Any]:
    """Measurement over coupled targets — no gate, no verdict.

    - ``exact_match``: fraction of tasks reproduced exactly;
    - ``coupled_token_accuracy``: position-wise accuracy against the coupled
      target (the metric a unigram-perfect shuffle fails);
    - ``by_kind``: exact match per task kind;
    - ``length_mismatch_fraction``: flagged, not crashed.
    """
    if not tasks:
        raise ValueError(
            "score_dependency_outputs over zero tasks — an empty slice run "
            "has no scores; record the failed cell instead (#159 review)"
        )
    if len(outputs) != len(tasks):
        raise ValueError(
            f"got {len(outputs)} outputs for {len(tasks)} tasks; every task "
            "needs an output (emit an empty sequence for a failed generation)"
        )
    exact = 0
    position_correct = 0
    position_total = 0
    length_mismatches = 0
    kind_totals: dict[str, list[int]] = {}
    for task, output in zip(tasks, outputs, strict=True):
        output = tuple(output)
        is_exact = output == task.target
        exact += is_exact
        kind_totals.setdefault(task.kind, [0, 0])
        kind_totals[task.kind][0] += is_exact
        kind_totals[task.kind][1] += 1
        if len(output) != len(task.target):
            length_mismatches += 1
        # max(), not len(target): a correct prefix followed by rambling junk
        # must not score a perfect coupled accuracy (#159 review).
        position_total += max(len(output), len(task.target))
        position_correct += sum(
            produced == expected
            for produced, expected in zip(output, task.target, strict=False)
        )
    return {
        "exact_match": exact / len(tasks),
        "coupled_token_accuracy": position_correct / position_total,
        "by_kind": {
            kind: matched / total for kind, (matched, total) in kind_totals.items()
        },
        "length_mismatch_fraction": length_mismatches / len(tasks),
        "task_count": len(tasks),
    }


_REQUIRED_KEYS = ("name", "kind", "prompt", "source", "target")


def load_external_dependency_records(path: Any) -> tuple[DependencyTask, ...]:
    """JSONL adapter for externally produced dependency tasks — loud on any
    missing key; never guesses a field."""
    tasks = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            absent = [key for key in _REQUIRED_KEYS if key not in row]
            if absent:
                raise ValueError(
                    f"{path}:{line_number}: external dependency record is "
                    f"missing required key(s) {absent}; required: "
                    f"{list(_REQUIRED_KEYS)}"
                )
            tasks.append(
                DependencyTask(
                    name=row["name"],
                    kind=row["kind"],
                    prompt=row["prompt"],
                    source=tuple(row["source"]),
                    target=tuple(row["target"]),
                )
            )
    return tuple(tasks)
