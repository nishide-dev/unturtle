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
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DependencyTask",
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
        position_total += len(task.target)
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
