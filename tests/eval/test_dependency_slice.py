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

"""#152 dependency-sensitive parallel-decoding slice (ParallelBench-inspired,
arXiv:2510.04767) — RED-first.

A compact deterministic fixture, not the external benchmark: tasks whose
outputs have strongly coupled token dependencies (exact copy, reversal,
key-value recall).  The load-bearing property, pinned below: an output that
is UNIGRAM-PERFECT (same tokens, wrong coupled order) scores near zero here
while generic distributional metrics see nothing wrong.  That separation is
what generic quality metrics miss about parallel decoding.

An adapter consumes externally produced task records (JSONL) for cases
where embedding external benchmark tasks is inappropriate.
"""

import json

import pytest


class TestDeterministicTasks:
    def test_tasks_are_deterministic_in_the_seed(self):
        """Compared by CONTENT (kind/source/target), not by name: the name
        embeds the seed, so a whole-task comparison would read 'different'
        even if the seed stopped driving the actual token content (a
        battery survivor caught exactly that)."""
        from unturtle.eval.dependency_slice import dependency_tasks

        def content(seed):
            return [
                (task.kind, task.source, task.target)
                for task in dependency_tasks(n_per_kind=4, seed=seed)
            ]

        assert content(0) == content(0)
        assert content(0) != content(1)

    def test_the_three_coupled_kinds_are_present(self):
        from unturtle.eval.dependency_slice import dependency_tasks

        tasks = dependency_tasks(n_per_kind=3, seed=0)
        assert len(tasks) == 9
        assert {task.kind for task in tasks} == {"copy", "reverse", "kv_recall"}

    def test_targets_actually_encode_the_coupling(self):
        from unturtle.eval.dependency_slice import dependency_tasks

        tasks = dependency_tasks(n_per_kind=2, seed=0)
        for task in tasks:
            if task.kind == "copy":
                assert task.target == task.source
            if task.kind == "reverse":
                assert task.target == tuple(reversed(task.source))
            if task.kind == "kv_recall":
                assert task.target  # the queried values, in query order

    def test_tasks_are_frozen(self):
        from unturtle.eval.dependency_slice import dependency_tasks

        task = dependency_tasks(n_per_kind=1, seed=0)[0]
        with pytest.raises(AttributeError):
            task.kind = "other"


class TestScoring:
    def test_perfect_outputs_score_one(self):
        from unturtle.eval.dependency_slice import (
            dependency_tasks,
            score_dependency_outputs,
        )

        tasks = dependency_tasks(n_per_kind=2, seed=0)
        scores = score_dependency_outputs(tasks, [task.target for task in tasks])
        assert scores["exact_match"] == 1.0
        assert scores["coupled_token_accuracy"] == 1.0
        assert all(kind_score == 1.0 for kind_score in scores["by_kind"].values())

    def test_unigram_perfect_but_order_broken_outputs_are_caught(self):
        """THE discriminating case: outputs containing exactly the right
        tokens in a dependency-breaking order.  Distributional metrics see
        nothing; the slice must."""
        from unturtle.eval.dependency_slice import (
            dependency_tasks,
            score_dependency_outputs,
        )

        tasks = [
            task
            for task in dependency_tasks(n_per_kind=4, seed=0)
            if task.kind == "reverse"
        ]
        # rotate instead of reversing: same unigrams, coupled order broken
        outputs = [task.source[1:] + task.source[:1] for task in tasks]
        for task, output in zip(tasks, outputs, strict=True):
            assert sorted(output) == sorted(task.target)  # unigram-perfect

        scores = score_dependency_outputs(tasks, outputs)
        assert scores["exact_match"] == 0.0
        assert scores["coupled_token_accuracy"] < 0.5

    def test_length_mismatches_are_flagged_not_crashed(self):
        from unturtle.eval.dependency_slice import (
            dependency_tasks,
            score_dependency_outputs,
        )

        tasks = dependency_tasks(n_per_kind=1, seed=0)
        outputs = [task.target[:-1] for task in tasks]  # truncated
        scores = score_dependency_outputs(tasks, outputs)
        assert scores["exact_match"] == 0.0
        assert scores["length_mismatch_fraction"] == 1.0

    def test_output_count_must_match_task_count(self):
        from unturtle.eval.dependency_slice import (
            dependency_tasks,
            score_dependency_outputs,
        )

        tasks = dependency_tasks(n_per_kind=1, seed=0)
        with pytest.raises(ValueError, match="outputs"):
            score_dependency_outputs(tasks, [])


class TestExternalAdapter:
    def test_external_records_load_and_score(self, tmp_path):
        """The licensing escape hatch: consume externally produced task
        records instead of embedding the benchmark."""
        from unturtle.eval.dependency_slice import (
            load_external_dependency_records,
            score_dependency_outputs,
        )

        path = tmp_path / "external.jsonl"
        rows = [
            {
                "name": "ext-0",
                "kind": "copy",
                "prompt": "copy: 5 3 9",
                "source": ["5", "3", "9"],
                "target": ["5", "3", "9"],
            }
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        tasks = load_external_dependency_records(path)
        assert tasks[0].kind == "copy"
        scores = score_dependency_outputs(tasks, [["5", "3", "9"]])
        assert scores["exact_match"] == 1.0

    def test_missing_keys_are_loud(self, tmp_path):
        from unturtle.eval.dependency_slice import load_external_dependency_records

        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"name": "x", "kind": "copy"}) + "\n")
        with pytest.raises(ValueError, match="target"):
            load_external_dependency_records(path)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
