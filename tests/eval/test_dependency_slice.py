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

    def test_over_length_output_is_penalized(self):
        """Review F3 (#159): the right prefix followed by rambling junk
        previously scored coupled_token_accuracy == 1.0 — the denominator
        must cover max(len(output), len(target))."""
        from unturtle.eval.dependency_slice import (
            dependency_tasks,
            score_dependency_outputs,
        )

        tasks = [
            task
            for task in dependency_tasks(n_per_kind=1, seed=0)
            if task.kind == "copy"
        ]
        outputs = [task.target + ("junk",) * len(task.target) for task in tasks]
        scores = score_dependency_outputs(tasks, outputs)
        assert scores["exact_match"] == 0.0
        assert scores["coupled_token_accuracy"] == pytest.approx(0.5)
        assert scores["length_mismatch_fraction"] == 1.0

    def test_empty_task_list_is_loud(self):
        """Review F5: an empty slice run raises actionably, not
        ZeroDivisionError."""
        from unturtle.eval.dependency_slice import score_dependency_outputs

        with pytest.raises(ValueError, match="task"):
            score_dependency_outputs([], [])

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


# --- #157 step 3: answer span and per-kind length diagnostics ---------------


class TestAnswerSpan:
    """Scoring must stop at the first EOS (#157 step 3, condition 1)."""

    @staticmethod
    def _span(*args, **kwargs):
        from unturtle.eval.dependency_slice import answer_span

        return answer_span(*args, **kwargs)

    def test_stops_at_first_eos(self):
        assert self._span([1, 2, 9, 3, 4], eos_id=9) == [1, 2]

    def test_whole_suffix_when_no_eos(self):
        assert self._span([1, 2, 3], eos_id=9) == [1, 2, 3]

    def test_eos_at_position_zero_is_an_empty_answer(self):
        assert self._span([9, 1, 2], eos_id=9) == []

    def test_does_not_splice_across_a_later_eos(self):
        """The #157 step-2 failure mode: decoding the whole canvas with
        skip_special_tokens deletes EOS and joins unrelated fragments."""
        assert self._span([1, 9, 2, 9, 3], eos_id=9) == [1]

    def test_does_not_mutate_the_input(self):
        suffix = [1, 2, 9, 3]
        self._span(suffix, eos_id=9)
        assert suffix == [1, 2, 9, 3]


class TestDependencyLengthDiagnostics:
    """Per-kind length reporting that cannot cancel out (condition 2)."""

    @staticmethod
    def _diag(*args, **kwargs):
        from unturtle.eval.dependency_slice import dependency_length_diagnostics

        return dependency_length_diagnostics(*args, **kwargs)

    def test_no_eos_rows_are_a_separate_column(self):
        result = self._diag([[1, 2, 9], [1, 2, 3]], eos_id=9, mask_id=7)
        assert result["no_eos_fraction"] == 0.5
        assert result["eos_bearing_rows"] == 1

    def test_first_eos_covers_eos_rows_only(self):
        """A row that never stopped must NOT be imputed as position 1024:
        'filled the canvas' and 'stopped late' are different findings."""
        result = self._diag([[1, 2, 9], [1, 2, 3]], eos_id=9, mask_id=7)
        assert result["first_eos_mean_over_eos_rows"] == 2.0

    def test_polarized_rows_stay_visible(self):
        """The maskgit pattern: one row fills the canvas, one stops instantly.
        Both facts must survive rather than averaging to something ordinary."""
        result = self._diag([[1] * 10, [9] + [1] * 9], eos_id=9, mask_id=7)
        assert result["no_eos_fraction"] == 0.5
        assert result["first_eos_mean_over_eos_rows"] == 0.0

    def test_all_rows_without_eos_report_none_not_a_number(self):
        result = self._diag([[1, 2], [3, 4]], eos_id=9, mask_id=7)
        assert result["first_eos_mean_over_eos_rows"] is None
        assert result["first_eos_median_over_eos_rows"] is None
        assert result["no_eos_fraction"] == 1.0

    def test_specials_excluded_and_masks_counted(self):
        result = self._diag([[7, 1, 9, 7]], eos_id=9, mask_id=7)
        assert result["residual_mask_total"] == 2
        assert result["mean_non_special_tokens"] == 1.0

    def test_zero_rows_is_refused(self):
        with pytest.raises(ValueError, match="zero rows"):
            self._diag([], eos_id=9, mask_id=7)


class TestExtractNumericAnswer:
    """The frozen PRIMARY extraction rule (#157 step 3)."""

    @staticmethod
    def _ex(text):
        from unturtle.eval.dependency_slice import extract_numeric_answer

        return extract_numeric_answer(text)

    def test_comma_separated_block_after_prose(self):
        assert self._ex(
            "Sure, here is the sequence:\n\n46, 85, 80, 87, 36, 79, 52, 46"
        )["values"] == ("46", "85", "80", "87", "36", "79", "52", "46")

    def test_concatenated_run_splits_into_two_digit_values(self):
        """A model that emits no separators must recover the same values as a
        comma-separated one — same task schema, different surface form."""
        assert self._ex("7155843774274627")["values"] == (
            "71",
            "55",
            "84",
            "37",
            "74",
            "27",
            "46",
            "27",
        )

    def test_prose_digits_do_not_become_the_answer(self):
        """`re.findall(r"\\d+")` would pick up the 4 of `k4`; this must not."""
        assert self._ex("Based on k4 and k6 the answer is:\n12 41 85 24")["values"] == (
            "12",
            "41",
            "85",
            "24",
        )

    def test_block_with_most_items_wins(self):
        assert self._ex("11 22\n33 44 55 66")["values"] == (
            "33",
            "44",
            "55",
            "66",
        )

    def test_later_block_wins_a_tie(self):
        assert self._ex("11 22\n33 44")["values"] == ("33", "44")

    def test_odd_length_run_is_invalid_not_dropped(self):
        """Discarding a malformed run would hide a bad answer and flatter the
        arm that produced it."""
        result = self._ex("answer: 123")
        assert result["values"] == ()
        assert result["invalid_runs"] == 1

    def test_surplus_is_not_truncated(self):
        assert len(self._ex("11 22 33 44 55 66 77 88 99")["values"]) == 9

    def test_shortfall_is_not_padded(self):
        assert self._ex("11 22")["values"] == ("11", "22")

    def test_absence_of_numbers_is_typed_not_guessed(self):
        result = self._ex("I cannot answer that.")
        assert result["values"] == ()
        assert result["status"] == "no_numeric_block"

    def test_nfkc_normalizes_full_width_digits(self):
        assert self._ex("４６ ８５")["values"] == ("46", "85")

    def test_a_line_break_ends_a_block(self):
        """Rule 2: blocks are separated by prose, and a LINE BREAK is a
        separator, not an in-block one.

        Pinned because the first implementation of this rule used `\\s` in the
        block character class, which merged separate lines into one block. A
        mutation battery caught it surviving: fixtures whose values all survive
        concatenation cannot tell the two apart, because the merged block ends
        with the same items. This fixture can — merging would report six values
        where the winning block has four.
        """
        assert self._ex("11 22 33 44\n55 66")["values"] == (
            "11",
            "22",
            "33",
            "44",
        )

    def test_a_stray_leading_number_is_not_merged_into_the_answer(self):
        """Same separation property from the other side: a stray value on its
        own line must not be prepended to the real answer block."""
        assert self._ex("42\n11 22 33")["values"] == ("11", "22", "33")

    def test_never_consults_the_target(self):
        """Rule 4 picks by COUNT. A longer wrong block must beat a shorter
        block even when the shorter one would have matched a target."""
        assert self._ex("46 85\n11 22 33 44 55 66")["values"] == (
            "11",
            "22",
            "33",
            "44",
            "55",
            "66",
        )


class TestAllNumericRuns:
    """The SECONDARY sensitivity extraction — never a verdict."""

    def test_is_deliberately_broader_than_the_primary_rule(self):
        from unturtle.eval.dependency_slice import all_numeric_runs

        assert all_numeric_runs("k4 -> 12 41") == ("4", "12", "41")


class TestAssembleDependencyCell:
    """The producer's record schema, exercised WITHOUT a GPU.

    The bug these tests exist for: the per-kind block read the flat pre-freeze
    schema while the floor check read the nested one. CI stayed green because
    it never executes the benchmark script, and re-scoring saved suffixes went
    through the shared scorer, so only a fresh multi-hour run would have hit
    the KeyError.
    """

    @staticmethod
    def _cell(**kwargs):
        from unturtle.eval.dependency_slice import assemble_dependency_cell

        return assemble_dependency_cell(**kwargs)

    @staticmethod
    def _fixture():
        from unturtle.eval.dependency_slice import dependency_tasks

        tasks = dependency_tasks(n_per_kind=1, seed=0, length=8)
        # One perfect answer per task, plus an EOS so the length diagnostics
        # see a stopping row.
        texts = [" ".join(task.target) for task in tasks]
        suffixes = [[11, 22, 9] for _ in tasks]
        return tasks, texts, suffixes

    def test_per_kind_carries_the_nested_extraction_schema(self):
        tasks, texts, suffixes = self._fixture()
        result = self._cell(
            tasks=tasks,
            texts=texts,
            suffixes=suffixes,
            eos_id=9,
            mask_id=7,
            reference_floor_accuracy=0.05,
        )
        for cell in result["per_kind"].values():
            assert "primary" in cell
            assert "exact_match" in cell["primary"]
            assert "coupled_token_accuracy" in cell["primary"]
            assert "secondary_all_numeric_runs" in cell
            assert "length" in cell

    def test_per_kind_has_no_flat_duplicate_of_a_nested_score(self):
        """The scores live at ONE level only.

        A mutant that adds a flat `exact_match` beside the nested `primary`
        block survived a presence-only assertion — which is exactly the
        schema-drift that caused the fresh-run KeyError, a flat reader and a
        nested reader coexisting.
        """
        tasks, texts, suffixes = self._fixture()
        result = self._cell(
            tasks=tasks,
            texts=texts,
            suffixes=suffixes,
            eos_id=9,
            mask_id=7,
            reference_floor_accuracy=0.05,
        )
        for cell in result["per_kind"].values():
            assert "exact_match" not in cell
            assert "coupled_token_accuracy" not in cell

    def test_the_floor_is_read_from_the_primary_extraction(self):
        """Reading it from the secondary must change the outcome, so the
        fixture is built where the two extractions DISAGREE: prose digits give
        the broad parser a match the schema-aware primary refuses."""
        from unturtle.eval.dependency_slice import dependency_tasks

        tasks = [
            task
            for task in dependency_tasks(n_per_kind=1, seed=0, length=8)
            if task.kind == "copy"
        ]
        # Correct values, then a LONGER junk block. Rule 4 picks by count, so
        # the primary takes the junk and scores 0; the secondary flattens every
        # run in order and still lines the correct values up at the front.
        texts = [" ".join(tasks[0].target) + "\n" + " ".join(["11"] * 12)]
        suffixes = [[11, 22, 9]]
        result = self._cell(
            tasks=tasks,
            texts=texts,
            suffixes=suffixes,
            eos_id=9,
            mask_id=7,
            reference_floor_accuracy=0.05,
        )
        cell = result["per_kind"]["copy"]
        assert cell["primary"]["coupled_token_accuracy"] == 0.0
        assert cell["secondary_all_numeric_runs"]["coupled_token_accuracy"] > 0.0
        assert result["reference_floor_kinds"] == ["copy"]

    def test_perfect_answers_are_not_at_the_reference_floor(self):
        tasks, texts, suffixes = self._fixture()
        result = self._cell(
            tasks=tasks,
            texts=texts,
            suffixes=suffixes,
            eos_id=9,
            mask_id=7,
            reference_floor_accuracy=0.05,
        )
        assert result["reference_floor_kinds"] == []

    def test_silent_reference_puts_every_kind_at_the_floor(self):
        tasks, _texts, suffixes = self._fixture()
        result = self._cell(
            tasks=tasks,
            texts=["" for _ in tasks],
            suffixes=suffixes,
            eos_id=9,
            mask_id=7,
            reference_floor_accuracy=0.05,
        )
        assert sorted(result["reference_floor_kinds"]) == [
            "copy",
            "kv_recall",
            "reverse",
        ]
        for cell in result["per_kind"].values():
            assert cell["measurement_status"] == "reference_floor / undecidable"

    def test_a_non_reference_arm_is_typed_by_the_reference_floor(self):
        """Condition 5: an arm that is merely as bad as the reference must not
        be scored as preservation, so the floor is imposed, not recomputed."""
        tasks, texts, suffixes = self._fixture()
        result = self._cell(
            tasks=tasks,
            texts=texts,  # this arm answers perfectly
            suffixes=suffixes,
            eos_id=9,
            mask_id=7,
            reference_floor_accuracy=0.05,
            floor_kinds={"copy"},  # but the REFERENCE had no signal on copy
        )
        assert result["reference_floor_kinds"] == ["copy"]
        assert (
            result["per_kind"]["copy"]["measurement_status"]
            == "reference_floor / undecidable"
        )
        assert "measurement_status" not in result["per_kind"]["reverse"]
