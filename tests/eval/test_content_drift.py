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

"""#157 step 2: paired content drift between decode paths.

The #157 baseline found prefix cache 3.07-3.59x faster at matched NFE on one
checkpoint. Because that is a **training-free change on identical weights**,
the question "is it quality-preserving" has a sharper form than a distribution
metric can answer: does the approximate-reuse path produce the SAME text as
the exact path, and if not, how far does it drift?

Pinned here (fakes only — no checkpoint):

- exact agreement is a fraction of samples, not a boolean;
- drift is reported even when agreement is 1.0, so a zero is a measurement
  rather than an absence;
- token-level and text-level distances are separate: identical tokens imply
  identical text, but near-identical text can come from very different tokens;
- an arm whose output length differs is drift, not a crash;
- when the two runs are not comparable (different sample counts), the cell is
  refused rather than silently truncated;
- non-determinism is stated as a reason, never hidden behind a low agreement
  number.
"""

import pytest


class TestExactAgreement:
    def test_identical_outputs_agree_completely(self):
        from unturtle.eval.content_drift import paired_content_drift

        out = paired_content_drift(
            reference_texts=["alpha beta", "gamma"],
            candidate_texts=["alpha beta", "gamma"],
            reference_ids=[[1, 2], [3]],
            candidate_ids=[[1, 2], [3]],
        )
        assert out["exact_text_agreement"] == pytest.approx(1.0)
        assert out["exact_token_agreement"] == pytest.approx(1.0)
        # a zero drift is still reported, not omitted
        assert out["mean_token_edit_distance"] == 0.0
        assert out["mean_normalized_token_distance"] == 0.0

    def test_agreement_is_a_fraction_not_a_boolean(self):
        from unturtle.eval.content_drift import paired_content_drift

        out = paired_content_drift(
            reference_texts=["a", "b", "c", "d"],
            candidate_texts=["a", "X", "c", "d"],
            reference_ids=[[1], [2], [3], [4]],
            candidate_ids=[[1], [9], [3], [4]],
        )
        assert out["exact_text_agreement"] == pytest.approx(0.75)
        assert out["exact_token_agreement"] == pytest.approx(0.75)

    def test_token_and_text_agreement_can_disagree(self):
        """Different tokens can decode to the same text (a retokenization), so
        the two agreements are separate columns and must not be merged."""
        from unturtle.eval.content_drift import paired_content_drift

        out = paired_content_drift(
            reference_texts=["hello world"],
            candidate_texts=["hello world"],
            reference_ids=[[15496, 995]],
            candidate_ids=[[71, 5439, 995]],
        )
        assert out["exact_text_agreement"] == pytest.approx(1.0)
        assert out["exact_token_agreement"] == pytest.approx(0.0)


class TestDistances:
    def test_token_edit_distance_counts_edits(self):
        from unturtle.eval.content_drift import paired_content_drift

        out = paired_content_drift(
            reference_texts=["x"],
            candidate_texts=["y"],
            reference_ids=[[1, 2, 3]],
            candidate_ids=[[1, 9, 3]],
        )
        assert out["mean_token_edit_distance"] == pytest.approx(1.0)
        assert out["mean_normalized_token_distance"] == pytest.approx(1 / 3)

    def test_length_difference_is_drift_not_an_error(self):
        from unturtle.eval.content_drift import paired_content_drift

        out = paired_content_drift(
            reference_texts=["a b c"],
            candidate_texts=["a b"],
            reference_ids=[[1, 2, 3]],
            candidate_ids=[[1, 2]],
        )
        assert out["exact_token_agreement"] == pytest.approx(0.0)
        assert out["mean_token_edit_distance"] == pytest.approx(1.0)
        assert out["length_mismatch_fraction"] == pytest.approx(1.0)

    def test_text_similarity_is_reported_alongside(self):
        from unturtle.eval.content_drift import paired_content_drift

        out = paired_content_drift(
            reference_texts=["the quick brown fox"],
            candidate_texts=["the quick brown cat"],
            reference_ids=[[1, 2, 3, 4]],
            candidate_ids=[[1, 2, 3, 9]],
        )
        # near-identical text, one token changed
        assert 0.5 < out["mean_text_similarity"] < 1.0
        assert out["exact_text_agreement"] == pytest.approx(0.0)


class TestRefusals:
    def test_mismatched_sample_counts_are_refused(self):
        from unturtle.eval.content_drift import paired_content_drift

        with pytest.raises(ValueError, match="sample count"):
            paired_content_drift(
                reference_texts=["a", "b"],
                candidate_texts=["a"],
                reference_ids=[[1], [2]],
                candidate_ids=[[1]],
            )

    def test_empty_input_is_refused(self):
        from unturtle.eval.content_drift import paired_content_drift

        with pytest.raises(ValueError, match="no samples"):
            paired_content_drift(
                reference_texts=[],
                candidate_texts=[],
                reference_ids=[],
                candidate_ids=[],
            )

    def test_ids_must_match_the_texts_in_count(self):
        from unturtle.eval.content_drift import paired_content_drift

        with pytest.raises(ValueError, match="ids"):
            paired_content_drift(
                reference_texts=["a", "b"],
                candidate_texts=["a", "b"],
                reference_ids=[[1]],
                candidate_ids=[[1], [2]],
            )


class TestDeterminismStatement:
    """A drift number is only interpretable if the two runs were supposed to
    agree. When they were not, the record says so instead of implying the
    decode path caused the difference."""

    def test_a_determinism_caveat_is_carried(self):
        from unturtle.eval.content_drift import paired_content_drift

        out = paired_content_drift(
            reference_texts=["a"],
            candidate_texts=["b"],
            reference_ids=[[1]],
            candidate_ids=[[2]],
            determinism="not_guaranteed: temperature 1.0 sampling",
        )
        assert out["determinism"] == "not_guaranteed: temperature 1.0 sampling"

    def test_the_default_states_the_assumption_explicitly(self):
        from unturtle.eval.content_drift import paired_content_drift

        out = paired_content_drift(
            reference_texts=["a"],
            candidate_texts=["a"],
            reference_ids=[[1]],
            candidate_ids=[[1]],
        )
        assert "determinism" in out
        assert out["determinism"] != ""


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
