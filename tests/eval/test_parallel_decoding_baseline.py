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

"""#157 baseline producer — RED-first.

The Part-0 audit (docs/parallel-decoding-reference-audit.md) froze what this
baseline must produce before any number existed.  Pinned here, on fakes so
the semantics are testable without an 8B checkpoint:

- the speed verdict is **wall-clock**; executed NFE rides along as an
  explanatory variable and is never the denominator;
- executed NFE is counted per family the way #165 established, and a
  requested count is refused;
- commitment-order metrics come from an observed commit trajectory, and
  `answer_before_reasoning_rate` is UNSUPPORTED (not 0) on a fixture with no
  declared output spans;
- `threshold=None` is a quota policy: a step may commit several tokens, so
  nothing may label it one-token-per-step;
- a cell that OOMs or is unsupported is typed data, never an omission;
- the cache axis and the commit axis are recorded separately, because a gain
  on the diagonal alone is a commit gain wearing a cache label.
"""

import math

import pytest
import torch


class TestWallClockVerdict:
    def test_speed_cell_reports_wall_clock_and_carries_nfe_alongside(self):
        """#157 review B5: NFE-normalized throughput is not the verdict."""
        from unturtle.eval.decoding_baseline import speed_cell

        cell = speed_cell(
            wall_seconds=4.0, batch_size=8, executed_nfe=256, sequence_length=1024
        )
        assert cell["samples_per_second"] == pytest.approx(2.0)
        assert cell["wall_seconds"] == 4.0
        # NFE is present but is NOT the denominator of the verdict metric.
        assert cell["executed_nfe"] == 256
        assert cell["nfe_role"] == "explanatory"
        assert "nfe_normalized_throughput" not in cell

    def test_zero_wall_time_is_refused(self):
        from unturtle.eval.decoding_baseline import speed_cell

        with pytest.raises(ValueError, match="wall"):
            speed_cell(
                wall_seconds=0.0, batch_size=8, executed_nfe=1, sequence_length=1024
            )

    def test_requested_nfe_is_refused(self):
        from unturtle.eval.decoding_baseline import speed_cell

        with pytest.raises(ValueError, match="executed"):
            speed_cell(
                wall_seconds=1.0, batch_size=1, executed_nfe=None, sequence_length=1024
            )


class TestCommitTrajectory:
    """Commitment order from an OBSERVED trajectory of committed states."""

    def test_normalized_commit_step_is_first_commit_over_executed_steps(self):
        from unturtle.eval.decoding_baseline import commit_order_metrics

        M = -1  # stand-in for "still masked"
        traj = [
            torch.tensor([[M, M, M, M]]),
            torch.tensor([[7, M, M, M]]),  # pos 0 commits at step 1
            torch.tensor([[7, M, 9, M]]),  # pos 2 at step 2
            torch.tensor([[7, 3, 9, 5]]),  # pos 1 and 3 at step 3
        ]
        out = commit_order_metrics(traj, mask_id=M)
        assert out["steps_executed"] == 3
        # first-commit step / executed steps
        assert out["normalized_commit_step"] == pytest.approx(
            [1 / 3, 3 / 3, 2 / 3, 3 / 3]
        )

    def test_tokens_committed_per_step_carries_position_distribution(self):
        from unturtle.eval.decoding_baseline import commit_order_metrics

        M = -1
        traj = [
            torch.tensor([[M, M, M, M]]),
            torch.tensor([[7, M, M, M]]),
            torch.tensor([[7, 3, 9, 5]]),
        ]
        out = commit_order_metrics(traj, mask_id=M)
        assert out["tokens_committed_per_step"] == [1, 3]
        # positions committed at each step, so a step's spatial spread is visible
        assert out["committed_position_mean"] == pytest.approx([0.0, 2.0])
        assert out["committed_position_std"][1] > 0

    def test_a_quota_step_committing_several_tokens_is_not_relabelled(self):
        """#157 review: threshold=None allocates floor(masked/steps) with the
        remainder spread over the first steps, so multi-token steps are the
        NORM, not an anomaly.  Nothing may call this one-token-per-step."""
        from unturtle.eval.decoding_baseline import commit_order_metrics

        M = -1
        traj = [
            torch.tensor([[M] * 6]),
            torch.tensor([[1, 2, M, M, M, M]]),  # 2 tokens
            torch.tensor([[1, 2, 3, 4, M, M]]),  # 2 tokens
            torch.tensor([[1, 2, 3, 4, 5, 6]]),  # 2 tokens
        ]
        out = commit_order_metrics(traj, mask_id=M)
        assert out["tokens_committed_per_step"] == [2, 2, 2]
        assert out["commit_policy_label"] != "one_token_per_step"

    def test_first_commit_is_kept_separate_from_revisions(self):
        """A position committed then changed counts at its FIRST commit; the
        change is reported as a revision, not merged into commit order."""
        from unturtle.eval.decoding_baseline import commit_order_metrics

        M = -1
        traj = [
            torch.tensor([[M, M]]),
            torch.tensor([[7, M]]),
            torch.tensor([[4, 9]]),  # pos 0 REVISED 7 -> 4, pos 1 first commit
        ]
        out = commit_order_metrics(traj, mask_id=M)
        assert out["normalized_commit_step"][0] == pytest.approx(1 / 2)
        assert out["revision_events"] == 1

    def test_first_commit_survives_a_re_masking_sampler(self):
        """Mutation target: recording the LAST commit instead of the first.

        My other fixtures never re-mask, so first and last coincide there and
        the distinction is invisible.  It is real for a sampler that CAN
        re-mask a position — `alg='origin'` redraws every masked slot each
        step with `p_transfer = 1 - s/t`, so a committed position can return
        to the mask state and commit again.  The recorded step must be the
        FIRST decision, with the round trip visible as revisions.
        """
        from unturtle.eval.decoding_baseline import commit_order_metrics

        M = -1
        traj = [
            torch.tensor([[M, M]]),
            torch.tensor([[7, M]]),  # pos 0 first commits at step 1
            torch.tensor([[M, M]]),  # ...and is re-masked
            torch.tensor([[4, 9]]),  # ...then commits again at step 3
        ]
        out = commit_order_metrics(traj, mask_id=M)
        # first commit, not the later one
        assert out["normalized_commit_step"][0] == pytest.approx(1 / 3)
        assert out["normalized_commit_step"][1] == pytest.approx(3 / 3)
        # the round trip is not silently absorbed into commit order
        assert out["tokens_committed_per_step"] == [1, 0, 2]

    def test_a_never_committed_position_is_reported_not_imputed(self):
        from unturtle.eval.decoding_baseline import commit_order_metrics

        M = -1
        traj = [torch.tensor([[M, M]]), torch.tensor([[7, M]])]
        out = commit_order_metrics(traj, mask_id=M)
        assert out["normalized_commit_step"][1] is None
        assert out["uncommitted_positions"] == 1

    def test_a_single_snapshot_cannot_yield_commit_order(self):
        from unturtle.eval.decoding_baseline import commit_order_metrics

        with pytest.raises(ValueError, match="at least two"):
            commit_order_metrics([torch.tensor([[1, 2]])], mask_id=-1)


class TestSpanMetricSupport:
    """#157 review B4: the span metric is UNSUPPORTED where no output spans
    are declared — reported as such, never as 0."""

    def test_unsupported_without_declared_spans(self):
        from unturtle.eval.decoding_baseline import answer_before_reasoning

        out = answer_before_reasoning(
            normalized_commit_step=[0.2, 0.4, 0.6], spans=None
        )
        assert out["status"] == "unsupported"
        assert "value" not in out
        assert "span" in out["reason"].lower()

    def test_computed_when_spans_are_task_declared(self):
        from unturtle.eval.decoding_baseline import answer_before_reasoning

        # reasoning = positions 0-1 (late), answer = positions 2-3 (early)
        out = answer_before_reasoning(
            normalized_commit_step=[0.8, 0.9, 0.1, 0.2],
            spans={"reasoning": (0, 2), "answer": (2, 4)},
        )
        assert out["status"] == "ok"
        assert out["answer_first"] is True
        assert out["reasoning_mean"] == pytest.approx(0.85)
        assert out["answer_mean"] == pytest.approx(0.15)

    def test_an_empty_span_is_excluded_with_a_reason(self):
        from unturtle.eval.decoding_baseline import answer_before_reasoning

        out = answer_before_reasoning(
            normalized_commit_step=[0.5, 0.5],
            spans={"reasoning": (0, 0), "answer": (0, 2)},
        )
        assert out["status"] == "excluded"
        assert "empty" in out["reason"].lower()

    def test_uncommitted_positions_do_not_silently_shrink_a_span(self):
        from unturtle.eval.decoding_baseline import answer_before_reasoning

        out = answer_before_reasoning(
            normalized_commit_step=[0.8, None, 0.1, 0.2],
            spans={"reasoning": (0, 2), "answer": (2, 4)},
        )
        assert out["status"] == "ok"
        assert out["reasoning_uncommitted"] == 1
        # the mean is over the committed members only, and says so
        assert out["reasoning_mean"] == pytest.approx(0.8)


class TestAxisSeparation:
    """The cache axis and the commit axis are recorded separately, so a
    diagonal-only gain cannot be reported as a cache gain."""

    def test_a_cell_names_both_axes(self):
        from unturtle.eval.decoding_baseline import baseline_cell_key

        key = baseline_cell_key(cache_path="prefix_cache", commit="quota")
        assert key == {"cache_path": "prefix_cache", "commit": "quota"}

    def test_unknown_axis_values_are_refused(self):
        from unturtle.eval.decoding_baseline import baseline_cell_key

        with pytest.raises(ValueError, match="cache_path"):
            baseline_cell_key(cache_path="magic", commit="quota")
        with pytest.raises(ValueError, match="commit"):
            baseline_cell_key(cache_path="no_cache", commit="one_token")

    def test_the_exact_path_is_labelled_exact(self):
        """#157 review: no_cache is the EXACT reference path, not approximate."""
        from unturtle.eval.decoding_baseline import cache_path_class

        assert cache_path_class("no_cache") == "exact"
        assert cache_path_class("prefix_cache") == "approximate_reuse"
        assert cache_path_class("dual_cache") == "approximate_reuse"


class TestTypedCells:
    def test_oom_is_typed_data(self):
        from unturtle.eval.decoding_baseline import run_typed_cell

        def boom(batch_size):
            raise torch.cuda.OutOfMemoryError("no room")

        cell = run_typed_cell(boom, batch_size=32)
        assert cell["status"] == "oom"
        assert "reason" in cell

    def test_unsupported_is_typed_data(self):
        from unturtle.eval.decoding_baseline import run_typed_cell

        cell = run_typed_cell(
            lambda batch_size: None, batch_size=32, unsupported="block_decode absent"
        )
        assert cell["status"] == "unsupported"
        assert cell["reason"] == "block_decode absent"

    def test_other_exceptions_propagate(self):
        from unturtle.eval.decoding_baseline import run_typed_cell

        def broken(batch_size):
            raise RuntimeError("a real bug")

        with pytest.raises(RuntimeError, match="a real bug"):
            run_typed_cell(broken, batch_size=1)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
