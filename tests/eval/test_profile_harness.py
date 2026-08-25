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

"""#166 Stage-1 harness — coverage arithmetic (docs/acceleration-profile-protocol.md)."""

from __future__ import annotations

import pytest


def _event(**kwargs):
    from unturtle.eval.profile_harness import OperationEvent

    base = {"name": "op", "inclusive_seconds": 1.0, "call_count": 1}
    return OperationEvent(**{**base, **kwargs})


def _cell(events, *, on=10.0, off=9.0, trials=(9.0, 9.1, 8.9)):
    from unturtle.eval.profile_harness import ProfileCell

    return ProfileCell(
        family="test",
        cell="c",
        batch_size=1,
        sequence_length=128,
        dtype="bf16",
        wall_seconds_instrumented_off=off,
        wall_seconds_instrumented_on=on,
        events=events,
        trial_seconds=list(trials),
    )


class TestCoverageIsExclusive:
    """The blocker this arithmetic exists for: nested taxonomies double-count
    when inclusive times are summed across a parent and its child."""

    def test_a_parent_and_child_both_eligible_is_refused(self):
        from unturtle.eval.profile_harness import coverage_seconds

        events = [
            _event(name="full_model_forward", inclusive_seconds=5.0),
            _event(
                name="attention_path",
                inclusive_seconds=3.0,
                parent="full_model_forward",
            ),
        ]
        with pytest.raises(ValueError, match="both"):
            coverage_seconds(events)

    def test_only_the_eligible_level_contributes(self):
        from unturtle.eval.profile_harness import coverage_seconds

        events = [
            _event(name="full_model_forward", inclusive_seconds=5.0),
            _event(
                name="attention_path",
                inclusive_seconds=3.0,
                parent="full_model_forward",
                coverage_eligible=False,
            ),
        ]
        assert coverage_seconds(events) == 5.0

    def test_sibling_events_sum(self):
        from unturtle.eval.profile_harness import coverage_seconds

        events = [
            _event(name="forward", inclusive_seconds=4.0),
            _event(name="backward", inclusive_seconds=2.0),
        ]
        assert coverage_seconds(events) == 6.0

    def test_exclusive_time_is_preferred_when_known(self):
        """An eligible parent with a measured exclusive time contributes that,
        not its inclusive total."""
        from unturtle.eval.profile_harness import coverage_seconds

        events = [
            _event(name="p", inclusive_seconds=5.0, exclusive_seconds=1.5),
        ]
        assert coverage_seconds(events) == 1.5

    def test_a_child_of_an_ineligible_parent_may_be_eligible(self):
        """Marking the child level is the other valid choice."""
        from unturtle.eval.profile_harness import coverage_seconds

        events = [
            _event(name="p", inclusive_seconds=5.0, coverage_eligible=False),
            _event(name="c1", inclusive_seconds=2.0, parent="p"),
            _event(name="c2", inclusive_seconds=1.0, parent="p"),
        ]
        assert coverage_seconds(events) == 3.0


class TestUnattributedRemainder:
    def test_remainder_is_the_instrumented_wall_minus_coverage(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(inclusive_seconds=4.0)], on=10.0))
        assert record["covered_seconds"] == 4.0
        assert record["unattributed_seconds"] == 6.0
        assert record["status"] == "ok"

    def test_a_negative_remainder_is_not_clamped_and_invalidates_the_cell(self):
        """Clamping to zero would hide the bookkeeping error that produced it."""
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(inclusive_seconds=12.0)], on=10.0))
        assert record["unattributed_seconds"] < 0
        assert record["status"] == "profile_invalid"
        assert any("negative" in r for r in record["invalid_reasons"])

    def test_over_coverage_invalidates_the_cell(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(inclusive_seconds=11.0)], on=10.0))
        assert record["status"] == "profile_invalid"
        assert any("exceeds" in r for r in record["invalid_reasons"])

    def test_coverage_within_tolerance_stays_ok(self):
        """A sync boundary between the two reads may put coverage a hair over."""
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(inclusive_seconds=10.0000005)], on=10.0))
        assert record["status"] == "ok"

    def test_shares_are_never_normalized_to_one(self):
        """A taxonomy covering 40% must read as 40%, not be scaled to 100%."""
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(inclusive_seconds=4.0)], on=10.0))
        assert record["covered_seconds"] / record["wall_seconds_instrumented_on"] == 0.4


class TestVerdictSource:
    def test_the_verdict_is_the_uninstrumented_wall_clock(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], on=10.0, off=9.0))
        assert record["verdict_source"] == "wall_seconds_instrumented_off"
        assert record["wall_seconds_instrumented_off"] == 9.0

    def test_instrumentation_overhead_is_reported_not_hidden(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], on=10.0, off=9.0))
        assert record["instrumentation_overhead_seconds"] == pytest.approx(1.0)

    def test_no_nfe_or_operation_sum_normalized_throughput_field(self):
        """Coverage must not be substitutable for the outer wall-clock."""
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()]))
        assert "operation_sum_seconds" not in record


class TestTrialStatistics:
    def test_single_trial_is_flagged(self):
        from unturtle.eval.profile_harness import trial_statistics

        assert trial_statistics([1.0])["single_trial"] is True

    def test_median_with_range(self):
        from unturtle.eval.profile_harness import trial_statistics

        stats = trial_statistics([1.0, 2.0, 5.0])
        assert stats["median_seconds"] == 2.0
        assert stats["min_seconds"] == 1.0
        assert stats["max_seconds"] == 5.0
        assert stats["single_trial"] is False

    def test_zero_trials_is_refused(self):
        from unturtle.eval.profile_harness import trial_statistics

        with pytest.raises(ValueError, match="not a measurement"):
            trial_statistics([])


class TestTreeValidation:
    """Coverage is computed FROM the parent declarations, so they are checked."""

    def test_an_eligible_ancestor_two_levels_up_is_refused(self):
        """The grandparent hole: `leaf`'s direct parent is ineligible, so a
        direct-parent check passes while root and leaf both contribute."""
        from unturtle.eval.profile_harness import coverage_seconds

        events = [
            _event(name="root", inclusive_seconds=10.0),
            _event(
                name="middle",
                inclusive_seconds=8.0,
                parent="root",
                coverage_eligible=False,
            ),
            _event(name="leaf", inclusive_seconds=6.0, parent="middle"),
        ]
        with pytest.raises(ValueError, match="ancestor"):
            coverage_seconds(events)

    def test_a_deep_chain_with_one_eligible_level_is_accepted(self):
        from unturtle.eval.profile_harness import coverage_seconds

        events = [
            _event(name="root", inclusive_seconds=10.0, coverage_eligible=False),
            _event(
                name="middle",
                inclusive_seconds=8.0,
                parent="root",
                coverage_eligible=False,
            ),
            _event(name="leaf", inclusive_seconds=6.0, parent="middle"),
        ]
        assert coverage_seconds(events) == 6.0

    def test_duplicate_event_names_are_refused(self):
        from unturtle.eval.profile_harness import coverage_seconds

        events = [_event(name="op"), _event(name="op")]
        with pytest.raises(ValueError, match="duplicate"):
            coverage_seconds(events)

    def test_a_dangling_parent_is_refused(self):
        from unturtle.eval.profile_harness import coverage_seconds

        with pytest.raises(ValueError, match="not in this cell"):
            coverage_seconds([_event(name="child", parent="ghost")])

    def test_a_parent_cycle_is_refused(self):
        from unturtle.eval.profile_harness import coverage_seconds

        events = [
            _event(name="a", parent="b", coverage_eligible=False),
            _event(name="b", parent="a", coverage_eligible=False),
        ]
        with pytest.raises(ValueError, match="cycle"):
            coverage_seconds(events)


class TestIntegrityFieldsCannotBeOverwritten:
    def test_extra_is_namespaced_not_spread(self):
        """A producer passing extra={"status": "ok"} must not overwrite a
        profile_invalid verdict the core just computed."""
        from unturtle.eval.profile_harness import ProfileCell, profile_cell

        cell = ProfileCell(
            family="f",
            cell="c",
            batch_size=1,
            sequence_length=128,
            dtype="bf16",
            wall_seconds_instrumented_off=9.0,
            wall_seconds_instrumented_on=10.0,
            events=[_event(inclusive_seconds=12.0)],
            trial_seconds=[9.0, 9.1, 8.9],
            extra={"status": "ok", "covered_seconds": 0},
        )
        record = profile_cell(cell)
        assert record["status"] == "profile_invalid"
        assert record["covered_seconds"] == 12.0
        assert record["extra"]["status"] == "ok"


class TestTrialProvenanceIsRequired:
    def test_a_cell_without_trials_is_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], trials=()))
        assert record["status"] == "profile_invalid"
        assert any("trial_seconds" in r for r in record["invalid_reasons"])

    def test_a_single_trial_cell_is_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], trials=(9.0,)))
        assert record["status"] == "profile_invalid"
        assert any("single trial" in r for r in record["invalid_reasons"])

    def test_replicated_trials_are_summarized(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], trials=(9.0, 9.2, 9.1)))
        assert record["status"] == "ok"
        assert record["trials"]["trials"] == 3
        assert record["trials"]["median_seconds"] == 9.1
