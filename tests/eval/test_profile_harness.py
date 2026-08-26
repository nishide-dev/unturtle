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


def _cell(events, *, on=10.0, off=9.0, off_trials=None, on_trials=None):
    """`on`/`off` are convenience scalars expanded into three identical trials;
    pass `off_trials`/`on_trials` to control provenance explicitly."""
    from unturtle.eval.profile_harness import ProfileCell

    return ProfileCell(
        family="test",
        cell="c",
        batch_size=1,
        sequence_length=128,
        dtype="bf16",
        wall_off_trials=list(off_trials if off_trials is not None else [off] * 3),
        wall_on_trials=list(on_trials if on_trials is not None else [on] * 3),
        events=events,
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
            wall_off_trials=[9.0, 9.1, 8.9],
            wall_on_trials=[10.0, 10.1, 9.9],
            events=[_event(inclusive_seconds=12.0)],
            extra={"status": "ok", "covered_seconds": 0},
        )
        record = profile_cell(cell)
        assert record["status"] == "profile_invalid"
        assert record["covered_seconds"] == 12.0
        assert record["extra"]["status"] == "ok"


class TestTrialProvenanceIsRequired:
    def test_a_cell_without_off_trials_is_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], off_trials=()))
        assert record["status"] == "profile_invalid"
        assert any("wall_off_trials is empty" in r for r in record["invalid_reasons"])

    def test_a_cell_without_on_trials_is_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], on_trials=()))
        assert record["status"] == "profile_invalid"
        assert any("wall_on_trials is empty" in r for r in record["invalid_reasons"])

    def test_a_single_trial_is_invalid_on_either_arm(self):
        from unturtle.eval.profile_harness import profile_cell

        assert (
            profile_cell(_cell([_event()], off_trials=(9.0,)))["status"]
            == "profile_invalid"
        )
        assert (
            profile_cell(_cell([_event()], on_trials=(10.0,)))["status"]
            == "profile_invalid"
        )

    def test_the_verdict_is_the_median_of_the_off_trials(self):
        """Not an independently injectable scalar: three valid trials plus an
        unrelated verdict number was the hole this closes."""
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(
            _cell([_event()], off_trials=(8.0, 9.0, 10.0), on_trials=(11.0, 12.0, 13.0))
        )
        assert record["wall_seconds_instrumented_off"] == 9.0
        assert record["wall_seconds_instrumented_on"] == 12.0
        assert record["wall_off_trials"]["trials"] == 3

    def test_overhead_is_replicated_against_replicated(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(
            _cell([_event()], off_trials=(9.0, 9.0, 9.0), on_trials=(10.0, 10.0, 10.0))
        )
        assert record["instrumentation_overhead_seconds"] == pytest.approx(1.0)


class TestOneShotArmLifecycle:
    """A one-shot measure function must leave nothing referencing its model.

    The earlier `(run, teardown)` closure-pair design did not: both closures
    captured the model, so `teardown` dropped only local aliases while the
    `run` closure and its default arguments kept it resident — and it was still
    alive while the NEXT arm was being constructed, which is exactly the
    cross-arm contamination the gate exists to avoid.
    """

    def test_the_closure_pair_pattern_leaks_and_the_one_shot_does_not(self):
        import gc
        import weakref

        class Model:
            pass

        # The old shape: teardown cannot release what run still captures.
        def make_arm_closure_pair():
            model = Model()
            trainer = {"m": model}

            def run():
                return len(trainer)

            def teardown(model=model, trainer=trainer):
                alias_model = model
                alias_trainer = trainer
                del alias_model, alias_trainer
                gc.collect()

            return run, teardown, weakref.ref(model)

        run, teardown, leaked = make_arm_closure_pair()
        run()
        teardown()
        assert leaked() is not None, "the leak this test documents is gone"
        del run, teardown
        gc.collect()

        # The one-shot shape: build, use, release, return — nothing survives.
        def measure_arm_one_shot(probe_out):
            model = Model()
            trainer = {"m": model}
            probe_out.append(weakref.ref(model))

            def step(model=model, trainer=trainer):
                return len(trainer)

            step()
            del step
            trainer = None
            model = None
            gc.collect()
            return {"ok": True}

        probes: list = []
        measure_arm_one_shot(probes)
        gc.collect()
        assert probes[0]() is None, "one-shot arm must not outlive its call"


class TestFiniteValidation:
    """NaN and inf must not pass as measurements.

    Every comparison against NaN is False, so without this check the coverage
    and remainder tests pass vacuously and a NaN cell reports `status: ok`.
    """

    def test_nan_wall_trials_are_measurement_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], off_trials=(float("nan"),) * 3))
        assert record["status"] == "measurement_invalid"
        assert any("not finite" in r for r in record["invalid_reasons"])

    def test_negative_wall_trials_are_measurement_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event()], on_trials=(-1.0, 2.0, 3.0)))
        assert record["status"] == "measurement_invalid"
        assert any("negative" in r for r in record["invalid_reasons"])

    def test_non_finite_event_time_is_measurement_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(inclusive_seconds=float("-inf"))]))
        assert record["status"] == "measurement_invalid"

    def test_negative_call_count_is_measurement_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(call_count=-5)]))
        assert record["status"] == "measurement_invalid"

    def test_a_clean_cell_is_still_ok(self):
        from unturtle.eval.profile_harness import profile_cell

        assert profile_cell(_cell([_event(inclusive_seconds=4.0)]))["status"] == "ok"


class TestCallCountIsAnInteger:
    """`call_count` is a count, and the annotation is not enforced at runtime.

    A bare `< 0` check lets NaN through — `nan < 0` is False — along with inf
    and fractional values, so a malformed event reported `status: ok`.
    """

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), float("-inf"), 1.5, -0.5]
    )
    def test_non_integer_counts_are_measurement_invalid(self, value):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(call_count=value)]))
        assert record["status"] == "measurement_invalid"

    def test_bool_is_not_accepted_as_a_count(self):
        """`True` is an Integral and would silently read as one call."""
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(_cell([_event(call_count=True)]))
        assert record["status"] == "measurement_invalid"
        assert any("call_count" in r for r in record["invalid_reasons"])

    def test_a_negative_integer_is_still_refused(self):
        from unturtle.eval.profile_harness import profile_cell

        assert (
            profile_cell(_cell([_event(call_count=-3)]))["status"]
            == "measurement_invalid"
        )

    def test_zero_and_positive_integers_are_accepted(self):
        from unturtle.eval.profile_harness import profile_cell

        assert profile_cell(_cell([_event(call_count=0)]))["status"] == "ok"
        assert profile_cell(_cell([_event(call_count=7)]))["status"] == "ok"


class TestExclusiveIsASubsetOfInclusive:
    """Exclusive time feeds coverage directly, so an exclusive value above its
    own inclusive total inflates `covered_seconds` while staying under the
    outer wall time — over-coverage the wall check cannot see."""

    def test_exclusive_above_inclusive_is_measurement_invalid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(
            _cell([_event(inclusive_seconds=1.0, exclusive_seconds=5.0)])
        )
        assert record["status"] == "measurement_invalid"
        assert any("exceeds its inclusive" in r for r in record["invalid_reasons"])

    def test_equal_values_are_valid_for_a_leaf(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(
            _cell([_event(inclusive_seconds=4.0, exclusive_seconds=4.0)])
        )
        assert record["status"] == "ok"

    def test_a_proper_subset_is_valid(self):
        from unturtle.eval.profile_harness import profile_cell

        record = profile_cell(
            _cell([_event(inclusive_seconds=4.0, exclusive_seconds=1.5)])
        )
        assert record["status"] == "ok"
        assert record["covered_seconds"] == 1.5
