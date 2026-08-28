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

"""#166 Stage 1 — ELF generation profile: instrumentation isolation and taxonomy.

The producer needs a GPU and the real checkpoint; these tests pin the properties
that decide whether its numbers mean anything, using a stub model.
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import pathlib
import tokenize

import pytest

pytest.importorskip("unturtle_elf", reason="ELF pack not installed")

import torch  # noqa: E402


def _code_only(source: str) -> str:
    """Source with comments and string literals removed: asserting a forbidden
    construct is absent fails on the very docstring that forbids it."""
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def _producer():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "elf"
        / "generation_profile.py"
    )
    spec = importlib.util.spec_from_file_location("_elf_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubModel(torch.nn.Module):
    """Returns the (aux, logits) pair `_dlm_decode_batch` expects."""

    def forward(self, *_args, **_kwargs):
        return None, torch.zeros(1, 2, 3)


def _open_step(recorder, name="_sde_step"):
    """Open a solver step through the RECORDER's own API.

    An earlier version hand-built the step dict, which silently diverged when
    the record grew `step_id` / `forwards_step_id` fields — a fixture that
    fabricates the structure under test cannot detect a change to it.
    """
    return recorder.solver_step(name)


class TestInstanceLocalPatching:
    """A class-level `type(model).forward` patch instruments EVERY instance of
    the class while it is live. Measured before the fix: calling an unrelated
    instance recorded a rollout forward on this recorder."""

    def test_an_unrelated_instance_is_not_observed(self):
        producer = _producer()
        target, bystander = StubModel(), StubModel()
        recorder = producer.Recorder(torch, mode="count")
        with producer.instrumented(target, recorder), _open_step(recorder):
            bystander(torch.zeros(1))
            assert recorder.forward_calls["rollout"] == 0, (
                "an unrelated instance of the same class was attributed"
            )
            target(torch.zeros(1))
        assert recorder.forward_calls["rollout"] == 1

    def test_the_class_is_never_modified(self):
        producer = _producer()
        original = StubModel.forward
        with producer.instrumented(StubModel(), producer.Recorder(torch, mode="count")):
            assert StubModel.forward is original
        assert StubModel.forward is original

    def test_the_instance_override_is_deleted_not_reassigned(self):
        """Reassigning the original bound method would leave a permanent
        instance-level shadow of the class descriptor."""
        producer = _producer()
        model = StubModel()
        assert "forward" not in model.__dict__
        with producer.instrumented(model, producer.Recorder(torch, mode="count")):
            assert "forward" in model.__dict__
        assert "forward" not in model.__dict__

    def test_the_instance_is_restored_when_the_forward_raises(self):
        producer = _producer()

        class Exploding(torch.nn.Module):
            def forward(self, *_args, **_kwargs):
                raise RuntimeError("forward exploded")

        model = Exploding()
        recorder = producer.Recorder(torch, mode="count")
        with (
            pytest.raises(RuntimeError, match="forward exploded"),
            producer.instrumented(model, recorder),
            _open_step(recorder),
        ):
            model(torch.zeros(1))
        assert "forward" not in model.__dict__

    def test_nested_installation_is_refused(self):
        """Attribution would otherwise depend on install order."""
        producer = _producer()
        model = StubModel()
        # The nesting IS the subject of this test: an outer install, then an
        # inner attempt on the same instance. Merging the two `with` statements
        # would remove the condition being checked.
        with producer.instrumented(  # noqa: SIM117
            model, producer.Recorder(torch, mode="count")
        ):
            with (
                pytest.raises(producer.InstrumentationError, match="refusing to nest"),
                producer.instrumented(model, producer.Recorder(torch, mode="count")),
            ):
                pass
        assert "forward" not in model.__dict__

    def test_nesting_raises_a_catchable_error_not_system_exit(self):
        """`SystemExit` slips past `except Exception` and would kill the
        producer before the remaining cells could be written."""
        producer = _producer()
        assert issubclass(producer.InstrumentationError, RuntimeError)
        assert not issubclass(producer.InstrumentationError, SystemExit)

    def test_the_producer_never_assigns_to_the_class(self):
        producer = _producer()
        code = _code_only(inspect.getsource(producer.instrumented))
        assert "type ( model ) . forward =" not in code
        assert "MethodType" in code

    def test_the_module_globals_are_restored_exactly(self):
        producer = _producer()
        from unturtle_elf._reference import generation_utils as gu

        before = (gu._sde_step, gu._ode_step, gu._dlm_decode_batch)
        with producer.instrumented(StubModel(), producer.Recorder(torch, mode="count")):
            assert (gu._sde_step, gu._ode_step, gu._dlm_decode_batch) != before
        assert (gu._sde_step, gu._ode_step, gu._dlm_decode_batch) == before

    def test_generation_utils_is_patched_not_sampling_utils(self):
        """`generation_utils` imports the step functions as aliases at import
        time, so patching `sampling_utils` would never reach the call sites."""
        producer = _producer()
        source = inspect.getsource(producer.instrumented)
        assert "generation_utils as gu" in source
        assert "sampling_utils" not in _code_only(source)

    def test_the_off_arm_installs_nothing(self):
        producer = _producer()
        from unturtle_elf._reference import generation_utils as gu

        model = StubModel()
        before = (gu._sde_step, gu._dlm_decode_batch)
        with producer.instrumented(model, None):
            assert (gu._sde_step, gu._dlm_decode_batch) == before
            assert "forward" not in model.__dict__


class TestPhaseAttribution:
    def test_the_endpoint_forward_is_not_a_denoiser_forward(self):
        producer = _producer()
        model = StubModel()
        recorder = producer.Recorder(torch, mode="count")
        with producer.instrumented(model, recorder):
            with _open_step(recorder):
                model(torch.zeros(1))
            with recorder.endpoint_projection():
                model(torch.zeros(1))
        assert recorder.forward_calls == {"rollout": 1, "endpoint": 1}

    def test_a_forward_outside_any_phase_is_attributed_to_neither(self):
        producer = _producer()
        model = StubModel()
        recorder = producer.Recorder(torch, mode="count")
        with producer.instrumented(model, recorder):
            model(torch.zeros(1))
        assert recorder.forward_calls == {"rollout": 0, "endpoint": 0}

    def test_the_rollout_child_lands_on_its_own_step(self):
        producer = _producer()
        model = StubModel()
        recorder = producer.Recorder(torch, mode="count")
        with producer.instrumented(model, recorder):
            for _ in range(3):
                with recorder.solver_step("_sde_step"):
                    model(torch.zeros(1))
        assert len(recorder.steps) == 3
        assert all(len(step["forwards"]) == 1 for step in recorder.steps)


class TestFrozenCounts:
    @pytest.mark.parametrize(("steps", "sde", "ode"), [(32, 31, 1), (64, 63, 1)])
    def test_the_final_step_is_always_ode(self, steps, sde, ode):
        """`t_steps` is length steps+1, the loop runs len-2 times, and the last
        step is unconditionally ODE."""
        assert _producer().expected_step_calls(steps) == {
            "_sde_step": sde,
            "_ode_step": ode,
        }

    @pytest.mark.parametrize(("steps", "rollout"), [(32, 32), (64, 64)])
    def test_the_endpoint_forward_is_counted_separately(self, steps, rollout):
        assert _producer().expected_forward_calls(steps) == {
            "rollout": rollout,
            "endpoint": 1,
        }

    @pytest.mark.parametrize(("steps", "churn"), [(32, 31), (64, 63)])
    def test_random_calls_are_frozen_per_callsite(self, steps, churn):
        """A total alone would absorb a future random op inside the model."""
        assert _producer().expected_random_calls(steps) == {
            "time_grid": 1,
            "initial_latent": 1,
            "sde_churn": churn,
        }

    @pytest.mark.parametrize(("steps", "total"), [(32, 33), (64, 65)])
    def test_the_random_total_matches_the_breakdown(self, steps, total):
        assert sum(_producer().expected_random_calls(steps).values()) == total

    def test_the_frozen_cell_configuration(self):
        producer = _producer()
        assert producer.SOLVER == "sde"
        assert producer.MAX_LENGTH == 1024
        assert producer.CFG_SCALE == 1.0
        assert producer.SELF_COND_CFG_SCALE == 3.0
        assert producer.TIME_SCHEDULE == "logit_normal"
        assert producer.STEPS_CELLS == (32, 64)
        assert producer.BATCH_SIZES == (1, 8, 32)
        assert producer.SDE_GAMMA == {32: 1.5, 64: 1.0}
        assert producer.TRIALS == 3

    def test_the_checkpoint_matches_the_pack_pin(self):
        from unturtle_elf.loader import DEFAULT_CHECKPOINT, DEFAULT_REVISION

        assert DEFAULT_CHECKPOINT == "embedded-language-flows/ELF-B-owt-torch"
        assert DEFAULT_REVISION == "146f84133c1389bfd4ef47f14ec7a955da22faa7"


class TestStepExclusiveArithmetic:
    @staticmethod
    def _fake_pair(seconds):
        class Event:
            def elapsed_time(self, other):
                return other.value

            def __init__(self, value=0.0):
                self.value = value

        return (Event(), Event(seconds * 1000.0))

    def test_state_update_is_a_per_step_difference(self):
        """`median(inclusive) - median(forward)` subtracts series whose medians
        can come from DIFFERENT steps, so it is not an exclusive time."""
        producer = _producer()
        recorder = producer.Recorder(torch, mode="time")
        # Inclusive 1.0/2.0/3.0 with children 0.9/0.5/0.1 -> exclusive
        # 0.1+1.5+2.9 = 4.5. A difference of medians would give 2.0-0.5 = 1.5.
        for inclusive, child in ((1.0, 0.9), (2.0, 0.5), (3.0, 0.1)):
            recorder.steps.append(
                {
                    "name": "_sde_step",
                    "inclusive": self._fake_pair(inclusive),
                    "forwards": [self._fake_pair(child)],
                }
            )
        denoiser, state_update, problems = producer.step_exclusive_seconds(recorder)
        assert problems == []
        assert denoiser == pytest.approx(1.5)
        assert state_update == pytest.approx(4.5)

    def test_a_step_without_a_child_forward_is_a_pairing_failure(self):
        """Counts can add up while the parent/child correspondence is broken."""
        producer = _producer()
        recorder = producer.Recorder(torch, mode="time")
        recorder.steps.append(
            {"name": "_sde_step", "inclusive": self._fake_pair(1.0), "forwards": []}
        )
        _, _, problems = producer.step_exclusive_seconds(recorder)
        assert problems and "expected exactly 1" in problems[0]

    def test_a_step_with_two_child_forwards_is_a_pairing_failure(self):
        producer = _producer()
        recorder = producer.Recorder(torch, mode="time")
        recorder.steps.append(
            {
                "name": "_sde_step",
                "inclusive": self._fake_pair(1.0),
                "forwards": [self._fake_pair(0.4), self._fake_pair(0.4)],
            }
        )
        _, _, problems = producer.step_exclusive_seconds(recorder)
        assert problems and "2 rollout forwards" in problems[0]


class TestStepIdentityPairing:
    """The most dangerous ELF regression is not a wrong count — it is
    subtracting a forward that belongs to a DIFFERENT step while every count
    still adds up."""

    def test_a_child_recorded_against_another_step_is_caught(self):
        producer = _producer()
        recorder = producer.Recorder(torch, mode="count")
        model = StubModel()
        with producer.instrumented(model, recorder):
            for _ in range(3):
                with recorder.solver_step("_sde_step"):
                    model(torch.zeros(1))
        assert producer.check_span_ordering(recorder) == []
        # Counts stay correct while the pairing is corrupted.
        recorder.steps[1]["forwards_step_id"] = [2]
        problems = producer.check_span_ordering(recorder)
        assert problems and "step id 2, not 1" in problems[0]
        assert all(len(step["forwards"]) == 1 for step in recorder.steps)

    def test_each_child_records_the_step_that_was_open(self):
        producer = _producer()
        recorder = producer.Recorder(torch, mode="count")
        model = StubModel()
        with producer.instrumented(model, recorder):
            for _ in range(4):
                with recorder.solver_step("_sde_step"):
                    model(torch.zeros(1))
        assert [s["step_id"] for s in recorder.steps] == [0, 1, 2, 3]
        assert [s["forwards_step_id"] for s in recorder.steps] == [[0], [1], [2], [3]]

    def test_an_unclosed_parent_span_is_caught(self):
        producer = _producer()
        recorder = producer.Recorder(torch, mode="count")
        recorder.steps.append(
            {"name": "_sde_step", "step_id": 0, "forwards": [], "forwards_step_id": []}
        )
        problems = producer.check_span_ordering(recorder)
        assert problems and "never closed" in problems[0]


class TestNegativeExclusiveTime:
    @staticmethod
    def _pair(seconds):
        class Event:
            def __init__(self, value=0.0):
                self.value = value

            def elapsed_time(self, other):
                return other.value

        return (Event(), Event(seconds * 1000.0))

    def test_a_parent_shorter_than_its_child_fails_the_cell(self):
        """On one stream a correctly nested parent cannot be shorter than the
        child it contains."""
        producer = _producer()
        recorder = producer.Recorder(torch, mode="time")
        recorder.steps.append(
            {
                "name": "_sde_step",
                "step_id": 0,
                "inclusive": self._pair(0.5),
                "forwards": [self._pair(0.9)],
                "forwards_step_id": [0],
            }
        )
        _, _, problems = producer.step_exclusive_seconds(recorder)
        assert problems and "is negative" in problems[0]

    def test_the_negative_exclusive_is_not_clamped(self):
        producer = _producer()
        code = _code_only(inspect.getsource(producer.step_exclusive_seconds))
        for clamping in ("max ( 0", "abs ("):
            assert clamping not in code, clamping

    def test_per_step_values_are_retained_for_audit(self):
        """So a later regression to median-of-medians is detectable from the
        artifact alone."""
        producer = _producer()
        recorder = producer.Recorder(torch, mode="time")
        for inclusive, child in ((1.0, 0.6), (2.0, 0.5)):
            recorder.steps.append(
                {
                    "name": "_sde_step",
                    "step_id": len(recorder.steps),
                    "inclusive": self._pair(inclusive),
                    "forwards": [self._pair(child)],
                    "forwards_step_id": [len(recorder.steps)],
                }
            )
        producer.step_exclusive_seconds(recorder)
        assert recorder.per_step_exclusive == pytest.approx([0.4, 1.5])


class TestRandomCallClassification:
    def test_the_three_callsites_are_distinguished(self):
        producer = _producer()

        class Frame:
            def __init__(self, filename, function):
                self.filename, self.function = filename, function

        cases = {
            "time_grid": Frame(
                "/x/unturtle_elf/_reference/sampling_utils.py", "sample_timesteps"
            ),
            "sde_churn": Frame(
                "/x/unturtle_elf/_reference/sampling_utils.py", "_sde_step"
            ),
            "initial_latent": Frame(
                "/x/unturtle_elf/sampler.py", "run_generation_request"
            ),
        }
        for expected, frame in cases.items():
            assert producer.classify_random_call([frame]) == expected

    def test_an_unrecognised_callsite_is_unknown_not_absorbed(self):
        """A new random op inside the model must not be silently folded into a
        matching total."""
        producer = _producer()

        class Frame:
            filename = "/x/unturtle_elf/_reference/model.py"
            function = "forward"

        assert producer.classify_random_call([Frame()]) == "unknown"

    def test_shape_alone_would_not_separate_the_callsites(self):
        """The initial latent and the SDE churn draw the same [B, L, d] shape,
        which is why the classifier keys on module AND function."""
        producer = _producer()
        source = inspect.getsource(producer.classify_random_call)
        assert "frame.function" in source
        assert "frame.filename" in source
        assert "SAME" in source


def _trial(
    *,
    index=0,
    off=1.0,
    on=1.0,
    denoiser=0.5,
    state=0.3,
    endpoint=0.1,
    inclusive=0.8,
    steps=32,
):
    """A trial record shaped exactly as `paired_trials` emits one."""
    return {
        "trial": index,
        "order": ["off", "on"] if index % 2 == 0 else ["on", "off"],
        "off_wall_seconds": off,
        "on_wall_seconds": on,
        "paired_overhead_seconds": on - off,
        "peak_allocated_bytes": 100 + index,
        "peak_reserved_bytes": 200 + index,
        "event_seconds": {
            "denoiser_forward": denoiser,
            "solver_state_update": state,
            "endpoint_projection": endpoint,
        },
        "audit_seconds": {"solver_step_inclusive": inclusive},
        "per_step_exclusive_seconds": [state],
        "step_calls": {"_sde_step": steps - 1, "_ode_step": 1},
        "forward_calls": {"rollout": steps, "endpoint": 1},
        "problems": [],
    }


class TestCellAssemblyBehaviour:
    """Asserted by CALLING `assemble_cell`, not by grepping it."""

    def test_a_clean_cell_reports_ok_with_the_audit_fields(self):
        producer = _producer()
        cell = producer.assemble_cell(32, [_trial(index=i) for i in range(3)])
        assert cell["status"] == "ok"
        assert cell["audit"]["per_step_exclusive_seconds"]
        assert cell["audit"]["solver_step_inclusive_seconds_trials"] == [0.8] * 3
        assert cell["forward_accounting"]["rollout_forward_count"] == 32
        assert cell["forward_accounting"]["endpoint_forward_count"] == 1
        assert cell["forward_accounting"]["extra_cfg_forward_count"] == 0

    def test_only_the_coverage_events_appear(self):
        """The audit parent contains its own children; including it would
        double-count."""
        producer = _producer()
        cell = producer.assemble_cell(32, [_trial(index=i) for i in range(3)])
        assert [e["name"] for e in cell["events"]] == list(producer.COVERAGE_EVENTS)
        assert "solver_step_inclusive" not in [e["name"] for e in cell["events"]]

    def test_events_exceeding_the_wall_invalidate_the_cell(self):
        """Caught by the RESIDUAL gate. There is no separate coverage gate:
        `coverage > 1` is exactly `residual < 0` because both derive from the
        same attributed sum and the same wall, so a coverage check would be
        strictly subsumed — and an unreachable gate reads as an independent
        guarantee it never provided."""
        producer = _producer()
        trials = [_trial(index=i) for i in range(3)]
        trials[1]["event_seconds"]["denoiser_forward"] = 1.5
        cell = producer.assemble_cell(32, trials)
        assert cell["status"] == "profile_invalid"
        assert cell["reason_code"] == "negative_residual_or_invalid_event"
        assert cell["latency"] is None

    def test_coverage_is_descriptive_and_never_classifies(self):
        producer = _producer()
        cell = producer.assemble_cell(32, [_trial(index=i) for i in range(3)])
        attribution = cell["attribution"]
        assert attribution["coverage_disposition"] == "descriptive_only"
        assert attribution["coverage_ratio_trials"]
        # No tolerance constant survives, and nothing branches on coverage.
        assert not hasattr(producer, "SHARE_TOLERANCE")
        code = _code_only(inspect.getsource(producer.assemble_cell))
        assert "coverage_per_trial [ index ] >" not in code

    def test_a_negative_residual_in_one_trial_invalidates_the_cell(self):
        producer = _producer()
        trials = [_trial(index=i) for i in range(3)]
        trials[2]["event_seconds"]["endpoint_projection"] = 0.9
        cell = producer.assemble_cell(32, trials)
        assert cell["status"] == "profile_invalid"

    def test_a_median_only_residual_check_would_have_passed(self):
        """Which is why the residual gate is per trial: two clean trials carry
        the median."""
        import statistics

        producer = _producer()
        trials = [_trial(index=i) for i in range(3)]
        trials[1]["event_seconds"]["denoiser_forward"] = 1.5
        residuals = [
            t["on_wall_seconds"] - sum(t["event_seconds"].values()) for t in trials
        ]
        assert statistics.median(residuals) >= 0
        assert min(residuals) < 0
        assert producer.assemble_cell(32, trials)["status"] == "profile_invalid"

    def test_shares_are_per_trial_ratios(self):
        """Walls differ across trials, so a ratio of medians would differ from
        the median of ratios."""
        producer = _producer()
        trials = [
            _trial(index=0, on=1.0, denoiser=0.5),
            _trial(index=1, on=2.0, denoiser=0.5),
            _trial(index=2, on=4.0, denoiser=0.5),
        ]
        cell = producer.assemble_cell(32, trials)
        share = next(
            e["share_of_on_wall"]
            for e in cell["events"]
            if e["name"] == "denoiser_forward"
        )
        assert share == pytest.approx(0.25)

    def test_a_ratio_of_medians_would_give_a_different_answer(self):
        """Constructed so the two aggregations genuinely diverge, which the
        equal-event fixture above cannot show."""
        import statistics

        producer = _producer()
        trials = [
            _trial(index=0, on=1.0, denoiser=0.10, state=0.1, endpoint=0.1),
            _trial(index=1, on=2.0, denoiser=1.60, state=0.1, endpoint=0.1),
            _trial(index=2, on=4.0, denoiser=0.80, state=0.1, endpoint=0.1),
        ]
        median_of_ratios = statistics.median([0.10 / 1.0, 1.60 / 2.0, 0.80 / 4.0])
        ratio_of_medians = statistics.median([0.10, 1.60, 0.80]) / statistics.median(
            [1.0, 2.0, 4.0]
        )
        assert median_of_ratios != pytest.approx(ratio_of_medians)
        cell = producer.assemble_cell(32, trials)
        share = next(
            e["share_of_on_wall"]
            for e in cell["events"]
            if e["name"] == "denoiser_forward"
        )
        assert share == pytest.approx(median_of_ratios)

    def test_peak_memory_is_a_per_trial_array_plus_the_max(self):
        producer = _producer()
        cell = producer.assemble_cell(32, [_trial(index=i) for i in range(3)])
        assert cell["peak_memory"]["allocated_bytes_trials"] == [100, 101, 102]
        assert cell["peak_memory"]["max_allocated_bytes"] == 102
        assert cell["peak_memory"]["basis"] == "instrumentation_off_trials"

    def test_the_overhead_is_never_adjudicated(self):
        producer = _producer()
        cell = producer.assemble_cell(32, [_trial(index=i, on=0.9) for i in range(3)])
        overhead = cell["latency"]["instrumentation_overhead"]
        assert overhead["median_paired_delta"] < 0
        assert overhead["resolvable"] is None
        assert cell["status"] == "ok", "a negative overhead must not invalidate"


class TestTrialGate:
    def test_a_clean_trial_passes(self):
        producer = _producer()
        assert producer.gate_trial(32, _trial()) == []

    def test_a_wrong_step_split_fails(self):
        """An SDE cell is (steps-1) SDE plus exactly one final ODE."""
        producer = _producer()
        trial = _trial()
        trial["step_calls"] = {"_sde_step": 32, "_ode_step": 0}
        assert producer.gate_trial(32, trial)

    def test_an_endpoint_forward_counted_as_rollout_fails(self):
        producer = _producer()
        trial = _trial()
        trial["forward_calls"] = {"rollout": 33, "endpoint": 0}
        assert producer.gate_trial(32, trial)

    def test_pairing_problems_propagate(self):
        producer = _producer()
        trial = _trial()
        trial["problems"] = ["solver step 4: child recorded against step id 7"]
        assert producer.gate_trial(32, trial)


class TestRandomSpyBehaviour:
    def test_the_spy_counts_by_callsite_and_restores(self):
        producer = _producer()
        counts: dict[str, int] = {}
        original = torch.randn
        with producer.random_call_spy(counts):
            assert torch.randn is not original
            torch.randn(2)
        assert torch.randn is original
        # This test's own callsite is not an ELF module.
        assert counts == {"unknown": 1}

    def test_the_spy_restores_when_the_body_raises(self):
        producer = _producer()
        original = torch.randn
        with pytest.raises(ZeroDivisionError), producer.random_call_spy({}):
            raise ZeroDivisionError
        assert torch.randn is original


class TestPreflightCompleteness:
    """`non_interference_preflight` compares six fields; dropping any one of
    them silently narrows what "identical" means."""

    def test_all_six_comparison_fields_are_checked(self):
        producer = _producer()
        source = inspect.getsource(producer.non_interference_preflight)
        fields = source.split("fields = [")[1].split("]")[0]
        for field in (
            "final_latent",
            "raw_endpoint_tokens",
            "masked_public_tokens",
            "terminal_rng_cpu",
            "executed_metadata",
        ):
            assert f'"{field}"' in fields, field
        assert '"terminal_rng_cuda"' in source

    def test_both_arms_are_recorded_not_only_the_verdict(self):
        producer = _producer()
        source = inspect.getsource(producer.non_interference_preflight)
        assert '"off": off' in source
        assert '"on": on' in source

    def test_terminal_rng_is_captured_after_the_decode_call(self):
        """An endpoint forward that wrongly consumed randomness would be
        invisible to an entry-time reading."""
        producer = _producer()
        source = inspect.getsource(producer.decode_diagnostics)
        call = source.index("tokens = original(z, *args, **kwargs)")
        rng = source.index('sink["terminal_rng"] = rng_state_hashes(device)')
        assert call < rng, "terminal RNG is read before the endpoint forward"

    def test_the_final_latent_is_captured_before_the_decode_call(self):
        producer = _producer()
        source = inspect.getsource(producer.decode_diagnostics)
        latent = source.index('sink["final_latent"] = stable_hash(z)')
        call = source.index("tokens = original(z, *args, **kwargs)")
        assert latent < call

    def test_the_decode_wrapper_restores_the_module_global(self):
        producer = _producer()
        from unturtle_elf._reference import generation_utils as gu

        original = gu._dlm_decode_batch
        sink: dict = {}
        with producer.decode_diagnostics(StubModel(), sink, None):
            assert gu._dlm_decode_batch is not original
        assert gu._dlm_decode_batch is original

    def test_the_decode_wrapper_restores_when_the_body_raises(self):
        producer = _producer()
        from unturtle_elf._reference import generation_utils as gu

        original = gu._dlm_decode_batch
        with (
            pytest.raises(ZeroDivisionError),
            producer.decode_diagnostics(StubModel(), {}, None),
        ):
            raise ZeroDivisionError
        assert gu._dlm_decode_batch is original

    def test_diagnostics_never_run_inside_a_timed_trial(self):
        producer = _producer()
        for function in (producer._run_off_trial, producer._run_on_trial):
            code = _code_only(inspect.getsource(function))
            for forbidden in ("decode_diagnostics", "stable_hash", "rng_state_hashes"):
                assert forbidden not in code, f"{function.__name__}: {forbidden}"


class TestRandomPreflightBehaviour:
    @staticmethod
    def _preflight(producer, counts, steps=32):
        """Drive `random_call_preflight`'s reporting with injected counts."""
        expected = producer.expected_random_calls(steps)
        observed = {key: counts.get(key, 0) for key in expected}
        unknown = counts.get("unknown", 0)
        return {
            "observed": observed,
            "expected": expected,
            "unknown": unknown,
            "matches": observed == expected and unknown == 0,
        }

    def test_an_unknown_callsite_fails_even_when_the_totals_match(self):
        """A new random op inside the model must not be absorbed."""
        producer = _producer()
        counts = {"time_grid": 1, "initial_latent": 1, "sde_churn": 31, "unknown": 1}
        assert self._preflight(producer, counts)["matches"] is False
        source = inspect.getsource(producer.random_call_preflight)
        assert "observed == expected and unknown == 0" in source

    def test_a_missing_callsite_fails(self):
        producer = _producer()
        counts = {"time_grid": 1, "initial_latent": 0, "sde_churn": 32}
        assert self._preflight(producer, counts)["matches"] is False

    def test_the_exact_frozen_counts_pass(self):
        producer = _producer()
        counts = {"time_grid": 1, "initial_latent": 1, "sde_churn": 31}
        assert self._preflight(producer, counts)["matches"] is True

    def test_the_observed_counts_are_read_from_the_spy(self):
        """Not copied from the expectation, which would make the gate
        tautological."""
        producer = _producer()
        source = inspect.getsource(producer.random_call_preflight)
        assert "counts.get(key, 0)" in source
        assert "observed = dict(expected)" not in source


class TestTimedWallBoundary:
    """Install/restore must sit OUTSIDE the timed span, and the window-closing
    synchronize INSIDE it."""

    @staticmethod
    def _on_source():
        return inspect.getsource(_producer()._run_on_trial)

    def test_instrumentation_is_installed_before_the_clock_starts(self):
        source = self._on_source()
        assert source.index("with instrumented(") < source.index(
            "begin = time.perf_counter()"
        )

    def test_the_sync_that_closes_the_window_is_inside_the_wall(self):
        """Synchronizing after the clock stops pushes the queue drain outside
        the wall — the #166 FMLM defect, where the ON wall came out shorter
        than the event total it should contain."""
        source = self._on_source()
        after_request = source.split("_public_request(model, request)")[1]
        sync = after_request.index("torch.cuda.synchronize()")
        stop = after_request.index("wall = time.perf_counter() - begin")
        assert sync < stop, "the drain is outside the timed wall"

    def test_the_restore_happens_after_the_clock_stops(self):
        source = self._on_source()
        wall_line = source.index("wall = time.perf_counter() - begin")
        # The `with` block closes (restoring) only after the wall is read.
        assert source.index("step_exclusive_seconds(recorder)") > wall_line

    def test_event_collection_does_not_synchronize(self):
        """The single window-closing sync already ran inside the timed span."""
        producer = _producer()
        collect = inspect.getsource(producer.step_exclusive_seconds)
        assert "synchronize" not in _code_only(collect)

    def test_the_off_arm_drain_is_inside_its_wall(self):
        """The verdict pass has the same async-drain hazard as the ON pass: a
        clock stopped before the queue empties makes the VERDICT itself short."""
        producer = _producer()
        source = inspect.getsource(producer._run_off_trial)
        after = source.split("_public_request(model, request)")[1]
        assert "torch.cuda.synchronize()" in after
        assert after.index("torch.cuda.synchronize()") < after.index(
            "time.perf_counter() - begin"
        ), "the OFF wall excludes the queue drain"

    def test_the_off_arm_installs_no_instrumentation(self):
        producer = _producer()
        code = _code_only(inspect.getsource(producer._run_off_trial))
        assert "instrumented" not in code
        assert "Recorder" not in code

    def test_both_arms_use_one_public_call_site(self):
        producer = _producer()
        whole = pathlib.Path(producer.__file__).read_text()
        assert whole.count("run_generation_request(model, request)") == 1
        for function in (producer._run_off_trial, producer._run_on_trial):
            assert "_public_request(model, request)" in inspect.getsource(function)

    def test_the_trial_order_reverses(self):
        source = inspect.getsource(_producer().paired_trials)
        assert 'order = ("off", "on") if index % 2 == 0 else ("on", "off")' in source


class TestFailureStaging:
    def test_an_instrumentation_error_is_profile_invalid_not_oom(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        block = source.split("except InstrumentationError")[1]
        assert 'reason_code="instrumentation_structure_invalid"' in block
        assert 'status="profile_invalid"' in block

    def test_timing_attempted_is_derived_from_the_stage(self):
        """A preflight failure must not claim a clock ever started."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert source.count('timing_attempted=stage == "paired_trials"') == 2

    def test_preflight_failures_report_no_timing(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        for stage in ("non_interference_preflight", "random_call_preflight"):
            # Up to the end of that failure_record call, not a fixed slice: the
            # random-callsite branch has a multi-line conditional reason_code.
            block = source.split(f'stage="{stage}"')[1].split(")\n")[0]
            assert "timing_attempted=False" in block, stage

    def test_an_unclassified_random_callsite_has_its_own_reason_code(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell)
        assert '"unclassified_random_callsite"' in source
        assert '"random_call_count_mismatch"' in source

    def test_a_failed_cell_emits_nulls_not_zeros(self):
        producer = _producer()
        record = producer.failure_record(
            stage="paired_trials", reason_code=None, timing_attempted=True
        )
        for field in ("latency", "events", "peak_memory", "attribution", "trials"):
            assert record[field] is None, field

    def test_a_host_out_of_memory_is_not_a_cuda_oom(self):
        producer = _producer()
        assert (
            producer.classify_failure(
                RuntimeError("CPU out of memory. Tried to allocate 6 GiB")
            )
            is None
        )
        assert (
            producer.classify_failure(
                RuntimeError("CUDA out of memory. Tried to allocate 6 GiB")
            )
            == "cuda_out_of_memory"
        )


class TestArtifactShape:
    def test_the_audit_fields_allow_detecting_median_of_medians(self):
        producer = _producer()
        source = inspect.getsource(producer.assemble_cell)
        for field in (
            "solver_step_inclusive_seconds_trials",
            "solver_state_update_seconds_trials",
            "per_step_exclusive_seconds",
        ):
            assert field in source, field

    def test_forward_accounting_separates_the_counts(self):
        producer = _producer()
        source = inspect.getsource(producer.assemble_cell)
        for field in (
            "rollout_forward_count",
            "endpoint_forward_count",
            "extra_cfg_forward_count",
            "total_top_level_model_calls",
            "sc_cfg_token_cost",
        ):
            assert field in source, field

    def test_mask_after_eos_is_declared_unattributed(self):
        producer = _producer()
        source = inspect.getsource(producer.assemble_cell)
        assert "mask_after_eos runs after" in source
        assert "deliberately unattributed" in source

    def test_the_overhead_is_descriptive_only(self):
        producer = _producer()
        estimate = producer.overhead_estimate(
            [
                {
                    "paired_overhead_seconds": -0.01,
                    "off_wall_seconds": 1.0,
                }
            ]
        )
        assert estimate["resolvable"] is None
        assert estimate["resolution_status"] == "not_assessed"

    def test_provenance_records_the_instrumentation_contract(self):
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert '"instrumentation_contract"' in source
        assert "instance-local" in source
        assert '"coverage_events"' in source


class TestTaxonomy:
    def test_the_audit_parent_is_excluded_from_coverage(self):
        """Including it would double-count its own children."""
        producer = _producer()
        assert "solver_step_inclusive" in producer.EVENT_TAXONOMY
        assert "solver_step_inclusive" not in producer.COVERAGE_EVENTS
        assert set(producer.COVERAGE_EVENTS) == {
            "denoiser_forward",
            "solver_state_update",
            "endpoint_projection",
        }

    def test_the_descriptions_state_their_boundaries(self):
        taxonomy = _producer().EVENT_TAXONOMY
        assert "EXCLUDES the endpoint decoder forward" in taxonomy["denoiser_forward"]
        assert "PER-STEP difference" in taxonomy["solver_state_update"]
        assert "NOT in denoiser_forward" in taxonomy["endpoint_projection"]
        assert "excluded from coverage" in taxonomy["solver_step_inclusive"]

    def test_the_sc_cfg_cost_is_declared_not_decomposed(self):
        assert "SC-CFG" in _producer().EVENT_TAXONOMY["denoiser_forward"]

    def test_no_description_quotes_a_measurement(self):
        """A taxonomy description must not carry figures from another run."""
        import re

        pattern = re.compile(r"\d+(?:\.\d+)?\s*(?:ms|milliseconds|s\b|%)")
        for name, text in _producer().EVENT_TAXONOMY.items():
            assert not pattern.findall(text), f"{name} quotes a measurement"
