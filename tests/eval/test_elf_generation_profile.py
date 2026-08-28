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
    recorder.phase = "rollout"
    recorder.steps.append({"name": name, "forwards": []})


class TestInstanceLocalPatching:
    """A class-level `type(model).forward` patch instruments EVERY instance of
    the class while it is live. Measured before the fix: calling an unrelated
    instance recorded a rollout forward on this recorder."""

    def test_an_unrelated_instance_is_not_observed(self):
        producer = _producer()
        target, bystander = StubModel(), StubModel()
        recorder = producer.Recorder(torch, mode="count")
        with producer.instrumented(target, recorder):
            _open_step(recorder)
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
        ):
            _open_step(recorder)
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
                pytest.raises(SystemExit, match="refusing to nest"),
                producer.instrumented(model, producer.Recorder(torch, mode="count")),
            ):
                pass
        assert "forward" not in model.__dict__

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
            _open_step(recorder)
            model(torch.zeros(1))
            recorder.phase = None
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
