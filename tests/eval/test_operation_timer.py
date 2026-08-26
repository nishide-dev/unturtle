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

"""#166 Stage-1 operation timer and ELF call-site attribution."""

from __future__ import annotations

import pytest


class TestOperationTimer:
    def test_accumulates_inclusive_time_and_calls(self):
        from unturtle.eval.operation_timer import OperationTimer

        timer = OperationTimer()
        for _ in range(3):
            with timer.measure("op"):
                pass
        result = timer.result()
        assert result["op"]["call_count"] == 3
        assert result["op"]["inclusive_seconds"] >= 0.0

    def test_distinct_names_stay_separate(self):
        """The load-bearing property: two operations must never merge."""
        from unturtle.eval.operation_timer import OperationTimer

        timer = OperationTimer()
        with timer.measure("a"):
            pass
        with timer.measure("b"):
            pass
        assert set(timer.result()) == {"a", "b"}

    def test_nested_measures_both_record(self):
        from unturtle.eval.operation_timer import OperationTimer

        timer = OperationTimer()
        with timer.measure("parent"), timer.measure("child"):
            pass
        result = timer.result()
        assert result["parent"]["call_count"] == 1
        assert result["child"]["call_count"] == 1

    def test_an_exception_still_records_the_operation(self):
        """A failing step must not silently vanish from the accounting."""
        from unturtle.eval.operation_timer import OperationTimer

        timer = OperationTimer()
        with pytest.raises(RuntimeError), timer.measure("op"):
            raise RuntimeError("boom")
        assert timer.result()["op"]["call_count"] == 1

    def test_reset_clears_everything(self):
        from unturtle.eval.operation_timer import OperationTimer

        timer = OperationTimer()
        with timer.measure("op"):
            pass
        timer.reset()
        assert timer.result() == {}


class TestCallerScope:
    """Call-site attribution is how the two ELF auxiliary forwards stay apart.

    They call the SAME callable from different functions, and Stage 0's
    hypothesis is about them being separate costs, so keying on the caller is
    what makes the hypothesis testable without editing the pack.
    """

    def test_reports_the_calling_function(self):
        from unturtle.eval.operation_timer import caller_scope

        def outer():
            return caller_scope(depth=2)

        assert outer() == "test_reports_the_calling_function"

    def test_two_call_sites_are_distinguished(self):
        from unturtle.eval.operation_timer import caller_scope

        def probe():
            return caller_scope(depth=2)

        def site_a():
            return probe()

        def site_b():
            return probe()

        assert site_a() == "site_a"
        assert site_b() == "site_b"
        assert site_a() != site_b()

    def test_depth_beyond_the_stack_is_typed_not_crashed(self):
        from unturtle.eval.operation_timer import caller_scope

        assert caller_scope(depth=10_000) == "unknown"


class TestElfTaxonomy:
    """The producer's declared taxonomy, checked without a GPU."""

    @staticmethod
    def _producer():
        import importlib.util
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "elf"
            / "training_profile.py"
        )
        spec = importlib.util.spec_from_file_location("_elf_profile", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_two_auxiliary_forwards_are_distinct_events(self):
        """Merging them into one `self_conditioning` event would destroy the
        finding this profile exists to test: they sit at different call sites
        under different conditions."""
        producer = self._producer()
        events = producer.CALL_SITE_EVENTS
        assert events["compute_shared_uncond"] == "sc_shared_uncond_forward"
        assert events["get_sc_cond_and_uncond"] == "sc_conditional_forward"
        assert events["elf_training_loss"] == "trained_forward"
        assert len(set(events.values())) == 3

    def test_the_measurement_window_is_frozen_not_cli(self):
        """Three #166 gates failed because a verdict depended on `--steps`, so
        the window must be module constants and the CLI must not expose it."""
        import inspect

        producer = self._producer()
        assert producer.TRIALS >= 2
        assert producer.STEPS >= 1
        assert producer.WARMUP >= 1
        source = inspect.getsource(producer.parse_args)
        assert '"--steps"' not in source
        assert '"--warmup"' not in source
        assert '"--trials"' not in source

    def test_the_timed_loop_uses_the_frozen_constants(self):
        """A loop reading `args` would reintroduce the same defect."""
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.timed_step_loop)
        assert "WARMUP" in source and "STEPS" in source
        assert "args." not in source

    def test_representative_batches_cover_the_protocol_cells(self):
        producer = self._producer()
        assert producer.BATCH_SIZES == (1, 8, 32)
