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


class TestElfProducerMeasurementValidity:
    """The producer's own measurement invariants, checked without a GPU."""

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
        spec = importlib.util.spec_from_file_location("_elf_profile2", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_default_bound_runner_must_be_deleted_before_its_objects(self):
        """The #173 closure leak, in the shape this producer had it.

        `run_off` binds the model as a DEFAULT ARGUMENT, so deleting the model
        while the function object survives keeps the weights resident — and the
        next trial's build would then run with the previous trial's allocation
        still live.
        """
        import gc
        import weakref

        class Model:
            pass

        def leaky():
            model = Model()

            def run(model=model):
                return model

            probe = weakref.ref(model)
            del model  # the object, but NOT the runner
            gc.collect()
            return run, probe

        run, probe = leaky()
        assert probe() is not None, "the leak this test documents is gone"
        del run
        gc.collect()
        assert probe() is None, "deleting the runner must release the model"

    def test_the_producer_deletes_its_runners_first(self):
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.profile_batch)
        # One interleaved `measure(arm)` replaced the separate run_off/run_on
        # loops; its runner is `run` and must still be deleted first.
        assert "del run, model" in source

    def test_events_and_wall_cover_the_same_window(self):
        """Warmup must be excluded from BOTH, not just from the wall.

        An earlier version divided event totals by `WARMUP + STEPS` while the
        wall kept only the timed steps, so coverage and wall described
        different intervals.
        """
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.timed_step_loop)
        assert "timer.reset()" in source
        # Check CODE, not prose: the docstring legitimately mentions
        # `WARMUP + STEPS` while explaining the bug it replaced.
        loop = inspect.getsource(producer.profile_batch)
        code = "\n".join(
            line for line in loop.splitlines() if not line.strip().startswith("#")
        )
        body = code.split('"""')[-1]
        assert "WARMUP + STEPS" not in body
        assert "/ STEPS" in body, "per-step division must use the timed window"

    def test_the_objective_publishes_an_exclusive_share(self):
        """Publishing only the inclusive parent pushes the objective's own work
        — CE/L2, target construction, masking — into the remainder."""
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.profile_batch)
        # The frozen taxonomy name is kept; the exclusive share rides on the
        # event's `exclusive_seconds` rather than a renamed event.
        assert "objective_loss_exclusive" not in source
        assert "exclusive_seconds=objective_exclusive" in source

    def test_required_events_are_asserted_not_inferred(self):
        """A broken caller lookup would otherwise drop an event silently."""
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.profile_batch)
        assert "measurement_invalid" in source
        assert 'body["call_count"] != STEPS' in source
        # Exact set equality, so an UNEXPECTED event is also refused.
        assert "set(ops) != required" in source

    def test_peak_memory_representative_is_fixed_in_advance(self):
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.profile_batch)
        assert "max(peak_allocated)" in source
        assert "peak_allocated_per_trial" in source

    def test_collation_covers_masks_and_transfer(self):
        """A dict comprehension over an already-moved batch measures nothing."""
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.collate)
        assert "build_self_attn_cond_masks" in source
        assert ".to(device)" in source

    def test_collation_masks_exclude_the_padded_region(self):
        """Behavioural, not textual: all-ones masks would train on padding.

        A source-only check let a mutant keep the call and overwrite its result
        with `ones`, which is exactly the defect `stage3_reduced_gate` warns
        about. This asserts the padded tail is actually masked out.
        """
        import numpy as np

        producer = self._producer()
        short = np.ones(12, dtype=np.int64)
        batch = producer.collate([short], device="cpu")
        mask = batch["attention_mask"]
        flat = mask.reshape(1, -1)
        assert float(flat.max()) != float(flat.min()), (
            "an all-uniform attention mask means the padded tail is not excluded"
        )


class TestElfArtifactIntegrity:
    """The artifact must describe what actually ran."""

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
        spec = importlib.util.spec_from_file_location("_elf_profile3", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_length_and_dtype_cannot_be_misreported_from_the_cli(self):
        """The producer recorded `sequence_length: 256` while truncating to
        1024, because the CLI exposed a `--seq-len` that changed only the
        record. Neither flag may exist."""
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.parse_args)
        assert '"--seq-len"' not in source
        assert '"--dtype"' not in source
        cell = inspect.getsource(producer.profile_batch)
        assert "sequence_length=SEQUENCE_LENGTH" in cell
        assert "args.seq_len" not in cell
        assert "args.dtype" not in cell

    def test_the_encoder_revision_is_pinned_and_verified(self):
        """Recording what a run fetched documents the past; pinning makes the
        next run reproducible."""
        import inspect

        producer = self._producer()
        assert len(producer.T5_REVISION) == 40
        source = inspect.getsource(producer.load_fixture)
        assert "revision=T5_REVISION" in source
        assert "!= T5_REVISION" in source

    def test_collation_pads_ragged_rows(self):
        """Truncation, padding and stacking are per-step collator work.

        Behavioural: hand in genuinely ragged rows and require a rectangular,
        correctly-masked batch out — a pre-stacked fixture would let the timed
        scope measure only the copy.
        """
        import numpy as np

        producer = self._producer()
        long_row = np.ones(producer.SEQUENCE_LENGTH + 50, dtype=np.int64)
        short_row = np.ones(10, dtype=np.int64)
        batch = producer.collate([long_row, short_row], device="cpu")
        assert batch["input_ids"].shape == (2, producer.SEQUENCE_LENGTH)
        mask = batch["attention_mask"]
        flat = mask.reshape(2, -1)
        # The truncated row is fully valid; the short row must have an excluded
        # tail, so the two rows cannot have identical masks.
        assert not bool((flat[0] == flat[1]).all()), (
            "a ragged pair must not produce identical attention masks"
        )

    def test_arms_are_interleaved_with_alternating_order(self):
        import inspect

        producer = self._producer()
        source = inspect.getsource(producer.profile_batch)
        assert 'order = ("off", "on") if trial % 2 == 0 else ("on", "off")' in source

    def test_oom_phases_are_distinguished(self):
        """`build`, `warmup` and `timed` are different findings."""
        import inspect

        producer = self._producer()
        assert hasattr(producer, "OomInPhase")
        cell = inspect.getsource(producer.profile_batch)
        assert 'oom_phase("build")' in cell
        assert 'oom_phase("timed")' in cell
        loop = inspect.getsource(producer.timed_step_loop)
        assert 'oom_phase("warmup")' in loop

    def test_oom_in_phase_carries_its_phase_and_cause(self):
        producer = self._producer()
        cause = RuntimeError("out of memory")
        error = producer.OomInPhase("warmup", cause)
        assert error.phase == "warmup"
        assert error.cause is cause
        assert "warmup" in str(error)

    def test_the_phase_context_tags_with_its_own_phase(self):
        """Behavioural: a mutant hardcoding one phase label passed a
        call-site-only check, because those tests never raised anything."""
        import torch

        producer = self._producer()
        for phase in ("build", "warmup", "timed"):
            with (
                pytest.raises(producer.OomInPhase) as caught,
                producer.oom_phase(phase),
            ):
                raise torch.cuda.OutOfMemoryError("synthetic")
            assert caught.value.phase == phase, (
                f"oom_phase({phase!r}) must tag with that phase, not "
                f"{caught.value.phase!r}"
            )

    def test_the_phase_context_leaves_other_errors_alone(self):
        producer = self._producer()
        with pytest.raises(ValueError), producer.oom_phase("build"):
            raise ValueError("unrelated")

    def test_successful_run_provenance_names_the_encoder(self):
        """The encoder identity must reach SUCCESSFUL runs, not only OOM
        records: an artifact whose frozen inputs are not all named cannot be
        reproduced from itself."""
        import argparse

        producer = self._producer()
        fixture = {
            "encoder_name": "t5-small",
            "encoder_revision": producer.T5_REVISION,
        }
        run = producer.provenance(argparse.Namespace(device="cpu"), fixture)
        encoder = run["fixture"]["encoder"]
        assert encoder == f"t5-small@{producer.T5_REVISION}"
        # An exact 40-character commit, not a bare name.
        assert len(encoder.split("@")[1]) == 40


class TestCudaEventTimerReset:
    def test_reset_discards_pending_scopes(self):
        """A mutant made `reset()` keep pending events, which then leaked into
        the next window's totals. Count-only checks missed it because the
        totals looked empty until the stale scope was collected."""
        from unturtle.eval.cuda_event_timer import CudaEventTimer

        timer = CudaEventTimer(device="cpu")
        with timer.measure("stale"):
            pass
        timer.reset()
        # Nothing pending, so collect() must be a no-op and results stay empty.
        timer.collect()
        assert timer.result() == {}

    def test_reset_then_new_window_counts_only_new_work(self):
        from unturtle.eval.cuda_event_timer import CudaEventTimer

        timer = CudaEventTimer(device="cpu")
        with timer.measure("warmup_only"):
            pass
        timer.reset()
        with timer.measure("counted"):
            pass
        timer.collect()
        assert set(timer.result()) == {"counted"}


class TestCollectSyncFlag:
    """`collect(synchronize=False)` must not add a second device sync.

    The flag is observed by WRAPPING the real synchronize, and the caller
    synchronizes after the scope closes — mirroring the producer, where the step
    boundary syncs before collection. Two earlier attempts were wrong for their
    own reasons rather than the code's: stubbing synchronize out removed the sync
    the CUDA events need before `elapsed_time`, and syncing before recording the
    events does not complete events created afterwards.
    """

    @staticmethod
    def _observe(synchronize: bool):
        import torch

        from unturtle.eval.cuda_event_timer import CudaEventTimer

        timer = CudaEventTimer(device="cuda:0")
        with timer.measure("op"):
            torch.zeros(8, device="cuda:0")
        if not synchronize:
            # What the producer does: the step boundary has already synced, so
            # the events are complete before collection is asked to skip it.
            torch.cuda.synchronize()

        calls = []
        original = torch.cuda.synchronize

        def observing(*args, **kwargs):
            calls.append(True)
            return original(*args, **kwargs)

        torch.cuda.synchronize = observing
        try:
            timer.collect(synchronize=synchronize)
        finally:
            torch.cuda.synchronize = original
        return calls, timer.result()

    def test_synchronize_false_does_not_sync(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA to have pending events")
        calls, result = self._observe(synchronize=False)
        assert calls == [], "collect(synchronize=False) must not synchronize"
        assert result["op"]["call_count"] == 1

    def test_synchronize_true_does_sync(self):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("needs CUDA to have pending events")
        calls, result = self._observe(synchronize=True)
        assert calls == [True]
        assert result["op"]["call_count"] == 1
