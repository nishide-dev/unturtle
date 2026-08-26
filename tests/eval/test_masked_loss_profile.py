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

"""#166 Stage-1 — the masked dense loss producer's measurement invariants.

Structural assertions only. Notably absent: any check that the realized mask
fraction is near 0.5. Twenty-four observations move materially with the seed
schedule, so pinning a value would fit this execution rather than the producer,
and the ledger's own regime caveat applies to the number either way.
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib

import pytest


def _producer():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "kernels"
        / "masked_loss_profile.py"
    )
    spec = importlib.util.spec_from_file_location("_masked_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFrozenConfiguration:
    def test_the_window_is_not_exposed_on_the_cli(self):
        """Three #166 gates failed when a verdict depended on `--steps`."""
        source = inspect.getsource(_producer().parse_args)
        for flag in ('"--steps"', '"--warmup"', '"--trials"', '"--seq-len"'):
            assert flag not in source

    def test_the_optimizer_is_pinned_not_defaulted(self):
        """A torch release changing an AdamW default would silently change the
        step this profile attributes."""
        producer = _producer()
        assert producer.OPTIMIZER == "torch.optim.AdamW"
        assert producer.BETAS == (0.9, 0.999)
        assert producer.EPS == 1e-8
        assert producer.WEIGHT_DECAY == 0.01
        assert producer.AMSGRAD is False
        source = inspect.getsource(producer.build)
        for keyword in ("betas=BETAS", "eps=EPS", "weight_decay=WEIGHT_DECAY"):
            assert keyword in source

    def test_the_trial_seeds_are_distinct(self):
        producer = _producer()
        assert len(set(producer.TRIAL_SEEDS)) == len(producer.TRIAL_SEEDS)
        assert len(producer.TRIAL_SEEDS) == producer.TRIALS

    def test_the_trainer_receives_the_trial_seed(self):
        """`Trainer.__init__` calls `set_seed(args.seed)`, default 42, which
        silently erased the per-trial seeding — all three trials then shared one
        RNG fingerprint."""
        source = inspect.getsource(_producer().build)
        assert "seed=seed," in source

    def test_the_taxonomy_is_exactly_six_events(self):
        producer = _producer()
        assert (
            frozenset(
                {
                    "noising",
                    "attention",
                    "lm_head_projection",
                    "loss",
                    "backward",
                    "optimizer_step",
                }
            )
            == producer.REQUIRED_EVENTS
        )

    def test_the_sparse_path_is_not_reachable(self):
        """Sparse is a separate producer: its hook topology and required event
        set differ, and it needs pre-noised batches where this cell measures
        `noising` itself."""
        source = inspect.getsource(_producer().build)
        assert "sparse_lm_head=False" in source


class TestHookTargets:
    def test_the_loss_symbol_is_patched_in_the_trainer_module(self):
        """`trainer.py` imports the loss directly, so patching the kernel
        module would wrap a name nobody calls — the wrong-symbol failure the
        #166 hybrid gate already produced once."""
        source = inspect.getsource(_producer().instrumented)
        assert "import unturtle.diffusion.trainer as trainer_module" in source
        # ASSIGNED, not just referenced: a mutant replacing the assignment with
        # `pass` kept the mention and survived.
        assert "trainer_module.fast_masked_diffusion_loss = timed_loss" in source
        assert "trainer_module.fast_masked_diffusion_loss = original_loss" in source

    def test_patching_the_kernel_module_would_not_be_observed(self):
        """The premise, verified: the trainer holds its own binding."""
        import unturtle.diffusion.trainer as trainer_module
        import unturtle.kernels.masked_diffusion_loss as kernel_module

        sentinel = object()
        original = kernel_module.fast_masked_diffusion_loss
        try:
            kernel_module.fast_masked_diffusion_loss = sentinel
            assert trainer_module.fast_masked_diffusion_loss is not sentinel
        finally:
            kernel_module.fast_masked_diffusion_loss = original

    def test_the_process_is_proxied_on_the_trainer_instance(self):
        source = inspect.getsource(_producer().instrumented)
        assert "trainer.forward_process" in source

    def test_every_hook_and_patch_is_restored_in_finally(self):
        source = inspect.getsource(_producer().instrumented)
        assert "finally:" in source
        tail = source.split("finally:")[1]
        assert "handle.remove()" in tail
        assert "trainer_module.fast_masked_diffusion_loss = original_loss" in tail
        assert "trainer.forward_process = original_process" in tail


class TestEventTimerDiscipline:
    def test_hooks_do_not_synchronize(self):
        """Per-scope synchronization would scale with layer count and change
        the thing being measured."""
        source = inspect.getsource(_producer().instrumented)
        body = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        # Strip docstrings: they legitimately discuss the rule being enforced.
        body = "".join(body.split('"""')[::2])
        assert "synchronize" not in body

    def test_elapsed_times_are_read_after_the_step_boundary(self):
        source = inspect.getsource(_producer().timed_step_loop)
        assert "timer.collect()" in source
        assert "timer.reset()" in source

    def test_pending_scopes_are_refused_not_dropped(self):
        """Uncollected work must fail loudly: totals that silently omit scopes
        would understate coverage."""
        from unturtle.eval.cuda_event_timer import CudaEventTimer

        timer = CudaEventTimer(device="cpu")
        with timer.measure("op"):
            pass
        with pytest.raises(RuntimeError, match="never collected"):
            timer.result()
        timer.collect()
        assert timer.result()["op"]["call_count"] == 1

    def test_reset_discards_completed_and_pending(self):
        from unturtle.eval.cuda_event_timer import CudaEventTimer

        timer = CudaEventTimer(device="cpu")
        with timer.measure("op"):
            pass
        timer.collect()
        timer.reset()
        assert timer.result() == {}


class TestMaskDiagnosticIsolation:
    def test_the_diagnostic_is_not_called_from_the_timed_arms(self):
        """Collecting the mask inside `measure()` put a GPU reduction and a
        Python append into the OFF wall time that IS the verdict."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell_for)
        measured = source.split("def measure(")[1].split("for trial in range")[0]
        assert "diffusion_mask" not in measured
        assert "mask_by_trial" not in measured

    def test_the_replay_runs_full_steps_not_the_process_alone(self):
        """Eight bare `forward_process` calls diverge from the real stream at
        the second draw, because the model forward also consumes the RNG."""
        source = inspect.getsource(_producer().profile_cell_for)
        replay = source.split("mask diagnostic")[1]
        assert "one_step(" in replay
        assert "range(WARMUP)" in replay
        # Two `one_step` calls: the warmup alignment loop and the counted loop.
        # A mutant replacing the counted one with `pass` left the other behind.
        assert replay.count("one_step(model, trainer, clean, optimizer)") == 2
        counted = replay.split("captured.clear()")[1]
        assert "one_step(model, trainer, clean, optimizer)" in counted, (
            "the counted loop must run full steps, not skip them"
        )

    def test_the_denominator_is_maskable_tokens(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert "labels != -100" in source
        assert "maskable_tokens" in source

    def test_a_zero_denominator_is_typed_not_divided(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert "if maskable == 0:" in source
        assert "mask_invalid" in source

    def test_the_capture_proxy_only_stores_output(self):
        """It must not sample or allocate: doing so would consume the RNG the
        replay exists to reproduce."""
        source = inspect.getsource(_producer().profile_cell_for)
        capture = source.split("def capturing(")[1].split("if original_process")[0]
        for forbidden in ("torch.rand", "randn", "randint", "manual_seed"):
            assert forbidden not in capture

    def test_each_trial_captures_into_its_own_list(self):
        """Behavioural: count alone would miss a mutant that appends 24 values
        to one list and slices it into three afterwards."""
        producer = _producer()
        source = inspect.getsource(producer.profile_cell_for)
        # A fresh `captured` list per trial, appended to `mask_by_trial`.
        replay = source.split("mask diagnostic")[1]
        assert "captured: list" in replay
        assert "mask_by_trial.append(draws)" in replay
        assert replay.index("captured: list") < replay.index("for _ in range(STEPS)")

    def test_the_interpretation_field_refuses_a_population_claim(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell_for)
        assert "sampling_contract" in source
        assert "realized_mask_fraction_interpretation" in source
        assert "not an estimate of the process's population" in source


class TestPairingAndLifecycle:
    def test_pairing_is_gated_on_state_not_seed(self):
        """Three distinct seed integers coexisted with one identical RNG state,
        so the seed value is not evidence of pairing."""
        source = inspect.getsource(_producer().profile_cell_for)
        assert "rng_fingerprint()" in source
        assert "state_fingerprint(" in source
        assert "paired arms disagree on" in source
        assert "mismatched" in source

    def test_a_paired_state_mismatch_is_refused(self):
        """Behavioural: a source-only check let `if mismatched:` become
        `if False:` and survive."""
        producer = _producer()
        # Drive the comparison the producer uses, with one arm perturbed.
        tables = {700: {"off": "aaa", "on": "bbb"}}
        mismatched = {
            seed: arms
            for seed, arms in tables.items()
            if len(arms) == 2 and arms["off"] != arms["on"]
        }
        assert mismatched, "the comparison itself must flag a differing pair"
        source = inspect.getsource(producer.profile_cell_for)
        # And the producer must ACT on it rather than compute it and move on.
        acted = source.split("mismatched = {")[1]
        assert "if mismatched:" in acted
        assert "return invalid(" in acted

    def test_trial_independence_is_gated(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert "independent draws" in source
        assert 'arms.get("off") for arms in rng_states.values()' in source

    def test_the_fingerprint_does_not_consume_rng(self):
        """Reading state must not advance it, or the measurement it guards
        would differ from the one that ran."""
        import torch

        from unturtle.eval.profile_harness import COVERAGE_TOLERANCE  # noqa: F401

        producer = _producer()
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()
        producer.rng_fingerprint()
        assert torch.equal(torch.get_rng_state(), before)

    def test_lifecycle_release_is_verdict_bearing(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert "weakref.ref(model)" in source
        assert "outlived its measurement call" in source

    def test_peak_memory_is_off_arm_only_with_max_fixed_in_advance(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert 'if arm == "off" and torch.cuda.is_available():' in source
        assert "max(peak_allocated)" in source
        assert "peak_allocated_per_trial" in source

    def test_oom_phases_are_distinguished(self):
        producer = _producer()
        assert hasattr(producer, "OomInPhase")
        source = inspect.getsource(producer.profile_cell_for)
        assert 'oom_phase("build")' in source
        assert 'oom_phase("timed")' in source
        assert 'oom_phase("warmup")' in inspect.getsource(producer.timed_step_loop)

    def test_the_phase_context_tags_with_its_own_phase(self):
        import torch

        producer = _producer()
        for phase in ("build", "warmup", "timed"):
            with (
                pytest.raises(producer.OomInPhase) as caught,
                producer.oom_phase(phase),
            ):
                raise torch.cuda.OutOfMemoryError("synthetic")
            assert caught.value.phase == phase


class TestTaxonomyIntegrity:
    def test_model_forward_is_a_diagnostic_not_an_event(self):
        """Publishing it as an event would double count attention and the LM
        head; publishing a `model_other` would invent a frozen-taxonomy entry."""
        source = inspect.getsource(_producer().profile_cell_for)
        assert "model_forward_inclusive_seconds" in source
        assert "model_forward_residual_seconds" in source
        # `model_other` must not be a published event NAME. The phrase appears
        # in a comment explaining why it is not, so check the event list.
        assert 'name="model_other"' not in source
        assert "OperationEvent(" in source

    def test_a_negative_residual_is_invalid_not_clamped(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert "if residual < 0:" in source
        assert "max(0" not in source.split("if residual < 0:")[1][:200]

    def test_exact_event_set_and_counts_are_asserted(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert "set(ops) != REQUIRED_EVENTS" in source
        assert "expected_counts" in source
        assert 'expected_counts["attention"] = layers * STEPS' in source

    def test_the_fixture_shape_is_recorded(self):
        producer = _producer()
        source = inspect.getsource(producer.profile_cell_for)
        for field in ("vocab_size", "layers", "steps_per_trial", "trials"):
            assert f'"{field}"' in source

    def test_the_mask_token_is_inside_the_vocabulary(self):
        """gpt2's EOS is 50256, out of range for a 32000-vocab model, and
        triggered a device-side assert the moment the process wrote it."""
        producer = _producer()
        tokenizer = producer._sparse_benchmark()._tokenizer()
        assert tokenizer.mask_token_id is not None
        assert tokenizer.mask_token_id < min(producer.VOCAB_SIZES)

    def test_importing_the_shared_benchmark_runs_nothing(self):
        """A shared fixture module must not execute its CLI on import."""
        producer = _producer()
        module = producer._sparse_benchmark()
        assert hasattr(module, "_tokenizer")
        assert hasattr(module, "main")


class TestReviewedMeasurementGates:
    """The five gates added after the Commit-A review."""

    def test_collection_happens_inside_the_timed_interval(self):
        """The clock used to stop BEFORE `collect()`, so collection overhead sat
        outside `wall_on_trials`."""
        source = inspect.getsource(_producer().timed_step_loop)
        body = source.split("seconds: list[float] = []")[1]
        collect_at = body.index("timer.collect(synchronize=False)")
        append_at = body.index("seconds.append")
        assert collect_at < append_at, (
            "collection must be folded in before the clock is read"
        )

    def test_the_timed_step_synchronizes_once(self):
        """`collect()` synchronized a second time, making it two per step."""
        source = inspect.getsource(_producer().timed_step_loop)
        body = source.split("seconds: list[float] = []")[1]
        assert body.count("sync()") == 2  # one before start, one at the boundary
        assert "collect(synchronize=False)" in body

    def test_collect_can_skip_its_own_sync(self):
        from unturtle.eval.cuda_event_timer import CudaEventTimer

        timer = CudaEventTimer(device="cpu")
        with timer.measure("op"):
            pass
        timer.collect(synchronize=False)
        assert timer.result()["op"]["call_count"] == 1

    def test_peak_memory_is_reset_after_warmup(self):
        """Resetting before warmup counted warmup transients and lazy
        optimizer-state creation in the peak."""
        producer = _producer()
        loop = inspect.getsource(producer.timed_step_loop)
        assert "on_warmup_done" in loop
        assert loop.index("warmup_seconds =") < loop.index("on_warmup_done()")
        cell = inspect.getsource(producer.profile_cell_for)
        assert "def after_warmup(" in cell
        assert "reset_peak_memory_stats()" in cell
        # Both call sites must pass the callback; `on_warmup_done=None` at
        # either one silently restores the pre-warmup reset.
        assert cell.count("on_warmup_done=after_warmup") == 2
        assert "on_warmup_done=None" not in cell

    def test_the_warmup_callback_actually_fires(self):
        """Behavioural: a mutant passing None kept every source string."""
        producer = _producer()
        fired = []
        producer.timed_step_loop(
            lambda: None,
            device="cpu",
            on_warmup_done=lambda: fired.append(True),
        )
        assert fired == [True], "the warmup hook must be invoked exactly once"

    def test_the_residual_is_checked_per_trial_before_any_median(self):
        """A difference of medians lets one trial's negative residual hide
        behind another's surplus."""
        source = inspect.getsource(_producer().profile_cell_for)
        assert "per_trial_residual" in source
        assert "zip(per_trial_model_forward, per_trial_ops, strict=True)" in source
        assert source.index("per_trial_residual = [") < source.index(
            "residual = statistics.median(per_trial_residual)"
        )
        # The check must iterate the real list: a mutant swapped it for `[]`,
        # keeping every other string intact.
        assert "enumerate(per_trial_residual)" in source
        assert "enumerate([])" not in source

    def test_replay_completeness_is_gated(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert "len(mask_by_trial) != TRIALS" in source
        assert "len(draws) != STEPS" in source
        assert "len(mask_fractions) != TRIALS * STEPS" in source
        assert "forward_process is None" in source

    def test_replay_start_state_must_match_the_measured_trial(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert "replay_fingerprints" in source
        assert "is not the one" in source
        assert "fingerprint != measured" in source
        assert "if False:" not in source

    def test_each_seed_runs_exactly_one_off_and_one_on(self):
        source = inspect.getsource(_producer().profile_cell_for)
        assert 'set(arms) != {"off", "on"}' in source

    def test_a_non_default_cuda_device_is_refused(self):
        """Sync, events, peak stats and the GPU name all target the default
        device, so any other index would be silently mis-recorded."""
        producer = _producer()
        producer.require_supported_device("cuda:0")
        producer.require_supported_device("cpu")
        with pytest.raises(SystemExit, match="not supported"):
            producer.require_supported_device("cuda:1")

    def test_optimizer_flags_are_resolved_values_not_placeholders(self):
        """`"default-resolved"` was a placeholder, not a value."""
        import torch

        producer = _producer()
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        record = producer.optimizer_fingerprint(optimizer)
        resolved = record["resolved_defaults"]
        for key in ("foreach", "fused", "capturable", "maximize", "differentiable"):
            assert resolved[key] != "default-resolved"
        assert record["param_groups"]
        assert "torch_version" in record
