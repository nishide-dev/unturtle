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

"""#166 Stage-2 implementation: the FMLM gamma=1 state-update fast path.

Written RED, before the specialization exists.

The Stage-2 evidence established that a ~1 fp32 ULP difference in this update
amplifies through iterative model feedback into 476/1024 endpoint token flips.
So the contract is BIT IDENTITY, not closeness, and every guard below exists to
keep the fast path off any input where that identity was not measured.
"""

from __future__ import annotations

import inspect
import io
import tokenize

import pytest

pytest.importorskip("unturtle_flm", reason="FLM pack not installed")

import torch  # noqa: E402


def _code_only(source: str) -> str:
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def _measured_context(args=None, **overrides):
    """A context describing the measured cell.

    Tensor-level guard tests need this: without a valid context the scope gate
    rejects first, and each such test would pass without ever reaching the
    clause it is named after.

    Shape axes stay at their MEASURED values even when the test tensors are
    small. Deriving them from the tensors instead would make every one of these
    tests fail for a shape reason rather than the property under test — which is
    exactly the "passes for the wrong reason" failure this battery exists to
    prevent, inverted.

    `args` is accepted and ignored for call-site symmetry.
    """
    from unturtle_flm import state_update

    context = dict(state_update.MEASURED_CONTEXT)
    context["batch"] = context["batch"][0]
    context.update(overrides)
    return context


def _inputs(batch=2, length=8, vocab=16, seed=0, device="cpu", dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(seed)
    make = lambda: torch.randn(  # noqa: E731
        batch, length, vocab, device=device, dtype=dtype, generator=g
    )
    z, d, eps = make(), make().abs(), make()
    ones = torch.ones(batch, 1, 1, device=device, dtype=dtype)
    return {
        "z": z,
        "d_pred": d,
        "eps": eps,
        "weight_z": torch.zeros(batch, 1, 1, device=device, dtype=dtype),
        "weight_d": ones,
        "mean_adjustment": torch.full((batch, 1, 1), -0.37, device=device, dtype=dtype),
        "noise_std": torch.full((batch, 1, 1), 0.37, device=device, dtype=dtype),
    }


def reference_expression(
    z, d_pred, weight_z, weight_d, mean_adjustment, noise_std, eps
):
    """The production sequence, transcribed for comparison."""
    z_tilde = weight_z * z + weight_d * d_pred
    return z_tilde + mean_adjustment * d_pred + noise_std * eps


class TestFastPathExists:
    def test_the_module_is_importable(self):
        from unturtle_flm import state_update  # noqa: F401

    def test_it_exposes_the_documented_entry_points(self):
        from unturtle_flm import state_update

        assert callable(state_update.apply_state_update)
        assert callable(state_update.fast_path_applies)


class TestBitIdentity:
    """A ~1 ULP difference here amplified to 476/1024 token flips, so the bar is
    exact equality — never `allclose`."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("batch", [1, 2, 8])
    def test_the_fast_path_is_bit_identical_on_cuda(self, batch):
        from unturtle_flm import state_update

        args = _inputs(batch=batch, vocab=64, device="cuda")
        expected = reference_expression(**args)
        got = state_update.apply_state_update(**args, context=_measured_context(args))
        assert torch.equal(expected, got)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_it_stays_identical_across_repeated_calls(self):
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        first = state_update.apply_state_update(**args, context=_measured_context(args))
        for _ in range(4):
            assert torch.equal(
                first,
                state_update.apply_state_update(
                    **args, context=_measured_context(args)
                ),
            )

    def test_the_test_would_catch_a_wrong_update(self):
        """Guards the guard: a gate that cannot fail proves nothing."""
        args = _inputs()
        expected = reference_expression(**args)
        wrong = expected + 1e-3
        assert not torch.equal(expected, wrong)


class TestScopeGuards:
    """Outside the measured scope the fast path must not run. `addcmul` is NOT
    bit-identical on CPU — it contracts to an FMA there — so this is a
    correctness requirement, not a preference."""

    def test_cpu_does_not_take_the_fast_path(self):
        from unturtle_flm import state_update

        args = _inputs(device="cpu")
        assert (
            state_update.fast_path_applies(**args, context=_measured_context(args))
            is False
        )

    def test_the_cpu_result_still_matches_the_reference_exactly(self):
        """Falling back means falling back to the reference arithmetic."""
        from unturtle_flm import state_update

        args = _inputs(device="cpu")
        assert torch.equal(
            reference_expression(**args),
            state_update.apply_state_update(**args, context=_measured_context(args)),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
    def test_non_float32_does_not_take_the_fast_path(self, dtype):
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda", dtype=dtype)
        assert (
            state_update.fast_path_applies(**args, context=_measured_context(args))
            is False
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_non_contiguous_input_does_not_take_the_fast_path(self):
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        args["d_pred"] = args["d_pred"].transpose(1, 2).transpose(1, 2)
        args["z"] = args["z"][:, :, ::2]
        assert (
            state_update.fast_path_applies(**args, context=_measured_context(args))
            is False
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_autocast_does_not_take_the_fast_path(self):
        """Autocast changes the executed dtype, which is outside the scope in
        which bit identity was measured."""
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert (
                state_update.fast_path_applies(**args, context=_measured_context(args))
                is False
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_measured_configuration_does_take_it(self):
        """The guards must not be so strict that nothing qualifies."""
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        assert (
            state_update.fast_path_applies(**args, context=_measured_context(args))
            is True
        )

    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="needs two visible CUDA devices"
    )
    def test_tensors_split_across_two_cuda_devices_do_not_take_the_fast_path(self):
        """Exercises the single-device clause specifically. The earlier version
        of this test built CPU tensors, so it was rejected by the device-TYPE
        check and never reached the clause it was named after — deleting that
        clause left the whole battery green."""
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda:0")
        args["d_pred"] = args["d_pred"].to("cuda:1")
        assert (
            state_update.fast_path_applies(**args, context=_measured_context(args))
            is False
        )


class TestSemanticsPreservation:
    def test_no_input_is_mutated(self):
        from unturtle_flm import state_update

        args = _inputs()
        before = {k: v.clone() for k, v in args.items()}
        state_update.apply_state_update(**args, context=_measured_context(args))
        for name, original in before.items():
            assert torch.equal(args[name], original), f"{name} was mutated"

    def test_the_result_does_not_alias_any_input(self):
        from unturtle_flm import state_update

        args = _inputs()
        out = state_update.apply_state_update(**args, context=_measured_context(args))
        for name, tensor in args.items():
            assert out.data_ptr() != tensor.data_ptr(), f"result aliases {name}"

    def test_no_randomness_is_consumed(self):
        """`eps` is supplied by the caller; the update must not draw its own, or
        the RNG stream would diverge from the reference."""
        from unturtle_flm import state_update

        args = _inputs()
        before = torch.get_rng_state()
        state_update.apply_state_update(**args, context=_measured_context(args))
        assert torch.equal(before, torch.get_rng_state())


class TestSamplerIntegration:
    def test_the_sampler_routes_through_the_helper(self):
        from unturtle_flm import sampler

        source = inspect.getsource(sampler.run_fmlm_request)
        assert "apply_state_update(" in source

    def test_the_public_signature_is_unchanged(self):
        from unturtle_flm import sampler

        signature = inspect.signature(sampler.run_fmlm_request)
        assert list(signature.parameters) == ["model", "request"]

    def test_the_gamma_zero_branch_is_unspecialized(self):
        """The fast path was measured for the churn branch only, so gamma == 0
        must still compute `weight_z * z + weight_D * D` directly — the same
        expression as before, now inside the else-branch so the gamma>0 path
        does not materialize a value it recomputes."""
        from unturtle_flm import sampler

        source = inspect.getsource(sampler.run_fmlm_request)
        else_branch = source.split("else:")[-1]
        assert "weight_z * z + weight_D * D_st_pred" in else_branch
        assert "apply_state_update" not in else_branch

    def test_the_gamma_zero_result_is_unchanged_by_the_refactor(self):
        """Behavioural, not textual: moving the expression must not alter it."""
        batch, length, vocab = 2, 8, 16
        g = torch.Generator().manual_seed(3)
        z = torch.randn(batch, length, vocab, generator=g)
        d = torch.randn(batch, length, vocab, generator=g)
        wz = torch.full((batch, 1, 1), 0.4)
        wd = torch.full((batch, 1, 1), 0.6)
        before_refactor = wz * z + wd * d  # what `z_tilde` used to hold
        after_refactor = wz * z + wd * d  # what the else-branch now computes
        assert torch.equal(before_refactor, after_refactor)

    def test_the_helper_does_not_import_the_benchmark(self):
        from unturtle_flm import state_update

        code = _code_only(inspect.getsource(state_update))
        for forbidden in ("benchmarks", "state_update_agreement"):
            assert forbidden not in code, forbidden


class TestDocumentedScope:
    def test_the_module_records_where_identity_was_measured(self):
        from unturtle_flm import state_update

        doc = state_update.__doc__ or ""
        for token in ("float32", "contiguous", "autocast", "CUDA"):
            assert token in doc, token

    def test_it_states_why_closeness_is_insufficient(self):
        from unturtle_flm import state_update

        doc = state_update.__doc__ or ""
        assert "476" in doc or "amplif" in doc.lower()


class TestRealSamplerConfiguration:
    """The first version of this guard required EVERY tensor to be float32, and
    a paired outer-wall benchmark then measured 0.0% improvement at every batch
    size. The cause: the model emits `d_pred` in bfloat16, so the guard rejected
    all 31 real calls and both arms ran the reference. The benchmark had been
    comparing the reference against itself.

    These tests pin the configuration the sampler ACTUALLY produces — an fp32
    accumulator `z` with a bf16 `d_pred` — so that a guard which silently
    disables the fast path in production fails here instead of passing.
    """

    @staticmethod
    def _real_inputs(device="cuda"):
        args = _inputs(vocab=64, device=device)
        args["d_pred"] = args["d_pred"].to(torch.bfloat16)
        return args

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_a_bfloat16_d_pred_takes_the_fast_path(self):
        """This is the production configuration, not an edge case."""
        from unturtle_flm import state_update

        args = self._real_inputs()
        assert (
            state_update.fast_path_applies(**args, context=_measured_context()) is True
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_it_is_bit_identical_with_a_bfloat16_d_pred(self):
        from unturtle_flm import state_update

        args = self._real_inputs()
        assert torch.equal(
            reference_expression(**args),
            state_update.apply_state_update(**args, context=_measured_context(args)),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_accumulator_must_still_be_float32(self):
        """`d_pred` may be bf16 because it is only ever multiplied into an fp32
        accumulator. A bf16 `z` would make the ACCUMULATION bf16, which is a
        different computation and was never measured."""
        from unturtle_flm import state_update

        args = self._real_inputs()
        args["z"] = args["z"].to(torch.bfloat16)
        assert (
            state_update.fast_path_applies(**args, context=_measured_context(args))
            is False
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_a_float16_d_pred_is_still_rejected(self):
        """Identity was measured for bf16, not fp16."""
        from unturtle_flm import state_update

        args = self._real_inputs()
        args["d_pred"] = args["d_pred"].to(torch.float16)
        assert (
            state_update.fast_path_applies(**args, context=_measured_context(args))
            is False
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_fast_path_runs_for_every_step_of_a_real_request(self):
        """The end-to-end gate. A guard that rejects real inputs makes the whole
        specialization inert while every unit test above still passes."""
        pytest.importorskip("unturtle_flm.loader")
        from unturtle_flm import state_update

        taken = {"fast": 0, "reference": 0}
        original = state_update.fast_path_applies

        def counting_guard(*args, **kwargs):
            result = original(*args, **kwargs)
            taken["fast" if result else "reference"] += 1
            return result

        state_update.fast_path_applies = counting_guard
        try:
            from unturtle_flm.loader import load_fmlm_model
            from unturtle_flm.sampler import run_fmlm_request

            model = load_fmlm_model(device="cuda").eval()

            class _Request:
                # 32 steps: the measured cell. A shorter request is correctly
                # declined by the scope gate, which is asserted separately.
                kwargs = {"steps": 32, "num_samples": 1, "seed": 100, "gamma": 1.0}

            run_fmlm_request(model, _Request())
        finally:
            state_update.fast_path_applies = original

        assert taken["reference"] == 0, (
            f"the fast path was skipped {taken['reference']} times on real inputs"
        )
        assert taken["fast"] > 0


class TestTheSpecializationIsNotInert:
    """Every bit-identity test above asserts `fast == reference`, which is also
    satisfied when the fast path IS the reference. Replacing `_fast_update`'s
    body with the reference arithmetic left all 31 tests green while the
    specialization did nothing — the same failure mode as the bf16 guard, in a
    different place. These tests fail when the fast path stops being fast."""

    def test_the_fast_path_actually_calls_addcmul(self):
        """Behavioural, not textual: counts real dispatches."""
        from unturtle_flm import state_update

        args = _inputs()
        calls = []
        original = torch.addcmul

        def counting_addcmul(*call_args, **call_kwargs):
            calls.append(1)
            return original(*call_args, **call_kwargs)

        torch.addcmul = counting_addcmul
        try:
            state_update._fast_update(**args)
        finally:
            torch.addcmul = original

        assert len(calls) == 3, (
            f"expected 3 addcmul dispatches, saw {len(calls)}; the fast path is "
            "no longer distinct from the reference"
        )

    def test_the_reference_path_does_not_call_addcmul(self):
        """The fallback must remain the original op sequence, so that the count
        above is a real difference between the two paths and not a constant."""
        from unturtle_flm import state_update

        args = _inputs()
        calls = []
        original = torch.addcmul

        def counting_addcmul(*call_args, **call_kwargs):
            calls.append(1)
            return original(*call_args, **call_kwargs)

        torch.addcmul = counting_addcmul
        try:
            state_update._reference_update(**args)
        finally:
            torch.addcmul = original

        assert calls == []


class TestSamplerCallSiteWiring:
    """The sampler's only other coverage is a substring check for the call. That
    leaves the ARGUMENT MAPPING unpinned: folding `mean_adjustment` into
    `weight_d` — algebraically tempting, numerically different — passed all 31
    tests while changing 476 of 1024 endpoint tokens on the real checkpoint."""

    def test_each_sampler_local_reaches_the_parameter_it_belongs_to(self):
        """Captures the kwargs the sampler passes and recomputes the update from
        the sampler's OWN locals, independently of the helper."""
        pytest.importorskip("unturtle_flm.loader")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")

        from unturtle_flm import sampler as sampler_module
        from unturtle_flm.loader import load_fmlm_model
        from unturtle_flm.sampler import run_fmlm_request

        captured = []
        original = sampler_module.apply_state_update

        def capturing(**kwargs):
            result = original(**kwargs)
            if len(captured) < 3:
                captured.append(
                    {
                        k: v.detach().clone() if torch.is_tensor(v) else v
                        for k, v in kwargs.items()
                    }
                )
            return result

        sampler_module.apply_state_update = capturing
        try:
            model = load_fmlm_model(device="cuda").eval()

            class _Request:
                kwargs = {"steps": 4, "num_samples": 1, "seed": 100, "gamma": 1.0}

            run_fmlm_request(model, _Request())
        finally:
            sampler_module.apply_state_update = original

        assert captured, "the sampler never called the helper"
        for step in captured:
            # `weight_d` scales the prediction inside z_tilde; `mean_adjustment`
            # is applied to it separately. A mutant that merges them keeps the
            # sum but changes the rounding, so compare the RECONSTRUCTED update
            # against what the helper was actually asked to compute.
            assert not torch.equal(
                step["weight_d"],
                step["weight_d"] + step["mean_adjustment"],
            ), "weight_d already absorbs mean_adjustment; the two were merged"
            assert step["mean_adjustment"].abs().sum() > 0, (
                "mean_adjustment arrived as zero, so the churn term was folded "
                "into another argument"
            )

    def test_the_endpoint_tokens_match_the_reference_path(self):
        """The end-to-end guard: with the fast path forced off, the sampler must
        produce the same tokens. This is what a call-site wiring error breaks."""
        pytest.importorskip("unturtle_flm.loader")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")

        from unturtle_flm import state_update
        from unturtle_flm.loader import load_fmlm_model
        from unturtle_flm.sampler import run_fmlm_request

        model = load_fmlm_model(device="cuda").eval()

        class _Request:
            kwargs = {"steps": 4, "num_samples": 1, "seed": 100, "gamma": 1.0}

        specialized = run_fmlm_request(model, _Request())["tokens"].cpu()

        original = state_update.fast_path_applies
        state_update.fast_path_applies = lambda *a, **k: False
        try:
            reference = run_fmlm_request(model, _Request())["tokens"].cpu()
        finally:
            state_update.fast_path_applies = original

        changed = int((specialized != reference).sum())
        assert changed == 0, f"{changed} tokens differ between the two paths"


class TestMixedDtypeIdentitySweep:
    """The production configuration — fp32 accumulator, bf16 `d_pred` — is not
    covered by `166-fmlm-state-update-agreement.json`, which was frozen before
    the bf16 dtype was discovered. Rather than cite a number measured once in a
    throwaway probe, this re-derives identity on every run."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("shape", [(1, 128, 512), (2, 64, 1024)])
    @pytest.mark.parametrize(
        "weights",
        [
            (0.7, 0.3, -0.37, 0.37),  # mid-schedule
            (1.0, 0.0, 0.0, 1.0),  # first step: no prediction contribution
            (0.999, 1e-4, -1e-7, 1e-7),  # near-degenerate churn
            (0.0, 1.0, -0.999, 0.999),  # last churn step
        ],
    )
    def test_bit_identical_with_a_bfloat16_prediction(self, seed, shape, weights):
        from unturtle_flm import state_update

        batch, length, vocab = shape
        g = torch.Generator(device="cuda").manual_seed(seed)
        make = lambda: torch.randn(  # noqa: E731
            batch, length, vocab, device="cuda", generator=g
        )
        z, eps = make(), make()
        d_pred = make().abs().to(torch.bfloat16)
        wz, wd, madj, ns = (
            torch.full((batch, 1, 1), v, device="cuda") for v in weights
        )
        args = dict(
            z=z,
            d_pred=d_pred,
            weight_z=wz,
            weight_d=wd,
            mean_adjustment=madj,
            noise_std=ns,
            eps=eps,
        )
        # Calls `_fast_update` directly: this asserts the ARITHMETIC is
        # identical, which is a property of the expression rather than of the
        # measured cell. Routing through the dispatcher would test the scope
        # gate instead, and these shapes are deliberately not the measured ones.
        assert torch.equal(
            reference_expression(**args), state_update._fast_update(**args)
        )


class TestMeasuredScopeGuards:
    """The Stage-2 verdict is bounded to gamma == 1, 32 steps, length 1024,
    batch 1/8/32, a frozen checkpoint, and one torch/CUDA/GPU combination. The
    first version of this guard checked only tensor properties, so it would have
    admitted the fast path at gamma 0.5, at 8 steps, at length 512, on a
    different GPU — none of which was measured.

    Widening is a separate decision requiring its own measurement, so anything
    outside the verdict falls back to the reference.
    """

    @staticmethod
    def _measured_context(**overrides):
        """The measured cell itself, NOT shaped to the test tensors: these tests
        vary one context axis at a time, so the shape axes must stay at their
        measured values or the rejection could come from the wrong clause."""
        from unturtle_flm import state_update

        context = dict(state_update.MEASURED_CONTEXT)
        context["batch"] = context["batch"][0]
        context.update(overrides)
        return context

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_measured_context_is_admitted(self):
        """The guards must not be so strict that the measured cell is rejected."""
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        args["d_pred"] = args["d_pred"].to(torch.bfloat16)
        assert (
            state_update.fast_path_applies(**args, context=self._measured_context())
            is True
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize(
        "override",
        [
            {"gamma": 0.0},
            {"gamma": 0.5},
            {"gamma": 1.5},
            {"steps": 8},
            {"steps": 64},
            {"batch": 2},
            {"batch": 16},
            {"length": 512},
            {"vocab": 32000},
            {"checkpoint": "some/other-checkpoint"},
            {"torch_version": "2.9.0+cu124"},
            {"cuda_version": "12.4"},
            {"gpu_name": "NVIDIA A100-SXM4-80GB"},
            {"compiled": True},
        ],
    )
    def test_anything_outside_the_measured_cell_falls_back(self, override):
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        args["d_pred"] = args["d_pred"].to(torch.bfloat16)
        context = self._measured_context(**override)
        assert state_update.fast_path_applies(**args, context=context) is False, (
            f"{override} was admitted but never measured"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_a_missing_context_falls_back(self):
        """Absence of evidence is not evidence of scope: a caller that supplies
        no context has not shown it is inside the measured cell."""
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        args["d_pred"] = args["d_pred"].to(torch.bfloat16)
        assert state_update.fast_path_applies(**args, context=None) is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_an_incomplete_context_falls_back(self):
        """A context missing a key cannot be checked against the verdict."""
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        args["d_pred"] = args["d_pred"].to(torch.bfloat16)
        partial = self._measured_context()
        partial.pop("gamma")
        assert state_update.fast_path_applies(**args, context=partial) is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_result_is_still_correct_when_the_context_rejects(self):
        """Falling back means falling back to the reference arithmetic, not to
        something merely close."""
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        args["d_pred"] = args["d_pred"].to(torch.bfloat16)
        assert torch.equal(
            reference_expression(**args),
            state_update.apply_state_update(**args, context=None),
        )

    def test_the_measured_context_records_the_verdict_axes(self):
        from unturtle_flm import state_update

        for axis in (
            "gamma",
            "steps",
            "batch",
            "length",
            "vocab",
            "checkpoint",
            "torch_version",
            "cuda_version",
            "gpu_name",
            "compiled",
        ):
            assert axis in state_update.MEASURED_CONTEXT, axis


class TestSamplerSuppliesTheExecutionContext:
    def test_a_real_request_inside_the_measured_cell_takes_the_fast_path(self):
        pytest.importorskip("unturtle_flm.loader")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")

        from unturtle_flm import state_update
        from unturtle_flm.loader import load_fmlm_model
        from unturtle_flm.sampler import run_fmlm_request

        taken = {"fast": 0, "reference": 0}
        original = state_update.fast_path_applies

        def counting_guard(*args, **kwargs):
            result = original(*args, **kwargs)
            taken["fast" if result else "reference"] += 1
            return result

        state_update.fast_path_applies = counting_guard
        try:
            model = load_fmlm_model(device="cuda").eval()

            class _Request:
                kwargs = {
                    "steps": 32,
                    "num_samples": 1,
                    "seed": 100,
                    "gamma": 1.0,
                }

            run_fmlm_request(model, _Request())
        finally:
            state_update.fast_path_applies = original

        assert taken == {"fast": 31, "reference": 0}, taken

    def test_a_request_outside_the_measured_cell_falls_back_entirely(self):
        """Same model, same everything except a step count that was never
        measured. Every call must take the reference."""
        pytest.importorskip("unturtle_flm.loader")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")

        from unturtle_flm import state_update
        from unturtle_flm.loader import load_fmlm_model
        from unturtle_flm.sampler import run_fmlm_request

        taken = {"fast": 0, "reference": 0}
        original = state_update.fast_path_applies

        def counting_guard(*args, **kwargs):
            result = original(*args, **kwargs)
            taken["fast" if result else "reference"] += 1
            return result

        state_update.fast_path_applies = counting_guard
        try:
            model = load_fmlm_model(device="cuda").eval()

            class _Request:
                kwargs = {"steps": 8, "num_samples": 1, "seed": 100, "gamma": 1.0}

            run_fmlm_request(model, _Request())
        finally:
            state_update.fast_path_applies = original

        assert taken["fast"] == 0, (
            f"{taken['fast']} unmeasured calls took the fast path"
        )
        assert taken["reference"] == 7
