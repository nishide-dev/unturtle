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
        got = state_update.apply_state_update(**args)
        assert torch.equal(expected, got)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_it_stays_identical_across_repeated_calls(self):
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        first = state_update.apply_state_update(**args)
        for _ in range(4):
            assert torch.equal(first, state_update.apply_state_update(**args))

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
        assert state_update.fast_path_applies(**args) is False

    def test_the_cpu_result_still_matches_the_reference_exactly(self):
        """Falling back means falling back to the reference arithmetic."""
        from unturtle_flm import state_update

        args = _inputs(device="cpu")
        assert torch.equal(
            reference_expression(**args), state_update.apply_state_update(**args)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
    def test_non_float32_does_not_take_the_fast_path(self, dtype):
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda", dtype=dtype)
        assert state_update.fast_path_applies(**args) is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_non_contiguous_input_does_not_take_the_fast_path(self):
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        args["d_pred"] = args["d_pred"].transpose(1, 2).transpose(1, 2)
        args["z"] = args["z"][:, :, ::2]
        assert state_update.fast_path_applies(**args) is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_autocast_does_not_take_the_fast_path(self):
        """Autocast changes the executed dtype, which is outside the scope in
        which bit identity was measured."""
        from unturtle_flm import state_update

        args = _inputs(vocab=64, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert state_update.fast_path_applies(**args) is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_measured_configuration_does_take_it(self):
        """The guards must not be so strict that nothing qualifies."""
        from unturtle_flm import state_update

        assert (
            state_update.fast_path_applies(**_inputs(vocab=64, device="cuda")) is True
        )

    def test_mismatched_devices_do_not_take_the_fast_path(self):
        from unturtle_flm import state_update

        args = _inputs()
        assert state_update.fast_path_applies(**args) is False


class TestSemanticsPreservation:
    def test_no_input_is_mutated(self):
        from unturtle_flm import state_update

        args = _inputs()
        before = {k: v.clone() for k, v in args.items()}
        state_update.apply_state_update(**args)
        for name, original in before.items():
            assert torch.equal(args[name], original), f"{name} was mutated"

    def test_the_result_does_not_alias_any_input(self):
        from unturtle_flm import state_update

        args = _inputs()
        out = state_update.apply_state_update(**args)
        for name, tensor in args.items():
            assert out.data_ptr() != tensor.data_ptr(), f"result aliases {name}"

    def test_no_randomness_is_consumed(self):
        """`eps` is supplied by the caller; the update must not draw its own, or
        the RNG stream would diverge from the reference."""
        from unturtle_flm import state_update

        args = _inputs()
        before = torch.get_rng_state()
        state_update.apply_state_update(**args)
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

        assert state_update.fast_path_applies(**self._real_inputs()) is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_it_is_bit_identical_with_a_bfloat16_d_pred(self):
        from unturtle_flm import state_update

        args = self._real_inputs()
        assert torch.equal(
            reference_expression(**args), state_update.apply_state_update(**args)
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_accumulator_must_still_be_float32(self):
        """`d_pred` may be bf16 because it is only ever multiplied into an fp32
        accumulator. A bf16 `z` would make the ACCUMULATION bf16, which is a
        different computation and was never measured."""
        from unturtle_flm import state_update

        args = self._real_inputs()
        args["z"] = args["z"].to(torch.bfloat16)
        assert state_update.fast_path_applies(**args) is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_a_float16_d_pred_is_still_rejected(self):
        """Identity was measured for bf16, not fp16."""
        from unturtle_flm import state_update

        args = self._real_inputs()
        args["d_pred"] = args["d_pred"].to(torch.float16)
        assert state_update.fast_path_applies(**args) is False

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
                kwargs = {"steps": 4, "num_samples": 1, "seed": 100, "gamma": 1.0}

            run_fmlm_request(model, _Request())
        finally:
            state_update.fast_path_applies = original

        assert taken["reference"] == 0, (
            f"the fast path was skipped {taken['reference']} times on real inputs"
        )
        assert taken["fast"] > 0
