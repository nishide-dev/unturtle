"""
Tests for runner-based generation dispatch (#69 PR B).

PR A made algorithm *selection* explicit.  The execution path still went
name -> flags -> booleans -> branch inside `_sample`, so a family with no
boolean to be encoded as had nowhere to go.  This pins the contract as each
algorithm gains a runner and `dispatch_generation` calls it directly.
"""

import pytest

# The doubles below mirror the REAL sampling-loop signatures on purpose.
# `*args, **kwargs` doubles accept any call shape, so they prove which method
# was reached and nothing about whether the call is well-formed — that is
# exactly how runners that TypeError on every real model passed review.
#
#   _sample(input_ids, attention_mask, generation_config)          [positional]
#   _sample_with_cache(input_ids, attention_mask, generation_config)
#   _sample_block_diffusion(input_ids, generation_config, attention_mask=None)


class _Recorder:
    """Records which sampling loop was entered, with real signatures."""

    def __init__(self):
        self.calls = []
        self.seen = {}

    def _sample(self, input_ids, attention_mask, generation_config):
        self.calls.append("mdlm")
        self.seen = {
            "inputs": input_ids,
            "config": generation_config,
            "attention_mask": attention_mask,
        }
        return "mdlm-result"

    def _sample_with_cache(self, input_ids, attention_mask, generation_config):
        self.calls.append("block_decode")
        self.seen = {
            "inputs": input_ids,
            "config": generation_config,
            "attention_mask": attention_mask,
        }
        return "block-decode-result"

    def _model_forward_with_cache(self, *args, **kwargs):
        return None

    def _sample_block_diffusion(
        self, input_ids, generation_config, attention_mask=None
    ):
        self.calls.append("bd3lm")
        self.seen = {
            "inputs": input_ids,
            "config": generation_config,
            "attention_mask": attention_mask,
        }
        return "bd3lm-result"


class _DreamLike:
    """Dream's `_sample` takes two extra required hook callables."""

    def __init__(self):
        self.calls = []

    def _sample(
        self,
        input_ids,
        attention_mask,
        generation_config,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ):
        self.calls.append("mdlm")
        return "dream-result"


class _MdlmOnly:
    def __init__(self):
        self.calls = []
        self.seen = {}

    def _sample(self, input_ids, attention_mask, generation_config):
        self.calls.append("mdlm")
        self.seen = {
            "inputs": input_ids,
            "config": generation_config,
            "attention_mask": attention_mask,
        }
        return "mdlm-result"


class _Canvas:
    def __init__(self):
        self.calls = []

    def _denoising_step(self, *args, **kwargs):
        return None

    def generate(self, *args, **kwargs):
        self.calls.append("upstream-generate")
        return "canvas-result"


def _request(**kwargs):
    from unturtle.models.generation.sampler import GenerationRequest

    return GenerationRequest(inputs=None, generation_config=None, kwargs=dict(kwargs))


class TestRunnersAreRegistered:
    @pytest.mark.parametrize("name", ["mdlm", "block_decode", "bd3lm", "block_ar"])
    def test_every_algorithm_has_a_runner(self, name):
        from unturtle.models.generation.sampler import find_algorithm

        algorithm = find_algorithm(name)
        assert algorithm is not None
        assert callable(algorithm.runner), f"{name} has no runner"


class TestDispatchCallsTheRightLoop:
    @pytest.mark.parametrize(
        ("algorithm", "expected"),
        [
            ("mdlm", "mdlm"),
            ("block_decode", "block_decode"),
            ("bd3lm", "bd3lm"),
        ],
    )
    def test_explicit_algorithm_reaches_its_own_loop(self, algorithm, expected):
        from unturtle.models.generation.sampler import dispatch_generation

        model = _Recorder()
        dispatch_generation(model, _request(), algorithm=algorithm)

        assert model.calls == [expected], (
            f"{algorithm} entered {model.calls}, not {expected}"
        )

    def test_auto_reaches_the_selected_loop(self):
        from unturtle.models.generation.sampler import dispatch_generation

        model = _Recorder()
        dispatch_generation(model, _request(), algorithm="auto")

        # block_decode outranks mdlm and the model supports it.
        assert model.calls == ["block_decode"]

    def test_mdlm_only_model_falls_to_mdlm(self):
        from unturtle.models.generation.sampler import dispatch_generation

        model = _MdlmOnly()
        dispatch_generation(model, _request(), algorithm="auto")

        assert model.calls == ["mdlm"]

    def test_block_ar_delegates_to_upstream_generate(self):
        """DiffusionGemma's loop is upstream's; Unturtle only selects it."""
        from unturtle.models.generation.sampler import dispatch_generation

        model = _Canvas()
        result = dispatch_generation(model, _request(), algorithm="block_ar")

        assert model.calls == ["upstream-generate"]
        assert result == "canvas-result"

    def test_block_ar_does_not_recurse_through_unturtle_generate(self):
        """#69 calls this out: the canvas runner must not re-enter dispatch."""
        from unturtle.models.generation.sampler import dispatch_generation

        depth = {"n": 0}

        class _Reentrant(_Canvas):
            def generate(self, *args, **kwargs):
                depth["n"] += 1
                if depth["n"] > 3:
                    raise AssertionError("block_ar recursed through generate")
                self.calls.append("upstream-generate")
                return "canvas-result"

        dispatch_generation(_Reentrant(), _request(), algorithm="block_ar")
        assert depth["n"] == 1


class TestUnsupportedStillRaisesBeforeExecuting:
    def test_explicit_unsupported_algorithm_never_runs_a_loop(self):
        from unturtle.models.generation.sampler import dispatch_generation

        model = _MdlmOnly()
        with pytest.raises(ValueError, match="block-decode"):
            dispatch_generation(model, _request(), algorithm="block_decode")

        assert model.calls == [], "a loop ran despite the capability check failing"

    def test_registered_algorithm_without_a_runner_raises(self):
        """Selection and execution are separate; a name alone cannot run."""
        from unturtle.models.generation import sampler
        from unturtle.models.generation.sampler import (
            GenerationAlgorithm,
            dispatch_generation,
            register_algorithm,
        )

        # Registered and supported, but no runner — a half-finished family.
        newcomer = GenerationAlgorithm(
            name="toy-runnerless",
            family="continuous_flow",
            supports=lambda model: True,
        )
        register_algorithm(newcomer)
        try:
            with pytest.raises(ValueError, match="no runner"):
                dispatch_generation(_MdlmOnly(), _request(), algorithm="toy-runnerless")
        finally:
            sampler._unregister_algorithm(newcomer)

    def test_unknown_algorithm_raises(self):
        from unturtle.models.generation.sampler import dispatch_generation

        model = _MdlmOnly()
        with pytest.raises(ValueError, match="Unknown decoding algorithm"):
            dispatch_generation(model, _request(), algorithm="nope")

        assert model.calls == []


class TestRequestCarriesTheCall:
    @pytest.mark.parametrize("algorithm", ["mdlm", "block_decode", "bd3lm"])
    def test_inputs_config_and_named_kwargs_reach_every_runner(self, algorithm):
        """Every runner, not just mdlm — plumbing was untested for three of four."""
        from unturtle.models.generation.sampler import (
            GenerationRequest,
            dispatch_generation,
        )

        model = _Recorder()
        sentinel_inputs = object()
        sentinel_config = object()
        sentinel_mask = object()

        dispatch_generation(
            model,
            GenerationRequest(
                inputs=sentinel_inputs,
                generation_config=sentinel_config,
                kwargs={"attention_mask": sentinel_mask},
            ),
            algorithm=algorithm,
        )

        assert model.seen["inputs"] is sentinel_inputs
        assert model.seen["config"] is sentinel_config
        assert model.seen["attention_mask"] is sentinel_mask

    def test_a_required_positional_defaults_to_none_when_unset(self):
        """`attention_mask` is required-positional on the real loops."""
        from unturtle.models.generation.sampler import dispatch_generation

        model = _MdlmOnly()
        dispatch_generation(model, _request(), algorithm="mdlm")

        assert model.calls == ["mdlm"]
        assert model.seen["attention_mask"] is None

    def test_a_loop_with_extra_required_args_is_still_callable(self):
        """Dream's `_sample` needs two hook callables the caller never passes."""
        from unturtle.models.generation.sampler import dispatch_generation

        model = _DreamLike()
        result = dispatch_generation(model, _request(), algorithm="mdlm")

        assert model.calls == ["mdlm"]
        assert result == "dream-result"


class TestNonMaskedFamilyNeedsNoMaskedHooks:
    def test_a_custom_runner_executes_without_masked_private_hooks(self):
        """#69's acceptance criterion, exercised rather than asserted."""
        from unturtle.models.generation import sampler
        from unturtle.models.generation.sampler import (
            GenerationAlgorithm,
            dispatch_generation,
            register_algorithm,
        )

        class _Latent:
            def __init__(self):
                self.calls = []

            def _integrate_ode(self, request):
                self.calls.append("ode")
                return "ode-result"

        newcomer = GenerationAlgorithm(
            name="toy-ode",
            family="continuous_flow",
            supports=lambda model: hasattr(model, "_integrate_ode"),
            runner=lambda model, request: model._integrate_ode(request),
        )
        register_algorithm(newcomer)
        try:
            model = _Latent()
            result = dispatch_generation(model, _request(), algorithm="toy-ode")

            assert model.calls == ["ode"]
            assert result == "ode-result"
            # No masked hooks, and no masked flags forced on it.
            assert not hasattr(model, "_sample")
            assert newcomer.flags == {}
        finally:
            sampler._unregister_algorithm(newcomer)


class TestAgainstRealModels:
    """Doubles prove routing; only a real model proves the call is well-formed.

    Every runner passed 17/17 against `*args, **kwargs` doubles while two of
    them raised `TypeError` on every real backbone, because `attention_mask` is
    a required positional on `_sample`/`_sample_with_cache`.
    """

    @pytest.fixture
    def llada(self):
        import torch

        from unturtle.models.backbones.llada import LLaDAConfig, LLaDAModelLM

        config = LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=512,
            mlp_ratio=4,
            max_sequence_length=64,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            rope=True,
            block_type="llama",
            activation_type="silu",
            init_device="cpu",
            mask_token_id=511,
        )
        torch.manual_seed(42)
        return LLaDAModelLM(config).eval()

    def _config(self, model):
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        return MaskedDiffusionGenerationConfig(
            max_length=16,
            steps=2,
            mask_token_id=511,
        )

    @pytest.mark.parametrize("algorithm", ["mdlm", "block_decode"])
    def test_dispatch_runs_on_a_real_model(self, llada, algorithm):
        import torch

        from unturtle.models.generation.sampler import (
            GenerationRequest,
            dispatch_generation,
            find_algorithm,
        )

        if not find_algorithm(algorithm).supports(llada):
            pytest.skip(f"{algorithm} unsupported on this backbone")

        input_ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            out = dispatch_generation(
                llada,
                GenerationRequest(
                    inputs=input_ids,
                    generation_config=self._config(llada),
                    kwargs={"attention_mask": None},
                ),
                algorithm=algorithm,
            )

        tokens = out if isinstance(out, torch.Tensor) else out.sequences
        assert tokens.shape[0] == 1
        assert tokens.shape[1] == 16

    def test_dispatch_matches_the_public_generate(self, llada):
        """Same tokens through dispatch as through `model.generate`."""
        import torch

        from unturtle.models.generation.sampler import (
            GenerationRequest,
            dispatch_generation,
        )

        input_ids = torch.tensor([[1, 2, 3, 4]])

        torch.manual_seed(0)
        with torch.no_grad():
            reference = llada.generate(
                input_ids, algorithm="mdlm", max_length=16, steps=2, mask_token_id=511
            )
        torch.manual_seed(0)
        with torch.no_grad():
            through_dispatch = dispatch_generation(
                llada,
                GenerationRequest(
                    inputs=input_ids,
                    generation_config=self._config(llada),
                    kwargs={"attention_mask": None},
                ),
                algorithm="mdlm",
            )

        ref = reference if isinstance(reference, torch.Tensor) else reference.sequences
        got = (
            through_dispatch
            if isinstance(through_dispatch, torch.Tensor)
            else through_dispatch.sequences
        )
        assert torch.equal(ref, got)
