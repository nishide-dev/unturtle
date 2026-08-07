"""
Tests for runner-based generation dispatch (#69 PR B).

PR A made algorithm *selection* explicit.  The execution path still went
name -> flags -> booleans -> branch inside `_sample`, so a family with no
boolean to be encoded as had nowhere to go.  This pins the contract as each
algorithm gains a runner and `dispatch_generation` calls it directly.
"""

import pytest


class _Recorder:
    """Records which sampling loop was entered."""

    def __init__(self, supports=("mdlm",)):
        self.calls = []
        self._supports = supports

    # --- capability probes the registry reads ---
    def _sample(self, *args, **kwargs):
        self.calls.append("mdlm")
        return "mdlm-result"

    def _sample_with_cache(self, *args, **kwargs):
        self.calls.append("block_decode")
        return "block-decode-result"

    def _model_forward_with_cache(self, *args, **kwargs):
        return None

    def _sample_block_diffusion(self, *args, **kwargs):
        self.calls.append("bd3lm")
        return "bd3lm-result"


class _MdlmOnly:
    def __init__(self):
        self.calls = []

    def _sample(self, *args, **kwargs):
        self.calls.append("mdlm")
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
    def test_kwargs_reach_the_runner(self):
        from unturtle.models.generation.sampler import dispatch_generation

        seen = {}

        class _Model:
            def _sample(self, *args, **kwargs):
                seen.update(kwargs)
                return None

        dispatch_generation(
            _Model(), _request(steps=7, temperature=0.3), algorithm="mdlm"
        )

        assert seen.get("steps") == 7
        assert seen.get("temperature") == 0.3

    def test_inputs_and_config_reach_the_runner(self):
        from unturtle.models.generation.sampler import (
            GenerationRequest,
            dispatch_generation,
        )

        seen = {}

        class _Model:
            def _sample(self, inputs=None, generation_config=None, **kwargs):
                seen["inputs"] = inputs
                seen["config"] = generation_config
                return None

        sentinel_inputs = object()
        sentinel_config = object()
        dispatch_generation(
            _Model(),
            GenerationRequest(
                inputs=sentinel_inputs,
                generation_config=sentinel_config,
                kwargs={},
            ),
            algorithm="mdlm",
        )

        assert seen["inputs"] is sentinel_inputs
        assert seen["config"] is sentinel_config


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
