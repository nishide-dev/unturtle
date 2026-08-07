"""
Tests for de-masking the registry core (#69 PR C).

Two things in `sampler.py` still assumed the algorithm set is the fixed masked
one:

  - the "no known decoding algorithm" message hardcoded three private hook
    names, so a model failing only a *newly registered* family's probe was told
    to go find hooks irrelevant to it;
  - the module docstring described the registry as if masked flags were
    universal.

The flags themselves stay — `use_cache` / `use_block_diffusion` are
user-visible `MaskedDiffusionGenerationConfig` fields with their own
cross-validation, so removing them is a behavior change, not a cleanup.
"""

import pytest


class _Bare:
    """No generation hooks of any kind."""


class _Masked:
    def _sample(self, input_ids, attention_mask, generation_config):
        return None


class TestNoAlgorithmMessageComesFromTheRegistry:
    def test_message_names_the_registered_algorithms(self):
        from unturtle.models.generation.sampler import resolve_algorithm

        with pytest.raises(ValueError) as excinfo:
            resolve_algorithm("auto", _Bare(), bd3lm_requested=False)

        message = str(excinfo.value)
        for name in ("mdlm", "block_decode", "block_ar"):
            assert name in message, f"{name} missing from {message!r}"

    def test_a_newly_registered_family_appears_in_the_message(self):
        """The old message hardcoded three masked hooks and could not say this."""
        from unturtle.models.generation import sampler
        from unturtle.models.generation.sampler import (
            GenerationAlgorithm,
            register_algorithm,
            resolve_algorithm,
        )

        newcomer = GenerationAlgorithm(
            name="toy-ode",
            family="continuous_flow",
            supports=lambda model: False,
            runner=lambda model, request: None,
        )
        register_algorithm(newcomer)
        try:
            with pytest.raises(ValueError) as excinfo:
                resolve_algorithm("auto", _Bare(), bd3lm_requested=False)
            assert "toy-ode" in str(excinfo.value)
        finally:
            sampler._unregister_algorithm(newcomer)

    def test_message_does_not_hardcode_masked_hook_names(self):
        """A continuous model should not be told to implement `_sample`."""
        from unturtle.models.generation.sampler import resolve_algorithm

        with pytest.raises(ValueError) as excinfo:
            resolve_algorithm("auto", _Bare(), bd3lm_requested=False)

        message = str(excinfo.value)
        assert "_denoising_step" not in message
        assert "_model_forward_with_cache" not in message


class TestFlagsAreNotUniversal:
    def test_a_non_masked_family_carries_no_masked_flags(self):
        from unturtle.models.generation.sampler import (
            algorithm_to_flags,
            find_algorithm,
        )

        # block_ar is the shipped non-masked family.
        assert algorithm_to_flags("block_ar") == {}
        assert find_algorithm("block_ar").family == "canvas"

    def test_the_core_never_reads_a_flag_by_name(self):
        """Selection and dispatch must not mention masked flag names.

        The flags remain as *registration data* for masked algorithms, and as
        user-visible config fields.  What PR C removes is the core treating
        them as concepts every family must have.
        """
        import ast
        import inspect
        import textwrap

        from unturtle.models.generation import sampler

        for function in (
            sampler.resolve_algorithm,
            sampler.dispatch_generation,
            sampler.find_algorithm,
            sampler.register_algorithm,
        ):
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            # Strings only, so a docstring *explaining* the flags does not
            # count as the core reading them.
            names = (
                {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
                | {
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                }
                | {
                    kw.arg
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    for kw in node.keywords
                    if kw.arg
                }
            )
            for flag in ("use_cache", "use_block_diffusion"):
                assert flag not in names, f"{function.__name__} reads {flag}"


class TestEveryFamilyIsDispatchable:
    def test_a_continuous_family_needs_no_masked_concept_at_all(self):
        """End-to-end: register, select, and run without masked hooks or flags."""
        from unturtle.models.generation import sampler
        from unturtle.models.generation.sampler import (
            GenerationAlgorithm,
            GenerationRequest,
            algorithm_to_flags,
            dispatch_generation,
            register_algorithm,
            resolve_algorithm,
        )

        class _Latent:
            def _integrate_ode(self, request):
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
            assert (
                resolve_algorithm("toy-ode", model, bd3lm_requested=False) == "toy-ode"
            )
            assert algorithm_to_flags("toy-ode") == {}
            assert (
                dispatch_generation(model, GenerationRequest(), algorithm="toy-ode")
                == "ode-result"
            )
            # And it still loses `auto` to the masked algorithms.
            assert resolve_algorithm("auto", _Masked(), bd3lm_requested=False) == "mdlm"
        finally:
            sampler._unregister_algorithm(newcomer)
