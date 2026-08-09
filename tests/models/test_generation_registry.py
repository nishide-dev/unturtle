"""
Tests for the generation strategy registry (#69 PR A).

`sampler.py` selected algorithms from three module-level dicts plus a chain of
`if algorithm == ...` capability checks.  This pins the selection contract as
it moves behind explicit strategy registrations — behavior-preserving, so most
of these assert the *existing* semantics.
"""

import pytest


class _Bare:
    """No generation hooks at all."""


class _Masked:
    def _sample(self, *a, **k):
        return None


class _BlockDecode(_Masked):
    def _model_forward_with_cache(self, *a, **k):
        return None


class _BlockDecodeOptedOut(_BlockDecode):
    supports_block_decode = False


class _BD3LM(_BlockDecode):
    def _sample_block_diffusion(self, *a, **k):
        return None


class _Canvas:
    def _denoising_step(self, *a, **k):
        return None


class TestRegistryContents:
    def test_every_algorithm_is_registered(self):
        from unturtle.models.generation.sampler import iter_algorithms

        names = {a.name for a in iter_algorithms()}
        assert names == {
            "mdlm",
            "block_decode",
            "bd3lm",
            "block_ar",
            "flowlm",
            "ladiff",
        }

    def test_families_are_distinguished(self):
        """`bd3lm` and `block_ar` are both 'block' but are not the same family."""
        from unturtle.models.generation.sampler import find_algorithm

        assert find_algorithm("bd3lm").family == "masked_discrete"
        assert find_algorithm("block_ar").family == "canvas"

    def test_duplicate_registration_is_rejected(self):
        from unturtle.models.generation.sampler import (
            GenerationAlgorithm,
            register_algorithm,
        )

        clash = GenerationAlgorithm(
            name="mdlm", family="masked_discrete", supports=lambda m: True
        )
        with pytest.raises(ValueError, match="mdlm"):
            register_algorithm(clash)


class TestFlagsUnchanged:
    @pytest.mark.parametrize(
        ("algorithm", "expected"),
        [
            ("mdlm", {"use_cache": False, "use_block_diffusion": False}),
            ("block_decode", {"use_cache": True, "use_block_diffusion": False}),
            ("bd3lm", {"use_cache": False, "use_block_diffusion": True}),
            # block_ar injects nothing: the upstream GenerationConfig governs.
            ("block_ar", {}),
        ],
    )
    def test_flag_sets(self, algorithm, expected):
        from unturtle.models.generation.sampler import algorithm_to_flags

        assert algorithm_to_flags(algorithm) == expected

    def test_returned_flags_are_a_copy(self):
        """Callers merge these into kwargs; mutation must not leak back."""
        from unturtle.models.generation.sampler import algorithm_to_flags

        first = algorithm_to_flags("mdlm")
        first["use_cache"] = True
        assert algorithm_to_flags("mdlm")["use_cache"] is False

    def test_unknown_algorithm_lists_the_registered_names(self):
        from unturtle.models.generation.sampler import algorithm_to_flags

        with pytest.raises(ValueError) as excinfo:
            algorithm_to_flags("nope")
        message = str(excinfo.value)
        for name in ("mdlm", "block_decode", "bd3lm", "block_ar"):
            assert name in message


class TestAutoSelection:
    """Priority: block_ar -> bd3lm (if requested) -> block_decode -> mdlm."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            (_Canvas(), "block_ar"),
            (_BlockDecode(), "block_decode"),
            (_Masked(), "mdlm"),
            # Inherits the mixin but opts out; must fall back, not crash.
            (_BlockDecodeOptedOut(), "mdlm"),
        ],
    )
    def test_auto_without_bd3lm_requested(self, model, expected):
        from unturtle.models.generation.sampler import resolve_algorithm

        assert resolve_algorithm("auto", model, bd3lm_requested=False) == expected

    def test_bd3lm_requested_beats_block_decode(self):
        from unturtle.models.generation.sampler import resolve_algorithm

        assert resolve_algorithm("auto", _BD3LM(), bd3lm_requested=True) == "bd3lm"

    def test_canvas_outranks_bd3lm_request(self):
        """block_ar is checked first, before the bd3lm request is consulted."""
        from unturtle.models.generation.sampler import resolve_algorithm

        assert resolve_algorithm("auto", _Canvas(), bd3lm_requested=True) == "block_ar"

    def test_bd3lm_requested_on_an_incapable_model_raises(self):
        """Must not silently fall back from an explicitly requested algorithm."""
        from unturtle.models.generation.sampler import resolve_algorithm

        with pytest.raises(ValueError, match="BD3LM"):
            resolve_algorithm("auto", _BlockDecode(), bd3lm_requested=True)

    def test_model_with_no_hooks_raises(self):
        from unturtle.models.generation.sampler import resolve_algorithm

        with pytest.raises(ValueError, match="registered decoding"):
            resolve_algorithm("auto", _Bare(), bd3lm_requested=False)


class TestExplicitSelection:
    @pytest.mark.parametrize(
        ("algorithm", "model"),
        [
            ("mdlm", _Masked()),
            ("block_decode", _BlockDecode()),
            ("bd3lm", _BD3LM()),
            ("block_ar", _Canvas()),
        ],
    )
    def test_supported_algorithm_resolves_to_itself(self, algorithm, model):
        from unturtle.models.generation.sampler import resolve_algorithm

        assert resolve_algorithm(algorithm, model, bd3lm_requested=False) == algorithm

    @pytest.mark.parametrize(
        ("algorithm", "model", "message_fragment"),
        [
            # Each message names the missing capability and a real alternative.
            ("block_ar", _Masked(), "block_ar"),
            ("mdlm", _Canvas(), "masked"),
            ("block_decode", _Masked(), "block-decode"),
            ("bd3lm", _BlockDecode(), "BD3LM"),
            ("block_decode", _BlockDecodeOptedOut(), "block-decode"),
        ],
    )
    def test_unsupported_algorithm_raises_before_execution(
        self, algorithm, model, message_fragment
    ):
        from unturtle.models.generation.sampler import resolve_algorithm

        with pytest.raises(ValueError, match=message_fragment):
            resolve_algorithm(algorithm, model, bd3lm_requested=False)

    def test_unknown_name_mentions_auto(self):
        from unturtle.models.generation.sampler import resolve_algorithm

        with pytest.raises(ValueError, match="auto"):
            resolve_algorithm("does-not-exist", _Masked(), bd3lm_requested=False)


class TestExtensibility:
    def test_a_registered_strategy_does_not_outrank_masked_by_default(self):
        """#69: a new algorithm must not win `auto` merely by being registered."""
        from unturtle.models.generation import sampler
        from unturtle.models.generation.sampler import (
            GenerationAlgorithm,
            register_algorithm,
            resolve_algorithm,
        )

        newcomer = GenerationAlgorithm(
            name="toy-flow",
            family="discrete_flow",
            # Claims to support everything — priority must still keep it last.
            supports=lambda model: True,
        )
        # Inserted at the FRONT: if selection ever fell back to registration
        # order, this would win, so the assertions below prove `auto_priority`
        # is what decides rather than list position.
        sampler._ALGORITHMS.insert(0, newcomer)
        try:
            assert resolve_algorithm("auto", _Masked(), bd3lm_requested=False) == "mdlm"
            assert (
                resolve_algorithm("auto", _Canvas(), bd3lm_requested=False)
                == "block_ar"
            )
            # But it is still explicitly selectable.
            assert (
                resolve_algorithm("toy-flow", _Masked(), bd3lm_requested=False)
                == "toy-flow"
            )
        finally:
            sampler._unregister_algorithm(newcomer)

        # The public path rejects it once it is genuinely registered.
        register_algorithm(newcomer)
        try:
            with pytest.raises(ValueError, match="toy-flow"):
                register_algorithm(newcomer)
        finally:
            sampler._unregister_algorithm(newcomer)

    def test_registration_order_does_not_decide_auto(self):
        """The built-ins happen to be listed in priority order; don't rely on it."""
        from unturtle.models.generation import sampler
        from unturtle.models.generation.sampler import resolve_algorithm

        original = list(sampler._ALGORITHMS)
        # mdlm first, block_ar last — the exact inverse of the intended order.
        sampler._ALGORITHMS[:] = sorted(original, key=lambda a: -a.auto_priority)
        try:
            assert resolve_algorithm("auto", _Masked(), bd3lm_requested=False) == "mdlm"
            assert (
                resolve_algorithm("auto", _BlockDecode(), bd3lm_requested=False)
                == "block_decode"
            )
            assert (
                resolve_algorithm("auto", _Canvas(), bd3lm_requested=False)
                == "block_ar"
            )
        finally:
            sampler._ALGORITHMS[:] = original

    def test_algorithm_without_a_custom_message_still_explains_itself(self):
        """The path a future family registered without a message would hit."""
        from unturtle.models.generation import sampler
        from unturtle.models.generation.sampler import (
            GenerationAlgorithm,
            register_algorithm,
            resolve_algorithm,
        )

        newcomer = GenerationAlgorithm(
            name="toy-bare",
            family="continuous",
            supports=lambda model: False,
            # No unsupported_message: exercise the default.
        )
        register_algorithm(newcomer)
        try:
            with pytest.raises(ValueError) as excinfo:
                resolve_algorithm("toy-bare", _Masked(), bd3lm_requested=False)
            message = str(excinfo.value)
            assert "toy-bare" in message
            assert "_Masked" in message
        finally:
            sampler._unregister_algorithm(newcomer)

    def test_registry_flags_cannot_be_mutated_in_place(self):
        """`frozen=True` blocks rebinding, not dict mutation — so guard it."""
        from unturtle.models.generation.sampler import find_algorithm

        entry = find_algorithm("mdlm")
        with pytest.raises((TypeError, AttributeError)):
            entry.flags["use_cache"] = True

    def test_bd3lm_is_never_chosen_automatically(self):
        """`auto` must not pick bd3lm without the explicit request.

        A BD3LM-capable model also supports block_decode, and bd3lm has the
        higher priority of the two — so only `auto_eligible=False` keeps the
        unrequested path on block_decode.
        """
        from unturtle.models.generation.sampler import resolve_algorithm

        assert (
            resolve_algorithm("auto", _BD3LM(), bd3lm_requested=False) == "block_decode"
        )

    def test_a_non_masked_strategy_needs_no_masked_hooks(self):
        """A strategy with no `_sample`/`_denoising_step` must still work."""
        from unturtle.models.generation import sampler
        from unturtle.models.generation.sampler import (
            GenerationAlgorithm,
            algorithm_to_flags,
            register_algorithm,
            resolve_algorithm,
        )

        class _Latent:
            def _integrate_ode(self):
                return None

        newcomer = GenerationAlgorithm(
            name="toy-ode",
            family="continuous",
            supports=lambda model: hasattr(model, "_integrate_ode"),
            unsupported_message=lambda model: f"{type(model).__name__} has no ODE hook",
        )
        register_algorithm(newcomer)
        try:
            model = _Latent()
            assert (
                resolve_algorithm("toy-ode", model, bd3lm_requested=False) == "toy-ode"
            )
            # No masked flags forced on a non-masked family.
            assert algorithm_to_flags("toy-ode") == {}
            with pytest.raises(ValueError, match="no ODE hook"):
                resolve_algorithm("toy-ode", _Masked(), bd3lm_requested=False)
        finally:
            sampler._unregister_algorithm(newcomer)


class TestBackwardCompatibleTables:
    """The old dicts had no external consumers, but keep them coherent."""

    def test_all_algorithms_still_maps_names_to_flags(self):
        from unturtle.models.generation.sampler import (
            ALL_ALGORITHMS,
            algorithm_to_flags,
        )

        assert set(ALL_ALGORITHMS) == {
            "mdlm",
            "block_decode",
            "bd3lm",
            "block_ar",
            "flowlm",
            "ladiff",
        }
        for name, flags in ALL_ALGORITHMS.items():
            assert flags == algorithm_to_flags(name)
