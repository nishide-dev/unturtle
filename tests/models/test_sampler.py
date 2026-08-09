from __future__ import annotations

import pytest

from unturtle.models.generation.sampler import (
    ALL_ALGORITHMS,
    CANVAS_ALGORITHMS,
    DISCRETE_ALGORITHMS,
    algorithm_to_flags,
    resolve_algorithm,
)


class _CacheCapable:
    """Stub model that supports block-decode (implements _model_forward_with_cache)."""

    def _sample(self, *a, **k):  # noqa: ANN002, ANN003
        ...

    def _model_forward_with_cache(self, *a, **k):  # noqa: ANN002, ANN003
        ...


class _NoCache:
    """Stub model without block-decode capability."""

    def _sample(self, *a, **k):  # noqa: ANN002, ANN003
        ...


class _BlockCapable:
    """Stub exposing the block-decode cache hook."""

    def _sample(self, *a, **k):  # noqa: ANN002, ANN003
        ...

    def _model_forward_with_cache(self, *a, **k):  # noqa: ANN002, ANN003
        ...


class _PlainDiffusion:
    """Stub without the cache hook."""

    def _sample(self, *a, **k):  # noqa: ANN002, ANN003
        ...


class _CacheCapableOptOut:
    """Stub with the cache hook but supports_block_decode = False (encoder-style opt-out)."""

    supports_block_decode = False

    def _sample(self, *a, **k):  # noqa: ANN002, ANN003
        ...

    def _model_forward_with_cache(self, *a, **k):  # noqa: ANN002, ANN003
        ...


class _BD3LMCapable:
    """Stub with both block-decode and BD3LM capability."""

    def _sample(self, *a, **k):  # noqa: ANN002, ANN003
        ...

    def _model_forward_with_cache(self, *a, **k):  # noqa: ANN002, ANN003
        ...

    def _sample_block_diffusion(self, *a, **k):  # noqa: ANN002, ANN003
        ...


class _BD3LMOnly:
    """Stub with BD3LM but no block-decode cache hook."""

    def _sample(self, *a, **k):  # noqa: ANN002, ANN003
        ...

    def _sample_block_diffusion(self, *a, **k):  # noqa: ANN002, ANN003
        ...


def test_known_algorithms_present() -> None:
    assert set(DISCRETE_ALGORITHMS) == {"mdlm", "block_decode", "bd3lm"}
    assert set(CANVAS_ALGORITHMS) == {"block_ar"}
    # The continuous_flow family (#66) is in neither historical table — those
    # tables exist for masked/canvas callers — but it is registered.
    assert set(ALL_ALGORITHMS) == set(DISCRETE_ALGORITHMS) | set(CANVAS_ALGORITHMS) | {
        "flowlm",
        "ladiff",
        "dfm",
    }


def test_algorithm_to_flags_mdlm() -> None:
    assert algorithm_to_flags("mdlm") == {
        "use_cache": False,
        "use_block_diffusion": False,
    }


def test_algorithm_to_flags_block_decode() -> None:
    flags = algorithm_to_flags("block_decode")
    assert flags["use_cache"] is True
    assert flags["use_block_diffusion"] is False


def test_algorithm_to_flags_bd3lm() -> None:
    flags = algorithm_to_flags("bd3lm")
    assert flags["use_block_diffusion"] is True
    assert flags["use_cache"] is False


def test_algorithm_to_flags_unknown_raises() -> None:
    with pytest.raises(ValueError) as exc:
        algorithm_to_flags("continuous_ddpm")
    assert "continuous_ddpm" in str(exc.value)
    assert "mdlm" in str(exc.value)


def test_resolve_auto_picks_block_decode_when_capable() -> None:
    assert (
        resolve_algorithm("auto", _CacheCapable(), bd3lm_requested=False)
        == "block_decode"
    )


def test_resolve_auto_picks_bd3lm_when_requested() -> None:
    assert resolve_algorithm("auto", _BD3LMCapable(), bd3lm_requested=True) == "bd3lm"


def test_resolve_auto_falls_back_to_mdlm_without_cache_capability() -> None:
    assert resolve_algorithm("auto", _NoCache(), bd3lm_requested=False) == "mdlm"


def test_resolve_explicit_passthrough() -> None:
    assert resolve_algorithm("mdlm", _CacheCapable(), bd3lm_requested=False) == "mdlm"


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError):
        resolve_algorithm("nope", _CacheCapable(), bd3lm_requested=False)


def test_resolve_auto_block_decode_still_works() -> None:
    assert (
        resolve_algorithm("auto", _BlockCapable(), bd3lm_requested=False)
        == "block_decode"
    )


def test_resolve_auto_mdlm_fallback_still_works() -> None:
    assert resolve_algorithm("auto", _PlainDiffusion(), bd3lm_requested=False) == "mdlm"


def test_algorithm_to_flags_unchanged() -> None:
    assert algorithm_to_flags("mdlm") == {
        "use_cache": False,
        "use_block_diffusion": False,
    }
    assert algorithm_to_flags("block_decode") == {
        "use_cache": True,
        "use_block_diffusion": False,
    }
    assert algorithm_to_flags("bd3lm") == {
        "use_cache": False,
        "use_block_diffusion": True,
    }


def test_resolve_ar_is_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="Unknown decoding algorithm"):
        resolve_algorithm("ar", _PlainDiffusion(), bd3lm_requested=False)


# ---------------------------------------------------------------------------
# Defect 1 & 2: capability checks
# ---------------------------------------------------------------------------


def test_auto_opts_out_via_supports_block_decode_false() -> None:
    """auto resolves to mdlm when model has the hook but supports_block_decode=False."""
    assert (
        resolve_algorithm("auto", _CacheCapableOptOut(), bd3lm_requested=False)
        == "mdlm"
    )


def test_explicit_block_decode_non_capable_raises() -> None:
    """explicit 'block_decode' on a non-capable model raises ValueError."""
    with pytest.raises(ValueError, match="block-decode"):
        resolve_algorithm("block_decode", _NoCache(), bd3lm_requested=False)


def test_explicit_block_decode_opted_out_raises() -> None:
    """explicit 'block_decode' on an opted-out model raises ValueError."""
    with pytest.raises(ValueError, match="block-decode"):
        resolve_algorithm("block_decode", _CacheCapableOptOut(), bd3lm_requested=False)


def test_explicit_bd3lm_without_capability_raises() -> None:
    """explicit 'bd3lm' on a model without _sample_block_diffusion raises ValueError."""
    with pytest.raises(ValueError, match="BD3LM"):
        resolve_algorithm("bd3lm", _CacheCapable(), bd3lm_requested=False)


def test_auto_bd3lm_requested_without_capability_raises() -> None:
    """auto + bd3lm_requested=True on a model without _sample_block_diffusion raises ValueError."""
    with pytest.raises(ValueError, match="BD3LM"):
        resolve_algorithm("auto", _CacheCapable(), bd3lm_requested=True)


def test_auto_bd3lm_requested_with_capability_resolves() -> None:
    """auto + bd3lm_requested=True resolves to 'bd3lm' when model has _sample_block_diffusion."""
    assert resolve_algorithm("auto", _BD3LMCapable(), bd3lm_requested=True) == "bd3lm"


def test_explicit_bd3lm_with_capability_resolves() -> None:
    """explicit 'bd3lm' resolves correctly when model has _sample_block_diffusion."""
    assert resolve_algorithm("bd3lm", _BD3LMCapable(), bd3lm_requested=False) == "bd3lm"


def test_explicit_block_decode_with_capability_resolves() -> None:
    """explicit 'block_decode' resolves correctly on a capable model."""
    assert (
        resolve_algorithm("block_decode", _CacheCapable(), bd3lm_requested=False)
        == "block_decode"
    )


# ---------------------------------------------------------------------------
# block_ar algorithm (DiffusionGemma / canvas block diffusion family)
# ---------------------------------------------------------------------------


class _BlockArCapable:
    """Stub mimicking DiffusionGemmaGenerationMixin capability.

    Deliberately no _sample: the canvas family has no mask semantics.
    """

    def _denoising_step(self, *a, **k):  # noqa: ANN002, ANN003
        ...


def test_resolve_auto_prefers_block_ar() -> None:
    assert (
        resolve_algorithm("auto", _BlockArCapable(), bd3lm_requested=False)
        == "block_ar"
    )


def test_resolve_explicit_block_ar_on_masked_model_raises() -> None:
    with pytest.raises(ValueError, match="block_ar"):
        resolve_algorithm("block_ar", _PlainDiffusion(), bd3lm_requested=False)


def test_resolve_explicit_mdlm_on_block_ar_model_raises() -> None:
    # block_ar families have no mask semantics -> mdlm is inapplicable.
    with pytest.raises(ValueError, match="masked"):
        resolve_algorithm("mdlm", _BlockArCapable(), bd3lm_requested=False)


def test_algorithm_to_flags_block_ar_is_empty() -> None:
    assert algorithm_to_flags("block_ar") == {}


def test_resolve_auto_block_ar_takes_priority_over_bd3lm_requested() -> None:
    # canvas family has no _sample; auto returns block_ar
    # instead of honoring (and then failing) the bd3lm request.
    assert (
        resolve_algorithm("auto", _BlockArCapable(), bd3lm_requested=True) == "block_ar"
    )


def test_resolve_auto_no_capability_raises() -> None:
    class _NoAlgorithms: ...

    # Message is now derived from the registry (#69 PR C) rather than naming
    # the three masked hooks, so it enumerates the registered algorithms.
    with pytest.raises(ValueError, match="registered decoding"):
        resolve_algorithm("auto", _NoAlgorithms(), bd3lm_requested=False)
