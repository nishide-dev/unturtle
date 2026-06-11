from __future__ import annotations

import pytest

from unturtle.models.generation.sampler import (
    DISCRETE_ALGORITHMS,
    algorithm_to_flags,
    resolve_algorithm,
)


class _CacheCapable:
    """Stub model that supports block-decode (implements _model_forward_with_cache)."""

    def _model_forward_with_cache(self, *a, **k):  # noqa: ANN002, ANN003
        ...


class _NoCache:
    """Stub model without block-decode capability."""


class _BlockCapable:
    """Stub exposing the block-decode cache hook."""

    def _model_forward_with_cache(self, *a, **k):  # noqa: ANN002, ANN003
        ...


class _PlainDiffusion:
    """Stub without the cache hook."""


def test_known_algorithms_present() -> None:
    assert set(DISCRETE_ALGORITHMS) == {"mdlm", "block_decode", "bd3lm"}


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
    assert resolve_algorithm("auto", _CacheCapable(), bd3lm_requested=True) == "bd3lm"


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
