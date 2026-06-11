from __future__ import annotations

import pytest

from unturtle.models.generation.sampler import (
    DISCRETE_ALGORITHMS,
    _supports_ar,
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


class _FakeConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class _ARModel:
    """Stub whose config model_type marks it AR-capable (TinyA2D family)."""

    def __init__(self, model_type):
        self.config = _FakeConfig(model_type)


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


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("tiny-a2d-llama", True),
        ("tiny-a2d-qwen2", True),
        ("tiny-a2d-qwen3", True),
        ("llada", False),
        ("dream", False),
        ("modernbert-diffusion", False),
    ],
)
def test_supports_ar_by_model_type(model_type: str, expected: bool) -> None:
    assert _supports_ar(_ARModel(model_type)) is expected


def test_supports_ar_missing_config_is_false() -> None:
    assert _supports_ar(_PlainDiffusion()) is False


def test_resolve_ar_for_ar_capable_returns_ar() -> None:
    model = _ARModel("tiny-a2d-llama")
    assert resolve_algorithm("ar", model, bd3lm_requested=False) == "ar"


def test_resolve_ar_for_non_ar_capable_raises() -> None:
    model = _ARModel("llada")
    with pytest.raises(ValueError, match="autoregressive"):
        resolve_algorithm("ar", model, bd3lm_requested=False)


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
