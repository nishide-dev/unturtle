from __future__ import annotations

import pytest
import torch

from unturtle.fast_diffusion_model import FastDiffusionModel


class _RecordingModel:
    """Stub dLLM model: records the kwargs diffusion_generate receives."""

    def __init__(self, *, cache_capable: bool = True) -> None:
        self.calls: list[dict] = []
        self._cache_capable = cache_capable

    def _model_forward_with_cache(self, *a, **k):  # noqa: ANN002, ANN003
        ...

    def diffusion_generate(self, inputs=None, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"inputs": inputs, **kwargs})
        return "GENERATED"


class _NoCacheModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def diffusion_generate(self, inputs=None, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"inputs": inputs, **kwargs})
        return "GENERATED"


def test_generate_auto_selects_block_decode_for_cache_capable() -> None:
    model = _RecordingModel(cache_capable=True)
    out = FastDiffusionModel.generate(model, inputs="X", steps=8)
    assert out == "GENERATED"
    call = model.calls[-1]
    assert call["use_cache"] is True
    assert call["use_block_diffusion"] is False
    assert call["steps"] == 8
    assert call["inputs"] == "X"


def test_generate_auto_falls_back_to_mdlm_without_cache() -> None:
    model = _NoCacheModel()
    FastDiffusionModel.generate(model, inputs="X")
    call = model.calls[-1]
    assert call["use_cache"] is False
    assert call["use_block_diffusion"] is False


def test_generate_explicit_mdlm_overrides_auto() -> None:
    model = _RecordingModel(cache_capable=True)
    FastDiffusionModel.generate(model, inputs="X", algorithm="mdlm")
    call = model.calls[-1]
    assert call["use_cache"] is False
    assert call["use_block_diffusion"] is False


def test_generate_explicit_bd3lm() -> None:
    model = _RecordingModel(cache_capable=True)
    FastDiffusionModel.generate(model, inputs="X", algorithm="bd3lm")
    call = model.calls[-1]
    assert call["use_block_diffusion"] is True
    assert call["use_cache"] is False


def test_generate_auto_with_use_block_diffusion_kwarg_picks_bd3lm() -> None:
    model = _RecordingModel(cache_capable=True)
    FastDiffusionModel.generate(model, inputs="X", use_block_diffusion=True)
    call = model.calls[-1]
    assert call["use_block_diffusion"] is True
    assert call["use_cache"] is False


def test_generate_unknown_algorithm_raises() -> None:
    model = _RecordingModel()
    with pytest.raises(ValueError):
        FastDiffusionModel.generate(model, inputs="X", algorithm="continuous_ddpm")


def test_generate_requires_diffusion_generate() -> None:
    class _NotADLLM:
        pass

    with pytest.raises(TypeError):
        FastDiffusionModel.generate(_NotADLLM(), inputs="X")


def test_generate_passes_through_gen_kwargs() -> None:
    model = _RecordingModel()
    FastDiffusionModel.generate(
        model, inputs="X", algorithm="block_decode", temperature=0.7, max_new_tokens=32
    )
    call = model.calls[-1]
    assert call["temperature"] == 0.7
    assert call["max_new_tokens"] == 32


def _tiny_a2d_model():
    from unturtle.models.conversion.a2d.tiny_a2d import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    config = TinyA2DLlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        mask_token_id=999,
    )
    model = TinyA2DLlamaLMHeadModel(config)
    model.eval()
    return model


@pytest.mark.parametrize(
    "algorithm,flags",
    [
        ("mdlm", {"use_cache": False, "use_block_diffusion": False}),
        ("block_decode", {"use_cache": True, "use_block_diffusion": False}),
    ],
)
def test_generate_parity_with_diffusion_generate(algorithm, flags) -> None:
    model = _tiny_a2d_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    gen_kwargs = dict(
        steps=4, max_new_tokens=4, temperature=0.0, mask_token_id=999, block_length=4
    )

    torch.manual_seed(0)
    out_direct = model.diffusion_generate(inputs=prompt, **flags, **gen_kwargs)

    torch.manual_seed(0)
    out_generate = FastDiffusionModel.generate(
        model, inputs=prompt, algorithm=algorithm, **gen_kwargs
    )

    seq_direct = (
        out_direct.sequences if hasattr(out_direct, "sequences") else out_direct
    )
    seq_generate = (
        out_generate.sequences if hasattr(out_generate, "sequences") else out_generate
    )
    assert torch.equal(seq_direct, seq_generate)
