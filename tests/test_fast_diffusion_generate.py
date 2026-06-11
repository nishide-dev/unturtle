from __future__ import annotations

import pytest
import torch

from unturtle.fast_diffusion_model import FastDiffusionModel


class _RecordingModel:
    """Stub dLLM model: records the args its generate() receives."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, inputs=None, *, algorithm="auto", **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"inputs": inputs, "algorithm": algorithm, **kwargs})
        return "GENERATED"


def test_facade_forwards_inputs_and_algorithm() -> None:
    model = _RecordingModel()
    out = FastDiffusionModel.generate(model, inputs="X", algorithm="mdlm", steps=8)
    assert out == "GENERATED"
    call = model.calls[-1]
    assert call["inputs"] == "X"
    assert call["algorithm"] == "mdlm"
    assert call["steps"] == 8


def test_facade_default_algorithm_is_auto() -> None:
    model = _RecordingModel()
    FastDiffusionModel.generate(model, inputs="X")
    assert model.calls[-1]["algorithm"] == "auto"


def test_facade_passes_through_gen_kwargs() -> None:
    model = _RecordingModel()
    FastDiffusionModel.generate(
        model, inputs="X", algorithm="block_decode", temperature=0.7, max_new_tokens=32
    )
    call = model.calls[-1]
    assert call["temperature"] == 0.7
    assert call["max_new_tokens"] == 32


def test_facade_requires_generate() -> None:
    class _NotADLLM:
        pass

    with pytest.raises(TypeError, match="generate"):
        FastDiffusionModel.generate(_NotADLLM(), inputs="X")


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


@pytest.mark.parametrize("algorithm", ["mdlm", "block_decode"])
def test_facade_parity_with_direct_generate(algorithm) -> None:
    """Facade output equals calling model.generate directly with the same algorithm."""
    model = _tiny_a2d_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    gen_kwargs = dict(
        steps=4, max_new_tokens=4, temperature=0.0, mask_token_id=999, block_length=4
    )

    torch.manual_seed(0)
    out_direct = model.generate(inputs=prompt, algorithm=algorithm, **gen_kwargs)

    torch.manual_seed(0)
    out_facade = FastDiffusionModel.generate(
        model, inputs=prompt, algorithm=algorithm, **gen_kwargs
    )

    seq_direct = (
        out_direct.sequences if hasattr(out_direct, "sequences") else out_direct
    )
    seq_facade = (
        out_facade.sequences if hasattr(out_facade, "sequences") else out_facade
    )
    assert torch.equal(seq_direct, seq_facade)
