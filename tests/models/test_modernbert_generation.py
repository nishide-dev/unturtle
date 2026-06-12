"""Tests for DiffusionModernBertForMaskedLM generation and algorithm resolution.

Covers:
- resolve_algorithm("auto", model) == "mdlm" (encoder backbone opts out of block-decode)
- model.generate(...) runs without error (pre-fix this crashed via past_key_values AttributeError)
- model.generate(..., algorithm="block_decode") raises ValueError
"""

from __future__ import annotations

import pytest
import torch

from unturtle.models.generation.sampler import resolve_algorithm


@pytest.fixture
def tiny_config():
    from unturtle.models.backbones.modernbert import DiffusionModernBertConfig

    return DiffusionModernBertConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        # Override default token IDs which exceed tiny vocab_size=256
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.fixture
def tiny_model(tiny_config):
    from unturtle.models.backbones.modernbert import DiffusionModernBertForMaskedLM

    torch.manual_seed(0)
    return DiffusionModernBertForMaskedLM(tiny_config).eval()


MASK_TOKEN_ID = 103  # arbitrary token within tiny vocab


def test_supports_block_decode_is_false(tiny_model) -> None:
    """DiffusionModernBertForMaskedLM must declare supports_block_decode=False."""
    assert getattr(tiny_model, "supports_block_decode", True) is False


def test_resolve_algorithm_auto_returns_mdlm(tiny_model) -> None:
    """auto on ModernBERT resolves to mdlm (encoder opts out of block-decode)."""
    result = resolve_algorithm("auto", tiny_model, bd3lm_requested=False)
    assert result == "mdlm"


def test_generate_default_runs_without_error(tiny_model) -> None:
    """model.generate() with auto algorithm runs without AttributeError.

    This is the regression test for Defect 1 — before the fix, auto resolved to
    block_decode, which crashed with:
      AttributeError: 'MaskedLMOutput' object has no attribute 'past_key_values'
    """
    B, L = 1, 8
    # Fill with mask tokens so all positions are generatable
    input_ids = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
    with torch.no_grad():
        out = tiny_model.generate(
            input_ids,
            steps=2,
            mask_token_id=MASK_TOKEN_ID,
            max_length=L + 1,
        )
    seq = out.sequences if hasattr(out, "sequences") else out
    assert seq.shape == (B, L + 1)


def test_generate_explicit_block_decode_raises(tiny_model) -> None:
    """Explicit algorithm='block_decode' on ModernBERT raises ValueError."""
    B, L = 1, 8
    input_ids = torch.full((B, L), MASK_TOKEN_ID, dtype=torch.long)
    with pytest.raises(ValueError, match="block-decode"):
        tiny_model.generate(
            input_ids,
            algorithm="block_decode",
            steps=2,
            mask_token_id=MASK_TOKEN_ID,
            max_length=L + 1,
        )
