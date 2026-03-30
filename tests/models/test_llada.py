"""Tests for LLaDA models.

CPU-only tests covering config instantiation, model instantiation with
random weights, and forward pass shapes. No pretrained checkpoints required.
"""

from __future__ import annotations

import pytest
import torch


class TestLLaDAConfig:
    def test_config_default_fields(self):
        from unturtle.models.llada import LLaDAConfig
        config = LLaDAConfig()
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_layers == 12
        assert config.vocab_size == 50257

    def test_config_custom_values(self):
        from unturtle.models.llada import LLaDAConfig
        config = LLaDAConfig(d_model=128, n_heads=4, n_layers=2, vocab_size=1000)
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 2
        assert config.vocab_size == 1000

    def test_config_hf_properties(self):
        """HF-compatible properties map to internal fields."""
        from unturtle.models.llada import LLaDAConfig
        config = LLaDAConfig(d_model=256, n_heads=8, n_layers=4)
        assert config.hidden_size == 256
        assert config.num_attention_heads == 8
        assert config.num_hidden_layers == 4

    def test_config_has_mask_token_id(self):
        from unturtle.models.llada import LLaDAConfig
        config = LLaDAConfig()
        assert hasattr(config, "mask_token_id")


class TestLLaDAModel:
    @pytest.fixture
    def config(self):
        from unturtle.models.llada import LLaDAConfig
        return LLaDAConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            vocab_size=1000,
            mlp_ratio=4,
            max_sequence_length=64,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            rope=True,         # LLaDA requires RoPE for MDM
            init_device="cpu",
        )

    def test_model_lm_instantiation(self, config):
        from unturtle.models.llada import LLaDAModelLM
        model = LLaDAModelLM(config).cpu()
        assert model is not None
        assert hasattr(model, "model") and hasattr(model.model, "transformer")

    def test_forward_logits_shape(self, config):
        from unturtle.models.llada import LLaDAModelLM
        model = LLaDAModelLM(config).cpu()
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        # LLaDA uses embedding_size (next multiple of 128 ≥ vocab_size) for logits
        effective_vocab = config.embedding_size if config.embedding_size else config.vocab_size
        assert out.logits.shape == (B, L, effective_vocab)
        assert out.logits.shape[0] == B
        assert out.logits.shape[1] == L

    def test_forward_backward(self, config):
        """Gradients flow through LLaDA."""
        from unturtle.models.llada import LLaDAModelLM
        model = LLaDAModelLM(config).cpu()
        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        out = model(input_ids=input_ids)
        # LLaDA does not compute loss internally — compute manually
        loss = out.logits[:, :, :config.vocab_size].reshape(-1, config.vocab_size).float().log_softmax(-1).mean().neg()
        assert not torch.isnan(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
