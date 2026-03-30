"""Tests for Dream diffusion language models.

CPU-only tests covering config instantiation, model instantiation with
random weights, and forward pass shapes. No pretrained checkpoints required.
"""

from __future__ import annotations

import pytest
import torch


class TestDreamConfig:
    def test_config_default_fields(self):
        from unturtle.models.dream import DreamConfig
        config = DreamConfig()
        assert config.vocab_size == 151936
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32

    def test_config_custom_values(self):
        from unturtle.models.dream import DreamConfig
        config = DreamConfig(
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        assert config.vocab_size == 10000
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 4

    def test_config_has_mask_token_id(self):
        from unturtle.models.dream import DreamConfig
        config = DreamConfig()
        assert hasattr(config, "mask_token_id")
        assert config.mask_token_id == 151666

    def test_config_use_cache_false(self):
        """Dream configs have use_cache=False by design."""
        from unturtle.models.dream import DreamConfig
        config = DreamConfig()
        assert config.use_cache is False


class TestDreamModel:
    @pytest.fixture
    def config(self):
        from unturtle.models.dream import DreamConfig
        return DreamConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
            mask_token_id=1,
        )

    def test_model_instantiation(self, config):
        from unturtle.models.dream import DreamModel
        model = DreamModel(config).cpu()
        assert model is not None

    def test_forward_logits_shape(self, config):
        from unturtle.models.dream import DreamModel
        model = DreamModel(config).cpu()
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_forward_backward(self, config):
        """Gradients flow through Dream."""
        from unturtle.models.dream import DreamModel
        model = DreamModel(config).cpu()
        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        labels = torch.randint(0, config.vocab_size, (B, L))
        labels[:, ::2] = -100
        out = model(input_ids=input_ids, labels=labels)
        assert out.loss is not None
        assert not torch.isnan(out.loss)
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestDreamGenerationUtils:
    def test_generation_config_creation(self):
        from unturtle.models.dream import DreamGenerationConfig
        gen_config = DreamGenerationConfig()
        assert gen_config is not None

    def test_generation_mixin_importable(self):
        from unturtle.models.dream import DreamGenerationMixin
        assert DreamGenerationMixin is not None
        assert hasattr(DreamGenerationMixin, "diffusion_generate")
