"""Tests for A2D (AutoRegressive→Diffusion) model adapters.

CPU-only tests covering config instantiation, model instantiation with
random weights, forward pass shapes, and AutoConfig/AutoModel registration.
No pretrained checkpoints are downloaded.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# A2D-Llama
# ---------------------------------------------------------------------------


class TestA2DLlama:
    @pytest.fixture
    def config(self):
        from unturtle.models.a2d import A2DLlamaConfig
        return A2DLlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )

    def test_config_model_type(self, config):
        assert config.model_type == "a2d-llama"

    def test_config_inherits_llama_config(self, config):
        import transformers
        assert isinstance(config, transformers.LlamaConfig)

    def test_model_instantiation(self, config):
        from unturtle.models.a2d import A2DLlamaLMHeadModel
        model = A2DLlamaLMHeadModel(config)
        assert model is not None
        assert hasattr(model, "lm_head")

    def test_forward_logits_shape(self, config):
        from unturtle.models.a2d import A2DLlamaLMHeadModel
        model = A2DLlamaLMHeadModel(config)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_autoconfig_registered(self):
        import transformers
        from unturtle.models.a2d import A2DLlamaConfig  # ensure registration
        assert "a2d-llama" in transformers.models.auto.configuration_auto.CONFIG_MAPPING


# ---------------------------------------------------------------------------
# A2D-Qwen2
# ---------------------------------------------------------------------------


class TestA2DQwen2:
    @pytest.fixture
    def config(self):
        from unturtle.models.a2d import A2DQwen2Config
        return A2DQwen2Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            use_sliding_window=False,
        )

    def test_config_model_type(self, config):
        assert config.model_type == "a2d-qwen2"

    def test_config_inherits_qwen2_config(self, config):
        import transformers
        assert isinstance(config, transformers.Qwen2Config)

    def test_model_instantiation(self, config):
        from unturtle.models.a2d import A2DQwen2LMHeadModel
        model = A2DQwen2LMHeadModel(config)
        assert model is not None

    def test_forward_logits_shape(self, config):
        from unturtle.models.a2d import A2DQwen2LMHeadModel
        model = A2DQwen2LMHeadModel(config)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_autoconfig_registered(self):
        import transformers
        from unturtle.models.a2d import A2DQwen2Config  # ensure registration
        assert "a2d-qwen2" in transformers.models.auto.configuration_auto.CONFIG_MAPPING


# ---------------------------------------------------------------------------
# A2D-Qwen3
# ---------------------------------------------------------------------------


class TestA2DQwen3:
    @pytest.fixture
    def config(self):
        from unturtle.models.a2d import A2DQwen3Config
        return A2DQwen3Config(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            use_sliding_window=False,
        )

    def test_config_model_type(self, config):
        assert config.model_type == "a2d-qwen3"

    def test_config_inherits_qwen3_config(self, config):
        import transformers
        assert isinstance(config, transformers.Qwen3Config)

    def test_model_instantiation(self, config):
        from unturtle.models.a2d import A2DQwen3LMHeadModel
        model = A2DQwen3LMHeadModel(config)
        assert model is not None

    def test_forward_logits_shape(self, config):
        from unturtle.models.a2d import A2DQwen3LMHeadModel
        model = A2DQwen3LMHeadModel(config)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_autoconfig_registered(self):
        import transformers
        from unturtle.models.a2d import A2DQwen3Config  # ensure registration
        assert "a2d-qwen3" in transformers.models.auto.configuration_auto.CONFIG_MAPPING
