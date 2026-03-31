"""Tests for A2D (AutoRegressive→Diffusion) model adapters.

CPU-only tests covering config instantiation, model instantiation with
random weights, forward pass shapes, AutoConfig/AutoModel registration,
and bidirectional attention verification.
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


# ---------------------------------------------------------------------------
# Bidirectional attention verification (all A2D variants)
# ---------------------------------------------------------------------------


class TestA2DBidirectional:
    """Verify that A2D models attend to future tokens (non-causal).

    The core property of A2D models is that the causal attention mask has been
    replaced with a padding-only mask, making attention fully bidirectional.
    We verify this by checking that the output at position 0 differs when only
    the last token changes — a causal model would produce identical outputs.
    """

    @pytest.mark.parametrize("model_cls,config_cls,model_type", [
        ("A2DLlamaLMHeadModel", "A2DLlamaConfig", "a2d-llama"),
        ("A2DQwen2LMHeadModel", "A2DQwen2Config", "a2d-qwen2"),
        ("A2DQwen3LMHeadModel", "A2DQwen3Config", "a2d-qwen3"),
    ])
    def test_attends_to_future_tokens(self, model_cls, config_cls, model_type):
        import importlib
        from unturtle.models import a2d as a2d_module

        Config = getattr(a2d_module, config_cls)
        Model = getattr(a2d_module, model_cls)

        config = Config(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )
        # Qwen3 requires head_dim
        if model_type == "a2d-qwen3":
            config.head_dim = config.hidden_size // config.num_attention_heads

        model = Model(config)
        model.eval()

        B, L = 1, 8
        ids_a = torch.randint(0, config.vocab_size, (B, L))
        ids_b = ids_a.clone()
        # Change only the last token
        ids_b[0, -1] = (ids_a[0, -1] + 1) % config.vocab_size

        with torch.no_grad():
            out_a = model(input_ids=ids_a).logits
            out_b = model(input_ids=ids_b).logits

        # Position 0 should differ — model attends to position L-1
        assert not torch.allclose(out_a[:, 0, :], out_b[:, 0, :]), (
            f"{model_type}: position-0 output is identical after changing position {L-1}. "
            "Model appears to be causal — check that A2DModel.forward uses a non-causal mask."
        )
