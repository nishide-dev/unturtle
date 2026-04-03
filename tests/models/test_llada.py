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


# ---------------------------------------------------------------------------
# LLaDA generation (diffusion_generate)
# ---------------------------------------------------------------------------


class TestLLaDAGeneration:
    """Tests for LLaDAGenerationMixin.diffusion_generate on tiny CPU models."""

    MASK_TOKEN_ID = 126336  # LLaDA default; overridden via config below

    @pytest.fixture
    def config(self):
        from unturtle.models.llada import LLaDAConfig
        return LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=512,
            mlp_ratio=4,
            max_sequence_length=64,
            attention_dropout=0.0,
            residual_dropout=0.0,
            embedding_dropout=0.0,
            rope=True,
            init_device="cpu",
            mask_token_id=511,  # use last token as [MASK] for tiny vocab
        )

    @pytest.fixture
    def model(self, config):
        from unturtle.models.llada import LLaDAModelLM
        return LLaDAModelLM(config).eval()

    TINY_MASK_ID = 511

    def test_has_diffusion_generate(self, model):
        from unturtle.models.llada import LLaDAGenerationMixin
        assert isinstance(model, LLaDAGenerationMixin)
        assert callable(model.diffusion_generate)

    def test_prepare_inputs_for_generation_removed(self, model):
        """prepare_inputs_for_generation (AR protocol) must no longer exist."""
        assert not hasattr(model, "prepare_inputs_for_generation"), (
            "prepare_inputs_for_generation should be removed from LLaDAModelLM — "
            "it implemented an AR (KV cache) protocol incompatible with dLLM generation."
        )

    def test_output_shape(self, model, config):
        B, L = 2, 10
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        with torch.no_grad():
            out = model.diffusion_generate(
                input_ids,
                steps=2,
                mask_token_id=self.TINY_MASK_ID,
                max_length=L + 1,
            )
        assert out.shape == (B, L + 1)

    def test_deterministic_with_seed(self, model):
        """Same random seed + same input → identical output."""
        B, L = 1, 8
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        with torch.no_grad():
            torch.manual_seed(42)
            out1 = model.diffusion_generate(
                input_ids.clone(),
                steps=2,
                mask_token_id=self.TINY_MASK_ID,
                temperature=0.0,
                max_length=L + 1,
            )
            torch.manual_seed(42)
            out2 = model.diffusion_generate(
                input_ids.clone(),
                steps=2,
                mask_token_id=self.TINY_MASK_ID,
                temperature=0.0,
                max_length=L + 1,
            )
        assert (out1 == out2).all(), "Same seed must produce identical output"

    def test_no_mask_token_id_raises(self, model):
        """Should raise ValueError if mask_token_id is not provided and not in config."""
        original = getattr(model.config, "mask_token_id", None)
        model.config.mask_token_id = None
        try:
            with pytest.raises(ValueError, match="mask_token_id"):
                B, L = 1, 4
                input_ids = torch.zeros((B, L), dtype=torch.long)
                model.diffusion_generate(input_ids, steps=1, max_length=L + 1)
        finally:
            model.config.mask_token_id = original
