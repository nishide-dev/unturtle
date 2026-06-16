"""Tests for the MDLM-DiT native diffusion backbone.

CPU-only: config, instantiation, forward shape, time-agnostic contract,
bidirectional attention, padding, generation, registration round-trip.
No pretrained checkpoints (native re-implementation baseline).
"""

from __future__ import annotations

import pytest
import torch


class TestMDLMDiTConfig:
    def test_config_default_fields(self):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTConfig

        config = MDLMDiTConfig()
        assert config.model_type == "mdlm-dit"
        assert config.hidden_size == 768
        assert config.num_attention_heads == 12
        assert config.num_hidden_layers == 12
        assert config.cond_dim == 128

    def test_config_custom_values(self):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTConfig

        config = MDLMDiTConfig(
            hidden_size=128, num_attention_heads=4, num_hidden_layers=2, vocab_size=1000
        )
        assert config.hidden_size == 128
        assert config.num_attention_heads == 4
        assert config.num_hidden_layers == 2
        assert config.vocab_size == 1000

    def test_config_has_mask_token_id(self):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTConfig

        config = MDLMDiTConfig(mask_token_id=42)
        assert config.mask_token_id == 42

    def test_config_use_cache_false(self):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTConfig

        # Bidirectional, no KV cache.
        assert MDLMDiTConfig().use_cache is False


@pytest.fixture
def tiny_config():
    from unturtle.models.backbones.mdlm_dit import MDLMDiTConfig

    return MDLMDiTConfig(
        vocab_size=512,
        hidden_size=64,
        cond_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
        mask_token_id=511,
    )


class TestMDLMDiTForward:
    def test_instantiation(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu()
        assert model is not None
        assert hasattr(model, "model")

    def test_forward_logits_shape(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        B, L = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert hasattr(out, "logits")
        assert out.logits.shape == (B, L, tiny_config.vocab_size)
        assert out.past_key_values is None

    def test_forward_is_time_agnostic(self, tiny_config):
        """forward must succeed with NO sigma/timesteps argument (Unturtle contract)."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        with torch.no_grad():
            # Passing a stray timesteps kwarg must be absorbed, not error.
            out = model(input_ids=input_ids, timesteps=torch.rand(2))
        assert out.logits.shape == (2, 8, tiny_config.vocab_size)

    def test_forward_backward(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        out = model(input_ids=input_ids)
        loss = out.logits.float().log_softmax(-1).mean().neg()
        assert not torch.isnan(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_adaln_zero_init(self, tiny_config):
        """adaLN_modulation weight & bias are zero-initialized (adaLN-Zero)."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu()
        for block in model.model.blocks:
            assert torch.all(block.adaLN_modulation.weight == 0)
            assert torch.all(block.adaLN_modulation.bias == 0)
        assert torch.all(model.model.output_layer.adaLN_modulation.weight == 0)
        assert torch.all(model.model.output_layer.adaLN_modulation.bias == 0)


def _activate_adaln(model) -> None:
    """Push adaLN gates off their zero-init so attention/MLP actually contribute."""
    torch.manual_seed(0)
    for block in model.model.blocks:
        block.adaLN_modulation.weight.data.normal_(0, 0.02)
        block.adaLN_modulation.bias.data.normal_(0, 0.02)
    model.model.output_layer.adaLN_modulation.weight.data.normal_(0, 0.02)
    model.model.output_layer.adaLN_modulation.bias.data.normal_(0, 0.02)


class TestMDLMDiTAttention:
    def test_bidirectional_attention(self, tiny_config):
        """Output at position i depends on tokens AFTER i (not causal)."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        _activate_adaln(model)
        L = 8
        a = torch.randint(0, tiny_config.vocab_size, (1, L))
        b = a.clone()
        b[0, -1] = (b[0, -1] + 1) % tiny_config.vocab_size  # perturb LAST token
        with torch.no_grad():
            out_a = model(input_ids=a).logits
            out_b = model(input_ids=b).logits
        # Position 0 must change when the last token changes => bidirectional.
        assert not torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-5)

    def test_attention_mask_2d_padding(self, tiny_config):
        """A 2-D [B,L] padding mask is accepted and changes the output."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        _activate_adaln(model)
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        full = torch.ones(1, 8, dtype=torch.long)
        partial = full.clone()
        partial[0, -2:] = 0  # mask out last two positions
        with torch.no_grad():
            out_full = model(input_ids=input_ids, attention_mask=full).logits
            out_part = model(input_ids=input_ids, attention_mask=partial).logits
        assert not torch.allclose(out_full[0, 0], out_part[0, 0], atol=1e-5)

    def test_attention_mask_4d_bool(self, tiny_config):
        """A 4-D [B,1,L,L] bool mask (as _sample passes) is accepted."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        m1d = torch.ones(1, 8, dtype=torch.bool)
        m4d = torch.logical_and(
            m1d.unsqueeze(1).unsqueeze(-2), m1d.unsqueeze(1).unsqueeze(-1)
        )  # [1,1,8,8]
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=m4d).logits
        assert out.shape == (1, 8, tiny_config.vocab_size)
