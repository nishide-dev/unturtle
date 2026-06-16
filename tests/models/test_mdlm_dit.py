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


class TestMDLMDiTGeneration:
    TINY_MASK_ID = 511

    @pytest.fixture
    def model(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        return MDLMDiTForMaskedDiffusionLM(tiny_config).eval()

    def test_is_generation_mixin(self, model):
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationMixin,
        )

        assert isinstance(model, MaskedDiffusionGenerationMixin)
        assert callable(model.generate)

    def test_resolve_algorithm_auto_is_mdlm(self, model):
        from unturtle.models.generation.sampler import resolve_algorithm

        assert resolve_algorithm("auto", model, bd3lm_requested=False) == "mdlm"

    def test_block_decode_not_supported(self, model):
        from unturtle.models.generation.sampler import _supports_block_decode

        assert _supports_block_decode(model) is False

    def test_generate_output_shape(self, model):
        B, L = 2, 10
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
                input_ids, steps=2, mask_token_id=self.TINY_MASK_ID, max_length=L + 1
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape == (B, L + 1)

    def test_generate_deterministic_with_seed(self, model):
        B, L = 1, 8
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        with torch.no_grad():
            torch.manual_seed(0)
            o1 = model.generate(
                input_ids.clone(),
                steps=2,
                mask_token_id=self.TINY_MASK_ID,
                temperature=0.0,
                max_length=L + 1,
            )
            torch.manual_seed(0)
            o2 = model.generate(
                input_ids.clone(),
                steps=2,
                mask_token_id=self.TINY_MASK_ID,
                temperature=0.0,
                max_length=L + 1,
            )
        s1 = o1.sequences if hasattr(o1, "sequences") else o1
        s2 = o2.sequences if hasattr(o2, "sequences") else o2
        assert (s1 == s2).all()


class TestMDLMDiTRegistration:
    def test_reexported_from_backbones(self):
        from unturtle.models.backbones import (
            MDLMDiTConfig,
            MDLMDiTForMaskedDiffusionLM,
        )

        assert MDLMDiTConfig.model_type == "mdlm-dit"
        assert MDLMDiTForMaskedDiffusionLM is not None

    def test_registered_in_native_classes(self):
        from unturtle.fast_diffusion_model import _native_model_classes

        classes = _native_model_classes()
        assert "mdlm-dit" in classes
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        assert classes["mdlm-dit"] is MDLMDiTForMaskedDiffusionLM

    def test_save_reload_forward_parity(self, tiny_config, tmp_path):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            ref = model(input_ids=input_ids).logits
        model.save_pretrained(tmp_path)
        reloaded = MDLMDiTForMaskedDiffusionLM.from_pretrained(tmp_path).cpu().eval()
        with torch.no_grad():
            got = reloaded(input_ids=input_ids).logits
        assert torch.allclose(ref, got, atol=1e-5)
