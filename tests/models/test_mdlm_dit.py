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
