# Copyright 2025-present nishide-dev & the Unturtle team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""kuleshov-group/mdlm-owt checkpoint -> native MDLM-DiT conversion (#130 PR0).

The conversion contract:

- every ``backbone.*`` tensor maps onto the structurally identical ``model.*``
  parameter, bitwise;
- the zero-sigma ``TimestepEmbedder`` collapses into the constant ``cond``
  vector (``cond := sigma_map(timestep_embedding(0))``), which is exact ONLY
  for ``time_conditioning=False`` checkpoints — anything else must refuse;
- nothing is dropped silently: an unrecognized source key raises, and the
  rotary ``inv_freq`` buffer (recomputed on our side, not copied) is verified
  against the checkpoint instead of being discarded on faith.
"""


import pytest
import torch
import torch.nn.functional as F

from unturtle.models.backbones.mdlm_dit import (
    MDLMDiTConfig,
    MDLMDiTForMaskedDiffusionLM,
)
from unturtle.models.backbones.mdlm_dit.convert_mdlm_owt import (
    FREQUENCY_EMBEDDING_SIZE,
    build_native_model,
    config_from_mdlm_owt,
    convert_mdlm_state_dict,
)

VOCAB = 12  # includes the appended mask token (mdlm: mask_index = vocab_size - 1)
HIDDEN = 8
HEADS = 2
COND = 4
LAYERS = 2


def tiny_config() -> MDLMDiTConfig:
    return MDLMDiTConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        cond_dim=COND,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        dropout=0.1,
        max_position_embeddings=16,
        mask_token_id=VOCAB - 1,
    )


def tiny_source_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    """An upstream-shaped (``backbone.*``) random state dict at tiny dims.

    Key names and shapes mirror kuleshov-group/mdlm-owt's ``model.safetensors``
    exactly (DITBackbone with TimestepEmbedder sigma_map); only sizes shrink.
    """
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    sd: dict[str, torch.Tensor] = {
        "backbone.vocab_embed.embedding": r(VOCAB, HIDDEN),
        "backbone.sigma_map.mlp.0.weight": r(COND, FREQUENCY_EMBEDDING_SIZE),
        "backbone.sigma_map.mlp.0.bias": r(COND),
        "backbone.sigma_map.mlp.2.weight": r(COND, COND),
        "backbone.sigma_map.mlp.2.bias": r(COND),
        "backbone.rotary_emb.inv_freq": 1.0
        / (10_000 ** (torch.arange(0, HIDDEN // HEADS, 2).float() / (HIDDEN // HEADS))),
        "backbone.output_layer.norm_final.weight": r(HIDDEN),
        "backbone.output_layer.linear.weight": r(VOCAB, HIDDEN),
        "backbone.output_layer.linear.bias": r(VOCAB),
        "backbone.output_layer.adaLN_modulation.weight": r(2 * HIDDEN, COND),
        "backbone.output_layer.adaLN_modulation.bias": r(2 * HIDDEN),
    }
    for i in range(LAYERS):
        p = f"backbone.blocks.{i}."
        sd[p + "norm1.weight"] = r(HIDDEN)
        sd[p + "attn_qkv.weight"] = r(3 * HIDDEN, HIDDEN)
        sd[p + "attn_out.weight"] = r(HIDDEN, HIDDEN)
        sd[p + "norm2.weight"] = r(HIDDEN)
        sd[p + "mlp.0.weight"] = r(4 * HIDDEN, HIDDEN)
        sd[p + "mlp.0.bias"] = r(4 * HIDDEN)
        sd[p + "mlp.2.weight"] = r(HIDDEN, 4 * HIDDEN)
        sd[p + "mlp.2.bias"] = r(HIDDEN)
        sd[p + "adaLN_modulation.weight"] = r(6 * HIDDEN, COND)
        sd[p + "adaLN_modulation.bias"] = r(6 * HIDDEN)
    return sd


UPSTREAM_CONFIG = {
    "hidden_dim": 768,
    "cond_dim": 128,
    "n_blocks": 12,
    "n_heads": 12,
    "dropout": 0.1,
    "model_length": 1024,
    "vocab_size": 50258,
    "time_conditioning": False,
}


class TestStateDictConversion:
    def test_converted_state_dict_loads_strict_and_preserves_weights(self):
        config = tiny_config()
        source = tiny_source_state_dict()
        converted = convert_mdlm_state_dict(source, config)

        model = MDLMDiTForMaskedDiffusionLM(config)
        model.load_state_dict(converted, strict=True)

        for src_key, dst_key in [
            ("backbone.vocab_embed.embedding", "model.vocab_embed.embedding"),
            ("backbone.blocks.1.attn_qkv.weight", "model.blocks.1.attn_qkv.weight"),
            ("backbone.blocks.0.mlp.2.bias", "model.blocks.0.mlp.2.bias"),
            (
                "backbone.blocks.1.adaLN_modulation.weight",
                "model.blocks.1.adaLN_modulation.weight",
            ),
            ("backbone.output_layer.linear.weight", "model.output_layer.linear.weight"),
            (
                "backbone.output_layer.norm_final.weight",
                "model.output_layer.norm_final.weight",
            ),
        ]:
            assert torch.equal(model.state_dict()[dst_key], source[src_key]), (
                f"{dst_key} is not bitwise-equal to {src_key}"
            )

    def test_cond_collapses_the_zero_sigma_timestep_embedder(self):
        """Upstream computes c = silu(sigma_map(sigma)) with sigma zeroed
        (time_conditioning=False still runs the MLP); our forward computes
        c = silu(cond).  Exactness therefore requires

            cond == mlp2(silu(mlp1(timestep_embedding(0))))

        where timestep_embedding(0, 256) = [cos(0)]*128 ++ [sin(0)]*128
        = [1]*128 ++ [0]*128, hand-derived here — not read back from the
        implementation under test."""
        config = tiny_config()
        source = tiny_source_state_dict()
        converted = convert_mdlm_state_dict(source, config)

        emb0 = torch.cat(
            [
                torch.ones(FREQUENCY_EMBEDDING_SIZE // 2),
                torch.zeros(FREQUENCY_EMBEDDING_SIZE // 2),
            ]
        )
        h = F.linear(
            emb0,
            source["backbone.sigma_map.mlp.0.weight"],
            source["backbone.sigma_map.mlp.0.bias"],
        )
        expected = F.linear(
            F.silu(h),
            source["backbone.sigma_map.mlp.2.weight"],
            source["backbone.sigma_map.mlp.2.bias"],
        )
        assert torch.equal(converted["model.cond"], expected)

    def test_rotary_inv_freq_mismatch_raises(self):
        config = tiny_config()
        source = tiny_source_state_dict()
        source["backbone.rotary_emb.inv_freq"] = (
            source["backbone.rotary_emb.inv_freq"] * 2.0
        )
        with pytest.raises(ValueError, match="inv_freq"):
            convert_mdlm_state_dict(source, config)

    def test_unconsumed_source_keys_raise_loudly(self):
        config = tiny_config()
        source = tiny_source_state_dict()
        source["backbone.bogus.weight"] = torch.zeros(3)
        with pytest.raises(ValueError, match="backbone.bogus.weight"):
            convert_mdlm_state_dict(source, config)

    def test_non_backbone_keys_raise_with_their_real_name(self):
        """A key outside the backbone.* namespace must be reported verbatim,
        not after a prefix rewrite that would mislabel it."""
        config = tiny_config()
        source = tiny_source_state_dict()
        source["ema.shadow_params"] = torch.zeros(3)
        with pytest.raises(ValueError, match="ema.shadow_params"):
            convert_mdlm_state_dict(source, config)


class TestConfigMapping:
    def test_upstream_fields_map_onto_native_names(self):
        config = config_from_mdlm_owt(UPSTREAM_CONFIG)
        assert config.vocab_size == 50258
        assert config.hidden_size == 768
        assert config.cond_dim == 128
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.dropout == pytest.approx(0.1)
        assert config.max_position_embeddings == 1024

    def test_mask_token_is_the_appended_last_vocab_row(self):
        """mdlm appends the mask token to the (mask-less) gpt2 tokenizer:
        mask_index = vocab_size - 1 = 50257 (diffusion.py).  The tokenizer
        cannot supply it — the config must (CLAUDE.md real-checkpoint rule)."""
        config = config_from_mdlm_owt(UPSTREAM_CONFIG)
        assert config.mask_token_id == 50257

    def test_time_conditioned_checkpoints_are_refused(self):
        """Collapsing sigma_map at sigma=0 is only exact when the checkpoint
        never conditions on time; converting a time-conditioned MDLM this way
        would silently change its function."""
        with pytest.raises(ValueError, match="time_conditioning"):
            config_from_mdlm_owt({**UPSTREAM_CONFIG, "time_conditioning": True})

    def test_a_config_lacking_time_conditioning_is_refused_not_assumed(self):
        """Refuse-don't-guess (#131 review): a missing key would otherwise be
        silently read as time-agnostic — a guess on exactly the field the
        refusal above exists to protect.  mdlm-owt's config carries the key
        explicitly, so requiring it costs nothing."""
        absent = {k: v for k, v in UPSTREAM_CONFIG.items() if k != "time_conditioning"}
        with pytest.raises(ValueError, match="time_conditioning"):
            config_from_mdlm_owt(absent)


class TestBuildNativeModel:
    def test_default_dtype_is_fp32_bitwise(self):
        config = tiny_config()
        source = tiny_source_state_dict()
        model = build_native_model(config, source)
        assert all(p.dtype == torch.float32 for p in model.parameters())
        assert torch.equal(
            model.model.vocab_embed.embedding,
            source["backbone.vocab_embed.embedding"],
        )

    def test_bf16_is_an_explicit_conversion(self):
        """bf16 never happens silently (#112): it is requested by dtype= and
        every parameter lands in bf16, equal to the fp32 source cast once."""
        config = tiny_config()
        source = tiny_source_state_dict()
        model = build_native_model(config, source, dtype=torch.bfloat16)
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())
        assert torch.equal(
            model.model.vocab_embed.embedding,
            source["backbone.vocab_embed.embedding"].to(torch.bfloat16),
        )

    def test_the_built_model_is_genuinely_native(self):
        """No chimera/class-stamping (#107/#112): the loader returns a real
        MDLMDiTForMaskedDiffusionLM constructed from MDLMDiTConfig, whose
        class hierarchy contains no foreign (remote-code) classes."""
        model = build_native_model(tiny_config(), tiny_source_state_dict())
        assert type(model) is MDLMDiTForMaskedDiffusionLM
        assert type(model.config) is MDLMDiTConfig

    def test_the_built_model_is_inference_ready(self):
        """#131 review Important: the checkpoint carries dropout=0.1, so a
        loader that returns a train-mode model hands every caller stochastic
        logits with no error (measured: two forwards differing by up to 7.5).
        A checkpoint loader returns eval mode, like from_pretrained."""
        model = build_native_model(tiny_config(), tiny_source_state_dict())
        assert not model.training
        ids = torch.randint(0, VOCAB - 1, (2, 8))
        with torch.no_grad():
            first = model(input_ids=ids).logits
            second = model(input_ids=ids).logits
        assert torch.equal(first, second), "forwards differ — dropout is live"

    def test_bf16_conversion_keeps_the_rope_table_fp32(self):
        """#131 review Important: a blanket Module.to(bf16) also casts the
        non-persistent inv_freq buffer, and bf16's ~3 significant digits alias
        the low-frequency lanes once multiplied by positions up to 1023
        (~0.45 rad angle error at the checkpoint's own 1024 context).
        Upstream keeps inv_freq fp32 and casts cos/sin at use time; so does
        native Rotary.forward — the buffer must stay fp32."""
        model = build_native_model(
            tiny_config(), tiny_source_state_dict(), dtype=torch.bfloat16
        )
        assert model.model.rotary.inv_freq.dtype == torch.float32
