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

"""Tests for FastDiffusionModel — CPU-only, no pretrained checkpoints downloaded.

Tests cover:
- apply_qkv / apply_o stub installation
- Forward pass through A2DAttention_fast_forward (CPU / SDPA path)
- LoRA application and Triton kernel patching (CPU, lora_dropout=0)
- Bidirectionality: model attends to future tokens (non-causal property)
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Shared tiny A2D-Llama config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_config():
    from unturtle.models.a2d import A2DLlamaConfig

    return A2DLlamaConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )


@pytest.fixture
def tiny_model(tiny_config):
    from unturtle.models.a2d import A2DLlamaLMHeadModel

    model = A2DLlamaLMHeadModel(tiny_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# _install_apply_stubs
# ---------------------------------------------------------------------------

class TestInstallApplyStubs:
    def test_stubs_installed_on_all_layers(self, tiny_model):
        from unturtle.fast_diffusion_model import _install_apply_stubs

        _install_apply_stubs(tiny_model)
        for layer in tiny_model.model.layers:
            attn = layer.self_attn
            assert hasattr(attn, "apply_qkv"), "apply_qkv stub missing"
            assert hasattr(attn, "apply_o"), "apply_o stub missing"

    def test_stub_output_shape(self, tiny_model):
        from unturtle.fast_diffusion_model import _install_apply_stubs

        _install_apply_stubs(tiny_model)
        attn = tiny_model.model.layers[0].self_attn
        B, L = 1, 8
        hidden = torch.randn(B, L, tiny_model.config.hidden_size)
        Q, K, V = attn.apply_qkv(attn, hidden)
        expected = (B, L, tiny_model.config.num_attention_heads * attn.head_dim)
        assert Q.shape == expected


# ---------------------------------------------------------------------------
# A2DAttention_fast_forward (CPU / SDPA path)
# ---------------------------------------------------------------------------

class TestA2DAttentionFastForward:
    def test_forward_returns_correct_shapes(self, tiny_model):
        from unturtle.fast_diffusion_model import _install_apply_stubs
        from unturtle.models.a2d._fast_forward import A2DAttention_fast_forward
        import types

        _install_apply_stubs(tiny_model)
        attn = tiny_model.model.layers[0].self_attn
        attn.forward = types.MethodType(A2DAttention_fast_forward, attn)

        B, L = 2, 8
        hidden = torch.randn(B, L, tiny_model.config.hidden_size)
        out, weights = attn(hidden)

        assert out.shape == (B, L, tiny_model.config.hidden_size)
        assert weights is None

    def test_bidirectional_attends_to_future_tokens(self, tiny_config):
        """A bidirectional model's output at position 0 should depend on
        tokens at positions 1+.  We verify this by comparing outputs with
        two sequences that differ only at position L-1: a truly causal model
        would produce identical output at position 0, a bidirectional model
        would produce different output.
        """
        from unturtle.models.a2d import A2DLlamaLMHeadModel
        from unturtle.fast_diffusion_model import _install_apply_stubs
        from unturtle.models.a2d._fast_forward import A2DAttention_fast_forward
        import types

        model = A2DLlamaLMHeadModel(tiny_config)
        model.eval()
        _install_apply_stubs(model)

        # Patch all attention layers
        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(
                A2DAttention_fast_forward, layer.self_attn
            )

        B, L = 1, 8
        # Two sequences identical everywhere except position L-1
        ids_a = torch.randint(0, tiny_config.vocab_size, (B, L))
        ids_b = ids_a.clone()
        ids_b[0, -1] = (ids_a[0, -1] + 1) % tiny_config.vocab_size

        with torch.no_grad():
            out_a = model(ids_a).logits
            out_b = model(ids_b).logits

        # Position 0 output should differ because model sees future token
        assert not torch.allclose(out_a[:, 0, :], out_b[:, 0, :]), (
            "Position 0 outputs are identical — attention appears to be causal!"
        )


# ---------------------------------------------------------------------------
# FastDiffusionModel.get_peft_model (CPU, no GPU kernel execution)
# ---------------------------------------------------------------------------

class TestGetPeftModel:
    def test_peft_model_wraps_base(self, tiny_model):
        """get_peft_model returns a PEFT-wrapped model."""
        from peft import PeftModel

        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        assert isinstance(peft_model, PeftModel)

    def test_apply_qkv_patched_to_lora(self, tiny_model):
        """After get_peft_model, apply_qkv on attention layers should be
        apply_lora_qkv (the Triton kernel) when lora_dropout=0 and bias='none'.
        """
        from unsloth.kernels.fast_lora import apply_lora_qkv

        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        for layer in peft_model.base_model.model.model.layers:
            assert layer.self_attn.apply_qkv is apply_lora_qkv, (
                f"apply_qkv not patched to apply_lora_qkv: {layer.self_attn.apply_qkv}"
            )

    def test_apply_o_patched_to_lora(self, tiny_model):
        from unsloth.kernels.fast_lora import apply_lora_o

        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        for layer in peft_model.base_model.model.model.layers:
            assert layer.self_attn.apply_o is apply_lora_o, (
                f"apply_o not patched to apply_lora_o: {layer.self_attn.apply_o}"
            )

    def test_fast_attn_forward_injected(self, tiny_model):
        """Attention forward should be replaced with A2DAttention_fast_forward."""
        import types

        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.a2d._fast_forward import A2DAttention_fast_forward

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        for layer in peft_model.base_model.model.model.layers:
            forward_fn = layer.self_attn.forward
            # types.MethodType wraps the function — extract __func__
            if isinstance(forward_fn, types.MethodType):
                forward_fn = forward_fn.__func__
            assert forward_fn is A2DAttention_fast_forward

    def test_peft_model_forward_runs(self, tiny_model):
        """Forward pass through a PEFT-wrapped dLLM should not raise."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        peft_model.eval()

        B, L = 2, 8
        input_ids = torch.randint(0, tiny_model.config.vocab_size, (B, L))
        with torch.no_grad():
            out = peft_model(input_ids)
        assert out.logits.shape == (B, L, tiny_model.config.vocab_size)

    def test_peft_save_load_roundtrip(self, tiny_model, tmp_path):
        """LoRA adapter weights can be saved and reloaded."""
        from peft import PeftModel

        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        save_dir = tmp_path / "adapter"
        peft_model.save_pretrained(str(save_dir))

        # Reload onto fresh base model
        from unturtle.models.a2d import A2DLlamaConfig, A2DLlamaLMHeadModel

        base = A2DLlamaLMHeadModel(tiny_model.config)
        loaded = PeftModel.from_pretrained(base, str(save_dir))
        assert loaded is not None

        # Shape of reloaded LoRA A should match original
        orig_lora_A = (
            peft_model.base_model.model.model.layers[0]
            .self_attn.q_proj.lora_A["default"].weight
        )
        loaded_lora_A = (
            loaded.base_model.model.model.layers[0]
            .self_attn.q_proj.lora_A["default"].weight
        )
        assert orig_lora_A.shape == loaded_lora_A.shape
        assert torch.allclose(orig_lora_A, loaded_lora_A)


# ---------------------------------------------------------------------------
# Dream model patching
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dream_config():
    from unturtle.models.dream.configuration_dream import DreamConfig

    return DreamConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        mask_token_id=1,
        pad_token_id=0,
    )


@pytest.fixture
def tiny_dream_model(tiny_dream_config):
    from unturtle.models.dream.modeling_dream import DreamModel

    model = DreamModel(tiny_dream_config)
    model.eval()
    return model


class TestDreamPatching:
    def test_dream_peft_o_proj_patched(self, tiny_dream_model):
        """After get_peft_model, o_proj layers should have apply_o=apply_lora_o."""
        from unsloth.kernels.fast_lora import apply_lora_o
        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_dream_model,
            r=4,
            target_modules=["o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        for layer in peft_model.base_model.model.model.layers:
            assert layer.self_attn.apply_o is apply_lora_o, (
                f"apply_o not patched: {layer.self_attn.apply_o}"
            )

    def test_dream_peft_qkv_skipped(self, tiny_dream_model):
        """QKV should NOT be patched for Dream (bias=True on q/k/v_proj)."""
        from unsloth.kernels.fast_lora import apply_lora_qkv
        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_dream_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        for layer in peft_model.base_model.model.model.layers:
            # apply_qkv should NOT be apply_lora_qkv (bias=True guard applies)
            assert getattr(layer.self_attn, "apply_qkv", None) is not apply_lora_qkv, (
                "apply_qkv should NOT be apply_lora_qkv for Dream (bias=True)"
            )

    def test_dream_peft_forward_runs(self, tiny_dream_model):
        """Forward pass through a PEFT-wrapped Dream model should not raise."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_dream_model,
            r=4,
            target_modules=["o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        peft_model.eval()

        B, L = 2, 8
        input_ids = torch.randint(0, tiny_dream_model.config.vocab_size, (B, L))
        with torch.no_grad():
            out = peft_model(input_ids=input_ids)
        # DreamModel returns MaskedLMOutput with logits
        assert out.logits.shape == (B, L, tiny_dream_model.config.vocab_size)


# ---------------------------------------------------------------------------
# LLaDA model patching
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_llada_config():
    from unturtle.models.llada.configuration_llada import LLaDAConfig

    return LLaDAConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_hidden_size=128,
        vocab_size=512,
        embedding_size=512,
        max_sequence_length=64,
        block_type="llama",
        activation_type="silu",  # LLaDALlamaBlock does gate*up with silu (not swiglu split)
        rope=True,
        include_bias=False,
        include_qkv_bias=False,
        weight_tying=False,
    )


@pytest.fixture
def tiny_llada_model(tiny_llada_config):
    from unturtle.models.llada.modeling_llada import LLaDAModelLM

    model = LLaDAModelLM(tiny_llada_config)
    model.eval()
    return model


class TestLLaDAPatching:
    def test_llada_peft_attn_out_patched(self, tiny_llada_model):
        """After get_peft_model, attn_out in LLaDALlamaBlocks should have apply_o=apply_lora_o."""
        from unsloth.kernels.fast_lora import apply_lora_o
        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_llada_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        from unturtle.models.llada.modeling_llada import LLaDALlamaBlock

        # LLaDAModelLM wraps LLaDAModel in self.model
        blocks = peft_model.base_model.model.model.transformer.blocks
        for block in blocks:
            if isinstance(block, LLaDALlamaBlock):
                assert block.apply_o is apply_lora_o, (
                    f"apply_o not patched on LLaDALlamaBlock: {block.apply_o}"
                )

    def test_llada_peft_qkv_patched(self, tiny_llada_model):
        """LLaDALlamaBlock q/k/v_proj without bias should get apply_qkv=apply_lora_qkv."""
        from unsloth.kernels.fast_lora import apply_lora_qkv
        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_llada_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        from unturtle.models.llada.modeling_llada import LLaDALlamaBlock

        # LLaDAModelLM wraps LLaDAModel in self.model
        blocks = peft_model.base_model.model.model.transformer.blocks
        for block in blocks:
            if isinstance(block, LLaDALlamaBlock):
                assert block.apply_qkv is apply_lora_qkv, (
                    f"apply_qkv not patched on LLaDALlamaBlock: {block.apply_qkv}"
                )

    def test_llada_peft_forward_runs(self, tiny_llada_model):
        """Forward pass through a PEFT-wrapped LLaDA model should not raise."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_llada_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        peft_model.eval()

        B, L = 2, 8
        input_ids = torch.randint(0, tiny_llada_model.config.vocab_size, (B, L))
        with torch.no_grad():
            out = peft_model(input_ids=input_ids)
        assert out.logits.shape == (B, L, tiny_llada_model.config.vocab_size)
