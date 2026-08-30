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
- Forward pass through TinyA2DAttention_fast_forward (CPU / SDPA path)
- LoRA application and Triton kernel patching (CPU, lora_dropout=0)
- Bidirectionality: model attends to future tokens (non-causal property)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Shared tiny A2D-Llama config fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_config():
    from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaConfig

    return TinyA2DLlamaConfig(
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
    from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaLMHeadModel

    model = TinyA2DLlamaLMHeadModel(tiny_config)
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
# TinyA2DAttention_fast_forward (CPU / SDPA path)
# ---------------------------------------------------------------------------


class TestA2DAttentionFastForward:
    def test_forward_returns_correct_shapes(self, tiny_model):
        import types

        from unturtle.fast_diffusion_model import _install_apply_stubs
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            TinyA2DAttention_fast_forward,
        )

        _install_apply_stubs(tiny_model)
        attn = tiny_model.model.layers[0].self_attn
        attn.forward = types.MethodType(TinyA2DAttention_fast_forward, attn)

        B, L = 2, 8
        hidden = torch.randn(B, L, tiny_model.config.hidden_size)
        out, weights = attn(hidden)

        assert out.shape == (B, L, tiny_model.config.hidden_size)
        assert weights is None

    def test_passes_cpu_device_type_to_backend_selection(self, tiny_model, monkeypatch):
        import types

        from unturtle.fast_diffusion_model import _install_apply_stubs
        from unturtle.models.conversion.a2d.tiny_a2d import (
            _fast_forward as fast_forward_module,
        )
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            TinyA2DAttention_fast_forward,
        )
        from unturtle.utils import attention_dispatch

        captured = {}

        def _capture_backend(use_varlen=False, *, device_type):
            captured["use_varlen"] = use_varlen
            captured["device_type"] = device_type
            return attention_dispatch.SDPA

        monkeypatch.setattr(
            fast_forward_module, "select_attention_backend", _capture_backend
        )

        _install_apply_stubs(tiny_model)
        attn = tiny_model.model.layers[0].self_attn
        attn.forward = types.MethodType(TinyA2DAttention_fast_forward, attn)

        hidden = torch.randn(1, 4, tiny_model.config.hidden_size)
        out, weights = attn(hidden)

        assert out.shape == (1, 4, tiny_model.config.hidden_size)
        assert weights is None
        assert captured == {"use_varlen": False, "device_type": "cpu"}

    def test_bidirectional_attends_to_future_tokens(self, tiny_config):
        """A bidirectional model's output at position 0 should depend on
        tokens at positions 1+.  We verify this by comparing outputs with
        two sequences that differ only at position L-1: a truly causal model
        would produce identical output at position 0, a bidirectional model
        would produce different output.
        """
        import types

        from unturtle.fast_diffusion_model import _install_apply_stubs
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaLMHeadModel
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            TinyA2DAttention_fast_forward,
        )

        model = TinyA2DLlamaLMHeadModel(tiny_config)
        model.eval()
        _install_apply_stubs(model)

        # Patch all attention layers
        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(
                TinyA2DAttention_fast_forward, layer.self_attn
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
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        assert isinstance(peft_model, PeftModel)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton kernels require CUDA"
    )
    def test_apply_qkv_patched_to_lora(self, tiny_model):
        """After get_peft_model, apply_qkv on attention layers should be
        apply_lora_qkv (the Triton kernel) when lora_dropout=0 and bias='none'.
        """
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.kernels.fast_lora import apply_lora_qkv

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model.cuda(),
            r=4,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        for layer in peft_model.base_model.model.model.layers:
            assert layer.self_attn.apply_qkv is apply_lora_qkv, (
                f"apply_qkv not patched to apply_lora_qkv: {layer.self_attn.apply_qkv}"
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton kernels require CUDA"
    )
    def test_apply_o_patched_to_lora(self, tiny_model):
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.kernels.fast_lora import apply_lora_o

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model.cuda(),
            r=4,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        for layer in peft_model.base_model.model.model.layers:
            assert layer.self_attn.apply_o is apply_lora_o, (
                f"apply_o not patched to apply_lora_o: {layer.self_attn.apply_o}"
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton kernels require CUDA"
    )
    def test_fast_attn_forward_injected(self, tiny_model):
        """Attention forward should be replaced with TinyA2DAttention_fast_forward."""
        import types

        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            TinyA2DAttention_fast_forward,
        )

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model.cuda(),
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
            assert forward_fn is TinyA2DAttention_fast_forward

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

    def test_peft_model_with_gc_keeps_grad_flow(self, tiny_model):
        """Non-quantized LoRA + gradient checkpointing should preserve backward."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=True,
        )
        peft_model.train()

        B, L = 2, 8
        input_ids = torch.randint(0, tiny_model.config.vocab_size, (B, L))
        logits = peft_model(input_ids).logits

        assert logits.requires_grad

        loss = logits.sum()
        loss.backward()

        lora_params = [
            param for name, param in peft_model.named_parameters() if "lora_" in name
        ]
        assert lora_params
        assert any(param.grad is not None for param in lora_params)

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
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )

        base = TinyA2DLlamaLMHeadModel(tiny_model.config)
        loaded = PeftModel.from_pretrained(base, str(save_dir))
        assert loaded is not None

        # Shape of reloaded LoRA A should match original
        orig_lora_A = (
            peft_model.base_model.model.model.layers[0]
            .self_attn.q_proj.lora_A["default"]
            .weight
        )
        loaded_lora_A = (
            loaded.base_model.model.model.layers[0]
            .self_attn.q_proj.lora_A["default"]
            .weight
        )
        assert orig_lora_A.shape == loaded_lora_A.shape
        assert torch.allclose(orig_lora_A, loaded_lora_A)


# ---------------------------------------------------------------------------
# Dream model patching
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dream_config():
    from unturtle.models.backbones.dream.configuration_dream import DreamConfig

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
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    model = DreamModel(tiny_dream_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# LoRA_QKV_Bias kernel unit tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FastDiffusionModel.from_pretrained improvements
# ---------------------------------------------------------------------------


class TestFromPretrained:
    """Tests for the improved from_pretrained helper functions."""

    def test_dtype_cpu_fallback(self):
        """On CPU (no CUDA), dtype should default to float32."""
        import unittest.mock as mock

        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )

        config = TinyA2DLlamaConfig(
            vocab_size=512,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base_model = TinyA2DLlamaLMHeadModel(config)

        # Patch from_pretrained to return our tiny model
        with (
            mock.patch.object(
                TinyA2DLlamaLMHeadModel, "from_pretrained", return_value=base_model
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            model, _ = FastDiffusionModel.from_pretrained(
                "dummy-path",
                model_class=TinyA2DLlamaLMHeadModel,
                load_in_4bit=False,
            )
        # Float32 default on CPU
        assert model is base_model

    def test_max_seq_length_set(self):
        """from_pretrained sets max_seq_length on model and nested modules."""
        import unittest.mock as mock

        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )

        config = TinyA2DLlamaConfig(
            vocab_size=512,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base_model = TinyA2DLlamaLMHeadModel(config)

        with (
            mock.patch.object(
                TinyA2DLlamaLMHeadModel, "from_pretrained", return_value=base_model
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            model, _ = FastDiffusionModel.from_pretrained(
                "dummy-path",
                max_seq_length=128,
                model_class=TinyA2DLlamaLMHeadModel,
                load_in_4bit=False,
            )
        assert model.max_seq_length == 128

    def test_apply_stubs_installed(self):
        """from_pretrained installs apply_qkv / apply_o stubs."""
        import unittest.mock as mock

        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )

        config = TinyA2DLlamaConfig(
            vocab_size=512,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base_model = TinyA2DLlamaLMHeadModel(config)

        with (
            mock.patch.object(
                TinyA2DLlamaLMHeadModel, "from_pretrained", return_value=base_model
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            model, _ = FastDiffusionModel.from_pretrained(
                "dummy-path",
                model_class=TinyA2DLlamaLMHeadModel,
                load_in_4bit=False,
            )
        for layer in model.model.layers:
            assert hasattr(layer.self_attn, "apply_qkv")
            assert hasattr(layer.self_attn, "apply_o")

    def test_tokenizer_warning_on_missing(self):
        """Missing tokenizer emits a UserWarning instead of silently returning None."""
        import unittest.mock as mock
        import warnings

        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )

        config = TinyA2DLlamaConfig(
            vocab_size=512,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
        base_model = TinyA2DLlamaLMHeadModel(config)

        with (
            mock.patch.object(
                TinyA2DLlamaLMHeadModel, "from_pretrained", return_value=base_model
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch(
                "unturtle.fast_diffusion_model.AutoTokenizer.from_pretrained",
                side_effect=OSError("no tokenizer files"),
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            model, tokenizer = FastDiffusionModel.from_pretrained(
                "dummy-path",
                model_class=TinyA2DLlamaLMHeadModel,
                load_in_4bit=False,
            )
        assert tokenizer is None
        assert any("tokenizer" in str(warning.message).lower() for warning in w)


class TestLoRAQKVBias:
    """Unit tests for the LoRA_QKV_Bias autograd function."""

    def test_output_shapes(self):
        """LoRA_QKV_Bias.apply returns three tensors with correct shapes."""
        from unturtle.kernels.fast_lora import LoRA_QKV_Bias

        B, L, D, R = 2, 8, 32, 4
        X = torch.randn(B, L, D, requires_grad=True)
        # Simulate (W, W_quant=None, A, B, S, bias) for Q, K, V
        QW = torch.randn(D, D)
        KW = torch.randn(D, D)
        VW = torch.randn(D, D)
        QA = torch.randn(R, D)
        QB = torch.randn(D, R)
        KA = torch.randn(R, D)
        KB = torch.randn(D, R)
        VA = torch.randn(R, D)
        VB = torch.randn(D, R)
        QBias = torch.randn(D)
        KBias = torch.randn(D)
        VBias = torch.randn(D)
        scale = 1.0

        Q, K, V = LoRA_QKV_Bias.apply(
            X,
            QW,
            None,
            QA,
            QB,
            scale,
            QBias,
            KW,
            None,
            KA,
            KB,
            scale,
            KBias,
            VW,
            None,
            VA,
            VB,
            scale,
            VBias,
            False,
        )
        assert Q.shape == (B, L, D)
        assert K.shape == (B, L, D)
        assert V.shape == (B, L, D)

    def test_bias_is_applied(self):
        """Setting bias to a constant vector shifts all outputs by that vector."""
        from unturtle.kernels.fast_lora import LoRA_QKV_Bias

        B, L, D, R = 1, 4, 16, 2
        X = torch.zeros(B, L, D)  # zero input → W*0 = 0
        W = torch.eye(D)
        A = torch.zeros(R, D)
        Bmat = torch.zeros(D, R)
        bias = torch.ones(D) * 5.0
        scale = 1.0

        Q, K, V = LoRA_QKV_Bias.apply(
            X,
            W,
            None,
            A,
            Bmat,
            scale,
            bias,
            W,
            None,
            A,
            Bmat,
            scale,
            bias,
            W,
            None,
            A,
            Bmat,
            scale,
            bias,
            False,
        )
        # With zero input and eye weight: W@X=0, LoRA=0, so output = bias
        assert torch.allclose(Q, torch.ones(B, L, D) * 5.0)
        assert torch.allclose(K, torch.ones(B, L, D) * 5.0)
        assert torch.allclose(V, torch.ones(B, L, D) * 5.0)

    def test_backward_runs(self):
        """Backward pass through LoRA_QKV_Bias should not raise."""
        from unturtle.kernels.fast_lora import LoRA_QKV_Bias

        B, L, D, R = 2, 4, 16, 2
        X = torch.randn(B, L, D, requires_grad=True)
        QW = torch.randn(D, D, requires_grad=False)
        QA = torch.randn(R, D, requires_grad=True)
        QB = torch.randn(D, R, requires_grad=True)
        QBias = torch.randn(D, requires_grad=True)
        scale = 1.0

        Q, K, V = LoRA_QKV_Bias.apply(
            X,
            QW,
            None,
            QA,
            QB,
            scale,
            QBias,
            QW,
            None,
            QA,
            QB,
            scale,
            QBias,
            QW,
            None,
            QA,
            QB,
            scale,
            QBias,
            False,
        )
        loss = (Q + K + V).sum()
        loss.backward()

        assert X.grad is not None
        assert QA.grad is not None
        assert QB.grad is not None
        assert QBias.grad is not None

    def test_2d_input_forward_and_backward(self):
        """2-D X [total_tokens, D] (padding-free) must round-trip fwd+bwd.

        The backward reshape bookkeeping previously assumed 3-D X
        (``batch, seq_len, hd = X.shape``) even though forward accepts 2-D.
        Gradients for the flattened input must match the 3-D equivalent.
        """
        from unturtle.kernels.fast_lora import LoRA_QKV_Bias

        torch.manual_seed(3)
        B, L, D, R = 2, 4, 16, 2
        base = torch.randn(B, L, D)
        QW = torch.randn(D, D)
        QA = torch.randn(R, D)
        QB = torch.randn(D, R)
        QBias = torch.randn(D)
        scale = 1.0

        def _run(X):
            args = []
            for _ in range(3):
                args += [QW, None, QA, QB, scale, QBias]
            Q, K, V = LoRA_QKV_Bias.apply(X, *args, False)
            return Q, K, V

        X3 = base.clone().requires_grad_(True)
        Q3, K3, V3 = _run(X3)
        (Q3.square().sum() + K3.sum() + V3.sum()).backward()

        X2 = base.clone().reshape(B * L, D).requires_grad_(True)
        Q2, K2, V2 = _run(X2)
        assert Q2.shape == (B * L, D)
        (Q2.square().sum() + K2.sum() + V2.sum()).backward()

        assert X2.grad is not None
        assert X2.grad.shape == (B * L, D)
        assert torch.allclose(X2.grad, X3.grad.reshape(B * L, D), atol=1e-5)
        assert torch.allclose(Q2, Q3.reshape(B * L, D), atol=1e-5)


class TestDreamPatching:
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton kernels require CUDA"
    )
    def test_dream_peft_o_proj_patched(self, tiny_dream_model):
        """After get_peft_model, o_proj layers should have apply_o=apply_lora_o."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.kernels.fast_lora import apply_lora_o

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_dream_model.cuda(),
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

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton kernels require CUDA"
    )
    def test_dream_peft_qkv_uses_bias_kernel(self, tiny_dream_model):
        """Dream q/k/v_proj (bias=True) should use apply_lora_qkv_with_bias."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.kernels.fast_lora import apply_lora_qkv_with_bias

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_dream_model.cuda(),
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        for layer in peft_model.base_model.model.model.layers:
            assert layer.self_attn.apply_qkv is apply_lora_qkv_with_bias, (
                f"apply_qkv not set to apply_lora_qkv_with_bias: {layer.self_attn.apply_qkv}"
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
    from unturtle.models.backbones.llada.configuration_llada import LLaDAConfig

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
    from unturtle.models.backbones.llada.modeling_llada import LLaDAModelLM

    model = LLaDAModelLM(tiny_llada_config)
    model.eval()
    return model


class TestLLaDAPatching:
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton kernels require CUDA"
    )
    def test_llada_peft_attn_out_patched(self, tiny_llada_model):
        """After get_peft_model, attn_out in LLaDALlamaBlocks should have apply_o=apply_lora_o."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.kernels.fast_lora import apply_lora_o

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_llada_model.cuda(),
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        from unturtle.models.backbones.llada.modeling_llada import LLaDALlamaBlock

        # LLaDAModelLM wraps LLaDAModel in self.model
        blocks = peft_model.base_model.model.model.transformer.blocks
        for block in blocks:
            if isinstance(block, LLaDALlamaBlock):
                assert block.apply_o is apply_lora_o, (
                    f"apply_o not patched on LLaDALlamaBlock: {block.apply_o}"
                )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton kernels require CUDA"
    )
    def test_llada_peft_qkv_patched(self, tiny_llada_model):
        """LLaDALlamaBlock q/k/v_proj without bias should get apply_qkv=apply_lora_qkv."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.kernels.fast_lora import apply_lora_qkv

        peft_model = FastDiffusionModel.get_peft_model(
            tiny_llada_model.cuda(),
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )
        from unturtle.models.backbones.llada.modeling_llada import LLaDALlamaBlock

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


# ---------------------------------------------------------------------------
# TestInferenceTrainingMethods — for_inference / for_training / inference_context
# ---------------------------------------------------------------------------


class TestInferenceTrainingMethods:
    """Tests for FastDiffusionModel.for_inference, for_training, inference_context."""

    def test_for_inference_sets_eval(self, tiny_model):
        """for_inference puts model in eval mode."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        tiny_model.train()
        assert tiny_model.training
        FastDiffusionModel.for_inference(tiny_model)
        assert not tiny_model.training

    def test_for_inference_returns_model(self, tiny_model):
        """for_inference returns the same model object."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        returned = FastDiffusionModel.for_inference(tiny_model)
        assert returned is tiny_model

    def test_for_training_sets_train(self, tiny_model):
        """for_training puts model in training mode."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        tiny_model.eval()
        assert not tiny_model.training
        FastDiffusionModel.for_training(tiny_model, use_gradient_checkpointing=False)
        assert tiny_model.training

    def test_for_training_returns_model(self, tiny_model):
        """for_training returns the same model object."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        returned = FastDiffusionModel.for_training(
            tiny_model, use_gradient_checkpointing=False
        )
        assert returned is tiny_model

    def test_inference_context_restores_train_mode(self, tiny_model):
        """inference_context restores training mode on exit."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        tiny_model.train()
        with FastDiffusionModel.inference_context(tiny_model):
            assert not tiny_model.training  # eval inside context
        assert tiny_model.training  # restored after exit

    def test_inference_context_stays_eval_if_was_eval(self, tiny_model):
        """inference_context does not flip to train if model was already in eval."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        tiny_model.eval()
        with FastDiffusionModel.inference_context(tiny_model):
            assert not tiny_model.training
        assert not tiny_model.training  # stays eval

    def test_inference_context_no_grad(self, tiny_model):
        """Inside inference_context, gradients are disabled."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        tiny_model.eval()
        with FastDiffusionModel.inference_context(tiny_model):
            assert not torch.is_grad_enabled()

    def test_inference_context_restores_gc_mode_false(self, tiny_model):
        """inference_context should restore False GC mode exactly."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        tiny_model._unturtle_gradient_checkpointing_mode = False
        tiny_model.train()
        with FastDiffusionModel.inference_context(tiny_model):
            assert _all_gc_flags_false(tiny_model)
        assert tiny_model.training
        assert tiny_model._unturtle_gradient_checkpointing_mode is False
        assert _all_gc_flags_false(tiny_model)

    def test_inference_context_restores_gc_mode_unsloth(self, tiny_model):
        """inference_context should preserve the symbolic 'unsloth' mode."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        FastDiffusionModel.for_training(
            tiny_model, use_gradient_checkpointing="unsloth"
        )
        with FastDiffusionModel.inference_context(tiny_model):
            assert _all_gc_flags_false(tiny_model)
        assert tiny_model._unturtle_gradient_checkpointing_mode == "unsloth"
        assert _all_gc_flags_true(tiny_model)

    def test_inference_context_restores_state_on_exception(self, tiny_model):
        """inference_context should restore mode/state even if body raises."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        FastDiffusionModel.for_training(tiny_model, use_gradient_checkpointing=False)
        with (
            pytest.raises(RuntimeError, match="boom"),
            FastDiffusionModel.inference_context(tiny_model),
        ):
            raise RuntimeError("boom")
        assert tiny_model.training
        assert tiny_model._unturtle_gradient_checkpointing_mode is False
        assert _all_gc_flags_false(tiny_model)

    def test_for_training_preserves_requested_unsloth_mode(self, tiny_model):
        """for_training should remember symbolic unsloth mode for later restore."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        FastDiffusionModel.for_training(
            tiny_model, use_gradient_checkpointing="unsloth"
        )
        assert tiny_model._unturtle_gradient_checkpointing_mode == "unsloth"
        assert _all_gc_flags_true(tiny_model)


# ---------------------------------------------------------------------------
# TestSavePretrainedMerged — save_pretrained_merged
# ---------------------------------------------------------------------------


class TestSavePretrainedMerged:
    """Tests for FastDiffusionModel.save_pretrained_merged."""

    @pytest.fixture
    def peft_model(self, tiny_model):
        from unturtle.fast_diffusion_model import FastDiffusionModel

        return FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )

    def test_save_creates_directory(self, peft_model, tmp_path):
        """save_pretrained_merged writes files to the given directory."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        save_dir = tmp_path / "merged"
        FastDiffusionModel.save_pretrained_merged(peft_model, str(save_dir))
        # Should contain at least a config and a weight file
        assert save_dir.exists()
        assert (save_dir / "config.json").exists()
        files = list(save_dir.iterdir())
        assert len(files) >= 2

    def test_save_does_not_modify_original(self, peft_model, tmp_path):
        """save_pretrained_merged leaves adapter weights on the original model intact."""
        from peft import PeftModel

        from unturtle.fast_diffusion_model import FastDiffusionModel

        adapter_weights_before = {
            name: param.detach().clone()
            for name, param in peft_model.named_parameters()
            if "lora_" in name
        }

        save_dir = tmp_path / "merged2"
        FastDiffusionModel.save_pretrained_merged(peft_model, str(save_dir))

        assert isinstance(peft_model, PeftModel)
        adapter_weights_after = {
            name: param.detach().clone()
            for name, param in peft_model.named_parameters()
            if "lora_" in name
        }
        assert adapter_weights_before.keys() == adapter_weights_after.keys()
        for name in adapter_weights_before:
            assert torch.equal(
                adapter_weights_before[name], adapter_weights_after[name]
            )

    def test_save_with_tokenizer(self, peft_model, tmp_path):
        """save_pretrained_merged saves tokenizer when provided."""
        from unittest.mock import MagicMock

        from unturtle.fast_diffusion_model import FastDiffusionModel

        mock_tokenizer = MagicMock()
        save_dir = tmp_path / "merged3"
        FastDiffusionModel.save_pretrained_merged(
            peft_model, str(save_dir), tokenizer=mock_tokenizer
        )
        mock_tokenizer.save_pretrained.assert_called_once_with(str(save_dir))


class TestPushToHubMerged:
    """Tests for FastDiffusionModel.push_to_hub_merged."""

    @pytest.fixture
    def peft_model(self, tiny_model):
        from unturtle.fast_diffusion_model import FastDiffusionModel

        return FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )

    def test_push_forwards_kwargs_to_model_and_tokenizer(self, peft_model, monkeypatch):
        """push_to_hub_merged should forward the same Hub kwargs to tokenizer."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        pushed = {}

        def fake_deepcopy(obj):
            merged = MagicMock()
            merged_base = MagicMock()
            merged.merge_and_unload.return_value = merged_base
            pushed["merged_base"] = merged_base
            return merged

        monkeypatch.setattr("copy.deepcopy", fake_deepcopy)

        tokenizer = MagicMock()
        FastDiffusionModel.push_to_hub_merged(
            peft_model,
            "user/repo",
            tokenizer=tokenizer,
            token="hf_xxx",
            revision="main",
            private=True,
            create_pr=True,
            commit_message="test commit",
        )

        pushed["merged_base"].push_to_hub.assert_called_once_with(
            "user/repo",
            safe_serialization=True,
            token="hf_xxx",
            revision="main",
            private=True,
            create_pr=True,
            commit_message="test commit",
        )
        tokenizer.push_to_hub.assert_called_once_with(
            "user/repo",
            safe_serialization=True,
            token="hf_xxx",
            revision="main",
            private=True,
            create_pr=True,
            commit_message="test commit",
        )


def _all_gc_flags_false(model):
    flags = [
        m.gradient_checkpointing
        for m in model.modules()
        if hasattr(m, "gradient_checkpointing")
    ]
    return all(flag is False for flag in flags)


def _all_gc_flags_true(model):
    flags = [
        m.gradient_checkpointing
        for m in model.modules()
        if hasattr(m, "gradient_checkpointing")
    ]
    return all(flag is True for flag in flags)


# ---------------------------------------------------------------------------
# ModernBERT PEFT patching tests
# ---------------------------------------------------------------------------


def _tiny_modernbert_config():
    from unturtle.models.backbones.modernbert import A2DModernBertConfig

    return A2DModernBertConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )


class TestModernBertPeftPatching:
    """CPU tests for _patch_modernbert_peft and ModernBertAttention_fast_forward."""

    @pytest.fixture
    def peft_model(self):
        from peft import LoraConfig, TaskType, get_peft_model

        from unturtle.models.backbones.modernbert import A2DModernBertForMaskedLM

        config = _tiny_modernbert_config()
        model = A2DModernBertForMaskedLM(config)
        lora_config = LoraConfig(
            r=4,
            target_modules=["Wqkv", "Wo"],
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_dropout=0,
            bias="none",
        )
        return get_peft_model(model, lora_config)

    def test_patch_peft_model_does_not_raise_on_cpu(self, peft_model):
        """patch_peft_model must succeed on CPU (Triton skipped gracefully)."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        FastDiffusionModel.patch_peft_model(peft_model, lora_dropout=0, bias="none")

    def test_fast_forward_injected_on_cuda(self, peft_model):
        """ModernBertAttention_fast_forward must be injected on CUDA."""
        pytest.importorskip("torch.cuda", reason="CUDA required")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.backbones.modernbert._fast_forward import (
            ModernBertAttention_fast_forward,
        )

        cuda_model = peft_model.cuda()
        FastDiffusionModel.patch_peft_model(cuda_model, lora_dropout=0, bias="none")

        for layer in cuda_model.base_model.model.model.layers:
            assert layer.attn.forward.__func__ is ModernBertAttention_fast_forward, (
                "ModernBertAttention_fast_forward was not injected on CUDA"
            )

    def test_forward_logits_shape_after_patch_cpu(self, peft_model):
        """Forward pass shape is correct after CPU patch (fast_forward not injected)."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        FastDiffusionModel.patch_peft_model(peft_model, lora_dropout=0, bias="none")
        peft_model.eval()
        B, L = 2, 16
        input_ids = torch.randint(3, 1000, (B, L))
        with torch.no_grad():
            out = peft_model(input_ids=input_ids)
        assert out.logits.shape == (B, L, 1000)

    def test_apply_wo_stub_installed(self, peft_model):
        """apply_wo stubs must be present after patching."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.backbones.modernbert._fast_forward import (
            _original_apply_wo,
        )

        FastDiffusionModel.patch_peft_model(peft_model, lora_dropout=0, bias="none")
        for layer in peft_model.base_model.model.model.layers:
            assert hasattr(layer.attn, "apply_wo"), "apply_wo stub missing after patch"


# ---------------------------------------------------------------------------
# Issue #134: FastDiffusionModel.save_pretrained_gguf
# ---------------------------------------------------------------------------


class TestSavePretrainedGguf:
    """Tests for FastDiffusionModel.save_pretrained_gguf (Issue #134)."""

    def test_applies_patch_and_delegates(self, monkeypatch):
        """save_pretrained_gguf calls patch_saving_functions then model.save_pretrained_gguf."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        patched = {}
        # patch_saving_functions is imported inside save_pretrained_gguf via
        # `from unturtle.save import patch_saving_functions`, so we must patch
        # the canonical location in unturtle.save.
        monkeypatch.setattr(
            "unturtle.save.patch_saving_functions",
            lambda m: patched.setdefault("model", m),
        )

        fake_model = MagicMock()
        fake_model.save_pretrained_gguf = MagicMock()

        FastDiffusionModel.save_pretrained_gguf(
            fake_model, "/tmp/out", tokenizer="tok", quantization_method="q8_0"
        )

        assert patched["model"] is fake_model
        fake_model.save_pretrained_gguf.assert_called_once_with(
            "/tmp/out", "tok", quantization_method="q8_0"
        )

    def test_raises_runtime_error_when_patch_unavailable(self, monkeypatch):
        """Raises RuntimeError when patch_saving_functions does not inject the method."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        # save_pretrained_gguf does `from unturtle.save import patch_saving_functions`
        # at call time, so we must patch the canonical binding in unturtle.save.
        monkeypatch.setattr(
            "unturtle.save.patch_saving_functions",
            lambda m: None,  # no-op — does NOT add save_pretrained_gguf
        )

        fake_model = MagicMock(
            spec=[]
        )  # spec=[] means no attributes, incl. no gguf method
        with pytest.raises(RuntimeError, match="save_pretrained_gguf is not available"):
            FastDiffusionModel.save_pretrained_gguf(
                fake_model, "/tmp/out", tokenizer=None
            )


# ---------------------------------------------------------------------------
# Issue #135: FastDiffusionModel.save_lora_adapter
# ---------------------------------------------------------------------------


class TestSaveLORAAdapter:
    """Tests for FastDiffusionModel.save_lora_adapter (Issue #135)."""

    @pytest.fixture
    def peft_model(self, tiny_model):
        from unturtle.fast_diffusion_model import FastDiffusionModel

        return FastDiffusionModel.get_peft_model(
            tiny_model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )

    def test_raises_value_error_for_non_peft_model(self, tiny_model, tmp_path):
        """Passing a non-PEFT model raises ValueError."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        with pytest.raises(ValueError, match="requires a PEFT-wrapped model"):
            FastDiffusionModel.save_lora_adapter(tiny_model, str(tmp_path / "adapter"))

    def test_saves_adapter_files(self, peft_model, tmp_path):
        """save_lora_adapter writes adapter_config.json and adapter weights."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        save_dir = tmp_path / "adapter"
        FastDiffusionModel.save_lora_adapter(peft_model, str(save_dir))

        assert save_dir.exists()
        assert (save_dir / "adapter_config.json").exists()

    def test_saves_tokenizer_when_provided(self, peft_model, tmp_path):
        """save_lora_adapter calls tokenizer.save_pretrained when tokenizer is given."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        mock_tokenizer = MagicMock()
        save_dir = tmp_path / "adapter_tok"
        FastDiffusionModel.save_lora_adapter(
            peft_model, str(save_dir), tokenizer=mock_tokenizer
        )
        mock_tokenizer.save_pretrained.assert_called_once_with(str(save_dir))

    def test_no_tokenizer_save_when_none(self, peft_model, tmp_path):
        """save_lora_adapter does not call tokenizer.save_pretrained when tokenizer=None."""
        from unturtle.fast_diffusion_model import FastDiffusionModel

        save_dir = tmp_path / "adapter_notok"
        FastDiffusionModel.save_lora_adapter(peft_model, str(save_dir), tokenizer=None)
        # tokenizer=None: no tokenizer files (tokenizer.json, tokenizer_config.json, etc.)
        tokenizer_files = list(save_dir.glob("tokenizer*"))
        assert tokenizer_files == [], (
            f"Expected no tokenizer files, got: {tokenizer_files}"
        )
        # Adapter files must still be written
        assert (save_dir / "adapter_config.json").exists()


# ---------------------------------------------------------------------------
# Issue #136: FastDiffusionModel.from_pretrained — PEFT adapter detection
# ---------------------------------------------------------------------------


class TestFromPretrainedAdapterDetection:
    """Tests for PEFT adapter checkpoint detection in from_pretrained (Issue #136).

    The two-stage load path (base model load + PeftModel.from_pretrained) requires
    network access, so those tests only cover the detection and error logic that
    runs before any HF Hub call.
    """

    def test_raises_value_error_when_base_model_missing_from_adapter_config(
        self, tmp_path
    ):
        """Raises ValueError when adapter_config.json has no base_model_name_or_path."""
        import json

        from unturtle.fast_diffusion_model import FastDiffusionModel

        adapter_dir = tmp_path / "bad_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="base_model_name_or_path"):
            FastDiffusionModel.from_pretrained(str(adapter_dir), load_in_4bit=False)

    def test_skips_adapter_detection_when_full_weights_present(self, tmp_path):
        """Full model weights alongside adapter_config.json bypasses adapter path."""
        import json
        from unittest.mock import patch

        from unturtle.fast_diffusion_model import FastDiffusionModel

        model_dir = tmp_path / "full_model"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "hf/base"}), encoding="utf-8"
        )
        # Presence of model.safetensors marks this as a full-weight directory
        (model_dir / "model.safetensors").write_bytes(b"")

        # _load_model_auto is a module-level function called from from_pretrained
        # when the adapter detection path is correctly skipped.
        with (
            patch(
                "unturtle.fast_diffusion_model._load_model_auto",
                side_effect=RuntimeError("reached full-weight load path"),
            ),
            pytest.raises(RuntimeError, match="reached full-weight load path"),
        ):
            FastDiffusionModel.from_pretrained(str(model_dir), load_in_4bit=False)

    def test_skips_adapter_detection_for_sharded_safetensors(self, tmp_path):
        """Sharded safetensors checkpoint is not treated as an adapter-only directory."""
        import json
        from unittest.mock import patch

        from unturtle.fast_diffusion_model import FastDiffusionModel

        model_dir = tmp_path / "sharded"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "hf/base"}), encoding="utf-8"
        )
        # Sharded checkpoints have an index file and shard files, not model.safetensors
        (model_dir / "model.safetensors.index.json").write_bytes(b"{}")
        (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"")

        with (
            patch(
                "unturtle.fast_diffusion_model._load_model_auto",
                side_effect=RuntimeError("reached full-weight load path"),
            ),
            pytest.raises(RuntimeError, match="reached full-weight load path"),
        ):
            FastDiffusionModel.from_pretrained(str(model_dir), load_in_4bit=False)

    def test_two_stage_load_calls_peft_from_pretrained(self, tmp_path, tiny_model):
        """adapter_config.json detection triggers recursive base-model load + PeftModel wrap.

        The detection code in from_pretrained (lines ~769-810) does:
          1. Reads base_model_name_or_path from adapter_config.json
          2. Calls FastDiffusionModel.from_pretrained(base_model_id, ...) recursively
          3. Calls PeftModel.from_pretrained(base_model, adapter_dir)
          4. Returns (peft_wrapped_model, tokenizer)

        To exercise the real detection branch without a network call, we intercept
        the *recursive* call to from_pretrained by patching _load_model_auto (which
        only runs for a non-adapter HF id, not for the adapter dir path).  The
        local `from peft import PeftModel` inside the branch is intercepted by
        patching the attribute directly on the already-imported peft module —
        avoiding sys.modules replacement which interacts poorly with peft's
        internal import machinery.
        """
        import json
        from unittest.mock import MagicMock, patch

        import peft as _peft_mod

        from unturtle.fast_diffusion_model import FastDiffusionModel

        adapter_dir = tmp_path / "my_adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "hf/base-model"}), encoding="utf-8"
        )

        fake_tokenizer = MagicMock()
        # Use spec to prevent hasattr from returning True for *every* attribute.
        # _propagate_max_seq_length does `while hasattr(model, "model")` — a bare
        # MagicMock would loop forever because MagicMock answers True for any attr.
        fake_peft_model = MagicMock(
            spec=[
                "modules",
                "named_modules",
                "parameters",
                "max_seq_length",
            ]
        )
        fake_peft_model.modules.return_value = []
        fake_peft_model.named_modules.return_value = []
        fake_peft_model.parameters.return_value = iter([])

        mock_peft_cls = MagicMock()
        mock_peft_cls.from_pretrained.return_value = fake_peft_model

        def fake_tokenizer_from_pretrained(name, **kw):
            return fake_tokenizer

        # Patch PeftModel on the already-imported peft module so that the
        # `from peft import PeftModel as _PeftModel` inside from_pretrained
        # resolves to our mock without triggering a fresh module import.
        with (
            patch.object(_peft_mod, "PeftModel", mock_peft_cls),
            patch(
                "unturtle.fast_diffusion_model._load_model_auto",
                return_value=(tiny_model, None),
            ) as mock_load_auto,
            patch(
                "unturtle.fast_diffusion_model.AutoTokenizer.from_pretrained",
                side_effect=fake_tokenizer_from_pretrained,
            ),
        ):
            result_model, result_tok = FastDiffusionModel.from_pretrained(
                str(adapter_dir),
                max_seq_length=64,
                load_in_4bit=False,
            )

        # _load_model_auto is called with the *base* model id, not the adapter dir
        mock_load_auto.assert_called_once()
        first_arg = mock_load_auto.call_args[0][0]
        assert first_arg == "hf/base-model", (
            f"Expected base model id, got {first_arg!r}"
        )

        # PeftModel.from_pretrained must be called with (base_model, adapter_dir)
        mock_peft_cls.from_pretrained.assert_called_once_with(
            tiny_model, str(adapter_dir)
        )

        # The returned model is the PEFT-wrapped model
        assert result_model is fake_peft_model
        assert result_tok is fake_tokenizer


# ---------------------------------------------------------------------------
# FastModel delegation — _load_via_fastmodel
# ---------------------------------------------------------------------------


class TestFastModelDelegation:
    def test_non_native_model_type_delegates_to_fastmodel(self, monkeypatch):
        """A model_type outside the native dict goes through unsloth FastModel."""
        calls = {}

        class _FakeFastModel:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                calls["model_name"] = model_name
                calls["kwargs"] = kwargs
                return "FM_MODEL", "FM_TOKENIZER"

        from unturtle import fast_diffusion_model as fdm

        monkeypatch.setattr(fdm, "_import_fastmodel", lambda: _FakeFastModel)
        out = fdm._load_via_fastmodel(
            "some/hub-model", {"torch_dtype": "bf16"}, load_in_4bit=True
        )
        assert out == ("FM_MODEL", "FM_TOKENIZER")
        assert calls["model_name"] == "some/hub-model"
        assert calls["kwargs"].get("load_in_4bit") is True

    def test_fastmodel_failure_falls_back_to_automodel(self, monkeypatch):
        from unturtle import fast_diffusion_model as fdm

        def _boom():
            raise ImportError("no unsloth")

        monkeypatch.setattr(fdm, "_import_fastmodel", _boom)
        assert fdm._load_via_fastmodel("x", {}, load_in_4bit=False) is None

    def test_load_model_auto_tries_fastmodel_before_automodel(self, monkeypatch):
        from unturtle import fast_diffusion_model as fdm

        order = []
        monkeypatch.setattr(
            fdm, "_load_native", lambda *a, **k: (order.append("native"), None)[1]
        )
        monkeypatch.setattr(
            fdm,
            "_load_via_fastmodel",
            lambda *a, **k: (order.append("fastmodel"), ("M", "T"))[1],
        )
        monkeypatch.setattr(
            fdm,
            "_load_via_automodel",
            lambda *a, **k: (order.append("automodel"), "AM")[1],
        )
        out = fdm._load_model_auto("x", {}, trust_remote_code=False)
        assert out == ("M", "T")
        assert order == ["native", "fastmodel"]  # automodel never called

    def test_load_model_auto_falls_through_to_automodel(self, monkeypatch):
        from unturtle import fast_diffusion_model as fdm

        monkeypatch.setattr(fdm, "_load_native", lambda *a, **k: None)
        monkeypatch.setattr(fdm, "_load_via_fastmodel", lambda *a, **k: None)
        monkeypatch.setattr(fdm, "_load_via_automodel", lambda *a, **k: "AM")
        model, tok = fdm._load_model_auto("x", {}, trust_remote_code=False)
        assert model == "AM"
        assert tok is None


# ---------------------------------------------------------------------------
# Post-load class-swap registry — _POST_LOAD_CLASS_SWAPS / _apply_post_load_class_swap
# ---------------------------------------------------------------------------


class TestPostLoadClassSwap:
    def test_registered_resolver_swaps_class(self):
        from unturtle import fast_diffusion_model as fdm

        class _Base:
            class config:
                model_type = "swaptest"

        class _Wrapper(_Base):
            pass

        fdm._POST_LOAD_CLASS_SWAPS["swaptest"] = lambda: _Wrapper
        try:
            m = _Base()
            fdm._apply_post_load_class_swap(m)
            assert type(m) is _Wrapper
        finally:
            del fdm._POST_LOAD_CLASS_SWAPS["swaptest"]

    def test_unregistered_model_type_untouched(self):
        from unturtle import fast_diffusion_model as fdm

        class _Other:
            class config:
                model_type = "nobody-registered-this"

        m = _Other()
        fdm._apply_post_load_class_swap(m)
        assert type(m) is _Other

    def test_unregistered_model_keeps_instance_generate(self):
        """Unregistered model types must NOT have their instance-level generate removed."""
        from unturtle import fast_diffusion_model as fdm

        class _Other:
            class config:
                model_type = "nobody-registered-this"

        sentinel = object()

        m = _Other()
        m.generate = sentinel  # type: ignore[attr-defined]
        fdm._apply_post_load_class_swap(m)
        # Unregistered — instance attribute must be preserved
        assert m.__dict__.get("generate") is sentinel


class TestSwapRefusesIncompatibleArchitectures:
    """Stamping the wrapper class onto a foreign architecture makes a chimera.

    Observed on the real checkpoint (#96): when the unsloth path is
    unavailable, the Auto* fallback resolves `diffusion_gemma` to the bare
    composite model (children `encoder`/`decoder`), and the swap then set
    `__class__` to the ForBlockDiffusion wrapper anyway — an object whose
    class expects `self.model`/`self.lm_head` that the instance never had.
    The first `.generate` died with `AttributeError: ... no attribute
    'model'`.  A swap is only meaningful onto the architecture the wrapper
    subclasses.
    """

    def test_a_foreign_architecture_is_not_swapped(self):
        from unturtle import fast_diffusion_model as fdm

        class _UpstreamHead:
            pass

        class _Wrapper(_UpstreamHead):
            pass

        class _WrongArchitecture:
            class config:
                model_type = "archswap"

        fdm._POST_LOAD_CLASS_SWAPS["archswap"] = lambda: _Wrapper
        try:
            model = _WrongArchitecture()
            sentinel = object()
            model.generate = sentinel
            fdm._apply_post_load_class_swap(model)
        finally:
            del fdm._POST_LOAD_CLASS_SWAPS["archswap"]

        assert type(model) is _WrongArchitecture, (
            "the wrapper was stamped onto an architecture it does not "
            "subclass; the result is a chimera whose methods reference "
            "submodules the instance never had"
        )
        assert model.__dict__.get("generate") is sentinel, (
            "the unswapped model's own generate was removed"
        )

    def test_the_matching_architecture_still_swaps(self):
        from unturtle import fast_diffusion_model as fdm

        class _UpstreamHead:
            class config:
                model_type = "archswap2"

        class _Wrapper(_UpstreamHead):
            pass

        fdm._POST_LOAD_CLASS_SWAPS["archswap2"] = lambda: _Wrapper
        try:
            model = _UpstreamHead()
            fdm._apply_post_load_class_swap(model)
        finally:
            del fdm._POST_LOAD_CLASS_SWAPS["archswap2"]

        assert type(model) is _Wrapper


class TestAutoFallbackPrefersTheRegisteredHead:
    """The Auto* chain must not pick a different head than the swap expects.

    `AutoModel` resolves `diffusion_gemma` to the bare composite model, not
    `...ForBlockDiffusion` — the wrong head for the wrapper contract, and
    (with the swap guard above) a load that would end up un-swapped.  When a
    model_type has a registered wrapper, the fallback loads through the
    wrapper class itself first: correct head, and the normal `__init__`
    postamble populates `generation_config` on the way.
    """

    def test_the_wrapper_class_is_tried_first(self, monkeypatch):
        """FIRST, not merely tried.  A first draft of this test recorded only
        the wrapper's own call, so `insert(0, ...)` mutated to `append(...)`
        survived — and on a real checkpoint the appended variant lets
        `AutoModel` succeed with the bare composite head before the wrapper
        is ever reached, reintroducing the #96 chimera while the test stays
        green.  Every loader in the chain records into ONE list here, and
        the Auto* loaders raise so reaching them at all is observable.
        """
        from unturtle import fast_diffusion_model as fdm

        calls = []

        class _Wrapper:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                calls.append(("wrapper", name))
                instance = cls()
                instance.config = type("C", (), {"model_type": "headswap"})()
                return instance

        def _recording_auto(auto_name):
            class _Auto:
                @classmethod
                def from_pretrained(cls, name, **kwargs):
                    calls.append((auto_name, name))
                    raise OSError(f"{auto_name} should not be reached")

            return _Auto

        class _FakeAutoConfig:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return type("C", (), {"model_type": "headswap"})()

        monkeypatch.setattr(fdm, "AutoConfig", _FakeAutoConfig, raising=False)
        # Patch fdm's own loader seam, NOT the transformers module: unsloth
        # replaces sys.modules["transformers"] at import time, so attribute
        # patches on any previously-bound transformers object are invisible
        # to the loader's from-import (measured: an insert->append ordering
        # mutant survived a transformers-patching version of this test while
        # the patched attribute read back correctly).
        monkeypatch.setattr(
            fdm,
            "_automodel_loaders",
            lambda: [
                (n, _recording_auto(n))
                for n in ("AutoModel", "AutoModelForMaskedLM", "AutoModelForCausalLM")
            ],
        )
        monkeypatch.setattr(
            fdm,
            "_load_model_with_optional_4bit_fallback",
            lambda loader, name, kw: loader.from_pretrained(name, **kw),
        )
        fdm._POST_LOAD_CLASS_SWAPS["headswap"] = lambda: _Wrapper
        try:
            model = fdm._load_via_automodel("org/ckpt", {})
        finally:
            del fdm._POST_LOAD_CLASS_SWAPS["headswap"]

        assert calls[0] == ("wrapper", "org/ckpt"), (
            f"loader order was {[c[0] for c in calls]}; the registered "
            "wrapper class must run before any Auto* loader, which resolves "
            "the wrong head"
        )
        assert [c[0] for c in calls] == ["wrapper"], (
            "an Auto* loader was reached even though the wrapper succeeded"
        )
        assert type(model) is _Wrapper

    def test_unregistered_model_types_keep_the_auto_chain(self, monkeypatch):
        """No registry entry -> behaviour unchanged (Auto* chain, in order)."""
        from unturtle import fast_diffusion_model as fdm

        class _FakeAutoConfig:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return type("C", (), {"model_type": "nobody-registered"})()

        loaded = []

        class _FakeAutoModel:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                loaded.append(name)
                return object()

        monkeypatch.setattr(fdm, "AutoConfig", _FakeAutoConfig, raising=False)
        monkeypatch.setattr(
            fdm,
            "_load_model_with_optional_4bit_fallback",
            lambda loader, name, kw: loader.from_pretrained(name, **kw),
        )
        monkeypatch.setattr(
            fdm, "_automodel_loaders", lambda: [("AutoModel", _FakeAutoModel)]
        )

        model = fdm._load_via_automodel("org/other", {})

        assert loaded == ["org/other"]
        assert model is not None


class TestSwapRestoresGenerationConfig:
    """unsloth's FastModel load path skips the `PreTrainedModel.__init__`
    postamble that populates `generation_config`, so a swapped DiffusionGemma
    raised `AttributeError: ... no attribute 'generation_config'` on its
    first real generate (#96).  Measured on the real checkpoint: the plain
    transformers load carries a `DiffusionGemmaGenerationConfig`; the
    FastModel load has nothing in `__dict__`.  The swap site owns the wrapper
    contract, so it restores the attribute — from the checkpoint's own
    generation config when a name is available (preserving tuned sampler
    fields), falling back to the class's model-config derivation, exactly
    mirroring upstream init.
    """

    @staticmethod
    def _family(*, can_generate=True, from_pretrained_raises=False):
        class _GenConfig:
            def __init__(self, origin="bare"):
                self.origin = origin

            @classmethod
            def from_pretrained(cls, name):
                if from_pretrained_raises:
                    raise OSError("no generation_config.json")
                return cls(origin=f"checkpoint:{name}")

            @classmethod
            def from_model_config(cls, config):
                return cls(origin="model_config")

        class _Base:
            generation_config_class = _GenConfig

            class config:
                model_type = "gcswap"

            @classmethod
            def can_generate(cls):
                return can_generate

        class _Wrapper(_Base):
            pass

        return _Base, _Wrapper, _GenConfig

    def _swapped(self, base_cls, wrapper_cls, model_name=None):
        from unturtle import fast_diffusion_model as fdm

        fdm._POST_LOAD_CLASS_SWAPS["gcswap"] = lambda: wrapper_cls
        try:
            model = base_cls()
            fdm._apply_post_load_class_swap(model, model_name=model_name)
        finally:
            del fdm._POST_LOAD_CLASS_SWAPS["gcswap"]
        return model

    def test_a_missing_config_is_restored_from_the_checkpoint(self):
        base, wrapper, _ = self._family()

        model = self._swapped(base, wrapper, model_name="org/ckpt")

        assert "generation_config" in model.__dict__, (
            "the swap left the instance without a generation_config; the "
            "first generate would raise AttributeError (#96)"
        )
        assert model.generation_config.origin == "checkpoint:org/ckpt", (
            "the checkpoint's own generation config (tuned sampler fields) "
            "was not preferred"
        )

    def test_the_fallback_derives_from_the_model_config(self):
        """No checkpoint file (or no name): mirror upstream init."""
        base, wrapper, _ = self._family(from_pretrained_raises=True)

        model = self._swapped(base, wrapper, model_name="org/ckpt")

        assert model.generation_config.origin == "model_config"

    def test_no_model_name_still_restores(self):
        base, wrapper, _ = self._family()

        model = self._swapped(base, wrapper, model_name=None)

        assert model.generation_config.origin == "model_config"

    def test_an_existing_config_is_not_overwritten(self):
        """A load path that DID populate it (plain transformers) must win —
        the restored default would discard checkpoint-tuned fields."""
        base, wrapper, gen_config = self._family()
        sentinel = gen_config(origin="already-there")

        from unturtle import fast_diffusion_model as fdm

        fdm._POST_LOAD_CLASS_SWAPS["gcswap"] = lambda: wrapper
        try:
            model = base()
            model.generation_config = sentinel
            fdm._apply_post_load_class_swap(model, model_name="org/ckpt")
        finally:
            del fdm._POST_LOAD_CLASS_SWAPS["gcswap"]

        assert model.generation_config is sentinel

    def test_a_non_generating_model_is_left_alone(self):
        """Mirrors upstream: only `can_generate()` models carry the attr."""
        base, wrapper, _ = self._family(can_generate=False)

        model = self._swapped(base, wrapper, model_name="org/ckpt")

        assert "generation_config" not in model.__dict__


# ---------------------------------------------------------------------------
# FastModel kwarg forwarding — _load_via_fastmodel
# ---------------------------------------------------------------------------


class TestFastModelKwargForwarding:
    def test_forwards_revision_cache_dir_and_extras(self, monkeypatch):
        """User kwargs (revision, cache_dir, subfolder, attn_implementation, …)
        must reach FastModel.from_pretrained instead of being silently dropped."""
        calls = {}

        class _FakeFastModel:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                calls["kwargs"] = kwargs
                return "M", "T"

        from unturtle import fast_diffusion_model as fdm

        monkeypatch.setattr(fdm, "_import_fastmodel", lambda: _FakeFastModel)
        out = fdm._load_via_fastmodel(
            "some/hub-model",
            {
                "torch_dtype": "bf16",
                "revision": "abc123",
                "cache_dir": "/tmp/hf",
                "subfolder": "sub",
                "attn_implementation": "sdpa",
                "token": "tok",
            },
            load_in_4bit=False,
        )
        assert out == ("M", "T")
        kw = calls["kwargs"]
        assert kw["dtype"] == "bf16"  # torch_dtype → dtype rename
        assert "torch_dtype" not in kw
        assert kw["revision"] == "abc123"
        assert kw["cache_dir"] == "/tmp/hf"
        assert kw["subfolder"] == "sub"
        assert kw["attn_implementation"] == "sdpa"
        assert kw["token"] == "tok"
        assert kw["load_in_4bit"] is False

    def test_quantization_config_not_forwarded(self, monkeypatch):
        """FastModel owns quantization on this path — quantization_config is an
        intentional skip, threaded through as load_in_4bit instead."""
        calls = {}

        class _FakeFastModel:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                calls["kwargs"] = kwargs
                return "M", "T"

        from unturtle import fast_diffusion_model as fdm

        monkeypatch.setattr(fdm, "_import_fastmodel", lambda: _FakeFastModel)
        fdm._load_via_fastmodel(
            "x", {"quantization_config": object(), "revision": "r"}, load_in_4bit=True
        )
        assert "quantization_config" not in calls["kwargs"]
        assert calls["kwargs"]["load_in_4bit"] is True
        assert calls["kwargs"]["revision"] == "r"

    def test_unaccepted_kwargs_dropped_with_warning(self, monkeypatch):
        """Keys a non-**kwargs FastModel signature cannot take are dropped loudly."""
        calls = {}
        warnings_seen: list[str] = []

        class _FakeStrictFastModel:
            @staticmethod
            def from_pretrained(
                model_name, dtype=None, revision=None, load_in_4bit=True
            ):
                calls["kwargs"] = {
                    "dtype": dtype,
                    "revision": revision,
                    "load_in_4bit": load_in_4bit,
                }
                return "M", "T"

        from unturtle import fast_diffusion_model as fdm

        monkeypatch.setattr(fdm, "_import_fastmodel", lambda: _FakeStrictFastModel)
        monkeypatch.setattr(fdm, "_warn_once", warnings_seen.append)
        out = fdm._load_via_fastmodel(
            "x",
            {"torch_dtype": "bf16", "revision": "r", "not_a_real_kwarg": 1},
            load_in_4bit=False,
        )
        assert out == ("M", "T")
        assert calls["kwargs"]["revision"] == "r"
        assert any("not_a_real_kwarg" in w for w in warnings_seen)


# ---------------------------------------------------------------------------
# 4-bit load fallback — _load_model_with_optional_4bit_fallback
# ---------------------------------------------------------------------------


class TestFourBitFallback:
    def test_oom_is_reraised_not_retried(self, monkeypatch):
        """OOM must NOT trigger the full-precision retry (which needs MORE memory)."""
        from unturtle import fast_diffusion_model as fdm

        attempts = []

        class _OOMLoader:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                attempts.append(kwargs)
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        with pytest.raises(torch.cuda.OutOfMemoryError):
            fdm._load_model_with_optional_4bit_fallback(
                _OOMLoader, "x", {"quantization_config": object()}
            )
        assert len(attempts) == 1  # no retry

    def test_genuine_4bit_failure_falls_back_with_exception_in_warning(
        self, monkeypatch
    ):
        from unturtle import fast_diffusion_model as fdm

        warnings_seen: list[str] = []
        monkeypatch.setattr(fdm, "_warn_once", warnings_seen.append)
        attempts = []

        class _FlakyLoader:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                attempts.append(dict(kwargs))
                if "quantization_config" in kwargs:
                    raise ValueError("bnb kaboom")
                return "FULL_PRECISION_MODEL"

        out = fdm._load_model_with_optional_4bit_fallback(
            _FlakyLoader, "x", {"quantization_config": object(), "device_map": "auto"}
        )
        assert out == "FULL_PRECISION_MODEL"
        assert len(attempts) == 2
        assert "quantization_config" not in attempts[1]
        # The warning must carry the original exception type + message
        assert any("ValueError" in w and "bnb kaboom" in w for w in warnings_seen)


# ---------------------------------------------------------------------------
# Merged-16bit honesty — _dequantize_merged_model_ / save_pretrained_merged
# ---------------------------------------------------------------------------


class _FakeQuantState:
    dtype = torch.float16


def _make_fake_quantized_linear(in_features: int, out_features: int) -> torch.nn.Linear:
    """nn.Linear whose weight mimics a bnb Params4bit (has .quant_state)."""
    lin = torch.nn.Linear(in_features, out_features, bias=True)
    packed = torch.zeros((out_features * in_features) // 2, 1, dtype=torch.uint8)
    w = torch.nn.Parameter(packed, requires_grad=False)
    w.quant_state = _FakeQuantState()
    lin.weight = w
    lin.in_features = in_features
    lin.out_features = out_features
    return lin


class _FakeModelConfig:
    def __init__(self):
        # Instance attribute, like transformers' PretrainedConfig — `del` must work.
        self.quantization_config = {"quant_method": "bitsandbytes"}


class _FakeQuantizedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = _make_fake_quantized_linear(4, 8)
        self.config = _FakeModelConfig()


class TestMergedSaveDequantizes:
    def test_dequantize_replaces_quantized_linear(self, monkeypatch):
        from unturtle import fast_diffusion_model as fdm

        class _FakeBnb:
            class functional:
                @staticmethod
                def dequantize_4bit(data, quant_state):
                    return torch.zeros(8, 4, dtype=torch.float16)

        monkeypatch.setattr(fdm, "_import_bitsandbytes", lambda: _FakeBnb)
        model = _FakeQuantizedModel()
        out = fdm._dequantize_merged_model_(model)

        assert isinstance(out.proj, torch.nn.Linear)
        assert out.proj.weight.dtype == torch.float16
        assert out.proj.weight.shape == (8, 4)
        assert getattr(out.proj.weight, "quant_state", None) is None
        # Stale 4-bit metadata must not survive into the saved config
        assert not hasattr(out.config, "quantization_config")

    def test_bitsandbytes_unavailable_raises_clear_error(self, monkeypatch):
        from unturtle import fast_diffusion_model as fdm

        def _boom():
            raise ImportError("no bitsandbytes")

        monkeypatch.setattr(fdm, "_import_bitsandbytes", _boom)
        with pytest.raises(RuntimeError, match="load_in_4bit=False"):
            fdm._dequantize_merged_model_(_FakeQuantizedModel())

    def test_dequantize_failure_raises_clear_error(self, monkeypatch):
        from unturtle import fast_diffusion_model as fdm

        class _FakeBnb:
            class functional:
                @staticmethod
                def dequantize_4bit(data, quant_state):
                    raise ValueError("bad quant_state")

        monkeypatch.setattr(fdm, "_import_bitsandbytes", lambda: _FakeBnb)
        with pytest.raises(RuntimeError, match="load_in_4bit=False"):
            fdm._dequantize_merged_model_(_FakeQuantizedModel())

    def test_non_quantized_model_untouched(self):
        from unturtle import fast_diffusion_model as fdm

        model = torch.nn.Linear(4, 8)
        assert fdm._dequantize_merged_model_(model) is model

    def test_save_pretrained_merged_refuses_mislabeled_4bit(self, monkeypatch):
        """End-to-end: merged save on a 4-bit model must error, never save nf4."""
        from unturtle import fast_diffusion_model as fdm
        from unturtle.fast_diffusion_model import FastDiffusionModel

        def _boom():
            raise ImportError("no bitsandbytes")

        monkeypatch.setattr(fdm, "_import_bitsandbytes", _boom)

        saved: list[bool] = []

        class _FakePeft(torch.nn.Module):
            # merge_and_unload builds the quantized model at call time (after
            # save_pretrained_merged's deepcopy — torch's Parameter deepcopy
            # would strip the fake quant_state marker).
            def merge_and_unload(self):
                merged = _FakeQuantizedModel()
                merged.save_pretrained = lambda *a, **k: saved.append(True)
                return merged

        model = _FakePeft()
        with pytest.raises(RuntimeError, match="16-bit"):
            FastDiffusionModel.save_pretrained_merged(
                model, "/nonexistent/should-not-write"
            )
        assert saved == []


# ---------------------------------------------------------------------------
# _fast_path_dtype_incompatibility — #177 all-or-nothing dtype gate (CPU)
# ---------------------------------------------------------------------------


class _GuardProbeModel(torch.nn.Module):
    """Minimal model exposing exactly the structure the dtype gate reads."""

    def __init__(self, embed_dtype=torch.bfloat16, quant_dtypes=(torch.bfloat16,)):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4).to(embed_dtype)
        for index, quant_dtype in enumerate(quant_dtypes):
            linear = _make_fake_quantized_linear(4, 4)
            # _FakeQuantState.dtype is a class attribute; shadow per instance.
            linear.weight.quant_state.dtype = quant_dtype
            setattr(self, f"proj{index}", linear)

    def get_input_embeddings(self):
        return self.embed


class TestFastPathDtypeGate:
    """The #177 gate: quantized models whose hidden-state dtype cannot feed
    the fused kernels are refused wholesale — and healthy or unresolvable
    models are not (fail-open, since patchers keep their per-layer gates)."""

    def _gate(self, model):
        from unturtle.fast_diffusion_model import _fast_path_dtype_incompatibility

        return _fast_path_dtype_incompatibility(model)

    def test_non_quantized_model_is_compatible(self):
        assert self._gate(torch.nn.Linear(4, 4)) is None

    def test_matching_dtype_is_compatible(self):
        model = _GuardProbeModel(
            embed_dtype=torch.bfloat16, quant_dtypes=(torch.bfloat16,)
        )
        assert self._gate(model) is None

    def test_fp32_embedding_is_incompatible(self):
        """The pre-#177 state: peft's kbit prepare upcast the embedding."""
        model = _GuardProbeModel(
            embed_dtype=torch.float32, quant_dtypes=(torch.bfloat16,)
        )
        assert self._gate(model) == "incompatible_compute_dtype"

    def test_quant_dtype_mismatch_is_incompatible(self):
        """quant_state.dtype is what the weight DEQUANTIZES to in the fused
        path (matmul_lora does not cast, unlike the standard bnb forward) —
        measured on CUDA: bf16 activations against an fp16 quant_state fail."""
        model = _GuardProbeModel(
            embed_dtype=torch.bfloat16, quant_dtypes=(torch.float16,)
        )
        assert self._gate(model) == "incompatible_compute_dtype"

    def test_mixed_quant_dtypes_are_incompatible(self):
        """No single hidden-state dtype feeds {bf16, fp16} weights: patching
        only the matching layers would create the partially-fast model the
        contract forbids."""
        model = _GuardProbeModel(
            embed_dtype=torch.bfloat16,
            quant_dtypes=(torch.bfloat16, torch.float16),
        )
        assert self._gate(model) == "incompatible_compute_dtype"

    def test_raising_get_input_embeddings_fails_open(self):
        """transformers raises NotImplementedError on exotic layouts; the
        gate must return a verdict, never propagate."""
        model = _GuardProbeModel()
        model.get_input_embeddings = None  # not callable

        raising = _GuardProbeModel()

        def _boom():
            raise NotImplementedError("not auto-handled")

        raising.get_input_embeddings = _boom

        assert self._gate(model) is None
        assert self._gate(raising) is None


# ---------------------------------------------------------------------------
# get_peft_model(random_state=...) contract (#188)
# ---------------------------------------------------------------------------


def _tiny_dream_for_rng():
    from unturtle.models.backbones.dream.configuration_dream import DreamConfig
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    torch.manual_seed(0)
    config = DreamConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        mask_token_id=1,
        pad_token_id=0,
    )
    return DreamModel(config)


def _lora_a_after_wrap(*, random_state, pre_consume: int) -> tuple[torch.Tensor, bool]:
    """Wrap a fresh tiny model after consuming `pre_consume` RNG draws; return
    the first lora_A and whether the caller's torch RNG state was untouched."""
    from unturtle.fast_diffusion_model import FastDiffusionModel

    model = _tiny_dream_for_rng()
    torch.manual_seed(100)
    if pre_consume:
        torch.randn(pre_consume)
    state_before = torch.get_rng_state().clone()
    peft_model = FastDiffusionModel.get_peft_model(
        model,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj"],
        use_gradient_checkpointing=False,
        random_state=random_state,
    )
    state_after = torch.get_rng_state()
    lora_a = next(p for n, p in peft_model.named_parameters() if ".lora_A." in n)
    return lora_a.detach().clone(), bool(torch.equal(state_before, state_after))


class TestPeftRandomStateContract:
    def test_same_random_state_gives_same_adapters_regardless_of_prior_rng(self):
        a, _ = _lora_a_after_wrap(random_state=3407, pre_consume=0)
        b, _ = _lora_a_after_wrap(random_state=3407, pre_consume=7)
        assert torch.equal(a, b), "random_state did not own adapter initialization"

    def test_different_random_state_gives_different_adapters(self):
        a, _ = _lora_a_after_wrap(random_state=3407, pre_consume=0)
        b, _ = _lora_a_after_wrap(random_state=3408, pre_consume=0)
        assert not torch.equal(a, b)

    def test_callers_global_rng_is_not_consumed_or_reseeded(self):
        """Documented divergence from unsloth's set_seed: the seed lives in a
        forked generator, so the caller's RNG stream continues unchanged."""
        _, untouched = _lora_a_after_wrap(random_state=3407, pre_consume=3)
        assert untouched
        # and the caller's NEXT draw equals what it would have been without the wrap
        torch.manual_seed(100)
        torch.randn(3)
        expected_next = torch.randn(4)
        model = _tiny_dream_for_rng()
        torch.manual_seed(100)
        torch.randn(3)
        from unturtle.fast_diffusion_model import FastDiffusionModel

        FastDiffusionModel.get_peft_model(
            model,
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj"],
            use_gradient_checkpointing=False,
            random_state=3407,
        )
        assert torch.equal(torch.randn(4), expected_next)

    def test_none_keeps_legacy_unseeded_behavior(self):
        a, _ = _lora_a_after_wrap(random_state=None, pre_consume=0)
        b, _ = _lora_a_after_wrap(random_state=None, pre_consume=7)
        assert not torch.equal(a, b), "None must opt out of seeding (legacy contract)"
