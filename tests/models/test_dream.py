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


class TestDreamFastRoPE:
    """Tests for DreamAttention_fast_forward Triton RoPE path."""

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

    def test_fast_forward_importable(self):
        from unturtle.models.dream.modeling_dream import DreamAttention_fast_forward
        assert callable(DreamAttention_fast_forward)

    def test_cpu_parity_simple(self, config):
        """DreamAttention_fast_forward CPU fallback matches original forward."""
        import types
        from unturtle.models.dream import DreamModel
        from unturtle.models.dream.modeling_dream import DreamAttention_fast_forward

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()

        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))

        # Reference output (original forward)
        with torch.no_grad():
            ref_out = model(input_ids=input_ids)

        # Install stubs and fast forward
        for module in model.modules():
            if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
                if not hasattr(module, "apply_qkv"):
                    from unturtle.fast_diffusion_model import _original_apply_qkv, _original_apply_o
                    module.apply_qkv = _original_apply_qkv
                    module.apply_o = _original_apply_o

        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(
                DreamAttention_fast_forward, layer.self_attn
            )

        with torch.no_grad():
            fast_out = model(input_ids=input_ids)

        assert torch.allclose(ref_out.logits, fast_out.logits, atol=1e-5), (
            f"CPU logits mismatch: max_diff={( ref_out.logits - fast_out.logits).abs().max().item():.2e}"
        )

    def test_cpu_parity_reset_position_ids(self, config):
        """CPU fallback is numerically stable with non-monotonic position_ids (packed pattern)."""
        import types
        from unturtle.models.dream import DreamModel
        from unturtle.models.dream.modeling_dream import DreamAttention_fast_forward, _apply_dream_rope

        torch.manual_seed(1)
        model = DreamModel(config).cpu().eval()

        # Install stubs
        for module in model.modules():
            if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
                if not hasattr(module, "apply_qkv"):
                    from unturtle.fast_diffusion_model import _original_apply_qkv, _original_apply_o
                    module.apply_qkv = _original_apply_qkv
                    module.apply_o = _original_apply_o

        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(
                DreamAttention_fast_forward, layer.self_attn
            )

        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        # Reset position_ids: row 0 = [0..7, 0..7], row 1 = [0..15]
        position_ids = torch.cat([
            torch.arange(L // 2).repeat(2).unsqueeze(0),
            torch.arange(L).unsqueeze(0),
        ], dim=0)

        with torch.no_grad():
            out = model(input_ids=input_ids, position_ids=position_ids)
        assert out.logits.shape == (B, L, config.vocab_size)
        assert not torch.isnan(out.logits).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton RoPE requires CUDA")
    def test_cuda_parity_vs_cpu(self, config):
        """CUDA fast_rope_embedding output matches CPU apply_rotary_pos_emb.

        DreamRotaryEmbedding returns cos/sin via cat(freqs, freqs) so the
        first and second halves are identical — this is the format that
        fast_rope_embedding requires (it only reads head_dim//2 elements).
        """
        from unturtle.models.dream.modeling_dream import _apply_dream_rope

        torch.manual_seed(42)
        B, n_heads, L, head_dim = 2, 4, 16, 32

        # Simulate DreamRotaryEmbedding output: cat(freqs, freqs) pattern
        freqs = torch.randn(B, L, head_dim // 2)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (B, L, head_dim)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

        q = torch.randn(B, n_heads, L, head_dim)
        k = torch.randn(B, n_heads, L, head_dim)

        # CPU reference (clone: fast_rope_embedding is in-place on CUDA)
        q_cpu, k_cpu = _apply_dream_rope(q.clone(), k.clone(), cos, sin, B, L)

        # CUDA fast path
        q_cuda, k_cuda = _apply_dream_rope(
            q.cuda().clone(), k.cuda().clone(), cos.cuda(), sin.cuda(), B, L
        )

        assert torch.allclose(q_cpu, q_cuda.cpu(), atol=1e-4), (
            f"Q mismatch: max_diff={(q_cpu - q_cuda.cpu()).abs().max().item():.2e}"
        )
        assert torch.allclose(k_cpu, k_cuda.cpu(), atol=1e-4), (
            f"K mismatch: max_diff={(k_cpu - k_cuda.cpu()).abs().max().item():.2e}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton RoPE requires CUDA")
    def test_cuda_no_double_index_reset_positions(self, config):
        """CUDA path with reset position_ids produces valid output (no NaN, correct shape).

        cos/sin use the cat(freqs, freqs) pattern from DreamRotaryEmbedding
        and are pre-indexed per batch row to simulate packed/reset positions.
        """
        from unturtle.models.dream.modeling_dream import _apply_dream_rope

        torch.manual_seed(7)
        B, n_heads, L, head_dim = 2, 4, 8, 32

        q = torch.randn(B, n_heads, L, head_dim).cuda()
        k = torch.randn(B, n_heads, L, head_dim).cuda()

        # DreamRotaryEmbedding-style cos/sin: cat(freqs, freqs) pattern
        freqs = torch.randn(B, L, head_dim // 2).cuda()
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

        # Clone before passing to each path (fast_rope_embedding is in-place)
        q_cuda_in = q.clone()
        k_cuda_in = k.clone()
        q_cpu_in = q.cpu().clone()
        k_cpu_in = k.cpu().clone()

        with torch.no_grad():
            q_out, k_out = _apply_dream_rope(q_cuda_in, k_cuda_in, cos, sin, B, L)
            q_cpu, k_cpu = _apply_dream_rope(q_cpu_in, k_cpu_in, cos.cpu(), sin.cpu(), B, L)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()

        assert torch.allclose(q_cpu, q_out.cpu(), atol=1e-4), (
            f"CUDA/CPU parity failed: max_diff={(q_cpu - q_out.cpu()).abs().max().item():.2e}"
        )
