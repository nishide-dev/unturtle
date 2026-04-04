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


# ---------------------------------------------------------------------------
# Packed sequence integration
# ---------------------------------------------------------------------------


class TestA2DPackedForward:
    """Verify that packed_seq_lengths propagates through A2DLlamaLMHeadModel forward."""

    @pytest.fixture
    def tiny_config(self):
        from unturtle.models.a2d import A2DLlamaConfig
        return A2DLlamaConfig(
            vocab_size=200,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
        )

    def test_forward_with_packed_seq_lengths(self, tiny_config):
        """Model forward must complete without error when packed_seq_lengths is passed."""
        from unturtle.models.a2d import A2DLlamaLMHeadModel
        from unturtle.fast_diffusion_model import _install_apply_stubs

        model = A2DLlamaLMHeadModel(tiny_config)
        _install_apply_stubs(model)
        model.eval()

        B, L = 2, 16
        # Simulate two packed rows, each with 2 samples of length 8
        input_ids = torch.randint(4, tiny_config.vocab_size, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
        # Each row contains 2 samples of length 8
        packed_seq_lengths = torch.tensor([8, 8, 8, 8], dtype=torch.int32)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                packed_seq_lengths=packed_seq_lengths,
            )

        assert out.logits.shape == (B, L, tiny_config.vocab_size), (
            f"Unexpected logits shape: {out.logits.shape}"
        )

    def test_get_packed_info_returns_non_none_when_key_present(self, tiny_config):
        """get_packed_info_from_kwargs must return non-None when packed_seq_lengths is in kwargs."""
        from unturtle.utils.packing import get_packed_info_from_kwargs

        packed_seq_lengths = torch.tensor([8, 8], dtype=torch.int32)
        kwargs = {"packed_seq_lengths": packed_seq_lengths}

        result = get_packed_info_from_kwargs(kwargs, device=torch.device("cpu"))
        assert result is not None, (
            "get_packed_info_from_kwargs returned None — "
            "packed_seq_lengths key is present but not recognized"
        )
        lengths, cu_seqlens, max_seqlen = result
        assert lengths.tolist() == [8, 8]
        assert cu_seqlens.tolist() == [0, 8, 16]
        assert max_seqlen == 8


# ---------------------------------------------------------------------------
# Flash varlen compaction helper
# ---------------------------------------------------------------------------


class TestFlashVarlenCompaction:
    """Tests for _flash_varlen_packed: compaction and scatter logic."""

    def test_compaction_token_count_and_values(self):
        """Compact slices have correct total count and preserve token values."""
        B, n_heads, L, head_dim = 2, 4, 16, 8
        # row0: 2 samples × 6 tokens = 12 real; row1: 1 sample × 10 tokens = 10 real
        seq_lengths_list = [
            torch.tensor([6, 6], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
        ]
        real_counts = [12, 10]
        total_tokens = 22

        Q_t = torch.randn(B, L, n_heads, head_dim)
        compact = torch.cat([Q_t[b, :real_counts[b]] for b in range(B)], dim=0)

        assert compact.shape[0] == total_tokens
        assert torch.allclose(compact[:12], Q_t[0, :12])
        assert torch.allclose(compact[12:], Q_t[1, :10])

    def test_scatter_is_inverse_of_compact(self):
        """Scatter back into padded buffer is lossless; padding positions remain zero."""
        B, n_heads, L, head_dim = 2, 4, 16, 8
        real_counts = [12, 10]
        total_tokens = 22

        fake_out = torch.randn(total_tokens, n_heads, head_dim)
        out_full = torch.zeros(B, L, n_heads * head_dim)

        offset = 0
        for b in range(B):
            rc = real_counts[b]
            out_full[b, :rc] = fake_out[offset:offset + rc].reshape(rc, n_heads * head_dim)
            offset += rc

        assert torch.all(out_full[0, 12:] == 0), "Row 0 padding must be zero"
        assert torch.all(out_full[1, 10:] == 0), "Row 1 padding must be zero"

        offset = 0
        for b in range(B):
            rc = real_counts[b]
            expected = fake_out[offset:offset + rc].reshape(rc, n_heads * head_dim)
            assert torch.allclose(out_full[b, :rc], expected), f"Row {b} values mismatch"
            offset += rc

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for Flash Attention"
    )
    def test_flash_varlen_packed_gpu_shape_and_bidirectionality(self):
        """Flash varlen output has correct shape; bidirectionality is preserved."""
        try:
            from flash_attn import flash_attn_varlen_func  # noqa: F401
        except ImportError:
            pytest.skip("flash_attn not installed")

        from unturtle.models.a2d._fast_forward import _flash_varlen_packed

        B, n_heads, L, head_dim = 2, 4, 16, 8
        n_kv_heads = n_heads
        device = "cuda"

        seq_lengths_list = [
            torch.tensor([6, 6], dtype=torch.int32),   # row0: 12 real, 4 padding
            torch.tensor([10], dtype=torch.int32),     # row1: 10 real, 6 padding
        ]

        torch.manual_seed(42)
        Q = torch.randn(B, n_heads, L, head_dim, device=device, dtype=torch.bfloat16)
        K = torch.randn(B, n_kv_heads, L, head_dim, device=device, dtype=torch.bfloat16)
        V = torch.randn(B, n_kv_heads, L, head_dim, device=device, dtype=torch.bfloat16)

        out = _flash_varlen_packed(
            Q, K, V,
            seq_lengths_list=seq_lengths_list,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )

        assert out.shape == (B, L, n_heads * head_dim)
        assert torch.all(out[0, 12:] == 0), "Row 0 padding must be zero"
        assert torch.all(out[1, 10:] == 0), "Row 1 padding must be zero"

        # Bidirectionality test: change V at position 5 (last token of sample0, positions 0-5).
        # Since causal=False, position 0 (same sample) attends to position 5 — output should change.
        # Position 6 (start of sample1, positions 6-11) must NOT change — cross-sample blocked.
        V_fwd = V.clone()
        V_fwd[0, :, 5, :] = torch.randn(n_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        out_fwd = _flash_varlen_packed(
            Q, K, V_fwd,
            seq_lengths_list=seq_lengths_list,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        # Position 0 (sample0) attends to position 5 (same sample) → output changes
        assert not torch.allclose(out[0, 0].float(), out_fwd[0, 0].float(), atol=1e-3), (
            "Position 0 should change when V at position 5 (same sample) changes — "
            "bidirectional attention not working"
        )
        # Position 6 (sample1) does NOT attend to position 5 (sample0) → output unchanged
        assert torch.allclose(out[0, 6].float(), out_fwd[0, 6].float(), atol=1e-3), (
            "Position 6 (sample1) should NOT change when V at position 5 (sample0) changes — "
            "cross-sample attention should be blocked by cu_seqlens"
        )


# ---------------------------------------------------------------------------
# A2D generation (diffusion_generate)
# ---------------------------------------------------------------------------


class TestA2DGeneration:
    """Tests for A2DGenerationMixin.diffusion_generate on tiny CPU models."""

    MASK_TOKEN_ID = 999

    @pytest.fixture
    def llama_config(self):
        from unturtle.models.a2d import A2DLlamaConfig
        return A2DLlamaConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            mask_token_id=self.MASK_TOKEN_ID,
        )

    @pytest.fixture
    def llama_model(self, llama_config):
        from unturtle.models.a2d import A2DLlamaLMHeadModel
        model = A2DLlamaLMHeadModel(llama_config).eval()
        return model

    def test_has_diffusion_generate(self, llama_model):
        from unturtle.models.a2d import A2DGenerationMixin
        assert isinstance(llama_model, A2DGenerationMixin)
        assert callable(llama_model.diffusion_generate)

    def test_output_shape(self, llama_model, llama_config):
        """Output shape should be [B, max_length].

        In dLLM generation the caller pre-fills completion slots with
        mask_token_id and passes the full sequence.  max_length must be
        > input length so we pass max_length explicitly.
        """
        B, L_prompt, L_new = 2, 4, 8
        L_total = L_prompt + L_new
        prompt_ids = torch.randint(0, 100, (B, L_prompt))
        mask_fill = torch.full((B, L_new), self.MASK_TOKEN_ID, dtype=torch.long)
        input_ids_full = torch.cat([prompt_ids, mask_fill], dim=1)
        with torch.no_grad():
            out = llama_model.diffusion_generate(
                input_ids_full,
                steps=3,
                mask_token_id=self.MASK_TOKEN_ID,
                max_length=L_total + 1,  # must be > input_length
            )
        assert out.shape == (B, L_total + 1)

    def test_prompt_tokens_preserved(self, llama_model, llama_config):
        """Prompt tokens (non-mask) must not be changed by generation."""
        B, L_prompt, L_new = 1, 4, 6
        L_total = L_prompt + L_new
        prompt_ids = torch.tensor([[1, 2, 3, 4]])
        mask_fill = torch.full((B, L_new), self.MASK_TOKEN_ID, dtype=torch.long)
        input_ids_full = torch.cat([prompt_ids, mask_fill], dim=1)
        with torch.no_grad():
            out = llama_model.diffusion_generate(
                input_ids_full,
                steps=3,
                mask_token_id=self.MASK_TOKEN_ID,
                max_length=L_total + 1,
            )
        # Original prompt positions were NOT mask tokens → should be preserved
        assert (out[0, :L_prompt] == prompt_ids[0]).all(), (
            "Prompt tokens should not be overwritten by diffusion_generate"
        )

    def test_deterministic_with_seed(self, llama_model, llama_config):
        """Same random seed + same input → identical output (regardless of alg)."""
        B, L = 1, 8
        input_ids = torch.full((B, L), self.MASK_TOKEN_ID, dtype=torch.long)
        with torch.no_grad():
            torch.manual_seed(42)
            out1 = llama_model.diffusion_generate(
                input_ids.clone(),
                steps=2,
                mask_token_id=self.MASK_TOKEN_ID,
                temperature=0.0,
                max_length=L + 1,
            )
            torch.manual_seed(42)
            out2 = llama_model.diffusion_generate(
                input_ids.clone(),
                steps=2,
                mask_token_id=self.MASK_TOKEN_ID,
                temperature=0.0,
                max_length=L + 1,
            )
        assert (out1 == out2).all(), "Same seed must produce identical output"

    def test_num_steps_one(self, llama_model):
        """steps=1 should complete in a single forward pass."""
        B, L = 1, 6
        input_ids = torch.full((B, L), self.MASK_TOKEN_ID, dtype=torch.long)
        with torch.no_grad():
            out = llama_model.diffusion_generate(
                input_ids,
                steps=1,
                mask_token_id=self.MASK_TOKEN_ID,
                max_length=L + 1,
            )
        assert out.shape == (B, L + 1)

    def test_return_dict(self, llama_model):
        """return_dict=True should return MaskedDiffusionModelOutput."""
        from unturtle.models.diffusion_generation_utils import MaskedDiffusionModelOutput
        B, L = 1, 4
        input_ids = torch.full((B, L), self.MASK_TOKEN_ID, dtype=torch.long)
        with torch.no_grad():
            out = llama_model.diffusion_generate(
                input_ids,
                steps=2,
                mask_token_id=self.MASK_TOKEN_ID,
                max_length=L + 1,
                return_dict=True,
            )
        assert isinstance(out, MaskedDiffusionModelOutput)
        assert out.sequences.shape == (B, L + 1)

    def test_maskgit_plus_alg(self, llama_model):
        """maskgit_plus algorithm should run without error."""
        B, L = 1, 6
        input_ids = torch.full((B, L), self.MASK_TOKEN_ID, dtype=torch.long)
        with torch.no_grad():
            out = llama_model.diffusion_generate(
                input_ids,
                steps=3,
                mask_token_id=self.MASK_TOKEN_ID,
                alg="maskgit_plus",
                max_length=L + 1,
            )
        assert out.shape == (B, L + 1)

    def test_num_return_sequences(self, llama_model):
        """num_return_sequences=2 should double the batch dimension."""
        B, L = 1, 6
        input_ids = torch.full((B, L), self.MASK_TOKEN_ID, dtype=torch.long)
        with torch.no_grad():
            out = llama_model.diffusion_generate(
                input_ids,
                steps=2,
                mask_token_id=self.MASK_TOKEN_ID,
                max_length=L + 1,
                num_return_sequences=2,
            )
        assert out.shape == (B * 2, L + 1)

    def test_attention_mask(self, llama_model):
        """Padded attention_mask should be handled without error."""
        B, L = 2, 8
        input_ids = torch.full((B, L), self.MASK_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.ones((B, L), dtype=torch.long)
        attention_mask[1, -2:] = 0  # simulate padding in second sample
        with torch.no_grad():
            out = llama_model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                steps=2,
                mask_token_id=self.MASK_TOKEN_ID,
                max_length=L + 1,
            )
        assert out.shape == (B, L + 1)


# ---------------------------------------------------------------------------
# RoPE unit tests
# ---------------------------------------------------------------------------


def _make_cos_sin(
    B: int,
    L: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build sequential cos/sin of shape (B, L, head_dim)."""
    position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1)
    return _make_cos_sin_from_position_ids(
        position_ids=position_ids,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )



def _make_cos_sin_from_position_ids(
    position_ids: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin of shape (B, L, head_dim) for arbitrary position_ids."""
    theta = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = position_ids[..., None].to(torch.float32) * theta.view(1, 1, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype=dtype).to(device)
    sin = emb.sin().to(dtype=dtype).to(device)
    return cos, sin


class TestA2DRoPE:
    """Unit tests for ``_rotate_half_rope`` (CPU RoPE fallback in A2D fast forward)."""

    def test_l2_norm_preserved_no_position_ids(self):
        """RoPE rotation must be an isometry: per-vector L2 norm is preserved."""
        from unturtle.models.a2d._fast_forward import _rotate_half_rope

        B, n_heads, L, head_dim = 2, 4, 8, 16
        torch.manual_seed(0)
        Q = torch.randn(B, n_heads, L, head_dim)
        K = torch.randn(B, n_heads, L, head_dim)
        cos, sin = _make_cos_sin(B, L, head_dim)

        Q_out, K_out = _rotate_half_rope(Q, K, cos, sin, position_ids=None)

        assert Q_out.shape == Q.shape
        assert K_out.shape == K.shape
        torch.testing.assert_close(
            Q_out.norm(dim=-1), Q.norm(dim=-1), atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            K_out.norm(dim=-1), K.norm(dim=-1), atol=1e-5, rtol=1e-5,
        )

    def test_l2_norm_preserved_with_position_ids(self):
        """RoPE norm preservation must hold for packed-style repeated position_ids."""
        from unturtle.models.a2d._fast_forward import _rotate_half_rope

        B, n_heads, L, head_dim = 2, 4, 8, 16
        torch.manual_seed(1)
        Q = torch.randn(B, n_heads, L, head_dim)
        K = torch.randn(B, n_heads, L, head_dim)
        position_ids: torch.LongTensor = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]],
            dtype=torch.long,
        )
        cos, sin = _make_cos_sin_from_position_ids(position_ids, head_dim)

        Q_out, K_out = _rotate_half_rope(Q, K, cos, sin, position_ids=position_ids)

        torch.testing.assert_close(
            Q_out.norm(dim=-1), Q.norm(dim=-1), atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            K_out.norm(dim=-1), K.norm(dim=-1), atol=1e-5, rtol=1e-5,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cpu_matches_cuda_no_position_ids(self):
        """CPU _rotate_half_rope must match CUDA fast_rope_embedding (no position_ids)."""
        from unturtle.models.a2d._fast_forward import _rotate_half_rope

        try:
            from unturtle.kernels import fast_rope_embedding
        except ImportError:
            pytest.skip("fast_rope_embedding not available")

        B, n_heads, L, head_dim = 1, 4, 8, 32
        torch.manual_seed(2)
        Q_cpu = torch.randn(B, n_heads, L, head_dim)
        K_cpu = torch.randn(B, n_heads, L, head_dim)
        cos_cpu, sin_cpu = _make_cos_sin(B, L, head_dim)

        Q_out_cpu, K_out_cpu = _rotate_half_rope(Q_cpu, K_cpu, cos_cpu, sin_cpu)

        # fast_rope_embedding expects cos/sin as (1, L, head_dim) or (L, head_dim);
        # it calls .squeeze() internally so (1, L, head_dim) → (L, head_dim).
        Q_cuda = Q_cpu.cuda().to(torch.bfloat16)
        K_cuda = K_cpu.cuda().to(torch.bfloat16)
        cos_cuda = cos_cpu.cuda().to(torch.bfloat16)   # (1, L, head_dim)
        sin_cuda = sin_cpu.cuda().to(torch.bfloat16)

        Q_out_cuda, K_out_cuda = fast_rope_embedding(Q_cuda, K_cuda, cos_cuda, sin_cuda)

        # Tolerance is relaxed to 1e-2 because CUDA path runs in bfloat16
        torch.testing.assert_close(
            Q_out_cuda.float().cpu(), Q_out_cpu, atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            K_out_cuda.float().cpu(), K_out_cpu, atol=1e-2, rtol=1e-2,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cpu_matches_cuda_with_position_ids(self):
        """CPU fallback must match CUDA RoPE for packed-style position_ids."""
        from unturtle.models.a2d._fast_forward import _rotate_half_rope

        try:
            from unturtle.kernels import fast_rope_embedding
        except ImportError:
            pytest.skip("fast_rope_embedding not available")

        B, n_heads, L, head_dim = 1, 4, 8, 32
        torch.manual_seed(3)
        Q_cpu = torch.randn(B, n_heads, L, head_dim)
        K_cpu = torch.randn(B, n_heads, L, head_dim)
        position_ids: torch.LongTensor = torch.tensor(
            [[0, 1, 0, 1, 2, 3, 4, 5]], dtype=torch.long
        )
        cos_cpu, sin_cpu = _make_cos_sin_from_position_ids(position_ids, head_dim)

        Q_out_cpu, K_out_cpu = _rotate_half_rope(
            Q_cpu, K_cpu, cos_cpu, sin_cpu, position_ids=position_ids
        )

        Q_cuda = Q_cpu.cuda().to(torch.bfloat16)
        K_cuda = K_cpu.cuda().to(torch.bfloat16)
        cos_cuda = cos_cpu.cuda().to(torch.bfloat16)
        sin_cuda = sin_cpu.cuda().to(torch.bfloat16)

        Q_out_cuda, K_out_cuda = fast_rope_embedding(
            Q_cuda, K_cuda, cos_cuda, sin_cuda,
        )

        torch.testing.assert_close(
            Q_out_cuda.float().cpu(), Q_out_cpu, atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            K_out_cuda.float().cpu(), K_out_cpu, atol=1e-2, rtol=1e-2,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flattened_preindexed_cuda_path_matches_cpu(self):
        """Flattened pre-indexed cos/sin with flat row indices must match CPU fallback."""
        from unturtle.models.a2d._fast_forward import _rotate_half_rope

        try:
            from unturtle.kernels import fast_rope_embedding
        except ImportError:
            pytest.skip("fast_rope_embedding not available")

        B, n_heads, L, head_dim = 2, 4, 8, 32
        torch.manual_seed(4)
        Q_cpu = torch.randn(B, n_heads, L, head_dim)
        K_cpu = torch.randn(B, n_heads, L, head_dim)
        position_ids: torch.LongTensor = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [0, 0, 1, 1, 2, 2, 3, 3]],
            dtype=torch.long,
        )
        cos_cpu, sin_cpu = _make_cos_sin_from_position_ids(position_ids, head_dim)

        Q_expected, K_expected = _rotate_half_rope(
            Q_cpu, K_cpu, cos_cpu, sin_cpu, position_ids=position_ids
        )

        flat_indices = torch.arange(B * L, dtype=torch.long, device="cuda")
        Q_out_cuda, K_out_cuda = fast_rope_embedding(
            Q_cpu.cuda().to(torch.bfloat16),
            K_cpu.cuda().to(torch.bfloat16),
            cos_cpu.reshape(B * L, head_dim).cuda().to(torch.bfloat16),
            sin_cpu.reshape(B * L, head_dim).cuda().to(torch.bfloat16),
            rope_embedding_indices=flat_indices,
        )

        torch.testing.assert_close(
            Q_out_cuda.float().cpu(), Q_expected, atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            K_out_cuda.float().cpu(), K_expected, atol=1e-2, rtol=1e-2,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_broadcasted_cos_sin_cuda_path_matches_cpu(self):
        """Broadcasted (1, L, D) cos/sin must stay on the non-indexed CUDA path."""
        from unturtle.models.a2d._fast_forward import _rotate_half_rope

        try:
            from unturtle.kernels import fast_rope_embedding
        except ImportError:
            pytest.skip("fast_rope_embedding not available")

        B, n_heads, L, head_dim = 2, 4, 8, 32
        torch.manual_seed(5)
        Q_cpu = torch.randn(B, n_heads, L, head_dim)
        K_cpu = torch.randn(B, n_heads, L, head_dim)
        cos_cpu, sin_cpu = _make_cos_sin(1, L, head_dim)

        Q_expected, K_expected = _rotate_half_rope(Q_cpu, K_cpu, cos_cpu, sin_cpu)
        Q_out_cuda, K_out_cuda = fast_rope_embedding(
            Q_cpu.cuda().to(torch.bfloat16),
            K_cpu.cuda().to(torch.bfloat16),
            cos_cpu.cuda().to(torch.bfloat16),
            sin_cpu.cuda().to(torch.bfloat16),
        )

        torch.testing.assert_close(
            Q_out_cuda.float().cpu(), Q_expected, atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            K_out_cuda.float().cpu(), K_expected, atol=1e-2, rtol=1e-2,
        )


# ---------------------------------------------------------------------------
# A2D-ModernBERT
# ---------------------------------------------------------------------------


def _tiny_modernbert_config():
    """Return a minimal A2DModernBertConfig suitable for CPU tests.

    ModernBertConfig defaults include token IDs (e.g. pad_token_id=50283) that
    exceed our tiny vocab_size=1000, so we override them explicitly.
    """
    from unturtle.models.a2d import A2DModernBertConfig

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


class TestA2DModernBert:
    @pytest.fixture
    def config(self):
        return _tiny_modernbert_config()

    def test_config_model_type(self, config):
        assert config.model_type == "a2d-modernbert"

    def test_config_inherits_modernbert_config(self, config):
        from transformers import ModernBertConfig
        assert isinstance(config, ModernBertConfig)

    def test_model_instantiation(self, config):
        from unturtle.models.a2d import A2DModernBertForMaskedLM, A2DModernBertModel
        model = A2DModernBertForMaskedLM(config)
        assert model is not None
        assert isinstance(model.model, A2DModernBertModel)
        assert hasattr(model, "decoder")

    def test_decoder_weight_tied_to_embeddings(self, config):
        """decoder.weight must be tied to tok_embeddings.weight after model swap."""
        from unturtle.models.a2d import A2DModernBertForMaskedLM
        model = A2DModernBertForMaskedLM(config)
        assert model.decoder.weight is model.model.embeddings.tok_embeddings.weight, (
            "decoder.weight and tok_embeddings.weight are not the same tensor — "
            "tie_weights() was not called after self.model replacement."
        )

    def test_forward_logits_shape(self, config):
        from unturtle.models.a2d import A2DModernBertForMaskedLM
        model = A2DModernBertForMaskedLM(config)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(3, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_autoconfig_registered(self):
        import transformers
        from unturtle.models.a2d import A2DModernBertConfig  # ensure registration
        assert "a2d-modernbert" in transformers.models.auto.configuration_auto.CONFIG_MAPPING

    def test_bidirectional_attention(self, config):
        """ModernBERT is already bidirectional — position-0 output changes when last token changes."""
        from unturtle.models.a2d import A2DModernBertForMaskedLM
        model = A2DModernBertForMaskedLM(config)
        model.eval()

        B, L = 1, 8
        ids_a = torch.randint(3, config.vocab_size, (B, L))
        ids_b = ids_a.clone()
        ids_b[0, -1] = (ids_a[0, -1] + 1) % config.vocab_size

        with torch.no_grad():
            out_a = model(input_ids=ids_a).logits
            out_b = model(input_ids=ids_b).logits

        assert not torch.allclose(out_a[:, 0, :], out_b[:, 0, :]), (
            "a2d-modernbert: position-0 output is identical after changing position L-1. "
            "Bidirectional attention is not working."
        )
