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
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaConfig

        return TinyA2DLlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )

    def test_config_model_type(self, config):
        assert config.model_type == "tiny-a2d-llama"

    def test_config_inherits_llama_config(self, config):
        import transformers

        assert isinstance(config, transformers.LlamaConfig)

    def test_model_instantiation(self, config):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaLMHeadModel

        model = TinyA2DLlamaLMHeadModel(config)
        assert model is not None
        assert hasattr(model, "lm_head")

    def test_forward_logits_shape(self, config):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaLMHeadModel

        model = TinyA2DLlamaLMHeadModel(config)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_autoconfig_registered(self):
        import transformers

        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,  # ensure registration
        )

        assert (
            "tiny-a2d-llama"
            in transformers.models.auto.configuration_auto.CONFIG_MAPPING
        )


# ---------------------------------------------------------------------------
# A2D-Qwen2
# ---------------------------------------------------------------------------


class TestA2DQwen2:
    @pytest.fixture
    def config(self):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DQwen2Config

        return TinyA2DQwen2Config(
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
        assert config.model_type == "tiny-a2d-qwen2"

    def test_config_inherits_qwen2_config(self, config):
        import transformers

        assert isinstance(config, transformers.Qwen2Config)

    def test_model_instantiation(self, config):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DQwen2LMHeadModel

        model = TinyA2DQwen2LMHeadModel(config)
        assert model is not None

    def test_forward_logits_shape(self, config):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DQwen2LMHeadModel

        model = TinyA2DQwen2LMHeadModel(config)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_autoconfig_registered(self):
        import transformers

        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DQwen2Config,  # ensure registration
        )

        assert (
            "tiny-a2d-qwen2"
            in transformers.models.auto.configuration_auto.CONFIG_MAPPING
        )


# ---------------------------------------------------------------------------
# A2D-Qwen3
# ---------------------------------------------------------------------------


class TestA2DQwen3:
    @pytest.fixture
    def config(self):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DQwen3Config

        return TinyA2DQwen3Config(
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
        assert config.model_type == "tiny-a2d-qwen3"

    def test_config_inherits_qwen3_config(self, config):
        import transformers

        assert isinstance(config, transformers.Qwen3Config)

    def test_model_instantiation(self, config):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DQwen3LMHeadModel

        model = TinyA2DQwen3LMHeadModel(config)
        assert model is not None

    def test_forward_logits_shape(self, config):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DQwen3LMHeadModel

        model = TinyA2DQwen3LMHeadModel(config)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_autoconfig_registered(self):
        import transformers

        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DQwen3Config,  # ensure registration
        )

        assert (
            "tiny-a2d-qwen3"
            in transformers.models.auto.configuration_auto.CONFIG_MAPPING
        )


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

    @pytest.mark.parametrize(
        "model_cls,config_cls,model_type",
        [
            ("TinyA2DLlamaLMHeadModel", "TinyA2DLlamaConfig", "tiny-a2d-llama"),
            ("TinyA2DQwen2LMHeadModel", "TinyA2DQwen2Config", "tiny-a2d-qwen2"),
            ("TinyA2DQwen3LMHeadModel", "TinyA2DQwen3Config", "tiny-a2d-qwen3"),
        ],
    )
    def test_attends_to_future_tokens(self, model_cls, config_cls, model_type):
        import importlib

        from unturtle.models.conversion.a2d import tiny_a2d as a2d_module

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
        if model_type == "tiny-a2d-qwen3":
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
            f"{model_type}: position-0 output is identical after changing position {L - 1}. "
            "Model appears to be causal — check that TinyA2DModel.forward uses a non-causal mask."
        )


# ---------------------------------------------------------------------------
# Packed sequence integration
# ---------------------------------------------------------------------------


class TestA2DPackedForward:
    """Verify that packed_seq_lengths propagates through TinyA2DLlamaLMHeadModel forward."""

    @pytest.fixture
    def tiny_config(self):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaConfig

        return TinyA2DLlamaConfig(
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
        from unturtle.fast_diffusion_model import _install_apply_stubs
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaLMHeadModel

        model = TinyA2DLlamaLMHeadModel(tiny_config)
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
        _seq_lengths_list = [
            torch.tensor([6, 6], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
        ]
        real_counts = [12, 10]
        total_tokens = 22

        Q_t = torch.randn(B, L, n_heads, head_dim)
        compact = torch.cat([Q_t[b, : real_counts[b]] for b in range(B)], dim=0)

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
            out_full[b, :rc] = fake_out[offset : offset + rc].reshape(
                rc, n_heads * head_dim
            )
            offset += rc

        assert torch.all(out_full[0, 12:] == 0), "Row 0 padding must be zero"
        assert torch.all(out_full[1, 10:] == 0), "Row 1 padding must be zero"

        offset = 0
        for b in range(B):
            rc = real_counts[b]
            expected = fake_out[offset : offset + rc].reshape(rc, n_heads * head_dim)
            assert torch.allclose(out_full[b, :rc], expected), (
                f"Row {b} values mismatch"
            )
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

        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            _flash_varlen_packed,
        )

        B, n_heads, L, head_dim = 2, 4, 16, 8
        n_kv_heads = n_heads
        device = "cuda"

        seq_lengths_list = [
            torch.tensor([6, 6], dtype=torch.int32),  # row0: 12 real, 4 padding
            torch.tensor([10], dtype=torch.int32),  # row1: 10 real, 6 padding
        ]

        torch.manual_seed(42)
        Q = torch.randn(B, n_heads, L, head_dim, device=device, dtype=torch.bfloat16)
        K = torch.randn(B, n_kv_heads, L, head_dim, device=device, dtype=torch.bfloat16)
        V = torch.randn(B, n_kv_heads, L, head_dim, device=device, dtype=torch.bfloat16)

        out = _flash_varlen_packed(
            Q,
            K,
            V,
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
        V_fwd[0, :, 5, :] = torch.randn(
            n_kv_heads, head_dim, device=device, dtype=torch.bfloat16
        )
        out_fwd = _flash_varlen_packed(
            Q,
            K,
            V_fwd,
            seq_lengths_list=seq_lengths_list,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        # Position 0 (sample0) attends to position 5 (same sample) → output changes
        assert not torch.allclose(
            out[0, 0].float(), out_fwd[0, 0].float(), atol=1e-3
        ), (
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
    """Tests for TinyA2DGenerationMixin.diffusion_generate on tiny CPU models."""

    MASK_TOKEN_ID = 999

    @pytest.fixture
    def llama_config(self):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaConfig

        return TinyA2DLlamaConfig(
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
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaLMHeadModel

        model = TinyA2DLlamaLMHeadModel(llama_config).eval()
        return model

    def test_has_diffusion_generate(self, llama_model):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DGenerationMixin

        assert isinstance(llama_model, TinyA2DGenerationMixin)
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
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionModelOutput,
        )

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

    def test_generate_accepts_algorithm_kwarg(self, llama_model):
        B, L_prompt, L_new = 1, 4, 4
        L_total = L_prompt + L_new
        prompt_ids = torch.tensor([[1, 2, 3, 4]])
        mask_fill = torch.full((B, L_new), self.MASK_TOKEN_ID, dtype=torch.long)
        input_ids_full = torch.cat([prompt_ids, mask_fill], dim=1)
        with torch.no_grad():
            out = llama_model.generate(
                input_ids_full,
                algorithm="mdlm",
                steps=3,
                mask_token_id=self.MASK_TOKEN_ID,
                max_length=L_total + 1,
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape == (B, L_total + 1)

    def test_generate_auto_matches_block_decode(self, llama_model):
        B, L_prompt, L_new = 1, 4, 4
        L_total = L_prompt + L_new
        prompt_ids = torch.tensor([[1, 2, 3, 4]])
        mask_fill = torch.full((B, L_new), self.MASK_TOKEN_ID, dtype=torch.long)
        input_ids_full = torch.cat([prompt_ids, mask_fill], dim=1)
        gen = dict(steps=3, mask_token_id=self.MASK_TOKEN_ID, max_length=L_total + 1)

        torch.manual_seed(0)
        out_auto = llama_model.generate(input_ids_full, **gen)
        torch.manual_seed(0)
        out_block = llama_model.generate(
            input_ids_full, algorithm="block_decode", **gen
        )

        s_auto = out_auto.sequences if hasattr(out_auto, "sequences") else out_auto
        s_block = out_block.sequences if hasattr(out_block, "sequences") else out_block
        assert torch.equal(s_auto, s_block)

    def test_generate_auto_with_use_block_diffusion_resolves_bd3lm(self, llama_model):
        # auto + use_block_diffusion=True must follow the same path as explicit bd3lm
        prompt = torch.tensor([[1, 2, 3, 4]])
        gen = dict(
            use_block_diffusion=True,
            bd3lm_block_size=4,
            max_new_tokens=4,
            steps=2,
            mask_token_id=self.MASK_TOKEN_ID,
            pad_token_id=0,
        )

        torch.manual_seed(0)
        out_auto = llama_model.generate(prompt, **gen)
        torch.manual_seed(0)
        out_bd3lm = llama_model.generate(prompt, algorithm="bd3lm", **gen)

        s_auto = out_auto.sequences if hasattr(out_auto, "sequences") else out_auto
        s_bd3lm = out_bd3lm.sequences if hasattr(out_bd3lm, "sequences") else out_bd3lm
        assert torch.equal(s_auto, s_bd3lm)


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
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            _rotate_half_rope,
        )

        B, n_heads, L, head_dim = 2, 4, 8, 16
        torch.manual_seed(0)
        Q = torch.randn(B, n_heads, L, head_dim)
        K = torch.randn(B, n_heads, L, head_dim)
        cos, sin = _make_cos_sin(B, L, head_dim)

        Q_out, K_out = _rotate_half_rope(Q, K, cos, sin, position_ids=None)

        assert Q_out.shape == Q.shape
        assert K_out.shape == K.shape
        torch.testing.assert_close(
            Q_out.norm(dim=-1),
            Q.norm(dim=-1),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            K_out.norm(dim=-1),
            K.norm(dim=-1),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_l2_norm_preserved_with_position_ids(self):
        """RoPE norm preservation must hold for packed-style repeated position_ids."""
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            _rotate_half_rope,
        )

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
            Q_out.norm(dim=-1),
            Q.norm(dim=-1),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            K_out.norm(dim=-1),
            K.norm(dim=-1),
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cpu_matches_cuda_no_position_ids(self):
        """CPU _rotate_half_rope must match CUDA fast_rope_embedding (no position_ids)."""
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            _rotate_half_rope,
        )

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
        cos_cuda = cos_cpu.cuda().to(torch.bfloat16)  # (1, L, head_dim)
        sin_cuda = sin_cpu.cuda().to(torch.bfloat16)

        Q_out_cuda, K_out_cuda = fast_rope_embedding(Q_cuda, K_cuda, cos_cuda, sin_cuda)

        # Tolerance is relaxed to 1e-2 because CUDA path runs in bfloat16
        torch.testing.assert_close(
            Q_out_cuda.float().cpu(),
            Q_out_cpu,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            K_out_cuda.float().cpu(),
            K_out_cpu,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cpu_matches_cuda_with_position_ids(self):
        """CPU fallback must match CUDA RoPE for packed-style position_ids."""
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            _rotate_half_rope,
        )

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
            Q_cuda,
            K_cuda,
            cos_cuda,
            sin_cuda,
        )

        torch.testing.assert_close(
            Q_out_cuda.float().cpu(),
            Q_out_cpu,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            K_out_cuda.float().cpu(),
            K_out_cpu,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_flattened_preindexed_cuda_path_matches_cpu(self):
        """Flattened pre-indexed cos/sin with flat row indices must match CPU fallback."""
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            _rotate_half_rope,
        )

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
            Q_out_cuda.float().cpu(),
            Q_expected,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            K_out_cuda.float().cpu(),
            K_expected,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_broadcasted_cos_sin_cuda_path_matches_cpu(self):
        """Broadcasted (1, L, D) cos/sin must stay on the non-indexed CUDA path."""
        from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
            _rotate_half_rope,
        )

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
            Q_out_cuda.float().cpu(),
            Q_expected,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            K_out_cuda.float().cpu(),
            K_expected,
            atol=1e-2,
            rtol=1e-2,
        )


# ---------------------------------------------------------------------------
# A2D-ModernBERT
# ---------------------------------------------------------------------------


def _tiny_modernbert_config():
    """Return a minimal A2DModernBertConfig suitable for CPU tests.

    ModernBertConfig defaults include token IDs (e.g. pad_token_id=50283) that
    exceed our tiny vocab_size=1000, so we override them explicitly.
    """
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


class TestA2DModernBert:
    @pytest.fixture
    def config(self):
        return _tiny_modernbert_config()

    def test_config_model_type(self, config):
        assert config.model_type == "modernbert-diffusion"

    def test_config_inherits_modernbert_config(self, config):
        from transformers import ModernBertConfig

        assert isinstance(config, ModernBertConfig)

    def test_model_instantiation(self, config):
        from unturtle.models.backbones.modernbert import (
            A2DModernBertForMaskedLM,
            A2DModernBertModel,
        )

        model = A2DModernBertForMaskedLM(config)
        assert model is not None
        assert isinstance(model.model, A2DModernBertModel)
        assert hasattr(model, "decoder")

    def test_decoder_weight_tied_to_embeddings(self, config):
        """decoder.weight must be tied to tok_embeddings.weight after model swap."""
        from unturtle.models.backbones.modernbert import A2DModernBertForMaskedLM

        model = A2DModernBertForMaskedLM(config)
        assert model.decoder.weight is model.model.embeddings.tok_embeddings.weight, (
            "decoder.weight and tok_embeddings.weight are not the same tensor — "
            "tie_weights() was not called after self.model replacement."
        )

    def test_forward_logits_shape(self, config):
        from unturtle.models.backbones.modernbert import A2DModernBertForMaskedLM

        model = A2DModernBertForMaskedLM(config)
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(3, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_autoconfig_registered(self):
        import transformers

        from unturtle.models.backbones.modernbert import (
            A2DModernBertConfig,  # ensure import still works
        )

        assert (
            "modernbert-diffusion"
            in transformers.models.auto.configuration_auto.CONFIG_MAPPING
        )

    def test_bidirectional_attention(self, config):
        """ModernBERT is already bidirectional — position-0 output changes when last token changes."""
        from unturtle.models.backbones.modernbert import A2DModernBertForMaskedLM

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


class TestA2DBlockDecode:
    """Test block-decode KV cache functionality (Phase M)."""

    @pytest.fixture
    def tiny_config(self):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaConfig

        return TinyA2DLlamaConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=128,
        )

    @pytest.fixture
    def tiny_model(self, tiny_config):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaLMHeadModel

        model = TinyA2DLlamaLMHeadModel(tiny_config)
        model.eval()
        return model

    def test_block_decode_baseline(self, tiny_model, tiny_config):
        """Phase M.1: Block-decode with tuple cache trimming runs without error."""
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        # Generate short sequence with block-decode
        input_ids = torch.randint(3, tiny_config.vocab_size, (1, 4))
        mask_token_id = 1
        gen_config = MaskedDiffusionGenerationConfig(
            max_new_tokens=8,
            steps=2,
            alg="origin",
            use_cache=True,
            block_length=4,
            mask_token_id=mask_token_id,
        )

        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=input_ids, generation_config=gen_config
            )

        # Basic shape check
        assert output.shape[0] == 1
        assert output.shape[1] == 4 + 8  # prompt + generated

    def test_block_decode_correctness(self, tiny_model, tiny_config):
        """Phase M.1: Block-decode generates valid (non-mask) tokens in the generated region.

        Block-decode uses a trimmed KV-cache so attended context differs from the full
        no-cache forward pass in a bidirectional model — exact value equivalence with the
        no-cache baseline is not guaranteed.  We verify output correctness instead:
        correct shape, prompt preserved, no remaining mask tokens.
        """
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        torch.manual_seed(42)
        input_ids = torch.randint(3, tiny_config.vocab_size, (1, 4))
        mask_token_id = 1
        prompt_len = input_ids.shape[1]
        max_new = 8

        gen_config_block = MaskedDiffusionGenerationConfig(
            max_new_tokens=max_new,
            steps=4,
            alg="origin",
            use_cache=True,
            block_length=4,
            mask_token_id=mask_token_id,
            temperature=0.0,
        )
        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=input_ids, generation_config=gen_config_block
            )

        assert output.shape == (1, prompt_len + max_new)
        assert torch.equal(output[:, :prompt_len], input_ids)
        assert not torch.any(output[:, prompt_len:] == mask_token_id), (
            "Block-decode should produce no remaining mask tokens in generated region"
        )

    def test_block_decode_disables_replace_cache_for_a2d(
        self, tiny_model, tiny_config, monkeypatch
    ):
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        input_ids = torch.randint(3, tiny_config.vocab_size, (1, 4))
        gen_config = MaskedDiffusionGenerationConfig(
            max_new_tokens=8,
            steps=2,
            alg="origin",
            use_cache=True,
            block_length=4,
            mask_token_id=1,
            use_replace_cache=True,
        )
        captured = {}

        def fake_block_decode_loop(*, input_ids, attention_mask, generation_config):
            captured["use_replace_cache"] = generation_config.use_replace_cache
            return torch.cat(
                [
                    input_ids,
                    torch.full(
                        (input_ids.shape[0], 8),
                        7,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )

        monkeypatch.setattr(tiny_model, "_block_decode_loop", fake_block_decode_loop)

        output = tiny_model.diffusion_generate(
            inputs=input_ids, generation_config=gen_config
        )

        assert captured == {"use_replace_cache": False}
        assert output.shape == (1, 12)


# ---------------------------------------------------------------------------
# BD3LM generation config fields
# ---------------------------------------------------------------------------


class TestBD3LMGenerationConfig:
    """Tests for BD3LM-related config fields."""

    def test_new_fields_have_correct_defaults(self):
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        cfg = MaskedDiffusionGenerationConfig()
        assert cfg.use_block_diffusion is False
        assert cfg.bd3lm_block_size == 32
        assert cfg.cfg_scale == 0.0

    def test_fields_settable_via_kwargs(self):
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        cfg = MaskedDiffusionGenerationConfig(
            use_block_diffusion=True,
            bd3lm_block_size=16,
            cfg_scale=1.5,
        )
        assert cfg.use_block_diffusion is True
        assert cfg.bd3lm_block_size == 16
        assert cfg.cfg_scale == 1.5

    def test_use_block_diffusion_and_use_cache_raises(self):
        import pytest

        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            MaskedDiffusionGenerationConfig(
                use_block_diffusion=True,
                use_cache=True,
            )

    def test_use_block_diffusion_and_return_dict_raises(self):
        import pytest

        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        with pytest.raises(ValueError, match="return_dict"):
            MaskedDiffusionGenerationConfig(
                use_block_diffusion=True,
                return_dict=True,
            )


# ---------------------------------------------------------------------------
# BD3LM block-diffusion generation
# ---------------------------------------------------------------------------


class TestA2DBlockDiffusionGeneration:
    """Tests for A2D BD3LM generation via use_block_diffusion=True."""

    MASK_ID = 100
    PAD_ID = 0

    @pytest.fixture
    def tiny_model(self):
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )

        config = TinyA2DLlamaConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
            mask_token_id=self.MASK_ID,
            pad_token_id=self.PAD_ID,
        )
        model = TinyA2DLlamaLMHeadModel(config)
        model.eval()
        return model

    def test_use_block_diffusion_runs(self, tiny_model):
        """BD3LM generation completes and returns correct shape."""
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 8
        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
            )
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, prompt.shape[1] + max_new_tokens)

    def test_no_mask_tokens_in_generated_region(self, tiny_model):
        """Generated region contains no remaining mask tokens."""
        prompt = torch.tensor([[1, 2, 3, 4]])
        padded_prompt_len = 4  # 4 is multiple of block_size=4
        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=4,
                max_new_tokens=8,
                steps=4,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
            )
        generated = out[:, padded_prompt_len:]
        assert not (generated == self.MASK_ID).any(), (
            "Generated region must have no remaining mask tokens"
        )

    def test_cfg_scale_accepted(self, tiny_model):
        """cfg_scale > 0 runs and produces correct shape with no mask tokens."""
        import math

        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 4
        padded_prompt_len = math.ceil(prompt.shape[1] / block_size) * block_size
        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                cfg_scale=1.0,
                temperature=0.0,
            )
        assert out.shape == (1, padded_prompt_len + max_new_tokens)
        assert not (out[:, padded_prompt_len:] == self.MASK_ID).any()

    def test_right_shift_logits_accepted(self, tiny_model):
        """right_shift_logits=True runs and produces correct shape with no mask tokens."""
        import math

        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 4
        padded_prompt_len = math.ceil(prompt.shape[1] / block_size) * block_size
        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                right_shift_logits=True,
                temperature=0.0,
            )
        assert out.shape == (1, padded_prompt_len + max_new_tokens)
        assert not (out[:, padded_prompt_len:] == self.MASK_ID).any()

    def test_non_aligned_prompt_length_does_not_leak_left_padding(self, tiny_model):
        prompt = torch.tensor([[7, 8, 9]])

        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=4,
                max_new_tokens=4,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
            )

        assert out.shape == (1, prompt.shape[1] + 4)
        assert torch.equal(out[:, : prompt.shape[1]], prompt)
        assert not (out[:, prompt.shape[1] :] == self.PAD_ID).any()

    def test_block_diffusion_accepts_max_length_without_max_new_tokens(
        self, tiny_model
    ):
        prompt = torch.tensor([[7, 8, 9, 10]])

        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=4,
                max_length=8,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
            )

        assert out.shape == (1, 8)
        assert torch.equal(out[:, : prompt.shape[1]], prompt)

    def test_block_diffusion_kv_cache_is_reused_from_prefix_only(
        self, tiny_model, monkeypatch
    ):
        from types import SimpleNamespace

        from unturtle.models.generation import (
            masked_diffusion_block_mixin as generation_utils_module,
        )

        class FakeCache:
            def __init__(self, counter=0):
                self.counter = counter

        def fake_snapshot_prefix_cache(past_key_values):
            return ("prefix", past_key_values.counter)

        def fake_rewrap_prefix_cache(past_key_values, device):
            _, counter = past_key_values
            return FakeCache(counter=counter)

        def fake_forward(
            input_ids,
            attention_mask=None,
            position_ids=None,
            use_cache=False,
            past_key_values=None,
            **kwargs,
        ):
            vocab = tiny_model.config.vocab_size
            batch, seq_len = input_ids.shape
            logits = torch.zeros(batch, seq_len, vocab)

            if use_cache:
                logits[..., 11] = 1.0
                return SimpleNamespace(logits=logits, past_key_values=FakeCache())

            if past_key_values is not None:
                token_id = 11 + past_key_values.counter
                logits[..., token_id] = 1.0
                past_key_values.counter += 1
                return SimpleNamespace(logits=logits)

            logits[..., 99] = 1.0
            return SimpleNamespace(logits=logits)

        monkeypatch.setattr(
            generation_utils_module,
            "_snapshot_prefix_cache",
            fake_snapshot_prefix_cache,
        )
        monkeypatch.setattr(
            generation_utils_module, "_rewrap_prefix_cache", fake_rewrap_prefix_cache
        )
        monkeypatch.setattr(tiny_model, "forward", fake_forward)

        prompt = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=2,
                max_new_tokens=2,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
            )

        assert torch.equal(out[:, : prompt.shape[1]], prompt)
        assert torch.equal(out[:, prompt.shape[1] :], torch.tensor([[11, 11]]))

    def test_block_diffusion_rewraps_prefix_cache_without_deepcopy(
        self, tiny_model, monkeypatch
    ):
        from types import SimpleNamespace

        class FakeCache:
            def __deepcopy__(self, memo):
                raise AssertionError(
                    "deepcopy should not be used for prefix cache reuse"
                )

        def fake_forward(
            input_ids,
            attention_mask=None,
            position_ids=None,
            use_cache=False,
            past_key_values=None,
            **kwargs,
        ):
            vocab = tiny_model.config.vocab_size
            batch, seq_len = input_ids.shape
            logits = torch.zeros(batch, seq_len, vocab)
            logits[..., 11] = 1.0
            if use_cache:
                return SimpleNamespace(logits=logits, past_key_values=FakeCache())
            return SimpleNamespace(logits=logits)

        monkeypatch.setattr(tiny_model, "forward", fake_forward)

        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=torch.tensor([[1, 2, 3, 4]]),
                use_block_diffusion=True,
                bd3lm_block_size=2,
                max_new_tokens=2,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
            )

        assert out.shape == (1, 6)

    def test_eos_stops_finished_rows_in_batch(self, tiny_model, monkeypatch):
        from types import SimpleNamespace

        eos_id = 4

        def fake_forward(
            input_ids,
            attention_mask=None,
            position_ids=None,
            use_cache=False,
            past_key_values=None,
            **kwargs,
        ):
            vocab = tiny_model.config.vocab_size
            batch, seq_len = input_ids.shape
            logits = torch.zeros(batch, seq_len, vocab)
            logits[:, -1, 9] = 1.0
            logits[0, -1, eos_id] = 10.0
            if use_cache:
                return SimpleNamespace(logits=logits, past_key_values=object())
            return SimpleNamespace(logits=logits)

        monkeypatch.setattr(tiny_model, "forward", fake_forward)

        prompt = torch.tensor([[1, 2, 3, 10], [5, 6, 7, 8]])
        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=2,
                max_new_tokens=4,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                eos_token_id=eos_id,
                temperature=0.0,
            )

        assert out.shape == (2, 8)
        assert (out[0, 4:6] == eos_id).any()
        first_eos = (out[0] == eos_id).nonzero(as_tuple=True)[0][0].item()
        assert torch.equal(
            out[0, first_eos + 1 :],
            torch.tensor([self.PAD_ID] * (out.shape[1] - first_eos - 1)),
        )
        assert not (out[1, 4:] == eos_id).any()
        assert (out[1, 4:] != self.PAD_ID).any()

    def test_stream_callback_called_each_inner_step(self, tiny_model):
        """stream_callback fires once per inner denoising step across all blocks."""
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 8
        steps = 4
        import math

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = max(1, math.ceil(steps / num_blocks))
        expected_calls = num_blocks * steps_per_block

        stream_calls = []

        def stream_cb(step, total, x):
            stream_calls.append((step, total, x.shape))

        with torch.no_grad():
            tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=steps,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
                stream_callback=stream_cb,
            )

        assert len(stream_calls) == expected_calls
        steps_seen = [s for s, _, _ in stream_calls]
        assert steps_seen == list(range(1, expected_calls + 1))
        totals_seen = {t for _, t, _ in stream_calls}
        assert totals_seen == {expected_calls}

    def test_step_callback_called_each_inner_step(self, tiny_model):
        """step_callback fires once per inner denoising step across all blocks."""
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 4
        steps = 2
        import math

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = max(1, math.ceil(steps / num_blocks))
        expected_calls = num_blocks * steps_per_block

        step_calls = []

        def step_cb(step, total):
            step_calls.append((step, total))

        with torch.no_grad():
            tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=steps,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
                step_callback=step_cb,
            )

        assert len(step_calls) == expected_calls
        steps_seen = [s for s, _ in step_calls]
        assert steps_seen == list(range(1, expected_calls + 1))

    def test_stream_callback_excludes_left_padding_for_non_aligned_prompt(
        self, tiny_model
    ):
        prompt = torch.tensor([[1, 2, 3]])
        stream_shapes = []

        def stream_cb(step, total, x):
            stream_shapes.append(tuple(x.shape))
            assert x.shape[1] >= prompt.shape[1]
            assert torch.equal(x[0, : prompt.shape[1]], prompt[0])

        with torch.no_grad():
            tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=4,
                max_new_tokens=4,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
                stream_callback=stream_cb,
            )

        assert stream_shapes

    def test_callback_errors_do_not_abort_block_diffusion(self, tiny_model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        stream_calls = 0
        step_calls = 0
        expected_calls = 2

        def stream_cb(step, total, x):
            nonlocal stream_calls
            stream_calls += 1
            raise RuntimeError("stream callback boom")

        def step_cb(step, total):
            nonlocal step_calls
            step_calls += 1
            raise RuntimeError("step callback boom")

        with torch.no_grad():
            out = tiny_model.diffusion_generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=4,
                max_new_tokens=4,
                steps=2,
                mask_token_id=self.MASK_ID,
                pad_token_id=self.PAD_ID,
                temperature=0.0,
                stream_callback=stream_cb,
                step_callback=step_cb,
            )

        assert out.shape == (1, 8)
        assert stream_calls == expected_calls
        assert step_calls == expected_calls
