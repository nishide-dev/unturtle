"""Tests for LLaDA models.

CPU-only tests covering config instantiation, model instantiation with
random weights, and forward pass shapes. No pretrained checkpoints required.
"""

from __future__ import annotations

import pytest
import torch


class TestLLaDAConfig:
    def test_config_default_fields(self):
        from unturtle.models.backbones.llada import LLaDAConfig

        config = LLaDAConfig()
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_layers == 12
        assert config.vocab_size == 50257

    def test_config_custom_values(self):
        from unturtle.models.backbones.llada import LLaDAConfig

        config = LLaDAConfig(d_model=128, n_heads=4, n_layers=2, vocab_size=1000)
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 2
        assert config.vocab_size == 1000

    def test_config_hf_properties(self):
        """HF-compatible properties map to internal fields."""
        from unturtle.models.backbones.llada import LLaDAConfig

        config = LLaDAConfig(d_model=256, n_heads=8, n_layers=4)
        assert config.hidden_size == 256
        assert config.num_attention_heads == 8
        assert config.num_hidden_layers == 4

    def test_config_has_mask_token_id(self):
        from unturtle.models.backbones.llada import LLaDAConfig

        config = LLaDAConfig()
        assert hasattr(config, "mask_token_id")


class TestLLaDAModel:
    @pytest.fixture
    def config(self):
        from unturtle.models.backbones.llada import LLaDAConfig

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
            rope=True,  # LLaDA requires RoPE for MDM
            init_device="cpu",
        )

    def test_model_lm_instantiation(self, config):
        from unturtle.models.backbones.llada import LLaDAModelLM

        model = LLaDAModelLM(config).cpu()
        assert model is not None
        assert hasattr(model, "model") and hasattr(model.model, "transformer")

    def test_forward_logits_shape(self, config):
        from unturtle.models.backbones.llada import LLaDAModelLM

        model = LLaDAModelLM(config).cpu()
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        # LLaDA uses embedding_size (next multiple of 128 ≥ vocab_size) for logits
        effective_vocab = (
            config.embedding_size if config.embedding_size else config.vocab_size
        )
        assert out.logits.shape == (B, L, effective_vocab)
        assert out.logits.shape[0] == B
        assert out.logits.shape[1] == L

    def test_forward_backward(self, config):
        """Gradients flow through LLaDA."""
        from unturtle.models.backbones.llada import LLaDAModelLM

        model = LLaDAModelLM(config).cpu()
        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        out = model(input_ids=input_ids)
        # LLaDA does not compute loss internally — compute manually
        loss = (
            out.logits[:, :, : config.vocab_size]
            .reshape(-1, config.vocab_size)
            .float()
            .log_softmax(-1)
            .mean()
            .neg()
        )
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
        from unturtle.models.backbones.llada import LLaDAConfig

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
        from unturtle.models.backbones.llada import LLaDAModelLM

        return LLaDAModelLM(config).eval()

    TINY_MASK_ID = 511

    def test_has_diffusion_generate(self, model):
        from unturtle.models.backbones.llada import LLaDAGenerationMixin

        assert isinstance(model, LLaDAGenerationMixin)
        assert callable(model.diffusion_generate)

    def test_generate_is_mixin_generate(self, model):
        """MRO identity pin: LLaDAModelLM.generate is the diffusion mixin's generate.

        GenerationMixin is not in the LLaDAModelLM MRO, so this pin guards against
        a future base-class or mixin insertion that could re-route generate to
        transformers' autoregressive path.
        """
        from transformers.generation import GenerationMixin

        from unturtle.models.backbones.llada import LLaDAModelLM
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationMixin,
        )

        assert LLaDAModelLM.generate is MaskedDiffusionGenerationMixin.generate
        assert LLaDAModelLM.generate is not GenerationMixin.generate

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

    def test_attention_mask_2d(self, model):
        """Padded attention_mask (2-D) should be forwarded correctly to LLaDA.

        Exercises the LLaDAGenerationMixin override that keeps the mask 2-D
        instead of expanding to 4-D (which LLaDAModel cannot handle).
        """
        B, L = 2, 10
        # Second sequence is shorter — last 2 positions are padding
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        attention_mask = torch.ones((B, L), dtype=torch.long)
        attention_mask[1, -2:] = 0  # simulate padding in second sample
        with torch.no_grad():
            out = model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                steps=2,
                mask_token_id=self.TINY_MASK_ID,
                max_length=L + 1,
            )
        assert out.shape == (B, L + 1)

    def test_generate_redirects_to_diffusion_generate(self, model):
        """model.generate() must route to diffusion_generate(), not HF AR generate()."""
        B, L = 1, 6
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                steps=2,
                mask_token_id=self.TINY_MASK_ID,
                max_length=L + 1,
            )
        assert out.shape == (B, L + 1)

    def test_num_return_sequences(self, model):
        """num_return_sequences=2 should double the batch dimension."""
        B, L = 1, 6
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        with torch.no_grad():
            out = model.diffusion_generate(
                input_ids,
                steps=2,
                mask_token_id=self.TINY_MASK_ID,
                max_length=L + 1,
                num_return_sequences=2,
            )
        assert out.shape == (B * 2, L + 1)

    def test_llada_generate_runs_diffusion(self, model):
        """model.generate(algorithm="mdlm") runs the diffusion denoising loop end-to-end."""
        B, L_prompt, L_new = 1, 4, 4
        L_total = L_prompt + L_new
        prompt_ids = torch.tensor([[1, 2, 3, 4]])
        mask_fill = torch.full((B, L_new), self.TINY_MASK_ID, dtype=torch.long)
        input_ids_full = torch.cat([prompt_ids, mask_fill], dim=1)
        with torch.no_grad():
            out = model.generate(
                input_ids_full,
                algorithm="mdlm",
                steps=3,
                mask_token_id=self.TINY_MASK_ID,
                max_length=L_total + 1,
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape == (B, L_total + 1)

    def test_llada_generate_ar_raises(self, model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        # No "ar" algorithm exists; pure dLLMs reject it at algorithm resolution.
        with pytest.raises(ValueError, match="Unknown decoding algorithm"):
            model.generate(prompt, algorithm="ar", max_new_tokens=4)


# ---------------------------------------------------------------------------
# LLaDA Triton RoPE fast path (_make_llada_fast_rope_forward)
# ---------------------------------------------------------------------------

cuda = torch.cuda.is_available()


class TestLLaDAFastRoPE:
    """Tests for _make_llada_fast_rope_forward — CPU parity and CUDA correctness."""

    @pytest.fixture
    def rotary_emb(self):
        from collections import defaultdict

        from unturtle.models.backbones.llada import LLaDAConfig
        from unturtle.models.backbones.llada.modeling_llada import RotaryEmbedding

        config = LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=512,
            max_sequence_length=64,
            rope=True,
            init_device="cpu",
        )
        cache = defaultdict(lambda: None)
        return RotaryEmbedding(config, cache)

    def test_fast_forward_importable(self):
        from unturtle.models.backbones.llada.modeling_llada import (
            _make_llada_fast_rope_forward,
        )

        assert callable(_make_llada_fast_rope_forward)

    def test_cpu_parity(self, rotary_emb):
        """CPU fast forward matches original on CPU (falls back to original)."""
        import types

        from unturtle.models.backbones.llada.modeling_llada import (
            _make_llada_fast_rope_forward,
        )

        B, n_heads, T, head_dim = 2, 4, 8, 16
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)

        original_forward = type(rotary_emb).forward
        fast_forward = _make_llada_fast_rope_forward(original_forward)
        rotary_emb.forward = types.MethodType(fast_forward, rotary_emb)

        q_fast, k_fast = rotary_emb(q.clone(), k.clone())
        q_orig, k_orig = original_forward(rotary_emb, q.clone(), k.clone())

        assert torch.allclose(q_fast, q_orig, atol=1e-5), (
            f"Q mismatch: {(q_fast - q_orig).abs().max()}"
        )
        assert torch.allclose(k_fast, k_orig, atol=1e-5), (
            f"K mismatch: {(k_fast - k_orig).abs().max()}"
        )

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_cuda_parity_vs_cpu(self, rotary_emb):
        """CUDA Triton fast RoPE matches original on CPU (within float32 tolerance)."""
        import types

        from unturtle.models.backbones.llada.modeling_llada import (
            _make_llada_fast_rope_forward,
        )

        B, n_heads, T, head_dim = 2, 4, 8, 16
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)

        original_forward = type(rotary_emb).forward

        # CPU reference
        q_cpu, k_cpu = original_forward(rotary_emb, q.clone(), k.clone())

        # CUDA Triton path
        rotary_emb_cuda = rotary_emb
        fast_forward = _make_llada_fast_rope_forward(original_forward)
        rotary_emb_cuda.forward = types.MethodType(fast_forward, rotary_emb_cuda)

        q_cuda = q.clone().cuda()
        k_cuda = k.clone().cuda()
        q_out, k_out = rotary_emb_cuda(q_cuda, k_cuda)

        assert torch.allclose(q_out.cpu(), q_cpu, atol=1e-4), (
            f"Q max diff: {(q_out.cpu() - q_cpu).abs().max()}"
        )
        assert torch.allclose(k_out.cpu(), k_cpu, atol=1e-4), (
            f"K max diff: {(k_out.cpu() - k_cpu).abs().max()}"
        )

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_patch_applied_via_fast_diffusion_model(self):
        """_patch_llada_peft injects Triton RoPE into rotary_emb on CUDA."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.backbones.llada import LLaDAConfig, LLaDAModelLM

        config = LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=512,
            max_sequence_length=64,
            rope=True,
            block_type="llama",  # LLaDALlamaBlock has split q/k/v + rotary_emb
            activation_type="silu",  # LLaDALlamaBlock requires silu (not swiglu)
            init_device="cpu",
        )
        model = LLaDAModelLM(config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
            lora_dropout=0,
            bias="none",
        )
        # Check that at least one block's rotary_emb was patched
        inner = peft_model.base_model.model
        if hasattr(inner, "model") and hasattr(inner.model, "transformer"):
            blocks = inner.model.transformer.blocks
        else:
            blocks = inner.transformer.blocks

        patched = [
            b
            for b in blocks
            if hasattr(b, "rotary_emb")
            and getattr(b.rotary_emb, "_fast_rope_patched", False)
        ]
        assert len(patched) > 0, (
            "Expected at least one block to have Triton RoPE patched"
        )

    def test_kv_cache_fallback_cpu_parity(self, rotary_emb):
        """query_len < key_len (KV-cache prefix) falls back to original — CPU parity."""
        import types

        from unturtle.models.backbones.llada.modeling_llada import (
            _make_llada_fast_rope_forward,
        )

        B, n_heads, query_len, key_len, head_dim = 2, 4, 3, 8, 16
        q = torch.randn(B, n_heads, query_len, head_dim)
        k = torch.randn(B, n_heads, key_len, head_dim)

        original_forward = type(rotary_emb).forward
        fast_forward = _make_llada_fast_rope_forward(original_forward)
        rotary_emb.forward = types.MethodType(fast_forward, rotary_emb)

        q_fast, k_fast = rotary_emb(q.clone(), k.clone())
        q_orig, k_orig = original_forward(rotary_emb, q.clone(), k.clone())

        assert torch.allclose(q_fast, q_orig, atol=1e-6), (
            f"Q mismatch: {(q_fast - q_orig).abs().max()}"
        )
        assert torch.allclose(k_fast, k_orig, atol=1e-6), (
            f"K mismatch: {(k_fast - k_orig).abs().max()}"
        )

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_kv_cache_fallback_cuda_parity(self, rotary_emb):
        """query_len < key_len on CUDA also falls back to original (no Triton path)."""
        import types

        from unturtle.models.backbones.llada.modeling_llada import (
            _make_llada_fast_rope_forward,
        )

        B, n_heads, query_len, key_len, head_dim = 2, 4, 3, 8, 16
        q = torch.randn(B, n_heads, query_len, head_dim)
        k = torch.randn(B, n_heads, key_len, head_dim)

        original_forward = type(rotary_emb).forward
        q_ref, k_ref = original_forward(rotary_emb, q.clone(), k.clone())

        fast_forward = _make_llada_fast_rope_forward(original_forward)
        rotary_emb.forward = types.MethodType(fast_forward, rotary_emb)
        q_out, k_out = rotary_emb(q.clone().cuda(), k.clone().cuda())

        assert torch.allclose(q_out.cpu(), q_ref, atol=1e-6), (
            f"Q max diff: {(q_out.cpu() - q_ref).abs().max()}"
        )
        assert torch.allclose(k_out.cpu(), k_ref, atol=1e-6), (
            f"K max diff: {(k_out.cpu() - k_ref).abs().max()}"
        )

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_gqa_cuda_parity(self):
        """GQA (n_kv_heads=1): CUDA fast RoPE matches original for Q; K passthrough."""
        import types
        from collections import defaultdict

        from unturtle.models.backbones.llada import LLaDAConfig
        from unturtle.models.backbones.llada.modeling_llada import (
            RotaryEmbedding,
            _make_llada_fast_rope_forward,
        )

        config = LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_kv_heads=1,
            n_layers=2,
            vocab_size=512,
            max_sequence_length=64,
            rope=True,
            init_device="cpu",
        )
        cache = defaultdict(lambda: None)
        rotary_emb = RotaryEmbedding(config, cache)

        B, n_heads, n_kv_heads, T, head_dim = 2, 4, 1, 8, 16
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_kv_heads, T, head_dim)

        original_forward = type(rotary_emb).forward
        q_ref, k_ref = original_forward(rotary_emb, q.clone(), k.clone())

        fast_forward = _make_llada_fast_rope_forward(original_forward)
        rotary_emb.forward = types.MethodType(fast_forward, rotary_emb)
        q_out, k_out = rotary_emb(q.clone().cuda(), k.clone().cuda())

        assert torch.allclose(q_out.cpu(), q_ref, atol=1e-4), (
            f"Q max diff: {(q_out.cpu() - q_ref).abs().max()}"
        )
        assert torch.allclose(k_out.cpu(), k_ref, atol=1e-4), (
            f"K max diff: {(k_out.cpu() - k_ref).abs().max()}"
        )

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_double_patch_idempotent(self):
        """Calling _patch_llada_peft twice does not stack the fast forward wrapper."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.backbones.llada import LLaDAConfig, LLaDAModelLM

        config = LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=512,
            max_sequence_length=64,
            rope=True,
            block_type="llama",
            activation_type="silu",  # LLaDALlamaBlock requires silu (not swiglu)
            init_device="cpu",
        )
        model = LLaDAModelLM(config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
            lora_dropout=0,
            bias="none",
        )

        inner = peft_model.base_model.model
        blocks = (
            inner.model.transformer.blocks
            if hasattr(inner, "model") and hasattr(inner.model, "transformer")
            else inner.transformer.blocks
        )

        # Collect forward bindings after first patch
        forwards_after_first = [
            id(b.rotary_emb.forward) for b in blocks if hasattr(b, "rotary_emb")
        ]

        # Simulate a second patch call
        from unturtle.fast_diffusion_model import _patch_llada_peft

        _patch_llada_peft(peft_model, lora_dropout=0, bias="none")

        forwards_after_second = [
            id(b.rotary_emb.forward) for b in blocks if hasattr(b, "rotary_emb")
        ]

        assert forwards_after_first == forwards_after_second, (
            "double patch changed rotary_emb.forward bindings — wrapper was stacked"
        )


# ---------------------------------------------------------------------------
# LLaDA Triton MLP LoRA (apply_lora_mlp_swiglu via apply_mlp stub)
# ---------------------------------------------------------------------------


class TestLLaDAFastMLP:
    """Tests for Triton MLP LoRA patching on LLaDALlamaBlock."""

    @pytest.fixture
    def llama_config(self):
        from unturtle.models.backbones.llada import LLaDAConfig

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
            block_type="llama",
            activation_type="silu",  # LLaDALlamaBlock requires silu (not swiglu)
            init_device="cpu",
        )

    def test_apply_mlp_stub_exists(self, llama_config):
        """LLaDALlamaBlock has apply_mlp stub after instantiation."""
        from unturtle.models.backbones.llada import LLaDAModelLM

        model = LLaDAModelLM(llama_config)
        blocks = model.model.transformer.blocks
        for block in blocks:
            assert hasattr(block, "apply_mlp"), (
                "apply_mlp stub missing from LLaDALlamaBlock"
            )
            assert callable(block.apply_mlp)

    def test_cpu_forward_with_stub(self, llama_config):
        """CPU forward with default apply_mlp stub produces correct output shape."""
        from unturtle.models.backbones.llada import LLaDAModelLM

        model = LLaDAModelLM(llama_config).eval()
        B, L = 2, 8
        input_ids = torch.randint(0, llama_config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape[:2] == (B, L)

    @pytest.mark.skipif(not cuda, reason="Triton MLP LoRA requires CUDA")
    def test_mlp_patched_to_triton(self, llama_config):
        """_patch_llada_peft replaces apply_mlp with apply_lora_mlp_swiglu on CUDA."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.kernels.fast_lora import apply_lora_mlp_swiglu
        from unturtle.models.backbones.llada import LLaDAModelLM

        model = LLaDAModelLM(llama_config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "attn_out",
                "ff_proj",
                "up_proj",
                "ff_out",
            ],
            lora_dropout=0,
            bias="none",
        )
        inner = peft_model.base_model.model
        blocks = (
            inner.model.transformer.blocks
            if hasattr(inner, "model") and hasattr(inner.model, "transformer")
            else inner.transformer.blocks
        )

        patched = [b for b in blocks if b.apply_mlp is apply_lora_mlp_swiglu]
        assert len(patched) > 0, (
            "Expected apply_mlp to be replaced with apply_lora_mlp_swiglu"
        )

    @pytest.mark.skipif(not cuda, reason="Triton MLP LoRA requires CUDA")
    def test_mlp_gate_down_aliases_set(self, llama_config):
        """gate_proj and down_proj aliases are set on block after patching."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.backbones.llada import LLaDAModelLM

        model = LLaDAModelLM(llama_config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "attn_out",
                "ff_proj",
                "up_proj",
                "ff_out",
            ],
            lora_dropout=0,
            bias="none",
        )
        inner = peft_model.base_model.model
        blocks = (
            inner.model.transformer.blocks
            if hasattr(inner, "model") and hasattr(inner.model, "transformer")
            else inner.transformer.blocks
        )

        for block in blocks:
            assert hasattr(block, "gate_proj"), "gate_proj alias missing"
            assert hasattr(block, "down_proj"), "down_proj alias missing"
            assert block.gate_proj is block.ff_proj
            assert block.down_proj is block.ff_out

    @pytest.mark.skipif(not cuda, reason="Triton MLP LoRA requires CUDA")
    def test_cuda_forward_with_triton_mlp(self, llama_config):
        """Full model forward pass works on CUDA with Triton MLP LoRA patched."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.backbones.llada import LLaDAModelLM

        model = LLaDAModelLM(llama_config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "attn_out",
                "ff_proj",
                "up_proj",
                "ff_out",
            ],
            lora_dropout=0,
            bias="none",
        )
        peft_model.eval()
        B, L = 2, 8
        input_ids = torch.randint(0, llama_config.vocab_size, (B, L)).cuda()
        with torch.no_grad():
            out = peft_model(input_ids=input_ids)
        assert out.logits.shape[:2] == (B, L)

    def test_default_apply_mlp_numerics(self, llama_config):
        """_default_apply_mlp output matches the original inline MLP formula."""
        from unturtle.models.backbones.llada import LLaDAModelLM
        from unturtle.models.backbones.llada.modeling_llada import LLaDALlamaBlock

        model = LLaDAModelLM(llama_config).eval()
        block = model.model.transformer.blocks[0]

        B, L, d = 2, 8, llama_config.d_model
        x = torch.randn(B, L, d)

        # Reference: inline formula
        with torch.no_grad():
            x_ref = block.ff_norm(x)
            gate, up = block.ff_proj(x_ref), block.up_proj(x_ref)
            gate = block.act(gate)
            ref_out = block.ff_out(gate * up)

        # Via stub
        with torch.no_grad():
            stub_out = LLaDALlamaBlock._default_apply_mlp(block, block.ff_norm(x))

        assert torch.allclose(stub_out, ref_out, atol=1e-6), (
            f"_default_apply_mlp mismatch: {(stub_out - ref_out).abs().max()}"
        )

    @pytest.mark.skipif(not cuda, reason="Triton MLP LoRA requires CUDA")
    def test_swiglu_block_not_patched(self):
        """LLaDALlamaBlock with swiglu activation is NOT patched with Triton MLP."""
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.backbones.llada import LLaDAConfig, LLaDAModelLM
        from unturtle.models.backbones.llada.modeling_llada import LLaDALlamaBlock

        swiglu_config = LLaDAConfig(
            d_model=64,
            n_heads=4,
            n_layers=2,
            vocab_size=512,
            mlp_ratio=4,
            max_sequence_length=64,
            rope=True,
            block_type="llama",
            activation_type="swiglu",  # swiglu → ff_out.in_features=128
            init_device="cpu",
        )
        model = LLaDAModelLM(swiglu_config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "attn_out",
                "ff_proj",
                "up_proj",
                "ff_out",
            ],
            lora_dropout=0,
            bias="none",
        )
        inner = peft_model.base_model.model
        blocks = (
            inner.model.transformer.blocks
            if hasattr(inner, "model") and hasattr(inner.model, "transformer")
            else inner.transformer.blocks
        )

        # No block should have apply_mlp replaced with apply_lora_mlp_swiglu
        from unturtle.kernels.fast_lora import apply_lora_mlp_swiglu

        patched = [b for b in blocks if b.apply_mlp is apply_lora_mlp_swiglu]
        assert len(patched) == 0, (
            "swiglu blocks should NOT be patched with Triton MLP kernel"
        )


# ---------------------------------------------------------------------------
# Phase M.1: Block-decode KV cache (replace_position)
# ---------------------------------------------------------------------------


class TestLLaDABlockDecode:
    """Phase M.1 tests: replace_position cache handling and block-decode generation."""

    @pytest.fixture
    def tiny_config(self):
        from unturtle.models.backbones.llada import LLaDAConfig

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
            block_type="llama",
            activation_type="silu",
            init_device="cpu",
            mask_token_id=511,
        )

    @pytest.fixture
    def tiny_model(self, tiny_config):
        from unturtle.models.backbones.llada import LLaDAModelLM

        torch.manual_seed(42)
        return LLaDAModelLM(tiny_config).eval()

    def test_replace_position_cache_update(self, tiny_model, tiny_config):
        """Verify cache replacement logic works within block-decode context.

        This test verifies that replace_kv_cache correctly updates marked positions.
        Note: replace_position is intended for use within block-decode loops, where
        query_len matches the block being refined. Testing the full generation loop
        is covered by test_block_decode_baseline.
        """
        from unturtle.models.generation.cache_utils import replace_kv_cache

        B, L = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, L))

        # Step 1: Initial forward to build cache
        with torch.no_grad():
            out1 = tiny_model.model(input_ids=input_ids, use_cache=True)
        past_kv = out1.attn_key_values
        assert past_kv is not None
        assert len(past_kv) == tiny_config.n_layers

        # Step 2: Simulate block-decode update (replace middle 4 positions)
        block_start, block_end = 8, 12
        replace_position = torch.zeros(B, L, dtype=torch.bool)
        replace_position[:, block_start:block_end] = True

        # Generate new K/V for the marked block (simulate denoising step)
        _new_hidden = torch.randn(B, 4, tiny_config.d_model)  # block_length=4
        # Project to K/V (simplified: just use random tensors matching cache shape)
        _, n_heads, _, head_dim = past_kv[0][0].shape
        new_k = torch.randn(B, n_heads, 4, head_dim)
        new_v = torch.randn(B, n_heads, 4, head_dim)

        # Apply cache replacement for layer 0
        updated_cache = replace_kv_cache(
            past_kv, new_k, new_v, replace_position, layer_idx=0
        )

        # Verify cache was updated at marked positions
        old_k, old_v = past_kv[0]
        upd_k, upd_v = updated_cache[0]

        # Positions outside [block_start, block_end] should be unchanged
        assert torch.allclose(
            old_k[:, :, :block_start, :], upd_k[:, :, :block_start, :]
        )
        assert torch.allclose(old_k[:, :, block_end:, :], upd_k[:, :, block_end:, :])

        # Positions inside [block_start, block_end] should be updated
        assert not torch.allclose(
            old_k[:, :, block_start:block_end, :], upd_k[:, :, block_start:block_end, :]
        )

    def test_block_decode_baseline(self, tiny_model, tiny_config):
        """Block-decode generation runs without error (baseline)."""
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        B, prompt_len = 2, 8
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, prompt_len))

        gen_config = MaskedDiffusionGenerationConfig(
            max_new_tokens=8,
            steps=2,
            alg="origin",
            mask_token_id=tiny_config.mask_token_id,
            use_cache=True,
            block_length=4,
        )

        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=input_ids,
                generation_config=gen_config,
            )

        assert output.shape == (B, prompt_len + 8)

    def test_block_decode_correctness(self, tiny_model, tiny_config):
        """Block-decode generates valid (non-mask) tokens in the generated region.

        Block-decode uses a trimmed KV-cache so attended context differs from the
        full no-cache forward pass in a bidirectional model — exact value equivalence
        with the no-cache baseline is not guaranteed.  We verify output correctness:
        correct shape, prompt preserved, no remaining mask tokens.
        """
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        B, prompt_len = 1, 6
        max_new = 8
        torch.manual_seed(42)
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, prompt_len))

        gen_config_with_cache = MaskedDiffusionGenerationConfig(
            max_new_tokens=max_new,
            steps=4,
            alg="origin",
            mask_token_id=tiny_config.mask_token_id,
            use_cache=True,
            use_replace_cache=False,
            block_length=4,
            temperature=1.0,  # stochastic: argmax on random weights always picks mask_token_id
        )

        torch.manual_seed(42)
        with torch.no_grad():
            output_with_cache = tiny_model.diffusion_generate(
                inputs=input_ids.clone(),
                generation_config=gen_config_with_cache,
            )

        assert output_with_cache.shape == (B, prompt_len + max_new)
        assert torch.equal(output_with_cache[:, :prompt_len], input_ids)
        assert not torch.any(
            output_with_cache[:, prompt_len:] == tiny_config.mask_token_id
        ), "Block-decode should produce no remaining mask tokens in generated region"

    def test_parallel_decode_trim_cache_runs(self, tiny_model, tiny_config):
        """LLaDA parallel_decode runs in trim-cache mode."""
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        gen_config = MaskedDiffusionGenerationConfig(
            max_new_tokens=8,
            steps=4,
            alg="maskgit_plus",
            mask_token_id=tiny_config.mask_token_id,
            use_cache=True,
            use_replace_cache=False,
            parallel_decode=True,
            confidence_threshold=0.05,
            block_length=4,
            temperature=0.0,
        )

        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=input_ids, generation_config=gen_config
            )

        assert output.shape == (1, 16)
        assert not torch.any(output[:, 8:] == tiny_config.mask_token_id)

    def test_parallel_decode_completes_block_when_steps_per_block_is_small(
        self, tiny_model, tiny_config, monkeypatch
    ):
        """Trim-cache threshold mode keeps denoising until the block finishes."""
        import unturtle.models.generation.block_decode_mixin as block_decode_mixin
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        def select_single_token(masked_confidence, mask_index_block, threshold):
            transfer_mask = torch.zeros_like(mask_index_block, dtype=torch.bool)
            for row_idx in range(mask_index_block.shape[0]):
                masked_positions = mask_index_block[row_idx].nonzero(as_tuple=True)[0]
                if masked_positions.numel() > 0:
                    transfer_mask[row_idx, masked_positions[0]] = True
            return transfer_mask

        monkeypatch.setattr(
            block_decode_mixin, "select_threshold_transfer_mask", select_single_token
        )

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        gen_config = MaskedDiffusionGenerationConfig(
            max_new_tokens=8,
            steps=2,
            alg="maskgit_plus",
            mask_token_id=tiny_config.mask_token_id,
            use_cache=True,
            use_replace_cache=False,
            parallel_decode=True,
            confidence_threshold=0.99,
            block_length=4,
            temperature=0.0,
        )

        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=input_ids, generation_config=gen_config
            )

        assert output.shape == (1, 16)
        assert not torch.any(output[:, 8:] == tiny_config.mask_token_id)

    def test_parallel_decode_dual_cache_runs(self, tiny_model, tiny_config):
        """LLaDA parallel_decode runs in dual-cache replace_position mode."""
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        gen_config = MaskedDiffusionGenerationConfig(
            max_new_tokens=8,
            steps=4,
            alg="maskgit_plus",
            mask_token_id=tiny_config.mask_token_id,
            use_cache=True,
            use_replace_cache=True,
            parallel_decode=True,
            confidence_threshold=0.05,
            block_length=4,
            temperature=0.0,
        )

        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=input_ids, generation_config=gen_config
            )

        assert output.shape == (1, 16)
        assert not torch.any(output[:, 8:] == tiny_config.mask_token_id)

    def test_parallel_decode_dual_cache_completes_block_when_steps_per_block_is_small(
        self, tiny_model, tiny_config, monkeypatch
    ):
        """Dual-cache threshold mode keeps denoising until the block finishes."""
        import unturtle.models.generation.block_decode_mixin as block_decode_mixin
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )

        def select_single_token(masked_confidence, mask_index_block, threshold):
            transfer_mask = torch.zeros_like(mask_index_block, dtype=torch.bool)
            for row_idx in range(mask_index_block.shape[0]):
                masked_positions = mask_index_block[row_idx].nonzero(as_tuple=True)[0]
                if masked_positions.numel() > 0:
                    transfer_mask[row_idx, masked_positions[0]] = True
            return transfer_mask

        monkeypatch.setattr(
            block_decode_mixin, "select_threshold_transfer_mask", select_single_token
        )

        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        gen_config = MaskedDiffusionGenerationConfig(
            max_new_tokens=8,
            steps=2,
            alg="maskgit_plus",
            mask_token_id=tiny_config.mask_token_id,
            use_cache=True,
            use_replace_cache=True,
            parallel_decode=True,
            confidence_threshold=0.99,
            block_length=4,
            temperature=0.0,
        )

        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=input_ids, generation_config=gen_config
            )

        assert output.shape == (1, 16)
        assert not torch.any(output[:, 8:] == tiny_config.mask_token_id)

    def test_rope_with_replace_position(self, tiny_config):
        """RoPE with block_end_index bounds position indexing (block-decode context).

        In block-decode, query_len matches the current block being refined, and
        block_end_index is the end of that block in the full sequence.
        """
        from collections import defaultdict

        from unturtle.models.backbones.llada.modeling_llada import RotaryEmbedding

        cache = defaultdict(lambda: None)
        rotary_emb = RotaryEmbedding(tiny_config, cache)

        B, n_heads, head_dim = 2, 4, 16
        query_len, key_len = 4, 16
        q = torch.randn(B, n_heads, query_len, head_dim)
        k = torch.randn(B, n_heads, key_len, head_dim)

        q1, k1 = rotary_emb(q.clone(), k.clone())
        q2, k2 = rotary_emb(q.clone(), k.clone(), block_end_index=16)
        assert torch.allclose(q1, q2, atol=1e-6), (
            "block_end_index=key_len should match standard RoPE"
        )

        q3, k3 = rotary_emb(q.clone(), k.clone(), block_end_index=12)
        assert not torch.allclose(q1, q3, atol=1e-6), (
            "block_end_index=12 should produce different positions"
        )
