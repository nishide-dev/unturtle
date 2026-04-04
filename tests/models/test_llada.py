"""Tests for LLaDA models.

CPU-only tests covering config instantiation, model instantiation with
random weights, and forward pass shapes. No pretrained checkpoints required.
"""

from __future__ import annotations

import pytest
import torch


class TestLLaDAConfig:
    def test_config_default_fields(self):
        from unturtle.models.llada import LLaDAConfig
        config = LLaDAConfig()
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_layers == 12
        assert config.vocab_size == 50257

    def test_config_custom_values(self):
        from unturtle.models.llada import LLaDAConfig
        config = LLaDAConfig(d_model=128, n_heads=4, n_layers=2, vocab_size=1000)
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 2
        assert config.vocab_size == 1000

    def test_config_hf_properties(self):
        """HF-compatible properties map to internal fields."""
        from unturtle.models.llada import LLaDAConfig
        config = LLaDAConfig(d_model=256, n_heads=8, n_layers=4)
        assert config.hidden_size == 256
        assert config.num_attention_heads == 8
        assert config.num_hidden_layers == 4

    def test_config_has_mask_token_id(self):
        from unturtle.models.llada import LLaDAConfig
        config = LLaDAConfig()
        assert hasattr(config, "mask_token_id")


class TestLLaDAModel:
    @pytest.fixture
    def config(self):
        from unturtle.models.llada import LLaDAConfig
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
            rope=True,         # LLaDA requires RoPE for MDM
            init_device="cpu",
        )

    def test_model_lm_instantiation(self, config):
        from unturtle.models.llada import LLaDAModelLM
        model = LLaDAModelLM(config).cpu()
        assert model is not None
        assert hasattr(model, "model") and hasattr(model.model, "transformer")

    def test_forward_logits_shape(self, config):
        from unturtle.models.llada import LLaDAModelLM
        model = LLaDAModelLM(config).cpu()
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        # LLaDA uses embedding_size (next multiple of 128 ≥ vocab_size) for logits
        effective_vocab = config.embedding_size if config.embedding_size else config.vocab_size
        assert out.logits.shape == (B, L, effective_vocab)
        assert out.logits.shape[0] == B
        assert out.logits.shape[1] == L

    def test_forward_backward(self, config):
        """Gradients flow through LLaDA."""
        from unturtle.models.llada import LLaDAModelLM
        model = LLaDAModelLM(config).cpu()
        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        out = model(input_ids=input_ids)
        # LLaDA does not compute loss internally — compute manually
        loss = out.logits[:, :, :config.vocab_size].reshape(-1, config.vocab_size).float().log_softmax(-1).mean().neg()
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
        from unturtle.models.llada import LLaDAConfig
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
        from unturtle.models.llada import LLaDAModelLM
        return LLaDAModelLM(config).eval()

    TINY_MASK_ID = 511

    def test_has_diffusion_generate(self, model):
        from unturtle.models.llada import LLaDAGenerationMixin
        assert isinstance(model, LLaDAGenerationMixin)
        assert callable(model.diffusion_generate)

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


# ---------------------------------------------------------------------------
# LLaDA Triton RoPE fast path (_make_llada_fast_rope_forward)
# ---------------------------------------------------------------------------

cuda = torch.cuda.is_available()


class TestLLaDAFastRoPE:
    """Tests for _make_llada_fast_rope_forward — CPU parity and CUDA correctness."""

    @pytest.fixture
    def rotary_emb(self):
        from unturtle.models.llada import LLaDAConfig
        from unturtle.models.llada.modeling_llada import RotaryEmbedding
        from collections import defaultdict

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
        from unturtle.models.llada.modeling_llada import _make_llada_fast_rope_forward
        assert callable(_make_llada_fast_rope_forward)

    def test_cpu_parity(self, rotary_emb):
        """CPU fast forward matches original on CPU (falls back to original)."""
        from unturtle.models.llada.modeling_llada import _make_llada_fast_rope_forward
        import types

        B, n_heads, T, head_dim = 2, 4, 8, 16
        q = torch.randn(B, n_heads, T, head_dim)
        k = torch.randn(B, n_heads, T, head_dim)

        original_forward = type(rotary_emb).forward
        fast_forward = _make_llada_fast_rope_forward(original_forward)
        rotary_emb.forward = types.MethodType(fast_forward, rotary_emb)

        q_fast, k_fast = rotary_emb(q.clone(), k.clone())
        q_orig, k_orig = original_forward(rotary_emb, q.clone(), k.clone())

        assert torch.allclose(q_fast, q_orig, atol=1e-5), f"Q mismatch: {(q_fast - q_orig).abs().max()}"
        assert torch.allclose(k_fast, k_orig, atol=1e-5), f"K mismatch: {(k_fast - k_orig).abs().max()}"

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_cuda_parity_vs_cpu(self, rotary_emb):
        """CUDA Triton fast RoPE matches original on CPU (within float32 tolerance)."""
        from unturtle.models.llada.modeling_llada import _make_llada_fast_rope_forward
        import types

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

        assert torch.allclose(q_out.cpu(), q_cpu, atol=1e-4), f"Q max diff: {(q_out.cpu() - q_cpu).abs().max()}"
        assert torch.allclose(k_out.cpu(), k_cpu, atol=1e-4), f"K max diff: {(k_out.cpu() - k_cpu).abs().max()}"

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_patch_applied_via_fast_diffusion_model(self):
        """_patch_llada_peft injects Triton RoPE into rotary_emb on CUDA."""
        from unturtle.models.llada import LLaDAConfig, LLaDAModelLM
        from unturtle.fast_diffusion_model import FastDiffusionModel

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
            b for b in blocks
            if hasattr(b, "rotary_emb") and getattr(b.rotary_emb, "_fast_rope_patched", False)
        ]
        assert len(patched) > 0, "Expected at least one block to have Triton RoPE patched"

    def test_kv_cache_fallback_cpu_parity(self, rotary_emb):
        """query_len < key_len (KV-cache prefix) falls back to original — CPU parity."""
        from unturtle.models.llada.modeling_llada import _make_llada_fast_rope_forward
        import types

        B, n_heads, query_len, key_len, head_dim = 2, 4, 3, 8, 16
        q = torch.randn(B, n_heads, query_len, head_dim)
        k = torch.randn(B, n_heads, key_len, head_dim)

        original_forward = type(rotary_emb).forward
        fast_forward = _make_llada_fast_rope_forward(original_forward)
        rotary_emb.forward = types.MethodType(fast_forward, rotary_emb)

        q_fast, k_fast = rotary_emb(q.clone(), k.clone())
        q_orig, k_orig = original_forward(rotary_emb, q.clone(), k.clone())

        assert torch.allclose(q_fast, q_orig, atol=1e-6), f"Q mismatch: {(q_fast - q_orig).abs().max()}"
        assert torch.allclose(k_fast, k_orig, atol=1e-6), f"K mismatch: {(k_fast - k_orig).abs().max()}"

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_kv_cache_fallback_cuda_parity(self, rotary_emb):
        """query_len < key_len on CUDA also falls back to original (no Triton path)."""
        from unturtle.models.llada.modeling_llada import _make_llada_fast_rope_forward
        import types

        B, n_heads, query_len, key_len, head_dim = 2, 4, 3, 8, 16
        q = torch.randn(B, n_heads, query_len, head_dim)
        k = torch.randn(B, n_heads, key_len, head_dim)

        original_forward = type(rotary_emb).forward
        q_ref, k_ref = original_forward(rotary_emb, q.clone(), k.clone())

        fast_forward = _make_llada_fast_rope_forward(original_forward)
        rotary_emb.forward = types.MethodType(fast_forward, rotary_emb)
        q_out, k_out = rotary_emb(q.clone().cuda(), k.clone().cuda())

        assert torch.allclose(q_out.cpu(), q_ref, atol=1e-6), f"Q max diff: {(q_out.cpu() - q_ref).abs().max()}"
        assert torch.allclose(k_out.cpu(), k_ref, atol=1e-6), f"K max diff: {(k_out.cpu() - k_ref).abs().max()}"

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_gqa_cuda_parity(self):
        """GQA (n_kv_heads=1): CUDA fast RoPE matches original for Q; K passthrough."""
        from unturtle.models.llada import LLaDAConfig
        from unturtle.models.llada.modeling_llada import RotaryEmbedding, _make_llada_fast_rope_forward
        from collections import defaultdict
        import types

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

        assert torch.allclose(q_out.cpu(), q_ref, atol=1e-4), f"Q max diff: {(q_out.cpu() - q_ref).abs().max()}"
        assert torch.allclose(k_out.cpu(), k_ref, atol=1e-4), f"K max diff: {(k_out.cpu() - k_ref).abs().max()}"

    @pytest.mark.skipif(not cuda, reason="Triton fast RoPE requires CUDA")
    def test_double_patch_idempotent(self):
        """Calling _patch_llada_peft twice does not stack the fast forward wrapper."""
        from unturtle.models.llada import LLaDAConfig, LLaDAModelLM
        from unturtle.fast_diffusion_model import FastDiffusionModel

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
            model, r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
            lora_dropout=0, bias="none",
        )

        inner = peft_model.base_model.model
        blocks = (inner.model.transformer.blocks
                  if hasattr(inner, "model") and hasattr(inner.model, "transformer")
                  else inner.transformer.blocks)

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
        from unturtle.models.llada import LLaDAConfig
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
        from unturtle.models.llada import LLaDAModelLM
        model = LLaDAModelLM(llama_config)
        blocks = model.model.transformer.blocks
        for block in blocks:
            assert hasattr(block, "apply_mlp"), "apply_mlp stub missing from LLaDALlamaBlock"
            assert callable(block.apply_mlp)

    def test_cpu_forward_with_stub(self, llama_config):
        """CPU forward with default apply_mlp stub produces correct output shape."""
        from unturtle.models.llada import LLaDAModelLM
        model = LLaDAModelLM(llama_config).eval()
        B, L = 2, 8
        input_ids = torch.randint(0, llama_config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape[:2] == (B, L)

    @pytest.mark.skipif(not cuda, reason="Triton MLP LoRA requires CUDA")
    def test_mlp_patched_to_triton(self, llama_config):
        """_patch_llada_peft replaces apply_mlp with apply_lora_mlp_swiglu on CUDA."""
        from unturtle.models.llada import LLaDAModelLM
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.kernels.fast_lora import apply_lora_mlp_swiglu

        model = LLaDAModelLM(llama_config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"],
            lora_dropout=0,
            bias="none",
        )
        inner = peft_model.base_model.model
        blocks = (inner.model.transformer.blocks
                  if hasattr(inner, "model") and hasattr(inner.model, "transformer")
                  else inner.transformer.blocks)

        patched = [b for b in blocks if b.apply_mlp is apply_lora_mlp_swiglu]
        assert len(patched) > 0, "Expected apply_mlp to be replaced with apply_lora_mlp_swiglu"

    @pytest.mark.skipif(not cuda, reason="Triton MLP LoRA requires CUDA")
    def test_mlp_gate_down_aliases_set(self, llama_config):
        """gate_proj and down_proj aliases are set on block after patching."""
        from unturtle.models.llada import LLaDAModelLM
        from unturtle.fast_diffusion_model import FastDiffusionModel

        model = LLaDAModelLM(llama_config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"],
            lora_dropout=0,
            bias="none",
        )
        inner = peft_model.base_model.model
        blocks = (inner.model.transformer.blocks
                  if hasattr(inner, "model") and hasattr(inner.model, "transformer")
                  else inner.transformer.blocks)

        for block in blocks:
            assert hasattr(block, "gate_proj"), "gate_proj alias missing"
            assert hasattr(block, "down_proj"), "down_proj alias missing"
            assert block.gate_proj is block.ff_proj
            assert block.down_proj is block.ff_out

    @pytest.mark.skipif(not cuda, reason="Triton MLP LoRA requires CUDA")
    def test_cuda_forward_with_triton_mlp(self, llama_config):
        """Full model forward pass works on CUDA with Triton MLP LoRA patched."""
        from unturtle.models.llada import LLaDAModelLM
        from unturtle.fast_diffusion_model import FastDiffusionModel

        model = LLaDAModelLM(llama_config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"],
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
        from unturtle.models.llada import LLaDAModelLM
        from unturtle.models.llada.modeling_llada import LLaDALlamaBlock

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
        from unturtle.models.llada import LLaDAConfig, LLaDAModelLM
        from unturtle.fast_diffusion_model import FastDiffusionModel
        from unturtle.models.llada.modeling_llada import LLaDALlamaBlock

        swiglu_config = LLaDAConfig(
            d_model=64, n_heads=4, n_layers=2, vocab_size=512,
            mlp_ratio=4, max_sequence_length=64, rope=True,
            block_type="llama", activation_type="swiglu",  # swiglu → ff_out.in_features=128
            init_device="cpu",
        )
        model = LLaDAModelLM(swiglu_config).cuda()
        peft_model = FastDiffusionModel.get_peft_model(
            model, r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"],
            lora_dropout=0, bias="none",
        )
        inner = peft_model.base_model.model
        blocks = (inner.model.transformer.blocks
                  if hasattr(inner, "model") and hasattr(inner.model, "transformer")
                  else inner.transformer.blocks)

        # No block should have apply_mlp replaced with apply_lora_mlp_swiglu
        from unturtle.kernels.fast_lora import apply_lora_mlp_swiglu
        patched = [b for b in blocks if b.apply_mlp is apply_lora_mlp_swiglu]
        assert len(patched) == 0, (
            "swiglu blocks should NOT be patched with Triton MLP kernel"
        )
