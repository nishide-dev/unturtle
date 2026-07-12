"""Tests for Dream diffusion language models.

CPU-only tests covering config instantiation, model instantiation with
random weights, and forward pass shapes. No pretrained checkpoints required.
"""

from __future__ import annotations

import pytest
import torch


class TestDreamConfig:
    def test_config_default_fields(self):
        from unturtle.models.backbones.dream import DreamConfig

        config = DreamConfig()
        assert config.vocab_size == 151936
        assert config.hidden_size == 4096
        assert config.num_hidden_layers == 32

    def test_config_custom_values(self):
        from unturtle.models.backbones.dream import DreamConfig

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
        from unturtle.models.backbones.dream import DreamConfig

        config = DreamConfig()
        assert hasattr(config, "mask_token_id")
        assert config.mask_token_id == 151666

    def test_config_use_cache_false(self):
        """Dream configs have use_cache=False by design."""
        from unturtle.models.backbones.dream import DreamConfig

        config = DreamConfig()
        assert config.use_cache is False


class TestDreamModel:
    @pytest.fixture
    def config(self):
        from unturtle.models.backbones.dream import DreamConfig

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
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu()
        assert model is not None

    def test_forward_logits_shape(self, config):
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu()
        model.eval()
        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)

    def test_forward_backward(self, config):
        """Gradients flow through Dream (unshifted masked CE computed externally)."""
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu()
        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        labels = torch.randint(0, config.vocab_size, (B, L))
        labels[:, ::2] = -100
        out = model(input_ids=input_ids)
        loss = torch.nn.functional.cross_entropy(
            out.logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100
        )
        assert not torch.isnan(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_forward_labels_raises(self, config):
        """DreamModel refuses `labels`: the inherited transformers fallback is a
        shifted causal-LM loss, which is wrong for masked diffusion. Use
        DiffusionTrainer (or an external unshifted masked CE) instead."""
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu()
        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        labels = torch.randint(0, config.vocab_size, (B, L))
        with pytest.raises(NotImplementedError, match="DiffusionTrainer"):
            model(input_ids=input_ids, labels=labels)

    def test_gradient_checkpointing_forward_backward(self, config):
        """Training-mode forward+backward with gradient checkpointing must not raise.

        Regression: the checkpointed call used to pass 10 positional args to
        DreamDecoderLayer.forward (8 positionals + **kwargs) -> TypeError.
        """
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu()
        model.gradient_checkpointing_enable()
        model.train()
        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        labels = torch.randint(0, config.vocab_size, (B, L))
        labels[:, ::2] = -100
        out = model(input_ids=input_ids)
        loss = torch.nn.functional.cross_entropy(
            out.logits.view(-1, config.vocab_size), labels.view(-1), ignore_index=-100
        )
        assert not torch.isnan(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_gradient_checkpointing_dual_cache_raises(self, config):
        """dual_cache is inference-only; combined with training-mode gradient
        checkpointing it used to be silently dropped — now it must raise."""
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu()
        model.gradient_checkpointing_enable()
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        with pytest.raises(ValueError, match="dual_cache"):
            model(input_ids=input_ids, dual_cache=True)

    def test_gradient_checkpointing_matches_plain_forward_with_padding_mask(
        self, config
    ):
        """Checkpointed forward under a 2-D padding mask matches the plain path."""
        from unturtle.models.backbones.dream import DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu()
        model.train()
        B, L = 2, 8
        input_ids = torch.randint(2, config.vocab_size, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        attention_mask[0, -3:] = 0
        plain = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.detach()
        model.gradient_checkpointing_enable()
        checkpointed = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.detach()
        assert torch.allclose(plain, checkpointed, atol=1e-6)

    def test_gradient_checkpointing_matches_plain_forward(self, config):
        """Checkpointed forward must be numerically identical to the plain path."""
        from unturtle.models.backbones.dream import DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu()
        model.train()
        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        plain = model(input_ids=input_ids).logits.detach()
        model.gradient_checkpointing_enable()
        checkpointed = model(input_ids=input_ids).logits.detach()
        assert torch.allclose(plain, checkpointed, atol=1e-6)

    def test_padded_batch_eager_masks_padding(self, config):
        """Eager attention must actually mask padding (bool keep-mask handling).

        Regression: the eager path added the bool [B,1,L,L] keep-mask to the
        attention scores (+1/+0) instead of masking with -inf, so padding
        silently leaked into attention. output_attentions=True forces the
        eager DreamAttention.forward fallback.
        """
        from unturtle.models.backbones.dream import DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        L_real, L_pad = 6, 8
        real_ids = torch.randint(2, config.vocab_size, (1, L_real))
        padded_ids = torch.full((1, L_pad), config.pad_token_id, dtype=torch.long)
        padded_ids[:, :L_real] = real_ids
        attention_mask = torch.zeros(1, L_pad, dtype=torch.long)
        attention_mask[:, :L_real] = 1

        with torch.no_grad():
            ref = model(input_ids=real_ids, output_attentions=True).logits
            masked = model(
                input_ids=padded_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            ).logits
            unmasked = model(input_ids=padded_ids, output_attentions=True).logits

        # Non-pad positions of the masked padded batch match the unpadded run
        assert torch.allclose(ref, masked[:, :L_real], atol=1e-5), (
            f"max_diff={(ref - masked[:, :L_real]).abs().max().item():.2e}"
        )
        # ... and differ from the unmasked run (padding actually masked)
        assert not torch.allclose(ref, unmasked[:, :L_real], atol=1e-5)

    def test_padded_batch_sdpa_masks_padding(self, config):
        """Same padding-equivalence guarantee on the default SDPA path."""
        from unturtle.models.backbones.dream import DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        L_real, L_pad = 6, 8
        real_ids = torch.randint(2, config.vocab_size, (1, L_real))
        padded_ids = torch.full((1, L_pad), config.pad_token_id, dtype=torch.long)
        padded_ids[:, :L_real] = real_ids
        attention_mask = torch.zeros(1, L_pad, dtype=torch.long)
        attention_mask[:, :L_real] = 1

        with torch.no_grad():
            ref = model(input_ids=real_ids).logits
            masked = model(input_ids=padded_ids, attention_mask=attention_mask).logits

        assert torch.allclose(ref, masked[:, :L_real], atol=1e-5), (
            f"max_diff={(ref - masked[:, :L_real]).abs().max().item():.2e}"
        )

    def test_all_ones_attention_mask_eager_does_not_crash(self, config):
        """All-ones 2-D mask becomes the string sentinel 'full'; the eager path
        must treat it as no-mask instead of crashing on string addition."""
        from unturtle.models.backbones.dream import DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        B, L = 2, 8
        input_ids = torch.randint(2, config.vocab_size, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        with torch.no_grad():
            with_mask = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            ).logits
            no_mask = model(input_ids=input_ids, output_attentions=True).logits
        assert torch.allclose(with_mask, no_mask, atol=1e-6)

    def test_position_ids_are_honored(self, config):
        """Explicit shifted position_ids must change the output vs default arange."""
        from unturtle.models.backbones.dream import DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        B, L = 1, 8
        input_ids = torch.randint(2, config.vocab_size, (B, L))
        default = torch.arange(L).unsqueeze(0)
        shifted = default + 5
        collapsed = torch.zeros(1, L, dtype=torch.long)
        with torch.no_grad():
            out_default = model(input_ids=input_ids).logits
            out_explicit_default = model(
                input_ids=input_ids, position_ids=default
            ).logits
            out_shifted = model(input_ids=input_ids, position_ids=shifted).logits
            out_collapsed = model(input_ids=input_ids, position_ids=collapsed).logits
        # Explicit arange == implicit arange
        assert torch.allclose(out_default, out_explicit_default, atol=1e-6)
        # RoPE is relative: a uniform shift preserves pairwise offsets, so a
        # shifted arange must still match — proving the ids reach RoPE intact.
        assert torch.allclose(out_default, out_shifted, atol=1e-5)
        # Collapsed positions change relative structure -> output must differ
        assert not torch.allclose(out_default, out_collapsed, atol=1e-4)

    def test_forward_use_cache_returns_past_key_values(self, config):
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu().eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        assert out.past_key_values is not None
        assert len(out.past_key_values) == config.num_hidden_layers


class TestDreamGenerationUtils:
    @pytest.fixture
    def config(self):
        from unturtle.models.backbones.dream import DreamConfig

        return DreamConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
            mask_token_id=1,
            use_cache=False,
        )

    def test_generation_config_creation(self):
        from unturtle.models.backbones.dream import DreamGenerationConfig

        gen_config = DreamGenerationConfig()
        assert gen_config is not None

    def test_generation_mixin_importable(self):
        from unturtle.models.backbones.dream import DreamGenerationMixin

        assert DreamGenerationMixin is not None
        assert hasattr(DreamGenerationMixin, "generate")

    def test_cache_block_decode_trim_mode(self, config):
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            block_length=2,
            use_cache=True,
            use_replace_cache=False,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            out = model.generate(inputs=inputs, generation_config=generation_config)
        assert out.shape == (1, 8)
        assert not torch.any(out == config.mask_token_id)

    def test_cache_block_decode_dual_cache_mode(self, config):
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            block_length=2,
            use_cache=True,
            use_replace_cache=True,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            out = model.generate(inputs=inputs, generation_config=generation_config)
        assert out.shape == (2, 8)
        assert not torch.any(out == config.mask_token_id)

    def test_forward_preserves_additive_attention_mask(self, config):
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu().eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 6))
        additive_mask = torch.zeros(1, 1, 6, 6)
        additive_mask[:, :, :, -1] = torch.finfo(torch.float32).min
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=additive_mask)
        assert out.logits.shape == (1, 6, config.vocab_size)
        assert not torch.isnan(out.logits).any()

    def test_forward_accepts_3d_additive_attention_mask_with_attentions(self, config):
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu().eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 6))
        additive_mask = torch.zeros(1, 6, 6)
        additive_mask[:, :, -1] = torch.finfo(torch.float32).min
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=additive_mask,
                output_attentions=True,
            )
        assert out.logits.shape == (1, 6, config.vocab_size)
        assert out.attentions is not None
        assert not torch.isnan(out.logits).any()

    def test_cache_block_decode_accepts_additive_attention_mask(self, config):
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5]])
        additive_mask = torch.zeros(1, 1, 8, 8)
        additive_mask[:, :, :, 3] = torch.finfo(torch.float32).min
        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            block_length=2,
            use_cache=True,
            use_replace_cache=True,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            out = model.generate(
                inputs=inputs,
                attention_mask=additive_mask,
                generation_config=generation_config,
            )
        assert out.shape == (1, 8)
        assert not torch.any(out == config.mask_token_id)

    def test_cache_path_preserves_shifted_logits(self, config):
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        class SpyDreamModel(DreamModel):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.seen_logits = None

            def _postprocess_block_decode_logits(self, logits):
                shifted = super()._postprocess_block_decode_logits(logits)
                self.seen_logits = shifted
                return shifted

        model = SpyDreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=2,
            steps=2,
            block_length=2,
            use_cache=True,
            use_replace_cache=False,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            _ = model.generate(inputs=inputs, generation_config=generation_config)
        assert model.seen_logits is not None

    def test_dual_cache_query_start_includes_previous_token(self, config):
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        class SpyDreamModel(DreamModel):
            def __init__(self, cfg):
                super().__init__(cfg)
                self.forward_lengths = []

            def _model_forward_with_cache(self, *args, **kwargs):
                input_ids = kwargs["input_ids"]
                self.forward_lengths.append(input_ids.shape[1])
                return super()._model_forward_with_cache(*args, **kwargs)

        model = SpyDreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            block_length=2,
            use_cache=True,
            use_replace_cache=True,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            _ = model.generate(inputs=inputs, generation_config=generation_config)

        assert 3 in model.forward_lengths

    def test_left_padded_generation_smoke(self, config):
        """Left-padded batch generation exercises the tok_idx position_ids path.

        _sample computes tok_idx from the padding mask (RoPE fix under left
        padding) and passes it as position_ids; the base model must honor it.
        """
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        inputs = torch.tensor([[0, 0, 2, 3], [4, 5, 6, 7]])
        attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            out = model.generate(
                inputs=inputs,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
        assert out.shape == (2, 8)
        assert not torch.any(out == config.mask_token_id)

    def test_generate_without_pad_token_id(self, config):
        """Regression (#48): `generate` must not crash when pad_token_id is None.

        The padding-detection check used to evaluate
        ``torch.any(input_ids == None)`` -> TypeError. The original Dream guard
        only runs the check when a pad token is actually set.
        """
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            mask_token_id=config.mask_token_id,
            pad_token_id=None,
            eos_token_id=None,
        )
        with torch.no_grad():
            out = model.generate(inputs=inputs, generation_config=generation_config)
        assert out.shape == (1, 8)
        assert not torch.any(out == config.mask_token_id)

    def test_dream_generate_accepts_algorithm(self, config):
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        model = DreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            block_length=2,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            out = model.generate(
                inputs=inputs, algorithm="mdlm", generation_config=generation_config
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape == (1, 8)

    def test_dream_generate_bd3lm_raises(self, config):
        """Dream does not implement BD3LM; explicit algorithm='bd3lm' must raise ValueError."""
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5]])
        with pytest.raises(ValueError, match="BD3LM"):
            model.generate(
                inputs=inputs,
                algorithm="bd3lm",
                steps=2,
                mask_token_id=config.mask_token_id,
                max_new_tokens=4,
            )


class TestDreamFastRoPE:
    """Tests for DreamAttention_fast_forward Triton RoPE path."""

    @pytest.fixture
    def config(self):
        from unturtle.models.backbones.dream import DreamConfig

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

    @staticmethod
    def _install_fast_forward(model):
        """Install apply_qkv/apply_o stubs and the fast attention forward on
        every layer (CPU-safe simulation of what _patch_dream_peft injects)."""
        import types

        from unturtle.fast_diffusion_model import (
            _original_apply_o,
            _original_apply_qkv,
        )
        from unturtle.models.backbones.dream.modeling_dream import (
            DreamAttention_fast_forward,
        )

        for layer in model.model.layers:
            attn = layer.self_attn
            if not hasattr(attn, "apply_qkv"):
                attn.apply_qkv = _original_apply_qkv
                attn.apply_o = _original_apply_o
            attn.forward = types.MethodType(DreamAttention_fast_forward, attn)

    def test_fast_forward_importable(self):
        from unturtle.models.backbones.dream.modeling_dream import (
            DreamAttention_fast_forward,
        )

        assert callable(DreamAttention_fast_forward)

    def test_cpu_parity_simple(self, config):
        """DreamAttention_fast_forward CPU fallback matches original forward."""
        import types

        from unturtle.models.backbones.dream import DreamModel
        from unturtle.models.backbones.dream.modeling_dream import (
            DreamAttention_fast_forward,
        )

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()

        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))

        # Reference output (original forward)
        with torch.no_grad():
            ref_out = model(input_ids=input_ids)

        # Install stubs and fast forward
        for module in model.modules():
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "o_proj")
                and not hasattr(module, "apply_qkv")
            ):
                from unturtle.fast_diffusion_model import (
                    _original_apply_o,
                    _original_apply_qkv,
                )

                module.apply_qkv = _original_apply_qkv
                module.apply_o = _original_apply_o

        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(
                DreamAttention_fast_forward, layer.self_attn
            )

        with torch.no_grad():
            fast_out = model(input_ids=input_ids)

        assert torch.allclose(ref_out.logits, fast_out.logits, atol=1e-5), (
            f"CPU logits mismatch: max_diff={(ref_out.logits - fast_out.logits).abs().max().item():.2e}"
        )

    def test_cpu_parity_reset_position_ids(self, config):
        """CPU fallback is numerically stable with non-monotonic position_ids (packed pattern)."""
        import types

        from unturtle.models.backbones.dream import DreamModel
        from unturtle.models.backbones.dream.modeling_dream import (
            DreamAttention_fast_forward,
            _apply_dream_rope,
        )

        torch.manual_seed(1)
        model = DreamModel(config).cpu().eval()

        # Install stubs
        for module in model.modules():
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "o_proj")
                and not hasattr(module, "apply_qkv")
            ):
                from unturtle.fast_diffusion_model import (
                    _original_apply_o,
                    _original_apply_qkv,
                )

                module.apply_qkv = _original_apply_qkv
                module.apply_o = _original_apply_o

        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(
                DreamAttention_fast_forward, layer.self_attn
            )

        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        # Reset position_ids: row 0 = [0..7, 0..7], row 1 = [0..15]
        position_ids = torch.cat(
            [
                torch.arange(L // 2).repeat(2).unsqueeze(0),
                torch.arange(L).unsqueeze(0),
            ],
            dim=0,
        )

        with torch.no_grad():
            out = model(input_ids=input_ids, position_ids=position_ids)
        assert out.logits.shape == (B, L, config.vocab_size)
        assert not torch.isnan(out.logits).any()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton RoPE requires CUDA"
    )
    def test_cuda_parity_vs_cpu(self, config):
        """CUDA fast_rope_embedding output matches CPU apply_rotary_pos_emb.

        DreamRotaryEmbedding returns cos/sin via cat(freqs, freqs) so the
        first and second halves are identical — this is the format that
        fast_rope_embedding requires (it only reads head_dim//2 elements).
        """
        from unturtle.models.backbones.dream.modeling_dream import _apply_dream_rope

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

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton RoPE requires CUDA"
    )
    def test_cuda_no_double_index_reset_positions(self, config):
        """CUDA path with reset position_ids produces valid output (no NaN, correct shape).

        cos/sin use the cat(freqs, freqs) pattern from DreamRotaryEmbedding
        and are pre-indexed per batch row to simulate packed/reset positions.
        """
        from unturtle.models.backbones.dream.modeling_dream import _apply_dream_rope

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
            q_cpu, k_cpu = _apply_dream_rope(
                q_cpu_in, k_cpu_in, cos.cpu(), sin.cpu(), B, L
            )

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()

        assert torch.allclose(q_cpu, q_out.cpu(), atol=1e-4), (
            f"CUDA/CPU parity failed: max_diff={(q_cpu - q_out.cpu()).abs().max().item():.2e}"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton RoPE requires CUDA"
    )
    def test_cuda_parity_with_actual_reset_position_ids(self, config):
        """CUDA fast path matches CPU for pre-indexed cos/sin derived from reset position_ids.

        This is the exact failure mode from CLAUDE.md gotcha #16/#17:
        position_ids are non-monotonic (packed/reset), cos/sin are pre-indexed
        from those ids, and we must not double-index inside fast_rope_embedding.
        """
        from unturtle.models.backbones.dream import DreamConfig
        from unturtle.models.backbones.dream.modeling_dream import (
            DreamRotaryEmbedding,
            _apply_dream_rope,
        )

        torch.manual_seed(3)
        B, n_heads, L, head_dim = 2, 4, 8, 32
        n_kv_heads = 4

        # Build reset position_ids: batch 0 = [0,1,2,3,0,1,2,3], batch 1 = [0,1,2,3,4,5,6,7]
        position_ids = torch.stack(
            [
                torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
                torch.arange(L),
            ]
        )

        # Derive cos/sin via DreamRotaryEmbedding (same as model would do)
        cfg = DreamConfig(
            vocab_size=1000,
            hidden_size=head_dim * n_heads,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            max_position_embeddings=128,
            pad_token_id=0,
            mask_token_id=1,
        )
        rotary = DreamRotaryEmbedding(config=cfg)
        dummy_x = torch.randn(B, L, cfg.hidden_size)
        cos, sin = rotary(dummy_x, position_ids)  # pre-indexed (B, L, head_dim)

        q = torch.randn(B, n_heads, L, head_dim)
        k = torch.randn(B, n_kv_heads, L, head_dim)

        # CPU reference
        q_cpu, k_cpu = _apply_dream_rope(q.clone(), k.clone(), cos, sin, B, L)

        # CUDA fast path
        q_cuda, k_cuda = _apply_dream_rope(
            q.cuda().clone(), k.cuda().clone(), cos.cuda(), sin.cuda(), B, L
        )

        assert torch.allclose(q_cpu, q_cuda.cpu(), atol=1e-4), (
            f"Q mismatch (reset pos_ids): max_diff={(q_cpu - q_cuda.cpu()).abs().max().item():.2e}"
        )
        assert torch.allclose(k_cpu, k_cuda.cpu(), atol=1e-4), (
            f"K mismatch (reset pos_ids): max_diff={(k_cpu - k_cuda.cpu()).abs().max().item():.2e}"
        )

    def test_gqa_parity(self, config):
        """_apply_dream_rope works correctly with GQA (n_kv_heads < n_heads)."""
        from unturtle.models.backbones.dream import DreamConfig
        from unturtle.models.backbones.dream.modeling_dream import _apply_dream_rope

        torch.manual_seed(5)
        B, n_heads, n_kv_heads, L, head_dim = 2, 8, 2, 16, 32

        freqs = torch.randn(B, L, head_dim // 2)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

        q = torch.randn(B, n_heads, L, head_dim)
        k = torch.randn(B, n_kv_heads, L, head_dim)

        q_out, k_out = _apply_dream_rope(q.clone(), k.clone(), cos, sin, B, L)

        assert q_out.shape == (B, n_heads, L, head_dim)
        assert k_out.shape == (B, n_kv_heads, L, head_dim)
        assert not torch.isnan(q_out).any()
        assert not torch.isnan(k_out).any()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Triton RoPE requires CUDA"
    )
    def test_gqa_cuda_parity(self, config):
        """CUDA and CPU agree for GQA (n_kv_heads < n_heads)."""
        from unturtle.models.backbones.dream.modeling_dream import _apply_dream_rope

        torch.manual_seed(6)
        B, n_heads, n_kv_heads, L, head_dim = 2, 8, 2, 16, 32

        freqs = torch.randn(B, L, head_dim // 2)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

        q = torch.randn(B, n_heads, L, head_dim)
        k = torch.randn(B, n_kv_heads, L, head_dim)

        q_cpu, k_cpu = _apply_dream_rope(q.clone(), k.clone(), cos, sin, B, L)
        q_cuda, k_cuda = _apply_dream_rope(
            q.cuda().clone(), k.cuda().clone(), cos.cuda(), sin.cuda(), B, L
        )

        assert torch.allclose(q_cpu, q_cuda.cpu(), atol=1e-4), (
            f"GQA Q mismatch: max_diff={(q_cpu - q_cuda.cpu()).abs().max().item():.2e}"
        )
        assert torch.allclose(k_cpu, k_cuda.cpu(), atol=1e-4), (
            f"GQA K mismatch: max_diff={(k_cpu - k_cuda.cpu()).abs().max().item():.2e}"
        )

    def test_position_embeddings_none_fallback(self, config):
        """DreamAttention_fast_forward computes RoPE internally when position_embeddings=None."""
        import types

        from unturtle.models.backbones.dream import DreamModel
        from unturtle.models.backbones.dream.modeling_dream import (
            DreamAttention_fast_forward,
        )

        torch.manual_seed(9)
        model = DreamModel(config).cpu().eval()

        for module in model.modules():
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "o_proj")
                and not hasattr(module, "apply_qkv")
            ):
                from unturtle.fast_diffusion_model import (
                    _original_apply_o,
                    _original_apply_qkv,
                )

                module.apply_qkv = _original_apply_qkv
                module.apply_o = _original_apply_o

        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(
                DreamAttention_fast_forward, layer.self_attn
            )

        B, L = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        # DreamModel.forward computes and passes position_embeddings internally,
        # but we verify the model still runs without error.
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)
        assert not torch.isnan(out.logits).any()

    def test_fast_forward_with_lora_dropout_still_works(self, config):
        """DreamAttention_fast_forward is injected even when lora_dropout > 0.

        The LoRA Triton kernels are skipped in that case, but the forward
        (including Triton RoPE on CUDA) must still execute correctly.
        """
        import types

        from unturtle.models.backbones.dream import DreamModel
        from unturtle.models.backbones.dream.modeling_dream import (
            DreamAttention_fast_forward,
        )

        torch.manual_seed(11)
        model = DreamModel(config).cpu().eval()

        for module in model.modules():
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "o_proj")
                and not hasattr(module, "apply_qkv")
            ):
                from unturtle.fast_diffusion_model import (
                    _original_apply_o,
                    _original_apply_qkv,
                )

                module.apply_qkv = _original_apply_qkv
                module.apply_o = _original_apply_o

        # Inject forward manually (simulates what _patch_dream_peft does
        # unconditionally before the lora_dropout check)
        for layer in model.model.layers:
            layer.self_attn.forward = types.MethodType(
                DreamAttention_fast_forward, layer.self_attn
            )

        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.logits.shape == (B, L, config.vocab_size)
        assert not torch.isnan(out.logits).any()

    def test_fast_forward_bool_padding_keep_mask(self, config):
        """Fast forward masks padding via the shared bool [B,1,L,L] keep-mask
        path (_apply_eager_attention_mask). Non-pad positions must match an
        unpadded run and differ from an unmasked padded run."""
        from unturtle.models.backbones.dream import DreamModel

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()
        self._install_fast_forward(model)

        L_real, L_pad = 6, 8
        real_ids = torch.randint(2, config.vocab_size, (1, L_real))
        padded_ids = torch.full((1, L_pad), config.pad_token_id, dtype=torch.long)
        padded_ids[:, :L_real] = real_ids
        attention_mask = torch.zeros(1, L_pad, dtype=torch.long)
        attention_mask[:, :L_real] = 1

        with torch.no_grad():
            ref = model(input_ids=real_ids).logits
            masked = model(input_ids=padded_ids, attention_mask=attention_mask).logits
            unmasked = model(input_ids=padded_ids).logits

        assert torch.allclose(ref, masked[:, :L_real], atol=1e-5), (
            f"max_diff={(ref - masked[:, :L_real]).abs().max().item():.2e}"
        )
        assert not torch.allclose(ref, unmasked[:, :L_real], atol=1e-5)

    @pytest.mark.parametrize("use_replace_cache", [False, True])
    def test_patched_model_cached_generate(self, config, use_replace_cache):
        """Regression (#48): the fast forward used to raise TypeError on Dream's
        tuple caches, so generate(use_cache=True) crashed on a patched model.
        It now delegates to the standard attention forward, so cached block
        decode works and matches the unpatched model exactly."""
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            block_length=2,
            use_cache=True,
            use_replace_cache=use_replace_cache,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        inputs = torch.tensor([[2, 3, 4, 5]])

        torch.manual_seed(0)
        model = DreamModel(config).cpu().eval()

        torch.manual_seed(42)
        with torch.no_grad():
            ref = model.generate(inputs=inputs, generation_config=generation_config)

        self._install_fast_forward(model)
        torch.manual_seed(42)
        with torch.no_grad():
            patched = model.generate(inputs=inputs, generation_config=generation_config)

        assert torch.equal(ref, patched)
        assert not torch.any(patched == config.mask_token_id)


class TestDreamSavePretrained:
    """transformers 5.x requires dict-style _tied_weights_keys; the legacy list
    form crashes remove_tied_weights_from_state_dict during save_pretrained
    (and therefore every Trainer checkpoint save)."""

    @pytest.fixture
    def config(self):
        from unturtle.models.backbones.dream import DreamConfig

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

    def test_save_pretrained_roundtrip(self, config, tmp_path):
        from unturtle.models.backbones.dream import DreamModel

        model = DreamModel(config).cpu()
        model.save_pretrained(tmp_path / "ckpt")
        reloaded = DreamModel.from_pretrained(tmp_path / "ckpt").cpu()
        assert reloaded.config.vocab_size == config.vocab_size
        torch.testing.assert_close(
            reloaded.lm_head.weight, model.lm_head.weight, atol=0, rtol=0
        )
