# MDLM-DiT Native Diffusion Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the MDLM DiT (Diffusion Transformer from kuleshov-group/mdlm) as a native, from-scratch-trainable diffusion backbone that rides Unturtle's existing `mdlm` algorithm with no collator/trainer/sampler changes.

**Architecture:** A time-agnostic port of the adaLN-Zero DiT. `TimestepEmbedder(sigma)` is replaced by a single learnable constant conditioning vector `nn.Parameter(zeros(cond_dim))` — equivalent to the paper's `time_conditioning=False`. The model conforms to Unturtle's masked-diffusion backbone contract: `forward(input_ids, attention_mask) -> CausalLMOutputWithPast(logits)`, bidirectional (`is_causal=False`), and inherits `MaskedDiffusionGenerationMixin` for `model.generate(algorithm="mdlm")`. Native re-implementation baseline (NOT weight-compatible with kuleshov checkpoints).

**Tech Stack:** PyTorch, transformers (`PretrainedConfig`, `PreTrainedModel`, `CausalLMOutputWithPast`, Auto* registration), `unturtle.models.generation.MaskedDiffusionGenerationMixin`, pytest (CPU, tiny configs).

**Spec:** `docs/superpowers/specs/2026-06-16-mdlm-dit-backbone-design.md`
**Issue:** #31. **Branch:** `feat/31-mdlm-dit-backbone` (already checked out, holds the spec commit).

---

## File Structure

- Create: `unturtle/models/backbones/mdlm_dit/__init__.py` — public exports.
- Create: `unturtle/models/backbones/mdlm_dit/configuration_mdlm_dit.py` — `MDLMDiTConfig`.
- Create: `unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py` — layers, `MDLMDiTPreTrainedModel`, `MDLMDiTModel`, `MDLMDiTForMaskedDiffusionLM`, Auto* registration.
- Modify: `unturtle/models/backbones/__init__.py` — re-export the new symbols.
- Modify: `unturtle/fast_diffusion_model.py:648` (`_native_model_classes`) — register `"mdlm-dit"`.
- Modify: `docs/dllm-gap-map.md` — add the MDLM-DiT backbone row.
- Modify: `CLAUDE.md` — add `mdlm_dit` to the backbones list.
- Create: `tests/models/test_mdlm_dit.py` — all tests.

---

## Task 1: Config (`MDLMDiTConfig`)

**Files:**
- Create: `unturtle/models/backbones/mdlm_dit/configuration_mdlm_dit.py`
- Create: `unturtle/models/backbones/mdlm_dit/__init__.py`
- Test: `tests/models/test_mdlm_dit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_mdlm_dit.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTConfig -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'unturtle.models.backbones.mdlm_dit'`

- [ ] **Step 3: Write the config**

Create `unturtle/models/backbones/mdlm_dit/configuration_mdlm_dit.py` (Apache header as in sibling files, then):

```python
from __future__ import annotations

from transformers import PretrainedConfig


class MDLMDiTConfig(PretrainedConfig):
    """Config for the MDLM-DiT native diffusion backbone.

    Time-agnostic adaLN-Zero Diffusion Transformer (kuleshov-group/mdlm DiT,
    ``time_conditioning=False`` equivalent). Field names are HF-standard so no
    ``@property`` mapping is needed.
    """

    model_type = "mdlm-dit"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        cond_dim: int = 128,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        max_position_embeddings: int = 1024,
        mask_token_id: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # mask_token_id is not a standard PretrainedConfig arg; set after super().
        self.mask_token_id = mask_token_id
        self.architectures = self.architectures or ["MDLMDiTForMaskedDiffusionLM"]
```

Create `unturtle/models/backbones/mdlm_dit/__init__.py` (Apache header, then):

```python
"""MDLM-DiT native diffusion backbone (kuleshov-group/mdlm DiT, time-agnostic).

Reference: https://arxiv.org/abs/2406.07524 (Sahoo et al., NeurIPS 2024).
Native re-implementation baseline — not weight-compatible with the published
kuleshov checkpoints.
"""

from .configuration_mdlm_dit import MDLMDiTConfig

__all__ = ["MDLMDiTConfig"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTConfig -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add unturtle/models/backbones/mdlm_dit/ tests/models/test_mdlm_dit.py
git commit -m "✨ feat(backbones): MDLMDiTConfig for MDLM-DiT backbone (#31)"
```

---

## Task 2: Internal modules + forward (time-agnostic DiT)

**Files:**
- Create: `unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py`
- Modify: `unturtle/models/backbones/mdlm_dit/__init__.py`
- Test: `tests/models/test_mdlm_dit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_mdlm_dit.py`:

```python
@pytest.fixture
def tiny_config():
    from unturtle.models.backbones.mdlm_dit import MDLMDiTConfig

    return MDLMDiTConfig(
        vocab_size=512,
        hidden_size=64,
        cond_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0.0,
        max_position_embeddings=64,
        mask_token_id=511,
    )


class TestMDLMDiTForward:
    def test_instantiation(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu()
        assert model is not None
        assert hasattr(model, "model")

    def test_forward_logits_shape(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        B, L = 2, 16
        input_ids = torch.randint(0, tiny_config.vocab_size, (B, L))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert hasattr(out, "logits")
        assert out.logits.shape == (B, L, tiny_config.vocab_size)
        assert out.past_key_values is None

    def test_forward_is_time_agnostic(self, tiny_config):
        """forward must succeed with NO sigma/timesteps argument (Unturtle contract)."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        with torch.no_grad():
            # Passing a stray timesteps kwarg must be absorbed, not error.
            out = model(input_ids=input_ids, timesteps=torch.rand(2))
        assert out.logits.shape == (2, 8, tiny_config.vocab_size)

    def test_forward_backward(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        out = model(input_ids=input_ids)
        loss = out.logits.float().log_softmax(-1).mean().neg()
        assert not torch.isnan(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_adaln_zero_init(self, tiny_config):
        """adaLN_modulation weight & bias are zero-initialized (adaLN-Zero)."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu()
        for block in model.model.blocks:
            assert torch.all(block.adaLN_modulation.weight == 0)
            assert torch.all(block.adaLN_modulation.bias == 0)
        assert torch.all(model.model.output_layer.adaLN_modulation.weight == 0)
        assert torch.all(model.model.output_layer.adaLN_modulation.bias == 0)


def _activate_adaln(model) -> None:
    """Push adaLN gates off their zero-init so attention/MLP actually contribute.

    adaLN-Zero means every gate is 0 at init, so `bias_dropout_add_scale` returns
    the bare residual and NO cross-position mixing happens — a freshly-initialized
    model is position-independent by construction. Tests that probe attention must
    first move the modulation weights away from zero. We do that deterministically
    by filling each adaLN_modulation weight/bias with a small constant.
    """
    torch.manual_seed(0)
    for block in model.model.blocks:
        block.adaLN_modulation.weight.data.normal_(0, 0.02)
        block.adaLN_modulation.bias.data.normal_(0, 0.02)
    model.model.output_layer.adaLN_modulation.weight.data.normal_(0, 0.02)
    model.model.output_layer.adaLN_modulation.bias.data.normal_(0, 0.02)


class TestMDLMDiTAttention:
    def test_bidirectional_attention(self, tiny_config):
        """Output at position i depends on tokens AFTER i (not causal).

        Requires non-zero adaLN gates (see `_activate_adaln`): at zero-init the
        model has no cross-position mixing, so this would trivially (and falsely)
        pass-by-failing. Activating the gates makes the attention path live.
        """
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        _activate_adaln(model)
        L = 8
        a = torch.randint(0, tiny_config.vocab_size, (1, L))
        b = a.clone()
        b[0, -1] = (b[0, -1] + 1) % tiny_config.vocab_size  # perturb LAST token
        with torch.no_grad():
            out_a = model(input_ids=a).logits
            out_b = model(input_ids=b).logits
        # Position 0 must change when the last token changes => bidirectional.
        assert not torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-5)

    def test_attention_mask_2d_padding(self, tiny_config):
        """A 2-D [B,L] padding mask is accepted and changes the output.

        Also requires active adaLN gates (otherwise attention contributes nothing
        and the mask cannot affect any output).
        """
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        _activate_adaln(model)
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        full = torch.ones(1, 8, dtype=torch.long)
        partial = full.clone()
        partial[0, -2:] = 0  # mask out last two positions
        with torch.no_grad():
            out_full = model(input_ids=input_ids, attention_mask=full).logits
            out_part = model(input_ids=input_ids, attention_mask=partial).logits
        assert not torch.allclose(out_full[0, 0], out_part[0, 0], atol=1e-5)

    def test_attention_mask_4d_bool(self, tiny_config):
        """A 4-D [B,1,L,L] bool mask (as _sample passes) is accepted."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        m1d = torch.ones(1, 8, dtype=torch.bool)
        m4d = torch.logical_and(
            m1d.unsqueeze(1).unsqueeze(-2), m1d.unsqueeze(1).unsqueeze(-1)
        )  # [1,1,8,8]
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=m4d).logits
        assert out.shape == (1, 8, tiny_config.vocab_size)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTForward tests/models/test_mdlm_dit.py::TestMDLMDiTAttention -v`
Expected: FAIL — `ImportError: cannot import name 'MDLMDiTForMaskedDiffusionLM'`

- [ ] **Step 3: Write the modeling module**

Create `unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py` (Apache header, then):

```python
from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from einops import rearrange
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationMixin,
)

from .configuration_mdlm_dit import MDLMDiTConfig


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def bias_dropout_add_scale(
    x: torch.Tensor,
    scale: torch.Tensor,
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return (x * self.weight[None, None, :]).to(self.weight.dtype)


class Rotary(nn.Module):
    """RoPE cos/sin cache. Returns per-position cos/sin shaped [1, L, 1, head_dim]."""

    def __init__(self, dim: int, base: int = 10_000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, head_dim]
        cos = emb.cos()[None, :, None, :]  # [1, L, 1, head_dim]
        sin = emb.sin()[None, :, None, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, L, H, D]; cos/sin: [1, L, 1, D]
    return (x * cos) + (rotate_half(x) * sin)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


class DDiTBlock(nn.Module):
    """adaLN-Zero transformer block (bidirectional)."""

    def __init__(self, dim: int, n_heads: int, cond_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        self.flash_attn_func = None
        try:
            from flash_attn import flash_attn_func  # type: ignore

            self.flash_attn_func = flash_attn_func
        except ModuleNotFoundError:
            pass

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # q/k/v: [B, L, H, D]
        if self.flash_attn_func is not None and q.device.type == "cuda" and attn_bias is None:
            out = self.flash_attn_func(q, k, v, dropout_p=0.0, causal=False)  # [B, L, H, D]
            return rearrange(out, "b l h d -> b l (h d)")
        # SDPA expects [B, H, L, D]
        qs, ks, vs = (rearrange(t, "b l h d -> b h l d") for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            qs, ks, vs, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )
        return rearrange(out, "b h l d -> b l (h d)")

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, c: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        h = modulate(self.norm1(x), shift_msa.squeeze(1), scale_msa.squeeze(1))
        qkv = self.attn_qkv(h)
        qkv = rearrange(qkv, "b l (three h d) -> three b l h d", three=3, h=self.n_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rotary(k, cos.to(k.dtype), sin.to(k.dtype))
        attn = self._attention(q, k, v, attn_bias)
        x = bias_dropout_add_scale(
            self.attn_out(attn), gate_msa.squeeze(1), x_skip, self.dropout, self.training
        )

        h = modulate(self.norm2(x), shift_mlp.squeeze(1), scale_mlp.squeeze(1))
        x = bias_dropout_add_scale(
            self.mlp(h), gate_mlp.squeeze(1), x, self.dropout, self.training
        )
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift.squeeze(1), scale.squeeze(1))
        return self.linear(x)


class MDLMDiTModel(nn.Module):
    """Time-agnostic adaLN-Zero DiT trunk.

    The kuleshov ``TimestepEmbedder(sigma)`` is replaced by a single learnable
    constant conditioning vector (``time_conditioning=False`` equivalent).
    """

    def __init__(self, config: MDLMDiTConfig) -> None:
        super().__init__()
        self.config = config
        dim, n_heads = config.hidden_size, config.num_attention_heads
        self.vocab_embed = EmbeddingLayer(dim, config.vocab_size)
        # Constant conditioning vector (replaces sigma time-embedding).
        self.cond = nn.Parameter(torch.zeros(config.cond_dim))
        self.rotary = Rotary(dim // n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(dim, n_heads, config.cond_dim, dropout=config.dropout)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.output_layer = DDitFinalLayer(dim, config.vocab_size, config.cond_dim)

    def forward(self, input_ids: torch.Tensor, attn_bias: Optional[torch.Tensor]) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.vocab_embed(input_ids)
        c = F.silu(self.cond).unsqueeze(0).expand(B, -1)  # [B, cond_dim]
        cos, sin = self.rotary(L, input_ids.device)
        for block in self.blocks:
            x = block(x, cos, sin, c, attn_bias)
        return self.output_layer(x, c)


class MDLMDiTPreTrainedModel(PreTrainedModel):
    config_class = MDLMDiTConfig
    base_model_prefix = "model"
    _no_split_modules = ["DDiTBlock"]
    supports_gradient_checkpointing = False


def _normalize_attention_mask(
    attention_mask: Optional[torch.Tensor], dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Convert a [B,L] padding mask or [B,1,L,L] bool mask to an additive SDPA bias.

    Returns None when no positions are masked (lets the flash fast path run).
    """
    if attention_mask is None:
        return None
    if attention_mask.dim() == 2:
        # [B, L] -> [B, 1, L, L] keep-mask
        keep = torch.logical_and(
            attention_mask.bool().unsqueeze(1).unsqueeze(-2),
            attention_mask.bool().unsqueeze(1).unsqueeze(-1),
        )
    else:
        keep = attention_mask.bool()
    if keep.all():
        return None
    bias = torch.zeros_like(keep, dtype=dtype)
    bias = bias.masked_fill(~keep, float("-inf"))
    return bias


class MDLMDiTForMaskedDiffusionLM(MDLMDiTPreTrainedModel, MaskedDiffusionGenerationMixin):
    """MDLM-DiT masked-diffusion LM head.

    Bidirectional, time-agnostic. Rides the ``mdlm`` algorithm via
    ``MaskedDiffusionGenerationMixin``. No KV cache (``supports_block_decode`` is
    absent → ``_supports_block_decode`` is False).
    """

    # Opt out of the block-decode fast path: DiT has no KV cache.
    supports_block_decode = False

    def __init__(self, config: MDLMDiTConfig) -> None:
        super().__init__(config)
        self.model = MDLMDiTModel(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # absorbs timesteps / past_key_values / use_cache (time-agnostic)
    ) -> CausalLMOutputWithPast:
        attn_bias = _normalize_attention_mask(attention_mask, self.model.cond.dtype)
        logits = self.model(input_ids, attn_bias)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)


with contextlib.suppress(ValueError):
    transformers.AutoConfig.register("mdlm-dit", MDLMDiTConfig)
with contextlib.suppress(ValueError):
    transformers.AutoModel.register(MDLMDiTConfig, MDLMDiTForMaskedDiffusionLM)
with contextlib.suppress(ValueError):
    transformers.AutoModelForMaskedLM.register(MDLMDiTConfig, MDLMDiTForMaskedDiffusionLM)
```

Update `unturtle/models/backbones/mdlm_dit/__init__.py` to also export the models:

```python
from .configuration_mdlm_dit import MDLMDiTConfig
from .modeling_mdlm_dit import (
    MDLMDiTForMaskedDiffusionLM,
    MDLMDiTModel,
    MDLMDiTPreTrainedModel,
)

__all__ = [
    "MDLMDiTConfig",
    "MDLMDiTForMaskedDiffusionLM",
    "MDLMDiTModel",
    "MDLMDiTPreTrainedModel",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTForward tests/models/test_mdlm_dit.py::TestMDLMDiTAttention -v`
Expected: PASS (all forward + attention tests).

IMPORTANT correctness note: at zero-init, every adaLN gate is 0, so each block returns the bare residual (`bias_dropout_add_scale` with `scale=0`) and there is **no cross-position mixing** — a freshly-initialized model is position-independent by construction. The attention tests therefore call `_activate_adaln(model)` first to push the gates off zero; without that, `test_bidirectional_attention` / `test_attention_mask_2d_padding` would falsely fail. `test_adaln_zero_init` (in `TestMDLMDiTForward`) deliberately checks the *un-activated* model to confirm the zero-init contract. If the attention tests still fail after activation, verify RoPE is applied to BOTH q and k and that `_normalize_attention_mask` returns a `[B,1,L,L]` additive bias (not a bool mask) for SDPA.

- [ ] **Step 5: Commit**

```bash
git add unturtle/models/backbones/mdlm_dit/ tests/models/test_mdlm_dit.py
git commit -m "✨ feat(backbones): time-agnostic MDLM-DiT modeling (forward, adaLN-Zero, bidirectional) (#31)"
```

---

## Task 3: Generation (mdlm algorithm)

**Files:**
- Test: `tests/models/test_mdlm_dit.py`
- (No new source — generation comes from the inherited mixin.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_mdlm_dit.py`:

```python
class TestMDLMDiTGeneration:
    TINY_MASK_ID = 511

    @pytest.fixture
    def model(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        return MDLMDiTForMaskedDiffusionLM(tiny_config).eval()

    def test_is_generation_mixin(self, model):
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationMixin,
        )

        assert isinstance(model, MaskedDiffusionGenerationMixin)
        assert callable(model.generate)

    def test_resolve_algorithm_auto_is_mdlm(self, model):
        from unturtle.models.generation.sampler import resolve_algorithm

        assert resolve_algorithm("auto", model, bd3lm_requested=False) == "mdlm"

    def test_block_decode_not_supported(self, model):
        from unturtle.models.generation.sampler import _supports_block_decode

        assert _supports_block_decode(model) is False

    def test_generate_output_shape(self, model):
        B, L = 2, 10
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
                input_ids, steps=2, mask_token_id=self.TINY_MASK_ID, max_length=L + 1
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape == (B, L + 1)

    def test_generate_deterministic_with_seed(self, model):
        B, L = 1, 8
        input_ids = torch.full((B, L), self.TINY_MASK_ID, dtype=torch.long)
        with torch.no_grad():
            torch.manual_seed(0)
            o1 = model.generate(
                input_ids.clone(), steps=2, mask_token_id=self.TINY_MASK_ID,
                temperature=0.0, max_length=L + 1,
            )
            torch.manual_seed(0)
            o2 = model.generate(
                input_ids.clone(), steps=2, mask_token_id=self.TINY_MASK_ID,
                temperature=0.0, max_length=L + 1,
            )
        s1 = o1.sequences if hasattr(o1, "sequences") else o1
        s2 = o2.sequences if hasattr(o2, "sequences") else o2
        assert (s1 == s2).all()
```

- [ ] **Step 2: Run tests to verify they fail / pass**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTGeneration -v`
Expected: These may PASS immediately (generation is inherited). If `test_generate_output_shape` errors on output unpacking, adjust the `seq = out.sequences if ...` extraction. The point of this task is to LOCK the contract; if all pass, proceed to commit. If `resolve_algorithm` returns something other than `"mdlm"`, verify `supports_block_decode = False` is set and no `_denoising_step`/`_sample_block_diffusion`/`_model_forward_with_cache` leaked in via the mixin.

- [ ] **Step 3: Fix if needed**

If `test_block_decode_not_supported` fails, ensure the class attribute `supports_block_decode = False` is present on `MDLMDiTForMaskedDiffusionLM` (Task 2). No other source change expected.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTGeneration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_mdlm_dit.py
git commit -m "✅ test(backbones): MDLM-DiT generation rides mdlm algorithm (#31)"
```

---

## Task 4: Loader registration + save/reload round-trip

**Files:**
- Modify: `unturtle/models/backbones/__init__.py`
- Modify: `unturtle/fast_diffusion_model.py` (`_native_model_classes`, ~L648)
- Test: `tests/models/test_mdlm_dit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_mdlm_dit.py`:

```python
class TestMDLMDiTRegistration:
    def test_reexported_from_backbones(self):
        from unturtle.models.backbones import (
            MDLMDiTConfig,
            MDLMDiTForMaskedDiffusionLM,
        )

        assert MDLMDiTConfig.model_type == "mdlm-dit"
        assert MDLMDiTForMaskedDiffusionLM is not None

    def test_registered_in_native_classes(self):
        from unturtle.fast_diffusion_model import _native_model_classes

        classes = _native_model_classes()
        assert "mdlm-dit" in classes
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        assert classes["mdlm-dit"] is MDLMDiTForMaskedDiffusionLM

    def test_save_reload_forward_parity(self, tiny_config, tmp_path):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().eval()
        input_ids = torch.randint(0, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            ref = model(input_ids=input_ids).logits
        model.save_pretrained(tmp_path)
        reloaded = MDLMDiTForMaskedDiffusionLM.from_pretrained(tmp_path).cpu().eval()
        with torch.no_grad():
            got = reloaded(input_ids=input_ids).logits
        assert torch.allclose(ref, got, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTRegistration -v`
Expected: FAIL — `ImportError` (not re-exported) and `assert "mdlm-dit" in classes` fails.

- [ ] **Step 3: Wire the re-export and native registration**

In `unturtle/models/backbones/__init__.py`, add the import (after the existing `.llada` import) and the `__all__` entries:

```python
from .mdlm_dit import MDLMDiTConfig, MDLMDiTForMaskedDiffusionLM, MDLMDiTModel
```

Add to `__all__`: `"MDLMDiTConfig"`, `"MDLMDiTForMaskedDiffusionLM"`, `"MDLMDiTModel"`.

In `unturtle/fast_diffusion_model.py`, inside `_native_model_classes()` (the function starting ~L648), add a registration block mirroring the LLaDA block (use `contextlib.suppress(Exception)` if the existing blocks do, else a plain import — match the surrounding style exactly):

```python
    with contextlib.suppress(Exception):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        classes["mdlm-dit"] = MDLMDiTForMaskedDiffusionLM
```

Note: match the exact guard/style used by the LLaDA/Dream blocks already in that function (they were read to use `try`/import; replicate that form precisely rather than inventing one).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTRegistration -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add unturtle/models/backbones/__init__.py unturtle/fast_diffusion_model.py tests/models/test_mdlm_dit.py
git commit -m "✨ feat(loader): register MDLM-DiT as a native backbone + re-export (#31)"
```

---

## Task 5: DiffusionTrainer one-step training smoke

**Files:**
- Test: `tests/models/test_mdlm_dit.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/models/test_mdlm_dit.py`:

```python
class TestMDLMDiTTrainingSmoke:
    def test_one_training_step(self, tiny_config):
        """A single masked-diffusion loss + backward step runs and lowers nothing absurd.

        Mirrors DiffusionTrainer.compute_loss without spinning a full Trainer:
        forward -> fast_masked_diffusion_loss on masked positions.
        """
        from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        torch.manual_seed(0)
        model = MDLMDiTForMaskedDiffusionLM(tiny_config).cpu().train()
        B, L = 2, 12
        labels = torch.randint(0, tiny_config.vocab_size, (B, L))
        input_ids = labels.clone()
        diffusion_mask = torch.zeros(B, L, dtype=torch.bool)
        diffusion_mask[:, ::2] = True  # mask every other position
        input_ids[diffusion_mask] = tiny_config.mask_token_id

        logits = model(input_ids=input_ids).logits
        loss = fast_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_weights=None,
            loss_norm_type="token",
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTTrainingSmoke -v`
Expected: PASS if the loss kernel signature matches; if the import path/signature differs, fix the call to match `fast_masked_diffusion_loss`'s actual signature (verify by reading `unturtle/kernels/masked_diffusion_loss.py`). This test exists to prove the backbone composes with the existing SUBS loss path.

- [ ] **Step 3: Adjust call if signature differs**

If `fast_masked_diffusion_loss` rejects a kwarg, read its signature and align the call. No backbone source change.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTTrainingSmoke -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/models/test_mdlm_dit.py
git commit -m "✅ test(backbones): MDLM-DiT composes with masked-diffusion loss (#31)"
```

---

## Task 6: Docs (gap-map + CLAUDE.md)

**Files:**
- Modify: `docs/dllm-gap-map.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the gap-map**

In `docs/dllm-gap-map.md`, add a backbone row to the Gap-map table (after the Block-AR / DiffusionGemma row):

```markdown
| MDLM DiT backbone | **MDLM-DiT** (kuleshov-group/mdlm DiT; adaLN-Zero, time-agnostic native baseline) | ✅ (#31) | `unturtle/models/backbones/mdlm_dit/` | medium | maintain |
```

And add a bullet under "### Done":

```markdown
- MDLM-DiT native diffusion backbone — time-agnostic adaLN-Zero Diffusion
  Transformer (kuleshov-group/mdlm DiT, `time_conditioning=False` equivalent),
  rides the existing `mdlm` algorithm; native re-implementation baseline,
  not weight-compatible with published checkpoints (#31).
```

- [ ] **Step 2: Update CLAUDE.md backbones list**

In `CLAUDE.md`, in the "Backbone architecture" bullet under "Model taxonomy", add `mdlm_dit` to the path list, e.g. change:

```
(`unturtle.models.backbones.{llada,dream,modernbert,diffusion_gemma}`)
```

to:

```
(`unturtle.models.backbones.{llada,dream,modernbert,diffusion_gemma,mdlm_dit}`)
```

and add a sentence: "MDLM-DiT is a native, time-agnostic adaLN-Zero Diffusion Transformer baseline (kuleshov-group/mdlm DiT) trained via `DiffusionTrainer`'s SUBS objective."

Also add `mdlm_dit` to the repo-map `backbones/` comment line.

- [ ] **Step 3: Commit**

```bash
git add docs/dllm-gap-map.md CLAUDE.md
git commit -m "📚 docs: record MDLM-DiT backbone in gap-map and CLAUDE.md (#31)"
```

---

## Task 7: Lint, format, full fast-test suite

**Files:** none (verification only).

- [ ] **Step 1: Ruff format + check**

Run: `.venv/bin/python -m ruff format unturtle/models/backbones/mdlm_dit/ tests/models/test_mdlm_dit.py`
Run: `.venv/bin/python -m ruff check unturtle/models/backbones/mdlm_dit/ tests/models/test_mdlm_dit.py unturtle/fast_diffusion_model.py unturtle/models/backbones/__init__.py`
Expected: "All checks passed!" (fix any reported issues, e.g. unused imports).

- [ ] **Step 2: Run the full MDLM-DiT test module**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py -v`
Expected: all PASS.

- [ ] **Step 3: Run the focused fast suite (regression guard)**

Run: `.venv/bin/python -m pytest tests/models/ tests/diffusion/ tests/test_fast_diffusion_model.py -m "not slow" -q`
Expected: all PASS — proves the new backbone's import-time Auto* registration did not break sibling backbones or the sampler.

- [ ] **Step 4: Commit any lint fixups**

```bash
git add -A
git commit -m "🔧 chore(backbones): ruff fixups for MDLM-DiT (#31)" || echo "nothing to commit"
```

---

## Task 8: PR + review

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/31-mdlm-dit-backbone
```

- [ ] **Step 2: Open a Draft PR**

```bash
gh pr create --draft \
  --title "✨ feat(backbones): MDLM-DiT native diffusion backbone (time-agnostic baseline) (#31)" \
  --body "Implements #31. Adds the kuleshov-group/mdlm DiT as a native, time-agnostic adaLN-Zero diffusion backbone. Rides the existing mdlm algorithm; no collator/trainer/sampler changes. Native re-implementation baseline (not weight-compatible with published checkpoints). See docs/superpowers/specs/2026-06-16-mdlm-dit-backbone-design.md."
```

- [ ] **Step 3: Run pr-review-toolkit**

Run the repo PR review (code-reviewer + pr-test-analyzer + comment-analyzer, per CLAUDE.md). Focus: reference alignment (kuleshov DiT), transformers registration/tie-weights, CUDA guards on the flash path, bidirectional preservation (`is_causal=False`), test coverage. Fix all critical/high findings.

- [ ] **Step 4: Mark ready + squash merge**

After CI is green and review findings are addressed, mark ready and squash-merge (default strategy). Confirm #31 closes and the branch is deleted.

---

## Notes for the implementer

- **Always use `.venv/bin/python`**, never `uv run python` (dep re-resolution fails on this repo's `requires-python`).
- **CPU-only tests**; flash_attn is import-guarded and only used on CUDA — the SDPA branch is what CPU tests exercise.
- **Reference**: `dev/repos/mdlm/models/dit.py` (source DiT), `unturtle/models/backbones/llada/modeling_llada.py` (attention + CausalLMOutputWithPast pattern), `unturtle/models/generation/diffusion_generation_utils.py:953` (the `_sample` loop the backbone plugs into).
- **Do NOT** port kuleshov's global `torch._C._jit_set_profiling_*` flags or `@torch.jit.script` fusions (global side effects on other backbones).
- If `_native_model_classes` uses a different guard idiom than shown, **match the existing blocks exactly**.
```
