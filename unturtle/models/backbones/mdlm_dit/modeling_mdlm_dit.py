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

"""Time-agnostic MDLM-DiT modeling (kuleshov-group/mdlm DiT, time-agnostic variant).

The kuleshov ``forward(indices, sigma)`` adaLN conditioning on ``sigma`` is dropped
entirely: Unturtle replaces the sigma path with a single learnable constant
conditioning vector, so the model matches the backbone contract
``forward(input_ids, attention_mask)``. This is Unturtle's own simplification — it is
*functionally* (not structurally) equivalent to kuleshov's ``time_conditioning=False``,
which zeroes sigma but still runs it through ``TimestepEmbedder``; both collapse to a
per-forward constant ``c``. The adaLN-Zero modulation machinery (shift/scale/gate,
zero-initialized) is retained. Attention is bidirectional (``is_causal=False``). No
global jit fusion flags are set.
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
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

    def forward(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(
        self, dim: int, n_heads: int, cond_dim: int, dropout: float = 0.1
    ) -> None:
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
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # q/k/v: [B, L, H, D]
        b, length = q.shape[0], q.shape[1]
        if (
            self.flash_attn_func is not None
            and q.device.type == "cuda"
            and attn_bias is None
        ):
            out = self.flash_attn_func(
                q, k, v, dropout_p=0.0, causal=False
            )  # [B,L,H,D]
            return out.reshape(b, length, -1)
        # SDPA expects [B, H, L, D]
        qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            qs, ks, vs, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
        )
        return out.transpose(1, 2).reshape(b, length, -1)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        c: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        h = modulate(self.norm1(x), shift_msa.squeeze(1), scale_msa.squeeze(1))
        b, length = h.shape[0], h.shape[1]
        qkv = self.attn_qkv(h)
        # [B, L, 3*H*D] -> [3, B, L, H, D]
        qkv = qkv.view(b, length, 3, self.n_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary(q, cos.to(q.dtype), sin.to(q.dtype))
        k = apply_rotary(k, cos.to(k.dtype), sin.to(k.dtype))
        attn = self._attention(q, k, v, attn_bias)
        # gate_* keep their [B, 1, dim] shape so they broadcast over the L dimension.
        x = bias_dropout_add_scale(
            self.attn_out(attn), gate_msa, x_skip, self.dropout, self.training
        )

        h = modulate(self.norm2(x), shift_mlp.squeeze(1), scale_mlp.squeeze(1))
        x = bias_dropout_add_scale(
            self.mlp(h), gate_mlp, x, self.dropout, self.training
        )
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int) -> None:
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        # NOTE: kuleshov zero-inits this final projection, which (with adaLN-Zero
        # gates) makes the whole network output identically zero at init. That is
        # fine for their training-from-scratch loop but leaves a freshly built model
        # unable to produce input-dependent logits. As a native Unturtle baseline
        # (not weight-compatible with the published checkpoints) we keep the
        # adaLN-Zero *gate* contract (modulation weights/bias zeroed) but give the
        # output projection a standard init so forward() is meaningfully exercised.
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift.squeeze(1), scale.squeeze(1))
        return self.linear(x)


class MDLMDiTModel(nn.Module):
    """Time-agnostic adaLN-Zero DiT trunk.

    Unturtle drops the kuleshov sigma path entirely: rather than feeding a zeroed
    sigma through ``TimestepEmbedder`` (what kuleshov's ``time_conditioning=False``
    does — it zeroes sigma but still runs the MLP), the whole conditioning is a single
    learnable constant vector ``cond``. Both yield a per-forward constant ``c``, so the
    model is functionally time-agnostic.
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

    def forward(
        self, input_ids: torch.Tensor, attn_bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
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

    def _init_weights(self, module) -> None:
        # No-op: every submodule self-initializes in its own ``__init__`` (adaLN-Zero
        # gates are zero-initialized there). Re-initializing here via ``post_init`` would
        # break the zero-init contract, so weight init is intentionally a no-op.
        return


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
    # Use finfo.min, not -inf: SDPA can emit NaNs on fully-masked query rows
    # with -inf (see unturtle/models/backbones/llada/modeling_llada.py NOTE).
    bias = bias.masked_fill(~keep, torch.finfo(dtype).min)
    return bias


class MDLMDiTForMaskedDiffusionLM(
    MDLMDiTPreTrainedModel, MaskedDiffusionGenerationMixin
):
    """MDLM-DiT masked-diffusion LM head.

    Bidirectional, time-agnostic. Rides the ``mdlm`` algorithm via
    ``MaskedDiffusionGenerationMixin``. No KV cache (``supports_block_decode`` is
    False).
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
    transformers.AutoModelForMaskedLM.register(
        MDLMDiTConfig, MDLMDiTForMaskedDiffusionLM
    )
