# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Vendored from unsloth/utils/attention_dispatch.py for issue #67.

"""Shared helpers for attention backend selection and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from unsloth.models._utils import HAS_FLASH_ATTENTION, xformers, xformers_attention

from .packing import (
    build_sdpa_packed_attention_mask,
    build_sdpa_packed_bidirectional_attention_mask,
    build_xformers_block_causal_mask,
)

if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
else:
    flash_attn_func = None
    flash_attn_varlen_func = None
HAS_XFORMERS = xformers is not None

if HAS_XFORMERS and torch.cuda.is_available():
    _cc = torch.cuda.get_device_capability()
    if _cc[0] >= 12:
        HAS_XFORMERS = False
SDPA_HAS_GQA = "enable_gqa" in (scaled_dot_product_attention.__doc__ or "")

FLASH_VARLEN = "flash_varlen"
FLASH_DENSE = "flash_dense"
XFORMERS = "xformers"
SDPA = "sdpa"


XFORMERS_BLOCK_DIAG_CLS = (
    xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None
)


@dataclass
class AttentionConfig:
    backend: str
    n_kv_heads: int
    n_groups: int
    flash_dense_kwargs: Optional[dict[str, Any]] = None
    flash_varlen_kwargs: Optional[dict[str, Any]] = None
    sdpa_kwargs: Optional[dict[str, Any]] = None
    xformers_kwargs: Optional[dict[str, Any]] = None
    # Whether run_attention may inject *causal* masking on the SDPA path
    # (packed block mask, 2-D mask expansion, and the is_causal fallback).
    # Defaults to True to preserve the vendored unsloth (AR) semantics.
    # Bidirectional dLLM callers MUST set causal=False — with it, run_attention
    # never constructs a causal mask on their behalf.  Note this flag does NOT
    # rewrite caller-supplied kwargs (flash_*_kwargs["causal"] /
    # sdpa_kwargs["is_causal"]); keep those consistent with this field.
    causal: bool = True


@dataclass
class AttentionContext:
    bsz: int
    q_len: int
    kv_seq_len: int
    n_heads: int
    head_dim: int
    requires_grad: bool
    seq_info: Optional[Tuple[Tensor, Tensor, int]]
    attention_mask: Optional[Tensor]
    causal_mask: Optional[Any]
    sliding_window: Optional[int] = None


def select_attention_backend(
    use_varlen: bool = False,
    *,
    device_type: str,
) -> str:
    if device_type == "cuda" and HAS_FLASH_ATTENTION:
        if use_varlen:
            return FLASH_VARLEN
        return FLASH_DENSE
    if device_type == "cuda" and HAS_XFORMERS:
        return XFORMERS
    return SDPA


def run_attention(
    *,
    config: AttentionConfig,
    context: AttentionContext,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
) -> Tensor:
    backend = config.backend
    if backend == FLASH_VARLEN and context.seq_info is None:
        backend = FLASH_DENSE if HAS_FLASH_ATTENTION else SDPA

    if context.attention_mask is not None and backend in (
        FLASH_DENSE,
        FLASH_VARLEN,
        XFORMERS,
    ):
        backend = SDPA

    if backend == XFORMERS and context.seq_info is not None and not config.causal:
        # The xformers packed path builds a *causal* block-diagonal bias
        # (build_xformers_block_causal_mask).  Bidirectional configs must not
        # receive causal masking, so fall back to the SDPA packed path which
        # builds a bidirectional block mask below.
        backend = SDPA

    flash_dense_kwargs = config.flash_dense_kwargs or {}
    flash_varlen_kwargs = config.flash_varlen_kwargs or {}
    sdpa_kwargs = config.sdpa_kwargs or {}
    xformers_kwargs = config.xformers_kwargs or {}

    bsz = context.bsz
    n_heads = context.n_heads
    q_len = context.q_len
    head_dim = context.head_dim
    kv_seq_len = context.kv_seq_len
    requires_grad = context.requires_grad
    sliding_window = context.sliding_window

    if backend == FLASH_VARLEN:
        Q_f = Q.transpose(1, 2).reshape(bsz * q_len, n_heads, head_dim)
        K_f = K.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        V_f = V.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        _, cu_seqlens, max_seqlen = context.seq_info
        return flash_attn_varlen_func(
            Q_f,
            K_f,
            V_f,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            **flash_varlen_kwargs,
        ).view(bsz, q_len, n_heads, head_dim)
    if backend == FLASH_DENSE:
        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        return flash_attn_func(Q_t, K_t, V_t, **flash_dense_kwargs).reshape(
            bsz, q_len, n_heads, head_dim
        )
    if backend == XFORMERS:
        attn_bias = build_xformers_block_causal_mask(
            context.seq_info,
            sliding_window=sliding_window,
            base_mask=context.causal_mask,
        )

        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)

        K_mod = K_t
        V_mod = V_t
        Q_mod = Q_t

        if config.n_groups != 1:
            K_mod = K_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            V_mod = V_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            K_mod = K_mod.expand(
                bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
            )
            V_mod = V_mod.expand(
                bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
            )

            if requires_grad:
                K_mod = K_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q_mod = Q_t.view(
                    bsz, q_len, config.n_kv_heads, config.n_groups, head_dim
                )

        has_block = XFORMERS_BLOCK_DIAG_CLS is not None and isinstance(
            attn_bias, XFORMERS_BLOCK_DIAG_CLS
        )

        if config.n_groups != 1 and has_block:
            if not requires_grad:
                Q_mod = Q_mod.view(
                    1, bsz * q_len, config.n_kv_heads, config.n_groups, head_dim
                )
                K_mod = K_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
                V_mod = V_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
            else:
                Q_mod = Q_mod.view(1, bsz * q_len, n_heads, head_dim)
                K_mod = K_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)

        out = xformers_attention(
            Q_mod,
            K_mod,
            V_mod,
            attn_bias=attn_bias,
            **xformers_kwargs,
        )

        if config.n_groups != 1 and not requires_grad:
            out = out.view(bsz, q_len, config.n_kv_heads, config.n_groups, head_dim)
            out = out.reshape(bsz, q_len, n_heads, head_dim)
        else:
            out = out.view(bsz, q_len, n_heads, head_dim)
        return out

    local_mask = context.attention_mask
    is_causal_local = False
    if context.seq_info is not None and local_mask is None:
        packed_mask_builder = (
            build_sdpa_packed_attention_mask
            if config.causal
            else build_sdpa_packed_bidirectional_attention_mask
        )
        local_mask = packed_mask_builder(
            context.seq_info,
            dtype=Q.dtype,
            device=Q.device,
            sliding_window=sliding_window,
        )
        # Debug guard (#49): packed lengths silently summing past the actual
        # sequence length would otherwise surface as an opaque SDPA shape error.
        if local_mask.shape[-1] != K.shape[-2]:
            raise ValueError(
                f"Packed seq_info lengths sum to {local_mask.shape[-1]} but the "
                f"key sequence length is {K.shape[-2]}; packed metadata does "
                "not match the batch."
            )
    else:
        q_len_local = Q.shape[-2]
        k_len_local = K.shape[-2]
        if local_mask is not None and isinstance(local_mask, torch.Tensor):
            local_mask = local_mask.to(device=Q.device)

            if local_mask.dim() == 2:
                if local_mask.dtype == torch.bool:
                    key_keep = local_mask
                else:
                    key_keep = local_mask != 0

                past_len = k_len_local - q_len_local
                q_pos = torch.arange(past_len, past_len + q_len_local, device=Q.device)
                k_pos = torch.arange(k_len_local, device=Q.device)

                if config.causal:
                    pos_keep = k_pos[None, :] <= q_pos[:, None]
                    if sliding_window is not None:
                        pos_keep &= k_pos[None, :] >= (
                            q_pos[:, None] - (sliding_window - 1)
                        )
                elif sliding_window is not None:
                    # Bidirectional: symmetric band, never a causal triangle.
                    pos_keep = (q_pos[:, None] - k_pos[None, :]).abs() < sliding_window
                else:
                    pos_keep = torch.ones(
                        (q_len_local, k_len_local),
                        dtype=torch.bool,
                        device=Q.device,
                    )

                local_mask = pos_keep[None, None, :, :] & key_keep[:, None, None, :]

            elif local_mask.dim() == 3:
                local_mask = local_mask[:, None, :, :]

            elif local_mask.dim() == 4:
                if local_mask.dtype != torch.bool:
                    local_mask = local_mask.eq(0)
            else:
                raise ValueError(
                    f"Unsupported SDPA attention_mask rank: {local_mask.dim()}"
                )

            if local_mask.dtype == torch.bool:
                no_allowed = ~local_mask.any(dim=-1, keepdim=True)
                local_mask = local_mask | no_allowed

        is_causal_local = (
            config.causal and local_mask is None and q_len_local == k_len_local
        )

    kwargs = dict(sdpa_kwargs)
    kwargs.setdefault("attn_mask", local_mask)
    kwargs.setdefault("is_causal", is_causal_local)

    use_sdpa_gqa = SDPA_HAS_GQA and config.n_groups != 1
    if (
        use_sdpa_gqa
        and (not requires_grad)
        and isinstance(local_mask, torch.Tensor)
        and local_mask.dim() >= 3
        and local_mask.shape[0] > 1
    ):
        use_sdpa_gqa = False

    if use_sdpa_gqa:
        kwargs.setdefault("enable_gqa", True)
        out = scaled_dot_product_attention(Q, K, V, **kwargs)
        return out.transpose(1, 2)

    K_mod = K
    V_mod = V
    if config.n_groups != 1:
        K_mod = K[:, :, None, :, :].expand(
            bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
        )
        V_mod = V[:, :, None, :, :].expand(
            bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
        )
        K_mod = K_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)
        V_mod = V_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)

    out = scaled_dot_product_attention(
        Q.contiguous(),
        K_mod.contiguous(),
        V_mod.contiguous(),
        **kwargs,
    )
    return out.transpose(1, 2).contiguous()


__all__ = [
    "AttentionConfig",
    "AttentionContext",
    "FLASH_DENSE",
    "FLASH_VARLEN",
    "HAS_FLASH_ATTENTION",
    "HAS_XFORMERS",
    "SDPA",
    "XFORMERS",
    "build_sdpa_packed_attention_mask",
    "build_sdpa_packed_bidirectional_attention_mask",
    "build_xformers_block_causal_mask",
    "run_attention",
    "select_attention_backend",
]
