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
    build_hybrid_prefix_attention_mask,
    build_sdpa_packed_attention_mask,
    build_sdpa_packed_bidirectional_attention_mask,
    build_xformers_block_bidirectional_mask,
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


# Base class covering both the causal (BlockDiagonalCausalMask) and the
# non-causal (BlockDiagonalMask) packed biases — the causal mask subclasses
# the non-causal one in xformers.
XFORMERS_BLOCK_DIAG_CLS = (
    getattr(
        xformers.attn_bias,
        "BlockDiagonalMask",
        xformers.attn_bias.BlockDiagonalCausalMask,
    )
    if HAS_XFORMERS
    else None
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

    # Bidirectional packed attention uses the non-causal BlockDiagonalMask
    # bias.  When it cannot be built (xformers class unavailable, or a
    # sliding window is set — xformers has no *bidirectional* local block
    # bias), fall back to the SDPA packed path which builds a dense
    # bidirectional block mask below.
    if (
        backend == XFORMERS
        and context.seq_info is not None
        and not config.causal
        and build_xformers_block_bidirectional_mask(
            context.seq_info, sliding_window=context.sliding_window
        )
        is None
    ):
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
        if not config.causal and context.seq_info is not None:
            # Non-causal packed: block-diagonal bias WITHOUT the causal
            # triangle (guaranteed non-None by the routing check above).
            attn_bias = build_xformers_block_bidirectional_mask(
                context.seq_info, sliding_window=sliding_window
            )
        else:
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


def hybrid_prefix_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    prompt_lengths: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """PreDiff-LM eq. (3) attention without materializing its mask (#63).

    Equation (3) splits **by query row** into two independent blocks::

        prompt queries -> prompt keys   causal        (target keys blocked)
        target queries -> all keys      bidirectional

    A prompt query never attends to a target key, so the prompt rows are
    exactly a causal attention over the prompt prefix *alone*; a target query
    attends to everything, so the target rows are exactly an unmasked attention
    over the full sequence.  Neither call needs a bias tensor, which is the
    whole point: a caller-supplied mask forces SDPA (see `run_attention`), so
    the dense form gives up Flash and xFormers and allocates an ``L x L`` bias
    per row.

    Measured on CUDA (bf16, 16 heads, head_dim 64, Lp = L/4), arms interleaved
    with `gc.collect()` between them, median of 5 replicates.  Two baselines,
    because they answer different questions:

        B x L      build+attn   hoisted   fast ms   vs build   vs hoisted
        4 x 1024        0.260     0.173     0.122      2.20x        1.42x
        4 x 2048        0.847     0.678     0.327      2.72x        2.07x
        8 x 2048        1.912     1.849     0.882      2.58x        2.10x
        2 x 4096        2.083     1.669     0.827      2.98x        2.02x

    "build+attn" rebuilds the mask inside the timed region; "hoisted" reuses a
    prebuilt one, isolating the attention win alone.

    **The hoisted column is the fair per-layer comparison.**
    `TinyA2D*Model.forward` builds the mask *once per forward* and shares it
    across every layer, so the build cost amortizes over `num_hidden_layers`
    and the rebuild column flatters this change.  The honest attention-only
    number is **~1.4-2.1x**; the larger figure additionally counts avoiding one
    `L x L` construction per forward, which is a real saving but not an
    attention one and is divided by the layer count in practice.

    Peak activation memory falls too (120 -> 85 MiB at 4 x 2048, 184 -> 85 MiB
    at 2 x 4096), which is the un-materialized bias.

    **Uniform vs ragged prompt lengths.**  The split point is a slice index, so
    one boundary shared by the batch takes two batched calls.  Ragged
    boundaries would need ``2B`` calls, which is slower than a single dense
    SDPA call, so those fall back to the dense mask.  The *answer* is identical
    either way -- verified to 2.2e-16 -- only the speed differs.

    **Not yet wired into the Tiny-A2D forwards.**  Those build the dense mask
    in ``TinyA2D*Model.forward`` and hand it down through every decoder layer,
    so routing them here needs ``prompt_lengths`` threaded through the layer
    stack -- a change across three shipped model families, separable from the
    kernel and deliberately left to its own slice.  Recovering the boundary
    from the mask instead was considered and rejected: it is derivable for
    ``0 < Lp < L`` but silently yields ``L - 1`` for an all-prompt row (the
    last causal row blocks nothing), and it re-derives what the caller already
    knew.

    Args:
        Q, K, V:         ``[B, H, L, D]``.
        prompt_lengths:  ``[B]`` prompt prefix length per row.
        attention_mask:  Optional ``[B, L]`` padding mask.  Padding breaks the
                         pure row split (an excluded key is excluded for every
                         query), so this also takes the dense path.

    Returns:
        ``[B, H, L, D_v]`` — the trailing dim follows ``V``, since the output
        is a weighted sum of value vectors.

        Agrees with attending through
        :func:`build_hybrid_prefix_attention_mask` to 0.0 in fp32/fp16/bf16
        and 4.4e-16 in fp64.  Not *mathematically* identical, though: the
        dense path blocks with ``finfo.min``, a large finite penalty rather
        than ``-inf``, so it admits a ~1e-38-weight contribution from blocked
        keys that this path omits entirely.  Where they differ, this one is
        the more correct.
    """
    lengths = prompt_lengths.reshape(-1)
    seq_len = Q.shape[-2]
    batch_size = Q.shape[0]

    # Validate BEFORE branching, so the fast and dense paths agree on errors
    # as well as on answers.  `build_hybrid_prefix_attention_mask` raises for
    # an out-of-range boundary; without this the same call would raise or
    # silently clamp depending only on whether the batch happened to have
    # uniform lengths -- the two branches drifting apart in exactly the way
    # the fallback design exists to prevent.  (Measured before the check:
    # Lp=-1 silently returned full bidirectional attention and Lp>L returned
    # full causal.)
    if lengths.shape[0] != batch_size:
        raise ValueError(
            f"prompt_lengths has {lengths.shape[0]} entries but the batch has "
            f"{batch_size} rows; one boundary broadcast across the batch would "
            "split every other row in the wrong place"
        )
    if lengths.numel() and bool(((lengths < 0) | (lengths > seq_len)).any()):
        raise ValueError(
            f"prompt_lengths must satisfy 0 <= Lp <= seq_len={seq_len}, got "
            f"{lengths.tolist()}"
        )

    # `B == 0` is a legitimate degenerate shape; there is no row to split and
    # `lengths[0]` below would raise an opaque IndexError.  The result takes
    # **V's** head_dim, not Q's -- attention output is a weighted sum of value
    # vectors, so `Q.clone()` here returned the wrong trailing dim whenever
    # `D_v != D_qk`.  Invisible while every test builds Q/K/V with equal dims.
    if batch_size == 0:
        return V.new_empty((0, V.shape[1], seq_len, V.shape[3]))

    # `bool(...)` on a CUDA tensor forces a device sync: measured 28 us in
    # isolation against 14 us for the same check on CPU.  In situ it costs far
    # less than that suggests -- scaling queued GPU work does not scale the
    # gap (measured at queue depths 1/4/16: -0.22, -0.32, +0.99 ms, i.e. noise
    # around zero), because the sync overlaps with work already in flight.
    # Callers that own the boundary can avoid it outright by keeping the small
    # `[B]` `prompt_lengths` on CPU; the slice index below is a Python int
    # either way.  Not cached here: this function is stateless and the tensor
    # may change between calls.
    uniform = bool((lengths == lengths[0]).all()) if lengths.numel() else True
    if attention_mask is not None or not uniform:
        # Padding and ragged boundaries both break the batched two-call form.
        # Falling back keeps one implementation of the semantics rather than
        # two that can drift apart.
        mask = build_hybrid_prefix_attention_mask(
            prompt_lengths=lengths,
            seq_len=seq_len,
            dtype=Q.dtype,
            device=Q.device,
            attention_mask=attention_mask,
        )
        return scaled_dot_product_attention(Q, K, V, attn_mask=mask)

    prompt_len = int(lengths[0])

    # The degenerate ends are not special cases bolted on -- they are what the
    # split reduces to, and returning early avoids a zero-length SDPA call
    # (which some backends reject outright).
    if prompt_len <= 0:
        return scaled_dot_product_attention(Q, K, V)
    if prompt_len >= seq_len:
        return scaled_dot_product_attention(Q, K, V, is_causal=True)

    # K/V are sliced to the prompt as well as Q.  That slice is *not* needed
    # for correctness -- `is_causal` with `q_len < kv_len` aligns top-left in
    # torch, so keys past the prompt are already blocked for these rows, and
    # passing full K/V is bit-identical (verified at several shapes; a mutant
    # removing the slice cannot be killed by any test).  It is a performance
    # choice: the prompt block becomes `Lp x Lp` instead of `Lp x L`, which is
    # part of where the measured speedup comes from.  Do not "simplify" it
    # away, and do not rely on the top-left alignment for correctness either --
    # it is a torch convention, not a guarantee this code should lean on.
    prompt = scaled_dot_product_attention(
        Q[:, :, :prompt_len],
        K[:, :, :prompt_len],
        V[:, :, :prompt_len],
        is_causal=True,
    )
    target = scaled_dot_product_attention(Q[:, :, prompt_len:], K, V)
    return torch.cat((prompt, target), dim=2)


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
    "build_hybrid_prefix_attention_mask",
    "build_xformers_block_bidirectional_mask",
    "hybrid_prefix_attention",
    "build_xformers_block_causal_mask",
    "run_attention",
    "select_attention_backend",
]
