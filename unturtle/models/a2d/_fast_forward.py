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

"""
Non-causal (bidirectional) fast attention forward for A2D models.

Replaces the standard ``LlamaAttention.forward`` with a version that:
1. Dispatches through ``self.apply_qkv`` / ``self.apply_o`` so unsloth's
   Triton-fused LoRA kernels are invoked when patched in by
   ``FastDiffusionModel.patch_peft_model``.
2. Passes ``causal=False`` to Flash Attention and ``is_causal=False`` to SDPA,
   preventing any causal masking.

Signature follows the **transformers 5.x** ``LlamaAttention.forward`` API:
- ``position_embeddings`` is passed from the model level (no ``self.rotary_emb``)
- Returns a 2-tuple ``(attn_output, attn_weights)``

dLLMs require bidirectional attention because they condition on all visible
tokens (not just past tokens) to predict masked positions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache

from unsloth.kernels.rope_embedding import fast_rope_embedding
from unsloth.utils.attention_dispatch import (
    HAS_FLASH_ATTENTION,
    SDPA,
    AttentionConfig,
    AttentionContext,
    run_attention,
    select_attention_backend,
)
from unsloth.utils.packing import get_packed_info_from_kwargs

if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_varlen_func


def _rotate_half_rope(
    Q: torch.Tensor,
    K: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU-safe rotate-half RoPE.  cos/sin shape: (B, L, head_dim)."""
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    # cos/sin: (B, L, head_dim) → unsqueeze head axis → (B, 1, L, head_dim)
    cos = cos.unsqueeze(1).to(dtype=Q.dtype)
    sin = sin.unsqueeze(1).to(dtype=Q.dtype)
    Q_out = Q * cos + _rotate_half(Q) * sin
    K_out = K * cos + _rotate_half(K) * sin
    return Q_out, K_out


def _flash_varlen_packed(
    Q: torch.Tensor,              # [B, n_heads, L, head_dim]
    K: torch.Tensor,              # [B, n_kv_heads, L, head_dim]
    V: torch.Tensor,              # [B, n_kv_heads, L, head_dim]
    seq_lengths_list: list,       # list[Tensor] — seq_lengths_list[b] = int32 lengths for row b
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Compact Q/K/V to padding-free buffers, run flash_attn_varlen_func, scatter back.

    Each row b in the batch has ``sum(seq_lengths_list[b])`` real tokens followed by
    padding.  This function:
      1. Strips padding from each row and concatenates into a flat compacted buffer.
      2. Calls ``flash_attn_varlen_func`` with ``causal=False`` (bidirectional dLLM).
      3. Scatters the output back into a zero-padded ``[B, L, n_heads * head_dim]`` tensor.

    flash_attn handles GQA natively: Q has ``n_heads`` heads, K/V have ``n_kv_heads``.

    Returns:
        Tensor of shape ``[B, L, n_heads * head_dim]`` with zeros at padding positions.
    """
    bsz = Q.shape[0]
    q_len = Q.shape[2]

    # Compute real token count per row (tiny CPU ops — seq_lengths are small)
    real_counts = [int(sl.sum().item()) for sl in seq_lengths_list]
    total_tokens = sum(real_counts)

    # Build per-sample cu_seqlens for flash_attn_varlen_func.
    # Concatenate all per-row length tensors → flat [total_samples] vector.
    # cu_seqlens indexes into the compacted buffer (sample-level boundaries).
    flat_lengths = torch.cat(
        [sl.to(device=Q.device, dtype=torch.int32) for sl in seq_lengths_list]
    )  # [total_samples_across_all_rows]
    cu_seqlens = torch.zeros(flat_lengths.numel() + 1, dtype=torch.int32, device=Q.device)
    torch.cumsum(flat_lengths, dim=0, out=cu_seqlens[1:])
    max_seqlen = int(flat_lengths.max().item())

    # Compact: remove padding, concatenate real tokens from all rows.
    # Q: [B, n_heads, L, head_dim] → transpose → [B, L, n_heads, head_dim] → slice + cat
    Q_t = Q.transpose(1, 2)   # [B, L, n_heads, head_dim]
    K_t = K.transpose(1, 2)   # [B, L, n_kv_heads, head_dim]
    V_t = V.transpose(1, 2)   # [B, L, n_kv_heads, head_dim]

    Q_compact = torch.cat([Q_t[b, :real_counts[b]] for b in range(bsz)], dim=0)
    K_compact = torch.cat([K_t[b, :real_counts[b]] for b in range(bsz)], dim=0)
    V_compact = torch.cat([V_t[b, :real_counts[b]] for b in range(bsz)], dim=0)
    # shapes: [total_tokens, n_heads/n_kv_heads, head_dim]

    # Flash varlen — bidirectional (causal=False), no dropout during fine-tuning.
    attn_out = flash_attn_varlen_func(
        Q_compact,
        K_compact,
        V_compact,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_p=0.0,
        causal=False,
    )  # [total_tokens, n_heads, head_dim]

    # Scatter back into zero-padded [B, L, n_heads * head_dim].
    # new_zeros preserves dtype and device from Q.
    out_full = Q.new_zeros(bsz, q_len, n_heads * head_dim)
    offset = 0
    for b in range(bsz):
        rc = real_counts[b]
        out_full[b, :rc] = attn_out[offset:offset + rc].reshape(rc, n_heads * head_dim)
        offset += rc

    return out_full


def A2DAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Bidirectional (non-causal) fast forward for A2D attention layers.

    Dispatches through ``self.apply_qkv`` / ``self.apply_o`` hooks, enabling
    unsloth's Triton-fused LoRA kernels when they are patched in.
    Attention is fully bidirectional (``causal=False``).
    """
    bsz, q_len, _ = hidden_states.size()

    n_heads = self.config.num_attention_heads
    n_groups = self.num_key_value_groups
    n_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    assert n_kv_heads * n_groups == n_heads

    # Dispatch through apply_qkv — uses Triton fused kernel when patched
    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    seq_info = get_packed_info_from_kwargs(kwargs, Q.device)

    kv_seq_len = K.shape[-2]
    if past_key_values is not None:
        kv_seq_len += past_key_values.get_seq_length(self.layer_idx)

    # Apply rotary embeddings
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if Q.device.type == "cuda":
            Q, K = fast_rope_embedding(Q, K, cos, sin, position_ids)
        else:
            # CPU fallback: plain rotate_half RoPE
            Q, K = _rotate_half_rope(Q, K, cos, sin, position_ids)
    # If position_embeddings is None, skip RoPE (shouldn't happen for A2D models
    # since A2DLlamaModel.forward always computes and passes them)

    if past_key_values is not None:
        cache_kwargs = {"cache_position": cache_position}
        K, V = past_key_values.update(K, V, self.layer_idx, cache_kwargs)

    # Determine attention backend.
    #
    # Packed sequence handling (seq_info present, no KV cache):
    #   The 4D attention_mask from A2DLlamaModel.forward is an all-ones padding mask for
    #   packed inputs — it is semantically irrelevant and must not be passed to the kernel.
    #
    #   Flash varlen requires padding-free compacted Q/K/V (total_tokens, heads, dim).
    #   The current batched tensors are [B, L] with padding, so feeding them to flash
    #   varlen would produce metadata mismatches when any row is not fully packed.
    #   Flash varlen support with proper compaction is deferred to a follow-up PR.
    #
    #   SDPA packed path: use block_attention_mask from the collator ([B, 1, L, L] bool).
    #   Do NOT fall back to build_sdpa_packed_attention_mask() — that upstream helper
    #   builds causal (upper-triangular) blocks, which is incorrect for bidirectional dLLM.
    #   If block_attention_mask is absent (non-packed forward), use the standard mask path.
    use_varlen = seq_info is not None and past_key_values is None

    if use_varlen:
        seq_lengths_list = kwargs.get("seq_lengths", None)
        if HAS_FLASH_ATTENTION and seq_lengths_list is not None:
            # Flash varlen path: compact padding tokens out, call flash_attn_varlen_func
            # directly (bypasses run_attention which does uncompacted reshape).
            A = _flash_varlen_packed(
                Q, K, V,
                seq_lengths_list=seq_lengths_list,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
            )
            # _flash_varlen_packed returns [B, L, n_heads * head_dim] already
            attn_output = self.apply_o(self, A)
            return attn_output, None

        # SDPA fallback: flash not available or seq_lengths absent.
        # Use block_attention_mask (bidirectional block-diagonal) from collator when present.
        # Do NOT fall back to build_sdpa_packed_attention_mask() — it builds causal blocks.
        effective_mask = kwargs.get("block_attention_mask", None)
        backend = SDPA
    else:
        effective_mask = attention_mask
        backend = SDPA if attention_mask is not None else select_attention_backend(False)

    # Key difference from LlamaAttention: causal=False everywhere.
    # sdpa_kwargs prevents SDPA from auto-inferring causal when q_len == k_len.
    config = AttentionConfig(
        backend=backend,
        n_kv_heads=n_kv_heads,
        n_groups=n_groups,
        flash_dense_kwargs={"causal": False},
        flash_varlen_kwargs={"dropout_p": 0.0, "causal": False},
        sdpa_kwargs={"is_causal": False},
    )
    context = AttentionContext(
        bsz=bsz,
        q_len=q_len,
        kv_seq_len=K.shape[-2],
        n_heads=n_heads,
        head_dim=head_dim,
        requires_grad=hidden_states.requires_grad,
        seq_info=seq_info,
        attention_mask=effective_mask,
        causal_mask=None,
    )

    A = run_attention(config=config, context=context, Q=Q, K=K, V=V)
    attn_output = A.reshape(bsz, q_len, n_heads * head_dim)
    # Dispatch through apply_o — uses Triton fused kernel when patched
    attn_output = self.apply_o(self, attn_output)
    return attn_output, None
