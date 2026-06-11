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

"""Block Diffusion attention mask construction.

Builds the specialized attention mask for BD3LM training where the input is a
concatenated ``[x_t, x_0]`` sequence of length ``2L``.  The mask is the union of:

  - **M_BD** (Block Diagonal): self-attention within noised blocks and within clean blocks
  - **M_OBC** (Offset Block Causal): cross-attention from noised block k to clean blocks 0..k-1
  - **M_BC** (Block Causal): causal (lower-triangular over blocks) within clean positions

Reference:
    dev/repos/dllm/dllm/core/trainers/bd3lm.py  _create_bd3lm_attention_mask
    Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models
    https://arxiv.org/abs/2503.09573
"""

from __future__ import annotations

import torch


def create_block_diffusion_attention_mask(
    seq_len: int,
    block_size: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build the Block Diffusion attention mask for a ``[x_t, x_0]`` sequence.

    Args:
        seq_len:   Length of the *original* sequence ``L``.
                   The concatenated sequence has length ``2L``.
        block_size: Size of each block for partitioning.
        device:    Target device for the returned tensor.

    Returns:
        Boolean tensor of shape ``(1, 1, 2*seq_len, 2*seq_len)`` where ``True``
        means attention is allowed.
    """
    if seq_len % block_size != 0:
        raise ValueError(
            f"seq_len ({seq_len}) must be divisible by block_size ({block_size})."
        )
    total_len = 2 * seq_len
    q_idx = torch.arange(total_len, device=device)[:, None]  # [2L, 1]
    kv_idx = torch.arange(total_len, device=device)[None, :]  # [1, 2L]

    # x_0 positions: second half (L .. 2L-1)
    x0_flag_q = q_idx >= seq_len
    x0_flag_kv = kv_idx >= seq_len

    # Block indices: offset for x_0 positions so blocks restart at 0
    block_q = torch.where(
        x0_flag_q, (q_idx - seq_len) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv, (kv_idx - seq_len) // block_size, kv_idx // block_size
    )

    # M_BD: self-attention within same block, same domain (x_t or x_0)
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # M_OBC: cross-attention from x_t block k to x_0 blocks 0..k-1
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q

    # M_BC: causal over blocks within x_0
    block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q

    mask = block_diagonal | offset_block_causal | block_causal
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 2L, 2L]
