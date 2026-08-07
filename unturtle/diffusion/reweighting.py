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
CART — Context-Adaptive Token-Level Noise Rescheduling.

Dream's key training innovation for improving dLLM quality on structured
outputs (code, math).  Instead of applying a uniform diffusion weight to
all masked tokens, CART dynamically re-weights each masked position based
on its proximity to clean (unmasked) context tokens.

The weight for a masked position ``n`` is:

    w(n) = 0.5 * Σ_i  p * (1-p)^(|n-i| - 1)  for all clean positions i ≠ n

where ``p = cart_p ∈ (0, 1]`` controls sharpness (locality):
  - Larger ``p`` → weight decays rapidly with distance → more local.
  - Smaller ``p`` → weight spreads further across the sequence.

Masked positions that are not adjacent to any clean context (fully masked
blocks) receive a near-zero weight, so the loss focuses on boundary tokens
where context is informative.

Usage in DiffusionTrainer::

    args = DiffusionTrainingArguments(
        loss_weight_type="cart",
        cart_p=0.8,
    )

Reference implementation:
    dev/repos/Dream/src/trainer/fsdp_sft_trainer.py  context_adaptive_reweight()
    dev/repos/Dream/src/trainer/fsdp_sft_trainer.py  L805–821 (weight application)

Paper:
    Dream: Scaling Diffusion Language Models via Adaptation from Autoregressive Models
    https://arxiv.org/abs/2508.15487
"""

from __future__ import annotations

import math
from typing import Any

import torch


def context_adaptive_reweight(
    seq_len: int,
    cart_p: float = 0.8,
) -> torch.Tensor:
    """Build the CART weight matrix for a given sequence length.

    Returns an ``(L, L)`` float32 tensor ``M`` where ``M[n, i]`` is the
    contribution of clean position ``i`` to the weight of masked position ``n``.

    To obtain per-token weights for a batch, multiply by a binary clean-position
    mask::

        clean_mask = ~diffusion_mask   # [B, L] — True at clean positions
        weight = clean_mask.float() @ M  # [B, L]
        weight = weight.masked_fill(clean_mask, 0.0)  # zero out clean positions

    The matrix is computed on CPU and moved to the target device by the caller.
    For a fixed ``seq_len`` within a training run, cache the result externally.

    Args:
        seq_len:  Sequence length ``L``.
        cart_p:   Geometric distribution parameter ``p ∈ (0, 1]``.
                  Controls how quickly weight decays with distance.

    Returns:
        Float32 tensor of shape ``(seq_len, seq_len)``.

    Raises:
        ValueError: If ``cart_p`` is not in ``(0, 1]``.
    """
    if not (0 < cart_p <= 1.0):
        raise ValueError(f"cart_p must be in (0, 1], got {cart_p}")

    # position_ids_l[n, i] = n - i  (signed distance from i to n)
    positions = torch.arange(seq_len, dtype=torch.float32)
    distance = positions.unsqueeze(1) - positions.unsqueeze(0)  # [L, L], M[n, i] = n-i
    abs_dist = distance.abs()

    if cart_p == 1.0:
        # Special-case: Geo(1, k) = p*(1-p)^(k-1) = 1*0^(k-1), which is 1 for k=1
        # and 0 for k>1 (and 0 for k=0 by convention).
        # So w(k) = 0.5 * [k == 1].  Avoid log(0) NaN from (abs_dist-1)*(-inf).
        weight = torch.where(abs_dist == 1.0, torch.tensor(0.5), torch.tensor(0.0))
    else:
        # Geometric distribution:  w(k) = 0.5 * p * (1-p)^(|k|-1)  for k ≠ 0
        # log form for numerical stability:
        #   log w(k) = log(0.5) + log(p) + (|k|-1) * log(1-p)
        log_p = math.log(cart_p)
        log_1mp = math.log(1.0 - cart_p)

        # log_weight[n, i] = log(0.5) + log(p) + (|n-i| - 1) * log(1-p)
        log_weight = math.log(0.5) + log_p + (abs_dist - 1.0) * log_1mp
        weight = log_weight.exp()

        # Zero out diagonal (distance = 0, a position does not contribute to itself)
        weight.fill_diagonal_(0.0)

    return weight  # [L, L], float32


def cart_loss_weights(
    diffusion_mask: torch.Tensor,
    cart_p: float,
    attention_mask: torch.Tensor | None = None,
    seq_lengths: Any | None = None,
) -> torch.Tensor:
    """Per-token CART weights for a batch (Dream context-adaptive reweighting).

    For each masked position ``n``, the weight is the geometric-decayed sum of
    contributions from every *clean* (unmasked, real) position.

    Shared by ``DiffusionTrainer`` and ``MaskedDiffusionEvaluator`` so the two
    cannot drift — in particular so eval honors packed-sample boundaries and
    stays comparable with the training loss.

    Args:
        diffusion_mask: ``[B, L]`` bool, ``True`` at masked positions.
        cart_p:         Geometric sharpness.
        attention_mask: Optional ``[B, L]`` real-token mask.  Padding must not
                        contribute clean context, otherwise identical samples
                        padded to different lengths get different weights.
        seq_lengths:    Optional per-row packed sample lengths (one entry per
                        batch row).  Clean context must not cross packed-sample
                        boundaries; the geometric weight is translation
                        invariant, so restricting the matmul to each sample's
                        diagonal block reproduces the unpacked weights exactly.

    Returns:
        ``[B, L]`` weights, zero everywhere the diffusion loss is not computed.
    """
    device = diffusion_mask.device
    _, L = diffusion_mask.shape
    weight_matrix = context_adaptive_reweight(L, cart_p=cart_p).to(device)

    clean_mask = ~diffusion_mask
    if attention_mask is not None:
        clean_mask = clean_mask & attention_mask.to(device=device, dtype=torch.bool)
    clean_f = clean_mask.float()

    if seq_lengths is not None:
        weight = torch.zeros(
            diffusion_mask.shape, dtype=weight_matrix.dtype, device=device
        )
        for b, lengths in enumerate(seq_lengths):
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.tolist()
            offset = 0
            for slen in lengths:
                end = min(offset + int(slen), L)
                if end <= offset:
                    continue
                weight[b, offset:end] = clean_f[b, offset:end].matmul(
                    weight_matrix[offset:end, offset:end]
                )
                offset = end
    else:
        weight = clean_f.matmul(weight_matrix)

    return weight.masked_fill(~diffusion_mask, 0.0)
