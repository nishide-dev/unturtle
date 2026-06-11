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
