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
Fused masked diffusion loss — eliminates the ``labels.clone()`` overhead.

``fast_masked_diffusion_loss`` (Phase 1) builds a masked label tensor via::

    masked_labels = labels.clone()        # B×L alloc + copy
    masked_labels[~diffusion_mask] = -100 # B×L scatter write

``fused_masked_diffusion_loss`` replaces those two steps with a single
``torch.where`` call that executes as one GPU pass::

    fused_labels = torch.where(diffusion_mask_flat, labels_flat, NEG100)

This saves one ``(B, L)`` allocation and one indexed-write kernel launch.
The improvement is most significant for large vocab (128K+) models, where
the label tensor is proportionally expensive to clone.

Gradient computation is unchanged — ``Fast_CrossEntropyLoss`` handles it
internally based on the fused label tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss

# Singleton -100 tensors reused across calls to avoid repeated scalar allocs.
# Created lazily per device.
_NEG100_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_neg100(device: torch.device) -> torch.Tensor:
    """Return a scalar int64 tensor with value -100 on *device* (cached)."""
    if device not in _NEG100_CACHE:
        _NEG100_CACHE[device] = torch.tensor(-100, dtype=torch.long, device=device)
    return _NEG100_CACHE[device]


def fused_masked_diffusion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    loss_weights=None,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
) -> torch.Tensor:
    """Triton CE loss with fused diffusion-mask application (no label clone).

    Drop-in replacement for ``fast_masked_diffusion_loss``.
    On GPU, replaces ``labels.clone() + scatter`` with a single
    ``torch.where`` call.  On CPU, uses the same ``F.cross_entropy`` path.

    Args:
        logits:           ``(B, L, V)`` — raw model output logits.
        labels:           ``(B, L)``    — clean token ids ``x_0``.
        diffusion_mask:   ``(B, L)`` bool — ``True`` at masked positions
                          (loss is computed here).
        loss_weights:     ``(B,)`` or ``(B, L)`` float — per-token weights.
                          Pass ``None`` for uniform weighting.
        logit_softcapping: Gemma-2 style softcap (0 = disabled).
        logit_scaling:    Cohere style logit scale (0 = disabled).

    Returns:
        Scalar loss tensor.
    """
    B, L, V = logits.shape
    assert labels.shape == (B, L), f"labels shape mismatch: {labels.shape}"
    assert diffusion_mask.shape == (B, L), (
        f"diffusion_mask shape mismatch: {diffusion_mask.shape}"
    )

    flat_labels = labels.view(-1)       # [B*L]
    flat_mask = diffusion_mask.view(-1)  # [B*L] bool

    if logits.device.type == "cuda":
        # Fused: single torch.where instead of clone + scatter.
        fused_labels = torch.where(flat_mask, flat_labels, _get_neg100(flat_labels.device))
        per_token_loss = Fast_CrossEntropyLoss.apply(
            logits.view(B * L, V),
            fused_labels,
            logit_softcapping,
            logit_scaling,
        )  # [B*L], float32
    else:
        # CPU fallback — identical semantics to fast_masked_diffusion_loss.
        fused_labels = torch.where(
            flat_mask,
            flat_labels,
            torch.tensor(-100, dtype=torch.long),
        )
        per_token_loss = F.cross_entropy(
            logits.view(B * L, V),
            fused_labels,
            ignore_index=-100,
            reduction="none",
        ).float()  # [B*L]

    # Normalize by n_maskable = number of tokens eligible for masking (labels != -100).
    # This matches the MDLM reference (dev/repos/dllm/dllm/core/trainers/mdlm.py L202:
    #   token_nll /= maskable_mask.sum().clamp_min(1)  # maskable_mask = labels != -100)
    # and d1 SFT (sft_trainer.py L25: loss / (numel - num_prompt_tokens)).
    #
    # Using n_maskable (rather than n_masked = diffusion_mask.sum()) keeps the loss scale
    # consistent with reference implementations so that learning rates transfer directly.
    # At mask_rate=0.15, n_masked ≈ 0.15 * n_maskable, so the previous n_masked denominator
    # produced ~6.7x larger loss values than MDLM for the same data.
    n_maskable = (flat_labels != -100).sum().clamp_min(1)

    if loss_weights is None:
        return per_token_loss.sum() / n_maskable

    per_token_loss = per_token_loss.view(B, L)

    if loss_weights.shape == (B,):
        loss_weights = loss_weights.unsqueeze(1)  # [B, 1]

    assert loss_weights.shape == (B, L) or loss_weights.shape == (B, 1), (
        f"loss_weights must be (B,), (B,1) or (B,L), got {loss_weights.shape}"
    )

    weighted = per_token_loss * loss_weights.to(per_token_loss.dtype)
    return weighted.sum() / n_maskable
