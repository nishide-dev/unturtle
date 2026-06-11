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

Loss normalization modes (``loss_norm_type``):
  - ``"token"``    – divide by total maskable tokens across the batch (default).
                     Matches the MDLM reference (dllm MDLMTrainer) and keeps
                     loss scale stable as batch size changes.
  - ``"sequence"`` – divide by per-sequence maskable count, then mean over B.
                     Equivalent to averaging per-example NLL.
  - ``"batch"``    – divide by batch size B only (simple mean over sequences).

Reference:
  zhziszz/dllm  dllm/core/trainers/mdlm.py  L200–210
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
    loss_norm_type: str = "token",
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
        loss_norm_type:   How to normalise the accumulated loss.
                          ``"token"``    – divide by total maskable tokens (default).
                          ``"sequence"`` – per-sequence maskable count, then mean over B.
                          ``"batch"``    – divide by B only.

    Returns:
        Scalar loss tensor.
    """
    B, L, V = logits.shape
    assert labels.shape == (B, L), f"labels shape mismatch: {labels.shape}"
    assert diffusion_mask.shape == (B, L), (
        f"diffusion_mask shape mismatch: {diffusion_mask.shape}"
    )

    flat_labels = labels.view(-1)  # [B*L]
    flat_mask = diffusion_mask.view(-1)  # [B*L] bool

    if logits.device.type == "cuda":
        # Fused: single torch.where instead of clone + scatter.
        fused_labels = torch.where(
            flat_mask, flat_labels, _get_neg100(flat_labels.device)
        )
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

    # maskable_mask: positions eligible for masking (labels != -100), shape [B, L]
    maskable_mask = labels != -100  # [B, L]

    # Apply per-token weights before normalization (if provided).
    if loss_weights is not None:
        per_token_loss = per_token_loss.view(B, L)
        w = loss_weights
        if w.shape == (B,):
            w = w.unsqueeze(1)  # [B, 1]
        assert w.shape == (B, L) or w.shape == (B, 1), (
            f"loss_weights must be (B,), (B,1) or (B,L), got {w.shape}"
        )
        per_token_loss = per_token_loss * w.to(per_token_loss.dtype)  # [B, L]

    # --- Normalization ---
    if loss_norm_type == "token":
        # Normalize by total maskable tokens in the batch.
        # Matches MDLM reference (dllm/core/trainers/mdlm.py L202) and d1 SFT.
        n_maskable = maskable_mask.sum().clamp_min(1)
        if loss_weights is None:
            return per_token_loss.sum() / n_maskable
        return per_token_loss.sum() / n_maskable

    if loss_norm_type == "sequence":
        # Per-sequence normalisation, then mean over B.
        # Equivalent to averaging per-example NLL.
        per_token_loss = per_token_loss.view(B, L)
        n_per_seq = maskable_mask.sum(dim=-1, keepdim=True).clamp_min(1)  # [B, 1]
        return (per_token_loss / n_per_seq.to(per_token_loss.dtype)).sum() / B

    if loss_norm_type == "batch":
        # Simple mean over sequences.
        per_token_loss = per_token_loss.view(B, L)
        return per_token_loss.sum() / B

    raise ValueError(
        f"Unknown loss_norm_type '{loss_norm_type}'. "
        "Choose from: 'token', 'sequence', 'batch'."
    )
