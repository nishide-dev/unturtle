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
Masked Diffusion Language Model (dLLM) loss functions.

Implements the masked diffusion cross-entropy loss used by LLaDA / MDLM / d1-style
training.  The core CE computation reuses the existing Triton-optimised
``Fast_CrossEntropyLoss`` kernel; the dLLM-specific additions are:

1. Only masked positions contribute to the loss (unmasked → label = -100).
2. Optional per-batch-element timestep weighting  ``w(t)``  (used by d1 SFT and
   MDLM's scheduler-based weighting).

References:
    LLaDA  – https://arxiv.org/abs/2406.04329
    MDLM   – https://arxiv.org/abs/2406.07524
    d1     – https://arxiv.org/abs/2504.12216
"""

import torch
from unturtle.kernels.fused_masked_diffusion_loss import fused_masked_diffusion_loss


def fast_masked_diffusion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
) -> torch.Tensor:
    """Triton-accelerated cross-entropy loss for masked diffusion language models.

    Only the positions indicated by ``diffusion_mask`` contribute to the loss.
    An optional per-token weight tensor ``loss_weights`` allows timestep-based
    weighting (e.g. ``w(t) = -α'(t) / (1 - α(t))`` from MDLM, or ``1/t`` from d1).

    This function reuses the existing ``Fast_CrossEntropyLoss`` Triton kernel by
    setting ``label = -100`` at unmasked positions, which the kernel already treats
    as "ignore".  Timestep weights are applied at the Python level so that no new
    Triton kernel is required for Phase 1.

    Args:
        logits:           ``(B, L, V)`` – raw model output logits.
        labels:           ``(B, L)``    – clean token ids ``x_0``.
        diffusion_mask:   ``(B, L)`` bool – ``True`` at positions that were masked
                          during the forward diffusion process (loss is computed here).
        loss_weights:     ``(B, L)`` float or ``(B,)`` float (broadcast over L) –
                          per-token weights.  Pass ``None`` for uniform weighting
                          (MDLM / LLaDA style).  Pass ``1/t`` expanded to ``(B, L)``
                          for d1-style timestep weighting.
        logit_softcapping: Gemma-2 style softcap value (0 = disabled).
        logit_scaling:    Cohere style logit scale (0 = disabled).

    Returns:
        Scalar loss.
    """
    B, L, V = logits.shape
    assert labels.shape == (B, L), f"labels shape mismatch: {labels.shape}"
    assert diffusion_mask.shape == (B, L), f"diffusion_mask shape mismatch: {diffusion_mask.shape}"

    # Delegate to fused_masked_diffusion_loss which eliminates the labels.clone()
    # overhead via a single torch.where call (no separate scatter write).
    return fused_masked_diffusion_loss(
        logits=logits,
        labels=labels,
        diffusion_mask=diffusion_mask,
        loss_weights=loss_weights,
        logit_softcapping=logit_softcapping,
        logit_scaling=logit_scaling,
    )


def masked_diffusion_loss_from_timesteps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    timesteps: torch.Tensor,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
) -> torch.Tensor:
    """Convenience wrapper: d1-style ``loss / t`` timestep weighting.

    Computes ``fast_masked_diffusion_loss`` with per-sequence weights ``1 / t``
    where ``t`` is the diffusion timestep used during the forward process.

    Args:
        logits:         ``(B, L, V)``
        labels:         ``(B, L)``
        diffusion_mask: ``(B, L)`` bool
        timesteps:      ``(B,)`` float in ``(eps, 1]`` – the diffusion timestep
                        for each sequence in the batch.

    Returns:
        Scalar loss.
    """
    assert timesteps.shape == (logits.shape[0],), (
        f"timesteps must have shape (B,)={logits.shape[0]}, got {timesteps.shape}"
    )
    loss_weights = 1.0 / timesteps.clamp_min(1e-6)  # [B]
    return fast_masked_diffusion_loss(
        logits=logits,
        labels=labels,
        diffusion_mask=diffusion_mask,
        loss_weights=loss_weights,
        logit_softcapping=logit_softcapping,
        logit_scaling=logit_scaling,
    )
