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
    loss_norm_type: str = "token",
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
        loss_norm_type:   How to normalise the accumulated loss.
                          ``"token"``    – divide by total maskable tokens (default).
                          ``"sequence"`` – per-sequence maskable count, then mean over B.
                          ``"batch"``    – divide by B only.

    Returns:
        Scalar loss.
    """
    B, L, V = logits.shape
    assert labels.shape == (B, L), f"labels shape mismatch: {labels.shape}"
    assert diffusion_mask.shape == (B, L), (
        f"diffusion_mask shape mismatch: {diffusion_mask.shape}"
    )

    # Delegate to fused_masked_diffusion_loss which eliminates the labels.clone()
    # overhead via a single torch.where call (no separate scatter write).
    return fused_masked_diffusion_loss(
        logits=logits,
        labels=labels,
        diffusion_mask=diffusion_mask,
        loss_weights=loss_weights,
        logit_softcapping=logit_softcapping,
        logit_scaling=logit_scaling,
        loss_norm_type=loss_norm_type,
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

    Exported as part of the public API for callers writing their own training
    loop.  ``DiffusionTrainer`` does not route through here — it builds the
    weights itself in ``_build_loss_weights`` and calls
    ``fast_masked_diffusion_loss`` directly — so changes to this wrapper do not
    affect the trainer, and vice versa.

    Args:
        logits:         ``(B, L, V)``
        labels:         ``(B, L)``
        diffusion_mask: ``(B, L)`` bool
        timesteps:      ``(B,)`` float in ``(eps, 1]`` – one timestep per
                        sequence, broadcast over ``L``.  Or ``(B, L)``: one
                        timestep per *position*, for packed rows whose segments
                        each own their own ``t`` (#62).

    Returns:
        Scalar loss.

    Note:
        When ``B == L`` the shape check cannot distinguish a ``(B, L)`` tensor
        from its transpose, so a transposed ``timesteps`` is accepted and
        silently yields a different loss.  Orientation is the caller's
        responsibility; only the rank/extent are validated here.
    """
    # `[B, L]` is accepted since #62 PR3: a packed row holds several original
    # samples, each owning its own timestep, so one value per row cannot
    # represent them.
    B, L = logits.shape[0], logits.shape[1]
    if timesteps.shape not in ((B,), (B, L)):
        raise ValueError(
            f"timesteps must have shape (B,)={(B,)} or (B, L)={(B, L)}, "
            f"got {tuple(timesteps.shape)}"
        )
    loss_weights = 1.0 / timesteps.clamp_min(1e-6)  # [B] or [B, L]
    return fast_masked_diffusion_loss(
        logits=logits,
        labels=labels,
        diffusion_mask=diffusion_mask,
        loss_weights=loss_weights,
        logit_softcapping=logit_softcapping,
        logit_scaling=logit_scaling,
    )
