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
Sparse masked LM-head loss (#61).

The dense path computes ``[B, L, V]`` logits and then discards every unmasked
position.  With a ~128K vocabulary that wasted GEMM, and the activation memory
for its output, is one of the larger Unturtle-specific training costs.  This
path gathers first::

    hidden [B, L, H] -> gather [M, H] -> project [M, V] -> CE over M targets

so the output projection runs on the masked positions only.

**Peak memory only improves below roughly a 40% mask ratio.**  Measured on an
RTX 6000 Ada, bf16, B=4, L=1024, H=1024, forward + loss + backward:

===========  ==========  ===========  ==========
vocab        mask 15%    mask 50%     mask 75%
===========  ==========  ===========  ==========
32000        -28%        +8%          +35%
128256       -41%        +10%         +49%
===========  ==========  ===========  ==========

(negative = sparse uses less).  The dense path is harder to beat than the
``[B, L, V]`` shape suggests: ``Fast_CrossEntropyLoss`` upcasts per tile in
registers and never materializes an fp32 logits tensor, so dense holds one
bf16 ``[B, L, V]`` while sparse holds a bf16 ``[M, V]`` plus its autograd
graph.  Past ``M/(B·L) ≈ 0.4`` the gather stops paying for itself.

That matters because MDLM-style training samples ``t ~ U(0, 1)``, giving ~50%
average masking — the regime where this path is *not* a memory win.  It is a
win for low-mask-ratio schedules and, separately, for step time (compute
scales with ``M``, not ``B·L``).  Callers should pick a path on measured mask
ratio rather than assuming.

Numerically identical to :func:`~unturtle.kernels.masked_diffusion_loss.fast_masked_diffusion_loss`
— same loss, same gradients — because cross-entropy at unmasked positions
contributes exactly zero under ``ignore_index=-100``.  Gathering removes terms
that were already zero; it does not approximate them away.

The one thing that does **not** survive the gather is normalization.
``n_maskable`` counts ``labels != -100`` over the full ``[B, L]``, which is a
different (larger) number than the ``M`` positions actually masked, so it must
be taken *before* gathering.  Getting that wrong rescales the loss without
changing its shape — the kind of drift that trains a subtly different objective
while every curve still looks healthy.

Model-specific access comes from the BackboneIntegration registry (#68), so this
module contains no ``model_type`` branches and no model hierarchy knowledge.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from unturtle.models.integrations import resolve_sparse_output


def supports_sparse_masked_loss(model: Any) -> bool:
    """Whether ``model`` can take the sparse path.

    Callers should fall back to the dense loss when this is ``False`` rather
    than treating it as an error.
    """
    return resolve_sparse_output(model) is not None


def sparse_masked_diffusion_loss(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    loss_norm_type: str = "token",
    **forward_kwargs: Any,
) -> torch.Tensor:
    """Masked-diffusion CE loss without materializing ``[B, L, V]`` logits.

    Args:
        model:          A model whose integration declares
                        ``sparse_output_projection`` (Tiny-A2D today).
        input_ids:      ``[B, L]`` noised token ids.
        labels:         ``[B, L]`` clean targets; ``-100`` where not supervised.
        diffusion_mask: ``[B, L]`` bool, ``True`` where the loss is computed.
        loss_weights:   ``[B]`` or ``[B, L]`` per-token weights, or ``None``.
        loss_norm_type: ``"token"`` (default, by ``n_maskable``), ``"sequence"``
                        or ``"batch"`` — matching the dense loss.
        forward_kwargs: Passed to the backbone (``attention_mask``,
                        ``position_ids``, …).  Only pass what the backbone
                        forward consumes: unknown keys are silently absorbed by
                        ``TransformersKwargs`` and would have no effect.

    Returns:
        Scalar loss, equal to the dense path's.

    Raises:
        ValueError: if the model cannot take the sparse path.  Raising rather
            than falling back keeps a silent perf regression from hiding;
            call :func:`supports_sparse_masked_loss` to choose a path.
    """
    access = resolve_sparse_output(model)
    if access is None:
        raise ValueError(
            f"{type(model).__name__} does not support the sparse masked-diffusion "
            "loss (no sparse_output_projection capability). Use "
            "fast_masked_diffusion_loss, or check supports_sparse_masked_loss() "
            "first."
        )

    if logit_softcapping != 0 or logit_scaling != 0:
        # Accepted only to reject: `fast_masked_diffusion_loss` applies these to
        # the logits, and a caller switching paths must not lose them silently.
        # Measured on a toy vocab, ignoring `logit_scaling=0.0625` (Cohere) puts
        # the loss 142% off.  Neither Tiny-A2D backbone uses them today.
        raise ValueError(
            "sparse_masked_diffusion_loss does not implement logit_softcapping "
            "or logit_scaling; use fast_masked_diffusion_loss for models that "
            "need them (Gemma-2 softcap, Cohere scaling)."
        )

    B, L = labels.shape
    if diffusion_mask.shape != (B, L):
        raise ValueError(
            f"diffusion_mask shape {tuple(diffusion_mask.shape)} does not match "
            f"labels {(B, L)}"
        )

    # Counted before the gather: `n_maskable` is over every supervised position,
    # not over the masked subset, and the gather destroys that structure.
    maskable_mask = labels != -100

    hidden = access.hidden_states(model, input_ids=input_ids, **forward_kwargs)

    # Intersect with the supervised positions.  This is a *size* optimization,
    # not a correctness one: `F.cross_entropy` defaults to `ignore_index=-100`,
    # so a masked-but-unsupervised position would contribute zero loss and zero
    # gradient either way.  Excluding it keeps those rows out of the projection,
    # which is the whole point of this path — a completion-only or packed batch
    # can mark `-100` positions as masked.
    active = diffusion_mask & maskable_mask
    if not bool(active.any()):
        # No terms at all.  Project a single row anyway so `lm_head.weight`
        # stays in the backward graph: with untied weights the dense path
        # yields a zero grad for it, and a parameter that never participates
        # trips DDP's find_unused_parameters / desyncs FSDP buckets.  The
        # multiply by zero keeps the value and every gradient at zero.
        probe = access.project(model, hidden.reshape(-1, hidden.shape[-1])[:1])
        return probe.sum() * 0.0

    selected_hidden = hidden[active]  # [M, H]
    selected_labels = labels[active]  # [M]

    logits = access.project(model, selected_hidden)  # [M, V]
    # Upcast the [M] losses, never the [M, V] logits.  An fp32 copy of the
    # projection is 2 bytes/element on top of the bf16 original, both retained
    # by autograd — enough to make this path use *more* peak memory than the
    # dense one past roughly a two-thirds mask ratio, which is the regime
    # MDLM-style training actually runs in.  The dense path upcasts the
    # per-token loss for the same reason (fused_masked_diffusion_loss L137).
    per_token = F.cross_entropy(
        logits,
        selected_labels,
        reduction="none",
    ).float()  # [M]

    if loss_weights is not None:
        weights = loss_weights
        if weights.shape == (B,):
            weights = weights.unsqueeze(1).expand(B, L)
        elif weights.shape == (B, 1):
            weights = weights.expand(B, L)
        elif weights.shape != (B, L):
            raise ValueError(
                f"loss_weights must be (B,), (B,1) or (B,L), got {tuple(weights.shape)}"
            )
        per_token = per_token * weights[active].to(per_token.dtype)

    if loss_norm_type == "token":
        n_maskable = maskable_mask.sum().clamp_min(1)
        return per_token.sum() / n_maskable

    if loss_norm_type == "sequence":
        # Per-sequence sums, each divided by that sequence's maskable count,
        # then averaged over B — as the dense path does over the [B, L] grid.
        row_index = torch.arange(B, device=labels.device).unsqueeze(1).expand(B, L)
        rows = row_index[active]
        per_seq = torch.zeros(B, dtype=per_token.dtype, device=per_token.device)
        per_seq = per_seq.index_add(0, rows, per_token)
        n_per_seq = maskable_mask.sum(dim=-1).clamp_min(1).to(per_token.dtype)
        return (per_seq / n_per_seq).sum() / B

    if loss_norm_type == "batch":
        return per_token.sum() / B

    raise ValueError(
        f"Unknown loss_norm_type '{loss_norm_type}'. "
        "Choose from: 'token', 'sequence', 'batch'."
    )
