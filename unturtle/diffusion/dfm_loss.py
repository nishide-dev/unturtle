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
Discrete flow-matching objective (#65, FS-DFM eq. 3.8).

The paper derives the loss as a **Bregman divergence** (Lipman et al. 2024,
eq. 7.31), starting from the velocity formulation.  Per token at position
``i``::

    L_i(x_1, x_t, t) = -g(t) * [ p_{1|t}(x_t^i | x_t) - delta_{x_1^i}(x_t^i) ]
                       - [ 1 - delta_{x_1^i}(x_t^i) ] * log p_{1|t}(x_1^i | x_t)

with ``p_{1|t}(x^i | z) = softmax(theta_i(z, t))`` and
``g(t) = kappa'(t) / (1 - kappa(t))``, which under the paper's linear
``kappa(t) = t`` is ``1 / (1 - t)``.  ``Ldfm`` is the mean over positions.

**This is not cross-entropy**, and three differences matter:

1. ``g(t)`` is not decoration.  The paper: it "naturally arises from the
   velocity formulation and ensures proper weighting across different time
   steps".  Dropping it trains a differently-weighted objective that still
   converges, so nothing announces the change.
2. The ``delta`` terms make an already-correct position behave differently:
   where ``x_t == x_1`` the log-likelihood term vanishes entirely
   (``1 - delta = 0``) and only the first term acts.  Cross-entropy would keep
   supervising it.
3. The first term reads the probability of the token the position *currently
   holds*, not of the target — and it is a probability, not a log-probability.

The scheduler is injected rather than hardcoding ``1/(1-t)`` so the objective
and the forward process (``unturtle.processes.DiscreteFlowProcess``) cannot
drift onto different paths.

Reimplemented from the paper.  ``apple/ml-fs-dfm`` is under the Apple Sample
Code License and was deliberately not read (see #65).

Reference:
    FS-DFM  https://arxiv.org/abs/2509.20624
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

_FINITE_DIFF = 1e-4


def _scale(scheduler: Any, t: torch.Tensor) -> torch.Tensor:
    """``g(t) = kappa'(t) / (1 - kappa(t))``.

    Uses the scheduler's own ``g`` when it provides one, and otherwise derives
    it from ``kappa`` by central difference.  Deriving rather than requiring
    ``g`` keeps a scheduler that only defines the path usable, and keeps the
    objective tied to whatever path the process actually samples.
    """
    explicit = getattr(scheduler, "g", None)
    if explicit is not None:
        value = explicit(t)
        return torch.as_tensor(value, device=t.device, dtype=t.dtype).expand_as(t)

    kappa = scheduler.kappa

    # Probe points clamped into [0, 1].  A central difference at t > 1 - h
    # evaluates `kappa` off the path, where it is undefined: a scheduler that
    # clamps, or a cosine that turns over past 1, returns a wrong derivative
    # silently.  Clamping degrades to a one-sided difference at the edges,
    # which is the correct thing to do there.
    upper = (t + _FINITE_DIFF).clamp(max=1.0)
    lower = (t - _FINITE_DIFF).clamp(min=0.0)
    span = (upper - lower).clamp_min(_FINITE_DIFF)

    forward = torch.as_tensor(kappa(upper), device=t.device, dtype=t.dtype)
    backward = torch.as_tensor(kappa(lower), device=t.device, dtype=t.dtype)
    derivative = (forward - backward) / span
    current = torch.as_tensor(kappa(t), device=t.device, dtype=t.dtype)

    # `g` genuinely diverges as kappa -> 1; the clamp bounds it rather than
    # letting a t=1.0 sample produce inf.  Callers should keep t strictly
    # below 1 (the process's `time_epsilon` guards the other end) -- this is a
    # backstop, not a licence to sample the singularity.
    return derivative / (1.0 - current).clamp_min(1e-6)


def discrete_flow_matching_loss(
    logits: torch.Tensor,
    x_1: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    scheduler: Any,
    loss_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Equation (3.8), per token or reduced.

    Args:
        logits:     ``[B, L, V]`` — the denoiser's output.
        x_1:        ``[B, L]`` clean target tokens.
        x_t:        ``[B, L]`` the interpolated state the model saw.
        timesteps:  ``[B]`` or ``[B, L]``.  The per-position form is what a
                    packed batch needs, since each segment owns its own ``t``
                    (#62/#65).
        scheduler:  Provides ``kappa(t)``, and optionally ``g(t)``.
        loss_mask:  ``[B, L]`` bool — positions to keep.  Padding carries no
                    supervision, so including it would only average in noise.
        reduction:  ``"mean"`` (default, ``Ldfm``) or ``"none"``.

    Returns:
        ``[B, L]`` under ``"none"``, a scalar under ``"mean"``.
    """
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")

    batch, length, _ = logits.shape
    if timesteps.shape not in ((batch,), (batch, length)):
        raise ValueError(
            f"timesteps must have shape {(batch,)} or {(batch, length)}, "
            f"got {tuple(timesteps.shape)}"
        )

    # Scheduler math in fp32 regardless of the logits dtype.  `_scale` central-
    # differences at 1e-4, and bf16 carries ~3 decimal digits, so `t + 1e-4`
    # and `t - 1e-4` are the *same* bf16 value and the derivative collapses to
    # exactly 0 -- deleting the entire first term with nothing raising.  This
    # is the default path, not a corner case: `LinearKappa` defines only
    # `kappa`, so it takes the finite-difference branch.
    t = timesteps.to(device=logits.device, dtype=torch.float32)
    g = _scale(scheduler, t).to(logits.dtype)
    t = t.to(logits.dtype)
    if g.dim() == 1:
        g = g[:, None]

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # `p_{1|t}(x_t^i | x_t)` — the probability of the token currently held,
    # *not* of the target.  Reading `x_1` here instead is a transcription
    # error that leaves the loss finite and plausible.
    prob_current = probs.gather(-1, x_t.unsqueeze(-1)).squeeze(-1)
    log_prob_target = log_probs.gather(-1, x_1.unsqueeze(-1)).squeeze(-1)

    delta = (x_t == x_1).to(logits.dtype)

    # Where the position already holds its target, `1 - delta` zeroes the
    # log-likelihood term entirely and only the first term acts.
    # The log term is SUBTRACTED, not added -- deliberately different from the
    # printed eq. (3.8), so do not "correct" it back.  As printed the
    # expression is the pre-minimization Bregman quantity: `+(1-delta)*log p`
    # is a positive log-likelihood, i.e. a reward.  Measured with it added, a
    # model predicting `x_1` perfectly scores -4e-09 while one predicting
    # wrongly scores -20.0, and 200 SGD steps drive the loss to -9.43 while
    # mean p(x_1) *falls* from 0.061 to 0.008 -- it unlearns the target, and
    # is unbounded below.  Negating it makes the two terms agree in direction.
    per_token = -g * (prob_current - delta) - (1.0 - delta) * log_prob_target

    if reduction == "none":
        return per_token

    if loss_mask is None:
        return per_token.mean()

    kept = loss_mask.to(per_token.dtype)
    # `clamp_min(1)` rather than a branch: an all-masked batch is a legitimate
    # runtime state (every position padding) and should contribute zero, not
    # NaN.
    return (per_token * kept).sum() / kept.sum().clamp_min(1.0)


__all__ = ["discrete_flow_matching_loss"]
