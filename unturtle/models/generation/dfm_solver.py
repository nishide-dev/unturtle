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
Discrete flow-matching solver (#65, FS-DFM App. B.1).

Simulates the jump process.  Choose a budget ``S`` with grid ``t_s`` and steps
``h_s = t_{s+1} - t_s`` (uniform ``h = 1/S``).  At each step::

    p_{1|t}      = softmax(logits / T)
    lambda_s^i   = gbar_{t,h} * (1 - p_{1|t}(X_t^i | X_t))    # exit rate
    J_s^i        ~ Bernoulli(1 - exp(-h * lambda_s^i))        # jump?

and where ``J = 1``, resample from ``p_{1|t}`` **restricted to the off-diagonals**
(the current token excluded).

Under the paper's linear ``kappa(t) = t`` the Cumulative Scalar has a closed
form::

    gbar_{t,h} = (1/h) * ln((1 - t) / (1 - t - h))

which is the whole trick: it replaces the *instantaneous* ``g(t) = 1/(1-t)``
with the flow actually integrated over a finite step.  Using ``g(t)`` directly
under-moves at large ``h``, which is why few-step sampling stalls near
``t = 0``.

**Two things the exclusion is not.**  It applies to the *sampler*: on a jump,
the next token is drawn from the off-diagonals.  The *forward process*
(``DiscreteFlowProcess``) deliberately does not exclude — §B.3 is plain "a
uniform source over tokens", so ``x_0 == x_1`` collisions happen at rate 1/V
and are correct.  Different mechanism, different stage.

Reimplemented from the paper.  ``apple/ml-fs-dfm`` is under the Apple Sample
Code License and was deliberately not read (see #65).

Reference:
    FS-DFM  https://arxiv.org/abs/2509.20624
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

_EPS = 1e-6


def cumulative_scalar(t: torch.Tensor, h: float) -> torch.Tensor:
    """``gbar_{t,h} = (1/h) * ln((1-t) / (1-t-h))``, for linear ``kappa``.

    Integrates ``g`` over ``[t, t+h]`` and normalizes by the width, so it
    approaches the instantaneous ``g(t) = 1/(1-t)`` as ``h -> 0``.
    """
    if h <= 0:
        raise ValueError(f"step size h must be > 0, got {h}")

    remaining = 1.0 - t
    after = remaining - h
    if bool((after <= 0).any()):
        raise ValueError(
            f"step of h={h} runs past the end of the path from t={t.tolist()}; "
            "the closed form takes ln of a non-positive number there"
        )
    return torch.log(remaining / after) / h


def jump_probability(
    prob_current: torch.Tensor, gbar: torch.Tensor, h: float
) -> torch.Tensor:
    """``1 - exp(-h * lambda)`` with ``lambda = gbar * (1 - p(current))``.

    The exit rate is proportional to how much probability mass the model puts
    *elsewhere*: a position the denoiser already agrees with barely moves.
    """
    while gbar.dim() < prob_current.dim():
        gbar = gbar.unsqueeze(-1)
    rate = gbar * (1.0 - prob_current)
    return 1.0 - torch.exp(-h * rate)


def sample_jump_targets(
    probs: torch.Tensor,
    current: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw from ``probs`` with the current token excluded.

    Zeroing the current token *is* the paper's off-diagonal draw.  There is
    deliberately no renormalization: ``torch.multinomial`` takes unnormalized
    weights and normalizes internally, so dividing by the remaining mass would
    change nothing about what is sampled.  (Measured: with
    ``[0.95, 0.025, 0.025]`` and the first token excluded, normalized and
    unnormalized weights draw token 1 at 0.48 and 0.47 over 400 draws.)  An
    earlier version divided here and claimed it corrected a sampling bias; it
    did not, and a mutation that removed the division was undetectable.
    """
    masked = probs.clone()
    masked.scatter_(-1, current.unsqueeze(-1), 0.0)

    # What *does* need handling: a row whose off-diagonal mass is ~0 (a
    # one-hot prediction, or underflow in low precision).  `multinomial` would
    # draw an essentially arbitrary token from the numerical dust; there is
    # genuinely nowhere to jump, so fall back to `probs` and let the position
    # stay where it is.
    exhausted = masked.sum(dim=-1, keepdim=True) <= _EPS
    masked = torch.where(exhausted, probs, masked)

    flat = masked.reshape(-1, masked.shape[-1])
    drawn = torch.multinomial(flat, num_samples=1, generator=generator)
    return drawn.reshape(current.shape)


def solve_discrete_flow(
    denoiser: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    x_0: torch.Tensor,
    *,
    steps: int,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Simulate the jump process from ``x_0`` over a uniform grid.

    Args:
        denoiser:    ``(x_t, t, h) -> logits``.  Step-aware by design: FS-DFM's
                     model is ``theta(x_t, t; h)``, and dropping ``h`` reduces
                     it to an ordinary DFM model — precisely what Phase B
                     improves on.
        x_0:         ``[B, L]`` initial state, from the source distribution.
        steps:       Budget ``S``; the grid is uniform with ``h = 1/S``.
        temperature: ``T`` in ``softmax(logits / T)``.  The paper always uses 1.

    Returns:
        ``[B, L]`` — the state at ``t = 1``.  ``x_0`` is never mutated.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    x_t = x_0.clone()
    h = 1.0 / steps

    for step in range(steps):
        t = torch.full((x_t.shape[0],), step * h, device=x_t.device)

        logits = denoiser(x_t, t, h)
        probs = F.softmax(logits / temperature, dim=-1)

        # The final step would put `t + h` exactly at 1, where the closed form
        # divides by zero.  The rate diverges there, so every position resolves
        # to a draw from `p_{1|t}` rather than a rate-governed jump.
        #
        # That draw is over the FULL distribution -- deliberately not
        # `sample_jump_targets`.  The exclusion belongs to a *jump*, which by
        # definition moves; the terminal draw is not a jump and may legitimately
        # return the token a position already holds.  Excluding here forced
        # every position to move at t=1 and gave the model's own argmax
        # probability exactly zero: measured with p = [0.70, 0.20, 0.07, 0.03]
        # and all positions starting on token 0, the argmax was emitted 0.0% of
        # the time, and the error grew with the step budget rather than
        # vanishing (TV 0.11 -> 0.42 over steps 1 -> 128) because more steps
        # means more positions settled onto the token the final step evicts.
        if step == steps - 1:
            flat = probs.reshape(-1, probs.shape[-1])
            drawn = torch.multinomial(flat, num_samples=1, generator=generator)
            x_t = drawn.reshape(x_t.shape)
            continue

        gbar = cumulative_scalar(t, h)
        prob_current = probs.gather(-1, x_t.unsqueeze(-1)).squeeze(-1)
        jumping = torch.rand(
            x_t.shape, device=x_t.device, generator=generator
        ) < jump_probability(prob_current, gbar, h)
        targets = sample_jump_targets(probs, x_t, generator=generator)
        x_t = torch.where(jumping, targets, x_t)

    return x_t


__all__ = [
    "cumulative_scalar",
    "jump_probability",
    "sample_jump_targets",
    "solve_discrete_flow",
]
