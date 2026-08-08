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
FS-DFM few-step training components (#65 Phase B).

The paper's §4 makes the step budget a training-time quantity in three
pieces, implemented here as pure functions (the fine-tuning loop that samples
``(t, h)``, maintains the EMA teacher weights of eq. 4.1 and drives these is
a separate slice):

- **Step-aware path loss** — lives in
  :func:`unturtle.diffusion.dfm_loss.discrete_flow_matching_loss` via its
  ``step_size`` argument (eq. 4.3/4.4: the Cumulative Scalar ``gbar_{t,h}``
  replaces the instantaneous ``g(t)``).
- **RK-4 shortcut teacher** (Algorithm 1, :func:`rk_teacher_logits`): the
  fine-grained ground truth for a large jump, built by actually *advancing
  the token state* through three half-step CTMC jumps and averaging the four
  logits with the classic RK-4 weights.
- **Distillation loss** (§4.3, :func:`few_step_distillation_loss`) and the
  budget blend (eq. 4.5, :func:`blend_losses`): small steps train on the path
  loss, large steps distill onto the teacher, per batch row.

Reimplemented from the paper.  ``apple/ml-fs-dfm`` is under the Apple Sample
Code License and was deliberately not read (see #65).

Reference:
    FS-DFM  https://arxiv.org/abs/2509.20624
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from unturtle.models.generation.dfm_solver import (
    cumulative_scalar,
    jump_probability,
    sample_jump_targets,
)


def few_step_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``L_dist = mean_j KL(p_tea,j || p_theta,j)`` with a stopped teacher.

    §4.3.  The KL direction matters: mass the teacher assigns must be covered
    by the student (mode-covering), which is what pulls a one-big-step
    prediction onto where many small steps would land.  The teacher is
    detached here rather than trusting the caller — a teacher that receives
    gradient drifts toward the student and the consistency target collapses.

    Args:
        student_logits: ``[B, L, V]`` from ``theta(x_t, t; h)``.
        teacher_logits: ``[B, L, V]`` from :func:`rk_teacher_logits` (or any
                        frozen estimate).  Detached internally.
        loss_mask:      ``[B, L]`` bool — positions to keep.

    Returns:
        Scalar mean KL over kept positions.
    """
    teacher = teacher_logits.detach()
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    per_position = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)

    if loss_mask is None:
        return per_position.mean()
    kept = loss_mask.to(per_position.dtype)
    return (per_position * kept).sum() / kept.sum().clamp_min(1.0)


def rk_teacher_logits(
    denoiser: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    x_t: torch.Tensor,
    t: torch.Tensor,
    step_size: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Algorithm 1 — the "Shortcut RK-4" teacher estimate for a jump of ``h``.

    ::

        h' = h/2;  t_mid = t + h';  t_next = t + h
        l1 = theta'(x_t,  t;      h');  x1 = Jump(x_t, l1, t,     h')
        l2 = theta'(x1,   t_mid;  h');  x2 = Jump(x1,  l2, t_mid, h')
        l3 = theta'(x2,   t_mid;  h');  x3 = Jump(x2,  l3, t_mid, h')
        l4 = theta'(x3,   t_next; h')
        return (l1 + 2*l2 + 2*l3 + l4) / 6

    Two properties are load-bearing and deliberately not simplified away:

    - **The state advances.**  Each ``Jump`` is a real CTMC step (exit rate
      from the Cumulative Scalar, off-diagonal resample) — the same mechanics
      as ``solve_discrete_flow``'s loop body, built from the solver's own
      exported pieces so the two cannot drift.  Evaluating the model four
      times at the *same* ``x_t`` would also produce four logits and a
      plausible average, but the teacher would never see the intermediate
      states the fine trajectory visits, and the integration would integrate
      nothing.  Per the paper: "we use Jump only to obtain the relevant state
      at t + h/2 in order to compute the logits there."
    - **Every evaluation is conditioned on** ``h' = h/2``, including the last:
      the teacher describes the *fine-grained* model, which is what the
      ``h``-conditioned student is trained to match in one step.

    EMA weight management (eq. 4.1) belongs to the caller: pass a denoiser
    closed over ``theta'``.  No gradient flows out of this function — the
    caller's stop-grad is enforced again in
    :func:`few_step_distillation_loss`.

    Args:
        denoiser:  ``(x_t, t, h) -> logits`` over the *teacher* weights.
        x_t:       ``[B, L]`` current state.
        t:         ``[B]`` current time; requires ``t + h < 1`` (the last
                   half-jump starts at ``t_mid`` and the Cumulative Scalar is
                   singular at the end of the path).
        step_size: The student's step ``h`` being distilled.

    Returns:
        ``[B, L, V]`` averaged teacher logits.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    t = t.reshape(-1).to(torch.float32)
    # Time-based precondition rather than trusting the per-jump scalar to
    # raise: fp32 rounding can leave `1 - t_mid - h'` a hair above zero when
    # t + h == 1 exactly (measured: t=0.9, h=0.1 slipped through with
    # gbar = 306), and the docstring's contract is t + h < 1.
    if bool((t + step_size >= 1.0).any()):
        raise ValueError(
            f"step of h={step_size} from t={t.tolist()} reaches or passes the "
            "end of the path; the teacher's last half-jump would rate at a "
            "singular Cumulative Scalar"
        )
    half = step_size / 2.0

    def jump(state: torch.Tensor, logits: torch.Tensor, at: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        gbar = cumulative_scalar(at, half)  # raises past the end of the path
        prob_current = probs.gather(-1, state.unsqueeze(-1)).squeeze(-1)
        jumping = torch.rand(
            state.shape, device=state.device, generator=generator
        ) < jump_probability(prob_current, gbar, half)
        targets = sample_jump_targets(probs, state, generator=generator)
        return torch.where(jumping, targets, state)

    with torch.no_grad():
        t_mid = t + half
        t_next = t + step_size

        logits_1 = denoiser(x_t, t, half)
        x_1 = jump(x_t, logits_1, t)
        logits_2 = denoiser(x_1, t_mid, half)
        x_2 = jump(x_1, logits_2, t_mid)
        logits_3 = denoiser(x_2, t_mid, half)
        x_3 = jump(x_2, logits_3, t_mid)
        logits_4 = denoiser(x_3, t_next, half)

    return (logits_1 + 2.0 * logits_2 + 2.0 * logits_3 + logits_4) / 6.0


def blend_losses(
    path_loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    *,
    step_sizes: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """eq. (4.5): ``mean_b( m_b * L_dfm_b + (1 - m_b) * L_dist_b )``.

    ``m_b = 1[h_b < tau]``, **strict**: §5.1 pairs ``tau = 2^-9`` with the
    training grid ``h in {2^-10 .. 2^0}``, so only ``h = 2^-10`` takes the
    path loss and ``h = tau`` itself distills.  Reading it as ``<=`` doubles
    the path-loss share of the grid, silently.

    Args:
        path_loss:          ``[B]`` per-row step-aware DFM loss.
        distillation_loss:  ``[B]`` per-row distillation loss.
        step_sizes:         ``[B]`` the ``h`` each row was trained at.
        tau:                The budget threshold.
    """
    take_path = (step_sizes < tau).to(path_loss.dtype)
    return (take_path * path_loss + (1.0 - take_path) * distillation_loss).mean()


__all__ = [
    "blend_losses",
    "few_step_distillation_loss",
    "rk_teacher_logits",
]
