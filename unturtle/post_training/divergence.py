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
Teacher-student divergences for on-policy AR->diffusion distillation (#64).

The supervision objective from OPDLM: a frozen AR teacher scores states the
diffusion student produced, and the student is trained to match it.  Forward
KL, blended reverse KL, and JSD are supported, each in a dense
(full-vocabulary) and a sparse (teacher top-k) form.

Semantics follow the official implementation
(``dev/repos/opdlm/train/rl_sdar.py``, MIT), not the paper summary.  Two
properties there are load-bearing:

**Non-finite teacher logprobs contribute zero.**  ``p * log p -> 0`` as
``p -> 0``, but evaluated directly it is ``0 * -inf = NaN``, which poisons the
whole batch.  Top-k makes this reachable rather than theoretical: when the
teacher has fewer than ``k`` tokens with non-zero probability, some gathered
indices carry ``-inf``.

**Top-k is an unnormalized partial sum.**  It restricts the sum to the
teacher's top-k tokens without renormalizing the truncated distribution.  That
is what makes ``k = vocab_size`` reproduce the dense value exactly.  It also
means the truncated value is *not* bounded above by the dense one: the summand
``p_t * (log p_t - log p_s)`` is negative wherever the student assigns more
mass than the teacher, so dropping tail terms can increase the total.  Do not
"fix" this with a clamp — it would no longer be the reference objective.

Reference:
    OPDLM  https://arxiv.org/abs/2606.06712
           https://github.com/divelab/OPDLM  (train/rl_sdar.py)
"""

from __future__ import annotations

import torch

_DIVERGENCES = ("kl", "jsd")


def _finite_weighted(log_p: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """``p * delta`` with non-finite ``log_p`` contributing exactly zero.

    ``torch.where`` alone is not enough: it evaluates both branches, so a
    ``-inf`` entry still produces a NaN that the gradient carries even when
    the forward value is discarded.  The logprob is sanitized first.
    """
    finite = torch.isfinite(log_p)
    safe = torch.where(finite, log_p, torch.zeros_like(log_p))
    return torch.where(finite, safe.exp() * delta, torch.zeros_like(delta))


def teacher_student_divergence(
    teacher_logprobs: torch.Tensor,
    student_logprobs: torch.Tensor,
    *,
    divergence: str = "kl",
    top_k: int | None = None,
    reverse_kl_weight: float = 0.0,
    jsd_alpha: float = 0.5,
) -> torch.Tensor:
    """Per-position divergence between a teacher and student distribution.

    Args:
        teacher_logprobs:  ``[..., V]`` log-probabilities from the frozen
                           teacher.  Pass these already detached; this
                           function does not detach for you, so a caller that
                           wants teacher gradients blocked must say so.
        student_logprobs:  ``[..., V]`` log-probabilities from the student.
        divergence:        ``"kl"`` (default) or ``"jsd"``.
        top_k:             Restrict supervision to the teacher's top-k tokens.
                           ``None`` (default) uses the full vocabulary.
                           Values ``>= V`` are clamped and then agree exactly
                           with the dense path.
        reverse_kl_weight: Blend weight ``w`` for the reverse direction:
                           ``(1-w) * KL(teacher||student) + w * KL(student||teacher)``.
                           Ignored for JSD.
        jsd_alpha:         Mixture weight for JSD:
                           ``M = alpha * teacher + (1 - alpha) * student``.
                           ``0.5`` gives the symmetric Jensen-Shannon
                           divergence.

    Returns:
        ``[...]`` — the input shape with the vocabulary axis reduced.
    """
    if divergence not in _DIVERGENCES:
        raise ValueError(
            f"Unknown divergence {divergence!r}. Choose from: {_DIVERGENCES}."
        )

    if top_k is not None:
        vocab = teacher_logprobs.shape[-1]
        k = min(int(top_k), vocab)
        if k <= 0:
            raise ValueError(f"top_k must be >= 1 when given, got {top_k}")
        # Gather before any elementwise work, so no dense [..., V] intermediate
        # is ever built — that is the point of this path.
        teacher_logprobs, indices = teacher_logprobs.topk(k=k, dim=-1)
        student_logprobs = student_logprobs.gather(-1, indices)

    if divergence == "jsd":
        # log M, where M mixes the two distributions.  The teacher's
        # contribution is dropped where its logprob is non-finite, matching
        # the reference; the student's is always finite by construction.
        teacher_probs = _finite_weighted(
            teacher_logprobs, torch.ones_like(teacher_logprobs)
        )
        student_probs = student_logprobs.exp()
        mixture = jsd_alpha * teacher_probs + (1.0 - jsd_alpha) * student_probs
        log_mixture = mixture.log()

        forward = _finite_weighted(teacher_logprobs, teacher_logprobs - log_mixture)
        reverse = student_probs * (student_logprobs - log_mixture)
        return jsd_alpha * forward.sum(dim=-1) + (1.0 - jsd_alpha) * reverse.sum(dim=-1)

    forward_kl = _finite_weighted(
        teacher_logprobs, teacher_logprobs - student_logprobs
    ).sum(dim=-1)
    if reverse_kl_weight <= 0.0:
        return forward_kl

    # The reverse direction weights by the *student*, so a non-finite teacher
    # logprob appears in the delta rather than the weight and cannot be
    # dropped the same way.  Student logprobs come from a log_softmax and are
    # finite; a -inf teacher entry here is a genuine infinity in the objective
    # (the student puts mass where the teacher assigns none), so it is not
    # silently zeroed.
    reverse_kl = (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(
        dim=-1
    )
    return (1.0 - reverse_kl_weight) * forward_kl + reverse_kl_weight * reverse_kl


__all__ = ["teacher_student_divergence"]
