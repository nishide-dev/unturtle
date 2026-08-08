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
Frozen AR teacher for OPD distillation (#64).

The teacher scores states the diffusion student produced.  It is supervision,
never a parameter to optimize, and the reference enforces that three separate
ways::

    teacher_model.requires_grad_(False)   # no parameter gradients
    teacher_model.eval()                  # no dropout
    with torch.no_grad():                 # no autograd graph at all
        teacher_model(...)

They are not redundant.  Freezing stops the optimizer; ``no_grad`` stops the
graph existing, which is where the memory saving comes from; and ``.eval()``
stops dropout making the supervision target stochastic — without it the
student chases noise it can never match.  Each has a distinct failure mode, so
each is tested separately.

**The shift is load-bearing.**  ``logits[t]`` predicts token ``t+1``, so the
reference slices ``[:, :-1, :]`` for the student *and* the teacher
(``rl_sdar.py:1108,1112``).  A teacher returning ``L`` positions against a
student's ``L-1`` misaligns every position by one and still yields a finite,
plausible loss.  This wrapper owns the shift so the two cannot drift.

**One thing the reference does not do:** check that student and teacher share a
vocabulary.  A mismatch produces either a shape error deep inside the
divergence or, if the sizes coincidentally match, silently wrong supervision.
#64 asks for the same tokenization/vocabulary contract as the converted
student, so it is verified where the teacher is built.

Reference:
    OPDLM  https://arxiv.org/abs/2606.06712
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F


def resolve_top_k_logits(value: Optional[int]) -> Optional[int]:
    """Convert the config sentinel to the divergence API's contract.

    ``training.top_k_logits: 0`` means "full vocabulary" upstream, while
    :func:`~unturtle.post_training.divergence.teacher_student_divergence`
    takes ``None`` and rejects ``0`` — deliberately, since accepting both
    would make them synonyms while ``-1`` raised.

    Converting here means callers do it once rather than at every call site,
    which is the footgun this removes.

    **Not for rollout sampling ``top_k``.**  The reference config carries both
    ``rollout.top_k`` (sampling width) and ``training.top_k_logits`` (sparse
    KL).  They default to the same number and mean opposite things; wiring the
    wrong one trains against a different objective with no error.
    """
    if value is None or value == 0:
        return None
    if value < 0:
        raise ValueError(f"top_k_logits must be >= 0, got {value}")
    return value


class FrozenTeacher:
    """A causal LM used only to produce supervision targets.

    Args:
        model:      Any causal LM exposing ``.logits``.
        vocab_size: The *student's* vocabulary size, checked against the
                    teacher's.  Required rather than inferred: inferring it
                    from the teacher would make the check vacuous.
    """

    def __init__(self, model: Any, *, vocab_size: int) -> None:
        teacher_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if teacher_vocab is not None and teacher_vocab != vocab_size:
            raise ValueError(
                f"teacher vocab_size {teacher_vocab} does not match the "
                f"student's {vocab_size}; distillation across different "
                "vocabularies is not defined, and matching sizes with "
                "different tokenizers would supervise silently wrong targets"
            )

        model.requires_grad_(False)
        model.eval()
        self.model = model
        self.vocab_size = vocab_size

    @torch.no_grad()
    def log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **forward_kwargs: Any,
    ) -> torch.Tensor:
        """Shifted teacher log-probabilities, ``[B, L-1, V]``.

        Args:
            attention_mask: The teacher's own padding mask.  Deliberately a
                            separate argument rather than reusing whatever the
                            student saw: in the reference's BD3LM branch the
                            student gets a 4-D causal mask while the teacher
                            gets the plain 2-D padding mask
                            (``rl_sdar.py:1096-1112``), so sharing one tensor
                            would silently change what the teacher sees.
        """
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, **forward_kwargs
        )
        # `logits[t]` predicts token `t+1`; drop the final position so index
        # `t` aligns with token `t`, matching the student's slice.  Upcast to
        # float32 before the log_softmax as the reference does — a bf16
        # teacher's targets are otherwise quantized enough to matter.
        return F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)


__all__ = ["FrozenTeacher", "resolve_top_k_logits"]
