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

**Alignment is the hazard, and the reference has two different answers.**
A causal teacher's ``logits[t]`` predicts token ``t+1``, while a diffusion
student's ``logits[t]`` predicts token ``t``.  Something must realign them,
and *which* thing depends on the path:

- ``forward_process_causal`` (``rl_sdar.py:1080``) slices ``[:, :-1, :]`` on
  **both** sides.  This runs only when ``block_size == 1`` — the degenerate
  AR case (``rl_sdar.py:1555``).
- ``forward_process`` (``rl_sdar.py:1253``) is the real OPD path, since the
  shipped config uses ``block_size: 4``.  There the **teacher alone** is
  realigned, with ``logits_teacher.roll(dims=1, shifts=1)`` and no slice, so
  every position is kept and the student is left untouched.

Getting this wrong misaligns every position by one and still yields a finite,
plausible loss.  So ``alignment`` is an explicit argument rather than a
hardcoded slice — ``"roll"`` (the diffusion default) or ``"truncate"`` (the
``block_size == 1`` AR case).

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
    return value


class FrozenTeacher:
    """A causal LM used only to produce supervision targets.

    Args:
        model:      Any causal LM exposing ``.logits``.
        vocab_size: The *student's* vocabulary size, checked against the
                    teacher's.  Required rather than inferred: inferring it
                    from the teacher would make the check vacuous.
        alignment:  ``"roll"`` (default, the OPD path: realign the teacher and
                    keep every position) or ``"truncate"`` (the
                    ``block_size == 1`` AR case: both sides drop a position).
    """

    def __init__(self, model: Any, *, vocab_size: int, alignment: str = "roll") -> None:
        teacher_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if teacher_vocab is not None and teacher_vocab != vocab_size:
            raise ValueError(
                f"teacher vocab_size {teacher_vocab} does not match the "
                f"student's {vocab_size}; distillation across different "
                "vocabularies is not defined, and matching sizes with "
                "different tokenizers would supervise silently wrong targets"
            )

        if alignment not in ("roll", "truncate"):
            raise ValueError(
                f"alignment must be 'roll' or 'truncate', got {alignment!r}"
            )
        self.alignment = alignment

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
        """Teacher log-probabilities, realigned to the student's convention.

        Returns ``[B, L, V]`` under ``alignment="roll"`` (the OPD default) and
        ``[B, L-1, V]`` under ``"truncate"``.

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
        logits = outputs.logits

        if self.alignment == "roll":
            # The OPD path.  A causal teacher's `logits[t]` predicts token
            # `t+1`; rolling right by one puts each prediction at the index of
            # the token it describes, matching a diffusion student that
            # predicts token `t` at position `t`.  Every position is kept, and
            # the student is not touched.
            logits = logits.roll(dims=1, shifts=1)
        else:
            # `block_size == 1` only: both sides drop a position instead.
            logits = logits[:, :-1, :]

        # Upcast before the log_softmax as the reference does — a bf16
        # teacher's targets are otherwise quantized enough to matter.
        return F.log_softmax(logits.float(), dim=-1)


__all__ = ["FrozenTeacher", "resolve_top_k_logits"]
