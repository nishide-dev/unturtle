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
On-policy supervision state for OPD distillation (#64).

The student rolls out a denoising trajectory; the frozen teacher scores one
state from it.  This module owns **which** state, and the identity that keeps
it paired with its own supervision after buffering.

Why an object rather than parallel tensors.  The reference implementation
(``dev/repos/opdlm/train/rl_sdar.py:825-838``) maintains three parallel lists —
states, masks, rewards — kept in sync by append order alone::

    extended_input_ids_list.append(ext_b)
    pmask_list.append(pm_b)
    reward_list.append(reward[b])
    ...
    if not per_prompt_ext:
        continue        # skips all three, but only by construction

Any later filter applied to a subset, or a `continue` that misses one list,
pairs state *i* with reward *j*.  Nothing downstream detects it: the loss is
finite, training proceeds, and the model learns against the wrong target.
That is the DiffuGRPO buffering lesson #64 cites, and it is present upstream.

Here every state carries a ``sample_id``, batches are looked up by id rather
than position, and duplicates are rejected at construction — so a misalignment
is a ``KeyError`` rather than a silent one.

Block provenance.  ``_combine_rounds_one_state_per_block`` stitches one row per
prompt by drawing an *independent denoising round per block*, exploiting
BD3LM's block-causal factorization.  ``round_indices`` records which round each
block came from, so a state can be reproduced after the fact.

Reference:
    OPDLM  https://arxiv.org/abs/2606.06712
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass(frozen=True)
class SupervisionState:
    """One student state, and everything needed to score and reproduce it.

    Args:
        sample_id:        Stable identifier surviving micro-batching.  Pairing
                          is by this, never by list position.
        input_ids:        ``[L]`` — the exact tensor the teacher will score,
                          where ``L = prompt_length + response_length``.
        supervision_mask: ``[L]`` bool — positions receiving divergence
                          supervision.  Must exclude the prompt.
        prompt_length:    Observed conditioning prefix, never supervised.
        block_size:       Denoising block width.
        clean_input_ids:  Optional ``[L]`` clean targets ``x_1``.  The
                          reference packs these and the noised state into one
                          ``L0 + 2*L1`` tensor and slices ``[:, :L]`` for
                          labels; keeping them as a separate field of matching
                          length says the same thing without a layout
                          convention every consumer must re-derive.
        round_indices:    One entry per response block: which denoising round
                          that block was taken from.

    Note the shape contract deliberately differs from the reference's packed
    layout, and slice B will need the difference spelled out to rebuild the
    model input.  Upstream, ``extended_input_ids`` is::

        [ clean prompt+response (L0 + L1) | noised tail (L1) ]   -> L0 + 2*L1

    so the noised region is **only the trailing L1**, not the whole tensor;
    ``p_mask`` is ``L0 + L1``; ``labels = extended_input_ids[:, :L]``; and the
    model's logits are re-sliced ``cat([logits[:, :L0], logits[:, L0+L1:]])``
    back to ``L0 + L1``.  The duplicated tail is a scratch region for the
    noised copy, not extra sequence.

    The supervision-bearing state is therefore ``L0 + L1``.  Keeping
    ``input_ids`` (noised) and ``clean_input_ids`` (targets) as separate
    equal-length fields carries the same information with the off-by-``L1``
    indexing removed — which is precisely the class of error this contract
    exists to prevent.
    """

    sample_id: str
    input_ids: torch.Tensor
    supervision_mask: torch.Tensor
    prompt_length: int
    block_size: int
    clean_input_ids: torch.Tensor | None = None
    round_indices: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        length = self.input_ids.shape[-1]
        if self.supervision_mask.shape[-1] != length:
            raise ValueError(
                f"supervision_mask has length {self.supervision_mask.shape[-1]} "
                f"but input_ids has {length}"
            )
        if self.supervision_mask.device != self.input_ids.device:
            # A split-device state is the silent variant of mispairing: the
            # mask VALUES are right, `torch.stack` in `from_states` happily
            # stacks each field on its own device, and nothing fails until an
            # op finally mixes them -- far from the construction site.  Reject
            # at the contract boundary instead (found via a rollout stitcher
            # that allocated its mask on CPU against CUDA rows, #109 review).
            raise ValueError(
                f"supervision_mask is on {self.supervision_mask.device} but "
                f"input_ids is on {self.input_ids.device}; a split-device "
                "state stacks without error and fails far from here"
            )
        if not 0 <= self.prompt_length <= length:
            raise ValueError(
                f"prompt_length must satisfy 0 <= p <= {length}, "
                f"got {self.prompt_length}"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if bool(self.supervision_mask[: self.prompt_length].any()):
            raise ValueError(
                "supervision_mask marks positions inside the prompt; the "
                "prompt is observed conditioning, and supervising it trains "
                "the student to reproduce its own input"
            )
        if self.clean_input_ids is not None and (
            self.clean_input_ids.shape[-1] != length
        ):
            raise ValueError(
                f"clean_input_ids has length {self.clean_input_ids.shape[-1]} "
                f"but input_ids has {length}; this contract keeps them aligned "
                "rather than packing them into one L0+2*L1 tensor"
            )
        expected_blocks = math.ceil((length - self.prompt_length) / self.block_size)
        if self.round_indices and len(self.round_indices) != expected_blocks:
            raise ValueError(
                f"round_indices has {len(self.round_indices)} entries but the "
                f"response spans {expected_blocks} blocks of {self.block_size}"
            )


@dataclass(frozen=True)
class SupervisionBatch:
    """A stacked batch that can still answer "which state is sample X?"."""

    sample_ids: tuple[str, ...]
    input_ids: torch.Tensor
    supervision_mask: torch.Tensor
    states: tuple[SupervisionState, ...]

    @classmethod
    def from_states(cls, states: Sequence[SupervisionState]) -> SupervisionBatch:
        """Stack states, rejecting anything that would make pairing ambiguous.

        Tensors are **copied**, not aliased: a caller mutating its own tensor
        afterwards must not change what the teacher scores.
        """
        if not states:
            # Distinct from the ragged-length error below: an empty rollout
            # (every sample filtered) is a plausible runtime state, and
            # "differing length []" would send someone hunting a shape bug
            # that does not exist.
            raise ValueError("cannot build a SupervisionBatch from no states")

        ids = [state.sample_id for state in states]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(
                f"duplicate sample_id(s) {sorted(duplicates)}: pairing by id "
                "would be ambiguous"
            )

        lengths = {int(state.input_ids.shape[-1]) for state in states}
        if len(lengths) != 1:
            raise ValueError(
                f"states have differing length {sorted(lengths)}; stacking "
                "would crash or pad silently"
            )

        # `torch.stack` already allocates a new tensor, so the batch never
        # shares storage with the states a caller still holds — the teacher
        # scores exactly what the student produced, and a later mutation on
        # either side cannot reach the other.  An explicit `.clone()` here
        # would be redundant (mutation-verified: removing it changes nothing).
        #
        # PyTorch has no read-only tensor flag, so non-aliasing is the
        # enforceable half of immutability -- and it covers the *stacked*
        # tensors only.  `select()` deliberately returns the state object
        # itself, so `batch.select(id).input_ids.mul_(0)` does reach the
        # caller's tensor.  That is the price of keeping `states` the single
        # ordering source of truth; callers holding a state should treat it as
        # borrowed.
        input_ids = torch.stack([s.input_ids for s in states])
        supervision = torch.stack([s.supervision_mask for s in states])

        return cls(
            sample_ids=tuple(ids),
            input_ids=input_ids,
            supervision_mask=supervision,
            states=tuple(states),
        )

    def select(self, sample_id: str) -> SupervisionState:
        """Look up a state by identity, not position."""
        for state in self.states:
            if state.sample_id == sample_id:
                return state
        raise KeyError(
            f"{sample_id!r} is not in this batch (have: {list(self.sample_ids)})"
        )

    def split(self, size: int) -> tuple[SupervisionBatch, ...]:
        """Chunk for gradient accumulation, keeping ids attached.

        The remainder is returned as a short final chunk rather than dropped —
        silently discarding it would quietly reduce the effective batch.
        """
        if size <= 0:
            raise ValueError(f"split size must be > 0, got {size}")
        return tuple(
            SupervisionBatch.from_states(self.states[i : i + size])
            for i in range(0, len(self.states), size)
        )


__all__ = ["SupervisionState", "SupervisionBatch"]
