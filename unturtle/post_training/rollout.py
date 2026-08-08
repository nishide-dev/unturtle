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
Rollout trajectory stitchers for OPDLM distillation (#64).

The reference stitches a prompt's T denoising rounds into one training row by
drawing **one round per block** — BD3LM's block-causal factorization makes
the blocks independently combinable — taking that round's noised block and
its supervision mask; blocks no round ever masked stay clean and carry no
supervision (``rl_sdar.py:52-75``).  A synthetic alternative skips the
trajectory: per block, a uniform mask ratio with at least one masked position
guaranteed (``:78-110``).

Two deliberate departures from the reference, both recorded:

- **Layout.**  The reference works on the packed ``L0 + 2*L1`` row whose
  trailing ``L1`` is a scratch region for the noised copy.  These functions
  work on the slice-A contract instead — equal-length ``input_ids`` (noised)
  and ``clean_input_ids`` — which carries the same information with the
  off-by-``L1`` indexing removed.
- **Provenance.**  The reference draws with ``random.choice`` and discards
  which round each block came from.  These record the draw as
  ``round_indices``, which :class:`SupervisionState` validates — the whole
  post_training layer exists because parallel-list pairing drifts silently.

The model-driving loop (capturing per-round states out of a live block
decode) is a separate slice; these are the pure combiners it will feed.
"""

from __future__ import annotations

from typing import Optional

import torch

from .trajectory import SupervisionState


def combine_rounds_one_state_per_block(
    round_states: torch.Tensor,
    round_masks: torch.Tensor,
    clean_input_ids: torch.Tensor,
    *,
    sample_id: str,
    prompt_length: int,
    block_size: int,
    generator: Optional[torch.Generator] = None,
) -> SupervisionState:
    """Stitch T denoising rounds into one supervision state, one round per block.

    Args:
        round_states:    ``[T, L]`` — round ``r``'s full row, with its noised
                         positions holding whatever the round held there.
        round_masks:     ``[T, L]`` bool — round ``r``'s noised positions.
                         Must be prompt-clean; the response region starts at
                         ``prompt_length``.
        clean_input_ids: ``[L]`` clean prompt + response.
        generator:       Seeded draws; the round choice per block is the only
                         randomness.

    Returns:
        A validated :class:`SupervisionState` whose ``round_indices`` records
        the chosen round per response block.  For a block no round masked,
        the entry is ``-1``, the tokens stay clean, and no position is
        supervised — same semantics as the reference's ``continue``.
    """
    if round_states.shape != round_masks.shape:
        raise ValueError(
            f"round_states {tuple(round_states.shape)} and round_masks "
            f"{tuple(round_masks.shape)} disagree; rounds would be paired "
            "with the wrong masks"
        )
    length = clean_input_ids.shape[-1]
    if round_states.shape[-1] != length:
        raise ValueError(
            f"round rows have length {round_states.shape[-1]} but the clean "
            f"row has {length}"
        )

    input_ids = clean_input_ids.clone()
    # Allocated on the input's device -- the reference threads
    # `device = clean_input_ids.device` through every allocation, and the
    # first draft dropped that in translation.  A CPU `supervision` against
    # CUDA rows does not even fail loudly here: the slice-assign below copies
    # device-to-host silently and the state comes out split across devices.
    supervision = torch.zeros(length, dtype=torch.bool, device=clean_input_ids.device)
    chosen: list[int] = []

    for start in range(prompt_length, length, block_size):
        end = min(start + block_size, length)
        # Rounds that noised anything in this block are the eligible pool --
        # the reference's `pm[:, block].any(dim=1)`.
        eligible = round_masks[:, start:end].any(dim=1).nonzero().flatten()
        if eligible.numel() == 0:
            chosen.append(-1)
            continue
        pick = int(
            eligible[int(torch.randint(0, eligible.numel(), (1,), generator=generator))]
        )
        chosen.append(pick)
        # Both the mask AND the tokens come from the same round -- taking
        # them from different rounds is the finite-and-plausible mispairing
        # this module exists to prevent.
        supervision[start:end] = round_masks[pick, start:end]
        input_ids[start:end] = round_states[pick, start:end]

    return SupervisionState(
        sample_id=sample_id,
        input_ids=input_ids,
        supervision_mask=supervision,
        prompt_length=prompt_length,
        block_size=block_size,
        clean_input_ids=clean_input_ids.clone(),
        round_indices=tuple(chosen),
    )


def random_mask_state(
    clean_input_ids: torch.Tensor,
    *,
    sample_id: str,
    prompt_length: int,
    block_size: int,
    mask_token_id: int,
    generator: Optional[torch.Generator] = None,
) -> SupervisionState:
    """The synthetic alternative: per-block uniform masking of the clean row.

    Per response block: draw ``t ~ U(0, 1)``, Bernoulli-mask each position at
    rate ``t``, and force at least one masked position so every block
    contributes gradient signal — the reference's explicit guarantee.  No
    trajectory and no provenance, so ``round_indices`` stays empty (the
    contract treats that as "no provenance to validate").
    """
    length = clean_input_ids.shape[-1]
    device = clean_input_ids.device
    input_ids = clean_input_ids.clone()
    supervision = torch.zeros(length, dtype=torch.bool, device=device)

    for start in range(prompt_length, length, block_size):
        end = min(start + block_size, length)
        block_length = end - start
        # Draws happen on CPU against the caller's (CPU) generator -- the
        # repo's seeded pipelines all hand CPU generators around -- and only
        # the resulting boolean mask moves to the input's device.  The values
        # are identical either way; what must NOT happen is a CPU mask meeting
        # CUDA rows (the first draft raised on line 154 for exactly that).
        rate = float(torch.rand((), generator=generator))
        mask = (torch.rand(block_length, generator=generator) < rate).to(device)
        if not bool(mask.any()):
            mask[int(torch.randint(0, block_length, (1,), generator=generator))] = True
        supervision[start:end] = mask
        input_ids[start:end] = torch.where(
            mask,
            torch.full_like(input_ids[start:end], mask_token_id),
            input_ids[start:end],
        )

    return SupervisionState(
        sample_id=sample_id,
        input_ids=input_ids,
        supervision_mask=supervision,
        prompt_length=prompt_length,
        block_size=block_size,
        clean_input_ids=clean_input_ids.clone(),
    )


__all__ = ["combine_rounds_one_state_per_block", "random_mask_state"]
