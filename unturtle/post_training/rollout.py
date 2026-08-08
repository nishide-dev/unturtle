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


def replay_rounds(
    input_ids: torch.Tensor,
    step_map: torch.Tensor,
    *,
    prompt_length: int,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a decode's intermediate states from its commit trace.

    ``one_round_vectorized`` (rl_sdar.py:590-651) iterated to exhaustion, on
    the unpacked layout.  ``step_map[i]`` is the denoising step at which
    response position ``i`` committed its final token.  Per block, each round
    consumes the minimum remaining step: positions committed AT that step are
    the round's supervised set; positions committed at that step or later are
    still masked in the round's state; earlier ones hold their final tokens.
    Consumed steps are retired, so every response position is supervised
    exactly once across the replay.

    No sampler hook is needed -- the trace suffices, which is the reference's
    own design (its generation loop records ``step_map`` and everything else
    is replayed after the fact).

    Args:
        input_ids: ``[L]`` final decoded row (prompt + response).
        step_map:  ``[L - prompt_length]`` commit step per response position.

    Returns:
        ``(states [T, L], masks [T, L])`` where ``T`` is the number of rounds
        (the max number of distinct commit steps in any block).
    """
    length = input_ids.shape[-1]
    response = length - prompt_length
    if step_map.shape[-1] != response:
        raise ValueError(
            f"step_map has {step_map.shape[-1]} entries but the response "
            f"spans {response} positions"
        )
    device = input_ids.device
    remaining = step_map.clone().to(device)
    big = torch.iinfo(remaining.dtype).max

    states: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    while True:
        supervised_tail = torch.zeros(response, dtype=torch.bool, device=device)
        still_masked_tail = torch.zeros(response, dtype=torch.bool, device=device)
        any_selected = False
        for start in range(0, response, block_size):
            end = min(start + block_size, response)
            block = remaining[start:end]
            valid = block.ge(0)
            if not bool(valid.any()):
                continue
            minimum = block.masked_fill(~valid, big).min()
            # Supervised this round: committed exactly at the block minimum.
            # Still masked: committed at the minimum OR LATER -- the state the
            # decode saw just before committing these positions.
            supervised_tail[start:end] = block.eq(minimum) & valid
            still_masked_tail[start:end] = block.ge(minimum) & valid
            any_selected = True
        if not any_selected:
            break

        state = input_ids.clone()
        state[prompt_length:][still_masked_tail] = mask_token_id
        mask = torch.zeros(length, dtype=torch.bool, device=device)
        mask[prompt_length:] = supervised_tail
        states.append(state)
        masks.append(mask)
        # Retire the consumed minima (the reference marks them -1).
        remaining = remaining.masked_fill(supervised_tail, -1)
        # Progress guard: each round retires at least one position, so the
        # replay can never exceed `response` rounds.  Without this, a retire
        # that stopped retiring loops forever selecting the same minima --
        # the mutant HANGS rather than fails, which no assertion can see.
        if len(states) > response:
            raise RuntimeError(
                f"replay exceeded {response} rounds without exhausting the "
                "step map; a round is not retiring the steps it consumes"
            )

    return torch.stack(states), torch.stack(masks)


def commit_steps_from_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    """Derive the commit trace from a recorded state sequence.

    A position's commit step is one past its LAST change -- the index of the
    state in which it first holds its final value for good.  A flip-flop
    (A -> B -> A) therefore commits late: it was not final-valued *stably*
    until its last change, and replaying it as early-committed would show
    the teacher a state the decode never stabilized.  Positions that never
    change commit at 0.

    Args:
        trajectory: ``[T + 1, ...]`` states, index 0 the initial state and
                    index ``T`` the final one (:class:`TrajectoryRecorder`
                    produces exactly this).

    Returns:
        Commit steps with the trajectory's trailing shape.
    """
    if trajectory.shape[0] < 2:
        raise ValueError(
            f"trajectory needs at least 2 states (initial and final), got "
            f"{trajectory.shape[0]}"
        )
    changed = trajectory[1:] != trajectory[:-1]
    step_index = torch.arange(1, trajectory.shape[0], device=trajectory.device).reshape(
        -1, *([1] * (trajectory.dim() - 1))
    )
    return (changed * step_index).amax(dim=0)


class TrajectoryRecorder:
    """Wrap a denoiser so a solver run leaves its full state trajectory behind.

    The solver hands the denoiser each pre-step state; the recorder keeps a
    clone of every one, and :meth:`finish` appends the solver's returned
    final state.  ``recorder -> commit_steps_from_trajectory -> replay_rounds``
    is the mechanical path from a live ``solve_discrete_flow`` call to the
    round states the stitcher consumes -- no hook inside the sampler.  Mind
    the monotone-commitment caveat on :func:`replay_rounds`: for the
    jump-process solver the replayed states are an idealized reconstruction,
    not the recorded trajectory itself.

    Single-use by design: a recorder reused across rollouts would silently
    concatenate two prompts' trajectories into one commit trace.
    """

    def __init__(self, denoiser) -> None:
        self._denoiser = denoiser
        self._states: list[torch.Tensor] = []
        self._finished = False

    def __call__(self, x_t: torch.Tensor, t: torch.Tensor, h) -> torch.Tensor:
        if self._finished:
            raise RuntimeError("this recorder is finished; build a new one")
        self._states.append(x_t.detach().clone())
        return self._denoiser(x_t, t, h)

    def finish(self, final_state: torch.Tensor) -> torch.Tensor:
        """Append the solver's return and yield the ``[T + 1, ...]`` trajectory."""
        if self._finished:
            raise RuntimeError("finish() may only be called once per recorder")
        self._finished = True
        self._states.append(final_state.detach().clone())
        return torch.stack(self._states)


__all__ = [
    "TrajectoryRecorder",
    "combine_rounds_one_state_per_block",
    "commit_steps_from_trajectory",
    "random_mask_state",
    "replay_rounds",
]
