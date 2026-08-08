"""
Rollout trajectory stitchers (#64, rollout provider part 1).

The reference (`rl_sdar.py:52-75`) stitches a prompt's T denoising rounds
into one training row by independently sampling **one round per block** —
exploiting BD3LM's block-causal factorization — and taking that round's
noised block plus its supervision mask.  Blocks no round ever masked stay
clean and unsupervised.  A synthetic alternative (`:78-110`) skips the
trajectory entirely: per block, a uniform mask ratio, at least one masked
position guaranteed.

Unturtle's versions produce `SupervisionState`s on the unpacked layout the
slice-A contract chose deliberately (separate equal-length `input_ids` /
`clean_input_ids`, no `L0+2*L1` scratch region, no off-by-`L1` indexing) and
record **which round each block came from** as `round_indices` — provenance
the reference discards, and which the contract validates when present.

The hazard here is the same one the whole post_training layer is built
around: a stitcher that pairs block `b` with round `r`'s mask but round
`r'`'s tokens produces a finite, plausible loss forever.  So the tests pin
content-to-provenance agreement, not just shapes.
"""

import pytest
import torch

from unturtle.post_training.trajectory import SupervisionState

PROMPT = 4
RESPONSE = 8
LENGTH = PROMPT + RESPONSE
BLOCK = 4
MASK_ID = 99


def _rounds(n_rounds=3, seed=0):
    """Synthetic per-round states: round r's noised positions hold 100 + r.

    Encoding the round id into the token values makes provenance directly
    readable off the stitched row — block content, mask, and round_indices
    can each be checked against the others.
    """
    generator = torch.Generator().manual_seed(seed)
    clean = torch.randint(0, 50, (LENGTH,), generator=generator)
    states, masks = [], []
    for r in range(n_rounds):
        mask = torch.zeros(LENGTH, dtype=torch.bool)
        mask[PROMPT:] = torch.rand(RESPONSE, generator=generator) < 0.5
        state = clean.clone()
        state[mask] = 100 + r
        states.append(state)
        masks.append(mask)
    return torch.stack(states), torch.stack(masks), clean


class TestOneStatePerBlock:
    def test_each_block_comes_whole_from_its_recorded_round(self):
        """Content must match provenance, position by position.

        For every response block: supervised positions hold the chosen
        round's tokens (encoded 100 + r), unsupervised positions hold the
        clean tokens, and the mask equals that round's mask on the block.
        This is the assertion that catches a mask-from-round-r,
        tokens-from-round-r' mismatch — the finite-and-plausible failure.
        """
        from unturtle.post_training.rollout import combine_rounds_one_state_per_block

        states, masks, clean = _rounds()

        state = combine_rounds_one_state_per_block(
            states,
            masks,
            clean,
            sample_id="s0",
            prompt_length=PROMPT,
            block_size=BLOCK,
            generator=torch.Generator().manual_seed(7),
        )

        assert isinstance(state, SupervisionState)
        for block_index, start in enumerate(range(PROMPT, LENGTH, BLOCK)):
            end = min(start + BLOCK, LENGTH)
            chosen = state.round_indices[block_index]
            assert torch.equal(
                state.supervision_mask[start:end], masks[chosen, start:end]
            ), f"block {block_index}: mask does not match round {chosen}"
            expected = torch.where(
                masks[chosen, start:end],
                torch.full((end - start,), 100 + chosen),
                clean[start:end],
            )
            assert torch.equal(state.input_ids[start:end], expected), (
                f"block {block_index}: tokens do not match round {chosen}"
            )

    def test_blocks_can_disagree_about_their_round(self):
        """The point of the mechanism: rounds are drawn per block.

        A stitcher that draws once and reuses it produces valid-looking rows
        forever; over many seeds the recorded indices must actually differ
        within a row.
        """
        from unturtle.post_training.rollout import combine_rounds_one_state_per_block

        states, masks, clean = _rounds(n_rounds=4)

        saw_disagreement = False
        for seed in range(20):
            state = combine_rounds_one_state_per_block(
                states,
                masks,
                clean,
                sample_id=f"s{seed}",
                prompt_length=PROMPT,
                block_size=BLOCK,
                generator=torch.Generator().manual_seed(seed),
            )
            if len(set(state.round_indices)) > 1:
                saw_disagreement = True
                break

        assert saw_disagreement, (
            "20 seeds never picked different rounds for different blocks; "
            "the draw is per-row, not per-block"
        )

    def test_a_block_no_round_masked_stays_clean_and_unsupervised(self):
        from unturtle.post_training.rollout import combine_rounds_one_state_per_block

        states, masks, clean = _rounds()
        # Erase every round's noise in the second block.
        start, end = PROMPT + BLOCK, LENGTH
        for r in range(states.shape[0]):
            states[r, start:end] = clean[start:end]
            masks[r, start:end] = False

        state = combine_rounds_one_state_per_block(
            states,
            masks,
            clean,
            sample_id="s0",
            prompt_length=PROMPT,
            block_size=BLOCK,
            generator=torch.Generator().manual_seed(0),
        )

        assert torch.equal(state.input_ids[start:end], clean[start:end])
        assert not bool(state.supervision_mask[start:end].any())
        # Provenance is the ONLY observable that separates "no round was
        # eligible" from "an eligible round happened to hold clean tokens
        # here": in both cases the tokens and the mask come out identical
        # (mutation-verified -- an eligibility check that ignored the block
        # survived every content assertion).  -1 is the recorded sentinel.
        assert state.round_indices[1] == -1, (
            f"round_indices {state.round_indices}: a block no round masked "
            "must record -1, not a round that contributed nothing"
        )

    def test_the_prompt_is_invariant_and_never_supervised(self):
        from unturtle.post_training.rollout import combine_rounds_one_state_per_block

        states, masks, clean = _rounds()

        state = combine_rounds_one_state_per_block(
            states,
            masks,
            clean,
            sample_id="s0",
            prompt_length=PROMPT,
            block_size=BLOCK,
            generator=torch.Generator().manual_seed(3),
        )

        assert torch.equal(state.input_ids[:PROMPT], clean[:PROMPT])
        assert not bool(state.supervision_mask[:PROMPT].any())
        assert torch.equal(state.clean_input_ids, clean)

    def test_the_draw_is_reproducible_under_a_seeded_generator(self):
        from unturtle.post_training.rollout import combine_rounds_one_state_per_block

        states, masks, clean = _rounds(n_rounds=4)

        first = combine_rounds_one_state_per_block(
            states,
            masks,
            clean,
            sample_id="a",
            prompt_length=PROMPT,
            block_size=BLOCK,
            generator=torch.Generator().manual_seed(5),
        )
        second = combine_rounds_one_state_per_block(
            states,
            masks,
            clean,
            sample_id="a",
            prompt_length=PROMPT,
            block_size=BLOCK,
            generator=torch.Generator().manual_seed(5),
        )

        assert first.round_indices == second.round_indices
        assert torch.equal(first.input_ids, second.input_ids)

    def test_mismatched_round_stacks_are_rejected(self):
        from unturtle.post_training.rollout import combine_rounds_one_state_per_block

        states, masks, clean = _rounds()

        with pytest.raises(ValueError, match="round"):
            combine_rounds_one_state_per_block(
                states[:2],
                masks,
                clean,
                sample_id="s0",
                prompt_length=PROMPT,
                block_size=BLOCK,
            )


class TestRandomMaskState:
    def test_masked_positions_hold_the_mask_token_and_the_rest_stay_clean(self):
        from unturtle.post_training.rollout import random_mask_state

        generator = torch.Generator().manual_seed(0)
        clean = torch.randint(0, 50, (LENGTH,), generator=generator)

        state = random_mask_state(
            clean,
            sample_id="s0",
            prompt_length=PROMPT,
            block_size=BLOCK,
            mask_token_id=MASK_ID,
            generator=generator,
        )

        masked = state.supervision_mask
        assert bool((state.input_ids[masked] == MASK_ID).all())
        assert torch.equal(state.input_ids[~masked], clean[~masked])
        assert torch.equal(state.clean_input_ids, clean)

    def test_every_block_carries_at_least_one_supervised_position(self):
        """The reference's explicit guarantee — a block that drew t ~ 0 and
        masked nothing contributes no gradient, so one position is forced."""
        from unturtle.post_training.rollout import random_mask_state

        for seed in range(30):
            generator = torch.Generator().manual_seed(seed)
            clean = torch.randint(0, 50, (LENGTH,), generator=generator)
            state = random_mask_state(
                clean,
                sample_id=f"s{seed}",
                prompt_length=PROMPT,
                block_size=BLOCK,
                mask_token_id=MASK_ID,
                generator=generator,
            )
            for start in range(PROMPT, LENGTH, BLOCK):
                end = min(start + BLOCK, LENGTH)
                assert bool(state.supervision_mask[start:end].any()), (
                    f"seed {seed}: block at {start} has no supervision"
                )

    def test_the_mask_ratio_varies_between_blocks(self):
        """Per-block t ~ U(0,1), not one row-wide ratio.

        Row-wide masking is the MDLM default this variant exists to replace;
        with one shared t the per-block densities are binomially tied, so over
        enough draws the two blocks' counts must decorrelate visibly.
        """
        from unturtle.post_training.rollout import random_mask_state

        densities = []
        for seed in range(40):
            generator = torch.Generator().manual_seed(seed)
            clean = torch.randint(0, 50, (LENGTH,), generator=generator)
            state = random_mask_state(
                clean,
                sample_id=f"s{seed}",
                prompt_length=PROMPT,
                block_size=BLOCK,
                mask_token_id=MASK_ID,
                generator=generator,
            )
            counts = [
                int(state.supervision_mask[s : s + BLOCK].sum())
                for s in range(PROMPT, LENGTH, BLOCK)
            ]
            densities.append(counts)

        differing = sum(1 for a, b in densities if a != b)
        assert differing > 5, (
            f"block mask counts agreed in {40 - differing}/40 rows; the ratio "
            "looks row-wide rather than per-block"
        )

        # Between-block disagreement alone cannot see a FIXED rate: two
        # independent Binomial(4, 0.5) draws also differ constantly
        # (mutation-verified survivor).  The uniform mixture is pinned by its
        # tail weight instead: P(full block) = E[t^4] = 1/5 under t ~ U(0,1)
        # against 1/16 at a fixed 0.5 -- over these 80 blocks, ~16 vs ~5
        # (measured 0.196 vs 0.059 over 4000 simulated blocks).
        full_blocks = sum(count == BLOCK for row in densities for count in row)
        assert full_blocks >= 9, (
            f"only {full_blocks}/80 blocks were fully masked; the mask rate "
            "is not drawn from U(0,1) per block"
        )

    def test_the_prompt_is_never_masked(self):
        from unturtle.post_training.rollout import random_mask_state

        generator = torch.Generator().manual_seed(1)
        clean = torch.randint(0, 50, (LENGTH,), generator=generator)

        state = random_mask_state(
            clean,
            sample_id="s0",
            prompt_length=PROMPT,
            block_size=BLOCK,
            mask_token_id=MASK_ID,
            generator=generator,
        )

        assert torch.equal(state.input_ids[:PROMPT], clean[:PROMPT])
        assert not bool(state.supervision_mask[:PROMPT].any())


class TestDeviceThreading:
    """The one thing the port dropped: the reference threads
    `device = clean_input_ids.device` through every allocation.

    Review found both failure shapes on a real GPU: `random_mask_state`
    RAISED mixing a CPU mask into CUDA rows, and the round stitcher was
    worse — it *succeeded*, returning a state with `input_ids` on cuda and
    `supervision_mask` on cpu, which `SupervisionBatch.from_states` stacks
    without complaint and which fails far from the construction site.  The
    contract now rejects split devices at construction, and these pin the
    stitchers producing device-coherent states.
    """

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_rows_produce_cuda_coherent_states(self):
        from unturtle.post_training.rollout import (
            combine_rounds_one_state_per_block,
            random_mask_state,
        )

        states, masks, clean = _rounds()
        stitched = combine_rounds_one_state_per_block(
            states.cuda(),
            masks.cuda(),
            clean.cuda(),
            sample_id="s0",
            prompt_length=PROMPT,
            block_size=BLOCK,
            generator=torch.Generator().manual_seed(0),
        )
        assert stitched.input_ids.device == stitched.supervision_mask.device

        random = random_mask_state(
            clean.cuda(),
            sample_id="s1",
            prompt_length=PROMPT,
            block_size=BLOCK,
            mask_token_id=MASK_ID,
            generator=torch.Generator().manual_seed(0),
        )
        assert random.input_ids.device == random.supervision_mask.device
        assert random.input_ids.device.type == "cuda"

    def test_the_contract_rejects_a_split_device_state(self):
        """The guard itself, pinned where CUDA is available.

        Without it a split-device state constructs, stacks, and fails at
        whatever op first mixes the fields — the silent variant of the
        mispairing this layer exists to prevent.
        """
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA to build a split-device state")

        mask = torch.zeros(LENGTH, dtype=torch.bool)  # cpu
        mask[PROMPT:] = True
        with pytest.raises(ValueError, match="device"):
            SupervisionState(
                sample_id="split",
                input_ids=torch.arange(LENGTH).cuda(),
                supervision_mask=mask,
                prompt_length=PROMPT,
                block_size=BLOCK,
            )


class TestStatesDoNotAliasTheCallerTensor:
    def test_mutating_the_caller_row_does_not_reach_the_state(self):
        """`clean_input_ids.clone()` is load-bearing, not defensive noise.

        The contract documents non-aliasing for stacked tensors; the
        stitcher extends it to the state fields, so a caller reusing its
        buffer for the next prompt cannot rewrite an emitted state.
        (Review found the clone uncovered — a mutant removing it survived.)
        """
        from unturtle.post_training.rollout import random_mask_state

        generator = torch.Generator().manual_seed(0)
        clean = torch.randint(0, 50, (LENGTH,), generator=generator)
        state = random_mask_state(
            clean,
            sample_id="s0",
            prompt_length=PROMPT,
            block_size=BLOCK,
            mask_token_id=MASK_ID,
            generator=generator,
        )
        snapshot = state.clean_input_ids.clone()

        clean.fill_(0)

        assert torch.equal(state.clean_input_ids, snapshot), (
            "the caller's buffer aliases the emitted state"
        )
