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


class TestReplayRounds:
    """`one_round_vectorized` (rl_sdar.py:590-651) iterated to exhaustion, on
    the unpacked layout.

    The reference reconstructs a decode's intermediate states from
    ``(final tokens, step_map)`` — per block, each round consumes the minimum
    remaining commit step: positions committed AT that step are this round's
    supervised set, positions committed at that step OR LATER are still
    masked in this round's state, earlier ones hold their final tokens.  No
    sampler hook is needed; the trace is the per-position commit step.
    """

    @staticmethod
    def _fixture():
        # Response layout (PROMPT=4, RESPONSE=8, BLOCK=4):
        #   block 0 commit steps: [0, 1, 0, 2]
        #   block 1 commit steps: [1, 1, 3, 3]
        clean = torch.arange(10, 10 + LENGTH)
        step_map = torch.tensor([0, 1, 0, 2, 1, 1, 3, 3])
        return clean, step_map

    def test_rounds_replay_the_decode_block_by_block(self):
        from unturtle.post_training.rollout import replay_rounds

        clean, step_map = self._fixture()
        # Snapshot BEFORE the call.  Review measured that a mutant removing
        # the production `.clone()` corrupts the caller's row in place —
        # and an expectation built from the already-corrupted `clean` agrees
        # with the corrupted output (the co-transcribed failure mode).  All
        # expectations below derive from this snapshot.
        snapshot = clean.clone()

        states, masks = replay_rounds(
            clean,
            step_map,
            prompt_length=PROMPT,
            block_size=BLOCK,
            mask_token_id=MASK_ID,
        )

        assert torch.equal(clean, snapshot), (
            "replay_rounds mutated the caller's row in place"
        )

        # Block 0 has 3 distinct steps {0,1,2}, block 1 has 2 {1,3}; rounds
        # continue while ANY block has steps left -> 3 rounds.
        assert states.shape == (3, LENGTH)

        # Round 0: block 0 consumes step 0 (positions 4, 6 supervised; all of
        # block 0 still masked since every step >= 0); block 1 consumes its
        # min step 1 (positions 8, 9 supervised; whole block masked).
        assert masks[0].tolist() == [False] * 4 + [
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
        ]
        assert (states[0, PROMPT:] == MASK_ID).tolist() == [True] * 8

        # Round 1: block 0 consumes step 1 (position 5); block 1 consumes
        # step 3 (positions 10, 11).  Earlier-committed positions now hold
        # final tokens; later-or-equal ones are masked.
        assert masks[1].tolist() == [False] * 4 + [
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
        ]
        expected_state_1 = snapshot.clone()
        expected_state_1[torch.tensor([5, 7, 10, 11])] = MASK_ID
        assert torch.equal(states[1], expected_state_1)

        # Round 2: only block 0 has step 2 left (position 7); block 1 is
        # exhausted and contributes nothing.
        assert masks[2].tolist() == [False] * 4 + [
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
        ]
        expected_state_2 = snapshot.clone()
        expected_state_2[7] = MASK_ID
        assert torch.equal(states[2], expected_state_2)

    def test_the_prompt_is_never_masked_or_supervised(self):
        from unturtle.post_training.rollout import replay_rounds

        clean, step_map = self._fixture()

        states, masks = replay_rounds(
            clean,
            step_map,
            prompt_length=PROMPT,
            block_size=BLOCK,
            mask_token_id=MASK_ID,
        )

        assert torch.equal(states[:, :PROMPT], clean[:PROMPT].expand(3, -1))
        assert not bool(masks[:, :PROMPT].any())

    def test_every_response_position_is_supervised_exactly_once(self):
        """Each commit step is consumed exactly once across the replay —
        the reference marks consumed minima as -1.  A position supervised
        twice would double-count its gradient; one supervised never would
        silently drop it."""
        from unturtle.post_training.rollout import replay_rounds

        clean, step_map = self._fixture()

        _, masks = replay_rounds(
            clean,
            step_map,
            prompt_length=PROMPT,
            block_size=BLOCK,
            mask_token_id=MASK_ID,
        )

        counts = masks[:, PROMPT:].sum(dim=0)
        assert bool((counts == 1).all()), counts.tolist()

    def test_replayed_rounds_feed_the_stitcher(self):
        """The integration the module exists for: replay -> combine ->
        contract-valid SupervisionState."""
        from unturtle.post_training.rollout import (
            combine_rounds_one_state_per_block,
            replay_rounds,
        )

        clean, step_map = self._fixture()
        states, masks = replay_rounds(
            clean,
            step_map,
            prompt_length=PROMPT,
            block_size=BLOCK,
            mask_token_id=MASK_ID,
        )

        state = combine_rounds_one_state_per_block(
            states,
            masks,
            clean,
            sample_id="replayed",
            prompt_length=PROMPT,
            block_size=BLOCK,
            generator=torch.Generator().manual_seed(0),
        )

        assert isinstance(state, SupervisionState)
        assert all(index >= 0 for index in state.round_indices)

    def test_step_map_length_must_match_the_response(self):
        from unturtle.post_training.rollout import replay_rounds

        clean, _ = self._fixture()

        with pytest.raises(ValueError, match="step_map"):
            replay_rounds(
                clean,
                torch.tensor([0, 1]),
                prompt_length=PROMPT,
                block_size=BLOCK,
                mask_token_id=MASK_ID,
            )


class TestCommitStepsFromTrajectory:
    def test_the_commit_step_is_the_last_change(self):
        """A position's commit step is the index of the state in which it
        first holds its final value FOR GOOD — i.e. one past its last change.
        Positions that never change commit at 0."""
        from unturtle.post_training.rollout import commit_steps_from_trajectory

        M = MASK_ID
        trajectory = torch.tensor(
            [
                [M, M, 7, M],  # state before step 0
                [3, M, 7, M],  # pos 0 committed at step... changed between 0->1
                [3, 5, 7, 2],
                [3, 5, 7, 2],  # final
            ]
        )

        steps = commit_steps_from_trajectory(trajectory)

        # pos 0 last changed entering state 1 -> commit step 1
        # pos 1 and 3 last changed entering state 2 -> commit step 2
        # pos 2 never changed -> 0
        assert steps.tolist() == [1, 2, 0, 2]

    def test_a_flip_flop_commits_at_its_last_change(self):
        """A -> B -> A still commits late: the position was NOT final-valued
        for good until its last change, and replaying it as early-committed
        would show the teacher a state the decode never stabilized."""
        from unturtle.post_training.rollout import commit_steps_from_trajectory

        trajectory = torch.tensor([[1], [2], [1], [1]])

        steps = commit_steps_from_trajectory(trajectory)

        assert steps.tolist() == [2]


class TestTrajectoryRecorder:
    def test_it_records_every_denoiser_call_plus_the_final_state(self):
        """The recorder turns any (x_t, t, h) denoiser into a trajectory
        source: the solver hands it each pre-step state, and the caller
        appends the solver's return — no hook into the sampler."""
        from unturtle.models.generation.dfm_solver import solve_discrete_flow
        from unturtle.post_training.rollout import TrajectoryRecorder

        def denoiser(x_t, t, h):
            logits = torch.zeros(*x_t.shape, MASK_ID + 1)
            logits[..., 3] = 10.0
            return logits

        recorder = TrajectoryRecorder(denoiser)
        x_0 = torch.full((1, RESPONSE), MASK_ID, dtype=torch.long)
        final = solve_discrete_flow(
            recorder, x_0, steps=4, generator=torch.Generator().manual_seed(0)
        )
        trajectory = recorder.finish(final)

        assert trajectory.shape == (5, 1, RESPONSE)
        assert torch.equal(trajectory[0], x_0)
        assert torch.equal(trajectory[-1], final)

    def test_finishing_twice_is_rejected(self):
        """A recorder reused across rollouts would silently concatenate two
        prompts' trajectories into one step_map."""
        from unturtle.post_training.rollout import TrajectoryRecorder

        recorder = TrajectoryRecorder(lambda x, t, h: torch.zeros(*x.shape, 4))
        recorder(torch.zeros(1, 4, dtype=torch.long), torch.tensor([0.0]), 0.5)
        recorder.finish(torch.zeros(1, 4, dtype=torch.long))

        with pytest.raises(RuntimeError, match="finish"):
            recorder.finish(torch.zeros(1, 4, dtype=torch.long))

    def test_recorded_states_are_snapshots_not_references(self):
        """A sampler that mutates its state in place must not rewrite history.

        `solve_discrete_flow` happens to allocate fresh tensors each step, so
        a recorder that skipped the clone would pass every solver-driven test
        (mutation-verified survivor) — but the recorder's contract is any
        (x_t, t, h) sampler, including in-place ones.
        """
        from unturtle.post_training.rollout import TrajectoryRecorder

        recorder = TrajectoryRecorder(lambda x, t, h: torch.zeros(*x.shape, 4))
        state = torch.zeros(1, 4, dtype=torch.long)
        recorder(state, torch.tensor([0.0]), 0.5)

        state.fill_(9)  # in-place sampler behaviour

        trajectory = recorder.finish(torch.ones(1, 4, dtype=torch.long))
        assert torch.equal(trajectory[0], torch.zeros(1, 4, dtype=torch.long)), (
            "mutating the passed state rewrote the recorded trajectory"
        )


class TestCommitStepsEdge:
    def test_a_single_state_trajectory_is_rejected_loudly(self):
        from unturtle.post_training.rollout import commit_steps_from_trajectory

        with pytest.raises(ValueError, match="trajectory"):
            commit_steps_from_trajectory(torch.zeros(1, 4, dtype=torch.long))
