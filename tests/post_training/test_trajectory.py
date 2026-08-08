"""
On-policy supervision state contract for OPD (#64 slice A).

`#64` is explicit that at this stage "the main risk here is alignment, not the
KL formula anymore".  The reference
(`dev/repos/opdlm/train/rl_sdar.py:825-838`) keeps three parallel lists —
states, masks, rewards — in sync **by append order alone**, guarded only by a
`continue` that happens to skip all three.  Any later filtering applied to a
subset silently pairs state *i* with reward *j*, and nothing detects it: the
loss stays finite and the model learns the wrong thing.

`SupervisionState` exists so that pairing is by identity rather than position.
These tests pin the properties that make buffering safe, not the tensor maths.
"""

import pytest
import torch


def _state(sample_id="s0", length=8, prompt_length=3, seed=0):
    import math

    from unturtle.post_training.trajectory import SupervisionState

    torch.manual_seed(seed)
    supervision = torch.zeros(length, dtype=torch.bool)
    supervision[prompt_length:] = True
    block_size = 2
    # Derived, not hardcoded: the constructor validates that round_indices
    # covers every block, so a fixed tuple silently constrains `length`.
    n_blocks = math.ceil((length - prompt_length) / block_size)
    return SupervisionState(
        sample_id=sample_id,
        input_ids=torch.randint(1, 50, (length,)),
        supervision_mask=supervision,
        prompt_length=prompt_length,
        block_size=block_size,
        round_indices=tuple(i % 2 for i in range(n_blocks)),
    )


class TestIdentityIsCarried:
    def test_a_state_knows_which_sample_it_came_from(self):
        assert _state("abc").sample_id == "abc"

    def test_states_pair_by_id_not_by_position(self):
        """The property that makes buffering safe.

        Reordering a batch must not change which teacher score belongs to
        which state.  Under the reference's parallel-list scheme this is only
        true by convention; here it is checked.
        """
        from unturtle.post_training.trajectory import SupervisionBatch

        states = [_state(f"s{i}", seed=i) for i in range(4)]
        batch = SupervisionBatch.from_states(states)

        shuffled = SupervisionBatch.from_states(list(reversed(states)))

        for sample_id in ("s0", "s1", "s2", "s3"):
            assert torch.equal(
                batch.select(sample_id).input_ids,
                shuffled.select(sample_id).input_ids,
            ), f"{sample_id} resolved to a different state after reordering"

    def test_input_order_is_preserved(self):
        """`split()` slices positionally, so order is load-bearing.

        Reordering inside `from_states` would still resolve `select()`
        correctly — the mutant is internally consistent — but it changes which
        states share a micro-batch, and desyncs any external parallel array
        the caller still indexes by position (rewards, teacher logprobs: the
        reference's `reward_list`).  Mutation-verified: sorting by `sample_id`
        passes every other test in this file.
        """
        from unturtle.post_training.trajectory import SupervisionBatch

        states = [_state(sid, seed=i) for i, sid in enumerate(["s3", "s1", "s2", "s0"])]

        batch = SupervisionBatch.from_states(states)

        assert batch.sample_ids == ("s3", "s1", "s2", "s0"), (
            "from_states reordered its input; positional consumers would desync"
        )
        assert torch.equal(batch.input_ids[0], states[0].input_ids)

    def test_an_empty_batch_says_so(self):
        """An empty rollout is a plausible runtime state, not a bug here.

        Without this guard the failure still occurs, but as "states have
        differing length []" from the ragged check — sending someone hunting a
        shape bug that does not exist.
        """
        from unturtle.post_training.trajectory import SupervisionBatch

        with pytest.raises(ValueError, match="no states"):
            SupervisionBatch.from_states([])

    def test_selecting_an_unknown_id_raises(self):
        """Silent `None` here would resurface as a shape error much later."""
        from unturtle.post_training.trajectory import SupervisionBatch

        batch = SupervisionBatch.from_states([_state("s0")])

        with pytest.raises(KeyError, match="s9"):
            batch.select("s9")

    def test_duplicate_ids_are_rejected(self):
        """Two states sharing an id makes `select` ambiguous.

        Rejecting at construction is the whole point: a duplicate that
        survives into a buffer produces a wrong pairing that no downstream
        assertion can catch.
        """
        from unturtle.post_training.trajectory import SupervisionBatch

        with pytest.raises(ValueError, match="duplicate"):
            SupervisionBatch.from_states([_state("s0"), _state("s0", seed=1)])


class TestSupervisionSemantics:
    def test_the_prompt_is_never_supervised(self):
        state = _state(prompt_length=3)

        assert not bool(state.supervision_mask[:3].any()), (
            "prompt positions are observed conditioning, not targets"
        )

    def test_a_supervision_mask_of_the_wrong_length_is_rejected(self):
        from unturtle.post_training.trajectory import SupervisionState

        with pytest.raises(ValueError, match="supervision_mask"):
            SupervisionState(
                sample_id="s0",
                input_ids=torch.zeros(8, dtype=torch.long),
                supervision_mask=torch.ones(7, dtype=torch.bool),
                prompt_length=3,
                block_size=2,
                round_indices=(0,),
            )

    def test_a_prompt_longer_than_the_sequence_is_rejected(self):
        from unturtle.post_training.trajectory import SupervisionState

        with pytest.raises(ValueError, match="prompt_length"):
            SupervisionState(
                sample_id="s0",
                input_ids=torch.zeros(4, dtype=torch.long),
                supervision_mask=torch.ones(4, dtype=torch.bool),
                prompt_length=9,
                block_size=2,
                round_indices=(0,),
            )

    def test_supervised_positions_inside_the_prompt_are_rejected(self):
        """Catches an off-by-one in the caller's mask construction.

        A mask that leaks one position into the prompt trains the student to
        reproduce its own conditioning — plausible-looking and hard to spot in
        a loss curve.
        """
        from unturtle.post_training.trajectory import SupervisionState

        mask = torch.ones(8, dtype=torch.bool)  # includes the prompt

        with pytest.raises(ValueError, match="prompt"):
            SupervisionState(
                sample_id="s0",
                input_ids=torch.zeros(8, dtype=torch.long),
                supervision_mask=mask,
                prompt_length=3,
                block_size=2,
                round_indices=(0,),
            )


class TestCleanTargetsAreASeparateField:
    """The layout deliberately differs from the reference's packed one.

    Upstream, `_combine_rounds_one_state_per_block` returns
    `torch.cat([clean_input_ids, tail])` of length `L0 + 2*L1`, while `p_mask`
    is `L0 + L1`.  Every consumer then slices `[:, :L]` to recover the clean
    half (`rl_sdar.py:843,848,851`).  That off-by-`L1` indexing convention,
    re-derived at each use site, is exactly the class of error this contract
    exists to remove — so clean targets live in their own equal-length field.
    """

    def test_clean_targets_are_optional(self):
        assert _state().clean_input_ids is None

    def test_clean_targets_must_match_the_state_length(self):
        from unturtle.post_training.trajectory import SupervisionState

        with pytest.raises(ValueError, match="clean_input_ids"):
            SupervisionState(
                sample_id="s0",
                input_ids=torch.zeros(8, dtype=torch.long),
                supervision_mask=torch.cat(
                    [torch.zeros(3, dtype=torch.bool), torch.ones(5, dtype=torch.bool)]
                ),
                prompt_length=3,
                block_size=2,
                # The reference's packed length, which this contract rejects.
                clean_input_ids=torch.zeros(13, dtype=torch.long),
                round_indices=(0, 1, 0),
            )

    def test_a_state_can_carry_both_noised_and_clean(self):
        from unturtle.post_training.trajectory import SupervisionState

        state = SupervisionState(
            sample_id="s0",
            input_ids=torch.arange(8),
            supervision_mask=torch.cat(
                [torch.zeros(3, dtype=torch.bool), torch.ones(5, dtype=torch.bool)]
            ),
            prompt_length=3,
            block_size=2,
            clean_input_ids=torch.arange(8) * 2,
            round_indices=(0, 1, 0),
        )

        assert state.clean_input_ids is not None
        assert state.clean_input_ids.shape == state.input_ids.shape
        # The two halves are distinguishable, which the packed layout makes
        # depend on getting the slice offset right.
        assert not torch.equal(state.clean_input_ids, state.input_ids)


class TestFrozenIsNotDeep:
    def test_frozen_prevents_reassignment_but_not_tensor_mutation(self):
        """States the limit of `frozen=True` rather than implying more.

        A reader could assume a frozen dataclass makes the state immutable.
        It only blocks attribute rebinding — the tensors inside are still
        writable, which is why `SupervisionBatch` relies on non-aliasing
        rather than on this.
        """
        import dataclasses

        state = _state()

        with pytest.raises(dataclasses.FrozenInstanceError):
            state.prompt_length = 5

        # But this is allowed, and is why non-aliasing carries the contract.
        state.input_ids[0] = 12345
        assert state.input_ids[0] == 12345


class TestRoundProvenance:
    def test_per_block_round_indices_are_recorded(self):
        """Each block independently draws its own denoising round.

        `_combine_rounds_one_state_per_block` stitches blocks from different
        rounds into one row.  Without recording which, a state cannot be
        reproduced or debugged after the fact.
        """
        state = _state()  # length 8, prompt 3 -> 3 blocks of 2

        assert len(state.round_indices) == 3
        assert state.round_indices == (0, 1, 0)

    def test_round_indices_must_cover_every_block(self):
        from unturtle.post_training.trajectory import SupervisionState

        # length 8, prompt 3 -> 5 response positions -> 3 blocks of size 2
        with pytest.raises(ValueError, match="round_indices"):
            SupervisionState(
                sample_id="s0",
                input_ids=torch.zeros(8, dtype=torch.long),
                supervision_mask=torch.cat(
                    [torch.zeros(3, dtype=torch.bool), torch.ones(5, dtype=torch.bool)]
                ),
                prompt_length=3,
                block_size=2,
                round_indices=(0,),  # only one, needs three
            )


class TestBatching:
    def test_stacking_preserves_every_state(self):
        from unturtle.post_training.trajectory import SupervisionBatch

        states = [_state(f"s{i}", seed=i) for i in range(3)]
        batch = SupervisionBatch.from_states(states)

        assert batch.input_ids.shape == (3, 8)
        assert batch.supervision_mask.shape == (3, 8)
        for i, state in enumerate(states):
            assert torch.equal(batch.input_ids[i], state.input_ids)

    def test_ragged_lengths_are_rejected(self):
        """Stacking would either crash or pad silently; say so instead."""
        from unturtle.post_training.trajectory import SupervisionBatch

        with pytest.raises(ValueError, match="length"):
            SupervisionBatch.from_states([_state("a"), _state("b", length=6)])

    def test_a_micro_batch_keeps_its_ids(self):
        """Gradient accumulation splits a batch; identity must survive."""
        from unturtle.post_training.trajectory import SupervisionBatch

        states = [_state(f"s{i}", seed=i) for i in range(4)]
        batch = SupervisionBatch.from_states(states)

        first, second = batch.split(2)

        assert first.sample_ids == ("s0", "s1")
        assert second.sample_ids == ("s2", "s3")
        assert torch.equal(second.input_ids[0], states[2].input_ids)

    def test_splitting_does_not_silently_drop_a_remainder(self):
        from unturtle.post_training.trajectory import SupervisionBatch

        batch = SupervisionBatch.from_states(
            [_state(f"s{i}", seed=i) for i in range(5)]
        )

        chunks = batch.split(2)

        assert [len(c.sample_ids) for c in chunks] == [2, 2, 1]
        assert sum(len(c.sample_ids) for c in chunks) == 5


class TestNoResamplingBetweenCaptureAndScoring:
    def test_capturing_does_not_reach_back_into_the_source_state(self):
        """No re-noising between student capture and teacher scoring.

        PyTorch has no read-only tensor flag, so "immutable" is not
        enforceable directly.  The enforceable half is that the batch and the
        source state do not share storage in *either* direction: mutating the
        batch must not corrupt the state a caller still holds.
        """
        from unturtle.post_training.trajectory import SupervisionBatch

        state = _state()
        original = state.input_ids.clone()
        batch = SupervisionBatch.from_states([state])

        batch.input_ids[0, 0] = 999

        assert torch.equal(state.input_ids, original), (
            "writing to the batch changed the source state; they share storage"
        )

    def test_select_returns_a_borrowed_state_not_a_copy(self):
        """Bounds the non-aliasing guarantee to the stacked tensors.

        `select()` returns the state object itself, so a caller can reach the
        original tensor through it.  Deliberate — `states` is the single
        ordering source of truth — but the docstring previously claimed
        nothing hands storage back out, which was wrong.
        """
        from unturtle.post_training.trajectory import SupervisionBatch

        state = _state()
        batch = SupervisionBatch.from_states([state])

        assert batch.select("s0") is state, (
            "select() now copies; the borrowed-state caveat can be dropped"
        )

    def test_from_states_copies_rather_than_aliasing(self):
        """A caller mutating its own tensor must not change a captured state."""
        from unturtle.post_training.trajectory import SupervisionBatch

        state = _state()
        batch = SupervisionBatch.from_states([state])
        before = batch.input_ids.clone()

        state.input_ids.mul_(0)

        assert torch.equal(batch.input_ids, before), (
            "the batch aliased the caller's tensor; a later mutation would "
            "change what the teacher scores"
        )
