"""
On-policy supervision buffer (#64 slice C).

Slices A and B built the trajectory contract (#91) and the frozen teacher
(#93).  What sits between them and a training step is accumulation: a rollout
produces states in whatever grouping the sampler yields, and the optimizer
consumes them in gradient-accumulation microbatches of a fixed size.  Those two
groupings do not line up, so something has to buffer.

**The failure this exists to prevent is silent mispairing.**  If a state's
identity is carried by list position rather than by `sample_id`, then any
reordering, drop, or uneven regrouping scores a student against a *different*
sample's teacher logits.  The loss stays finite, training still descends, and
the model just learns a worse thing — the same shape of failure as an inverted
objective (#97) or a biased sampler (#95).  So the tests here are about
identity and conservation, not about shapes.

Deliberately no torch device work and no model: this is bookkeeping, and it
should be verifiable without either.
"""

import pytest
import torch

from unturtle.post_training.trajectory import SupervisionState


def _state(sample_id, length=6, prompt_length=2, block_size=2):
    mask = torch.zeros(length, dtype=torch.bool)
    mask[prompt_length:] = True
    return SupervisionState(
        sample_id=sample_id,
        input_ids=torch.arange(length),
        supervision_mask=mask,
        prompt_length=prompt_length,
        block_size=block_size,
    )


class TestItAccumulatesUntilAskedToDrain:
    def test_states_are_held_until_the_target_is_reached(self):
        """A buffer that yields early defeats the point of accumulation.

        Gradient accumulation needs a *fixed* microbatch size; a buffer that
        emits whatever it has would produce uneven batches and silently change
        the effective batch size between steps.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)

        assert buffer.extend([_state("a"), _state("b")]) == []
        assert len(buffer) == 2

        emitted = buffer.extend([_state("c"), _state("d")])

        assert len(emitted) == 1
        assert emitted[0].sample_ids == ("a", "b", "c", "d")
        assert len(buffer) == 0

    def test_a_long_push_emits_several_full_batches(self):
        """Rollout grouping and microbatch size are unrelated.

        Ten states at batch_size 4 must give two full batches and keep the
        remainder, not one batch and a silent drop.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)

        emitted = buffer.extend([_state(str(i)) for i in range(10)])

        assert [len(batch.sample_ids) for batch in emitted] == [4, 4]
        assert len(buffer) == 2

    def test_drain_emits_a_short_final_batch(self):
        """The tail must be reachable, but only when explicitly asked for.

        An epoch boundary has to flush; a mid-epoch push must not.  Keeping
        that distinction explicit is what stops a short batch from appearing
        where a full one was assumed.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)
        buffer.extend([_state("a"), _state("b")])

        drained = buffer.drain()

        assert len(drained) == 1
        assert drained[0].sample_ids == ("a", "b")
        assert len(buffer) == 0

    def test_draining_an_empty_buffer_yields_nothing(self):
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)

        assert buffer.drain() == []


class TestIdentityIsPreserved:
    """The property whose absence is invisible in a loss curve."""

    def test_every_pushed_id_comes_back_exactly_once(self):
        """Conservation across an uneven push/emit pattern.

        Neither dropped nor duplicated: a dropped state silently shrinks the
        effective batch, and a duplicated one double-counts its gradient.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=3)
        pushed = [str(i) for i in range(17)]

        seen = []
        for sample_id in pushed:
            for batch in buffer.extend([_state(sample_id)]):
                seen.extend(batch.sample_ids)
        for batch in buffer.drain():
            seen.extend(batch.sample_ids)

        assert seen == pushed, (
            "ids came back reordered, dropped or duplicated; supervision would "
            "be paired with the wrong sample"
        )

    def test_a_duplicate_id_is_rejected(self):
        """Two states claiming one id makes pairing ambiguous.

        `SupervisionBatch.from_states` already rejects duplicates *within* a
        batch, but a buffer spans batches — the same id arriving in two
        separate pushes would slip past that check and only surface as a
        mispaired teacher score.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        # batch_size=2 so the first push is *emitted* and nothing is pending
        # when the duplicate arrives.  Pushing into a half-full buffer would
        # pass even if the check only looked at pending states, which is the
        # implementation this test exists to rule out.
        buffer = SupervisionBuffer(batch_size=2)
        buffer.extend([_state("a"), _state("b")])
        assert len(buffer) == 0, "the first push must have been emitted"

        with pytest.raises(ValueError, match="duplicate"):
            buffer.extend([_state("a")])

    def test_a_duplicate_within_a_single_push_is_rejected(self):
        """Both copies arrive together, so neither is in `_seen` yet."""
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=8)

        with pytest.raises(ValueError, match="duplicate"):
            buffer.extend([_state("a"), _state("b"), _state("a")])

    def test_an_id_stays_spent_across_a_drain(self):
        """ "Spent for the buffer's lifetime" — including past a flush.

        A drain that cleared the seen-set would let a replayed epoch silently
        re-admit every id.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)
        buffer.extend([_state("a")])
        buffer.drain()

        with pytest.raises(ValueError, match="duplicate"):
            buffer.extend([_state("a")])

    def test_order_is_first_in_first_out(self):
        """Not required for correctness — pairing is by id — but required for
        *reproducibility*: a seeded run must emit the same batches twice.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=2)

        emitted = buffer.extend([_state(c) for c in "abcd"])

        assert [batch.sample_ids for batch in emitted] == [("a", "b"), ("c", "d")]


class TestItRejectsIncoherentConfiguration:
    @pytest.mark.parametrize("size", [0, -1])
    def test_a_non_positive_batch_size_is_rejected(self, size):
        """Silently looping forever or emitting empty batches is worse."""
        from unturtle.post_training.buffer import SupervisionBuffer

        with pytest.raises(ValueError, match="batch_size"):
            SupervisionBuffer(batch_size=size)

    def test_mixed_block_sizes_are_rejected(self):
        """A batch is scored under one denoising block width.

        Mixing widths in one batch would apply the wrong block structure to
        some rows — finite, plausible, and wrong.  Caught at push time, where
        the offending state is still identifiable, rather than at loss time.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)
        buffer.extend([_state("a", block_size=2)])

        with pytest.raises(ValueError, match="block_size"):
            buffer.extend([_state("b", block_size=4)])


class TestAPushIsAllOrNothing:
    """A rejected push must leave the buffer exactly as it was.

    Committing states one at a time and raising partway through loses every
    state after the offending one — they are never recorded, so no diagnostic
    can name them and the caller's effective batch silently shrinks.  It also
    makes the push unretryable: the states already committed come back as
    duplicates, and the second error blames the wrong state entirely.
    """

    def test_a_rejected_push_commits_nothing(self):
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=8)
        buffer.extend([_state("a")])

        with pytest.raises(ValueError, match="block_size"):
            buffer.extend([_state("b"), _state("c", block_size=4), _state("d")])

        assert len(buffer) == 1, (
            f"{len(buffer)} states pending after a rejected push; states before "
            "the offending one were committed and states after it were dropped"
        )

    def test_a_rejected_push_can_be_retried_once_corrected(self):
        """The practical consequence of atomicity.

        A caller that catches the error, fixes the offending state and retries
        must not be told its own untouched states are duplicates.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=8)
        buffer.extend([_state("a")])

        with pytest.raises(ValueError, match="block_size"):
            buffer.extend([_state("b"), _state("c", block_size=4), _state("d")])

        buffer.extend([_state("b"), _state("c"), _state("d")])

        assert buffer.drain()[0].sample_ids == ("a", "b", "c", "d")

    def test_a_bad_first_state_does_not_pin_the_block_size(self):
        """The pin must not outlive the push that was rejected.

        Otherwise one malformed state poisons the buffer permanently: every
        subsequent well-formed state is rejected against a width that was
        never accepted.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)

        with pytest.raises(ValueError):
            buffer.extend([_state("bad", block_size=3), _state("x", block_size=2)])

        buffer.extend([_state("good", block_size=2)])

        assert len(buffer) == 1


class TestDrainStaysWithinTheMicrobatch:
    def test_drain_can_never_emit_an_oversized_batch(self):
        """The invariant that makes `drain`'s single-batch return sound.

        `extend` loops until fewer than `batch_size` states remain, so pending
        is always below `batch_size` on return — which is why `drain` can emit
        one batch rather than re-chunking.  Asserting the invariant rather than
        the return shape: if `extend` ever stopped maintaining it, `drain`
        would hand the optimizer an oversized microbatch.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)

        for pushed in range(1, 20):
            buffer.extend([_state(f"s{pushed}")])
            assert len(buffer) < 4, (
                f"{len(buffer)} pending after pushing {pushed}; extend left a "
                "full batch unemitted"
            )

        drained = buffer.drain()

        assert len(drained) == 1
        assert len(drained[0].sample_ids) < 4


class TestTheBatchShapeSurvivesADrain:
    """A flush empties the pending queue, not the buffer's contract.

    `drain` is an epoch boundary, not a reset: the same buffer keeps scoring
    one denoising block width.  Without this, a drained buffer silently accepts
    a different width, and the next batch is scored under a block structure
    that disagrees with everything before it.
    """

    def test_the_block_size_pin_persists_across_a_drain(self):
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)
        buffer.extend([_state("a", block_size=2)])
        buffer.drain()

        with pytest.raises(ValueError, match="block_size"):
            buffer.extend([_state("b", block_size=4)])

    def test_the_prompt_length_pin_persists_across_a_drain(self):
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)
        buffer.extend([_state("a", prompt_length=2)])
        buffer.drain()

        with pytest.raises(ValueError, match="prompt_length"):
            buffer.extend([_state("b", prompt_length=4)])


class TestPromptLengthMustAgree:
    def test_mixed_prompt_lengths_are_rejected(self):
        """`block_size` alone does not make a batch coherent.

        `SupervisionState` derives its block count from
        `(length - prompt_length) / block_size`, so two rows sharing a length
        and a block width can still span different numbers of blocks.  A
        consumer indexing block boundaries uniformly then reads the wrong
        positions for some rows — finite, plausible, wrong.  Neither
        `from_states` nor the buffer checked this.
        """
        from unturtle.post_training.buffer import SupervisionBuffer

        buffer = SupervisionBuffer(batch_size=4)
        buffer.extend([_state("a", prompt_length=2)])

        with pytest.raises(ValueError, match="prompt_length"):
            buffer.extend([_state("b", prompt_length=4)])
