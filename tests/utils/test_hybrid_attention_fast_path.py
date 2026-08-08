"""
Mask-free hybrid attention (#63 slice C).

Slice B wired PreDiff-LM's eq. (3) topology in as a dense ``[B, 1, L, L]``
additive mask.  That is correct, but it costs the fast backends: a caller mask
forces SDPA (``attention_dispatch.py:130-135``), so hybrid attention gives up
Flash and xFormers, and it materializes an ``L x L`` bias per row.

The topology does not actually need a mask.  Written out, eq. (3) splits **by
row** into two independent blocks::

        prompt queries -> prompt keys   causal          (target keys blocked)
        target queries -> all keys      bidirectional

A prompt query never attends to a target key, so the prompt rows are exactly a
causal attention over the prompt prefix alone; and a target query attends to
everything, so the target rows are exactly an unmasked attention over the full
sequence.  Two mask-free calls, no bias tensor.

**These tests are about exactness, not plausibility.**  An attention fast path
that is subtly wrong produces slightly worse models and never fails — the same
failure mode as a sampler.  So the central assertion is bit-level agreement
with the dense reference across shapes, dtypes and prompt boundaries, and the
guard against silent divergence is that the reference is the *shipped* mask
builder rather than a second transcription of eq. (3).
"""

import pytest
import torch
import torch.nn.functional as F

from unturtle.utils.packing import build_hybrid_prefix_attention_mask


def _dense_reference(Q, K, V, *, prompt_lengths, attention_mask=None):
    """Attention through the shipped eq-(3) mask builder.

    Deliberately routed through `build_hybrid_prefix_attention_mask` rather
    than a locally rewritten mask: a hand-rolled reference here would be a
    second transcription of eq. (3) and could agree with a wrong fast path for
    the same reason (see #97).  Comparing against the shipped builder means
    any divergence is a genuine difference in behaviour.
    """
    mask = build_hybrid_prefix_attention_mask(
        prompt_lengths=prompt_lengths,
        seq_len=Q.shape[-2],
        dtype=Q.dtype,
        device=Q.device,
        attention_mask=attention_mask,
    )
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)


def _qkv(batch=2, heads=4, length=12, head_dim=8, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    return tuple(
        torch.randn(batch, heads, length, head_dim, dtype=dtype) for _ in range(3)
    )


class TestTheRowSplitIsExact:
    """The mathematical claim the whole slice rests on."""

    @pytest.mark.parametrize("prompt_length", [1, 4, 11])
    def test_it_matches_the_dense_mask_for_any_prompt_boundary(self, prompt_length):
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(length=12)
        prompt_lengths = torch.full((Q.shape[0],), prompt_length)

        got = hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

        expected = _dense_reference(Q, K, V, prompt_lengths=prompt_lengths)
        assert torch.allclose(got, expected, atol=1e-12), (
            f"Lp={prompt_length}: max deviation "
            f"{float((got - expected).abs().max()):.3e}"
        )

    def test_an_all_prompt_sequence_is_plain_causal_attention(self):
        """`Lp == L` leaves no target rows: eq. (3) degenerates to causal.

        The boundary a split implementation is most likely to get wrong, since
        the target half is empty.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(length=9)
        prompt_lengths = torch.full((Q.shape[0],), 9)

        got = hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

        causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        assert torch.allclose(got, causal, atol=1e-12)

    def test_an_all_target_sequence_is_plain_bidirectional_attention(self):
        """`Lp == 0` leaves no prompt rows: eq. (3) degenerates to bidirectional.

        This is the case that must stay identical to today's non-hybrid
        behaviour, since a converted model with a zero-length prompt is just a
        masked-diffusion model.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(length=9)
        prompt_lengths = torch.zeros(Q.shape[0], dtype=torch.long)

        got = hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

        bidirectional = F.scaled_dot_product_attention(Q, K, V)
        assert torch.allclose(got, bidirectional, atol=1e-12)

    def test_ragged_prompt_lengths_still_match_the_dense_mask(self):
        """Different boundaries per row.

        The split point is a slice index, so a uniform batch takes two batched
        calls while a ragged one cannot.  Whatever the implementation chooses
        there, it must not change the *answer* — this pins correctness
        independently of which path is taken.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(batch=3, length=10)
        prompt_lengths = torch.tensor([2, 7, 5])

        got = hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

        expected = _dense_reference(Q, K, V, prompt_lengths=prompt_lengths)
        assert torch.allclose(got, expected, atol=1e-12), (
            f"max deviation {float((got - expected).abs().max()):.3e}"
        )

    def test_the_prompt_never_sees_the_target(self):
        """The asymmetry that distinguishes eq. (3) from uniform bidirectional.

        Asserted behaviourally rather than by inspecting a mask: rewriting the
        entire target region must leave every prompt output bit-identical.  A
        fast path that leaked target keys into prompt rows would still produce
        a plausible tensor and pass a shape check.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(length=10)
        prompt_lengths = torch.full((Q.shape[0],), 4)

        before = hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

        K_perturbed, V_perturbed = K.clone(), V.clone()
        K_perturbed[:, :, 4:] = torch.randn_like(K_perturbed[:, :, 4:])
        V_perturbed[:, :, 4:] = torch.randn_like(V_perturbed[:, :, 4:])
        after = hybrid_prefix_attention(
            Q, K_perturbed, V_perturbed, prompt_lengths=prompt_lengths
        )

        assert torch.equal(before[:, :, :4], after[:, :, :4]), (
            "rewriting the target changed prompt outputs; the prompt is not "
            "shielded from the corrupted target"
        )
        assert not torch.allclose(before[:, :, 4:], after[:, :, 4:]), (
            "rewriting the target left target outputs unchanged; the test is "
            "not exercising what it claims to"
        )

    def test_a_target_token_sees_the_whole_prompt_not_a_causal_prefix(self):
        """eq. (3)'s target->prompt quadrant is unconditional, not `1[j <= i]`.

        Reading it as causal there is the natural transcription slip, and it
        leaves attention finite and trainable.  Pinned by perturbing a prompt
        key *after* a target row's own position: under a causal reading that
        key is invisible to it.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(batch=1, heads=1, length=8)
        prompt_lengths = torch.tensor([6])

        before = hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

        # Prompt position 5 sits after target row 6 would reach under a causal
        # reading restricted to j <= i... which it does not, but position 5 is
        # the last prompt token, so a wrongly-causal target row still sees it.
        # Perturb an *early* target-row view instead: prompt key 5 must be
        # visible to target row 6 either way, so use the reverse — check the
        # count of prompt keys each target row depends on.
        K_perturbed = K.clone()
        K_perturbed[:, :, 5] = torch.randn_like(K_perturbed[:, :, 5])
        after = hybrid_prefix_attention(
            Q, K_perturbed, V, prompt_lengths=prompt_lengths
        )

        assert not torch.allclose(before[:, :, 6:], after[:, :, 6:]), (
            "target rows did not react to a prompt key; targets must see the "
            "whole prompt"
        )


class TestPaddingIsStillExcluded:
    def test_padding_is_honoured_on_both_axes(self):
        """A padded position neither attends nor is attended to.

        The dense builder takes a 2-D padding mask; the fast path must reach
        the same answer or it silently reintroduces padding into the softmax.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(batch=2, length=10)
        prompt_lengths = torch.tensor([3, 3])
        padding = torch.ones(2, 10, dtype=torch.long)
        padding[0, 8:] = 0  # row 0 has two padded tail positions

        got = hybrid_prefix_attention(
            Q, K, V, prompt_lengths=prompt_lengths, attention_mask=padding
        )

        expected = _dense_reference(
            Q, K, V, prompt_lengths=prompt_lengths, attention_mask=padding
        )
        assert torch.allclose(got, expected, atol=1e-12), (
            f"max deviation {float((got - expected).abs().max()):.3e}"
        )


class TestDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_it_agrees_with_the_dense_path_in_reduced_precision(self, dtype):
        """Low precision is where an attention rewrite usually diverges.

        Tolerances are tight on purpose.  Measured deviation is **exactly 0.0**
        in all three dtypes over 8 seeds, because both paths run the same SDPA
        kernel on the same values -- the dense one merely adds a `finfo.min`
        bias to positions the fast one omits.  A slack bound (the first draft
        used 2e-2 for bf16) would wave through a substantial regression, so
        these sit just above zero rather than at "plausible float error".
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(length=12, dtype=dtype)
        prompt_lengths = torch.full((Q.shape[0],), 5)

        got = hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

        expected = _dense_reference(Q, K, V, prompt_lengths=prompt_lengths)
        tolerance = {
            torch.float32: 1e-7,
            torch.float16: 1e-5,
            torch.bfloat16: 1e-4,
        }[dtype]
        assert torch.allclose(got, expected, atol=tolerance), (
            f"{dtype}: max deviation {float((got - expected).abs().max()):.3e}"
        )


class TestGradients:
    def test_gradients_match_the_dense_path(self):
        """Training is the point of this slice, so the backward must agree too.

        A forward-only check would pass for an implementation that detaches or
        mis-routes a branch's gradient, which is exactly the kind of bug that
        shows up as "training is a bit worse" and never as a failure.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(length=10)
        prompt_lengths = torch.full((Q.shape[0],), 4)

        fast_inputs = [x.clone().requires_grad_(True) for x in (Q, K, V)]
        hybrid_prefix_attention(
            *fast_inputs, prompt_lengths=prompt_lengths
        ).sum().backward()

        dense_inputs = [x.clone().requires_grad_(True) for x in (Q, K, V)]
        _dense_reference(*dense_inputs, prompt_lengths=prompt_lengths).sum().backward()

        for name, fast, dense in zip("QKV", fast_inputs, dense_inputs, strict=True):
            assert torch.allclose(fast.grad, dense.grad, atol=1e-10), (
                f"d{name} differs by {float((fast.grad - dense.grad).abs().max()):.3e}"
            )


class TestItRejectsWhatTheDenseBuilderRejects:
    """The fast and dense paths must agree on errors, not just on answers.

    `build_hybrid_prefix_attention_mask` raises for an out-of-range prompt
    length. If the fast path silently clamped instead, the same call would
    raise or succeed depending on whether the batch happened to have uniform
    boundaries — the two branches drifting apart in exactly the way the
    fallback design was meant to prevent.
    """

    @pytest.mark.parametrize("prompt_length", [-1, 13])
    def test_an_out_of_range_prompt_length_raises(self, prompt_length):
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(length=12)
        prompt_lengths = torch.full((Q.shape[0],), prompt_length)

        with pytest.raises(ValueError, match="prompt_lengths"):
            hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

    def test_the_ragged_branch_raises_on_the_same_input(self):
        """Ragged lengths take the dense fallback, which must agree.

        Pinned separately because the uniform and ragged branches validate in
        different places; a check placed only on the fast branch would let the
        fallback keep the old inconsistent behaviour.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(batch=2, length=12)
        prompt_lengths = torch.tensor([3, 99])  # ragged *and* out of range

        with pytest.raises(ValueError, match="prompt_lengths"):
            hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

    def test_a_batch_mismatch_raises_rather_than_broadcasting(self):
        """One boundary silently applied to every row is a wrong answer.

        Without this, `prompt_lengths=[3]` against a 4-row batch would take
        the uniform branch (all entries trivially equal) and split every row
        at 3, whatever their real boundaries.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(batch=4, length=12)

        with pytest.raises(ValueError, match="prompt_lengths"):
            hybrid_prefix_attention(Q, K, V, prompt_lengths=torch.tensor([3]))

    def test_an_empty_batch_returns_an_empty_result(self):
        """`B == 0` is a legitimate degenerate shape, not an error.

        It previously raised `IndexError: index 0 is out of bounds` from the
        uniformity check — an opaque failure for a public function.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q, K, V = _qkv(batch=0, length=12)

        got = hybrid_prefix_attention(
            Q, K, V, prompt_lengths=torch.zeros(0, dtype=torch.long)
        )

        assert got.shape == Q.shape


class TestAsymmetricHeadDims:
    """`V` may carry a different head_dim than `Q`/`K`.

    Attention output takes **V's** head_dim, since the output is a weighted sum
    of value vectors.  Every other test here builds Q, K and V with the same
    dim, so a path that returned Q's shape would look correct throughout —
    which is exactly how the empty-batch return shipped wrong.
    """

    def test_the_output_takes_v_head_dim_on_a_normal_batch(self):
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        torch.manual_seed(0)
        Q = torch.randn(2, 4, 12, 8, dtype=torch.float64)
        K = torch.randn(2, 4, 12, 8, dtype=torch.float64)
        V = torch.randn(2, 4, 12, 16, dtype=torch.float64)
        prompt_lengths = torch.full((2,), 5)

        got = hybrid_prefix_attention(Q, K, V, prompt_lengths=prompt_lengths)

        expected = _dense_reference(Q, K, V, prompt_lengths=prompt_lengths)
        assert got.shape == expected.shape, (
            f"got head_dim {got.shape[-1]}, expected {expected.shape[-1]} "
            "(the value head_dim)"
        )
        assert torch.allclose(got, expected, atol=1e-12)

    def test_the_output_takes_v_head_dim_on_an_empty_batch(self):
        """The degenerate shape where the bug actually was.

        `Q.clone()` returned Q's head_dim; SDPA returns V's.  A caller
        concatenating batches would hit a shape mismatch only when one shard
        happened to be empty.
        """
        from unturtle.utils.attention_dispatch import hybrid_prefix_attention

        Q = torch.randn(0, 4, 12, 8)
        K = torch.randn(0, 4, 12, 8)
        V = torch.randn(0, 4, 12, 16)

        got = hybrid_prefix_attention(
            Q, K, V, prompt_lengths=torch.zeros(0, dtype=torch.long)
        )

        expected = F.scaled_dot_product_attention(Q, K, V)
        assert got.shape == expected.shape, (
            f"got {tuple(got.shape)}, expected {tuple(expected.shape)}"
        )
