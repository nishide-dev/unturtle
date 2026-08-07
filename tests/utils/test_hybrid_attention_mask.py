"""
Hybrid causal-bidirectional attention mask (#63, PreDiff-LM).

Implements equation (3) of arXiv:2607.25157 §3.2, over a prompt prefix of
length ``Lp`` followed by a target region::

            | 1[j <= i]   i, j in prompt
    M_ij =  | 1           i in target, j in prompt
            | 1           i, j in target
            | 0           i in prompt, j in target

Two entries are easy to get wrong from the paper's prose alone, so they are
pinned individually below:

- **target -> prompt is unconditional**, not `1[j <= i]`.  A target token sees
  the whole prompt, including prompt positions after it.  "Preserves causal
  attention within the observed prompt" reads like the causal constraint
  extends across the boundary; it does not.
- **prompt -> target is blocked**, which is the actual mechanism.  The paper's
  stated reason is to prevent the corrupted target from perturbing prompt
  representations — keeping prompt-side computation in the regime the AR
  weights were pretrained in.  Uniform bidirectional attention already gives
  targets full context, so this asymmetry is what distinguishes the recipe.
"""

import pytest
import torch


def _mask(prompt_lengths, seq_len, **kwargs):
    from unturtle.utils.packing import build_hybrid_prefix_attention_mask

    return build_hybrid_prefix_attention_mask(
        prompt_lengths=torch.tensor(prompt_lengths),
        seq_len=seq_len,
        dtype=torch.float32,
        device=torch.device("cpu"),
        **kwargs,
    )


def _allowed(mask):
    """Additive mask -> bool `can attend` matrix."""
    return mask == 0


class TestEquationThree:
    def test_prompt_region_is_causal(self):
        allowed = _allowed(_mask([3], seq_len=5))[0, 0]

        for i in range(3):
            for j in range(3):
                assert bool(allowed[i, j]) == (j <= i), (
                    f"prompt[{i}] -> prompt[{j}] should be {j <= i}"
                )

    def test_target_region_is_bidirectional(self):
        allowed = _allowed(_mask([2], seq_len=5))[0, 0]

        for i in range(2, 5):
            for j in range(2, 5):
                assert bool(allowed[i, j]), f"target[{i}] -> target[{j}] must be open"

    def test_target_sees_the_entire_prompt(self):
        allowed = _allowed(_mask([3], seq_len=6))[0, 0]

        for i in range(3, 6):
            for j in range(3):
                assert bool(allowed[i, j]), f"target[{i}] must see prompt[{j}]"

    def test_a_target_row_is_open_across_its_full_width(self):
        """Cases 2 and 3 together: a target query attends everywhere.

        Note what this *cannot* establish. Equation (3) case 2 is `1`, not
        `1[j <= i]`, but under a prompt prefix those are indistinguishable —
        every target index already exceeds every prompt index, so the extra
        term is vacuously true. Verified rather than assumed: writing
        `key <= query` into the target branch produces byte-identical output
        across all 45 (L <= 8, Lp) configurations. It is a semantically null
        change here, not a coverage gap, and it stops being null only if
        interleaved prompt/target is ever supported — which the paper does
        not define.
        """
        allowed = _allowed(_mask([3], seq_len=7))[0, 0]

        for i in range(3, 7):
            assert bool(allowed[i, :].all()), (
                f"target row {i} must be open across the whole sequence"
            )

    def test_prompt_cannot_see_the_target(self):
        """The asymmetry that defines the recipe.

        Under uniform bidirectional attention this quadrant is open; blocking
        it is what keeps the corrupted target from changing the prompt's
        representations.
        """
        allowed = _allowed(_mask([3], seq_len=6))[0, 0]

        for i in range(3):
            for j in range(3, 6):
                assert not bool(allowed[i, j]), (
                    f"prompt[{i}] must NOT see target[{j}]: the corrupted "
                    "target would perturb pretrained prompt computation"
                )

    def test_matches_equation_three_exactly(self):
        """Whole-matrix check, so no quadrant can drift unnoticed."""
        Lp, L = 3, 7
        allowed = _allowed(_mask([Lp], seq_len=L))[0, 0]

        for i in range(L):
            for j in range(L):
                # One branch per line of equation (3), deliberately not
                # collapsed: cases 2 and 3 share a value but are distinct
                # claims (target->prompt, target->target), and merging them
                # would hide which one a failure came from.
                if i < Lp and j < Lp:  # noqa: SIM114
                    expected = j <= i
                elif i >= Lp and j < Lp:  # noqa: SIM114
                    expected = True
                elif i >= Lp and j >= Lp:
                    expected = True
                else:
                    expected = False
                assert bool(allowed[i, j]) == expected, (
                    f"M[{i},{j}]: expected {expected} (Lp={Lp})"
                )


class TestDegenerateCases:
    def test_zero_prompt_is_fully_bidirectional(self):
        """Lp=0 must reduce to uniform bidirectional attention."""
        allowed = _allowed(_mask([0], seq_len=4))[0, 0]

        assert bool(allowed.all()), "Lp=0 should leave every position open"

    def test_full_prompt_is_fully_causal(self):
        """Lp=L must reduce to the ordinary causal mask."""
        L = 4
        allowed = _allowed(_mask([L], seq_len=L))[0, 0]
        expected = torch.tril(torch.ones(L, L, dtype=torch.bool))

        assert torch.equal(allowed, expected), "Lp=L should be exactly lower-triangular"

    def test_degenerate_cases_equal_the_existing_builders_exactly(self):
        """Independent cross-check against code that predates this mask.

        At `Lp=0` the hybrid mask is a pure bidirectional mask, and at `Lp=L`
        a pure causal one.  The packed builders produce those two masks by a
        completely different construction (block-diagonal fill loops rather
        than index broadcasting), so exact equality is evidence the new
        implementation is right rather than merely self-consistent with its
        own tests.
        """
        from unturtle.utils.packing import (
            build_sdpa_packed_attention_mask,
            build_sdpa_packed_bidirectional_attention_mask,
        )

        L = 6
        dtype, device = torch.float32, torch.device("cpu")
        seq_info = (torch.tensor([L]), None, L)

        bidirectional = _mask([0], seq_len=L)[0, 0]
        assert torch.equal(
            bidirectional,
            build_sdpa_packed_bidirectional_attention_mask(
                seq_info, dtype=dtype, device=device
            )[0, 0],
        ), "Lp=0 must equal the bidirectional packed mask"

        causal = _mask([L], seq_len=L)[0, 0]
        assert torch.equal(
            causal,
            build_sdpa_packed_attention_mask(seq_info, dtype=dtype, device=device)[
                0, 0
            ],
        ), "Lp=L must equal the causal packed mask"

    def test_prompt_longer_than_sequence_is_rejected(self):
        with pytest.raises(ValueError, match="prompt_lengths"):
            _mask([5], seq_len=4)

    def test_negative_prompt_length_is_rejected(self):
        with pytest.raises(ValueError, match="prompt_lengths"):
            _mask([-1], seq_len=4)


class TestBatching:
    def test_each_row_uses_its_own_prompt_length(self):
        allowed = _allowed(_mask([1, 3], seq_len=4))

        # Row 0: Lp=1 -> position 0 sees only itself.
        assert not bool(allowed[0, 0, 0, 1]), "row 0 prompt must not see target"
        # Row 1: Lp=3 -> position 1 sees position 0 (causal), not position 3.
        assert bool(allowed[1, 0, 1, 0]), "row 1 prompt is causal within itself"
        assert not bool(allowed[1, 0, 1, 3]), "row 1 prompt must not see target"

    def test_shape_is_broadcastable_over_heads(self):
        mask = _mask([2, 2], seq_len=5)

        assert mask.shape == (2, 1, 5, 5), (
            f"expected [B, 1, L, L] for head broadcasting, got {tuple(mask.shape)}"
        )


class TestPadding:
    def test_padding_is_excluded_on_both_axes(self):
        """A padded position must neither attend nor be attended to.

        Leaving padding attendable is the classic silent bug: attention still
        runs, the loss still decreases, and real tokens quietly mix in
        embeddings of nothing.
        """
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        allowed = _allowed(
            _mask([2], seq_len=5, attention_mask=attention_mask),
        )[0, 0]

        assert not bool(allowed[:, 3:].any()), "nothing may attend to padding"
        assert not bool(allowed[3:, :].any()), "padding may not attend to anything"
        assert bool(allowed[2, 0]), "real target must still see real prompt"

    def test_padding_does_not_shift_the_prompt_boundary(self):
        """`Lp` counts real prompt tokens, and padding sits at the end."""
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
        allowed = _allowed(
            _mask([2], seq_len=5, attention_mask=attention_mask),
        )[0, 0]

        assert not bool(allowed[1, 2]), "prompt[1] must not see target[2]"
        assert bool(allowed[2, 1]), "target[2] must see prompt[1]"


class TestAdditiveMaskConvention:
    def test_blocked_entries_are_negative_infinity(self):
        """SDPA consumes additive masks; blocked must be -inf, not 0/1."""
        mask = _mask([2], seq_len=4)

        assert mask.dtype == torch.float32
        blocked = mask[0, 0, 0, 2]
        assert blocked == float("-inf"), f"blocked entry was {blocked}, want -inf"
        assert mask[0, 0, 0, 0] == 0.0, "allowed entry must be additive zero"

    def test_dtype_is_respected(self):
        from unturtle.utils.packing import build_hybrid_prefix_attention_mask

        mask = build_hybrid_prefix_attention_mask(
            prompt_lengths=torch.tensor([2]),
            seq_len=4,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )

        assert mask.dtype == torch.bfloat16
