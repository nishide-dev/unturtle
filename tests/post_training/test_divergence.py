"""
Teacher-student divergence objectives for OPD distillation (#64).

Semantics are taken from the official OPDLM implementation
(`dev/repos/opdlm/train/rl_sdar.py`, MIT), not from the paper summary.  Two
properties there are load-bearing and easy to lose:

- **Non-finite teacher logprobs must contribute zero.**  `p·log p -> 0` as
  `p -> 0`, but computed directly it is `0 * -inf = NaN`.  Top-k makes this
  reachable: if the teacher has fewer than `k` tokens with non-zero
  probability, some gathered indices carry `-inf`.
- **Top-k is a partial sum, not a renormalized distribution.**  It sums the
  teacher-weighted terms over the top-k only.  That makes `k=V` reproduce the
  dense value exactly, and it also means the truncated value is *not* bounded
  above by the dense one — the summand is negative wherever the student
  assigns more mass than the teacher.
"""

import pytest
import torch
import torch.nn.functional as F


def _logprobs(batch=2, length=3, vocab=32, seed=0):
    torch.manual_seed(seed)
    teacher = F.log_softmax(torch.randn(batch, length, vocab), dim=-1)
    student = F.log_softmax(torch.randn(batch, length, vocab), dim=-1)
    return teacher, student


class TestForwardKL:
    def test_matches_a_hand_computed_reference(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()
        expected = (teacher.exp() * (teacher - student)).sum(dim=-1)

        got = teacher_student_divergence(teacher, student)

        assert torch.allclose(got, expected, atol=1e-6)

    def test_is_zero_when_the_student_matches_the_teacher(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, _ = _logprobs()

        got = teacher_student_divergence(teacher, teacher.clone())

        assert torch.allclose(got, torch.zeros_like(got), atol=1e-6)

    def test_is_non_negative_for_the_dense_case(self):
        """Full-vocabulary KL is a genuine divergence, so it cannot go below 0.

        This is what makes the top-k case below surprising: truncation breaks
        the property, because the guarantee comes from summing over the whole
        distribution.
        """
        from unturtle.post_training.divergence import teacher_student_divergence

        for seed in range(5):
            teacher, student = _logprobs(seed=seed)
            got = teacher_student_divergence(teacher, student)
            assert bool((got >= -1e-6).all()), f"seed {seed}: {got.min().item()}"


class TestTopK:
    def test_k_equal_to_vocab_reproduces_the_dense_value(self):
        """#64 acceptance criterion.

        Holds only because top-k is an unnormalized partial sum: renormalizing
        the truncated teacher distribution would break this at every k < V and
        would not agree at k = V either, since the terms would be reweighted.
        """
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs(vocab=32)

        dense = teacher_student_divergence(teacher, student)
        full_k = teacher_student_divergence(teacher, student, top_k=32)

        assert torch.allclose(dense, full_k, atol=1e-6), (
            f"dense={dense.flatten()[:3]} top_k=V={full_k.flatten()[:3]}"
        )

    def test_top_k_is_an_unnormalized_partial_sum(self):
        """Pins the semantics at a `k` where truncation is actually real.

        `test_k_equal_to_vocab_reproduces_the_dense_value` was written to
        catch a renormalizing implementation, and cannot: at `k = V`,
        `log_softmax` over an already-normalized distribution is a no-op.
        Mutation-verified — inserting a renormalization after the `topk`
        passes every other test here while changing the k=8 loss by 2.3x.

        So the partial-sum property is pinned directly, at `k < V`.
        """
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs(vocab=32)
        k = 8

        top_teacher, indices = teacher.topk(k=k, dim=-1)
        expected = (
            top_teacher.exp() * (top_teacher - student.gather(-1, indices))
        ).sum(dim=-1)

        got = teacher_student_divergence(teacher, student, top_k=k)

        assert torch.allclose(got, expected, atol=1e-6), (
            f"got={got.flatten()[:3]} expected={expected.flatten()[:3]}"
        )

        # The defining property: the retained teacher mass is < 1, i.e. the
        # truncated distribution was NOT renormalized.
        retained = top_teacher.exp().sum(dim=-1)
        assert bool((retained < 0.999).any()), (
            "every position retained ~all teacher mass, so this fixture "
            "cannot distinguish a renormalized implementation"
        )

    def test_k_larger_than_vocab_is_clamped(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs(vocab=32)

        assert torch.allclose(
            teacher_student_divergence(teacher, student, top_k=1000),
            teacher_student_divergence(teacher, student),
            atol=1e-6,
        )

    def test_truncation_is_not_bounded_by_the_dense_value(self):
        """Pins a counter-intuitive property so nobody "fixes" it.

        `p_t·(log p_t − log p_s)` is negative wherever the student assigns
        more mass than the teacher, so dropping tail terms can *increase* the
        sum.  Anyone assuming `top_k <= dense` and adding a clamp would be
        changing the reference objective.
        """
        from unturtle.post_training.divergence import teacher_student_divergence

        exceeded = False
        for seed in range(8):
            teacher, student = _logprobs(vocab=64, seed=seed)
            dense = teacher_student_divergence(teacher, student)
            truncated = teacher_student_divergence(teacher, student, top_k=8)
            if bool((truncated > dense + 1e-6).any()):
                exceeded = True
                break

        assert exceeded, (
            "no seed produced a truncated value above the dense one; the "
            "fixture can no longer demonstrate the non-monotonicity"
        )

    def test_selects_the_teachers_top_tokens_not_the_students(self):
        """Supervision is restricted to what the *teacher* considers likely."""
        from unturtle.post_training.divergence import teacher_student_divergence

        vocab = 8
        teacher = torch.full((1, 1, vocab), -20.0)
        teacher[0, 0, 3] = 0.0  # teacher mass concentrated on token 3
        teacher = F.log_softmax(teacher, dim=-1)

        # Student disagrees, putting its mass on token 5.
        student = torch.full((1, 1, vocab), -20.0)
        student[0, 0, 5] = 0.0
        student = F.log_softmax(student, dim=-1)

        top1 = teacher_student_divergence(teacher, student, top_k=1)

        # Token 3 alone: p_t ~= 1, log p_t ~= 0, log p_s ~= -20 -> ~20.
        assert top1.item() > 10.0, (
            f"expected the teacher's argmax to dominate, got {top1.item()}"
        )


class TestNonFiniteTeacherLogprobs:
    def test_minus_inf_teacher_entries_contribute_zero_not_nan(self):
        """`p·log p -> 0` as `p -> 0`, but `0 * -inf` is NaN if computed."""
        from unturtle.post_training.divergence import teacher_student_divergence

        vocab = 8
        teacher = torch.full((1, 1, vocab), float("-inf"))
        teacher[0, 0, 0] = 0.0  # a one-hot teacher: every other logprob is -inf
        student = F.log_softmax(torch.randn(1, 1, vocab), dim=-1)

        got = teacher_student_divergence(teacher, student)

        assert torch.isfinite(got).all(), f"got {got}"
        expected = -student[0, 0, 0]  # 1 * (0 - log p_s[0])
        assert torch.allclose(got.flatten(), expected.reshape(1), atol=1e-6)

    def test_top_k_beyond_the_teachers_support_is_finite(self):
        """Reachable in practice: fewer than `k` tokens with non-zero mass."""
        from unturtle.post_training.divergence import teacher_student_divergence

        vocab = 8
        teacher = torch.full((1, 1, vocab), float("-inf"))
        teacher[0, 0, 0] = 0.0
        student = F.log_softmax(torch.randn(1, 1, vocab), dim=-1)

        got = teacher_student_divergence(teacher, student, top_k=4)

        assert torch.isfinite(got).all(), (
            f"top-k reached past the teacher's support and produced {got}"
        )


class TestGradientSafetyOfTheFiniteGuard:
    def test_teacher_gradients_stay_finite_with_minus_inf_entries(self):
        """Why the guard sanitizes before `exp` rather than only masking after.

        A bare `torch.where(finite, log_p.exp() * delta, 0)` suffices for the
        forward value and for the *student* gradient: `exp(-inf)` is exactly
        `0.0`, and a detached teacher has no gradient path through the weight.
        It is not enough when the teacher itself carries gradient (a soft or
        learned teacher, or a caller who forgot to detach).

        The bare form is constructed inline and asserted to fail, so this
        justification stays falsifiable — if PyTorch ever stops producing NaN
        there, the guard's extra step should be deleted, and this test says so.
        """
        from unturtle.post_training.divergence import teacher_student_divergence

        vocab = 6
        raw = torch.full((1, 1, vocab), float("-inf"))
        raw[0, 0, 0] = 0.0
        student = F.log_softmax(torch.randn(1, 1, vocab), dim=-1)

        naive_teacher = raw.clone().requires_grad_(True)
        finite = torch.isfinite(naive_teacher)
        torch.where(
            finite,
            naive_teacher.exp() * (naive_teacher - student),
            torch.zeros_like(naive_teacher),
        ).sum().backward()
        assert bool(torch.isnan(naive_teacher.grad).any()), (
            "the bare form no longer produces NaN teacher gradients, so the "
            "guard's extra step is unjustified — simplify it"
        )

        guarded_teacher = raw.clone().requires_grad_(True)
        teacher_student_divergence(guarded_teacher, student).sum().backward()
        assert torch.isfinite(guarded_teacher.grad).all(), (
            f"guarded teacher gradient was {guarded_teacher.grad}"
        )


class TestReverseKLIsUnguardedLikeTheReference:
    def test_trimmed_teacher_makes_dense_reverse_kl_infinite(self):
        """Documents a reachable hazard, matching upstream rather than diverging.

        `KL(student||teacher)` is genuinely infinite where the student puts
        mass and the teacher assigns none, and the reference computes it with
        no finiteness mask — its comment calls that "the mode-seeking property
        of reverse KL, NOT NaN".  Guarding it here would quietly make this a
        different objective.

        The trigger is not exotic: `rl_sdar.py` applies teacher-side top-k /
        top-p / min-p trimming just before the divergence, each writing -inf
        into the logits pre-`log_softmax`.  This fixture reproduces that.
        """
        from unturtle.post_training.divergence import teacher_student_divergence

        torch.manual_seed(1)
        vocab = 64
        logits = torch.randn(1, 2, vocab)
        cutoff = logits.topk(k=10, dim=-1).values[..., -1:]
        teacher = F.log_softmax(
            logits.masked_fill(logits < cutoff, float("-inf")), dim=-1
        )
        student = F.log_softmax(torch.randn(1, 2, vocab), dim=-1)

        assert torch.isfinite(teacher_student_divergence(teacher, student)).all(), (
            "forward KL must stay finite; only the reverse direction diverges"
        )

        blended = teacher_student_divergence(teacher, student, reverse_kl_weight=0.3)
        assert bool(torch.isinf(blended).any()), (
            "a trimmed teacher plus reverse_kl_weight should be infinite; if "
            "this now passes, a guard was added and the objective no longer "
            "matches the reference"
        )

    def test_a_full_support_teacher_keeps_reverse_kl_finite(self):
        """The ordinary case: nothing about reverse KL is broken per se."""
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()

        got = teacher_student_divergence(teacher, student, reverse_kl_weight=0.5)

        assert torch.isfinite(got).all()


class TestReverseKLAndJSD:
    def test_reverse_kl_weight_blends_the_two_directions(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()
        forward = teacher_student_divergence(teacher, student)
        reverse = (student.exp() * (student - teacher)).sum(dim=-1)

        blended = teacher_student_divergence(teacher, student, reverse_kl_weight=0.25)

        assert torch.allclose(blended, 0.75 * forward + 0.25 * reverse, atol=1e-6)

    def test_reverse_kl_weight_one_is_pure_reverse(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()
        reverse = (student.exp() * (student - teacher)).sum(dim=-1)

        got = teacher_student_divergence(teacher, student, reverse_kl_weight=1.0)

        assert torch.allclose(got, reverse, atol=1e-6)

    def test_jsd_is_symmetric_at_alpha_one_half(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()

        forward = teacher_student_divergence(
            teacher, student, divergence="jsd", jsd_alpha=0.5
        )
        swapped = teacher_student_divergence(
            student, teacher, divergence="jsd", jsd_alpha=0.5
        )

        assert torch.allclose(forward, swapped, atol=1e-6), (
            "JSD at alpha=0.5 must be symmetric in its arguments"
        )

    def test_jsd_is_zero_for_identical_distributions(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, _ = _logprobs()

        got = teacher_student_divergence(
            teacher, teacher.clone(), divergence="jsd", jsd_alpha=0.5
        )

        assert torch.allclose(got, torch.zeros_like(got), atol=1e-6)

    def test_jsd_top_k_at_k_equals_vocab_matches_dense_jsd(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs(vocab=32)

        dense = teacher_student_divergence(teacher, student, divergence="jsd")
        full_k = teacher_student_divergence(
            teacher, student, divergence="jsd", top_k=32
        )

        assert torch.allclose(dense, full_k, atol=1e-6)

    def test_jsd_alpha_weights_the_teacher_side(self):
        """Pins alpha away from 0.5, where two distinct mutants are invisible.

        Every other JSD test here uses alpha=0.5 (or identical distributions),
        and at 0.5 swapping the mixture weights — or the outer weights — is an
        identity transformation.  Mutation-verified: both swaps pass the rest
        of this file.  Correct value 0.143326 vs 0.240114 / 0.240627 swapped.
        """
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()
        alpha = 0.7

        teacher_probs, student_probs = teacher.exp(), student.exp()
        log_mixture = (alpha * teacher_probs + (1.0 - alpha) * student_probs).log()
        expected = alpha * (teacher_probs * (teacher - log_mixture)).sum(dim=-1) + (
            1.0 - alpha
        ) * (student_probs * (student - log_mixture)).sum(dim=-1)

        got = teacher_student_divergence(
            teacher, student, divergence="jsd", jsd_alpha=alpha
        )

        assert torch.allclose(got, expected, atol=1e-6), (
            f"got={got.flatten()[:3]} expected={expected.flatten()[:3]}"
        )

    def test_jsd_is_asymmetric_away_from_alpha_one_half(self):
        """The symmetry test above only establishes the degenerate case."""
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()

        forward = teacher_student_divergence(
            teacher, student, divergence="jsd", jsd_alpha=0.7
        )
        swapped = teacher_student_divergence(
            student, teacher, divergence="jsd", jsd_alpha=0.7
        )

        assert not torch.allclose(forward, swapped, atol=1e-4), (
            "JSD at alpha != 0.5 must not be symmetric; if it is, alpha is "
            "not reaching the mixture"
        )

    def test_unknown_divergence_is_rejected(self):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()

        with pytest.raises(ValueError, match="divergence"):
            teacher_student_divergence(teacher, student, divergence="wasserstein")


class TestItStaysSparse:
    def test_top_k_never_materializes_a_dense_vocab_tensor(self):
        """#64 acceptance criterion: no dense teacher probability tensor.

        Equivalence alone cannot show the sparse path was taken; this counts
        `[B, L, V]`-shaped allocations directly.  The dense arm's assertion is
        a self-check that the probe can detect anything at all.
        """
        from torch.overrides import TorchFunctionMode

        from unturtle.post_training.divergence import teacher_student_divergence

        B, L, V = 2, 3, 64
        teacher, student = _logprobs(batch=B, length=L, vocab=V)

        class CountFullVocabTensors(TorchFunctionMode):
            def __init__(self):
                self.seen = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                if isinstance(result, torch.Tensor) and tuple(result.shape) == (
                    B,
                    L,
                    V,
                ):
                    self.seen += 1
                return result

        sparse_probe = CountFullVocabTensors()
        with sparse_probe:
            teacher_student_divergence(teacher, student, top_k=8)

        dense_probe = CountFullVocabTensors()
        with dense_probe:
            teacher_student_divergence(teacher, student)

        assert dense_probe.seen > 0, (
            "the dense path allocated no [B, L, V] tensor, so this probe "
            "cannot distinguish the two paths"
        )
        assert sparse_probe.seen == 0, (
            f"top-k allocated {sparse_probe.seen} dense [B, L, V] tensors"
        )


class TestGradients:
    def test_gradient_flows_to_the_student_only(self):
        """The teacher is supervision, not a parameter to optimize."""
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs()
        teacher = teacher.requires_grad_(True)
        student = student.requires_grad_(True)

        teacher_student_divergence(teacher, student).sum().backward()

        assert student.grad is not None and bool(student.grad.abs().sum() > 0)

    @pytest.mark.parametrize("top_k", [None, 8])
    def test_gradients_are_finite(self, top_k):
        from unturtle.post_training.divergence import teacher_student_divergence

        teacher, student = _logprobs(vocab=32)
        student = student.requires_grad_(True)

        teacher_student_divergence(teacher, student, top_k=top_k).sum().backward()

        assert torch.isfinite(student.grad).all()
