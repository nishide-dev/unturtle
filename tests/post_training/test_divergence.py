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
