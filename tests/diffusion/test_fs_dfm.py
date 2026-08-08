"""
FS-DFM step-aware objective components (#65 Phase B).

Three pieces from the paper's §4, each tested by its fixed point or a
behavioural invariant per the #65 test rule — never by comparing against a
second transcription of the same equation (that failure shipped twice on
eq. (3.8) alone, #94/#97):

- the **step-aware path loss**: eq. (3.8) with the Cumulative Scalar
  ``gbar_{t,h}`` (eq. 4.3) replacing the instantaneous ``g(t)``, calibrating
  update strength for the finite step actually taken;
- the **RK-4 shortcut teacher** (Algorithm 1): three half-step CTMC jumps
  advancing the *state*, logits averaged with weights (1, 2, 2, 1)/6;
- the **distillation loss** (§4.3): per-position ``KL(p_tea || p_theta)``
  with a stopped-gradient teacher, blended with the path loss by
  ``m = 1[h < tau]`` (eq. 4.5).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from unturtle.processes.discrete_flow import LinearKappa


class _CosineKappa:
    """A non-linear path, so linear-kappa shortcuts cannot pass by accident."""

    def kappa(self, t):
        return 1.0 - torch.cos(t * math.pi / 2)


class TestStepAwareWeight:
    """`step_size` swaps g(t) for gbar_{t,h} in the same bracket."""

    def test_the_minimizer_is_still_the_clean_posterior(self):
        """gbar is a common positive factor over the whole bracket, exactly
        like g — so the optimum must stay `p*`, independent of t AND h.

        This is the TestTheOptimum property that would have caught the
        misplaced-g bug (#97); it is the first thing any reweighting of this
        loss has to preserve.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        vocab = 6
        target = torch.tensor([0.40, 0.25, 0.20, 0.10, 0.05, 0.0])
        generator = torch.Generator().manual_seed(0)
        x_1 = torch.multinomial(target.expand(4000, vocab), 1, generator=generator)
        x_1 = x_1.reshape(1, 4000)
        x_t = torch.full((1, 4000), 5)

        for t, h in ((0.1, 0.5), (0.5, 0.25), (0.9, 0.05)):
            logits = torch.zeros(1, 1, vocab, requires_grad=True)
            optimizer = torch.optim.Adam([logits], lr=0.05)
            for _ in range(900):
                loss = discrete_flow_matching_loss(
                    logits.expand(1, 4000, vocab),
                    x_1,
                    x_t,
                    torch.tensor([t]),
                    scheduler=LinearKappa(),
                    step_size=h,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            fitted = F.softmax(logits.detach(), dim=-1)[0, 0]
            distance = 0.5 * float((fitted - target).abs().sum())
            assert distance < 0.05, (
                f"at (t={t}, h={h}) the step-aware loss is minimized "
                f"{distance:.4f} away from the data distribution; the "
                "reweighting moved the fixed point"
            )

    def test_the_weight_approaches_g_as_h_shrinks(self):
        """gbar integrates g over [t, t+h]; the h->0 limit must recover g(t).

        Pinned through the loss itself (ratio of losses at shrinking h to the
        h-free loss), not through a transcribed gbar formula.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        torch.manual_seed(0)
        logits = torch.randn(1, 8, 6)
        x_1 = torch.randint(0, 6, (1, 8))
        x_t = torch.randint(0, 6, (1, 8))
        t = torch.tensor([0.4])

        instantaneous = float(
            discrete_flow_matching_loss(logits, x_1, x_t, t, scheduler=LinearKappa())
        )
        errors = [
            abs(
                float(
                    discrete_flow_matching_loss(
                        logits, x_1, x_t, t, scheduler=LinearKappa(), step_size=h
                    )
                )
                - instantaneous
            )
            for h in (0.2, 0.02, 0.002)
        ]

        assert errors == sorted(errors, reverse=True), (
            f"loss did not converge to the h-free loss as h shrank: {errors}"
        )
        assert errors[-1] < abs(instantaneous) * 0.01

    def test_the_weight_matches_the_solver_closed_form_under_linear_kappa(self):
        """One path, one scalar: the loss's gbar and the solver's must agree.

        The solver hardcodes the linear-kappa closed form
        (1/h)ln((1-t)/(1-t-h)); the loss derives gbar from the scheduler so
        non-linear paths work.  If the two drift, training calibrates for a
        different step than sampling takes.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
        from unturtle.models.generation.dfm_solver import cumulative_scalar

        torch.manual_seed(1)
        logits = torch.randn(1, 4, 6)
        x_1 = torch.randint(0, 6, (1, 4))
        x_t = torch.randint(0, 6, (1, 4))
        t, h = torch.tensor([0.3]), 0.25

        with_h = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=LinearKappa(), step_size=h, reduction="none"
        )
        base = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=LinearKappa(), reduction="none"
        )

        ratio = float((with_h / base).mean())
        expected = float(cumulative_scalar(t, h)) / (1.0 / (1.0 - 0.3))
        assert math.isclose(ratio, expected, rel_tol=1e-4), (
            f"loss gbar/g ratio {ratio:.6f} vs solver's {expected:.6f}; the "
            "objective and the sampler disagree on the same path"
        )

    def test_a_nonlinear_kappa_changes_the_weight(self):
        """The general eq. (4.3) form, not the linear shortcut.

        Under linear kappa `gbar(t,h)` has one value; under a cosine path it
        must differ.  An implementation that silently assumed kappa(t)=t
        would pass every linear-kappa test in this file.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        torch.manual_seed(2)
        logits = torch.randn(1, 4, 6)
        x_1 = torch.randint(0, 6, (1, 4))
        x_t = torch.randint(0, 6, (1, 4))
        t, h = torch.tensor([0.3]), 0.25

        linear = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=LinearKappa(), step_size=h, reduction="none"
        )
        cosine = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_CosineKappa(), step_size=h, reduction="none"
        )

        assert not torch.allclose(linear, cosine), (
            "linear and cosine paths produced identical step-aware losses; "
            "the weight is not reading the scheduler"
        )

    def test_bf16_logits_do_not_collapse_the_weight(self):
        """The #94-era bf16 finite-difference collapse, re-checked for gbar.

        `kappa(t + h)` at bf16 precision can equal `kappa(t)` for small h,
        zeroing the log ratio.  The scheduler math must run in fp32.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        torch.manual_seed(3)
        logits = torch.randn(1, 4, 6)
        x_1 = torch.randint(0, 6, (1, 4))
        x_t = torch.randint(0, 6, (1, 4))
        t, h = torch.tensor([0.4]), 1e-3

        fp32 = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=LinearKappa(), step_size=h
        )
        bf16 = discrete_flow_matching_loss(
            logits.bfloat16(), x_1, x_t, t, scheduler=LinearKappa(), step_size=h
        )

        assert float(bf16) != 0.0
        assert math.isclose(float(bf16), float(fp32), rel_tol=0.05), (
            f"bf16 step-aware loss {float(bf16):.4f} vs fp32 "
            f"{float(fp32):.4f}; the gbar computation collapsed in low "
            "precision"
        )

    def test_a_step_past_the_end_of_the_path_is_rejected(self):
        """kappa(t + h) with t + h > 1 is off the path; ln of it is garbage."""
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        with pytest.raises(ValueError, match="step"):
            discrete_flow_matching_loss(
                torch.randn(1, 2, 6),
                torch.randint(0, 6, (1, 2)),
                torch.randint(0, 6, (1, 2)),
                torch.tensor([0.9]),
                scheduler=LinearKappa(),
                step_size=0.2,
            )


class TestDistillationLoss:
    def test_the_minimizer_is_the_teacher_distribution(self):
        """KL(p_tea || p_theta) is minimized exactly at p_theta = p_tea."""
        from unturtle.diffusion.fs_dfm import few_step_distillation_loss

        torch.manual_seed(0)
        teacher_logits = torch.randn(1, 6, 8)
        student = torch.zeros(1, 6, 8, requires_grad=True)
        optimizer = torch.optim.Adam([student], lr=0.1)

        for _ in range(600):
            loss = few_step_distillation_loss(student, teacher_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fitted = F.softmax(student.detach(), dim=-1)
        target = F.softmax(teacher_logits, dim=-1)
        assert torch.allclose(fitted, target, atol=1e-3), (
            f"max deviation {float((fitted - target).abs().max()):.4f}; the "
            "distillation loss does not converge onto the teacher"
        )

    def test_it_is_non_negative(self):
        from unturtle.diffusion.fs_dfm import few_step_distillation_loss

        torch.manual_seed(1)
        worst = min(
            float(
                few_step_distillation_loss(torch.randn(1, 4, 8), torch.randn(1, 4, 8))
            )
            for _ in range(200)
        )
        assert worst >= -1e-6, f"KL went negative ({worst:.4g})"

    def test_the_teacher_gradient_is_stopped(self):
        """§4.3: stop-grad on the teacher logits.

        A teacher that receives gradient drifts toward the student, and the
        consistency target collapses.
        """
        from unturtle.diffusion.fs_dfm import few_step_distillation_loss

        student = torch.randn(1, 4, 8, requires_grad=True)
        teacher = torch.randn(1, 4, 8, requires_grad=True)

        few_step_distillation_loss(student, teacher).backward()

        assert student.grad is not None
        assert teacher.grad is None or torch.all(teacher.grad == 0), (
            "gradient reached the teacher logits"
        )

    def test_the_kl_direction_is_forward(self):
        """`KL(p_tea || p_theta)`, not the reverse — pinned by the gradient.

        Both directions share the same *minimizer* over an unconstrained
        student (the teacher itself), so no convergence test can tell them
        apart; they differ under limited capacity (mode-covering vs
        mode-seeking), which is exactly what the paper's choice buys.  The
        defining property that survives unconstrained fixtures is the update
        direction: forward KL over logits has
        ``dL/dlogits = (p_theta - p_tea) / N`` exactly.
        """
        from unturtle.diffusion.fs_dfm import few_step_distillation_loss

        torch.manual_seed(3)
        student = torch.randn(1, 4, 8, requires_grad=True)
        teacher = torch.randn(1, 4, 8)

        few_step_distillation_loss(student, teacher).backward()

        expected = (
            F.softmax(student.detach(), dim=-1) - F.softmax(teacher, dim=-1)
        ) / 4
        assert torch.allclose(student.grad, expected, atol=1e-6), (
            f"max gradient deviation "
            f"{float((student.grad - expected).abs().max()):.3e}; the KL "
            "direction (or its normalization) is not the forward "
            "KL(teacher || student)"
        )

    def test_padding_is_excluded(self):
        from unturtle.diffusion.fs_dfm import few_step_distillation_loss

        torch.manual_seed(2)
        student = torch.randn(1, 4, 8)
        teacher = torch.randn(1, 4, 8)
        mask = torch.tensor([[True, True, False, False]])

        masked = float(few_step_distillation_loss(student, teacher, loss_mask=mask))
        first_two = float(few_step_distillation_loss(student[:, :2], teacher[:, :2]))

        assert math.isclose(masked, first_two, rel_tol=1e-6), (
            "masked loss does not equal the loss over kept positions alone"
        )


class TestRKTeacher:
    """Algorithm 1: three half-step jumps, logits averaged (1, 2, 2, 1)/6."""

    def test_the_state_actually_advances_between_evaluations(self):
        """The defining choice of Algorithm 1, and the easiest one to fake.

        Evaluating theta'(x_t, t_k) four times with the SAME x_t also
        produces four logits and a plausible average — but then the teacher
        never sees the intermediate states the fine-grained trajectory visits,
        and the 'integration' integrates nothing.  Pinned by recording what
        the denoiser receives: from an all-mask start with a confident
        denoiser, later evaluations must see unmasked (jumped) states.
        """
        from unturtle.diffusion.fs_dfm import rk_teacher_logits

        MASK = 7
        seen = []

        def denoiser(x_t, t, h):
            seen.append(x_t.clone())
            logits = torch.zeros(*x_t.shape, 8)
            logits[..., 3] = 12.0  # confidently token 3
            return logits

        x_t = torch.full((2, 10), MASK, dtype=torch.long)
        rk_teacher_logits(
            denoiser,
            x_t,
            torch.tensor([0.1, 0.1]),
            0.4,
            generator=torch.Generator().manual_seed(0),
        )

        assert len(seen) == 4, f"expected 4 evaluations, saw {len(seen)}"
        assert torch.equal(seen[0], x_t), "first evaluation must be at x_t"
        moved = [float((s != MASK).float().mean()) for s in seen]
        assert moved[1] > 0.0 and moved[2] >= moved[1], (
            f"mask fraction never fell across evaluations ({moved}); the "
            "state is not being advanced by the jump process"
        )

    def test_the_average_uses_rk4_weights(self):
        """(1, 2, 2, 1)/6 — pinned by a denoiser whose logits encode which
        evaluation is which, so any other weighting produces a different
        number."""
        from unturtle.diffusion.fs_dfm import rk_teacher_logits

        calls = [0]

        def denoiser(x_t, t, h):
            calls[0] += 1
            return torch.full((*x_t.shape, 8), float(10 ** (calls[0] - 1)))

        out = rk_teacher_logits(
            denoiser,
            torch.zeros(1, 4, dtype=torch.long),
            torch.tensor([0.1]),
            0.4,
            generator=torch.Generator().manual_seed(0),
        )

        expected = (1 * 1 + 2 * 10 + 2 * 100 + 1 * 1000) / 6
        assert torch.allclose(out, torch.full_like(out, expected)), (
            f"got {float(out.reshape(-1)[0]):.3f}, expected {expected:.3f} "
            "under weights (1,2,2,1)/6"
        )

    def test_evaluation_times_and_step_conditioning_follow_algorithm_1(self):
        """Times (t, t+h/2, t+h/2, t+h); every evaluation conditioned on h/2."""
        from unturtle.diffusion.fs_dfm import rk_teacher_logits

        seen = []

        def denoiser(x_t, t, h):
            seen.append((float(t.reshape(-1)[0]), float(h)))
            return torch.zeros(*x_t.shape, 8)

        rk_teacher_logits(
            denoiser,
            torch.zeros(1, 4, dtype=torch.long),
            torch.tensor([0.2]),
            0.4,
            generator=torch.Generator().manual_seed(0),
        )

        times = [t for t, _ in seen]
        step_conditioning = [h for _, h in seen]
        assert times == pytest.approx([0.2, 0.4, 0.4, 0.6]), times
        assert step_conditioning == pytest.approx([0.2, 0.2, 0.2, 0.2]), (
            "Algorithm 1 conditions every teacher evaluation on h' = h/2"
        )

    def test_jump_rate_times_follow_the_rk4_convention(self):
        """The three jumps draw their rates at (t, t_mid, t_mid).

        Jumps 2 and 3 both rate at the midpoint — the RK4 convention, where
        k2 and k3 are both midpoint evaluations — even though the interval
        bookkeeping makes jump 3 look like it "should" rate at t + h.  Review
        measured the alternative reading is a material change (gbar 2.03 vs
        3.47 at t=0.3, h=0.4), and nothing observed the rate times before:
        they are internal to the jump closure, invisible to the
        denoiser-argument test above.  Recorded here so a future reader does
        not "fix" the convention away.
        """
        import unturtle.diffusion.fs_dfm as fs

        rate_times = []
        real = fs.cumulative_scalar

        def spy(at, h):
            rate_times.append(float(at.reshape(-1)[0]))
            return real(at, h)

        original = fs.cumulative_scalar
        fs.cumulative_scalar = spy
        try:
            fs.rk_teacher_logits(
                lambda x, t, h: torch.zeros(*x.shape, 8),
                torch.zeros(1, 4, dtype=torch.long),
                torch.tensor([0.2]),
                0.4,
                generator=torch.Generator().manual_seed(0),
            )
        finally:
            fs.cumulative_scalar = original

        assert rate_times == pytest.approx([0.2, 0.4, 0.4]), (
            f"jump rate times {rate_times}; Algorithm 1 rates the three jumps "
            "at (t, t_mid, t_mid)"
        )

    def test_all_three_jumps_actually_run(self):
        """Dropping the third jump survives the advancement test above.

        That test only compares evaluations 1 and 2 — `seen[3]` was never
        inspected, so `x_3 = x_2` passed everything.  With a confident
        denoiser from an all-mask start, the mask fraction must keep falling
        through the fourth evaluation.
        """
        from unturtle.diffusion.fs_dfm import rk_teacher_logits

        MASK = 7
        seen = []

        def denoiser(x_t, t, h):
            seen.append(x_t.clone())
            logits = torch.zeros(*x_t.shape, 8)
            logits[..., 3] = 12.0
            return logits

        rk_teacher_logits(
            denoiser,
            torch.full((4, 32), MASK, dtype=torch.long),
            torch.tensor([0.1] * 4),
            0.6,
            generator=torch.Generator().manual_seed(0),
        )

        moved = [float((s != MASK).float().mean()) for s in seen]
        assert moved[3] > moved[2], (
            f"mask fraction across evaluations: {moved}; the third jump did "
            "not advance the state, so the final evaluation saw a stale x_2"
        )

    def test_a_perfect_denoiser_is_a_fixed_point(self):
        """A state-and-time-independent denoiser must round-trip unchanged.

        All four evaluations return the same logits, so any convex weighting
        returns them too — if this fails, the teacher is transforming logits
        rather than averaging them.
        """
        from unturtle.diffusion.fs_dfm import rk_teacher_logits

        fixed = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1]))

        def denoiser(x_t, t, h):
            return fixed.expand(*x_t.shape, 4).clone()

        out = rk_teacher_logits(
            denoiser,
            torch.zeros(1, 6, dtype=torch.long),
            torch.tensor([0.3]),
            0.2,
            generator=torch.Generator().manual_seed(0),
        )

        assert torch.allclose(out, fixed.expand(1, 6, 4), atol=1e-6)

    def test_a_step_past_the_end_of_the_path_is_rejected(self):
        from unturtle.diffusion.fs_dfm import rk_teacher_logits

        with pytest.raises(ValueError, match="step"):
            rk_teacher_logits(
                lambda x, t, h: torch.zeros(*x.shape, 8),
                torch.zeros(1, 4, dtype=torch.long),
                torch.tensor([0.9]),
                0.2,
            )


class TestEndOfPathBoundaries:
    """t + h must stay strictly inside the path, checked on the time itself.

    Two measured escape routes closed here: a scheduler clamping kappa just
    below 1 keeps the kappa-based check silent for a far-off-path step (gbar
    27.3 from t=0.9, h=0.5), and fp32 rounding let t=0.9, h=0.1 through the
    teacher with gbar = 306 despite t + h == 1 exactly.
    """

    def test_the_loss_rejects_a_step_reaching_exactly_one(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        with pytest.raises(ValueError, match="step"):
            discrete_flow_matching_loss(
                torch.randn(1, 2, 6),
                torch.randint(0, 6, (1, 2)),
                torch.randint(0, 6, (1, 2)),
                torch.tensor([0.9]),
                scheduler=LinearKappa(),
                step_size=0.1,
            )

    def test_the_loss_rejects_a_clamping_scheduler_off_path(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        class _Clamping:
            def kappa(self, t):
                return torch.clamp(t, max=1.0 - 1e-7)

        with pytest.raises(ValueError, match="step"):
            discrete_flow_matching_loss(
                torch.randn(1, 2, 6),
                torch.randint(0, 6, (1, 2)),
                torch.randint(0, 6, (1, 2)),
                torch.tensor([0.9]),
                scheduler=_Clamping(),
                step_size=0.5,
            )

    def test_the_teacher_rejects_a_step_reaching_exactly_one(self):
        from unturtle.diffusion.fs_dfm import rk_teacher_logits

        with pytest.raises(ValueError, match="step"):
            rk_teacher_logits(
                lambda x, t, h: torch.zeros(*x.shape, 8),
                torch.zeros(1, 4, dtype=torch.long),
                torch.tensor([0.9]),
                0.1,
            )


class TestBlendedLoss:
    """eq. (4.5): m_b = 1[h_b < tau] selects the branch per batch row."""

    def test_small_steps_take_the_path_loss_and_large_steps_distill(self):
        from unturtle.diffusion.fs_dfm import blend_losses

        path = torch.tensor([1.0, 1.0, 1.0])
        distill = torch.tensor([100.0, 100.0, 100.0])
        step_sizes = torch.tensor([2.0**-10, 2.0**-9, 0.5])

        blended = blend_losses(path, distill, step_sizes=step_sizes, tau=2.0**-9)

        # Row 0: h < tau -> path (1.0).  Rows 1, 2: h >= tau -> distill (100).
        assert float(blended) == pytest.approx((1.0 + 100.0 + 100.0) / 3)

    def test_the_comparison_direction_is_small_steps_to_path(self):
        """`m = 1[h < tau]`, not 1[h > tau] — pinned by an asymmetric fixture.

        The two tests below are accidentally symmetric: one row at h == tau
        (both directions False), and a three-row fixture where either
        direction selects exactly one row of identical value.  Review measured
        a reversed comparison surviving both.  Distinct per-row values break
        the symmetry: correct gives (100 + 2)/2 = 51 only under reversal.
        """
        from unturtle.diffusion.fs_dfm import blend_losses

        path = torch.tensor([1.0, 2.0])
        distill = torch.tensor([100.0, 200.0])
        step_sizes = torch.tensor([2.0**-10, 0.5])

        blended = blend_losses(path, distill, step_sizes=step_sizes, tau=2.0**-9)

        # Row 0 (h < tau) -> path 1.0; row 1 -> distill 200.0.
        assert float(blended) == pytest.approx((1.0 + 200.0) / 2)

    def test_the_threshold_is_strict(self):
        """m = 1[h < tau], not <=: h == tau distills (§5.1 pairs tau = 2^-9
        with the h grid {2^-10..2^0}, so only h = 2^-10 takes the path loss)."""
        from unturtle.diffusion.fs_dfm import blend_losses

        path = torch.tensor([1.0])
        distill = torch.tensor([100.0])

        at_tau = blend_losses(
            path, distill, step_sizes=torch.tensor([2.0**-9]), tau=2.0**-9
        )

        assert float(at_tau) == pytest.approx(100.0)


class TestStepAwareWrapper:
    """App. C.1 adapted for time-agnostic backbones (an Unturtle choice).

    The paper fuses ``c = SiLU(W [phi_time(t); phi_dt(h)])`` inside its own
    architecture; Unturtle's masked-diffusion backbones are time-agnostic
    (the mask count carries the corruption level), so the wrapper adds the
    fused conditioning to the token embeddings and forwards through
    ``inputs_embeds`` — no backbone surgery.  What matters behaviourally is
    that the model can *tell budgets apart*, which is the whole premise of
    step-aware training.
    """

    @staticmethod
    def _base():
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )

        torch.manual_seed(0)
        return TinyA2DLlamaLMHeadModel(
            TinyA2DLlamaConfig(
                vocab_size=9,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=16,
            )
        )

    def test_the_step_size_is_visible_to_the_model(self):
        """Different h, same everything else -> different logits.

        Without this the student cannot be conditioned on its budget and
        "step-aware" degenerates to ordinary DFM.
        """
        from unturtle.diffusion.fs_dfm import StepAwareWrapper

        model = StepAwareWrapper(self._base()).eval()
        ids = torch.randint(0, 9, (2, 8))
        t = torch.tensor([0.3, 0.3])

        with torch.no_grad():
            small = model(ids, t, 2.0**-6)
            large = model(ids, t, 1.0)

        assert not torch.allclose(small, large), (
            "h = 2^-6 and h = 1 produced identical logits; the step size is "
            "not reaching the model"
        )

    def test_the_time_is_visible_to_the_model(self):
        from unturtle.diffusion.fs_dfm import StepAwareWrapper

        model = StepAwareWrapper(self._base()).eval()
        ids = torch.randint(0, 9, (2, 8))

        with torch.no_grad():
            early = model(ids, torch.tensor([0.1, 0.1]), 0.25)
            late = model(ids, torch.tensor([0.9, 0.9]), 0.25)

        assert not torch.allclose(early, late)

    def test_gradients_reach_both_the_base_and_the_fusion(self):
        """A wrapper that detaches either half silently freezes it."""
        from unturtle.diffusion.fs_dfm import StepAwareWrapper

        model = StepAwareWrapper(self._base()).train()
        ids = torch.randint(0, 9, (2, 8))

        model(ids, torch.tensor([0.3, 0.3]), 0.25).sum().backward()

        fuse_grad = model.fuse.weight.grad
        base_grad = model.base.model.embed_tokens.weight.grad
        assert fuse_grad is not None and float(fuse_grad.abs().sum()) > 0
        assert base_grad is not None and float(base_grad.abs().sum()) > 0

    def test_a_bf16_cast_wrapper_still_forwards(self):
        """Half precision is the realistic training dtype, and the first
        draft crashed on it: `torch.linspace` takes the default dtype, so the
        sinusoidal features stayed fp32 against a bf16 `fuse` and the matmul
        raised at the first forward.  Same dtype discipline as dfm_loss's
        fp32 scheduler-math rule, from the model side.
        """
        from unturtle.diffusion.fs_dfm import StepAwareWrapper

        model = StepAwareWrapper(self._base()).bfloat16().eval()
        ids = torch.randint(0, 9, (2, 8))

        with torch.no_grad():
            out = model(ids, torch.tensor([0.3, 0.3]), 0.25)

        assert out.dtype == torch.bfloat16
        assert bool(torch.isfinite(out.float()).all())

    def test_zero_fusion_reduces_to_the_plain_base_forward(self):
        """With `fuse` fully zeroed, the wrapper is the identity around the
        base — SiLU(0) = 0, so the conditioning vanishes exactly.

        NOTE: this alone cannot pin the SiLU (a linear fusion also maps 0 to
        0); the saturation test below does that.
        """
        from unturtle.diffusion.fs_dfm import StepAwareWrapper

        model = StepAwareWrapper(self._base()).eval()
        with torch.no_grad():
            model.fuse.weight.zero_()
            model.fuse.bias.zero_()
        ids = torch.randint(0, 9, (2, 8))

        with torch.no_grad():
            wrapped = model(ids, torch.tensor([0.3, 0.3]), 0.25)
            plain = model.base(input_ids=ids).logits

        assert torch.equal(wrapped, plain)

    def test_the_nonlinearity_saturates_negative_preactivations(self):
        """SiLU's defining behaviour, and the only one that separates it from
        a linear fusion at this call site.

        `silu(-20) = -20·sigmoid(-20) ≈ -4e-8` — a strongly negative
        pre-activation is *damped to nothing*, while a linear fusion passes
        -20 straight into every embedding and wrecks the logits.  The
        zero-fusion test above is blind to this (both map 0 to 0), which is
        exactly how a dropped SiLU survived the first battery: review
        measured it changes the fused vector by 54% under random weights, and
        nothing pinned it.
        """
        from unturtle.diffusion.fs_dfm import StepAwareWrapper

        model = StepAwareWrapper(self._base()).eval()
        with torch.no_grad():
            model.fuse.weight.zero_()
            model.fuse.bias.fill_(-20.0)
        ids = torch.randint(0, 9, (2, 8))

        with torch.no_grad():
            wrapped = model(ids, torch.tensor([0.3, 0.3]), 0.25)
            plain = model.base(input_ids=ids).logits

        assert torch.allclose(wrapped, plain, atol=1e-4), (
            "a -20 pre-activation reached the embeddings almost undamped; "
            "the fusion nonlinearity is missing"
        )

    def test_the_feature_layout_puts_time_first_and_step_second(self):
        """Columns [0, F) carry t and [F, 2F) carry h — pinned structurally.

        A swapped concatenation is invisible to any two-forward comparison:
        the mutant's output at (t=a, h=b) equals the correct output at
        (t=b, h=a), so both sides of an asymmetry assertion move together and
        it passes (measured: the swap survived exactly such a test).  The
        distinguishing observation is parameter-surgical: zeroing the fuse
        columns for the h-block must make the output h-invariant while t
        stays visible.  Under a swap, the same surgery silences t instead.
        """
        from unturtle.diffusion.fs_dfm import StepAwareWrapper

        model = StepAwareWrapper(self._base()).eval()
        feature_dim = model._feature_dim
        with torch.no_grad():
            model.fuse.weight[:, feature_dim:].zero_()  # silence the h block
        ids = torch.randint(0, 9, (2, 8))
        t = torch.tensor([0.3, 0.3])

        with torch.no_grad():
            h_small = model(ids, t, 2.0**-6)
            h_large = model(ids, t, 1.0)
            t_early = model(ids, torch.tensor([0.1, 0.1]), 0.25)
            t_late = model(ids, torch.tensor([0.9, 0.9]), 0.25)

        assert torch.allclose(h_small, h_large), (
            "h still reaches the output through the silenced h-columns; the "
            "feature layout is not [time; step]"
        )
        assert not torch.allclose(t_early, t_late), (
            "t vanished when the h-columns were silenced; the two feature "
            "blocks are swapped"
        )

    def test_the_conditioning_is_broadcast_to_every_position(self):
        """The base must receive `embed(ids) + c` at *every* position.

        Logit-level checks cannot see this: bidirectional attention
        propagates a single-position injection to all outputs, so "the last
        position moved" passes even when only position 0 was conditioned
        (measured — that mutant survived exactly such a test).  Observed at
        the wrapper's output boundary instead: capture the `inputs_embeds`
        the base is handed and assert the added vector is identical at every
        position and nonzero.
        """
        from unturtle.diffusion.fs_dfm import StepAwareWrapper

        model = StepAwareWrapper(self._base()).eval()
        ids = torch.randint(0, 9, (2, 8))
        captured = {}
        real_forward = model.base.forward

        def capture(*args, **kwargs):
            captured["inputs_embeds"] = kwargs["inputs_embeds"].detach().clone()
            return real_forward(*args, **kwargs)

        model.base.forward = capture
        with torch.no_grad():
            model(ids, torch.tensor([0.3, 0.3]), 0.25)

        delta = captured["inputs_embeds"] - model.base.model.embed_tokens(ids)
        per_position_spread = float((delta - delta[:, :1]).abs().max())
        assert per_position_spread < 1e-6, (
            "the conditioning vector differs across positions; it must be one "
            "broadcast vector per row"
        )
        assert float(delta.abs().max()) > 0, "no conditioning reached the base at all"

    def test_it_serves_as_a_solver_denoiser(self):
        """The (x_t, t, h) -> logits contract, end to end through the solver."""
        from unturtle.diffusion.fs_dfm import StepAwareWrapper
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        model = StepAwareWrapper(self._base()).eval()

        def denoise(x_t, t, h):
            with torch.no_grad():
                return model(x_t, t, h)

        out = solve_discrete_flow(
            denoise,
            torch.full((2, 8), 8, dtype=torch.long),
            steps=4,
            generator=torch.Generator().manual_seed(0),
        )

        assert out.shape == (2, 8)


class TestClipStepToPath:
    """The end-of-path adaptation the paper leaves unspecified.

    Training samples ``h`` up to 1, but eq. (4.3) and the teacher's jumps
    need ``t + h < 1`` strictly — at ``h = 1`` no valid ``t`` exists at all.
    The sampler absorbs the endpoint in its unconditional terminal draw; in
    training, Unturtle's choice (recorded as such) is to rescale ``t`` into
    the room the step leaves and clip the *integration* width, while the
    model keeps seeing the nominal ``h`` it will be conditioned on at
    inference.
    """

    def test_the_clipped_pair_always_stays_inside_the_path(self):
        from unturtle.diffusion.fs_dfm import clip_step_to_path

        generator = torch.Generator().manual_seed(0)
        for _ in range(200):
            t = torch.rand(16, generator=generator)
            h = float(2.0 ** -float(torch.randint(0, 11, (1,), generator=generator)))

            scaled_t, h_eff = clip_step_to_path(t, h)

            assert h_eff > 0
            assert bool((scaled_t + h_eff < 1.0).all()), (
                f"t + h_eff reached the path end at h={h}"
            )

    def test_a_step_that_fits_is_not_shrunk(self):
        """Clipping must be inert away from the boundary — otherwise every
        small-h path-loss batch trains on a silently different h."""
        from unturtle.diffusion.fs_dfm import clip_step_to_path

        t = torch.tensor([0.1, 0.4])
        _, h_eff = clip_step_to_path(t, 2.0**-6)

        assert h_eff == pytest.approx(2.0**-6)

    def test_h_equal_one_yields_near_zero_times(self):
        """The h = 1 row: no valid t exists, so t collapses toward 0 and the
        integration width fills (almost) the whole path."""
        from unturtle.diffusion.fs_dfm import clip_step_to_path

        scaled_t, h_eff = clip_step_to_path(torch.rand(8), 1.0)

        assert float(scaled_t.max()) < 0.01
        assert 0.9 < h_eff < 1.0

    def test_time_ordering_is_preserved(self):
        """Rescaling must not reorder the batch's times: the loss weights and
        the process state were drawn against these draws."""
        from unturtle.diffusion.fs_dfm import clip_step_to_path

        t = torch.tensor([0.9, 0.1, 0.5])
        scaled_t, _ = clip_step_to_path(t, 0.5)

        assert torch.equal(scaled_t.argsort(), t.argsort())
