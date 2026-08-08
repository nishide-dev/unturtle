"""
Discrete flow-matching objective (#65 Phase A, FS-DFM eq. 3.8).

The paper derives the loss as a **Bregman divergence**, not cross-entropy::

    L_i = -g(t)*[ p_{1|t}(x_t^i | x_t) - delta_{x_1^i}(x_t^i)
                  + ( 1 - delta_{x_1^i}(x_t^i) ) * log p_{1|t}(x_1^i | x_t) ]

Three properties would be lost by reaching for CE, and each is pinned below:

- `g(t) = kappa'(t)/(1 - kappa(t))` is not decoration.  The paper says it
  "naturally arises from the velocity formulation and ensures proper weighting
  across different time steps"; dropping it trains a differently-weighted
  objective that still converges.
- The `delta` terms make already-correct positions behave differently: where
  `x_t == x_1` the log-likelihood term vanishes entirely and only the first
  term acts.  CE would keep supervising them.
- The first term evaluates the probability of the token *currently held*, not
  of the target, and it is a probability rather than a log-probability.

Reimplemented from the paper; the Apple repository was not read (see #65).
"""

import math

import pytest
import torch
import torch.nn.functional as F


class _Linear:
    """kappa(t) = t, the paper's choice; g(t) = 1/(1-t)."""

    def kappa(self, t):
        return t

    def g(self, t):
        return 1.0 / (1.0 - t)


def _batch(batch=2, length=4, vocab=6, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(batch, length, vocab)
    x_1 = torch.randint(0, vocab, (batch, length))
    x_t = torch.randint(0, vocab, (batch, length))
    t = torch.full((batch,), 0.5)
    return logits, x_1, x_t, t


def _reference(logits, x_1, x_t, t, scheduler=_Linear()):
    """Equation (3.8), written out position by position.

    Note `g` scales the **whole bracket**, both terms:

        -g(t) * [ p(x_t) - delta + (1 - delta) * log p(x_1) ]

    An earlier version of this reference applied `g` to the first term only,
    matching an implementation that did the same — so the two agreed on being
    wrong and every test here passed while the objective's minimizer was the
    interpolant `p_t` rather than the clean posterior `p_{1|t}` (#97).  That is
    the standing hazard with a reference transcribed from the same line as the
    code, and the reason `TestTheOptimum` below asserts what the loss
    *converges to* rather than what it evaluates to.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    batch, length, _ = logits.shape

    total = torch.zeros(batch, length)
    for b in range(batch):
        g = scheduler.g(float(t[b]))
        for i in range(length):
            current = int(x_t[b, i])
            target = int(x_1[b, i])
            delta = 1.0 if current == target else 0.0
            bracket = (float(probs[b, i, current]) - delta) + (1.0 - delta) * float(
                log_probs[b, i, target]
            )
            total[b, i] = -g * bracket
    return total


class TestMatchesEquation38:
    def test_per_token_loss_matches_a_hand_computed_reference(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, t = _batch()

        got = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Linear(), reduction="none"
        )

        assert torch.allclose(got, _reference(logits, x_1, x_t, t), atol=1e-5)

    def test_the_default_reduction_is_the_mean_over_tokens(self):
        """`Ldfm` is the per-token loss averaged across positions."""
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, t = _batch()

        per_token = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Linear(), reduction="none"
        )
        reduced = discrete_flow_matching_loss(logits, x_1, x_t, t, scheduler=_Linear())

        assert torch.allclose(reduced, per_token.mean(), atol=1e-6)


class TestItIsActuallyAMinimizableLoss:
    """The property none of the formula tests could check.

    Every test comparing against `_reference()` is comparing two transcriptions
    of the same equation made with the same assumption — so a shared misreading
    passes all of them.  It did, twice: first a genuinely inverted objective
    (a model predicting `x_1` perfectly scored -4e-09 while a wrong one scored
    -20.0, and SGD drove mean p(x_1) from 0.061 down to 0.008), and later a
    misplaced `g` that left the loss minimizable but moved its minimizer.

    These tests are about what a loss is *for*, so they are independent of how
    it was transcribed.  They catch an inversion — but note they only check
    *ordering* between candidates, which a displaced minimum preserves.  That
    is why `TestTheOptimum` exists as well.
    """

    def test_a_better_prediction_gives_a_lower_loss(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        vocab = 8
        x_1, x_t, t = torch.tensor([[3]]), torch.tensor([[5]]), torch.tensor([0.5])

        confident = torch.full((1, 1, vocab), -10.0)
        confident[0, 0, 3] = 10.0
        misled = torch.full((1, 1, vocab), -10.0)
        misled[0, 0, 7] = 10.0

        good = float(
            discrete_flow_matching_loss(confident, x_1, x_t, t, scheduler=_Linear())
        )
        bad = float(
            discrete_flow_matching_loss(misled, x_1, x_t, t, scheduler=_Linear())
        )

        assert good < bad, (
            f"the worse model scored lower ({bad:.3f} < {good:.3f}); the "
            "objective is inverted and training would maximize error"
        )

    def test_descending_it_increases_the_target_probability(self):
        """The end-to-end statement: optimizing must teach, not unteach."""
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        torch.manual_seed(0)
        vocab = 8
        logits = torch.randn(1, 4, vocab, requires_grad=True)
        x_1 = torch.randint(0, vocab, (1, 4))
        x_t = torch.randint(0, vocab, (1, 4))
        t = torch.tensor([0.5])

        def target_probability():
            probs = F.softmax(logits.detach(), dim=-1)
            return float(probs.gather(-1, x_1.unsqueeze(-1)).mean())

        before = target_probability()
        optimizer = torch.optim.SGD([logits], lr=1.0)
        for _ in range(100):
            optimizer.zero_grad()
            discrete_flow_matching_loss(
                logits, x_1, x_t, t, scheduler=_Linear()
            ).backward()
            optimizer.step()

        assert target_probability() > before, (
            "descending the loss reduced p(x_1); the objective unlearns its own target"
        )

    def test_it_is_non_negative_like_a_bregman_divergence(self):
        """A Bregman divergence cannot go below zero.

        This is what distinguishes the correct reading from negating the whole
        expression, which is also minimizable but reaches -9.76 over the same
        200 cases.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        torch.manual_seed(0)
        worst = float("inf")
        for _ in range(200):
            value = float(
                discrete_flow_matching_loss(
                    torch.randn(1, 1, 8),
                    torch.randint(0, 8, (1, 1)),
                    torch.randint(0, 8, (1, 1)),
                    torch.tensor([0.5]),
                    scheduler=_Linear(),
                )
            )
            worst = min(worst, value)

        # Tolerance is relative to `g`, not absolute: the loss scales with
        # `g(t)`, so float error does too.  At t=0.5 here g=2, but a test
        # parametrized over t would breach a fixed 1e-6 near t=1 on rounding
        # alone (measured -1.39e-06 at t=0.99999).
        tolerance = 1e-6 * _Linear().g(0.5)
        assert worst >= -tolerance, (
            f"loss went negative ({worst:.4g}, tolerance {-tolerance:.4g}); "
            "not a divergence"
        )


class TestDtypeIndependence:
    def test_bf16_logits_do_not_collapse_g(self):
        """`_scale` central-differences at 1e-4; bf16 carries ~3 digits.

        Casting `t` to the logits dtype made `t + 1e-4 == t - 1e-4` exactly, so
        the derivative became 0 and the entire first term vanished — silently,
        on the default path, since `LinearKappa` defines only `kappa`.
        Measured before the fix: fp32 g(0.5)=2.00003, bf16 g(0.5)=0.00000.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
        from unturtle.processes.discrete_flow import LinearKappa

        logits, x_1, x_t, t = _batch()

        fp32 = discrete_flow_matching_loss(logits, x_1, x_t, t, scheduler=LinearKappa())
        bf16 = discrete_flow_matching_loss(
            logits.bfloat16(), x_1, x_t, t, scheduler=LinearKappa()
        )

        assert abs(float(fp32) - float(bf16)) < 0.1, (
            f"bf16 gave {float(bf16):.4f} against fp32 {float(fp32):.4f}; the "
            "scheduler math is being done in the logits dtype"
        )

    def test_the_repo_scheduler_takes_the_derived_branch(self):
        """Pins why the above is a default-path bug rather than a corner case."""
        from unturtle.processes.discrete_flow import LinearKappa

        assert not hasattr(LinearKappa(), "g"), (
            "LinearKappa now defines g; the finite-difference branch is no "
            "longer the default and the bf16 test above is less load-bearing"
        )


class TestDomainEdges:
    def test_t_near_one_does_not_step_outside_the_path(self):
        """Central differencing at t > 1-h evaluates kappa off the path.

        Harmless for a linear schedule, silently wrong for one that clamps or
        turns over past 1.  A one-sided difference at the edge is correct.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        class _Clamped:
            """Flat past t=1, so a probe at t+h there sees a *different*
            derivative than a one-sided difference inside the domain would.

            A linear schedule extrapolates harmlessly, which is why the
            earlier version of this test could not tell the two apart.
            """

            def kappa(self, t):
                return torch.as_tensor(t).clamp(max=1.0)

        logits, x_1, x_t, _ = _batch(batch=1, length=2)

        from unturtle.diffusion.dfm_loss import _scale

        # Analytic g just inside the domain: kappa'=1, so g = 1/(1-t).
        t = torch.tensor([0.99995])
        derived = float(_scale(_Clamped(), t))
        expected = 1.0 / (1.0 - float(t))

        assert abs(derived - expected) / expected < 0.05, (
            f"derived g={derived:.1f} against analytic {expected:.1f}; the "
            "probe stepped past t=1 where kappa is flat, halving the slope"
        )

        got = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Clamped(), reduction="none"
        )
        assert torch.isfinite(got).all()

    def test_t_at_one_is_bounded_rather_than_infinite(self):
        """`g` genuinely diverges as kappa -> 1; the clamp bounds it.

        Documented as a backstop, not a licence to sample the singularity.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, _ = _batch(batch=1, length=2)

        got = discrete_flow_matching_loss(
            logits,
            x_1,
            x_t,
            torch.tensor([1.0]),
            scheduler=_Linear2(),
            reduction="none",
        )

        assert torch.isfinite(got).all()


class _Linear2:
    """kappa-only, so `g` is derived and the clamp is exercised."""

    def kappa(self, t):
        return t


class TestTheGScaleIsLoadBearing:
    def test_the_loss_scales_with_g_of_t(self):
        """Not decoration: it weights the objective across timesteps.

        `g(0.5) = 2` and `g(0.75) = 4` under linear kappa, so the first term
        must double between them.  A CE-shaped implementation would be flat.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, _ = _batch()

        early = discrete_flow_matching_loss(
            logits,
            x_1,
            x_t,
            torch.full((2,), 0.5),
            scheduler=_Linear(),
            reduction="none",
        )
        late = discrete_flow_matching_loss(
            logits,
            x_1,
            x_t,
            torch.full((2,), 0.75),
            scheduler=_Linear(),
            reduction="none",
        )

        assert not torch.allclose(early, late, atol=1e-4), (
            "the loss did not change with t; g(t) is being ignored"
        )

    def test_a_custom_scheduler_is_honored(self):
        """DFM allows other monotone schedules; the paper just picks linear."""
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        class _Double:
            def kappa(self, t):
                return t

            def g(self, t):
                return 2.0 / (1.0 - t)

        logits, x_1, x_t, t = _batch()

        linear = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Linear(), reduction="none"
        )
        doubled = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Double(), reduction="none"
        )

        assert not torch.allclose(linear, doubled, atol=1e-4)

    def test_g_is_derived_from_kappa_when_not_supplied(self):
        """A scheduler exposing only `kappa` still works.

        `g(t) = kappa'(t)/(1 - kappa(t))`, so it is derivable — and deriving it
        keeps the objective tied to the same path the process samples, which
        is the point of injecting the scheduler rather than hardcoding 1/(1-t).
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        class _KappaOnly:
            def kappa(self, t):
                return t

        logits, x_1, x_t, t = _batch()

        derived = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_KappaOnly(), reduction="none"
        )
        explicit = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Linear(), reduction="none"
        )

        assert torch.allclose(derived, explicit, atol=1e-4)


class TestDeltaSemantics:
    def test_an_already_correct_position_drops_the_log_term(self):
        """`1 - delta = 0` there, so only the first term acts.

        Plain cross-entropy would keep supervising the position.  This is the
        clearest structural difference between eq. (3.8) and CE.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        vocab = 6
        logits = torch.randn(1, 1, vocab)
        x_1 = torch.tensor([[3]])
        t = torch.tensor([0.5])

        correct = discrete_flow_matching_loss(
            logits, x_1, x_1.clone(), t, scheduler=_Linear(), reduction="none"
        )

        probs = F.softmax(logits, dim=-1)
        expected = -2.0 * (float(probs[0, 0, 3]) - 1.0)  # g(0.5) = 2, delta = 1
        assert math.isclose(float(correct), expected, abs_tol=1e-5)

    def test_an_incorrect_position_keeps_both_terms(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        vocab = 6
        logits = torch.randn(1, 1, vocab)
        x_1 = torch.tensor([[3]])
        x_t = torch.tensor([[4]])
        t = torch.tensor([0.5])

        got = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Linear(), reduction="none"
        )

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # g(0.5) = 2 scales both terms, not just the first.
        expected = -2.0 * (float(probs[0, 0, 4]) + float(log_probs[0, 0, 3]))
        assert math.isclose(float(got), expected, abs_tol=1e-5)

    def test_the_first_term_reads_the_current_token_not_the_target(self):
        """`p_{1|t}(x_t^i | x_t)`, evaluated at what the position holds.

        Reading the target there instead is a plausible-looking transcription
        error that leaves the loss finite.
        """
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits = torch.zeros(1, 1, 4)
        logits[0, 0, 0] = 5.0  # mass on token 0
        x_1 = torch.tensor([[1]])
        t = torch.tensor([0.5])

        holding_0 = discrete_flow_matching_loss(
            logits, x_1, torch.tensor([[0]]), t, scheduler=_Linear(), reduction="none"
        )
        holding_2 = discrete_flow_matching_loss(
            logits, x_1, torch.tensor([[2]]), t, scheduler=_Linear(), reduction="none"
        )

        assert not torch.allclose(holding_0, holding_2, atol=1e-4), (
            "the current token did not affect the loss; the first term is "
            "reading the wrong index"
        )


class TestMasking:
    def test_padding_can_be_excluded(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, t = _batch(batch=1, length=4)
        mask = torch.tensor([[True, True, False, False]])

        masked = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Linear(), loss_mask=mask
        )
        first_two = discrete_flow_matching_loss(
            logits[:, :2], x_1[:, :2], x_t[:, :2], t, scheduler=_Linear()
        )

        assert torch.allclose(masked, first_two, atol=1e-5), (
            "masked reduction did not match computing on the kept positions"
        )

    def test_an_all_false_mask_does_not_divide_by_zero(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, t = _batch(batch=1, length=4)

        got = discrete_flow_matching_loss(
            logits,
            x_1,
            x_t,
            t,
            scheduler=_Linear(),
            loss_mask=torch.zeros(1, 4, dtype=torch.bool),
        )

        assert torch.isfinite(got)


class TestPerPositionTimesteps:
    def test_it_accepts_b_by_l_timesteps(self):
        """Packed rows carry one `t` per segment (#62/#65)."""
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, _ = _batch(batch=1, length=4)
        t = torch.tensor([[0.25, 0.25, 0.75, 0.75]])

        got = discrete_flow_matching_loss(
            logits, x_1, x_t, t, scheduler=_Linear(), reduction="none"
        )

        assert got.shape == (1, 4)
        # The two halves used different g(t), so they must not be interchangeable.
        flat = discrete_flow_matching_loss(
            logits,
            x_1,
            x_t,
            torch.tensor([0.25]),
            scheduler=_Linear(),
            reduction="none",
        )
        assert not torch.allclose(got[:, 2:], flat[:, 2:], atol=1e-4)

    def test_a_mismatched_timestep_shape_is_rejected(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, _ = _batch(batch=2, length=4)

        with pytest.raises(ValueError, match="timesteps"):
            discrete_flow_matching_loss(
                logits, x_1, x_t, torch.zeros(3), scheduler=_Linear()
            )


class TestGradients:
    def test_gradients_flow_to_the_logits(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        logits, x_1, x_t, t = _batch()
        logits = logits.requires_grad_(True)

        discrete_flow_matching_loss(logits, x_1, x_t, t, scheduler=_Linear()).backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestTheOptimum:
    """What the loss *converges to*, not what it evaluates to.

    Every other test in this file compares the implementation against a
    reference, or compares two hand-built predictions.  Neither can see a
    displaced minimum: a reference transcribed from the same equation shares
    the misreading, and two candidates that both move the wrong way are still
    correctly ordered relative to each other.

    That gap shipped twice on this one equation — an inverted sign (#94) and
    then a misplaced `g` whose minimizer was the interpolant `p_t` instead of
    the clean posterior `p_{1|t}` (#97).  The second one left an end-to-end
    model with 31% of sampled positions still holding the source token.

    So these tests descend the loss and ask where it lands.
    """

    @staticmethod
    def _optimum(x_t_token, t, target, *, vocab=6, samples=4000, iters=900):
        """Minimize over raw logits at a fixed ``(x_t, t)`` and return the fit."""
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

        generator = torch.Generator().manual_seed(0)
        x_1 = torch.multinomial(
            target.expand(samples, vocab), 1, generator=generator
        ).reshape(1, samples)
        x_t = torch.full((1, samples), x_t_token)

        logits = torch.zeros(1, 1, vocab, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.05)
        for _ in range(iters):
            loss = discrete_flow_matching_loss(
                logits.expand(1, samples, vocab),
                x_1,
                x_t,
                torch.tensor([t]),
                scheduler=_Linear(),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return F.softmax(logits.detach(), dim=-1)[0, 0]

    def test_the_minimizer_is_the_clean_posterior_at_every_t(self):
        """`p_{1|t}` is the distribution over CLEAN data, so it cannot vary with `t`.

        The bug this pins produced `q(x_t) = t` exactly — 0.9 at `t = 0.9`,
        where the correct answer is the data distribution regardless of `t`.
        Because `g` is a common positive factor over the whole bracket, a
        correct implementation's optimum is independent of `t`; that
        independence is the assertion.
        """
        target = torch.tensor([0.40, 0.25, 0.20, 0.10, 0.05, 0.0])

        for t in (0.1, 0.5, 0.9):
            fitted = self._optimum(5, t, target)

            distance = 0.5 * float((fitted - target).abs().sum())
            assert distance < 0.05, (
                f"at t={t} the loss is minimized by "
                f"{[round(v, 3) for v in fitted.tolist()]}, which is "
                f"{distance:.4f} in total variation from the data distribution "
                f"{target.tolist()}; the objective is not learning p_1|t"
            )

    def test_the_minimizer_does_not_favour_whatever_the_position_holds(self):
        """The specific failure: mass piling onto `x_t` because the first term pays.

        Checked on a token that carries real probability mass, not just on the
        mask id — with `p*(x_t) = 0` the log term gives `x_t` no pull at all,
        so a mask-only check sees a different (and weaker) version of this.
        """
        target = torch.tensor([0.30, 0.25, 0.20, 0.13, 0.07, 0.05])

        for held in (0, 2, 5):
            fitted = self._optimum(held, 0.9, target)

            assert abs(float(fitted[held]) - float(target[held])) < 0.05, (
                f"holding token {held} at t=0.9, the optimum assigns it "
                f"{float(fitted[held]):.3f} against a true {float(target[held]):.3f}; "
                "the first term is biasing the prediction toward the current state"
            )
