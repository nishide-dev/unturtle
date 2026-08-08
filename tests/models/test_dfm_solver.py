"""
Discrete flow-matching solver (#65 Phase A, FS-DFM App. B.1).

Per step ``s`` on a grid ``t_s`` with width ``h_s``::

    p_{1|t} = softmax(logits / T)
    lambda_s^i = gbar_{t,h} * (1 - p_{1|t}(X_t^i | X_t))     # per-position exit rate
    J_s^i ~ Bernoulli(1 - exp(-h * lambda_s^i))              # jump indicator

and on a jump, resample from ``p_{1|t}`` **restricted to the off-diagonals**
(the current token excluded).  Under the paper's linear ``kappa`` the Cumulative Scalar has the
closed form ``gbar_{t,h} = (1/h) * ln((1-t) / (1-t-h))``.

**These tests are deliberately about behaviour, not transcription.**  The
objective in this same issue shipped with an inverted sign that thirteen
formula tests could not see, because the reference was transcribed from the
same equation with the same assumption.  A sampler is worse: an error in the
jump law surfaces as merely-worse samples, never as a failure.  So the
load-bearing tests here are limits and monotonicity — properties that hold
regardless of how the formula was read.
"""

import math

import pytest
import torch


def _uniform_logits(batch=1, length=6, vocab=8):
    return torch.zeros(batch, length, vocab)


def _peaked_logits(target, batch=1, length=6, vocab=8, strength=20.0):
    logits = torch.zeros(batch, length, vocab)
    logits[..., target] = strength
    return logits


class TestCumulativeScalar:
    def test_it_matches_the_closed_form_under_linear_kappa(self):
        from unturtle.models.generation.dfm_solver import cumulative_scalar

        t, h = 0.3, 0.2
        expected = (1.0 / h) * math.log((1.0 - t) / (1.0 - t - h))

        got = cumulative_scalar(torch.tensor([t]), h)

        assert math.isclose(float(got), expected, rel_tol=1e-6)

    def test_it_approaches_the_instantaneous_g_as_h_shrinks(self):
        """The limit that a transcription test cannot check.

        `gbar` integrates `g` over `[t, t+h]` and normalizes by `h`, so as
        `h -> 0` it must approach `g(t) = 1/(1-t)`.  A sign error or an
        off-by-one inside the log would still produce a plausible finite
        number at any single `h`; only the limit exposes it.
        """
        from unturtle.models.generation.dfm_solver import cumulative_scalar

        t = torch.tensor([0.4])
        instantaneous = 1.0 / (1.0 - float(t))

        errors = [
            abs(float(cumulative_scalar(t, h)) - instantaneous)
            for h in (0.1, 0.01, 1e-3)
        ]

        assert errors == sorted(errors, reverse=True), (
            f"error did not shrink monotonically with h: {errors}"
        )
        assert errors[-1] < 1e-2, (
            f"gbar(t, 1e-3)={float(cumulative_scalar(t, 1e-3)):.4f} against "
            f"g(t)={instantaneous:.4f}"
        )

    def test_it_grows_as_t_approaches_one(self):
        """`g` diverges at the end of the path, so `gbar` must grow too."""
        from unturtle.models.generation.dfm_solver import cumulative_scalar

        early = float(cumulative_scalar(torch.tensor([0.1]), 0.05))
        late = float(cumulative_scalar(torch.tensor([0.8]), 0.05))

        assert late > early

    def test_a_step_past_the_end_of_the_path_is_rejected(self):
        """`t + h > 1` puts `ln` of a negative number in the closed form."""
        from unturtle.models.generation.dfm_solver import cumulative_scalar

        with pytest.raises(ValueError, match="step"):
            cumulative_scalar(torch.tensor([0.9]), 0.2)


class TestJumpLaw:
    def test_a_confident_correct_position_almost_never_jumps(self):
        """`lambda ~ gbar * (1 - p(current))`, so p(current)->1 means no jump."""
        from unturtle.models.generation.dfm_solver import jump_probability

        gbar = torch.tensor([2.0])
        p_current = torch.tensor([[0.999]])

        probability = jump_probability(p_current, gbar, h=0.1)

        assert float(probability) < 0.01

    def test_a_confidently_wrong_position_almost_always_jumps(self):
        from unturtle.models.generation.dfm_solver import jump_probability

        gbar = torch.tensor([50.0])
        p_current = torch.tensor([[0.001]])

        probability = jump_probability(p_current, gbar, h=0.5)

        assert float(probability) > 0.9

    def test_the_probability_is_monotone_in_the_step_size(self):
        """A longer step means more opportunity to jump.

        Independent of how the exponential law was transcribed.
        """
        from unturtle.models.generation.dfm_solver import jump_probability

        gbar = torch.tensor([2.0])
        p_current = torch.tensor([[0.5]])

        values = [
            float(jump_probability(p_current, gbar, h=h)) for h in (0.1, 0.3, 0.9)
        ]

        assert values == sorted(values), f"not monotone in h: {values}"

    def test_survival_compounds_across_split_steps(self):
        """The exponential law's defining property; a linear one lacks it.

        A holding time is memoryless: surviving `2h` must be exactly as likely
        as surviving `h` twice, since the process carries no memory of how long
        it has already waited.  So `(1 - P(h))**2 == 1 - P(2h)`.

        This is what pins `1 - exp(-h*lambda)` specifically.  Replacing it with
        a linear `clamp(h*lambda, max=1)` passes every other test in this file,
        including the monotonicity and range checks — yet the two diverge by
        13% at `h*lambda = 0.25` and 58% at `h*lambda = 1.0`, which is squarely
        the few-step regime FS-DFM exists to serve.  The linear form over-jumps
        there.
        """
        from unturtle.models.generation.dfm_solver import jump_probability

        gbar = torch.tensor([2.0])
        p_current = torch.tensor([[0.5]])  # rate = 2.0 * 0.5 = 1.0

        h = 0.25
        one_step = float(jump_probability(p_current, gbar, h=h))
        double_step = float(jump_probability(p_current, gbar, h=2 * h))

        survives_twice = (1.0 - one_step) ** 2
        survives_double = 1.0 - double_step

        assert math.isclose(survives_twice, survives_double, rel_tol=1e-6), (
            f"surviving two steps of h={h} has probability {survives_twice:.6f} "
            f"but surviving one step of h={2 * h} has {survives_double:.6f}; "
            "the holding time is not memoryless, so this is not the "
            "exponential law"
        )

    def test_it_stays_a_probability(self):
        from unturtle.models.generation.dfm_solver import jump_probability

        torch.manual_seed(0)
        for _ in range(50):
            probability = jump_probability(
                torch.rand(1, 4), torch.rand(1) * 100, h=float(torch.rand(1))
            )
            assert bool(((probability >= 0) & (probability <= 1)).all())


class TestResamplingExcludesTheCurrentToken:
    def test_a_jump_never_returns_the_token_it_left(self):
        """App. B.1: on a jump, resample from the off-diagonals.

        Without the exclusion a "jump" can land back where it started, which
        wastes the step and biases the trajectory toward staying put — and
        looks like nothing at all in a loss curve.
        """
        from unturtle.models.generation.dfm_solver import sample_jump_targets

        vocab = 6
        # Mass concentrated on token 2, which is also the current token.
        probs = torch.full((1, 1, vocab), 0.01)
        probs[0, 0, 2] = 0.95
        current = torch.tensor([[2]])

        generator = torch.Generator().manual_seed(0)
        for _ in range(50):
            drawn = sample_jump_targets(probs, current, generator=generator)
            assert int(drawn[0, 0]) != 2, "a jump returned the current token"

    def test_the_off_diagonals_keep_their_relative_weights(self):
        """Excluding the current token must not distort what remains.

        With the remaining two tokens at a 3:1 ratio, that ratio has to
        survive the exclusion — a jump is a draw from the off-diagonals, not a
        uniform pick among them.
        """
        from unturtle.models.generation.dfm_solver import sample_jump_targets

        probs = torch.tensor([[[0.60, 0.30, 0.10]]])
        current = torch.tensor([[0]])

        generator = torch.Generator().manual_seed(0)
        draws = [
            int(sample_jump_targets(probs, current, generator=generator)[0, 0])
            for _ in range(600)
        ]

        share_of_one = draws.count(1) / len(draws)
        assert 0.70 < share_of_one < 0.80, (
            f"token 1 drawn {share_of_one:.2f} of the time, expected ~0.75 "
            "(0.30 against 0.10); the off-diagonal weights were distorted"
        )

    def test_a_row_with_no_off_diagonal_mass_stays_put(self):
        """The one thing the `exhausted` guard is for.

        A one-hot prediction (or underflow in low precision) leaves only
        numerical dust off the diagonal, and `multinomial` would happily draw
        an arbitrary token from it.  There is nowhere to jump, so the position
        must stay.

        Note this is *not* testing renormalization: `multinomial` normalizes
        its weights internally, so dividing by the remaining mass is
        unobservable.  This guard is the only part of the exclusion path whose
        removal changes a sampled token.
        """
        from unturtle.models.generation.dfm_solver import sample_jump_targets

        probs = torch.tensor([[[1.0, 1e-30, 1e-30]]])
        current = torch.tensor([[0]])

        generator = torch.Generator().manual_seed(0)
        for _ in range(20):
            drawn = sample_jump_targets(probs, current, generator=generator)
            assert int(drawn[0, 0]) == 0, (
                "a position with no real off-diagonal mass jumped into "
                "numerical dust instead of staying put"
            )

    def test_a_degenerate_single_token_vocabulary_does_not_divide_by_zero(self):
        from unturtle.models.generation.dfm_solver import sample_jump_targets

        probs = torch.tensor([[[1.0]]])
        current = torch.tensor([[0]])

        drawn = sample_jump_targets(probs, current, generator=torch.Generator())

        assert int(drawn[0, 0]) == 0  # nowhere else to go; staying is correct


class TestTheSolverLoop:
    """End-to-end behaviour. This is where a wrong jump law actually shows."""

    def _model(self, target):
        """A denoiser that always predicts `target` confidently."""

        def denoise(x_t, t, h):
            return _peaked_logits(target, *x_t.shape)

        return denoise

    def test_it_converges_to_what_the_model_predicts(self):
        """The property that makes it a sampler at all.

        A wrong exit rate, a wrong holding-time law, or a wrong renormalization
        all still produce *some* trajectory — but only a correct one ends up
        where the denoiser points.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        target = 3
        x_0 = torch.zeros(1, 8, dtype=torch.long)

        final = solve_discrete_flow(
            self._model(target),
            x_0,
            steps=16,
            generator=torch.Generator().manual_seed(0),
        )

        agreement = float((final == target).float().mean())
        assert agreement > 0.9, (
            f"only {agreement:.2f} of positions reached the predicted token"
        )

    def test_more_steps_are_not_worse(self):
        """Monotonicity in the budget, the paper's central quality axis.

        Not a formula check: any implementation produces samples, and only a
        correct one improves (or holds) with more steps.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        target = 5
        x_0 = torch.zeros(1, 16, dtype=torch.long)

        scores = []
        for steps in (2, 8, 32):
            final = solve_discrete_flow(
                self._model(target),
                x_0,
                steps=steps,
                generator=torch.Generator().manual_seed(0),
            )
            scores.append(float((final == target).float().mean()))

        assert scores[-1] >= scores[0] - 1e-6, (
            f"more steps produced a worse sample: {scores}"
        )

    def test_the_intermediate_jump_law_actually_acts(self):
        """Mutation-verified gap: "always jump" and "never jump" both survived.

        The final step resamples unconditionally, so a confident denoiser
        reaches its target regardless of what the intermediate steps do — the
        very thing the exit rate governs was untested.

        Here the denoiser points at a *different* token on the first half of
        the schedule than the second.  If intermediate jumps never happen, the
        early target never appears; if they always happen, positions move on
        every step and the run stops depending on the rate at all.  Either
        way the observed jump count separates them.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        moves = []

        def denoise(x_t, t, h):
            moves.append(x_t.clone())
            # Barely favours a different token: p(current) stays high, so a
            # correct exit rate jumps rarely.
            logits = torch.zeros(*x_t.shape, 8)
            logits.scatter_(-1, x_t.unsqueeze(-1), 4.0)
            return logits

        final = solve_discrete_flow(
            denoise,
            torch.zeros(1, 32, dtype=torch.long),
            steps=8,
            generator=torch.Generator().manual_seed(0),
        )

        # Count how often the state changed between consecutive model calls.
        changes = sum(
            int((moves[i] != moves[i - 1]).sum()) for i in range(1, len(moves))
        )

        assert 0 < changes < 32 * (len(moves) - 1), (
            f"{changes} changes over {len(moves) - 1} intermediate steps on 32 "
            "positions: 0 means the jump law never fires, the maximum means it "
            "always fires — neither depends on the exit rate"
        )
        assert final.shape == (1, 32)

    def test_a_confident_denoiser_jumps_less_than_an_unsure_one(self):
        """The exit rate's defining behaviour: `lambda ~ 1 - p(current)`.

        Independent of how the holding-time law was transcribed — it only
        requires that agreement with the model suppresses movement.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        def make(strength):
            recorded = []

            def denoise(x_t, t, h):
                recorded.append(x_t.clone())
                logits = torch.zeros(*x_t.shape, 8)
                logits.scatter_(-1, x_t.unsqueeze(-1), strength)
                return logits

            return denoise, recorded

        counts = []
        for strength in (8.0, 0.0):  # agrees strongly, then indifferent
            denoise, recorded = make(strength)
            solve_discrete_flow(
                denoise,
                torch.zeros(1, 64, dtype=torch.long),
                steps=8,
                generator=torch.Generator().manual_seed(0),
            )
            counts.append(
                sum(
                    int((recorded[i] != recorded[i - 1]).sum())
                    for i in range(1, len(recorded))
                )
            )

        assert counts[0] < counts[1], (
            f"a denoiser agreeing with the state moved it {counts[0]} times "
            f"versus {counts[1]} for an indifferent one; the exit rate is not "
            "reading p(current)"
        )

    def test_the_observed_jump_frequency_matches_the_rate(self):
        """Pins the rate's *magnitude*, which the endpoint cannot see.

        With a state-independent denoiser the marginal at `t = 1` is `p` for
        any rate whatsoever, because the final step resamples unconditionally.
        So a rate that is off by a constant factor — halving it leaves the
        holding time memoryless and every other test here green — is invisible
        at the endpoint and only shows up in the trajectory.

        Here the denoiser reports a uniform `p`, so `p(current) = 1/V` exactly
        and each intermediate step has a closed-form jump probability
        `1 - exp(-h * gbar_{t,h} * (1 - 1/V))`.  Comparing the *measured*
        frequency against that is a property check, not a re-transcription:
        it fails for any rate scaled away from the paper's.
        """
        from unturtle.models.generation.dfm_solver import (
            cumulative_scalar,
            solve_discrete_flow,
        )

        vocab = 4
        uniform = torch.full((vocab,), 1.0 / vocab)
        log_uniform = torch.log(uniform)
        recorded = []

        def denoise(x_t, t, h):
            recorded.append(x_t.clone())
            return log_uniform.expand(*x_t.shape, vocab).clone()

        steps = 4
        h = 1.0 / steps
        solve_discrete_flow(
            denoise,
            torch.zeros(400, 60, dtype=torch.long),
            steps=steps,
            generator=torch.Generator().manual_seed(0),
        )

        for index in range(1, len(recorded)):
            t = (index - 1) * h
            gbar = float(cumulative_scalar(torch.tensor([t]), h))
            predicted = 1.0 - math.exp(-h * gbar * (1.0 - 1.0 / vocab))
            observed = float((recorded[index] != recorded[index - 1]).float().mean())

            assert abs(observed - predicted) < 0.05, (
                f"at t={t:.2f} the rate predicts P(jump)={predicted:.4f} but "
                f"{observed:.4f} of positions moved; the exit rate's magnitude "
                "does not match the Cumulative Scalar"
            )

    def test_it_is_reproducible_under_a_seeded_generator(self):
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        x_0 = torch.zeros(1, 8, dtype=torch.long)

        first = solve_discrete_flow(
            self._model(2), x_0, steps=8, generator=torch.Generator().manual_seed(7)
        )
        second = solve_discrete_flow(
            self._model(2), x_0, steps=8, generator=torch.Generator().manual_seed(7)
        )

        assert torch.equal(first, second)

    def test_the_input_state_is_not_mutated(self):
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        x_0 = torch.zeros(1, 8, dtype=torch.long)
        before = x_0.clone()

        solve_discrete_flow(self._model(1), x_0, steps=4)

        assert torch.equal(x_0, before)

    def test_zero_steps_is_rejected(self):
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        with pytest.raises(ValueError, match="steps"):
            solve_discrete_flow(
                self._model(1), torch.zeros(1, 4, dtype=torch.long), steps=0
            )

    def test_the_model_receives_the_step_size(self):
        """FS-DFM's denoiser is step-aware: `theta(x_t, t; h)`.

        Dropping `h` silently reduces it to an ordinary DFM model, which is
        precisely the thing Phase B improves on.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        seen = []

        def denoise(x_t, t, h):
            seen.append((float(t.reshape(-1)[0]), float(h)))
            return _uniform_logits(*x_t.shape)

        solve_discrete_flow(denoise, torch.zeros(1, 4, dtype=torch.long), steps=4)

        assert len(seen) == 4
        assert all(math.isclose(h, 0.25, rel_tol=1e-6) for _, h in seen)
        assert [t for t, _ in seen] == sorted(t for t, _ in seen), (
            "the time grid did not advance monotonically"
        )

    def test_it_recovers_a_known_target_distribution(self):
        """The property a sampler actually exists to have.

        Every other test here pins a piece of the mechanism.  This one pins
        the *outcome*: with a denoiser that reports a fixed `p*` regardless of
        state, the process at `t = 1` should be distributed as `p*`, so the
        empirical marginals over many positions must match it.

        This is the sampler's analogue of the property that would have caught
        the objective's inverted sign in this same issue — a wrong exit rate,
        a biased final step, or a botched exclusion all show up here as
        systematic drift, none of which any mechanism test would flag.

        Total-variation tolerance is loose (0.05) against sampling noise of
        roughly 0.005 at this sample count, so it fails on real bias and not
        on seed luck.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        target = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05])
        vocab = len(target)
        log_target = torch.log(target)

        def denoise(x_t, t, h):
            return log_target.expand(*x_t.shape, vocab).clone()

        for steps in (4, 16):
            generator = torch.Generator().manual_seed(0)
            x_0 = torch.randint(0, vocab, (200, 64), generator=generator)

            final = solve_discrete_flow(denoise, x_0, steps=steps, generator=generator)

            empirical = torch.bincount(final.reshape(-1), minlength=vocab).float()
            empirical /= empirical.sum()
            distance = 0.5 * float((empirical - target).abs().sum())

            assert distance < 0.05, (
                f"at steps={steps} the sampled marginals sit {distance:.4f} "
                f"in total variation from the denoiser's own distribution "
                f"{target.tolist()} (got {[round(v, 3) for v in empirical.tolist()]})"
            )

    def test_the_argmax_is_not_forbidden_at_the_final_step(self):
        """Regression: the terminal draw is not a jump.

        The final step originally routed through `sample_jump_targets`, which
        excludes the current token.  Exclusion is right for a *jump* — one that
        lands where it started did not move — but the draw at `t = 1` is not a
        jump, and must be able to return the token a position already holds.

        Excluding there gave the model's own argmax probability exactly zero.
        Starting every position on the argmax, it was emitted 0.0% of the time,
        and the distortion grew with the step budget (TV 0.11 -> 0.42 over
        steps 1 -> 128) rather than washing out, because more steps means more
        positions have settled onto the token the final step then evicts.

        Pinned at `steps=1`, where the final step is the *only* step, so
        nothing else can mask it.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        target = torch.tensor([0.70, 0.20, 0.07, 0.03])
        vocab = len(target)
        log_target = torch.log(target)

        def denoise(x_t, t, h):
            return log_target.expand(*x_t.shape, vocab).clone()

        # Every position starts on token 0 — the argmax the bug made unreachable.
        x_0 = torch.zeros(200, 100, dtype=torch.long)

        final = solve_discrete_flow(
            denoise, x_0, steps=1, generator=torch.Generator().manual_seed(0)
        )

        share_of_argmax = float((final == 0).float().mean())
        assert 0.60 < share_of_argmax < 0.80, (
            f"the model's argmax (p=0.70) was emitted {share_of_argmax:.3f} of "
            "the time; the terminal draw is excluding the current token"
        )

    def test_temperature_flattens_the_sampled_distribution(self):
        """`temperature` was plumbed and validated but asserted nowhere.

        Dropping `/ temperature` from the softmax left all other tests passing,
        so nothing pinned that the argument does anything at all.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        target = torch.tensor([0.70, 0.20, 0.07, 0.03])
        vocab = len(target)
        log_target = torch.log(target)

        def denoise(x_t, t, h):
            return log_target.expand(*x_t.shape, vocab).clone()

        shares = []
        for temperature in (1.0, 50.0):
            final = solve_discrete_flow(
                denoise,
                torch.zeros(200, 50, dtype=torch.long),
                steps=4,
                temperature=temperature,
                generator=torch.Generator().manual_seed(0),
            )
            shares.append(float((final == 0).float().mean()))

        # A high temperature pushes toward uniform, so the peak token's share
        # must fall from ~0.70 toward ~1/V.
        assert shares[1] < shares[0] - 0.15, (
            f"argmax share was {shares[0]:.3f} at T=1 and {shares[1]:.3f} at "
            "T=50; temperature is not reaching the softmax"
        )

    def test_a_non_trivial_batch_is_handled(self):
        """Every other test runs `B=1`, so no shape bug would surface.

        The `t` vector is built per batch row and broadcast against
        `[B, L, V]` logits; a mistake there is invisible at `B=1`.
        """
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        seen_shapes = []

        def denoise(x_t, t, h):
            seen_shapes.append((tuple(x_t.shape), tuple(t.shape)))
            return _uniform_logits(*x_t.shape)

        final = solve_discrete_flow(
            denoise,
            torch.zeros(3, 7, dtype=torch.long),
            steps=4,
            generator=torch.Generator().manual_seed(0),
        )

        assert final.shape == (3, 7)
        assert all(shape == ((3, 7), (3,)) for shape in seen_shapes), (
            f"the denoiser saw {seen_shapes}; `t` must carry one entry per batch row"
        )

    def test_a_non_positive_temperature_is_rejected(self):
        from unturtle.models.generation.dfm_solver import solve_discrete_flow

        with pytest.raises(ValueError, match="temperature"):
            solve_discrete_flow(
                lambda x_t, t, h: _uniform_logits(*x_t.shape),
                torch.zeros(1, 4, dtype=torch.long),
                steps=2,
                temperature=0.0,
            )
