"""
Discrete flow-matching objective (#65 Phase A, FS-DFM eq. 3.8).

The paper derives the loss as a **Bregman divergence**, not cross-entropy::

    L_i = -g(t)*[ p_{1|t}(x_t^i | x_t) - delta_{x_1^i}(x_t^i) ]
          + [ 1 - delta_{x_1^i}(x_t^i) ] * log p_{1|t}(x_1^i | x_t)

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
    """Equation (3.8), written out position by position."""
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
            first = -g * (float(probs[b, i, current]) - delta)
            second = (1.0 - delta) * float(log_probs[b, i, target])
            total[b, i] = first + second
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
        expected = -2.0 * float(probs[0, 0, 4]) + float(log_probs[0, 0, 3])
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
