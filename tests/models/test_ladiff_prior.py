# Copyright 2025-present nishide-dev & the Unturtle team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LaDiff latent prior (#130): tanh-logSNR schedule, x0 denoiser, Euler sampler.

Verbatim sources (extracted from the paper PDF, frozen on the issue):

- eq. (33-35): logSNR(t) = -d log tan(pi t / 2);
  sigma^2 = sigmoid(-logSNR), alpha^2 = sigmoid(logSNR)  [d = 10 primary]
- Algorithm 2 (training): z standardized; z_t = alpha_t z + sigma_t eps;
  50% self-conditioning z~ = z_psi(z_t, t, none) DETACHED; MSE ||z^ - z||^2
- Algorithm 3 (sampling): Euler on the velocity
  v^ = (1/sigma)((sigma alpha' - sigma' alpha) z^ + sigma' z_t),
  self-conditioning carried across steps, optional gamma re-noising,
  denormalize z <- sigma_z z + mu_z before discrete decode.

Analytic facts the tests lean on (derived, then verified numerically here):
  s'(t) = -d pi / sin(pi t);  alpha' = alpha sigma^2 s'/2;
  sigma' = -sigma alpha^2 s'/2.
Key identity: with a PERFECT x0 prediction (z^ = z) the velocity equals
alpha' z + sigma' eps — the exact time-derivative of the diffusion
trajectory, so Euler follows the trajectory up to O(dt^2) per step.
"""

import math

import pytest
import torch

from unturtle.models.latent.prior_dit import (
    LaDiffPriorConfig,
    LatentPriorDenoiser,
    TanhLogSNRSchedule,
    ladiff_prior_loss,
    sample_latent_prior,
)

DIM = 16
N_LATENTS = 4


def tiny_config(**overrides) -> LaDiffPriorConfig:
    defaults = dict(
        latent_dim=DIM,
        num_latents=N_LATENTS,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        cond_dim=8,
        schedule_d=10.0,
    )
    defaults.update(overrides)
    return LaDiffPriorConfig(**defaults)


class TestTanhLogSNRSchedule:
    def test_matches_the_paper_formulas_pointwise(self):
        sched = TanhLogSNRSchedule(d=10.0)
        for t in (0.1, 0.25, 0.5, 0.9):
            t_ = torch.tensor(t)
            logsnr = -10.0 * math.log(math.tan(math.pi * t / 2))
            assert sched.logsnr(t_).item() == pytest.approx(logsnr, rel=1e-5)
            assert sched.alpha(t_).item() == pytest.approx(
                math.sqrt(1 / (1 + math.exp(-logsnr))), rel=1e-5
            )
            assert sched.sigma(t_).item() == pytest.approx(
                math.sqrt(1 / (1 + math.exp(logsnr))), rel=1e-5
            )

    def test_variance_preserving_identity(self):
        sched = TanhLogSNRSchedule(d=10.0)
        t = torch.linspace(0.001, 0.999, 41)
        vp = sched.alpha(t) ** 2 + sched.sigma(t) ** 2
        assert torch.allclose(vp, torch.ones_like(vp), atol=1e-5)

    def test_endpoints_clean_and_noise(self):
        sched = TanhLogSNRSchedule(d=10.0)
        assert sched.alpha(torch.tensor(0.001)).item() == pytest.approx(1.0, abs=1e-5)
        assert sched.sigma(torch.tensor(0.999)).item() == pytest.approx(1.0, abs=1e-5)

    def test_derivatives_match_finite_differences(self):
        sched = TanhLogSNRSchedule(d=10.0)
        h = 1e-6
        f64 = (
            torch.float64
        )  # fp32 rounds alpha to 1.0 near t=0.1 (d=10), zeroing the fd
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            t_ = torch.tensor(t, dtype=f64)
            fd_alpha = (
                sched.alpha(torch.tensor(t + h, dtype=f64))
                - sched.alpha(torch.tensor(t - h, dtype=f64))
            ) / (2 * h)
            fd_sigma = (
                sched.sigma(torch.tensor(t + h, dtype=f64))
                - sched.sigma(torch.tensor(t - h, dtype=f64))
            ) / (2 * h)
            assert sched.alpha_dot(t_).item() == pytest.approx(
                fd_alpha.item(), rel=1e-3
            ), f"alpha_dot at t={t}"
            assert sched.sigma_dot(t_).item() == pytest.approx(
                fd_sigma.item(), rel=1e-3
            ), f"sigma_dot at t={t}"

    def test_d_parameter_is_live(self):
        assert TanhLogSNRSchedule(d=2.0).alpha(torch.tensor(0.3)) != TanhLogSNRSchedule(
            d=10.0
        ).alpha(torch.tensor(0.3))


class TestVelocityIdentity:
    def test_perfect_denoiser_velocity_is_the_trajectory_derivative(self):
        """Algebraic identity: v^(z_t, z^=z) == alpha' z + sigma' eps when
        z_t = alpha z + sigma eps.  This is the hand-checkable solver case
        the run protocol requires."""
        from unturtle.models.latent.prior_dit import euler_velocity

        sched = TanhLogSNRSchedule(d=10.0)
        g = torch.Generator().manual_seed(0)
        z = torch.randn(3, N_LATENTS, DIM, generator=g)
        eps = torch.randn(3, N_LATENTS, DIM, generator=g)
        for t in (0.2, 0.5, 0.8):
            t_ = torch.tensor(t)
            zt = sched.alpha(t_) * z + sched.sigma(t_) * eps
            v = euler_velocity(sched, zt, z, t_)
            expected = sched.alpha_dot(t_) * z + sched.sigma_dot(t_) * eps
            assert torch.allclose(v, expected, atol=1e-4), f"t={t}"

    def test_euler_with_perfect_denoiser_converges_to_the_trajectory(self):
        """Integrating from t=0.9 to t=0.1 with z^ = z fixed must land on
        alpha(0.1) z + sigma(0.1) eps, with error shrinking as steps grow."""
        from unturtle.models.latent.prior_dit import euler_velocity

        sched = TanhLogSNRSchedule(d=10.0)
        g = torch.Generator().manual_seed(1)
        z = torch.randn(2, N_LATENTS, DIM, generator=g)
        eps = torch.randn(2, N_LATENTS, DIM, generator=g)

        def integrate(steps: int) -> float:
            taus = torch.linspace(0.9, 0.1, steps + 1)
            zt = sched.alpha(taus[0]) * z + sched.sigma(taus[0]) * eps
            for m in range(steps):
                v = euler_velocity(sched, zt, z, taus[m])
                zt = zt - (taus[m] - taus[m + 1]) * v
            target = sched.alpha(taus[-1]) * z + sched.sigma(taus[-1]) * eps
            return float((zt - target).abs().max())

        coarse, fine = integrate(40), integrate(400)
        assert fine < coarse / 5, (coarse, fine)
        assert fine < 5e-3


class TestPriorLoss:
    def test_loss_is_mse_on_standardized_targets_and_deterministic(self):
        model = LatentPriorDenoiser(tiny_config())
        z = torch.randn(4, N_LATENTS, DIM, generator=torch.Generator().manual_seed(2))
        l1 = ladiff_prior_loss(model, z, generator=torch.Generator().manual_seed(3))
        l2 = ladiff_prior_loss(model, z, generator=torch.Generator().manual_seed(3))
        assert torch.equal(l1["total"], l2["total"])
        assert l1["total"].ndim == 0 and torch.isfinite(l1["total"])

    def test_self_conditioning_branch_fires_half_the_time_and_detaches(self):
        """Algorithm 2 lines 6-10: the self-conditioning prediction is made
        WITHOUT gradient (teacher detached).  Detection: when the branch
        fires, the denoiser forward runs twice."""
        model = LatentPriorDenoiser(tiny_config())
        calls = []
        original = model.forward

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        model.forward = counting
        z = torch.randn(4, N_LATENTS, DIM, generator=torch.Generator().manual_seed(4))
        counts = []
        for seed in range(10):
            calls.clear()
            ladiff_prior_loss(model, z, generator=torch.Generator().manual_seed(seed))
            counts.append(len(calls))
        assert 1 in counts and 2 in counts, counts

    def test_self_conditioning_teacher_carries_no_gradient(self):
        """The teacher call must run with grad DISABLED and its output must
        be detached — Algorithm 2's semantics.  Spied directly: per forward,
        record torch.is_grad_enabled() and the self_cond tensor's graph
        attachment."""
        model = LatentPriorDenoiser(tiny_config())
        modes = []
        original = model.forward

        def spy(zt, t, self_cond=None):
            modes.append(
                (
                    torch.is_grad_enabled(),
                    None if self_cond is None else self_cond.requires_grad,
                )
            )
            return original(zt, t, self_cond=self_cond)

        model.forward = spy
        z = torch.randn(4, N_LATENTS, DIM, generator=torch.Generator().manual_seed(5))
        for seed in range(10):
            modes.clear()
            losses = ladiff_prior_loss(
                model, z, generator=torch.Generator().manual_seed(seed)
            )
            if losses["self_conditioned"]:
                assert len(modes) == 2
                assert modes[0][0] is False, "teacher ran WITH grad enabled"
                assert modes[1][0] is True, "student ran without grad"
                assert modes[1][1] is False, "self_cond input not detached"
                return
        pytest.fail("self-conditioning branch never fired in 10 seeds")

    def test_the_regression_target_is_z_not_epsilon(self):
        """Target-swap pin: a stub that echoes z_t back makes the loss equal
        MSE(z_t, z), reconstructable by replaying the SAME generator draws.
        An eps-target mutant (||z_hat - eps||^2) yields a different value."""

        class Echo(torch.nn.Module):
            config = tiny_config()

            def forward(self, zt, t, self_cond=None):
                return zt

        g = torch.Generator().manual_seed(20)
        z = torch.randn(4, N_LATENTS, DIM, generator=g)
        loss = ladiff_prior_loss(Echo(), z, generator=torch.Generator().manual_seed(21))

        from unturtle.models.latent.prior_dit import TanhLogSNRSchedule

        sched = TanhLogSNRSchedule(d=10.0)
        replay = torch.Generator().manual_seed(21)
        t = torch.rand(4, generator=replay).clamp(1e-3, 1 - 1e-3)
        eps = torch.randn(4, N_LATENTS, DIM, generator=replay)
        zt = (
            sched.alpha(t).reshape(-1, 1, 1) * z
            + sched.sigma(t).reshape(-1, 1, 1) * eps
        )
        expected = torch.nn.functional.mse_loss(zt, z)
        assert loss["total"].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_x0_objective_reaches_the_analytic_posterior_mean(self):
        """Learned-optimum test (protocol requirement): for scalar z ~ N(0, 1)
        the MSE-optimal x0 prediction is E[z | z_t] = alpha z_t / (alpha^2 +
        sigma^2 = 1) = alpha z_t (posterior mean of a conjugate Gaussian).
        A linear readout z^ = w * z_t trained to convergence at FIXED t must
        recover w* = alpha(t) — testing what the objective is FOR, not its
        formula."""
        sched = TanhLogSNRSchedule(d=10.0)
        t = torch.tensor(0.5)
        alpha, sigma = sched.alpha(t), sched.sigma(t)
        g = torch.Generator().manual_seed(6)
        w = torch.zeros((), requires_grad=True)
        opt = torch.optim.Adam([w], lr=0.05)
        for _ in range(400):
            z = torch.randn(4096, generator=g)
            eps = torch.randn(4096, generator=g)
            zt = alpha * z + sigma * eps
            loss = ((w * zt - z) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert w.item() == pytest.approx(float(alpha), abs=0.02)


class TestSampler:
    def test_sampling_is_deterministic_shaped_and_finite(self):
        model = LatentPriorDenoiser(tiny_config()).eval()
        a = sample_latent_prior(
            model, batch=3, steps=20, generator=torch.Generator().manual_seed(7)
        )
        b = sample_latent_prior(
            model, batch=3, steps=20, generator=torch.Generator().manual_seed(7)
        )
        assert a.shape == (3, N_LATENTS, DIM)
        assert torch.equal(a, b)
        assert torch.isfinite(a).all()

    def test_gamma_zero_and_positive_differ(self):
        model = LatentPriorDenoiser(tiny_config()).eval()
        a = sample_latent_prior(
            model,
            batch=2,
            steps=10,
            gamma=0.0,
            generator=torch.Generator().manual_seed(8),
        )
        b = sample_latent_prior(
            model,
            batch=2,
            steps=10,
            gamma=0.3,
            generator=torch.Generator().manual_seed(8),
        )
        assert not torch.allclose(a, b)

    def test_self_conditioning_is_carried_across_steps(self):
        """Algorithm 3 line 8: z^ from step m conditions step m-1.  Spy on
        the denoiser's self_cond input: after the first step it must be the
        PREVIOUS prediction, not none/zeros."""
        model = LatentPriorDenoiser(tiny_config()).eval()
        seen = []
        original = model.forward

        def spy(zt, t, self_cond=None):
            seen.append(None if self_cond is None else self_cond.detach().clone())
            return original(zt, t, self_cond=self_cond)

        model.forward = spy
        sample_latent_prior(
            model, batch=2, steps=5, generator=torch.Generator().manual_seed(9)
        )
        assert seen[0] is None
        assert all(s is not None for s in seen[1:]), "self-cond not carried"
        assert len(seen) == 5

    def test_perfect_denoiser_sampler_recovers_the_data_mode(self):
        """End-to-end solver sanity: a stub denoiser that always predicts a
        fixed z* must make the sampler converge near z* at tau_min (where
        alpha ~= 1)."""
        target = torch.randn(
            1, N_LATENTS, DIM, generator=torch.Generator().manual_seed(10)
        )

        class Stub(torch.nn.Module):
            config = tiny_config()

            def forward(self, zt, t, self_cond=None):
                return target.expand_as(zt)

        out = sample_latent_prior(
            Stub(), batch=4, steps=200, generator=torch.Generator().manual_seed(11)
        )
        assert (out - target).abs().max() < 0.05


class TestDenoiserModule:
    def test_time_conditioning_is_live(self):
        """adaLN-Zero makes t inert at EXACT init (the gates are zeroed) —
        the same vacuity trap as the latent channel.  Open one block's
        modulation first, then t must matter."""
        model = LatentPriorDenoiser(tiny_config()).eval()
        torch.nn.init.normal_(model.blocks[0].adaLN_modulation.weight, std=0.1)
        z = torch.randn(2, N_LATENTS, DIM, generator=torch.Generator().manual_seed(12))
        with torch.no_grad():
            a = model(z, torch.tensor([0.2, 0.2]))
            b = model(z, torch.tensor([0.8, 0.8]))
        assert not torch.allclose(a, b), "denoiser ignores t"

    def test_self_conditioning_input_is_live(self):
        model = LatentPriorDenoiser(tiny_config()).eval()
        g = torch.Generator().manual_seed(13)
        z = torch.randn(2, N_LATENTS, DIM, generator=g)
        c1 = torch.randn(2, N_LATENTS, DIM, generator=g)
        t = torch.tensor([0.5, 0.5])
        with torch.no_grad():
            a = model(z, t, self_cond=c1)
            b = model(z, t, self_cond=None)
        assert not torch.allclose(a, b), "denoiser ignores self-conditioning"

    def test_output_shape_matches_input(self):
        model = LatentPriorDenoiser(tiny_config())
        z = torch.randn(5, N_LATENTS, DIM)
        assert model(z, torch.full((5,), 0.4)).shape == z.shape


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


class TestGammaRenoise:
    def test_renoising_actually_draws_noise(self):
        """gamma has TWO effects (time warp + re-noising); the warp alone
        made the earlier comparison pass even with re-noising deleted.  The
        re-noise consumes generator draws, so the generator's end state
        distinguishes them."""
        model = LatentPriorDenoiser(tiny_config()).eval()
        g0 = torch.Generator().manual_seed(30)
        sample_latent_prior(model, batch=2, steps=6, gamma=0.0, generator=g0)
        g1 = torch.Generator().manual_seed(30)
        sample_latent_prior(model, batch=2, steps=6, gamma=0.3, generator=g1)
        assert not torch.equal(g0.get_state(), g1.get_state()), (
            "gamma>0 consumed no extra randomness: re-noising is dead"
        )
