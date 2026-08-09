"""
DiLaDiff slice 2 (#117): MeanFlow self-distillation of the latent prior.

Method: arXiv:2605.23605 §3.3, eqs. 13-16.  The student learns the AVERAGE
velocity ``u(z_t, t, r)`` (mean displacement between two points on the ODE
path) from the frozen LaDiff prior:

    u_tgt = v(z_t, t) - (t - r) * (v . d_z u + d_t u)        (eq. 15)
    L     = || u(z_t, t, r) - stopgrad(u_tgt) ||^2            (eq. 14)

with the directional derivative taken by one JVP (tangents (v, 1, 0)), a
25% fraction of pure flow-matching rows (t = r, where the target degenerates
to the teacher velocity), and sampling by

    z_{tau'} = z_tau + (tau' - tau) * u(z_tau, tau, tau')     (eq. 16)

The teacher velocity comes from the slice-1 prior's x0 prediction on the
linear path: ``v(z_t, t) = (z_t - x0_pred) / t`` — the same identity #116's
average-velocity solver is built on, which is why a perfectly distilled
student collapses guided decoding to ONE latent step.

Deviation recorded: the paper's modified self-conditioning (2 teacher NFEs)
is omitted at prototype scale.
"""

import pytest
import torch

VOCAB = 16
HIDDEN = 32
LENGTH = 8
MASK_ID = VOCAB - 1
NUM_LATENTS = 2


def _flow_config(**overrides):
    from unturtle.models.latent import FlowLMConfig

    defaults = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=NUM_LATENTS,
        num_timesteps=4,
        time_scale=1000.0,
    )
    defaults.update(overrides)
    return FlowLMConfig(**defaults)


class TestTheTwoTimeDenoiser:
    def test_both_time_inputs_are_observable(self):
        """u(z, t, r) needs BOTH times: severing either collapses the
        average velocity to something r- (or t-) independent, which cannot
        represent eq. 13."""
        from unturtle.models.latent import MeanFlowDenoiser

        student = MeanFlowDenoiser(_flow_config()).eval()
        latents = torch.randn(2, NUM_LATENTS, HIDDEN)

        with torch.no_grad():
            base = student(
                latents,
                timesteps=torch.full((2,), 800.0),
                target_timesteps=torch.full((2,), 200.0),
            ).velocity
            moved_t = student(
                latents,
                timesteps=torch.full((2,), 400.0),
                target_timesteps=torch.full((2,), 200.0),
            ).velocity
            moved_r = student(
                latents,
                timesteps=torch.full((2,), 800.0),
                target_timesteps=torch.full((2,), 600.0),
            ).velocity

        assert base.shape == (2, NUM_LATENTS, HIDDEN)
        assert not torch.allclose(base, moved_t), "t conditioning is inert"
        assert not torch.allclose(base, moved_r), "r conditioning is inert"


class _AnchorTeacher:
    """Perfect x0 predictor: always returns the anchor — a straight flow."""

    def __init__(self, anchor):
        self.anchor = anchor

    def __call__(self, latents, timesteps):
        class Out:
            prediction = self.anchor.expand_as(latents)

        return Out()

    def parameters(self):
        return iter(())


class TestTheDistillationObjective:
    def test_a_velocity_exact_student_zeroes_the_pure_fm_rows(self):
        """TestTheOptimum, t = r branch: eq. 15 degenerates to
        ``u_tgt = v``, so a stub that returns the teacher velocity exactly
        has zero loss."""
        from unturtle.models.latent import meanflow_distillation_loss

        anchor = torch.randn(1, NUM_LATENTS, HIDDEN)
        teacher = _AnchorTeacher(anchor)

        class VelocityStub:
            def __call__(self, latents, timesteps, target_timesteps):
                t = (timesteps / 1000.0).view(-1, 1, 1)

                class Out:
                    velocity = (latents - anchor.expand_as(latents)) / t

                return Out()

            def parameters(self):
                return iter(())

        losses = meanflow_distillation_loss(
            VelocityStub(),
            teacher,
            torch.randn(8, NUM_LATENTS, HIDDEN),
            num_timesteps=4,
            time_scale=1000.0,
            pure_fm_fraction=1.0,
            generator=torch.Generator().manual_seed(0),
        )

        assert float(losses["total"]) < 1e-10, (
            f"the exact velocity is not the optimum: {float(losses['total'])}"
        )

    def test_the_straight_flow_average_velocity_zeroes_the_jvp_rows_too(self):
        """TestTheOptimum, r < t branch: on a straight flow the average
        velocity equals the instantaneous one, and the JVP correction
        vanishes analytically (v . d_z u = v/t, d_t u = -v/t) — the same
        stub must be a zero of the FULL objective, not just the degenerate
        rows."""
        from unturtle.models.latent import meanflow_distillation_loss

        anchor = torch.randn(1, NUM_LATENTS, HIDDEN)
        teacher = _AnchorTeacher(anchor)

        class VelocityStub:
            def __call__(self, latents, timesteps, target_timesteps):
                t = (timesteps / 1000.0).view(-1, 1, 1)

                class Out:
                    velocity = (latents - anchor.expand_as(latents)) / t

                return Out()

            def parameters(self):
                return iter(())

        losses = meanflow_distillation_loss(
            VelocityStub(),
            teacher,
            torch.randn(8, NUM_LATENTS, HIDDEN),
            num_timesteps=4,
            time_scale=1000.0,
            pure_fm_fraction=0.0,
            generator=torch.Generator().manual_seed(1),
        )

        assert float(losses["total"]) < 1e-8, (
            f"straight-flow optimum violated: {float(losses['total'])}"
        )

    def test_the_jvp_correction_matches_a_hand_computation(self):
        """A linear stub ``u = A z`` has known derivatives (d_z u = A,
        d_t u = 0), so eq. 15 is computable by hand:
        ``u_tgt = v - (t - r) A v`` — the loss value must match exactly.
        This is what pins the JVP's SIGN and the (t - r) factor; the
        straight-flow test cannot (its correction is zero)."""
        from unturtle.models.latent import meanflow_distillation_loss

        anchor = torch.zeros(1, NUM_LATENTS, HIDDEN)
        teacher = _AnchorTeacher(anchor)
        scale = 0.5

        class LinearStub:
            def __call__(self, latents, timesteps, target_timesteps):
                class Out:
                    velocity = latents * scale

                return Out()

            def parameters(self):
                return iter(())

        z0 = torch.randn(4, NUM_LATENTS, HIDDEN)
        losses = meanflow_distillation_loss(
            LinearStub(),
            teacher,
            z0,
            num_timesteps=4,
            time_scale=1000.0,
            pure_fm_fraction=0.0,
            generator=torch.Generator().manual_seed(2),
        )

        # Replay the loss's seeded draws in its own order.
        replay = torch.Generator().manual_seed(2)
        steps = torch.randint(1, 5, (4,), generator=replay)
        t = (steps.float() / 4).view(-1, 1, 1)
        noise = torch.randn(4, NUM_LATENTS, HIDDEN, generator=replay)
        z_t = (1 - t) * z0 + t * noise
        r = (torch.rand(4, generator=replay).view(-1, 1, 1)) * t
        v = (z_t - anchor) / t  # teacher velocity (x0 = anchor = 0)
        u = z_t * scale
        u_target = v - (t - r) * (v * scale)  # d_z u = scale*I, d_t u = 0
        expected = ((u - u_target) ** 2).mean()

        assert torch.allclose(losses["total"], expected, atol=1e-6), (
            f"JVP correction diverges from the hand computation: "
            f"{float(losses['total'])} vs {float(expected)}"
        )

    def test_the_teacher_receives_no_gradient(self):
        """eq. 14's stopgrad: the target is detached, and the teacher is
        frozen — any gradient reaching it means the distillation is
        optimizing the thing it distills from."""
        from unturtle.models.latent import (
            FlowLMDenoiser,
            MeanFlowDenoiser,
            meanflow_distillation_loss,
        )

        teacher = FlowLMDenoiser(_flow_config())
        student = MeanFlowDenoiser(_flow_config())

        losses = meanflow_distillation_loss(
            student,
            teacher,
            torch.randn(4, NUM_LATENTS, HIDDEN),
            num_timesteps=4,
            time_scale=1000.0,
            generator=torch.Generator().manual_seed(3),
        )
        losses["total"].backward()

        assert all(p.grad is None for p in teacher.parameters()), (
            "the frozen teacher received gradient through the target"
        )
        assert any(
            p.grad is not None and bool(p.grad.abs().sum() > 0)
            for p in student.parameters()
        ), "no gradient reached the student"

    def test_r_never_exceeds_t(self):
        """eq. 13 integrates from r to t with r <= t; a draw with r > t
        would ask the student for a backward average velocity the objective
        never defines."""
        from unturtle.models.latent import meanflow_distillation_loss

        seen = []

        class RecordingStub:
            def __call__(self, latents, timesteps, target_timesteps):
                seen.append((timesteps.clone(), target_timesteps.clone()))

                class Out:
                    velocity = torch.zeros_like(latents)

                return Out()

            def parameters(self):
                return iter(())

        teacher = _AnchorTeacher(torch.zeros(1, NUM_LATENTS, HIDDEN))
        for seed in range(5):
            meanflow_distillation_loss(
                RecordingStub(),
                teacher,
                torch.randn(16, NUM_LATENTS, HIDDEN),
                num_timesteps=4,
                time_scale=1000.0,
                generator=torch.Generator().manual_seed(seed),
            )

        assert seen, "the student was never called"
        for t_scaled, r_scaled in seen:
            assert bool((r_scaled <= t_scaled + 1e-5).all()), "sampled r exceeded t"


class TestFewStepSampling:
    def test_a_perfect_average_velocity_lands_in_one_step(self):
        """eq. 16 from tau=1 to tau'=0 with the straight-flow u lands
        exactly on the anchor: z + (0 - 1) * (z - anchor)/1 = anchor.  Any
        sign slip in the (tau' - tau) coefficient moves AWAY from data."""
        from unturtle.models.latent import sample_meanflow_latents

        anchor = torch.randn(1, NUM_LATENTS, HIDDEN)

        class StraightStub:
            config = _flow_config()

            def __call__(self, latents, timesteps, target_timesteps):
                t = (timesteps / 1000.0).view(-1, 1, 1)

                class Out:
                    velocity = (latents - anchor.expand_as(latents)) / t

                return Out()

            def parameters(self):
                return iter((torch.zeros(1),))

        for steps in (1, 3):
            final = sample_meanflow_latents(
                StraightStub(),
                batch_size=2,
                num_steps=steps,
                generator=torch.Generator().manual_seed(4),
            )
            assert torch.allclose(final, anchor.expand_as(final), atol=1e-5), (
                f"{steps}-step sampling missed the anchor"
            )


@pytest.mark.slow
def test_distillation_learns_the_average_velocity_end_to_end():
    """End-to-end: slice 1's pipeline (pretrain -> AE -> prior), then
    MeanFlow-distill the prior and measure the student against the ground
    truth — the teacher's own 4-step rollout displacement.

    **What is asserted, and what deliberately is not.**  The mechanism must
    work: the student's u(z, 1, 0) error against the rolled-out teacher
    displacement falls well below half its untrained value (measured
    1.11 -> 0.46 at these settings, monotone across checkpoints), and
    1-step distilled guidance stays in its measured stability band
    (30-32/64 intact rows at lr 3e-4; a plain floor of 24 sits below it).
    Teacher PARITY is deliberately NOT asserted: at this scale the teacher's
    own 1-step x0 jump already reaches 44/64 (the trajectory is short and
    nearly straight), so distillation's value-add is structurally small,
    and the self-referential JVP bootstrap oscillates without the paper's
    regime (25k steps at lr 5e-5, self-conditioning, a 200-step teacher) —
    measured: higher lr (2e-3) swings 6-54/64 across checkpoints, and
    Geng-style adaptive weighting diverges here.  Closing that gap is
    full-scale work, not what this slice pins."""
    from unturtle.models.latent import (
        LaDiffModel,
        MeanFlowDenoiser,
        flowlm_loss,
        latent_autoencoder_loss,
        meanflow_distillation_loss,
        sample_meanflow_latents,
    )
    from unturtle.processes.continuous_flow import ContinuousFlowProcess

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    from unturtle.models.latent import LaDiffConfig

    config = LaDiffConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=LENGTH,
        mask_token_id=MASK_ID,
        num_latents=NUM_LATENTS,
        num_timesteps=4,
    )
    model = LaDiffModel(config)
    patterns = torch.stack(
        [(torch.arange(LENGTH) * k + k) % (VOCAB - 1) for k in (1, 3, 5, 7)]
    )
    pattern_set = {tuple(row.tolist()) for row in patterns}

    def batch(n):
        picks = torch.randint(0, 4, (n,), generator=generator)
        return patterns[picks]

    # Phases A-C: pretrain, AE finetune, prior (compressed from slice 1).
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=2e-3)
    for _ in range(300):
        losses = latent_autoencoder_loss(
            model.codec, batch(32), latent_dropout=1.0, generator=generator
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
    optimizer = torch.optim.AdamW(
        list(model.codec.encoder.parameters()) + list(model.decoder.parameters()),
        lr=1e-3,
    )
    for _ in range(400):
        losses = latent_autoencoder_loss(
            model.codec, batch(32), latent_dropout=0.1, generator=generator
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
    process = ContinuousFlowProcess(num_timesteps=4)
    optimizer = torch.optim.AdamW(model.prior.parameters(), lr=2e-3)
    for _ in range(400):
        with torch.no_grad():
            clean = model.codec.encode(batch(32))
        out = process({"latents": clean}, generator=generator)
        pred = model.prior(
            out.model_inputs["latents"], timesteps=out.model_inputs["timesteps"]
        ).prediction
        losses = flowlm_loss(pred, out.objective_inputs["target_latents"])
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
    model.eval()

    # Ground truth: the teacher's OWN 4-step rollout displacement from z_1.
    def teacher_endpoint(z_start):
        z = z_start.clone()
        for k in range(4, 0, -1):
            t = k / 4
            pred = model.prior(
                z, timesteps=torch.full((z.shape[0],), t * 1000.0)
            ).prediction
            z = (1 - (1 / 4) / t) * z + ((1 / 4) / t) * pred
        return z

    probe = torch.randn(
        64, NUM_LATENTS, HIDDEN, generator=torch.Generator().manual_seed(99)
    )
    with torch.no_grad():
        true_displacement = probe - teacher_endpoint(probe)

    def probe_error(student):
        with torch.no_grad():
            u = student(
                probe,
                timesteps=torch.full((64,), 1000.0),
                target_timesteps=torch.zeros(64),
            ).velocity
        return float(((u - true_displacement) ** 2).mean())

    # Phase D: distill.
    student = MeanFlowDenoiser(_flow_config(num_timesteps=4)).train()
    initial_error = probe_error(student.eval())
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    for _ in range(1500):
        with torch.no_grad():
            clean = model.codec.encode(batch(32))
        losses = meanflow_distillation_loss(
            student,
            model.prior,
            clean,
            num_timesteps=4,
            time_scale=1000.0,
            pure_fm_fraction=0.25,
            generator=generator,
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
    student.eval()

    final_error = probe_error(student)
    distilled_latents = sample_meanflow_latents(
        student, batch_size=64, num_steps=1, generator=torch.Generator().manual_seed(13)
    )
    distilled_ids = model.sample_discrete(
        latents=distilled_latents,
        batch_size=64,
        num_discrete_steps=2,
        generator=torch.Generator().manual_seed(14),
    )
    distilled_intact = sum(tuple(row.tolist()) in pattern_set for row in distilled_ids)

    print(
        f"\nu-error {initial_error:.3f} -> {final_error:.3f}; "
        f"1-step distilled intact {distilled_intact}/64"
    )
    assert final_error < initial_error / 2, (
        f"distillation did not learn the average velocity: "
        f"{initial_error:.3f} -> {final_error:.3f}"
    )
    assert distilled_intact >= 24, (
        f"one-step distilled guidance fell out of its measured band: "
        f"{distilled_intact}/64"
    )
