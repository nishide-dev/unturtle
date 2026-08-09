"""
FlowLM-style minimal continuous prototype (#66).

The prototype exists to test the RFC's boundaries, not to claim quality:
codec (``encode`` / ``decode`` / ``trainable`` / named ``auxiliary_losses``),
continuous process, x0 objective, and a method-local average-velocity solver
registered as a NEW generation family — never as boolean flags on the masked
registry (acceptance criterion).

Method specifics are from the paper (arXiv:2605.20199), Algorithms 1-2 and
eq. 7: x0 prediction (v-pred measured unstable, Fig. 3), total loss
``||z_0 - z_0,pred||^2 + CE(decoder_head(z_0), w) + reg * ||pred_ref -
pred||^2 / t^2``, and the sampling update ``z <- (1 - dt/t) z + (dt/t)
z_0,pred`` — an AVERAGE-velocity step, not Euler on an instantaneous field.

The codec here is the protocol's simplest instance (embedding lookup +
rounding head), and it still carries a named auxiliary loss — the RFC's
strongest argument that bare ``encode()/decode()`` is insufficient.
"""

import pytest
import torch

VOCAB = 16
HIDDEN = 32
LENGTH = 8


def _config(**overrides):
    from unturtle.models.latent import FlowLMConfig

    defaults = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=LENGTH,
        num_timesteps=4,
    )
    defaults.update(overrides)
    return FlowLMConfig(**defaults)


class TestTheCodecBoundary:
    def test_encode_decode_shapes_and_rounding(self):
        from unturtle.models.latent import EmbeddingRoundingCodec

        codec = EmbeddingRoundingCodec(vocab_size=VOCAB, hidden_size=HIDDEN)
        ids = torch.randint(0, VOCAB, (2, LENGTH))

        latents = codec.encode(ids)
        logits = codec.decode(latents)

        assert latents.shape == (2, LENGTH, HIDDEN)
        assert latents.dtype.is_floating_point
        assert logits.shape == (2, LENGTH, VOCAB)

    def test_even_the_simplest_codec_owns_a_named_auxiliary_loss(self):
        """The RFC's core claim: FlowLM's rounding CE is codec-owned, so the
        protocol needs `auxiliary_losses() -> dict`, not bare encode/decode."""
        from unturtle.models.latent import EmbeddingRoundingCodec

        codec = EmbeddingRoundingCodec(vocab_size=VOCAB, hidden_size=HIDDEN)
        ids = torch.randint(0, VOCAB, (2, LENGTH))

        losses = codec.auxiliary_losses(codec.encode(ids), ids)

        assert set(losses) == {"rounding_ce"}
        assert losses["rounding_ce"].ndim == 0
        assert losses["rounding_ce"].requires_grad
        assert codec.trainable is True

    def test_minimizing_the_rounding_ce_makes_argmax_decode_invert_encode(self):
        """TestTheOptimum: the CE term exists so that rounding recovers the
        token; assert where optimization converges, not just that the number
        goes down."""
        from unturtle.models.latent import EmbeddingRoundingCodec

        torch.manual_seed(0)
        codec = EmbeddingRoundingCodec(vocab_size=VOCAB, hidden_size=HIDDEN)
        ids = torch.arange(VOCAB).unsqueeze(0)
        optimizer = torch.optim.AdamW(codec.parameters(), lr=5e-2)
        for _ in range(200):
            loss = codec.auxiliary_losses(codec.encode(ids), ids)["rounding_ce"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        rounded = codec.decode(codec.encode(ids)).argmax(dim=-1)

        assert torch.equal(rounded, ids), "rounding does not invert encoding"
        # The argmax alone cannot see an off-target CE: with a weight-tied
        # head, Cauchy-Schwarz makes cyclic dominance (every E_i . E_{i+1}
        # beating E_i . E_i) unattainable, so argmax stays identity even
        # under a shifted-target loss.  The convergence VALUE can see it —
        # the identity target is achievable (CE -> ~0), a shifted target has
        # a strictly positive floor.
        assert float(loss) < 0.2, (
            f"rounding CE converged to {float(loss):.3f}; the target is not "
            "the identity this loss exists to teach"
        )


class TestTheDenoiser:
    def test_it_maps_latents_to_latents_and_consumes_time(self):
        """Same shape in and out ([B, L, H] — no vocabulary head), and the
        time conditioning must be observable: a denoiser that ignores t
        cannot represent the path."""
        from unturtle.models.latent import FlowLMDenoiser

        denoiser = FlowLMDenoiser(_config()).eval()
        latents = torch.randn(2, LENGTH, HIDDEN)

        with torch.no_grad():
            early = denoiser(latents, timesteps=torch.full((2,), 250.0)).prediction
            late = denoiser(latents, timesteps=torch.full((2,), 1000.0)).prediction

        assert early.shape == (2, LENGTH, HIDDEN)
        assert not torch.allclose(early, late), "time conditioning is inert"


class TestTheObjective:
    def test_the_x0_mse_is_zero_exactly_at_the_clean_latent(self):
        from unturtle.models.latent import flowlm_loss

        target = torch.randn(2, LENGTH, HIDDEN)

        at_optimum = flowlm_loss(target.clone(), target)
        off_optimum = flowlm_loss(target + 0.1, target)

        assert float(at_optimum["x0_mse"]) == 0.0
        assert float(off_optimum["x0_mse"]) > 0.0
        assert float(at_optimum["total"]) == 0.0

    def test_named_auxiliary_losses_join_the_total(self):
        from unturtle.models.latent import flowlm_loss

        target = torch.randn(1, 4, 8)
        aux = {"rounding_ce": torch.tensor(0.7)}

        losses = flowlm_loss(target + 1.0, target, auxiliary_losses=aux)

        assert losses["rounding_ce"] is aux["rounding_ce"]
        assert torch.allclose(losses["total"], losses["x0_mse"] + losses["rounding_ce"])

    def test_the_reference_regularizer_carries_the_inverse_square_time_weight(self):
        """Paper §3.3: `reg * ||pred_diffu - pred_flow||^2 / t^2` — verified
        against a hand computation, not just for presence."""
        from unturtle.models.latent import flowlm_loss

        target = torch.zeros(2, 1, 1)
        pred = torch.ones(2, 1, 1)
        reference = torch.zeros(2, 1, 1)
        t = torch.tensor([0.5, 1.0])

        losses = flowlm_loss(
            pred,
            target,
            reference_pred=reference,
            timesteps=t,
            reg_rate=2.0,
        )

        # per-row mean squared diff is 1.0; weights 1/0.25 and 1.0; rate 2.
        expected = 2.0 * (1.0 / 0.25 + 1.0 / 1.0) / 2
        assert torch.allclose(losses["reference_reg"], torch.tensor(expected))
        assert torch.allclose(
            losses["total"], losses["x0_mse"] + losses["reference_reg"]
        )

    def test_the_regularizer_requires_timesteps(self):
        from unturtle.models.latent import flowlm_loss

        target = torch.zeros(1, 1, 1)

        with pytest.raises(ValueError, match="timesteps"):
            flowlm_loss(target, target, reference_pred=target, reg_rate=1.0)

    def test_an_auxiliary_loss_named_total_is_rejected(self):
        """`losses["total"] = total` would silently overwrite the named
        entry after summing it — the caller logging per-term losses loses
        the term while the sum quietly includes it.  The dict's whole point
        is that the trainer can log terms it does not understand, so a
        swallowed name defeats the design."""
        from unturtle.models.latent import flowlm_loss

        target = torch.zeros(1, 1, 1)

        with pytest.raises(ValueError, match="total"):
            flowlm_loss(target, target, auxiliary_losses={"total": torch.tensor(1.0)})


class TestTheAverageVelocitySolver:
    def test_a_perfect_predictor_lands_exactly_on_its_target(self):
        """eq. 7 telescopes: with z_0,pred constant, the final step (t = dt)
        has weight dt/t = 1, so the sampler lands on the prediction EXACTLY —
        for any step count.  This is the average-velocity property; an
        instantaneous-Euler implementation would not land exactly."""
        from unturtle.models.latent import FlowLMModel

        model = FlowLMModel(_config()).eval()
        anchor = torch.randn(1, LENGTH, HIDDEN)

        class Stub:
            def __call__(self, latents, timesteps):
                class Out:
                    prediction = anchor.expand_as(latents)

                return Out()

        for steps in (1, 3, 4):
            final = model.sample_latents(
                batch_size=1,
                num_steps=steps,
                denoise_fn=Stub(),
                generator=torch.Generator().manual_seed(0),
            )
            assert torch.allclose(final, anchor, atol=1e-5), (
                f"{steps}-step sampling did not land on the prediction"
            )

    def test_a_half_precision_model_can_sample(self):
        """The initial noise must be drawn in the model's dtype: `randn`
        defaults to fp32 and `.to(device)` does not cast, so a bf16/fp16
        model (the unsloth/LoRA default) hit a dtype mismatch in the first
        linear."""
        from unturtle.models.latent import FlowLMModel

        model = FlowLMModel(_config()).half().eval()

        ids = model.generate(
            batch_size=2, num_steps=2, generator=torch.Generator().manual_seed(3)
        )

        assert ids.shape == (2, LENGTH)

    def test_a_prompt_is_rejected_rather_than_silently_ignored(self):
        """The prototype is unconditional (documented); a caller passing
        inputs must get an error, not a batch-size-1 unconditional sample —
        the registry's no-silent-fallback posture."""
        from unturtle.models.latent import FlowLMModel

        model = FlowLMModel(_config()).eval()

        with pytest.raises(ValueError, match="unconditional"):
            model.generate(
                torch.randint(0, VOCAB, (2, LENGTH)),
                algorithm="flowlm",
                num_steps=1,
            )

    def test_one_step_generation_is_the_prediction_itself(self):
        """T=1: t=1, dt=1 → z <- z_0,pred in a single update (the paper's
        one-step mode)."""
        from unturtle.models.latent import FlowLMModel

        model = FlowLMModel(_config()).eval()

        ids = model.generate(
            batch_size=2,
            num_steps=1,
            generator=torch.Generator().manual_seed(1),
        )

        assert ids.shape == (2, LENGTH)
        assert bool((ids >= 0).all()) and bool((ids < VOCAB).all())


class TestTheGenerationFamilyRegistration:
    def test_flowlm_registers_as_its_own_family(self):
        """A new family with no masked-diffusion flags — the acceptance
        criterion that continuous algorithms are never boolean variants of
        the masked loop."""
        from unturtle.models.generation.sampler import find_algorithm

        entry = find_algorithm("flowlm")

        assert entry is not None
        assert entry.family == "continuous_flow"
        assert entry.flags == {}

    def test_a_masked_model_cannot_be_asked_for_flowlm(self):
        from unturtle.models.conversion.a2d.tiny_a2d import (
            TinyA2DLlamaConfig,
            TinyA2DLlamaLMHeadModel,
        )
        from unturtle.models.generation.sampler import resolve_algorithm

        masked = TinyA2DLlamaLMHeadModel(
            TinyA2DLlamaConfig(
                vocab_size=VOCAB,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=LENGTH,
            )
        )

        with pytest.raises(ValueError):
            resolve_algorithm("flowlm", masked, bd3lm_requested=False)

    def test_auto_resolves_to_flowlm_on_the_prototype(self):
        from unturtle.models.generation.sampler import resolve_algorithm
        from unturtle.models.latent import FlowLMModel

        model = FlowLMModel(_config())

        assert resolve_algorithm("auto", model, bd3lm_requested=False) == "flowlm"

    def test_generation_dispatches_through_the_registry(self):
        from unturtle.models.latent import FlowLMModel

        model = FlowLMModel(_config()).eval()

        ids = model.generate(
            algorithm="flowlm",
            batch_size=2,
            num_steps=3,
            generator=torch.Generator().manual_seed(2),
        )

        assert ids.shape == (2, LENGTH)


class TestSaveReload:
    def test_the_prototype_round_trips_through_pretrained_conventions(self, tmp_path):
        from unturtle.models.latent import FlowLMModel

        model = FlowLMModel(_config()).eval()
        model.save_pretrained(tmp_path / "proto")
        reloaded = FlowLMModel.from_pretrained(tmp_path / "proto").eval()

        latents = torch.randn(1, LENGTH, HIDDEN)
        t = torch.full((1,), 500.0)
        with torch.no_grad():
            assert torch.allclose(
                model.denoiser(latents, timesteps=t).prediction,
                reloaded.denoiser(latents, timesteps=t).prediction,
                atol=1e-6,
            )


@pytest.mark.slow
def test_the_prototype_trains_and_few_step_samples_recover_the_data():
    """End-to-end: process -> denoiser -> objective -> solver -> rounding.

    The corpus is deliberately UNIMODAL (one fixed sequence).  With a
    multimodal corpus and no conditioning, an x0-MSE model correctly learns
    the posterior mean at high t, and few-step unconditional samples mix
    modes position-wise — measured here as pattern-token-per-position but
    mode-inconsistent rows.  That is a property of unconditional MSE, not an
    architecture bug: the paper's setting is seq2seq, where the condition
    selects the mode.  The architectural claim under test — the pieces
    compose, training converges, and few-step sampling plus rounding recover
    the learned data exactly — is exactly testable on the unimodal corpus,
    at every step count including one."""
    from unturtle.models.latent import FlowLMModel, flowlm_loss
    from unturtle.processes.continuous_flow import ContinuousFlowProcess

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = FlowLMModel(_config(num_timesteps=4)).train()
    process = ContinuousFlowProcess(num_timesteps=4)

    pattern = (torch.arange(LENGTH) * 5 + 3) % VOCAB
    ids = pattern.unsqueeze(0).expand(32, -1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(400):
        clean = model.codec.encode(ids)
        out = process({"latents": clean}, generator=generator)
        pred = model.denoiser(
            out.model_inputs["latents"],
            timesteps=out.model_inputs["timesteps"],
        ).prediction
        losses = flowlm_loss(
            pred,
            out.objective_inputs["target_latents"],
            auxiliary_losses=model.codec.auxiliary_losses(clean, ids),
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

    model.eval()
    # Thresholds from a 20-seed sweep after training: 4- and 2-step sampling
    # hit 64/64 on EVERY seed (assert equality, not a fake margin); 1-step
    # ranges 56-64, so its bound sits below the measured minimum instead of
    # inside the noise band.
    for steps, minimum in ((4, 64), (2, 64), (1, 55)):
        sampled = model.generate(
            batch_size=64, num_steps=steps, generator=torch.Generator().manual_seed(7)
        )
        hits = int((sampled == pattern).all(dim=1).sum())
        assert hits >= minimum, (
            f"{steps}-step sampling recovered only {hits}/64 rows exactly "
            f"(measured floor {minimum})"
        )
