"""
DiLaDiff slice 1 (#117): trainable codec + latent prior + latent-guided decode.

The e2e inverts #116's measured failure.  There, an unconditional continuous
model on a 4-pattern corpus produced pattern tokens per position but
mode-inconsistent rows — the exact phenomenon DiLaDiff's eqs. 9-10 claim to
fix: conditioned on a latent that carries the token correlations, the
token-wise factorized posterior *truly* factorizes, so aggressive parallel
decoding stays coherent.  The tiny test is therefore:

- unconditional parallel masked decode on the same corpus → mode-mixed rows
  (the control),
- decode conditioned on the ENCODER's latent → the right pattern back
  (reconstruction),
- decode conditioned on a PRIOR-sampled latent → mode-consistent rows.

Deviations from the paper, recorded: additive zero-init latent conditioning
instead of cross-attention + zero-init conv; a Perceiver-lite encoder over
the decoder's own embeddings instead of BERT features; the prior reuses the
linear-interpolation path from #116 instead of tanh-logSNR VP.  ELBO
weighting is the paper's own simplification (constant -1 → plain mean CE on
masked positions).
"""

import pytest
import torch

VOCAB = 16
HIDDEN = 32
LENGTH = 8
MASK_ID = VOCAB - 1  # reserve the last id as [MASK]
NUM_LATENTS = 2


def _config(**overrides):
    from unturtle.models.latent import LaDiffConfig

    defaults = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=LENGTH,
        mask_token_id=MASK_ID,
        num_latents=NUM_LATENTS,
        num_timesteps=4,
    )
    defaults.update(overrides)
    return LaDiffConfig(**defaults)


def _patterns():
    return torch.stack(
        [(torch.arange(LENGTH) * k + k) % (VOCAB - 1) for k in (1, 3, 5, 7)]
    )


class TestTheLatentConditionedDecoder:
    def test_it_denoises_with_and_without_a_latent(self):
        from unturtle.models.latent import LatentConditionedMDLM

        decoder = LatentConditionedMDLM(_config()).eval()
        ids = torch.randint(0, VOCAB - 1, (2, LENGTH))
        z = torch.randn(2, NUM_LATENTS, HIDDEN)

        with torch.no_grad():
            unconditional = decoder(input_ids=ids).logits
            conditional = decoder(input_ids=ids, latents=z).logits

        assert unconditional.shape == (2, LENGTH, VOCAB)
        assert conditional.shape == (2, LENGTH, VOCAB)

    def test_conditioning_starts_inert_so_pretraining_is_preserved(self):
        """The latent projection is zero-initialized (the paper wraps its
        cross-attention in zero-init convolutions for the same reason): at
        AE-finetune start the decoder IS the pretrained masked dLLM, and the
        latent channel opens up only as training moves the projection."""
        from unturtle.models.latent import LatentConditionedMDLM

        decoder = LatentConditionedMDLM(_config()).eval()
        ids = torch.randint(0, VOCAB - 1, (2, LENGTH))

        with torch.no_grad():
            without = decoder(input_ids=ids).logits
            with_latent = decoder(
                input_ids=ids, latents=torch.randn(2, NUM_LATENTS, HIDDEN)
            ).logits

        assert torch.allclose(without, with_latent, atol=1e-6), (
            "a random latent moved logits at init; the conditioning is not "
            "zero-initialized and AE finetuning would start by destroying "
            "the pretrained decoder"
        )

    def test_an_open_channel_makes_latents_observable(self):
        """The liveness half of the zero-init pair: once the projection has
        moved (mid-training), different latents must produce different
        logits — a severed conditioning path would keep the inertness test
        green forever while the e2e trains an unconditional model."""
        from unturtle.models.latent import LatentConditionedMDLM

        decoder = LatentConditionedMDLM(_config()).eval()
        with torch.no_grad():
            decoder.latent_proj.weight.normal_(std=0.05)
        ids = torch.randint(0, VOCAB - 1, (2, LENGTH))

        with torch.no_grad():
            first = decoder(
                input_ids=ids, latents=torch.zeros(2, NUM_LATENTS, HIDDEN)
            ).logits
            second = decoder(
                input_ids=ids, latents=torch.ones(2, NUM_LATENTS, HIDDEN)
            ).logits

        assert not torch.allclose(first, second), (
            "latents do not reach the decoder through an opened channel"
        )


class TestTheTrainableCodec:
    def test_encode_produces_a_latent_sequence_and_decode_conditions_on_it(self):
        from unturtle.models.latent import LatentAutoencoderCodec, LatentConditionedMDLM

        decoder = LatentConditionedMDLM(_config())
        codec = LatentAutoencoderCodec(_config(), decoder)
        ids = torch.randint(0, VOCAB - 1, (2, LENGTH))

        z = codec.encode(ids)
        masked = torch.full_like(ids, MASK_ID)
        logits = codec.decode(z, input_ids=masked)

        assert z.shape == (2, NUM_LATENTS, HIDDEN)
        assert z.requires_grad, "the encoder must be trainable end-to-end"
        assert logits.shape == (2, LENGTH, VOCAB)
        assert codec.trainable is True

    def test_it_satisfies_the_codec_protocol(self):
        from unturtle.models.latent import (
            Codec,
            LatentAutoencoderCodec,
            LatentConditionedMDLM,
        )

        decoder = LatentConditionedMDLM(_config())

        assert isinstance(LatentAutoencoderCodec(_config(), decoder), Codec)


class TestTheAutoencoderObjective:
    def _setup(self):
        from unturtle.models.latent import LatentAutoencoderCodec, LatentConditionedMDLM

        config = _config()
        decoder = LatentConditionedMDLM(config)
        codec = LatentAutoencoderCodec(config, decoder)
        ids = torch.randint(0, VOCAB - 1, (4, LENGTH))
        return codec, ids

    def test_the_loss_is_named_and_masked_positions_only(self):
        """Constant (-1) ELBO weighting == plain mean CE over MASKED
        positions.  Computing it over every position would let the model
        earn loss by copying visible tokens — supervision the objective must
        not grant."""
        from unturtle.models.latent import latent_autoencoder_loss

        codec, ids = self._setup()

        losses = latent_autoencoder_loss(
            codec, ids, generator=torch.Generator().manual_seed(0)
        )

        assert "reconstruction_ce" in losses and "total" in losses
        assert losses["reconstruction_ce"].requires_grad

    def test_the_ce_is_computed_over_masked_positions_exactly(self):
        """Exact-equality pin using a stub codec that 'predicts' the
        corrupted input verbatim: visible positions would contribute ~0 CE,
        so averaging them in dilutes the loss — the masked-only value is
        recomputed by replaying the seeded mask draw and must match to the
        bit."""
        from types import SimpleNamespace

        import torch.nn.functional as F

        from unturtle.models.latent import latent_autoencoder_loss

        class _EchoCodec:
            config = SimpleNamespace(mask_token_id=MASK_ID)

            def encode(self, ids, **_):
                return torch.zeros(ids.shape[0], 1, 4, requires_grad=True)

            def decode(self, latents, input_ids, **_):
                return F.one_hot(input_ids, VOCAB).float() * 9.0

        ids = torch.randint(0, VOCAB - 1, (4, LENGTH))

        losses = latent_autoencoder_loss(
            _EchoCodec(),
            ids,
            latent_dropout=0.0,
            generator=torch.Generator().manual_seed(5),
        )

        # Replay the loss's own seeded corruption draw.
        replay = torch.Generator().manual_seed(5)
        t = torch.rand(4, 1, generator=replay).clamp_min(1e-3)
        masked = torch.rand(4, LENGTH, generator=replay) < t
        dead = ~masked.any(dim=1)
        if bool(dead.any()):
            masked[dead, 0] = True
        corrupted = ids.masked_fill(masked, MASK_ID)
        expected = F.cross_entropy(
            (F.one_hot(corrupted, VOCAB).float() * 9.0)[masked], ids[masked]
        )
        assert torch.equal(losses["reconstruction_ce"], expected), (
            "the CE does not match the masked-positions-only value; visible "
            "positions are leaking into the objective"
        )

    def test_full_latent_dropout_cuts_the_encoder_out_of_the_graph(self):
        """`p_dropout^z = 1` replaces the latent with pure noise every time —
        the mechanism that preserves an unconditional decoding mode.  If any
        encoder gradient survives at p=1, the dropout is not actually
        replacing the latent."""
        from unturtle.models.latent import latent_autoencoder_loss

        codec, ids = self._setup()

        losses = latent_autoencoder_loss(
            codec,
            ids,
            latent_dropout=1.0,
            generator=torch.Generator().manual_seed(1),
        )
        losses["total"].backward()

        encoder_grads = [
            p.grad for p in codec.encoder.parameters() if p.grad is not None
        ]
        assert not any(bool(g.abs().sum() > 0) for g in encoder_grads), (
            "encoder received gradient under full latent dropout"
        )

    def test_without_dropout_the_encoder_is_in_the_graph(self):
        """At init the zero-initialized latent projection blocks gradient
        INTO the encoder too (grad w.r.t. latents is Wᵀ·upstream = 0) — that
        is the inertness working as designed.  The property under test is
        mid-training: once the channel has opened, encoder gradients flow."""
        from unturtle.models.latent import latent_autoencoder_loss

        codec, ids = self._setup()
        with torch.no_grad():
            codec.decoder.latent_proj.weight.normal_(std=0.05)

        losses = latent_autoencoder_loss(
            codec,
            ids,
            latent_dropout=0.0,
            generator=torch.Generator().manual_seed(2),
        )
        losses["total"].backward()

        total = sum(
            float(p.grad.abs().sum())
            for p in codec.encoder.parameters()
            if p.grad is not None
        )
        assert total > 0, "no gradient reached the encoder without dropout"

    def test_a_seeded_generator_reproduces_the_loss(self):
        from unturtle.models.latent import latent_autoencoder_loss

        codec, ids = self._setup()

        first = latent_autoencoder_loss(
            codec, ids, latent_noise_std=0.1, generator=torch.Generator().manual_seed(3)
        )
        second = latent_autoencoder_loss(
            codec, ids, latent_noise_std=0.1, generator=torch.Generator().manual_seed(3)
        )

        assert torch.equal(first["total"], second["total"])


class TestTheGenerationFamily:
    def test_ladiff_registers_as_the_latent_guided_family(self):
        from unturtle.models.generation.sampler import find_algorithm

        entry = find_algorithm("ladiff")

        assert entry is not None
        assert entry.family == "latent_guided"
        assert entry.flags == {}

    def test_a_masked_model_cannot_be_asked_for_ladiff(self):
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
            resolve_algorithm("ladiff", masked, bd3lm_requested=False)

    def test_generation_dispatches_and_returns_token_rows(self):
        from unturtle.models.latent import LaDiffModel

        model = LaDiffModel(_config()).eval()

        ids = model.generate(
            algorithm="ladiff",
            batch_size=2,
            num_latent_steps=2,
            num_discrete_steps=2,
            generator=torch.Generator().manual_seed(4),
        )

        assert ids.shape == (2, LENGTH)
        assert bool((ids >= 0).all()) and bool((ids < MASK_ID).all()), (
            "generation left mask tokens or out-of-vocab ids in the output"
        )

    def test_the_mask_token_is_never_emitted_even_when_it_dominates(self):
        """Deterministic pin of the re-emission guard: bias the head so the
        mask token is the argmax everywhere — the sampler must still commit
        only real tokens (a decode that can emit [MASK] never terminates
        meaningfully)."""
        from unturtle.models.latent import LaDiffModel

        model = LaDiffModel(_config()).eval()
        with torch.no_grad():
            model.decoder.lm_head.bias[MASK_ID] = 25.0

        ids = model.sample_discrete(
            latents=None,
            batch_size=4,
            num_discrete_steps=2,
            generator=torch.Generator().manual_seed(6),
        )

        assert bool((ids != MASK_ID).all()), "the sampler re-emitted [MASK]"


class TestSaveReload:
    def test_the_bundle_round_trips(self, tmp_path):
        from unturtle.models.latent import LaDiffModel

        model = LaDiffModel(_config()).eval()
        model.save_pretrained(tmp_path / "ladiff")
        reloaded = LaDiffModel.from_pretrained(tmp_path / "ladiff").eval()

        ids = torch.randint(0, VOCAB - 1, (1, LENGTH))
        z = torch.randn(1, NUM_LATENTS, HIDDEN)
        with torch.no_grad():
            assert torch.allclose(
                model.decoder(input_ids=ids, latents=z).logits,
                reloaded.decoder(input_ids=ids, latents=z).logits,
                atol=1e-6,
            )


@pytest.mark.slow
def test_the_latent_guides_parallel_decoding_out_of_mode_mixing():
    """The slice-1 claim end-to-end, on the corpus that defeated #116.

    Aggressive parallel masked decoding samples each position from its
    marginal; on a 4-pattern corpus the marginals mix modes, so
    unconditional rows come back mode-inconsistent (the control below
    measures exactly that).  DiLaDiff's eqs. 9-10: conditioned on a latent
    carrying the correlations, the factorized posterior is faithful — the
    encoder's latent should reconstruct the right pattern, and a
    PRIOR-sampled latent should yield mode-CONSISTENT rows.

    The latent-vs-discrete cost split is measured and printed (a fact, not
    an assertion — #117's evaluation contract).
    """
    import time

    from unturtle.models.latent import (
        FlowLMDenoiser,
        LaDiffModel,
        flowlm_loss,
        latent_autoencoder_loss,
    )
    from unturtle.processes.continuous_flow import ContinuousFlowProcess

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    config = _config()
    model = LaDiffModel(config)
    patterns = _patterns()
    pattern_set = {tuple(row.tolist()) for row in patterns}

    def batch(n):
        picks = torch.randint(0, 4, (n,), generator=generator)
        return patterns[picks]

    # --- Phase A: pretrain the decoder as a plain masked dLLM (no latent).
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=2e-3)
    for _ in range(300):
        ids = batch(32)
        losses = latent_autoencoder_loss(
            model.codec, ids, latent_dropout=1.0, generator=generator
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

    # Control: unconditional aggressive parallel decode mixes modes.
    unconditional = model.sample_discrete(
        latents=None,
        batch_size=64,
        num_discrete_steps=2,
        generator=torch.Generator().manual_seed(11),
    )
    unconditional_intact = sum(
        tuple(row.tolist()) in pattern_set for row in unconditional
    )

    # --- Phase B: AE finetune — encoder + latent channel, decoder continues.
    optimizer = torch.optim.AdamW(
        list(model.codec.encoder.parameters()) + list(model.decoder.parameters()),
        lr=1e-3,
    )
    for _ in range(400):
        ids = batch(32)
        losses = latent_autoencoder_loss(
            model.codec, ids, latent_dropout=0.1, generator=generator
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        ids = patterns.repeat(4, 1)
        reconstructed = model.sample_discrete(
            latents=model.codec.encode(ids),
            batch_size=ids.shape[0],
            num_discrete_steps=2,
            generator=torch.Generator().manual_seed(12),
        )
    reconstruction_hits = int((reconstructed == ids).all(dim=1).sum())
    assert reconstruction_hits >= 12, (
        f"encoder latents reconstructed only {reconstruction_hits}/16 rows"
    )

    # --- Phase C: train the prior on encoder latents; sample; guided decode.
    model.train()
    process = ContinuousFlowProcess(num_timesteps=config.num_timesteps)
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
    start = time.perf_counter()
    latents = model.sample_prior_latents(
        batch_size=64, num_latent_steps=4, generator=torch.Generator().manual_seed(13)
    )
    latent_seconds = time.perf_counter() - start
    start = time.perf_counter()
    guided = model.sample_discrete(
        latents=latents,
        batch_size=64,
        num_discrete_steps=2,
        generator=torch.Generator().manual_seed(14),
    )
    discrete_seconds = time.perf_counter() - start
    guided_intact = sum(tuple(row.tolist()) in pattern_set for row in guided)

    print(
        f"\ncost split: latent {latent_seconds * 1e3:.1f}ms vs discrete "
        f"{discrete_seconds * 1e3:.1f}ms; intact rows unconditional "
        f"{unconditional_intact}/64 vs guided {guided_intact}/64"
    )
    assert guided_intact >= 40, (
        f"prior-guided decoding produced only {guided_intact}/64 intact rows"
    )
    assert guided_intact > unconditional_intact + 10, (
        f"guidance did not beat the unconditional control: "
        f"{guided_intact} vs {unconditional_intact}"
    )
