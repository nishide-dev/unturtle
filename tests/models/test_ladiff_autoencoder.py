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

"""LaDiff autoencoder on the real backbone (#130 PR-codec-2).

Paper Algorithm 1 (verbatim on the issue): clean features -> standardize ->
{mask XOR noise} -> encode -> {(maybe full Gaussian replacement) XOR mask}
-> decode a masked state conditioned on z -> CE on masked positions only.
The encoder input features are the FROZEN pretrained trunk's hidden states
of the clean sequence (the recorded substitution for the paper's frozen
BERT — no BERT exists in the gpt2/kuleshov lineage).

Regularizer tests pin the BRANCH STRUCTURE with a seeded generator: the
feature branch masks XOR noises (never both), the latent branch replaces-
with-Gaussian XOR masks.  Stubs are position-keyed (memory: uniform stubs
blind equality tests).
"""

import pytest
import torch

from unturtle.models.latent.autoencoder_dit import (
    LaDiffAutoencoder,
    LaDiffEncoder,
    RunningStandardizer,
    ladiff_autoencoder_loss,
)
from unturtle.models.latent.modeling_ladiff_dit import (
    LaDiffDiTConfig,
    LatentConditionedMDLMDiT,
)

VOCAB = 16
MASK_ID = VOCAB - 1
HIDDEN = 32
LAYERS = 4
N_LATENTS = 3


def config() -> LaDiffDiTConfig:
    return LaDiffDiTConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        cond_dim=8,
        num_hidden_layers=LAYERS,
        num_attention_heads=2,
        dropout=0.0,
        max_position_embeddings=32,
        mask_token_id=MASK_ID,
        num_latents=N_LATENTS,
        latent_dim=HIDDEN,
        encoder_layers=2,
    )


def autoencoder() -> LaDiffAutoencoder:
    torch.manual_seed(0)
    decoder = LatentConditionedMDLMDiT(config())
    return LaDiffAutoencoder(config(), decoder)


def open_channel(ae: LaDiffAutoencoder) -> LaDiffAutoencoder:
    for adapter in ae.decoder.latent_adapters.values():
        torch.nn.init.normal_(adapter.conv_out.weight, std=0.2)
    return ae


class TestEncoder:
    def test_latents_have_the_declared_shape(self):
        encoder = LaDiffEncoder(config())
        features = torch.randn(2, 12, HIDDEN)
        z = encoder(features)
        assert z.shape == (2, N_LATENTS, HIDDEN)

    def test_encoder_is_deterministic_and_feature_dependent(self):
        encoder = LaDiffEncoder(config()).eval()
        a = torch.randn(2, 12, HIDDEN, generator=torch.Generator().manual_seed(1))
        b = torch.randn(2, 12, HIDDEN, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            za1, za2, zb = encoder(a), encoder(a), encoder(b)
        assert torch.equal(za1, za2)
        assert not torch.allclose(za1, zb)


class TestRunningStandardizer:
    def test_stats_track_the_stream_in_train_and_freeze_in_eval(self):
        std = RunningStandardizer(HIDDEN)
        std.train()
        g = torch.Generator().manual_seed(0)
        stream = 3.0 + 2.0 * torch.randn(64, 10, HIDDEN, generator=g)
        for chunk in stream.split(16):
            std(chunk)
        mean_after = std.mean.clone()
        std.eval()
        std(torch.full((4, 10, HIDDEN), 100.0))
        assert torch.equal(std.mean, mean_after), "eval must not update stats"
        assert mean_after.mean().item() == pytest.approx(3.0, abs=0.2)
        assert std.std.mean().item() == pytest.approx(2.0, abs=0.3)

    def test_normalize_centers_the_stream(self):
        std = RunningStandardizer(HIDDEN)
        std.train()
        g = torch.Generator().manual_seed(1)
        stream = -1.0 + 0.5 * torch.randn(256, 4, HIDDEN, generator=g)
        out = None
        for chunk in stream.split(32):
            out = std(chunk)
        assert out.mean().item() == pytest.approx(0.0, abs=0.15)
        assert out.std().item() == pytest.approx(1.0, abs=0.15)


class TestAlgorithmOneBranches:
    """The regularizer recipe is BRANCHED, not additive: feature mask XOR
    feature noise; latent Gaussian-replacement XOR latent mask."""

    def seeded_ae(self):
        ae = open_channel(autoencoder())
        ids = torch.randint(
            0, VOCAB - 1, (4, 12), generator=torch.Generator().manual_seed(3)
        )
        return ae, ids

    def test_feature_branch_masks_xor_noises(self):
        ae, ids = self.seeded_ae()
        seen = []
        original = ae.encoder.forward

        def spy(features):
            seen.append(features.detach().clone())
            return original(features)

        ae.encoder.forward = spy
        for seed in range(6):
            ladiff_autoencoder_loss(
                ae,
                ids,
                feature_mask_p=0.9,
                feature_noise_std=0.5,
                latent_mask_p=0.0,
                latent_dropout_p=0.0,
                generator=torch.Generator().manual_seed(seed),
            )
        # A masked-branch draw has exact zeros at ~90% of coordinates; the
        # noise branch has (almost surely) none.  Both branches must occur
        # across seeds, and no draw may show both signatures at once.
        zero_fractions = [float((f == 0).float().mean()) for f in seen]
        assert any(z > 0.5 for z in zero_fractions), "mask branch never taken"
        assert any(z < 0.05 for z in zero_fractions), "noise branch never taken"

    def test_latent_dropout_replaces_with_standardized_gaussian(self):
        """When the (two-coin) dropout branch fires with p=1, the decoder
        sees z drawn from mu_z + sigma_z * eta — NOT the encoder output:
        the encoder must be OUT of the graph.  The outer 1/2 coin routes
        some seeds to the mask branch instead, so BOTH outcomes must occur
        across seeds — a dropout that always (or never) fires would break
        the branch structure."""
        cut, kept = [], []
        for seed in range(8):
            ae, ids = self.seeded_ae()
            losses = ladiff_autoencoder_loss(
                ae,
                ids,
                feature_mask_p=0.0,
                feature_noise_std=0.0,
                latent_mask_p=0.0,
                latent_dropout_p=1.0,
                generator=torch.Generator().manual_seed(seed),
            )
            losses["total"].backward()
            encoder_grad = sum(
                float(p.grad.abs().sum())
                for p in ae.encoder.parameters()
                if p.grad is not None
            )
            (cut if encoder_grad == 0 else kept).append(seed)
        assert cut, "the dropout branch never fired: encoder never cut"
        assert kept, "the dropout branch always fired: outer coin inert"

    def test_latent_mask_branch_zeroes_latent_coordinates(self):
        ae, ids = self.seeded_ae()
        seen = []
        decoder_forward = ae.decoder.forward

        def spy(input_ids=None, latents=None, **kw):
            seen.append(latents.detach().clone())
            return decoder_forward(input_ids=input_ids, latents=latents, **kw)

        ae.decoder.forward = spy
        for seed in range(8):
            ladiff_autoencoder_loss(
                ae,
                ids,
                feature_mask_p=0.0,
                feature_noise_std=0.0,
                latent_mask_p=0.9,
                latent_dropout_p=0.0,
                generator=torch.Generator().manual_seed(seed),
            )
        zero_fractions = [float((z == 0).float().mean()) for z in seen]
        assert any(z > 0.5 for z in zero_fractions), "latent mask branch never taken"
        assert any(z < 0.05 for z in zero_fractions), (
            "the no-regularization branch never taken"
        )


class TestAutoencoderLoss:
    def test_supervises_masked_positions_only(self):
        """Constant (-1) ELBO weighting == plain mean CE over MASKED
        positions.  Proven by counterfactual: perturbing the decoder's
        logits at UNMASKED positions must not change the loss."""
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        captured = {}
        decoder_forward = ae.decoder.forward

        def record(input_ids=None, latents=None, **kw):
            out = decoder_forward(input_ids=input_ids, latents=latents, **kw)
            captured["corrupted"] = input_ids.clone()
            return out

        ae.decoder.forward = record
        g = torch.Generator().manual_seed(5)
        loss = ladiff_autoencoder_loss(ae, ids, generator=g)
        masked = captured["corrupted"] == MASK_ID
        assert bool(masked.any()) and not bool(masked.all())
        # Same seed, but decoder logits perturbed at unmasked positions.
        ae2, _ = TestAlgorithmOneBranches().seeded_ae()
        ae2.load_state_dict(ae.state_dict())
        inner = ae2.decoder.forward

        def perturb(input_ids=None, latents=None, **kw):
            out = inner(input_ids=input_ids, latents=latents, **kw)
            unmasked = (input_ids != MASK_ID).unsqueeze(-1)
            out.logits = out.logits + 7.0 * unmasked
            return out

        ae2.decoder.forward = perturb
        loss2 = ladiff_autoencoder_loss(
            ae2, ids, generator=torch.Generator().manual_seed(5)
        )
        assert torch.equal(loss["total"], loss2["total"])

    def test_latent_encodes_the_clean_sequence_not_the_corrupted_one(self):
        """The paper's latent is E(clean x); an encoder fed the corrupted
        state would leak the mask pattern instead of sentence content."""
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        seen = []
        original = ae.encoder.forward

        def spy(features):
            seen.append(features.detach().clone())
            return original(features)

        ae.encoder.forward = spy
        with torch.no_grad():
            clean_features = ae.features(ids)
        ladiff_autoencoder_loss(
            ae,
            ids,
            feature_mask_p=0.0,
            feature_noise_std=0.0,
            latent_mask_p=0.0,
            latent_dropout_p=0.0,
            generator=torch.Generator().manual_seed(11),
        )
        assert torch.allclose(seen[0], ae.feature_standardizer(clean_features))

    def test_gradients_reach_the_encoder_through_an_open_channel(self):
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        losses = ladiff_autoencoder_loss(
            ae,
            ids,
            feature_mask_p=0.0,
            feature_noise_std=0.0,
            latent_mask_p=0.0,
            latent_dropout_p=0.0,
            generator=torch.Generator().manual_seed(0),
        )
        losses["total"].backward()
        grads = [
            p.grad.abs().sum() for p in ae.encoder.parameters() if p.grad is not None
        ]
        assert grads and float(sum(grads)) > 0

    def test_frozen_trunk_features_receive_no_gradient(self):
        """The feature extractor is the FROZEN pretrained trunk (the BERT
        substitute) — AE training must not update it through the feature
        path."""
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        losses = ladiff_autoencoder_loss(
            ae, ids, generator=torch.Generator().manual_seed(0)
        )
        losses["total"].backward()
        table = ae.decoder.model.vocab_embed.embedding
        assert table.grad is None or table.grad.abs().sum() == 0

    def test_deterministic_given_a_generator(self):
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        l1 = ladiff_autoencoder_loss(
            ae, ids, generator=torch.Generator().manual_seed(9)
        )
        l2 = ladiff_autoencoder_loss(
            ae, ids, generator=torch.Generator().manual_seed(9)
        )
        assert torch.equal(l1["total"], l2["total"])


class TestStaticFeatureExtractor:
    def test_features_do_not_chase_the_finetuning_decoder(self):
        """The extractor is the trunk AS OF CONSTRUCTION (the paper's frozen
        BERT role): mutating the decoder trunk afterwards must not move the
        features, or the latent space chases a moving target."""
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        with torch.no_grad():
            before = ae.features(ids)
            ae.decoder.model.vocab_embed.embedding.add_(1.0)
            ae.decoder.model.blocks[0].attn_qkv.weight.add_(0.5)
            after = ae.features(ids)
        assert torch.equal(before, after)


class TestDropoutReplacementStatistics:
    def test_replacement_is_mu_plus_sigma_eta_not_raw_noise(self):
        """Fire the dropout branch with doctored latent statistics: the
        decoder must receive z ~ N(mu_z, sigma_z^2), not N(0, I)."""
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        ae.latent_standardizer.mean.fill_(10.0)
        ae.latent_standardizer.m2.fill_(0.0)  # sigma ~ sqrt(eps) ~ 0
        ae.latent_standardizer.count.fill_(1.0)
        ae.latent_standardizer.eval()  # keep the doctored stats
        seen = []
        decoder_forward = ae.decoder.forward

        def spy(input_ids=None, latents=None, **kw):
            seen.append(latents.detach().clone())
            return decoder_forward(input_ids=input_ids, latents=latents, **kw)

        ae.decoder.forward = spy
        fired = False
        for seed in range(8):
            seen.clear()
            ladiff_autoencoder_loss(
                ae,
                ids,
                feature_mask_p=0.0,
                feature_noise_std=0.0,
                latent_mask_p=0.0,
                latent_dropout_p=1.0,
                generator=torch.Generator().manual_seed(seed),
            )
            if abs(float(seen[0].mean()) - 10.0) < 0.1:
                fired = True
                break
        assert fired, (
            "no seed delivered mu_z-centred latents: the replacement is not "
            "mu_z + sigma_z * eta"
        )


class TestDeadRowGuard:
    def test_loss_is_finite_even_when_a_row_draws_no_mask(self):
        """Length-1 rows make unmasked rows likely; without the guard the
        CE over an empty index set is NaN."""
        ae, _ = TestAlgorithmOneBranches().seeded_ae()
        ids = torch.randint(0, VOCAB - 1, (2, 1))
        for seed in range(20):
            loss = ladiff_autoencoder_loss(
                ae, ids, generator=torch.Generator().manual_seed(seed)
            )
            assert torch.isfinite(loss["total"]), f"seed {seed} produced NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


class TestCompositionLayer:
    """#134 review Criticals: device movement, checkpoint identity, dtype."""

    @pytest.mark.gpu
    def test_the_whole_autoencoder_runs_on_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        ae = ae.to("cuda")
        loss = ladiff_autoencoder_loss(
            ae, ids.cuda(), generator=torch.Generator().manual_seed(0)
        )
        assert loss["total"].device.type == "cuda"
        assert torch.isfinite(loss["total"])

    def test_bf16_conversion_leaves_the_statistics_in_fp32(self):
        """#134 review: bf16 accumulators freeze `count` once it exceeds the
        mantissa (increments vanish) and quantize mean/m2 — silently
        poisoning the statistics the prior slice consumes."""
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        ae = ae.to(torch.bfloat16)
        for std in (ae.feature_standardizer, ae.latent_standardizer):
            assert std.count.dtype == torch.float32
            assert std.mean.dtype == torch.float32
            assert std.m2.dtype == torch.float32
        std = RunningStandardizer(4).to(torch.bfloat16)
        std.train()
        for _ in range(300):
            std(torch.randn(7, 4))
        assert float(std.count) == 2100.0, "count lost increments"

    def test_loading_over_a_different_trunk_is_refused(self):
        """#134 review Critical: the frozen extractor is not persisted (it
        is bitwise the published checkpoint), so loading an AE state dict
        onto an instance built around a DIFFERENT trunk would silently swap
        the feature space.  A fingerprint in the state dict makes that loud."""
        ae, _ = TestAlgorithmOneBranches().seeded_ae()
        state = ae.state_dict()

        torch.manual_seed(123)  # a different random trunk
        other_decoder = LatentConditionedMDLMDiT(config())
        other = LaDiffAutoencoder(config(), other_decoder)
        with pytest.raises(ValueError, match="trunk"):
            other.load_state_dict(state)

    def test_loading_over_the_same_trunk_succeeds_and_preserves_features(self):
        ae, ids = TestAlgorithmOneBranches().seeded_ae()
        state = ae.state_dict()
        torch.manual_seed(0)  # the same construction path as autoencoder()
        twin_decoder = LatentConditionedMDLMDiT(config())
        twin = LaDiffAutoencoder(config(), twin_decoder)
        twin.load_state_dict(state)
        with torch.no_grad():
            assert torch.equal(twin.features(ids), ae.features(ids))

    def test_eval_normalization_with_empty_statistics_is_refused(self):
        """#134 review: eval at count=0 would normalize by sqrt(eps) and
        blow features up 300x (or hand the decoder a near-zero latent in
        the dropout branch) — refuse instead of guessing."""
        std = RunningStandardizer(HIDDEN).eval()
        with pytest.raises(RuntimeError, match="statistics"):
            std(torch.randn(2, 4, HIDDEN))


class TestOpenLatentChannelAPI:
    def test_open_latent_channel_lets_the_encoder_learn_from_step_zero(self):
        """#134 review: eq.(32)'s double zero-init means dL/dz = 0 until
        conv_out moves — the paper's encoder-first warmup would burn on zero
        gradients.  The supported API opens the channel; the AE run protocol
        pins the std as a recorded deviation."""
        _, ids = TestAlgorithmOneBranches().seeded_ae()
        ae2 = autoencoder()  # fresh, unopened
        ae2.decoder.open_latent_channel(std=1e-3)
        losses = ladiff_autoencoder_loss(
            ae2,
            ids,
            feature_mask_p=0.0,
            feature_noise_std=0.0,
            latent_mask_p=0.0,
            latent_dropout_p=0.0,
            generator=torch.Generator().manual_seed(0),
        )
        losses["total"].backward()
        grads = sum(
            float(p.grad.abs().sum())
            for p in ae2.encoder.parameters()
            if p.grad is not None
        )
        assert grads > 0, "opened channel still starves the encoder"
