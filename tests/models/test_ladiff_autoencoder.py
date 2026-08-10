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


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
