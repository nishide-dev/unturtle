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

"""LaDiff autoencoder on the real MDLM-DiT backbone (#130 PR-codec-2).

Paper Algorithm 1, with the recorded substitutions for the gpt2/kuleshov
lineage (issue #130):

- the frozen feature extractor is a FROZEN COPY of the pretrained trunk
  taken at construction (the paper uses a frozen BERT; the decoder trunk
  itself finetunes during AE training, so the live trunk cannot double as
  the extractor — the latent space would chase a moving target);
- features and latents are standardized coordinate-wise with running
  statistics (Welford), frozen at eval;
- the regularizer recipe is BRANCHED per batch, verbatim Algorithm 1:
  feature branch = {mask XOR noise}, latent branch = {(maybe replace the
  whole latent with mu_z + sigma_z * eta) XOR mask}.  The branch coins are
  per-batch, as written in the pseudocode.
- the objective is constant-weighted masked CE (eq. 8 with the -1
  weighting): supervision on masked positions only.

The frozen trunk copy is held by non-registered reference: it is bitwise
the published checkpoint's trunk, reconstructable via ``load_mdlm_owt`` —
persisting it would double every AE checkpoint for no information.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modeling_ladiff_dit import LaDiffDiTConfig, LatentConditionedMDLMDiT


class LaDiffEncoder(nn.Module):
    """Meshchaninov-style encoder: the latent z (learned queries) is refined
    per layer by cross-attention over the concatenation ``[h, z]``."""

    def __init__(self, config: LaDiffDiTConfig) -> None:
        super().__init__()
        if config.latent_dim != config.hidden_size:
            raise ValueError(
                "LaDiffEncoder concatenates features and latents along the "
                f"sequence axis; latent_dim ({config.latent_dim}) must equal "
                f"hidden_size ({config.hidden_size})."
            )
        dim = config.latent_dim
        self.queries = nn.Parameter(torch.randn(config.num_latents, dim) * 0.02)
        layers = getattr(config, "encoder_layers", 4)
        self.norms_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, config.num_attention_heads, batch_first=True)
                for _ in range(layers)
            ]
        )
        self.norms_f = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers)])
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 4 * dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(4 * dim, dim),
                )
                for _ in range(layers)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        z = self.queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        for norm_q, attn, norm_f, ffn in zip(
            self.norms_q, self.attns, self.norms_f, self.ffns
        ):
            kv = torch.cat([features, z], dim=1)
            attended, _ = attn(norm_q(z), kv, kv, need_weights=False)
            z = z + attended
            z = z + ffn(norm_f(z))
        return z


class RunningStandardizer(nn.Module):
    """Coordinate-wise standardization with running (Welford) statistics.

    Training forwards UPDATE the statistics then normalize; eval forwards
    only normalize.  Statistics live in buffers, so they persist through
    save/load and are shared with consumers needing ``mean``/``std``
    (Algorithm 1's Gaussian latent replacement, the prior slice's VP
    schedules)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))

    @property
    def std(self) -> torch.Tensor:
        variance = self.m2 / self.count.clamp_min(1.0)
        return torch.sqrt(variance + self.eps)

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1]).float()
        n = flat.shape[0]
        batch_mean = flat.mean(dim=0)
        batch_m2 = ((flat - batch_mean) ** 2).sum(dim=0)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * (n / total)
        self.m2 += batch_m2 + delta.square() * (self.count * n / total)
        self.count += n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update(x)
        return (x - self.mean) / self.std


class LaDiffAutoencoder(nn.Module):
    """Encoder + latent-conditioned decoder + standardizers.

    Owns (and registers) the decoder — its parameters train after warmup —
    and freezes the pretrained embedding table on construction (paper C.1).
    The frozen feature trunk is a non-registered deep copy of the decoder
    trunk at construction time.
    """

    def __init__(
        self, config: LaDiffDiTConfig, decoder: LatentConditionedMDLMDiT
    ) -> None:
        super().__init__()
        self.config = config
        self.decoder = decoder
        self.decoder.freeze_for_autoencoder_training()
        self.encoder = LaDiffEncoder(config)
        self.feature_standardizer = RunningStandardizer(config.hidden_size)
        self.latent_standardizer = RunningStandardizer(config.latent_dim)
        trunk = copy.deepcopy(decoder.model).eval()
        trunk.requires_grad_(False)
        # Non-registered: bitwise the published checkpoint's trunk (see
        # module docstring), rebuilt on load rather than persisted.
        object.__setattr__(self, "feature_trunk", trunk)

    @torch.no_grad()
    def features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Frozen-trunk hidden states of the clean sequence (the BERT-
        feature substitute): the trunk loop without the output head."""
        trunk = self.feature_trunk
        batch, length = input_ids.shape
        x = trunk.vocab_embed(input_ids)
        c = F.silu(trunk.cond).unsqueeze(0).expand(batch, -1)
        cos, sin = trunk.rotary(length, input_ids.device)
        for block in trunk.blocks:
            x = block(x, cos, sin, c, None)
        return x


def _draw(shape, generator, device):
    return torch.rand(shape, generator=generator).to(device)


def ladiff_autoencoder_loss(
    autoencoder: LaDiffAutoencoder,
    input_ids: torch.Tensor,
    *,
    feature_mask_p: float = 0.7,
    feature_noise_std: float = 0.5,
    latent_mask_p: float = 0.7,
    latent_dropout_p: float = 0.1,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Algorithm 1 (verbatim branch structure) + eq. (8) constant weighting.

    Defaults are the paper's best augmentation set (sigma_reg=0.5 noise,
    p_mask=0.7 on both features and latents); ``latent_dropout_p`` is not
    recoverable from the paper text and 0.1 is a recorded Unturtle choice
    (#130).  All randomness is drawn on ``generator``'s device (CPU by
    default) and transferred — the established seeded-reproducibility
    contract for these losses.
    """
    config = autoencoder.config
    mask_id = config.mask_token_id
    device = input_ids.device
    rows, length = input_ids.shape

    # --- masked training state (as in the prototype AE loss) ---
    t = torch.rand(rows, 1, generator=generator).clamp_min(1e-3)
    masked = (_draw((rows, length), generator, device) < t.to(device)).to(device)
    dead = ~masked.any(dim=1)
    if bool(dead.any()):
        masked[dead, 0] = True
    corrupted = input_ids.masked_fill(masked, mask_id)

    # --- encoder path (Algorithm 1 lines 1-9) ---
    features = autoencoder.features(input_ids)
    features = autoencoder.feature_standardizer(features)
    if float(torch.rand((), generator=generator)) < 0.5:
        keep = _draw(features.shape, generator, device) >= feature_mask_p
        features = features * keep
    elif feature_noise_std > 0:
        noise = torch.randn(features.shape, generator=generator).to(device)
        features = (
            1.0 - feature_noise_std**2
        ) ** 0.5 * features + feature_noise_std * noise
    latents = autoencoder.encoder(features)
    autoencoder.latent_standardizer(latents.detach())  # track stats only

    # --- latent branch (Algorithm 1 lines 10-16) ---
    if float(torch.rand((), generator=generator)) < 0.5:
        if float(torch.rand((), generator=generator)) < latent_dropout_p:
            eta = torch.randn(latents.shape, generator=generator).to(device)
            latents = (
                autoencoder.latent_standardizer.mean
                + autoencoder.latent_standardizer.std * eta
            )
    else:
        keep = _draw(latents.shape, generator, device) >= latent_mask_p
        latents = latents * keep

    # --- decoder path + eq. (8) ---
    logits = autoencoder.decoder(input_ids=corrupted, latents=latents).logits
    reconstruction = F.cross_entropy(logits[masked], input_ids[masked])
    return {"reconstruction_ce": reconstruction, "total": reconstruction}
