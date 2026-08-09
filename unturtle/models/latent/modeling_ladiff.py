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

"""
DiLaDiff slice 1 (#117): latent-conditioned masked decoder, Perceiver-lite
encoder, trainable codec, and the LaDiff bundle (prior + guided decoding).

Method: arXiv:2605.23605 (``dev/papers/diladiff.pdf``).  Eq. 8 trains
encoder + latent-conditioned masked-diffusion decoder jointly; eqs. 9-10 are
the reason this exists — conditioned on a latent that carries the token
correlations, the token-wise factorized posterior *truly* factorizes, so
parallel masked decoding stays coherent where the unconditional model mixes
modes (#116 measured that failure on the continuous side).  Eqs. 11-12:
sample the latent with a continuous solver, then run the DISCRETE masked
loop conditioned on it — the continuous <-> discrete boundary this slice
exists to exercise.

Recorded deviations (prototype scale): additive zero-init latent
conditioning instead of cross-attention wrapped in zero-init convolutions
(same inertness-at-init property, simpler); a Perceiver-lite encoder over
the decoder's own embeddings instead of BERT features; the prior reuses the
linear-interpolation path from #116 instead of tanh-logSNR VP.  MeanFlow
self-distillation (§3.3) is deliberately slice 2.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from .modeling_flowlm import FlowLMConfig, FlowLMDenoiser


class LaDiffConfig(PretrainedConfig):
    model_type = "ladiff-prototype"

    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        max_position_embeddings: int = 64,
        mask_token_id: int = 15,
        num_latents: int = 2,
        num_timesteps: int = 20,
        time_scale: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_latents = num_latents
        self.num_timesteps = num_timesteps
        self.time_scale = time_scale
        super().__init__(mask_token_id=mask_token_id, **kwargs)


class LatentConditionedMDLM(PreTrainedModel):
    """Tiny bidirectional masked denoiser with an additive latent channel.

    ``latents=None`` is the plain masked dLLM — the model pretrains
    unconditionally and the AE finetune CONTINUES the same weights (the
    paper's decoder-from-pretrained-MDLM initialization, literal at this
    scale).  The latent projection is zero-initialized so a latent changes
    nothing at init: finetuning opens the channel instead of starting by
    destroying the pretrained decoder.
    """

    config_class = LaDiffConfig

    def __init__(self, config: LaDiffConfig) -> None:
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=0.0,  # unseeded dropout would break generator-seeded
            batch_first=True,  # reproducibility of the AE loss
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.num_hidden_layers)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        # Zero-init: the latent channel is inert until trained (see class
        # docstring).  `post_init` runs first so it cannot re-randomize this.
        self.latent_proj = nn.Linear(
            config.num_latents * config.hidden_size, config.hidden_size
        )
        self.post_init()
        nn.init.zeros_(self.latent_proj.weight)
        nn.init.zeros_(self.latent_proj.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> MaskedLMOutput:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embedding(input_ids) + self.position_embedding(positions)
        if latents is not None:
            conditioning = self.latent_proj(latents.reshape(latents.shape[0], -1))
            hidden = hidden + conditioning.unsqueeze(1)
        hidden = self.blocks(hidden)
        return MaskedLMOutput(logits=self.lm_head(self.output_norm(hidden)))


class PerceiverLiteEncoder(nn.Module):
    """M learned queries cross-attending over an encoded token sequence."""

    def __init__(self, config: LaDiffConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=0.0,  # unseeded dropout would break generator-seeded
            batch_first=True,  # reproducibility of the AE loss
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=1)
        self.queries = nn.Parameter(
            torch.randn(config.num_latents, config.hidden_size) * 0.02
        )
        self.pool = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.blocks(
            self.embedding(input_ids) + self.position_embedding(positions)
        )
        queries = self.queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        pooled, _ = self.pool(queries, hidden, hidden)
        return pooled


class LatentAutoencoderCodec(nn.Module):
    """The ``Codec`` protocol's trainable end (#117, vs #116's simple end).

    ``encode`` is the Perceiver-lite encoder; ``decode`` is NOT a rounding
    head — it is the latent-conditioned masked dLLM evaluated on a masked
    state.  The codec's own recipe (latent dropout / noise) lives in
    :func:`latent_autoencoder_loss`, which returns its terms by name.
    """

    trainable = True

    def __init__(self, config: LaDiffConfig, decoder: LatentConditionedMDLM) -> None:
        super().__init__()
        self.config = config
        self.encoder = PerceiverLiteEncoder(config)
        # Held by REFERENCE, not registered as a submodule: `LaDiffModel`
        # already owns the decoder, and registering it here again would put
        # every decoder tensor in the state dict twice (safetensors rejects
        # the shared storage on save).  Training loops address the decoder's
        # parameters through the model, not through the codec.
        object.__setattr__(self, "decoder", decoder)

    def encode(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.encoder(input_ids)

    def decode(
        self, latents: torch.Tensor, input_ids: torch.Tensor, **_: Any
    ) -> torch.Tensor:
        return self.decoder(input_ids=input_ids, latents=latents).logits

    def auxiliary_losses(
        self, latents: torch.Tensor, input_ids: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        # Slice 1 keeps the paper's loss-side recipe inside
        # `latent_autoencoder_loss` (its regularizers are input transforms,
        # not loss terms); the protocol slot stays honest and empty.
        return {}


def latent_autoencoder_loss(
    codec: LatentAutoencoderCodec,
    input_ids: torch.Tensor,
    *,
    latent_dropout: float = 0.1,
    latent_noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """AE-MDLM objective (eq. 8) with the paper's constant (-1) weighting.

    Constant weighting == plain mean CE over MASKED positions; unmasked
    positions grant no supervision (copying visible tokens is free).  The
    latent is the CLEAN sequence's encoding, perturbed per the recipe:

    - ``latent_dropout``: with this probability per row, the latent is
      replaced by pure ``N(0, I)`` noise — the mechanism that preserves an
      unconditional decoding mode (and, at p=1, exactly plain MDLM
      pretraining: the encoder is out of the graph);
    - ``latent_noise_std``: Gaussian noise on the surviving latents.
    """
    mask_id = codec.config.mask_token_id
    rows, length = input_ids.shape

    # Drawn on the generator's device (CPU by default) then transferred —
    # drawing directly on CUDA from a CPU generator is itself an error, and
    # an untransferred CPU mask raises on `masked_fill` against CUDA ids.
    t = torch.rand(rows, 1, generator=generator).clamp_min(1e-3)
    masked = (torch.rand(rows, length, generator=generator) < t).to(input_ids.device)
    # Every row must supervise something.
    dead = ~masked.any(dim=1)
    if bool(dead.any()):
        masked[dead, 0] = True
    corrupted = input_ids.masked_fill(masked, mask_id)

    latents = codec.encode(input_ids)
    if latent_noise_std > 0:
        latents = latents + latent_noise_std * torch.randn(
            latents.shape, generator=generator
        ).to(latents.device)
    if latent_dropout > 0:
        drop = (torch.rand(rows, generator=generator) < latent_dropout).view(-1, 1, 1)
        pure_noise = torch.randn(latents.shape, generator=generator).to(latents.device)
        latents = torch.where(drop.to(latents.device), pure_noise, latents)

    logits = codec.decode(latents, input_ids=corrupted)
    reconstruction = F.cross_entropy(logits[masked], input_ids[masked])
    return {"reconstruction_ce": reconstruction, "total": reconstruction}


class LaDiffModel(PreTrainedModel):
    """Codec + decoder + latent prior; the ``latent_guided`` family citizen."""

    config_class = LaDiffConfig
    supports_ladiff_generation = True

    def __init__(self, config: LaDiffConfig) -> None:
        super().__init__(config)
        self.decoder = LatentConditionedMDLM(config)
        self.codec = LatentAutoencoderCodec(config, self.decoder)
        self.prior = FlowLMDenoiser(
            FlowLMConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                max_position_embeddings=max(config.num_latents, 1),
                num_timesteps=config.num_timesteps,
                time_scale=config.time_scale,
            )
        )
        self.post_init()

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> MaskedLMOutput:
        return self.decoder(input_ids=input_ids, **kwargs)

    @torch.no_grad()
    def sample_prior_latents(
        self,
        *,
        batch_size: int,
        num_latent_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Average-velocity sampling of the latent prior (#116 solver)."""
        steps = num_latent_steps or self.config.num_timesteps
        if steps < 1:
            raise ValueError(f"num_latent_steps must be >= 1, got {steps}")
        device = next(self.parameters()).device
        latents = torch.randn(
            (batch_size, self.config.num_latents, self.config.hidden_size),
            generator=generator,
            device=device if generator is None else generator.device,
        ).to(device=device, dtype=next(self.parameters()).dtype)
        dt = 1.0 / steps
        for k in range(steps, 0, -1):
            t = k / steps
            prediction = self.prior(
                latents,
                timesteps=torch.full(
                    (batch_size,), t * self.config.time_scale, device=device
                ),
            ).prediction
            latents = (1 - dt / t) * latents + (dt / t) * prediction
        return latents

    @torch.no_grad()
    def sample_discrete(
        self,
        *,
        latents: torch.Tensor | None,
        batch_size: int,
        num_discrete_steps: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Parallel masked decode, optionally latent-conditioned (eq. 12,
        simplified to confidence-ordered categorical unmasking).

        Aggressive on purpose: with few steps the factorized posterior's
        faithfulness is exactly what decides whether rows come back
        mode-consistent — the property eqs. 9-10 attribute to the latent.
        """
        if num_discrete_steps < 1:
            raise ValueError(
                f"num_discrete_steps must be >= 1, got {num_discrete_steps}"
            )
        length = self.config.max_position_embeddings
        mask_id = self.config.mask_token_id
        device = next(self.parameters()).device
        ids = torch.full((batch_size, length), mask_id, device=device)
        per_step = -(-length // num_discrete_steps)  # ceil

        for _ in range(num_discrete_steps):
            still_masked = ids == mask_id
            if not bool(still_masked.any()):
                break
            logits = self.decoder(input_ids=ids, latents=latents).logits
            # The mask token must not be re-emitted into the output.
            logits[..., mask_id] = torch.finfo(logits.dtype).min
            probabilities = F.softmax(logits, dim=-1)
            flat = probabilities.view(-1, probabilities.shape[-1])
            draws = (
                torch.multinomial(flat.cpu(), num_samples=1, generator=generator)
                .view(batch_size, length)
                .to(device)
            )
            confidence = probabilities.gather(-1, draws.unsqueeze(-1)).squeeze(-1)
            confidence = confidence.masked_fill(~still_masked, -1.0)
            order = confidence.argsort(dim=1, descending=True)
            chosen = order[:, :per_step]
            commit = torch.zeros_like(still_masked)
            commit.scatter_(1, chosen, True)
            commit &= still_masked
            ids = torch.where(commit, draws, ids)
        return ids

    def generate(  # type: ignore[override]
        self,
        inputs: Any = None,
        algorithm: str = "auto",
        **kwargs: Any,
    ) -> torch.Tensor:
        from unturtle.models.generation.sampler import (
            GenerationRequest,
            dispatch_generation,
        )

        return dispatch_generation(
            self,
            GenerationRequest(inputs=inputs, kwargs=kwargs),
            algorithm,
        )

    def _generate_ladiff(
        self,
        inputs: Any = None,
        *,
        batch_size: int = 1,
        num_latent_steps: int | None = None,
        num_discrete_steps: int = 1,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if inputs is not None:
            raise ValueError(
                "the LaDiff prototype is unconditional over prompts; the "
                "conditioning channel is the latent, not a token prefix"
            )
        latents = self.sample_prior_latents(
            batch_size=batch_size,
            num_latent_steps=num_latent_steps,
            generator=generator,
        )
        return self.sample_discrete(
            latents=latents,
            batch_size=batch_size,
            num_discrete_steps=num_discrete_steps,
            generator=generator,
        )


__all__ = [
    "LaDiffConfig",
    "LaDiffModel",
    "LatentAutoencoderCodec",
    "LatentConditionedMDLM",
    "PerceiverLiteEncoder",
    "latent_autoencoder_loss",
]
