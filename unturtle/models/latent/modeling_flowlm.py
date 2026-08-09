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
FlowLM prototype model (#66): tiny time-conditioned denoiser + codec.

Exists to exercise the RFC's boundaries end-to-end (process -> denoiser ->
objective -> solver -> rounding) with `PreTrainedModel` save/load — not to
claim quality.  Conditioning (FlowLM's ``Concat(z^x, z_t^y)``) is omitted:
the prototype is unconditional; adding a condition prefix is a
straightforward extension that changes no boundary.

The sampler implements Algorithm 2 / eq. 7 exactly:

    v = (z_t - z_0,pred) / t
    z <- z - v * dt   ==   z <- (1 - dt/t) * z + (dt/t) * z_0,pred

an **average**-velocity update, not Euler on an instantaneous field.  The
convex combination gives the property the tests pin: at the final step
``t = dt`` so ``dt/t = 1`` and the sampler lands exactly on the prediction —
the same guidance-toward-data behavior for any step count, which is what
makes one-step generation coherent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from .codec import EmbeddingRoundingCodec


class FlowLMConfig(PretrainedConfig):
    model_type = "flowlm-prototype"

    def __init__(
        self,
        vocab_size: int = 16,
        hidden_size: int = 32,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        max_position_embeddings: int = 64,
        num_timesteps: int = 20,
        time_scale: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_timesteps = num_timesteps
        self.time_scale = time_scale
        super().__init__(**kwargs)


@dataclass
class FlowLMDenoiserOutput(ModelOutput):
    prediction: Optional[torch.Tensor] = None


def _sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(-1) * frequencies
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class FlowLMDenoiser(PreTrainedModel):
    """[B, L, H] -> [B, L, H] x0 predictor with sinusoidal time conditioning.

    Bidirectional by construction (`TransformerEncoder`, no causal mask).
    The time embedding is added to every position; a denoiser that cannot
    see ``t`` cannot represent the interpolation path, and the tests assert
    the conditioning is observable.
    """

    config_class = FlowLMConfig

    def __init__(self, config: FlowLMConfig) -> None:
        super().__init__(config)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.num_hidden_layers)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> FlowLMDenoiserOutput:
        positions = torch.arange(latents.shape[1], device=latents.device)
        time_embedding = self.time_mlp(
            _sinusoidal_time_embedding(timesteps, self.config.hidden_size)
        ).to(latents.dtype)
        hidden = (
            latents + self.position_embedding(positions) + time_embedding.unsqueeze(1)
        )
        padding_mask = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0
        hidden = self.blocks(hidden, src_key_padding_mask=padding_mask)
        return FlowLMDenoiserOutput(
            prediction=self.output_proj(self.output_norm(hidden))
        )


class FlowLMModel(PreTrainedModel):
    """Codec + denoiser bundle; the registry's ``continuous_flow`` citizen."""

    config_class = FlowLMConfig
    # Capability marker for the generation registry's probe: attribute-based,
    # so `sampler.py` never imports this module.
    supports_flowlm_generation = True

    def __init__(self, config: FlowLMConfig) -> None:
        super().__init__(config)
        self.codec = EmbeddingRoundingCodec(config.vocab_size, config.hidden_size)
        self.denoiser = FlowLMDenoiser(config)
        self.post_init()

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs: Any,
    ) -> FlowLMDenoiserOutput:
        return self.denoiser(latents, timesteps=timesteps, **kwargs)

    @torch.no_grad()
    def sample_latents(
        self,
        *,
        batch_size: int,
        num_steps: int | None = None,
        seq_len: int | None = None,
        denoise_fn: Any = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Algorithm 2: average-velocity updates from pure noise to t=0."""
        steps = num_steps or self.config.num_timesteps
        if steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {steps}")
        length = seq_len or self.config.max_position_embeddings
        denoise = denoise_fn if denoise_fn is not None else self.denoiser
        device = next(self.parameters()).device

        latents = torch.randn(
            (batch_size, length, self.config.hidden_size), generator=generator
        ).to(device)
        dt = 1.0 / steps
        for k in range(steps, 0, -1):
            t = k / steps
            prediction = denoise(
                latents,
                timesteps=torch.full(
                    (batch_size,), t * self.config.time_scale, device=device
                ),
            ).prediction
            latents = (1 - dt / t) * latents + (dt / t) * prediction
        return latents

    def generate(  # type: ignore[override]
        self,
        inputs: Any = None,
        algorithm: str = "auto",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Registry-dispatched generation (no masked flags, no fallback)."""
        from unturtle.models.generation.sampler import (
            GenerationRequest,
            dispatch_generation,
        )

        return dispatch_generation(
            self,
            GenerationRequest(inputs=inputs, kwargs=kwargs),
            algorithm,
        )

    def _generate_flowlm(
        self,
        *,
        batch_size: int = 1,
        num_steps: int | None = None,
        seq_len: int | None = None,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> torch.Tensor:
        latents = self.sample_latents(
            batch_size=batch_size,
            num_steps=num_steps,
            seq_len=seq_len,
            generator=generator,
        )
        return self.codec.decode(latents).argmax(dim=-1)


__all__ = [
    "FlowLMConfig",
    "FlowLMDenoiser",
    "FlowLMDenoiserOutput",
    "FlowLMModel",
]
