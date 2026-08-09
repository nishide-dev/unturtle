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
MeanFlow self-distillation of the latent prior (#117 slice 2).

arXiv:2605.23605 §3.3, eqs. 13-16: the student learns the AVERAGE velocity
``u(z_t, t, r)`` — the mean displacement between two points on the teacher's
ODE path — so few latent steps replace many.  The target couples the
teacher's instantaneous velocity with the student's own derivatives:

    u_tgt = v(z_t, t) - (t - r) * (v . d_z u + d_t u)          (eq. 15)
    L     = || u(z_t, t, r) - stopgrad(u_tgt) ||^2             (eq. 14)

The directional derivative is one JVP with tangents ``(v, 1, 0)`` over
``(z, t, r)`` — chain-ruled through the model's time rescaling by defining
the differentiated function over UNSCALED time.  The teacher's velocity
comes from the slice-1 prior's x0 prediction on the linear path:
``v = (z_t - x0_pred) / t`` — the identity #116's average-velocity sampler
is built on.  Deviation recorded: the paper's modified self-conditioning
(two teacher NFEs) is omitted at prototype scale.

Sampling (eq. 16) over a decreasing grid ``tau``:

    z <- z + (tau_next - tau) * u(z, tau, tau_next)

Note the student OUTPUT is a velocity, not an x0 prediction — the opposite
convention from ``FlowLMDenoiser`` — so the two are deliberately distinct
classes rather than a flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .modeling_flowlm import FlowLMConfig, _sinusoidal_time_embedding


@dataclass
class MeanFlowOutput(ModelOutput):
    velocity: Optional[torch.Tensor] = None


class MeanFlowDenoiser(PreTrainedModel):
    """[B, M, H] -> [B, M, H] average-velocity student with (t, r) conditioning."""

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
        self.target_time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=0.0,
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
        target_timesteps: torch.Tensor,
        **_: Any,
    ) -> MeanFlowOutput:
        positions = torch.arange(latents.shape[1], device=latents.device)
        current = self.time_mlp(
            _sinusoidal_time_embedding(timesteps, self.config.hidden_size).to(
                self.dtype
            )
        )
        target = self.target_time_mlp(
            _sinusoidal_time_embedding(target_timesteps, self.config.hidden_size).to(
                self.dtype
            )
        )
        hidden = (
            latents
            + self.position_embedding(positions)
            + (current + target).unsqueeze(1).to(latents.dtype)
        )
        hidden = self.blocks(hidden)
        return MeanFlowOutput(velocity=self.output_proj(self.output_norm(hidden)))


def meanflow_distillation_loss(
    student: Any,
    teacher: Any,
    clean_latents: torch.Tensor,
    *,
    num_timesteps: int,
    time_scale: float = 1000.0,
    pure_fm_fraction: float = 0.25,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Eq. 14 with the eq. 15 JVP target; named terms.

    Rows are split per-batch: a ``pure_fm_fraction`` share trains with
    ``r = t`` (the target degenerates to the teacher velocity — the paper's
    25% pure flow-matching loss); the rest samples ``r ~ U(0, t)`` and takes
    the full JVP correction.  The whole target is built under ``no_grad``
    (eq. 14's stopgrad): only the student's plain forward carries gradient,
    and the frozen teacher receives none.
    """
    rows = clean_latents.shape[0]
    device = clean_latents.device

    steps = torch.randint(1, num_timesteps + 1, (rows,), generator=generator)
    t = (steps.to(clean_latents.dtype) / num_timesteps).to(device)
    noise = torch.randn(clean_latents.shape, generator=generator).to(device)
    t_b = t.view(-1, 1, 1)
    z_t = (1 - t_b) * clean_latents + t_b * noise

    r = (torch.rand(rows, generator=generator).to(device)) * t
    pure = torch.rand(rows, generator=generator).to(device) < pure_fm_fraction
    r = torch.where(pure, t, r)

    with torch.no_grad():
        teacher_x0 = teacher(z_t, timesteps=t * time_scale).prediction
        v = (z_t - teacher_x0) / t_b

        def u_fn(z: torch.Tensor, tt: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
            return student(
                z,
                timesteps=tt * time_scale,
                target_timesteps=rr * time_scale,
            ).velocity

        # Forward-mode AD is only implemented for the math SDPA backend;
        # the flash/efficient kernels raise NotImplementedError inside jvp.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            _, du = torch.func.jvp(
                u_fn,
                (z_t, t, r),
                (v, torch.ones_like(t), torch.zeros_like(r)),
            )
        u_target = v - (t - r).view(-1, 1, 1) * du

    u = student(z_t, timesteps=t * time_scale, target_timesteps=r * time_scale).velocity
    per_row = ((u - u_target) ** 2).flatten(1).mean(dim=1)

    losses: dict[str, torch.Tensor] = {}
    if bool((~pure).any()):
        losses["meanflow_mse"] = per_row[~pure].mean()
    if bool(pure.any()):
        losses["pure_fm_mse"] = per_row[pure].mean()
    losses["total"] = per_row.mean()
    return losses


@torch.no_grad()
def sample_meanflow_latents(
    student: Any,
    *,
    batch_size: int,
    num_steps: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Eq. 16 over a decreasing uniform grid from tau = 1 to tau = 0."""
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    config = student.config
    device = next(student.parameters()).device
    latents = torch.randn(
        (batch_size, config.max_position_embeddings, config.hidden_size),
        generator=generator,
        device=device if generator is None else generator.device,
    ).to(device=device, dtype=next(student.parameters()).dtype)

    for m in range(num_steps, 0, -1):
        tau = m / num_steps
        tau_next = (m - 1) / num_steps
        velocity = student(
            latents,
            timesteps=torch.full((batch_size,), tau * config.time_scale, device=device),
            target_timesteps=torch.full(
                (batch_size,), tau_next * config.time_scale, device=device
            ),
        ).velocity
        latents = latents + (tau_next - tau) * velocity
    return latents


__all__ = [
    "MeanFlowDenoiser",
    "MeanFlowOutput",
    "meanflow_distillation_loss",
    "sample_meanflow_latents",
]
