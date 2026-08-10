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

"""LaDiff latent prior (#130): tanh-logSNR VP schedule, x0 DiT denoiser,
Euler sampler with carried self-conditioning.

Verbatim sources (paper eq. 33-35, Algorithms 2-3, frozen on the issue):

- logSNR(t) = -d log tan(pi t/2); sigma^2 = sigmoid(-logSNR),
  alpha^2 = sigmoid(logSNR).  d = 10 is the paper's grid-searched optimum.
- Training: z (ALREADY standardized by the caller with the AE's frozen
  latent statistics) -> z_t = alpha z + sigma eps; with prob 1/2 the
  self-conditioning input is the denoiser's own no-grad prediction
  (teacher detached); MSE ||z_hat - z||^2 (x0 / data prediction).
- Sampling: Euler on v = (1/sigma)((sigma alpha' - sigma' alpha) z_hat
  + sigma' z_t), self-conditioning carried across steps, optional gamma
  re-noising; the caller denormalizes before discrete decode.

Analytic derivatives (verified by finite differences in the tests):
  s'(t) = -d pi / sin(pi t);  alpha' = alpha sigma^2 s'/2;
  sigma' = -sigma alpha^2 s'/2.
The time grid excludes the endpoints ([1e-3, 1-1e-3]): s' diverges at both,
and alpha(1-1e-3) is 0 to fp32, so z_T ~ N(0, I) matches the schedule.

The denoiser reuses the MDLM-DiT block stack (adaLN-Zero, bidirectional)
with REAL time conditioning — the same "DiT with noise-level conditioning"
family the paper cites — over the latent sequence [B, num_latents,
latent_dim]; self-conditioning enters by channel concatenation (zeros when
absent), a recorded implementation choice.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from unturtle.models.backbones.mdlm_dit.modeling_mdlm_dit import (
    DDiTBlock,
    DDitFinalLayer,
    Rotary,
)


class TanhLogSNRSchedule:
    """eq. (33-35); variance-preserving by construction."""

    def __init__(self, d: float = 10.0) -> None:
        self.d = d

    def logsnr(self, t: torch.Tensor) -> torch.Tensor:
        return -self.d * torch.log(torch.tan(math.pi * t / 2))

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logsnr(t)).sqrt()

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-self.logsnr(t)).sqrt()

    def _logsnr_dot(self, t: torch.Tensor) -> torch.Tensor:
        return -self.d * math.pi / torch.sin(math.pi * t)

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha(t) * self.sigma(t) ** 2 * self._logsnr_dot(t) / 2

    def sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return -self.sigma(t) * self.alpha(t) ** 2 * self._logsnr_dot(t) / 2


def euler_velocity(
    schedule: TanhLogSNRSchedule,
    z_t: torch.Tensor,
    z_hat: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Algorithm 3 line 9:
    v = (1/sigma) ((sigma alpha' - sigma' alpha) z_hat + sigma' z_t).

    With a perfect prediction (z_hat = z) this reduces algebraically to
    alpha' z + sigma' eps — the exact trajectory derivative (tested)."""
    alpha, sigma = schedule.alpha(t), schedule.sigma(t)
    alpha_dot, sigma_dot = schedule.alpha_dot(t), schedule.sigma_dot(t)
    return ((sigma * alpha_dot - sigma_dot * alpha) * z_hat + sigma_dot * z_t) / sigma


class LaDiffPriorConfig(PretrainedConfig):
    model_type = "ladiff-prior"

    def __init__(
        self,
        latent_dim: int = 768,
        num_latents: int = 512,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        cond_dim: int = 128,
        dropout: float = 0.0,
        schedule_d: float = 10.0,
        time_scale: float = 1000.0,
        **kwargs,
    ) -> None:
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.schedule_d = schedule_d
        self.time_scale = time_scale
        super().__init__(**kwargs)


def sinusoidal_time_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10_000.0
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class LatentPriorDenoiser(nn.Module):
    """x0-prediction DiT over the latent sequence, time-conditioned adaLN.

    ``forward(z_t, t, self_cond=None)`` returns the predicted CLEAN
    (standardized) latent.  ``self_cond=None`` means the zero channel —
    Algorithm 2/3's empty conditioning."""

    def __init__(self, config: LaDiffPriorConfig) -> None:
        super().__init__()
        self.config = config
        hidden, heads = config.hidden_size, config.num_attention_heads
        self.in_proj = nn.Linear(2 * config.latent_dim, hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(256, config.cond_dim),
            nn.SiLU(),
            nn.Linear(config.cond_dim, config.cond_dim),
        )
        self.rotary = Rotary(hidden // heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(hidden, heads, config.cond_dim, dropout=config.dropout)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.out = DDitFinalLayer(hidden, config.latent_dim, config.cond_dim)
        # DDitFinalLayer zero-inits its adaLN gate but our fork gives the
        # projection a real init, so the denoiser is trainable from step 0.

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self_cond is None:
            self_cond = torch.zeros_like(z_t)
        x = self.in_proj(torch.cat([z_t, self_cond], dim=-1))
        t = t.reshape(-1).to(z_t.device)
        if t.numel() == 1:
            t = t.expand(z_t.shape[0])
        c = F.silu(
            self.time_mlp(sinusoidal_time_embedding(t * self.config.time_scale, 256))
        )
        cos, sin = self.rotary(z_t.shape[1], z_t.device)
        for block in self.blocks:
            x = block(x, cos, sin, c, None)
        return self.out(x, c)


def ladiff_prior_loss(
    model: LatentPriorDenoiser,
    z: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    t_min: float = 1e-3,
) -> dict:
    """Algorithm 2.  ``z`` must already be standardized (the AE's frozen
    latent statistics are the caller's responsibility — the prior lives
    entirely in standardized space)."""
    schedule = TanhLogSNRSchedule(d=model.config.schedule_d)
    rows = z.shape[0]
    device = z.device

    t = torch.rand(rows, generator=generator).clamp(t_min, 1.0 - t_min).to(device)
    eps = torch.randn(z.shape, generator=generator).to(device)
    wide = (-1,) + (1,) * (z.ndim - 1)
    z_t = schedule.alpha(t).reshape(wide) * z + schedule.sigma(t).reshape(wide) * eps

    self_conditioned = bool(torch.rand((), generator=generator) < 0.5)
    self_cond = None
    if self_conditioned:
        with torch.no_grad():
            self_cond = model(z_t, t, self_cond=None).detach()
    z_hat = model(z_t, t, self_cond=self_cond)
    mse = F.mse_loss(z_hat, z)
    return {"total": mse, "self_conditioned": self_conditioned}


@torch.no_grad()
def sample_latent_prior(
    model,
    *,
    batch: int,
    steps: int,
    gamma: float = 0.0,
    generator: torch.Generator | None = None,
    tau_min: float = 1e-3,
    tau_max: float = 1.0 - 1e-3,
) -> torch.Tensor:
    """Algorithm 3, latent half: Euler with carried self-conditioning.
    Returns STANDARDIZED latents; the caller denormalizes (line 18)."""
    config = model.config
    schedule = TanhLogSNRSchedule(d=config.schedule_d)
    shape = (batch, config.num_latents, config.latent_dim)
    try:
        device = next(model.parameters()).device
    except StopIteration:  # parameter-free stubs in tests
        device = torch.device("cpu")
    z = torch.randn(shape, generator=generator).to(device)
    taus = torch.linspace(tau_max, tau_min, steps + 1)
    z_hat = None
    for m in range(steps):
        tau, tau_next = taus[m], taus[m + 1]
        if gamma > 0:
            tau_next = math.sqrt(1.0 - gamma**2) * tau_next
        t_batch = tau.expand(batch)
        z_hat = model(z, t_batch, self_cond=z_hat)
        v = euler_velocity(schedule, z, z_hat, tau)
        z = z - (tau - tau_next) * v
        if gamma > 0:
            noise = torch.randn(shape, generator=generator).to(z.device)
            z = math.sqrt(1.0 - gamma**2) * z + gamma * noise
    return z
