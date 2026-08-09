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
Continuous flow-matching forward process (#66).

Third implementation of the ``ForwardProcess`` contract, per the RFC
(``docs/rfcs/continuous-latent.md``): the training state is a continuous
latent and the supervision is a target tensor.  FlowLM Algorithm 1
(arXiv:2605.20199) fixes the specifics:

- linear interpolation path ``z_t = (1 - t) * z_0 + t * eps``,
  ``eps ~ N(0, I)`` — **t = 1 is noise, t = 0 is data**, the reverse of the
  masked convention (do not reuse ``alpha(t)``; there is no absorbing state);
- time sampled uniformly from the discrete grid ``{1/T, .., T/T}`` — the
  paper cuts T from 2000 to 20 to align training with the few-step sampling
  target and reports uniform (not loss-aware) sampling as essential;
- the model-facing time is ``t * time_scale`` (paper: x1000, preserving a
  DiffuSeq-pretrained conditioning range) while ``objective_inputs`` keeps
  the unscaled ``t`` for the 1/t^2 regularizer and the solver math.

The batch must already carry ``latents`` — encoding is the codec's job,
upstream of this process, so no collator ever handles continuous tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import ProcessOutput


@dataclass
class ContinuousFlowProcess:
    """Linear-interpolation corruption of an already-encoded latent batch.

    Args:
        num_timesteps: Size ``T`` of the discrete time grid.  The sampled
                       ``t`` is ``k/T`` with ``k ~ Uniform({1, .., T})``;
                       ``t = 0`` never occurs (nothing to denoise, and the
                       average-velocity sampler divides by ``t``).
        time_scale:    Multiplier applied to the model-facing timesteps only.
    """

    num_timesteps: int = 20
    time_scale: float = 1000.0

    def __post_init__(self) -> None:
        if self.num_timesteps < 1:
            raise ValueError(f"num_timesteps must be >= 1, got {self.num_timesteps}")
        if self.time_scale <= 0:
            raise ValueError(f"time_scale must be > 0, got {self.time_scale}")

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> ProcessOutput:
        latents = batch.get("latents")
        if latents is None:
            raise ValueError(
                "ContinuousFlowProcess needs `latents` in the batch; encoding "
                "token ids is the codec's job, upstream of this process"
            )
        if not latents.dtype.is_floating_point:
            raise ValueError(
                "`latents` must be a floating-point tensor; got "
                f"{latents.dtype} — token ids belong to the codec, not the "
                "process (Gaussian noise on integer ids is finite, plausible "
                "and wrong)"
            )

        rows = latents.shape[0]
        steps = torch.randint(
            1,
            self.num_timesteps + 1,
            (rows,),
            generator=generator,
            device=latents.device if generator is None else generator.device,
        ).to(latents.device)
        t = steps.to(latents.dtype) / self.num_timesteps

        noise = torch.randn(
            latents.shape,
            generator=generator,
            dtype=latents.dtype,
            device=latents.device if generator is None else generator.device,
        ).to(latents.device)

        t_broadcast = t.view(-1, *([1] * (latents.ndim - 1)))
        noised = (1 - t_broadcast) * latents + t_broadcast * noise

        model_inputs: dict[str, Any] = {
            key: value for key, value in batch.items() if key != "latents"
        }
        model_inputs["latents"] = noised
        model_inputs["timesteps"] = t * self.time_scale

        return ProcessOutput(
            model_inputs=model_inputs,
            objective_inputs={
                "target_latents": latents,
                "noise": noise,
                "timesteps": t,
            },
        )


__all__ = ["ContinuousFlowProcess"]
