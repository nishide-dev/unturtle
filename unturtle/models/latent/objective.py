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
FlowLM objective (#66): x0 MSE + codec-owned auxiliary terms + optional
reference regularizer.

Algorithm 1's total loss, term by term::

    ||z_0 - z_0,pred||^2                      x0 prediction (x-pred + x-loss;
                                              v-pred measured unstable, Fig. 3)
    + CE(decoder_head(z_0), w)                codec-owned; arrives here as a
                                              NAMED entry in `auxiliary_losses`
    + reg_rate * ||pred_ref - pred||^2 / t^2  fine-tuning-only anchor to the
                                              frozen source diffusion model
                                              (§3.3, prevents policy collapse)

Returned as a dict so the caller weights and logs terms by name — the
explicit-objective posture the issue demands instead of trainer flags.
There is deliberately no Trainer subclass here: the prototype trains with a
plain loop, and a fused Triton kernel would buy nothing (memory-bound
elementwise work; RFC acceleration inventory).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flowlm_loss(
    prediction: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    auxiliary_losses: dict[str, torch.Tensor] | None = None,
    reference_pred: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    reg_rate: float = 0.0,
) -> dict[str, torch.Tensor]:
    """FlowLM total loss, by named term plus ``"total"``."""
    losses: dict[str, torch.Tensor] = {"x0_mse": F.mse_loss(prediction, target_latents)}

    if auxiliary_losses:
        for name, value in auxiliary_losses.items():
            # "total" is checked explicitly: it is not in `losses` yet at
            # this point, and letting it through would sum the term but then
            # silently overwrite its named entry — a caller logging per-term
            # losses loses the term while the sum quietly includes it.
            if name in losses or name in ("total", "reference_reg"):
                raise ValueError(f"auxiliary loss name collides: {name!r}")
            losses[name] = value

    if reference_pred is not None and reg_rate != 0.0:
        if timesteps is None:
            raise ValueError(
                "the reference regularizer is weighted by 1/t^2 and needs "
                "`timesteps` (the unscaled t from the process)"
            )
        per_row = ((reference_pred.detach() - prediction) ** 2).flatten(1).mean(dim=1)
        losses["reference_reg"] = reg_rate * (per_row / timesteps**2).mean()

    total = losses["x0_mse"]
    for name, value in losses.items():
        if name != "x0_mse":
            total = total + value
    losses["total"] = total
    return losses


__all__ = ["flowlm_loss"]
