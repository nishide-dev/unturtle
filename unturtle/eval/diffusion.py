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

from __future__ import annotations

import math
from typing import Any

import torch

from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
from unturtle.diffusion.collator import MaskedDiffusionDataCollator
from unturtle.diffusion.packed_collator import PackedMaskedDiffusionDataCollator
from unturtle.diffusion.schedulers import LinearAlphaScheduler, make_alpha_scheduler

from .base import BaseEvaluator


class MaskedDiffusionEvaluator(BaseEvaluator):
    """Evaluate masked-diffusion loss metrics on a validation dataset."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        data_collator: Any | None = None,
        loss_weight_type: str = "uniform",
        alpha_scheduler: Any | None = None,
        time_epsilon: float = 1e-3,
        completion_only: bool = True,
        metric_key_prefix: str = "eval",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        if isinstance(alpha_scheduler, str):
            alpha_scheduler = make_alpha_scheduler(alpha_scheduler)
        self.alpha_scheduler = alpha_scheduler or LinearAlphaScheduler()
        self.loss_weight_type = loss_weight_type
        self.time_epsilon = time_epsilon
        self.completion_only = completion_only
        self.metric_key_prefix = metric_key_prefix
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(getattr(model, "config", None), "mask_token_id", None)
        self.data_collator = data_collator or MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            scheduler=self.alpha_scheduler,
            mask_token_id=mask_token_id,
            time_epsilon=time_epsilon,
            completion_only=completion_only,
        )
        if isinstance(self.data_collator, PackedMaskedDiffusionDataCollator) and self.loss_weight_type != "uniform":
            raise ValueError(
                "PackedMaskedDiffusionDataCollator is not supported for diffusion evaluation with "
                "loss_weight_type='timestep' or 'scheduler'. Use uniform weighting or an "
                "unpacked MaskedDiffusionDataCollator."
            )

    def _build_loss_weights(
        self,
        timesteps: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.loss_weight_type == "uniform":
            return None

        device = logits.device
        t = timesteps.to(device)

        if self.loss_weight_type == "timestep":
            return 1.0 / t.clamp_min(1e-6)

        if self.loss_weight_type == "scheduler":
            weights = self.alpha_scheduler.weight(t)
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, device=device)
            return weights.to(device)

        raise ValueError(
            f"Unknown loss_weight_type '{self.loss_weight_type}'. "
            "Choose from: 'uniform', 'timestep', 'scheduler'."
        )

    def evaluate(
        self,
        dataset: Any,
        batch_size: int = 1,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        dataloader = self._make_dataloader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.data_collator,
        )

        total_loss = 0.0
        total_unweighted_nll = 0.0
        total_maskable = 0
        total_masked = 0
        total_batches = 0

        with self.evaluation_mode():
            for batch in dataloader:
                if max_batches is not None and total_batches >= max_batches:
                    break

                batch = self._move_to_device(batch)
                labels: torch.Tensor = batch.pop("labels")
                diffusion_mask: torch.Tensor = batch.pop("diffusion_mask")
                timesteps: torch.Tensor = batch.pop("timesteps")

                outputs = self.model(**batch)
                logits: torch.Tensor = outputs.logits
                loss_weights = self._build_loss_weights(timesteps, logits)
                loss = fast_masked_diffusion_loss(
                    logits=logits,
                    labels=labels,
                    diffusion_mask=diffusion_mask,
                    loss_weights=loss_weights,
                )
                unweighted_nll = fast_masked_diffusion_loss(
                    logits=logits,
                    labels=labels,
                    diffusion_mask=diffusion_mask,
                    loss_weights=None,
                )

                maskable = int((labels != -100).sum().item())
                masked = int(diffusion_mask.sum().item())
                total_loss += float(loss.item()) * max(maskable, 1)
                total_unweighted_nll += float(unweighted_nll.item()) * max(maskable, 1)
                total_maskable += maskable
                total_masked += masked
                total_batches += 1

        denom = max(total_maskable, 1)
        avg_loss = total_loss / denom
        avg_unweighted_nll = total_unweighted_nll / denom
        perplexity = math.exp(avg_unweighted_nll) if avg_unweighted_nll < 80 else float("inf")
        mask_rate = float(total_masked) / denom
        prefix = self.metric_key_prefix

        return {
            self._metric_key(prefix, "loss"): avg_loss,
            self._metric_key(prefix, "masked_token_nll"): avg_unweighted_nll,
            self._metric_key(prefix, "perplexity"): perplexity,
            self._metric_key(prefix, "mask_rate"): mask_rate,
            self._metric_key(prefix, "num_batches"): float(total_batches),
        }
