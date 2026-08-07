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

from unturtle.diffusion.collator import MaskedDiffusionDataCollator
from unturtle.diffusion.mask_token import resolve_mask_token_id
from unturtle.diffusion.packed_collator import PackedMaskedDiffusionDataCollator
from unturtle.diffusion.schedulers import LinearAlphaScheduler, make_alpha_scheduler
from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
from unturtle.processes import MaskedDiffusionProcess

from .base import BaseEvaluator


class MaskedDiffusionEvaluator(BaseEvaluator):
    """Evaluate masked-diffusion loss metrics on a validation dataset.

    Metrics reported by :meth:`evaluate`:

    - ``eval_loss``: weighted training loss (may use timestep/scheduler weights).
    - ``eval_masked_token_nll``: unweighted per-maskable-token NLL, matching the
      MDLM trainer normalization ``token_nll / maskable_mask.sum()`` (mdlm.py L200).
    - ``eval_perplexity``: ``exp(eval_masked_token_nll)``.
    - ``eval_mask_rate``: fraction of maskable tokens that were actually masked.

    Note: ``eval_masked_token_nll`` is the *training-objective* NLL evaluated on the
    validation set with random masking.  It is **not** the MDLM Monte Carlo
    log-likelihood estimator (which corrects for mask probability via
    ``CE / p_mask[mask_indices]``).  For a direct comparison with MDLM eval metrics,
    use a dedicated Monte Carlo evaluator.
    """

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
        cart_p: float = 0.8,
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        if isinstance(alpha_scheduler, str):
            alpha_scheduler = make_alpha_scheduler(alpha_scheduler)
        self.alpha_scheduler = alpha_scheduler or LinearAlphaScheduler()
        self.loss_weight_type = loss_weight_type
        self.cart_p = cart_p
        self.time_epsilon = time_epsilon
        self.completion_only = completion_only
        self.metric_key_prefix = metric_key_prefix
        mask_token_id = resolve_mask_token_id(tokenizer, model)
        self.mask_token_id = mask_token_id

        # Corruption is applied device-side (#62), mirroring DiffusionTrainer.
        # Kept as None when no mask id resolves so an explicitly-supplied
        # noising collator still evaluates.
        self.forward_process: MaskedDiffusionProcess | None = (
            MaskedDiffusionProcess(
                scheduler=self.alpha_scheduler,
                mask_token_id=mask_token_id,
                time_epsilon=time_epsilon,
                completion_only=completion_only,
            )
            if mask_token_id is not None
            else None
        )

        self.data_collator = data_collator or MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            scheduler=self.alpha_scheduler,
            mask_token_id=mask_token_id,
            time_epsilon=time_epsilon,
            completion_only=completion_only,
            noise=False,
        )
        if (
            isinstance(self.data_collator, PackedMaskedDiffusionDataCollator)
            and self.loss_weight_type != "uniform"
        ):
            raise ValueError(
                "PackedMaskedDiffusionDataCollator is not supported for diffusion evaluation with "
                "loss_weight_type='timestep' or 'scheduler'. Use uniform weighting or an "
                "unpacked MaskedDiffusionDataCollator."
            )

    def _apply_forward_process(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Corrupt a clean batch device-side, or pass a pre-noised one through.

        Mirrors ``DiffusionTrainer._apply_forward_process``.  Presence of
        ``diffusion_mask`` means an explicitly-supplied noising collator (or
        the packed collator) already did the corruption — re-noising would
        mask the mask tokens themselves and skew every reported metric.
        """
        if "diffusion_mask" in batch:
            return batch

        if self.forward_process is None:
            raise ValueError(
                "MaskedDiffusionEvaluator received a clean batch (no "
                "'diffusion_mask') but has no forward process: mask_token_id "
                "could not be resolved from the tokenizer or model config.  "
                "Pass a tokenizer with a mask token, set "
                "model.config.mask_token_id, or supply a noising data_collator."
            )

        output = self.forward_process(batch)
        return {**output.model_inputs, **output.objective_inputs}

    def _build_loss_weights(
        self,
        timesteps: torch.Tensor,
        logits: torch.Tensor,
        diffusion_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
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

        if self.loss_weight_type == "cart":
            # `cart` is a valid DiffusionTrainingArguments value and
            # `DiffusionTrainer.build_diffusion_evaluator` forwards it here, so
            # evaluating a CART-trained model must not crash.  Reuse the
            # trainer's implementation rather than duplicating the geometric
            # reweighting.
            if diffusion_mask is None:
                raise ValueError(
                    "loss_weight_type='cart' needs the diffusion mask; this "
                    "evaluator was called without one."
                )
            from unturtle.diffusion.reweighting import context_adaptive_reweight

            _, L = diffusion_mask.shape
            weight_matrix = context_adaptive_reweight(L, cart_p=self.cart_p).to(device)
            clean_mask = ~diffusion_mask
            if attention_mask is not None:
                clean_mask = clean_mask & attention_mask.to(
                    device=clean_mask.device, dtype=torch.bool
                )
            weight = clean_mask.float().matmul(weight_matrix)
            return weight.masked_fill(~diffusion_mask, 0.0)

        raise ValueError(
            f"Unknown loss_weight_type '{self.loss_weight_type}'. "
            "Choose from: 'uniform', 'timestep', 'scheduler', 'cart'."
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
                batch = self._apply_forward_process(batch)
                labels: torch.Tensor = batch.pop("labels")
                diffusion_mask: torch.Tensor = batch.pop("diffusion_mask")
                timesteps: torch.Tensor = batch.pop("timesteps")

                # Read-only; the model forward consumes it too.
                attention_mask = batch.get("attention_mask")

                outputs = self.model(**batch)
                logits: torch.Tensor = outputs.logits
                loss_weights = self._build_loss_weights(
                    timesteps,
                    logits,
                    diffusion_mask=diffusion_mask,
                    attention_mask=attention_mask,
                )
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
        perplexity = (
            math.exp(avg_unweighted_nll) if avg_unweighted_nll < 80 else float("inf")
        )
        mask_rate = float(total_masked) / denom
        prefix = self.metric_key_prefix

        return {
            self._metric_key(prefix, "loss"): avg_loss,
            self._metric_key(prefix, "masked_token_nll"): avg_unweighted_nll,
            self._metric_key(prefix, "perplexity"): perplexity,
            self._metric_key(prefix, "mask_rate"): mask_rate,
            self._metric_key(prefix, "num_batches"): float(total_batches),
        }
