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
DiffusionTrainer – Unturtle trainer for masked diffusion language models.

Extends :class:`~unsloth.trainer.UnslothTrainer` (which in turn extends TRL's
``SFTTrainer``) with:

  1. A custom ``compute_loss`` that calls ``fast_masked_diffusion_loss``.
  2. Integration with :class:`~.collator.MaskedDiffusionDataCollator` as the
     default data collator.
  3. Support for three loss-weighting modes:
       - ``"uniform"``    – equal weight per masked token (LLaDA / MDLM default)
       - ``"timestep"``   – weight = ``1/t`` per sequence (d1 SFT style)
       - ``"scheduler"``  – weight = ``w(t) = -α'(t)/(1-α(t))`` (MDLM paper)
  4. A companion :class:`DiffusionTrainingArguments` dataclass.

Reference implementations:
  dllm-reasoning/d1   SFT/sft_trainer.py
  zhziszz/dllm        dllm/core/trainers/mdlm.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

from .collator import MaskedDiffusionDataCollator
from .schedulers import BaseAlphaScheduler, LinearAlphaScheduler, make_alpha_scheduler


@dataclass
class DiffusionTrainingArguments(UnslothTrainingArguments):
    """Training arguments for :class:`DiffusionTrainer`.

    Inherits all fields from :class:`~unsloth.trainer.UnslothTrainingArguments`
    and adds dLLM-specific options.

    Args:
        alpha_scheduler:  Name of the alpha scheduler.  One of ``"linear"``
                          (default) or ``"cosine"``.
        time_epsilon:     Minimum sampled timestep (avoids ``t → 0``).
        loss_weight_type: How to weight the per-token loss.
                          ``"uniform"``   – equal weight (LLaDA / MDLM default).
                          ``"timestep"``  – weight = ``1/t`` (d1 SFT style).
                          ``"scheduler"`` – MDLM paper weight ``w(t)``.
        completion_only:  Only mask completion tokens, not the prompt.
    """

    alpha_scheduler: str = field(
        default="linear",
        metadata={"help": "Alpha scheduler: 'linear' or 'cosine'."},
    )
    time_epsilon: float = field(
        default=1e-3,
        metadata={"help": "Minimum timestep value to avoid degenerate t→0."},
    )
    loss_weight_type: str = field(
        default="uniform",
        metadata={
            "help": (
                "Per-token loss weighting: "
                "'uniform' (LLaDA/MDLM), "
                "'timestep' (1/t, d1 SFT), "
                "'scheduler' (MDLM w(t))."
            )
        },
    )
    completion_only: bool = field(
        default=True,
        metadata={"help": "Only mask completion tokens (not the prompt)."},
    )


class DiffusionTrainer(UnslothTrainer):
    """Unturtle trainer for masked diffusion language models.

    Wraps the Triton-optimised ``fast_masked_diffusion_loss`` and wires in
    :class:`~.collator.MaskedDiffusionDataCollator` automatically.

    Args:
        args: A :class:`DiffusionTrainingArguments` instance.
        All other kwargs are forwarded to ``UnslothTrainer`` / ``SFTTrainer``.

    Example::

        from unsloth import FastLanguageModel
        from unsloth.diffusion import DiffusionTrainer, DiffusionTrainingArguments

        model, tokenizer = FastLanguageModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct", load_in_4bit=True
        )
        model = FastLanguageModel.get_peft_model(model, r=16)

        args = DiffusionTrainingArguments(
            output_dir="output",
            num_train_epochs=3,
            alpha_scheduler="linear",
            loss_weight_type="uniform",
        )
        trainer = DiffusionTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=dataset,
        )
        trainer.train()
    """

    def __init__(self, *pargs: Any, **kwargs: Any) -> None:
        # Extract DiffusionTrainingArguments (may have been passed positionally)
        args: DiffusionTrainingArguments | None = kwargs.get("args")
        if args is None and len(pargs) > 1:
            args = pargs[1]  # SFTTrainer(model, args, ...)

        # Build the alpha scheduler
        scheduler_name: str = getattr(args, "alpha_scheduler", "linear")
        self._alpha_scheduler: BaseAlphaScheduler = make_alpha_scheduler(scheduler_name)

        self._time_epsilon: float = getattr(args, "time_epsilon", 1e-3)
        self._loss_weight_type: str = getattr(args, "loss_weight_type", "uniform")
        completion_only: bool = getattr(args, "completion_only", True)

        # Inject MaskedDiffusionDataCollator unless the caller supplied one
        if "data_collator" not in kwargs or kwargs["data_collator"] is None:
            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if tokenizer is not None:
                kwargs["data_collator"] = MaskedDiffusionDataCollator(
                    tokenizer=tokenizer,
                    scheduler=self._alpha_scheduler,
                    time_epsilon=self._time_epsilon,
                    completion_only=completion_only,
                )

        super().__init__(*pargs, **kwargs)

    # ------------------------------------------------------------------ #
    #  Loss computation                                                   #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the masked diffusion CE loss using the Triton kernel.

        Expects ``inputs`` to contain:
          ``input_ids``      – noised token ids (from the data collator)
          ``labels``         – clean token ids (``x_0``); ``-100`` at unmasked positions
          ``diffusion_mask`` – bool tensor, True at masked positions
          ``timesteps``      – sampled ``t``, shape ``(B,)``
        """
        labels: torch.Tensor = inputs.pop("labels")  # [B, L]
        diffusion_mask: torch.Tensor = inputs.pop("diffusion_mask")
        timesteps: torch.Tensor = inputs.pop("timesteps")

        outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits  # [B, L, V]

        loss_weights = self._build_loss_weights(timesteps, logits, diffusion_mask)

        loss = fast_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_weights=loss_weights,
        )

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_loss_weights(
        self,
        timesteps: torch.Tensor,
        logits: torch.Tensor,
        diffusion_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return per-token loss weights based on ``loss_weight_type``."""
        if self._loss_weight_type == "uniform":
            return None

        B = logits.shape[0]
        device = logits.device
        t = timesteps.to(device)

        if self._loss_weight_type == "timestep":
            # d1 SFT: weight = 1/t per sequence, broadcast over L
            return (1.0 / t.clamp_min(1e-6))  # [B]

        if self._loss_weight_type == "scheduler":
            # MDLM: w(t) = -α'(t) / (1 - α(t))
            w: torch.Tensor = self._alpha_scheduler.weight(t)  # [B]
            return w.to(device)

        raise ValueError(
            f"Unknown loss_weight_type '{self._loss_weight_type}'. "
            "Choose from: 'uniform', 'timestep', 'scheduler'."
        )
