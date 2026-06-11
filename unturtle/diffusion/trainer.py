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

Extends :class:`~unturtle.trainer.UnturtleTrainer` (which in turn extends TRL's
``SFTTrainer``) with:

  1. A custom ``compute_loss`` that calls ``fast_masked_diffusion_loss``.
  2. Integration with :class:`~.collator.MaskedDiffusionDataCollator` as the
     default data collator.
  3. Support for three loss-weighting modes:
       - ``"uniform"``    – equal weight per masked token (LLaDA / MDLM default)
       - ``"timestep"``   – weight = ``1/t`` per sequence (d1 SFT style)
       - ``"scheduler"``  – weight = ``w(t) = -α'(t)/(1-α(t))`` (MDLM paper)
       - ``"cart"``       – context-adaptive geometric reweighting (Dream)
  4. Support for three loss normalisation modes (``loss_norm_type``):
       - ``"token"``      – divide by total maskable tokens (MDLM / LLaDA default)
       - ``"sequence"``   – per-sequence then mean over B
       - ``"batch"``      – divide by B only
  5. ``right_shift_logits`` for Dream-style AR-to-dLLM continual pre-training.
  6. A companion :class:`DiffusionTrainingArguments` dataclass.

Reference implementations:
  dllm-reasoning/d1   SFT/sft_trainer.py
  zhziszz/dllm        dllm/core/trainers/mdlm.py
  Dream               src/trainer/fsdp_sft_trainer.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from unturtle.eval import GenerationEvaluator, MaskedDiffusionEvaluator
from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
from unturtle.trainer import UnturtleTrainer, UnturtleTrainingArguments

from .collator import MaskedDiffusionDataCollator
from .packed_collator import PackedMaskedDiffusionDataCollator
from .reweighting import context_adaptive_reweight
from .schedulers import BaseAlphaScheduler, LinearAlphaScheduler, make_alpha_scheduler


@dataclass
class DiffusionTrainingArguments(UnturtleTrainingArguments):
    """Training arguments for :class:`DiffusionTrainer`.

    Inherits all fields from :class:`~unturtle.trainer.UnturtleTrainingArguments`
    and adds dLLM-specific options.

    Args:
        alpha_scheduler:    Name of the alpha scheduler.  One of ``"linear"``
                            (default) or ``"cosine"``.
        time_epsilon:       Minimum sampled timestep (avoids ``t → 0``).
        loss_weight_type:   How to weight the per-token loss.
                            ``"uniform"``    – equal weight (LLaDA / MDLM default).
                            ``"timestep"``   – weight = ``1/t`` (d1 SFT style).
                            ``"scheduler"``  – MDLM paper weight ``w(t)``.
                            ``"cart"``       – context-adaptive geometric reweighting
                                               (Dream; requires ``cart_p``).
        cart_p:             Geometric distribution sharpness for CART weighting.
                            Only used when ``loss_weight_type="cart"``.
                            Larger values concentrate weight on nearby clean tokens.
        loss_norm_type:     How to normalise the accumulated per-token loss.
                            ``"token"``      – divide by total maskable tokens (default).
                            ``"sequence"``   – per-sequence, then mean over B.
                            ``"batch"``      – divide by B only.
        completion_only:    Only mask completion tokens, not the prompt.
        right_shift_logits: Apply the Dream Shift Operation during training:
                            shift logits one position right so that ``logit[i]``
                            predicts token at position ``i+1``.  Required when
                            fine-tuning Dream checkpoints (which were pre-trained
                            with this shifted objective).  See Dream paper §3.1
                            and ``dev/repos/Dream/src/trainer/fsdp_sft_trainer.py``.
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
                "'scheduler' (MDLM w(t)), "
                "'cart' (Dream context-adaptive reweighting)."
            )
        },
    )
    cart_p: float = field(
        default=0.8,
        metadata={
            "help": (
                "Geometric distribution sharpness for CART reweighting "
                "(only used when loss_weight_type='cart'). "
                "Range (0, 1]; larger = more local concentration."
            )
        },
    )
    loss_norm_type: str = field(
        default="token",
        metadata={
            "help": (
                "Loss normalisation: "
                "'token' (total maskable tokens, MDLM default), "
                "'sequence' (per-sequence then mean over B), "
                "'batch' (divide by B only)."
            )
        },
    )
    completion_only: bool = field(
        default=True,
        metadata={"help": "Only mask completion tokens (not the prompt)."},
    )
    right_shift_logits: bool = field(
        default=False,
        metadata={
            "help": (
                "Apply Dream Shift Operation: shift logits one position right "
                "so logit[i] predicts position i+1.  Must be True when "
                "fine-tuning Dream checkpoints."
            )
        },
    )


class DiffusionTrainer(UnturtleTrainer):
    """Unturtle trainer for masked diffusion language models.

    Wraps the Triton-optimised ``fast_masked_diffusion_loss`` and wires in
    :class:`~.collator.MaskedDiffusionDataCollator` automatically.

    Args:
        args: A :class:`DiffusionTrainingArguments` instance.
        All other kwargs are forwarded to ``UnturtleTrainer`` / ``SFTTrainer``.

    Example (LLaDA / MDLM style SFT)::

        from unturtle import FastLanguageModel
        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

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

    Example (Dream fine-tuning — requires ``right_shift_logits=True``)::

        model, tokenizer = FastLanguageModel.from_pretrained(
            "Dream-org/Dream-v0-Instruct-7B", load_in_4bit=True
        )
        model = FastLanguageModel.get_peft_model(model, r=16)

        args = DiffusionTrainingArguments(
            output_dir="output",
            num_train_epochs=3,
            loss_weight_type="cart",      # Dream-style CART reweighting
            cart_p=0.8,
            right_shift_logits=True,      # required for Dream checkpoints
        )
        trainer = DiffusionTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=dataset,
        )
        trainer.train()

    Example (LLaDA pre-training / continual pre-training — full-sequence masking)::

        # Dataset items need only ``input_ids`` (no prompt/completion split).
        # All non-padding tokens are eligible for masking.  This matches the
        # LLaDA pre-training objective (arxiv 2502.09992 §3.1) and LLaDA 2.0
        # continual pre-training (arxiv 2512.15745 §3).

        model, tokenizer = FastLanguageModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Base", load_in_4bit=True
        )
        model = FastLanguageModel.get_peft_model(model, r=16)

        args = DiffusionTrainingArguments(
            output_dir="output",
            num_train_epochs=1,
            loss_weight_type="uniform",
            completion_only=False,   # mask all tokens, not just completion
        )
        trainer = DiffusionTrainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=dataset,   # items: {"input_ids": [...]}
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
        self._cart_p: float = getattr(args, "cart_p", 0.8)
        self._loss_norm_type: str = getattr(args, "loss_norm_type", "token")
        self._right_shift_logits: bool = getattr(args, "right_shift_logits", False)
        completion_only: bool = getattr(args, "completion_only", True)

        model = kwargs.get("model") or (pargs[0] if pargs else None)

        # Inject MaskedDiffusionDataCollator unless the caller supplied one
        if "data_collator" not in kwargs or kwargs["data_collator"] is None:
            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if tokenizer is not None:
                mask_token_id = getattr(tokenizer, "mask_token_id", None)
                if mask_token_id is None:
                    mask_token_id = getattr(
                        getattr(model, "config", None), "mask_token_id", None
                    )
                kwargs["data_collator"] = MaskedDiffusionDataCollator(
                    tokenizer=tokenizer,
                    scheduler=self._alpha_scheduler,
                    mask_token_id=mask_token_id,
                    time_epsilon=self._time_epsilon,
                    completion_only=completion_only,
                )

        super().__init__(*pargs, **kwargs)

        # DiffusionTrainer does NOT use num_items_in_batch in compute_loss.
        # Setting model_accepts_loss_kwargs=False tells the Transformers Trainer to apply
        # its standard gradient-accumulation scaling (loss / current_gradient_accumulation_steps).
        # Note: fused_masked_diffusion_loss already normalizes by n_maskable per microbatch,
        # so the effective accumulated loss is a mean-of-microbatch-means (approximate token
        # weighting when token counts vary across microbatches).  This matches the d1/MDLM
        # reference trainer behavior.  See transformers Trainer.training_step L1925.
        self.model_accepts_loss_kwargs = False

        if isinstance(
            self.data_collator, PackedMaskedDiffusionDataCollator
        ) and self._loss_weight_type not in ("uniform", "cart"):
            raise ValueError(
                "PackedMaskedDiffusionDataCollator is not supported for diffusion training with "
                "loss_weight_type='timestep' or 'scheduler'. Use uniform/cart weighting or an "
                "unpacked MaskedDiffusionDataCollator."
            )

    # ------------------------------------------------------------------ #
    #  Loss computation                                                   #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute the masked diffusion CE loss using the Triton kernel.

        Expects ``inputs`` to contain:
          ``input_ids``      – noised token ids (from the data collator)
          ``labels``         – clean token ids (``x_0``); prompt/padding may be ``-100``
          ``diffusion_mask`` – bool tensor, True at masked positions
          ``timesteps``      – sampled ``t``, shape ``(B,)``
        """
        labels: torch.Tensor = inputs.pop("labels")  # [B, L]
        diffusion_mask: torch.Tensor = inputs.pop("diffusion_mask")
        timesteps: torch.Tensor = inputs.pop("timesteps")

        outputs = model(**inputs)
        logits: torch.Tensor = outputs.logits  # [B, L, V]

        # Dream Shift Operation (#201): shift logits one position right so that
        # logit[i] predicts the token at position i+1.  This matches the shifted
        # objective used during Dream pre-training and is required for correct
        # fine-tuning of Dream checkpoints.
        #
        # Reference: dev/repos/Dream/src/trainer/fsdp_sft_trainer.py L777-779
        #   shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
        # Reference: dev/repos/dllm/dllm/core/trainers/mdlm.py _postprocess_outputs
        if self._right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1).contiguous()

        loss_weights = self._build_loss_weights(timesteps, logits, diffusion_mask)

        loss = fast_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_weights=loss_weights,
            loss_norm_type=self._loss_norm_type,
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

        device = logits.device
        t = timesteps.to(device)
        _, L = diffusion_mask.shape

        if self._loss_weight_type == "timestep":
            # d1 SFT: weight = 1/t per sequence, broadcast over L
            return 1.0 / t.clamp_min(1e-6)  # [B]

        if self._loss_weight_type == "scheduler":
            # MDLM: w(t) = -α'(t) / (1 - α(t))
            w: torch.Tensor = self._alpha_scheduler.weight(t)  # [B]
            return w.to(device)

        if self._loss_weight_type == "cart":
            # Dream CART: context-adaptive geometric reweighting (#202).
            # For each masked position n, weight = Σ_i Geo(cart_p, |n-i|-1)
            # summed over clean (unmasked) positions i.
            #
            # Reference: dev/repos/Dream/src/trainer/fsdp_sft_trainer.py L91-115, L805-821
            weight_matrix = context_adaptive_reweight(L, cart_p=self._cart_p).to(device)
            # clean positions: maskable but NOT currently masked
            # diffusion_mask is True where token is masked
            clean_mask = ~diffusion_mask  # [B, L] — True at clean/unmasked positions
            # weight[b, n] = sum of geometric weights from clean positions to n
            weight = clean_mask.float().matmul(weight_matrix)  # [B, L]
            # masked positions that are themselves clean get zero weight
            weight = weight.masked_fill(clean_mask, 0.0)
            return weight  # [B, L]

        raise ValueError(
            f"Unknown loss_weight_type '{self._loss_weight_type}'. "
            "Choose from: 'uniform', 'timestep', 'scheduler', 'cart'."
        )

    def build_diffusion_evaluator(
        self,
        tokenizer: Any | None = None,
        data_collator: Any | None = None,
        metric_key_prefix: str = "eval",
        **kwargs: Any,
    ) -> MaskedDiffusionEvaluator:
        tokenizer = (
            tokenizer
            or getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
        )
        if tokenizer is None:
            raise ValueError(
                "Tokenizer or processing_class is required to build a diffusion evaluator."
            )

        collator = data_collator or self.data_collator
        if isinstance(
            collator, PackedMaskedDiffusionDataCollator
        ) and self._loss_weight_type not in ("uniform", "cart"):
            raise ValueError(
                "PackedMaskedDiffusionDataCollator is not supported for diffusion evaluation with "
                "loss_weight_type='timestep' or 'scheduler'. Use uniform/cart weighting or pass an "
                "unpacked MaskedDiffusionDataCollator explicitly."
            )

        return MaskedDiffusionEvaluator(
            model=self.model,
            tokenizer=tokenizer,
            data_collator=collator,
            loss_weight_type=self._loss_weight_type,
            alpha_scheduler=self._alpha_scheduler,
            time_epsilon=self._time_epsilon,
            completion_only=getattr(self.args, "completion_only", True),
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

    def build_generation_evaluator(
        self,
        tokenizer: Any | None = None,
        metric_key_prefix: str = "gen",
        **kwargs: Any,
    ) -> GenerationEvaluator:
        tokenizer = (
            tokenizer
            or getattr(self, "processing_class", None)
            or getattr(self, "tokenizer", None)
        )

        return GenerationEvaluator(
            model=self.model,
            tokenizer=tokenizer,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

    def evaluate_diffusion(
        self,
        dataset: Any,
        batch_size: int = 1,
        max_batches: int | None = None,
        metric_key_prefix: str = "eval",
        **kwargs: Any,
    ) -> dict[str, float]:
        evaluator = self.build_diffusion_evaluator(
            metric_key_prefix=metric_key_prefix, **kwargs
        )
        return evaluator.evaluate(
            dataset=dataset, batch_size=batch_size, max_batches=max_batches
        )

    def evaluate_generation(
        self,
        dataset: Any,
        generation_config: Any | None = None,
        max_examples: int | None = None,
        metric_key_prefix: str = "gen",
        **kwargs: Any,
    ) -> dict[str, float]:
        evaluator = self.build_generation_evaluator(
            metric_key_prefix=metric_key_prefix, **kwargs
        )
        return evaluator.evaluate(
            dataset=dataset,
            generation_config=generation_config,
            max_examples=max_examples,
        )
