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

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from unturtle.eval import GenerationEvaluator, MaskedDiffusionEvaluator
from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
from unturtle.kernels.sparse_masked_loss import (
    sparse_masked_diffusion_loss,
    supports_sparse_masked_loss,
)
from unturtle.processes import MaskedDiffusionProcess
from unturtle.trainer import UnturtleTrainer, UnturtleTrainingArguments

from .collator import MaskedDiffusionDataCollator
from .mask_token import (
    classify_batch,
    require_mask_token_id,
    resolve_mask_token_id,
)
from .packed_collator import PackedMaskedDiffusionDataCollator
from .reweighting import cart_loss_weights
from .schedulers import BaseAlphaScheduler, LinearAlphaScheduler, make_alpha_scheduler

logger = logging.getLogger(__name__)

# Batch keys emitted by PackedMaskedDiffusionDataCollator that only a
# packed-aware attention forward consumes.
_PACKED_METADATA_KEYS = ("block_attention_mask", "packed_seq_lengths")


def _model_consumes_packed_metadata(model: torch.nn.Module) -> bool:
    """Best-effort check that ``model``'s forward path consumes packed metadata.

    ``block_attention_mask`` / ``packed_seq_lengths`` ride through
    ``**kwargs``-tolerant transformers forwards without error, so an unpatched
    model silently ignores them — attention is then NOT blocked at packed
    sample boundaries and the loss still decreases (cross-sample
    contamination).  Two signals, in order of reliability:

    1. An *instance-level* ``forward`` override carrying the explicit
       ``_consumes_packed_metadata`` marker.  ``FastDiffusionModel`` patching
       installs fast forwards via ``types.MethodType``; only the packed-aware
       ones (``TinyA2DAttention_fast_forward``) set the marker — other
       unturtle fast forwards (Dream / ModernBERT / LLaDA) never read the
       packed keys, so merely being patched from unturtle code is NOT a
       signal.
    2. A module class whose ``forward`` signature *explicitly* declares one of
       the packed metadata arguments (a backbone with native packed support).
       No current unturtle backbone does; a plain ``**kwargs`` sink is
       deliberately NOT accepted — silently swallowing these kwargs is exactly
       the failure mode this guard exists to catch.

    This is a documented heuristic: a model consuming packed metadata through
    an opaque ``**kwargs`` sink would be reported as non-consuming (the guard
    only warns, it never raises).
    """
    for module in model.modules():
        # Signal 1: MethodType-patched instance forward carrying the explicit
        # packed-aware marker (class-level forwards do not appear in the
        # instance __dict__).
        fwd = module.__dict__.get("forward")
        func = getattr(fwd, "__func__", fwd)
        if func is not None and getattr(func, "_consumes_packed_metadata", False):
            return True
        # Signal 2: explicit packed-metadata parameter in the class forward.
        try:
            signature = inspect.signature(type(module).forward)
        except (TypeError, ValueError):
            continue
        if any(key in signature.parameters for key in _PACKED_METADATA_KEYS):
            return True
    return False


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
    sparse_lm_head: bool = field(
        default=False,
        metadata={
            "help": (
                "Project only masked positions through the LM head instead of "
                "computing full [B, L, V] logits (#61).  Numerically identical "
                "to the dense path.  Defaults off because it is NOT a win at "
                "the ~50%% average mask ratio MDLM training produces: measured "
                "+8%% (32K vocab) to +10%% (128K) peak memory there, versus "
                "-28%% / -41%% at a 15%% mask ratio.  Enable for low-mask-ratio "
                "schedules or when step time matters more than peak memory.  "
                "Requires a backbone declaring 'sparse_output_projection'; "
                "raises at construction otherwise rather than silently "
                "falling back."
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
        self._sparse_lm_head: bool = getattr(args, "sparse_lm_head", False)
        completion_only: bool = getattr(args, "completion_only", True)

        model = kwargs.get("model") or (pargs[0] if pargs else None)

        # Checked at construction, not per step: a silent fallback would turn
        # an explicit opt-in into a no-op that surfaces only as unexplained
        # memory use.
        #
        # Deliberately no guard for logit_softcapping / logit_scaling: neither
        # is a field on DiffusionTrainingArguments, so a check against `args`
        # could never fire and would read as protection that does not exist.
        # `sparse_masked_diffusion_loss` rejects them where they can actually
        # be passed.  Add one here only alongside adding the fields.
        if (
            self._sparse_lm_head
            and model is not None
            and not supports_sparse_masked_loss(model)
        ):
            raise ValueError(
                f"sparse_lm_head=True but {type(model).__name__} does not "
                "declare the 'sparse_output_projection' capability. "
                "Supported today: the Tiny-A2D family. Set "
                "sparse_lm_head=False to use the dense path."
            )

        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        mask_token_id = resolve_mask_token_id(tokenizer, model)

        # The forward process owns corruption (#62).  It runs inside
        # compute_loss, after the Trainer has moved the batch to the
        # accelerator, so noising happens device-side rather than in CPU
        # DataLoader workers.
        self.forward_process: MaskedDiffusionProcess | None = (
            MaskedDiffusionProcess(
                scheduler=self._alpha_scheduler,
                mask_token_id=mask_token_id,
                time_epsilon=self._time_epsilon,
                completion_only=completion_only,
            )
            if mask_token_id is not None
            else None
        )

        # Inject a *clean* collator unless the caller supplied one: padding and
        # supervision only, with the process applying corruption later.
        if ("data_collator" not in kwargs or kwargs["data_collator"] is None) and (
            tokenizer is not None
        ):
            if self.forward_process is None:
                # Clean collator + no process means nothing would ever corrupt
                # the batch.  Unrecoverable, so fail here rather than minutes
                # later on the first compute_loss.
                require_mask_token_id(tokenizer, model, context="DiffusionTrainer")
            kwargs["data_collator"] = self._build_default_collator(
                tokenizer=tokenizer,
                mask_token_id=mask_token_id,
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

        # One-shot guard flag: on the first batch carrying packed metadata,
        # verify the model actually consumes it (see compute_loss).
        self._packed_metadata_checked = False

        # `timestep`/`scheduler` weighting needs a per-sample t.  The *noising*
        # packed collator collapses its samples to a row mean, which is the
        # wrong t for each of them, so those weightings stay barred there.  The
        # clean packed collator emits `segment_ids` and lets the process build
        # a full [B, L] instead, so it is fine (#62 PR3).
        collator = self.data_collator
        if (
            isinstance(collator, PackedMaskedDiffusionDataCollator)
            and getattr(collator, "noise", True)
            and self._loss_weight_type not in ("uniform", "cart")
        ):
            raise ValueError(
                "A *noising* PackedMaskedDiffusionDataCollator is not supported for "
                "diffusion training with loss_weight_type='timestep' or 'scheduler': "
                "it collapses each packed row's per-sample timesteps to a mean. "
                "Pass noise=False so the forward process samples per segment, use "
                "uniform/cart weighting, or use an unpacked collator."
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
        # Packed-batch guard (#57): warn once when packed metadata rides
        # through **kwargs into a model that never consumes it (cross-sample
        # attention; the loss still decreases, hiding the bug).  Warning only —
        # backbones that consume packed metadata through an opaque **kwargs
        # sink must not be broken by a false positive.
        if not getattr(self, "_packed_metadata_checked", False) and any(
            key in inputs for key in _PACKED_METADATA_KEYS
        ):
            self._packed_metadata_checked = True
            if not _model_consumes_packed_metadata(model):
                logger.warning(
                    "DiffusionTrainer: the batch carries packed-attention metadata "
                    "(%s) but the model does not appear to consume it — no "
                    "unturtle fast-attention forward is installed on any module "
                    "and no module forward declares these arguments.  The "
                    "metadata will ride through **kwargs unused, so attention is "
                    "NOT blocked at packed-sample boundaries (cross-sample "
                    "contamination; the loss still decreases, hiding the bug).  "
                    "Load the model via FastDiffusionModel.from_pretrained / "
                    "get_peft_model on CUDA so the packed fast forward is "
                    "installed, or use the unpacked MaskedDiffusionDataCollator.  "
                    "If your backbone consumes packed metadata natively through a "
                    "**kwargs sink, this warning is a false positive.",
                    " / ".join(key for key in _PACKED_METADATA_KEYS if key in inputs),
                )

        inputs = self._apply_forward_process(inputs)

        labels: torch.Tensor = inputs.pop("labels")  # [B, L]
        diffusion_mask: torch.Tensor = inputs.pop("diffusion_mask")
        timesteps: torch.Tensor = inputs.pop("timesteps")

        # Batch metadata consumed (read-only) by CART loss weighting: real-token
        # mask and, for packed batches, per-row sample lengths.  Left in
        # ``inputs`` — the model forward also consumes them.
        attention_mask = inputs.get("attention_mask")
        seq_lengths = inputs.get("seq_lengths")
        if seq_lengths is None:
            flat_lengths = inputs.get("packed_seq_lengths")
            if flat_lengths is not None and labels.shape[0] == 1:
                seq_lengths = [flat_lengths]

        # `getattr` default, not `self._sparse_lm_head`: test fixtures build
        # trainers without running `__init__` (SimpleNamespace / __new__), and
        # an opt-in optimization must never be the reason such a trainer fails
        # to compute a loss at all.  (BlockDiffusionTrainer is NOT an example —
        # it calls super().__init__ and gets the attribute normally.)
        if getattr(self, "_sparse_lm_head", False) and not return_outputs:
            # `return_outputs` forces the dense path: the caller wants the
            # model outputs, and the whole point of the sparse path is that
            # `[B, L, V]` logits are never built.  Silently returning outputs
            # without logits would break callers worse than being slower.
            loss_weights = self._build_loss_weights(
                timesteps,
                labels.device,
                diffusion_mask,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
            )
            forward_kwargs = {
                k: v for k, v in inputs.items() if k not in ("input_ids", "labels")
            }
            return sparse_masked_diffusion_loss(
                model=model,
                input_ids=inputs["input_ids"],
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_weights=loss_weights,
                loss_norm_type=self._loss_norm_type,
                right_shift=self._right_shift_logits,
                **forward_kwargs,
            )

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

        loss_weights = self._build_loss_weights(
            timesteps,
            logits.device,
            diffusion_mask,
            attention_mask=attention_mask,
            seq_lengths=seq_lengths,
        )

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

    def _build_default_collator(
        self,
        tokenizer: Any,
        mask_token_id: int | None,
        completion_only: bool,
    ) -> MaskedDiffusionDataCollator:
        """Build the collator injected when the caller supplies none.

        Overridden by subclasses whose objective needs different collation —
        BD3LM, for instance, must pad to a ``block_size`` multiple with EOS,
        and a plain collator's pad-token/``-100`` padding would silently
        change which positions are maskable.
        """
        return MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            scheduler=self._alpha_scheduler,
            mask_token_id=mask_token_id,
            time_epsilon=self._time_epsilon,
            completion_only=completion_only,
            noise=False,
        )

    def _apply_forward_process(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Corrupt a clean batch device-side, or pass a pre-noised one through.

        Two collator contracts coexist during the #62 migration: the clean
        collator (this trainer's default) emits no ``diffusion_mask``, while
        ``PackedMaskedDiffusionDataCollator`` and any explicitly-passed legacy
        collator still noise during collation.  A batch carrying *both*
        supervision keys is already corrupted and passes through — re-noising
        it would mask a fraction of the mask tokens themselves and silently
        change the objective.

        A batch carrying only one of the two keys is rejected rather than
        guessed at: passing it through dies later on a bare ``KeyError``, and
        noising it would silently discard the caller's ``timesteps``.

        RNG comes from the global torch stream, which Trainer/Accelerate
        already seed via ``set_seed``; this keeps gradient accumulation and
        multi-rank behavior on the existing rails.
        """
        pre_noised = classify_batch(inputs, "DiffusionTrainer")
        if pre_noised:
            return inputs

        if self.forward_process is None:
            raise ValueError(
                "DiffusionTrainer received a clean batch (no 'diffusion_mask') but "
                "has no forward process: mask_token_id could not be resolved from "
                "the tokenizer or model config.  Pass a tokenizer with a mask token, "
                "set model.config.mask_token_id, or supply a noising data_collator."
            )

        output = self.forward_process(inputs)
        return {**output.model_inputs, **output.objective_inputs}

    def _build_loss_weights(
        self,
        timesteps: torch.Tensor,
        device: torch.device,
        diffusion_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        seq_lengths: Any | None = None,
    ) -> torch.Tensor | None:
        """Return per-token loss weights based on ``loss_weight_type``.

        Takes a ``device`` rather than the logits tensor: every weighting is a
        function of ``timesteps`` and the mask, and none reads the vocabulary
        dimension.  Depending on ``[B, L, V]`` merely to reach ``.device``
        would force that tensor to exist on the sparse LM-head path (#61).

        Args:
            attention_mask: optional ``[B, L]`` real-token mask.  Used by CART
                so padding never counts as clean context.
            seq_lengths: optional per-row packed sample lengths (list of int
                tensors/lists, one entry per batch row — the structure the
                packed collator emits as ``seq_lengths``).  Used by CART so
                clean context never crosses packed-sample boundaries.
        """
        if self._loss_weight_type == "uniform":
            return None

        t = timesteps.to(device)

        if self._loss_weight_type == "timestep":
            # d1 SFT: weight = 1/t.  `t` is [B] unpacked (broadcast over L) or
            # [B, L] on a clean packed batch, where each position carries its
            # own segment's t (#62); the kernel accepts either.
            return 1.0 / t.clamp_min(1e-6)  # [B] or [B, L]

        if self._loss_weight_type == "scheduler":
            # MDLM: w(t) = -α'(t) / (1 - α(t))
            # Shape follows `t`: [B] unpacked, [B, L] on a clean packed batch.
            w: torch.Tensor = self._alpha_scheduler.weight(t)
            return w.to(device)

        if self._loss_weight_type == "cart":
            # Dream CART: context-adaptive geometric reweighting (#202).
            # For each masked position n, weight = Σ_i Geo(cart_p, |n-i|-1)
            # summed over clean (unmasked) positions i.
            #
            # Reference: dev/repos/Dream/src/trainer/fsdp_sft_trainer.py L91-115, L805-821
            return cart_loss_weights(
                diffusion_mask,
                cart_p=self._cart_p,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
            )  # [B, L]

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
        # Only the *noising* packed collator is incompatible: it collapses each
        # packed row's per-sample timesteps to a mean.  A clean packed collator
        # defers corruption to the forward process, which samples per segment
        # and emits `[B, L]` (#62).  Narrowed to match the guards in
        # `__init__` and `MaskedDiffusionEvaluator` — before this, a clean
        # packed setup that trained without complaint raised here.
        if (
            isinstance(collator, PackedMaskedDiffusionDataCollator)
            and getattr(collator, "noise", True)
            and self._loss_weight_type not in ("uniform", "cart")
        ):
            raise ValueError(
                "A *noising* PackedMaskedDiffusionDataCollator is not supported for "
                "diffusion evaluation with loss_weight_type='timestep' or "
                "'scheduler': it collapses each packed row's per-sample timesteps "
                "to a mean. Pass noise=False so the forward process samples per "
                "segment, use uniform/cart weighting, or pass an unpacked "
                "MaskedDiffusionDataCollator explicitly."
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
            cart_p=self._cart_p,
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
