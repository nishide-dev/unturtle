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
BlockDiffusionTrainer – Unturtle trainer for the BD3LM block diffusion objective.

Extends :class:`~.trainer.DiffusionTrainer` by overriding ``compute_loss`` to
implement the BD3LM training objective described in:

    "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models"
    https://arxiv.org/abs/2503.09573

The key idea: at each training step the input sequence is duplicated into a
concatenated ``[x_t, x_0]`` sequence of length ``2L``.  A block-structured
attention mask (implemented in :mod:`~.block_attention`) constrains each noised
block to attend to itself and to previously-clean blocks.  The loss is computed
only on the first-half (noised) logits ``logits[:, :L]``, matching the BD3LM
objective.

Reference implementations:
    dev/repos/dllm/dllm/core/trainers/bd3lm.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

from .block_attention import create_block_diffusion_attention_mask
from .block_diffusion_collator import BlockDiffusionDataCollator
from .mask_token import classify_batch
from .trainer import DiffusionTrainer, DiffusionTrainingArguments


@dataclass
class BlockDiffusionTrainingArguments(DiffusionTrainingArguments):
    """Training arguments for :class:`BlockDiffusionTrainer`.

    Inherits all fields from :class:`~.trainer.DiffusionTrainingArguments`
    and adds the BD3LM block-size parameter.

    Args:
        block_size: Size of each block for the block-wise partition.  Every
                    sequence length must be divisible by this value; use
                    :class:`~.block_diffusion_collator.BlockDiffusionDataCollator`
                    to enforce alignment automatically.
    """

    block_size: int = field(
        default=32,
        metadata={"help": "Block size for block-wise partitioning (BD3LM)."},
    )


class BlockDiffusionTrainer(DiffusionTrainer):
    """Unturtle trainer implementing the BD3LM block diffusion training objective.

    Extends :class:`~.trainer.DiffusionTrainer` with a ``compute_loss`` override
    that:

    1. Concatenates the noised sequence ``x_t`` and the clean sequence ``x_0``
       into a single ``[x_t, x_0]`` input of length ``2L``.
    2. Builds a block-structured attention mask of shape ``(B, 1, 2L, 2L)``
       using :func:`~.block_attention.create_block_diffusion_attention_mask`.
    3. Creates duplicated position IDs ``[0..L-1, 0..L-1]`` so both halves
       share the same positional encoding.
    4. Runs the forward pass on the concatenated sequence.
    5. Computes the diffusion loss on the first ``L`` logits only (noised half).

    Args:
        args: A :class:`BlockDiffusionTrainingArguments` instance.
        All other kwargs are forwarded to :class:`~.trainer.DiffusionTrainer`.

    Example::

        from unturtle.diffusion import BlockDiffusionTrainer, BlockDiffusionTrainingArguments
        from unturtle.diffusion.block_diffusion_collator import BlockDiffusionDataCollator
        from unturtle.diffusion.schedulers import LinearAlphaScheduler

        args = BlockDiffusionTrainingArguments(
            output_dir="output",
            block_size=32,
            loss_weight_type="uniform",
        )
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=32,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=tokenizer.mask_token_id,
        )
        trainer = BlockDiffusionTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )
        trainer.train()
    """

    def __init__(self, *pargs: Any, **kwargs: Any) -> None:
        # Extract block_size from args before delegating to super().__init__.
        # super().__init__ (DiffusionTrainer) reads the same args object for its
        # own attributes, so we just read block_size here and store it.
        args: BlockDiffusionTrainingArguments | None = kwargs.get("args")
        if args is None and len(pargs) > 1:
            args = pargs[1]

        self._block_size: int = getattr(args, "block_size", 32)

        super().__init__(*pargs, **kwargs)

    def _build_default_collator(
        self,
        tokenizer: Any,
        mask_token_id: int | None,
        completion_only: bool,
    ) -> BlockDiffusionDataCollator:
        """Inject a block-aware collator rather than the plain masked one.

        BD3LM needs every sequence padded to a ``block_size`` multiple with
        EOS (real, maskable tokens).  A plain collator pads with pad tokens,
        ``attention_mask=0`` and ``-100`` labels, so those positions become
        unmaskable — and when the batch length already happens to be a
        multiple of ``block_size`` the divisibility check in ``compute_loss``
        passes and the objective changes with no error.
        """
        return BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            scheduler=self._alpha_scheduler,
            mask_token_id=mask_token_id,
            time_epsilon=self._time_epsilon,
            completion_only=completion_only,
            block_size=self._block_size,
            noise=False,
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
        """Compute the BD3LM loss on the concatenated [x_t, x_0] sequence.

        Accepts either collator contract (#62).  From the clean
        :class:`~.block_diffusion_collator.BlockDiffusionDataCollator`:

          ``input_ids``      – clean token ids ``x_0``, shape ``(B, L)``
          ``labels``         – supervision targets; ``-100`` at prompt/padding
          ``attention_mask`` – standard padding mask ``(B, L)`` (consumed here;
                               replaced by the block attention mask)

        the forward process then produces ``x_t``, ``diffusion_mask``, and
        ``timesteps`` device-side.  A pre-noised batch (packed collator, or a
        legacy noising collator passed explicitly) is passed through untouched
        and ``x_0`` is reconstructed from ``labels`` as before.
        """
        # --- 1. Extract diffusion-specific keys ---
        # With the clean collator the batch still holds the true x_0 here, so
        # capture it before the process overwrites input_ids with x_t (#62).
        # Uses the same classifier as the base trainer so the two cannot
        # disagree about which contract a batch follows.
        clean_input_ids: torch.Tensor | None = (
            None
            if classify_batch(inputs, "BlockDiffusionTrainer")
            else inputs["input_ids"].clone()
        )

        inputs = self._apply_forward_process(inputs)

        labels: torch.Tensor = inputs.pop("labels")  # [B, L]
        diffusion_mask: torch.Tensor = inputs.pop("diffusion_mask")  # [B, L]
        timesteps: torch.Tensor = inputs.pop("timesteps")  # [B]

        # --- 2. Extract the original attention_mask ---
        # The block attention mask replaces it for the forward pass, but CART
        # loss weighting still needs the real-token mask so padding never
        # counts as clean context.
        padding_mask = inputs.pop("attention_mask", None)

        # --- 3. Obtain x_0 (directly on the clean path, else reconstructed) ---
        noised_ids: torch.Tensor = inputs.pop("input_ids")  # [B, L]
        B, L = noised_ids.shape

        if L % self._block_size != 0:
            raise ValueError(
                f"Sequence length ({L}) must be divisible by block_size "
                f"({self._block_size}). Use BlockDiffusionDataCollator with "
                "the same block_size to pad sequences before training."
            )

        if clean_input_ids is not None:
            # Clean-collator path: x_0 comes straight from the batch, so no
            # inference is needed.  (The reconstruction below is also correct
            # for the process's own output — it never leaves a position both
            # masked and labeled -100 — but taking x_0 directly keeps the
            # invariant local instead of spread across two components.)
            clean_ids = clean_input_ids
        else:
            # Pre-noised batch (packed or an explicitly-passed legacy collator):
            # x_0 must be reconstructed.  Where labels == -100 (prompt/padding)
            # use the noised ids; at masked positions labels already hold x_0,
            # and at unmasked positions input_ids == labels.
            clean_ids = torch.where(labels == -100, noised_ids, labels)  # [B, L]

        # --- 4. Concatenate [x_t, x_0] → (B, 2L) ---
        concat_ids = torch.cat([noised_ids, clean_ids], dim=1)  # [B, 2L]

        # --- 5. Build block-diffusion attention mask (B, 1, 2L, 2L) ---
        block_attn_mask = (
            create_block_diffusion_attention_mask(
                seq_len=L,
                block_size=self._block_size,
                device=noised_ids.device,
            )
            .expand(B, 1, 2 * L, 2 * L)
            .contiguous()
        )  # [B, 1, 2L, 2L]

        # --- 6. Build position IDs: duplicate [0..L-1] for both halves ---
        pos_half = torch.arange(L, device=noised_ids.device).unsqueeze(0)  # [1, L]
        concat_pos = pos_half.repeat(B, 2)  # [B, 2L]

        # --- 7. Forward pass on concatenated sequence ---
        outputs = model(
            input_ids=concat_ids,
            attention_mask=block_attn_mask,
            position_ids=concat_pos,
            **inputs,  # any remaining keys (e.g. token_type_ids)
        )
        logits: torch.Tensor = outputs.logits  # [B, 2L, V]

        # --- 8. Slice to first-half logits (noised positions only) ---
        logits = logits[:, :L, :].contiguous()  # [B, L, V]

        # --- 9. Dream Shift Operation (inherited flag) ---
        if self._right_shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1).contiguous()

        # --- 10. Build per-token loss weights (inherited machinery) ---
        loss_weights = self._build_loss_weights(
            timesteps, logits.device, diffusion_mask, attention_mask=padding_mask
        )

        # --- 11. Compute masked diffusion loss ---
        loss = fast_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_weights=loss_weights,
            loss_norm_type=self._loss_norm_type,
        )

        return (loss, outputs) if return_outputs else loss
