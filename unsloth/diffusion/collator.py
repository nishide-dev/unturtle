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
Data collator for masked diffusion language model training.

``MaskedDiffusionDataCollator`` applies the *forward noising process* to each
batch: it randomly masks a fraction of completion tokens (based on a sampled
diffusion timestep ``t``) and replaces them with the ``[MASK]`` token id.  The
resulting batch contains:

  ``input_ids``      – noised token ids (masked positions → mask_token_id)
  ``labels``         – clean token ids (``x_0``); unmasked positions → -100
  ``diffusion_mask`` – bool tensor, True at masked positions
  ``timesteps``      – sampled ``t`` values, shape ``(B,)``

Reference implementations:
  dllm-reasoning/d1   SFT/sft_trainer.py  ::  dLLMDataCollator
  zhziszz/dllm        dllm/core/trainers/mdlm.py  ::  MDLMTrainer.compute_loss
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import default_data_collator

from .schedulers import BaseAlphaScheduler, LinearAlphaScheduler


@dataclass
class MaskedDiffusionDataCollator:
    """Collate and apply forward diffusion noising to a batch.

    Args:
        tokenizer:       HuggingFace tokenizer.  Must expose ``mask_token_id``
                         or the caller must pass ``mask_token_id`` explicitly.
        scheduler:       Alpha scheduler that defines the masking rate α(t).
                         Defaults to :class:`~.schedulers.LinearAlphaScheduler`.
        mask_token_id:   Id of the ``[MASK]`` token.  Falls back to
                         ``tokenizer.mask_token_id`` when ``None``.
        time_epsilon:    Minimum timestep value (avoids degenerate ``t → 0``).
        completion_only: If ``True``, only mask *completion* tokens (i.e. those
                         whose initial ``labels`` value is not ``-100``).  This
                         corresponds to the "completion-only SFT" mode used by
                         LLaDA / d1.  If ``False``, all tokens (including the
                         prompt) are eligible for masking.
    """

    tokenizer: PreTrainedTokenizerBase
    scheduler: BaseAlphaScheduler = field(
        default_factory=LinearAlphaScheduler
    )
    mask_token_id: int | None = None
    time_epsilon: float = 1e-3
    completion_only: bool = True

    def __post_init__(self) -> None:
        if self.mask_token_id is None:
            if self.tokenizer.mask_token_id is None:
                raise ValueError(
                    "Tokenizer has no mask_token_id.  Pass mask_token_id explicitly "
                    "to MaskedDiffusionDataCollator."
                )
            self.mask_token_id = self.tokenizer.mask_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of dataset items and apply forward noising.

        Each item in ``features`` must have at least ``input_ids``.
        If a ``labels`` key is present it is used to determine which positions
        are *maskable* (non ``-100``).  This is the standard completion-only
        convention used by TRL / SFT.

        Returns a dict with keys:
          ``input_ids``, ``attention_mask`` (if present), ``labels``,
          ``diffusion_mask``, ``timesteps``.
        """
        # --- standard HF collation (padding, stacking) ---
        batch = default_data_collator(features)

        input_ids: torch.Tensor = batch["input_ids"]  # [B, L]
        B, L = input_ids.shape

        # --- determine maskable positions ---
        if self.completion_only and "labels" in batch:
            # Positions where the original label is not -100 are completion tokens
            maskable: torch.Tensor = batch["labels"] != -100  # [B, L]
        else:
            # Mask any non-padding position
            if "attention_mask" in batch:
                maskable = batch["attention_mask"].bool()
            else:
                maskable = torch.ones(B, L, dtype=torch.bool)

        # --- sample diffusion timestep per sequence ---
        device = input_ids.device
        t: torch.Tensor = self.time_epsilon + (
            1.0 - self.time_epsilon
        ) * torch.rand(B, device=device)  # [B], in (eps, 1]

        # --- compute per-token masking probability p_mask = 1 - alpha(t) ---
        alpha_t: torch.Tensor = self.scheduler.alpha(t)  # [B]
        p_mask: torch.Tensor = (1.0 - alpha_t).unsqueeze(1).expand(B, L)  # [B, L]

        # --- stochastic masking (Bernoulli) ---
        rand = torch.rand(B, L, device=device)
        diffusion_mask: torch.Tensor = (rand < p_mask) & maskable  # [B, L] bool

        # --- apply noising ---
        noised_input_ids = input_ids.clone()
        noised_input_ids[diffusion_mask] = self.mask_token_id

        # --- build labels: clean tokens at masked positions, -100 elsewhere ---
        labels = input_ids.clone()
        labels[~diffusion_mask] = -100

        batch["input_ids"] = noised_input_ids
        batch["labels"] = labels
        batch["diffusion_mask"] = diffusion_mask
        batch["timesteps"] = t

        return batch
