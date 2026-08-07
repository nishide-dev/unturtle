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
Masked-discrete diffusion forward process.

``MaskedDiffusionProcess`` applies the MDLM/LLaDA absorbing-state corruption to
an already-collated clean batch: sample one timestep per sequence, mask each
eligible token with probability ``1 - alpha(t)``, and replace it with
``mask_token_id``.

This is a *masked-discrete* implementation.  It is one point in the process
layer, not a universal contract — discrete flow matching and continuous/latent
methods will produce different tensors under different keys.

Semantics are pinned to ``unturtle.diffusion.collator.MaskedDiffusionDataCollator``
so the two paths can be compared directly during the #62 migration.  This module
performs no collation: padding and label construction remain the collator's job.

Reference implementations:
  dllm-reasoning/d1   SFT/sft_trainer.py  ::  dLLMDataCollator
  zhziszz/dllm        dllm/core/trainers/mdlm.py  ::  MDLMTrainer.compute_loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .base import AlphaSchedule, ProcessOutput

# Fields the process owns and therefore rebuilds rather than passes through.
_SUPERVISION_KEYS = ("labels", "diffusion_mask", "timesteps")

# Packing topology the process consumes but the model forward does not: which
# original sample owns each position.  Kept out of `model_inputs` so a packed
# batch does not hand the model an argument it has no parameter for.
_TOPOLOGY_KEYS = ("segment_ids",)


@dataclass
class MaskedDiffusionProcess:
    """Absorbing-state (mask-token) forward process for masked diffusion LMs.

    Args:
        scheduler:       Alpha schedule defining the masking rate ``alpha(t)``.
                         Consumed structurally; construct and inject it from the
                         diffusion config rather than resolving it here.
        mask_token_id:   Id of the ``[MASK]`` token.  Resolving this from a
                         tokenizer or model config is orchestration's job.
        time_epsilon:    Lower bound on the sampled timestep, avoiding a
                         degenerate ``t -> 0``.
        completion_only: If ``True``, only positions whose ``labels`` are not
                         ``-100`` are eligible for masking (completion-only SFT,
                         as used by LLaDA / d1).  When ``True`` but the batch
                         carries no ``labels``, eligibility falls back to
                         non-padding positions, matching the legacy collator.
    """

    scheduler: AlphaSchedule
    mask_token_id: int
    time_epsilon: float = 1e-3
    completion_only: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.time_epsilon < 1.0:
            raise ValueError(
                f"time_epsilon must satisfy 0 <= time_epsilon < 1, got {self.time_epsilon}"
            )
        if self.mask_token_id < 0:
            raise ValueError(f"mask_token_id must be >= 0, got {self.mask_token_id}")

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> ProcessOutput:
        """Corrupt a clean batch into a masked-diffusion training state.

        Args:
            batch:     Must contain ``input_ids`` ``[B, L]``.  ``attention_mask``,
                       ``labels``, and any model metadata are optional and pass
                       through untouched.
            generator: Optional explicit RNG.  Two calls with equally-seeded
                       generators produce identical timesteps and masks.

        Returns:
            A :class:`~.base.ProcessOutput` whose ``model_inputs`` carry the
            noised ``input_ids`` plus every field the process does not own
            (see ``_SUPERVISION_KEYS``), and whose ``objective_inputs`` carry
            ``labels``, ``diffusion_mask``, and ``timesteps``.  A batch that
            already holds supervision keys has them rebuilt, not passed
            through.

        The input batch and its tensors are never mutated.  Note that
        pass-through values are the *same* tensor objects, not copies —
        only process-owned fields are freshly allocated — so a consumer that
        mutates e.g. ``model_inputs["attention_mask"]`` in place writes
        through to the caller's batch.
        """
        input_ids: torch.Tensor = batch["input_ids"]
        B, L = input_ids.shape
        device = input_ids.device

        attention_mask = batch.get("attention_mask")
        labels_in = batch.get("labels")
        use_labels = self.completion_only and labels_in is not None

        # --- determine maskable positions ---
        if use_labels:
            maskable = labels_in != -100
        elif attention_mask is not None:
            maskable = attention_mask.bool()
        else:
            maskable = torch.ones_like(input_ids, dtype=torch.bool)

        # --- sample diffusion timesteps, t in [eps, 1) ---
        # Packed rows hold several original samples, each of which owns its own
        # timestep.  With `segment_ids` we sample per segment and broadcast, so
        # `timesteps` is [B, L]; without it, one per row as before ([B]).
        # A packed row previously collapsed its samples to a mean t, which is
        # the wrong t for every sample in the row and is why `timestep` and
        # `scheduler` weighting had to reject packed batches outright.
        segment_ids = batch.get("segment_ids")
        if segment_ids is not None:
            if segment_ids.shape != (B, L):
                raise ValueError(
                    f"segment_ids must have shape {(B, L)} to match input_ids, "
                    f"got {tuple(segment_ids.shape)}"
                )
            n_segments = int(segment_ids.max()) + 1 if segment_ids.numel() else 0
            per_segment = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
                (B, max(n_segments, 1)), device=device, generator=generator
            )
            t = torch.gather(per_segment, 1, segment_ids.to(torch.long))  # [B, L]
        else:
            t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
                B, device=device, generator=generator
            )

        # --- per-token masking probability p_mask = 1 - alpha(t) ---
        # Normalized without assuming the schedule already returns the right
        # dtype/device.  Deliberately unclamped, matching the legacy collator.
        alpha_t = torch.as_tensor(self.scheduler.alpha(t), device=device, dtype=t.dtype)
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.expand(t.shape)
        elif alpha_t.shape != t.shape:
            # A [1] result would broadcast one row's rate over the whole batch.
            raise ValueError(
                f"scheduler.alpha(t) must return a scalar or a "
                f"{tuple(t.shape)} tensor, got shape {tuple(alpha_t.shape)}"
            )
        p_mask = 1.0 - alpha_t

        # --- Bernoulli corruption over eligible positions ---
        rand = torch.rand((B, L), device=device, generator=generator)
        threshold = p_mask if p_mask.dim() == 2 else p_mask[:, None]
        diffusion_mask = (rand < threshold) & maskable

        # --- apply noising ---
        noised_input_ids = input_ids.clone()
        noised_input_ids[diffusion_mask] = self.mask_token_id

        # --- build labels: clean targets at every maskable position ---
        # Unmasked-but-maskable positions keep their target; the masked-diffusion
        # loss selects contributing positions via ``diffusion_mask``.
        if use_labels:
            labels = labels_in.clone()
        else:
            labels = input_ids.clone()
            if attention_mask is not None:
                labels[~attention_mask.bool()] = -100

        model_inputs = {
            k: v
            for k, v in batch.items()
            if k not in _SUPERVISION_KEYS and k not in _TOPOLOGY_KEYS
        }
        model_inputs["input_ids"] = noised_input_ids

        return ProcessOutput(
            model_inputs=model_inputs,
            objective_inputs={
                "labels": labels,
                "diffusion_mask": diffusion_mask,
                "timesteps": t,
            },
        )
