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
Sequence-packed data collator for masked diffusion language model training.

``PackedMaskedDiffusionDataCollator`` concatenates multiple short sequences into
a single long sequence (up to ``max_seq_length``), eliminating padding tokens and
improving GPU utilisation.  Each packed sequence carries ``cu_seqlens`` metadata
so Flash Attention can enforce per-sample attention boundaries (block-diagonal,
non-causal).  When Flash Attention is unavailable it falls back to a dense
block-diagonal attention mask compatible with PyTorch SDPA.

The collator also applies the dLLM *forward noising process* identically to
:class:`~unturtle.diffusion.collator.MaskedDiffusionDataCollator`: it samples a
per-sample diffusion timestep ``t`` and randomly replaces maskable tokens with
``mask_token_id``.

Reference:
    unsloth/utils/packing.py  —  pack_dataset / pack_input_ids
    dllm-reasoning/d1  SFT/sft_trainer.py  ::  dLLMDataCollator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from .schedulers import BaseAlphaScheduler, LinearAlphaScheduler

logger = logging.getLogger(__name__)

try:
    import flash_attn  # noqa: F401
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False


def _build_block_diagonal_mask(
    seq_lengths: list[int],
    total_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a dense ``[1, 1, total_len, total_len]`` boolean attention mask where
    only positions within the same original sample may attend to each other.

    Used as a fallback when Flash Attention is not available.
    """
    mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=device)
    offset = 0
    for length in seq_lengths:
        end = offset + length
        mask[offset:end, offset:end] = True
        offset = end
    # Expand to [1, 1, total_len, total_len] for HF attention API
    return mask.unsqueeze(0).unsqueeze(0)


@dataclass
class PackedMaskedDiffusionDataCollator:
    """Collate, pack, and apply forward diffusion noising to a batch.

    Multiple short samples are concatenated into a single sequence of length
    ``≤ max_seq_length``.  Several packed samples are then batched together.

    Args:
        tokenizer:        HuggingFace tokenizer.  Must expose ``mask_token_id``
                          or the caller must pass ``mask_token_id`` explicitly.
        max_seq_length:   Maximum length of each packed sequence (tokens).
        scheduler:        Alpha scheduler.  Defaults to ``LinearAlphaScheduler``.
        mask_token_id:    Id of the ``[MASK]`` token.  Falls back to
                          ``tokenizer.mask_token_id`` when ``None``.
        pad_token_id:     Id used to pad packed sequences to ``max_seq_length``.
                          Falls back to ``tokenizer.pad_token_id``, then ``0``.
        time_epsilon:     Minimum timestep value (avoids degenerate ``t → 0``).
        completion_only:  If ``True``, only mask completion tokens (positions
                          where ``labels != -100``).  Set ``False`` to mask all
                          non-padding tokens.
        truncate_long:    How to handle samples longer than ``max_seq_length``:
                          ``"truncate"`` silently clips to ``max_seq_length``;
                          ``"drop"`` discards the sample.
    """

    tokenizer: PreTrainedTokenizerBase
    max_seq_length: int = 2048
    scheduler: BaseAlphaScheduler = field(default_factory=LinearAlphaScheduler)
    mask_token_id: int | None = None
    pad_token_id: int | None = None
    time_epsilon: float = 1e-3
    completion_only: bool = True
    truncate_long: str = "truncate"  # "truncate" | "drop"

    def __post_init__(self) -> None:
        if self.mask_token_id is None:
            if self.tokenizer.mask_token_id is None:
                raise ValueError(
                    "Tokenizer has no mask_token_id.  Pass mask_token_id "
                    "explicitly to PackedMaskedDiffusionDataCollator."
                )
            self.mask_token_id = self.tokenizer.mask_token_id

        if self.pad_token_id is None:
            self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None) or 0

        if self.truncate_long not in ("truncate", "drop"):
            raise ValueError(
                f"truncate_long must be 'truncate' or 'drop', got {self.truncate_long!r}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_sample(
        self, feature: dict[str, Any]
    ) -> tuple[list[int], list[int], list[bool]] | None:
        """Return ``(input_ids, labels, maskable)`` lists for one sample.

        Returns ``None`` if the sample should be dropped.
        """
        ids: list[int] = list(feature["input_ids"])
        length = len(ids)

        if length > self.max_seq_length:
            if self.truncate_long == "drop":
                return None
            ids = ids[: self.max_seq_length]
            length = self.max_seq_length

        # Determine maskable positions: completion tokens have labels != -100
        if self.completion_only and "labels" in feature:
            raw_labels = list(feature["labels"])[:length]
            maskable = [lbl != -100 for lbl in raw_labels]
        else:
            # attention_mask or all-True
            if "attention_mask" in feature:
                maskable = [bool(m) for m in list(feature["attention_mask"])[:length]]
            else:
                maskable = [True] * length

        labels = list(ids)  # clean token ids (will be modified in-place below)
        return ids, labels, maskable

    def _pack_samples(
        self, samples: list[tuple[list[int], list[int], list[bool]]]
    ) -> list[tuple[list[tuple[list[int], list[int], list[bool]]], int]]:
        """Greedy first-fit packing.

        Returns a list of *packed groups*.  Each group is
        ``(list_of_samples, total_length)``.  Groups have ``total_length ≤
        max_seq_length``.
        """
        groups: list[list[tuple[list[int], list[int], list[bool]]]] = []
        lengths: list[int] = []

        for sample in samples:
            slen = len(sample[0])
            placed = False
            for i, gl in enumerate(lengths):
                if gl + slen <= self.max_seq_length:
                    groups[i].append(sample)
                    lengths[i] += slen
                    placed = True
                    break
            if not placed:
                groups.append([sample])
                lengths.append(slen)

        return [(g, l) for g, l in zip(groups, lengths)]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a list of dataset items, pack them, apply forward noising.

        Returns a dict with keys:
          ``input_ids``       – [B, max_seq_length] packed + padded noised ids
          ``labels``          – [B, max_seq_length] clean ids at masked pos, -100 else
          ``attention_mask``  – [B, max_seq_length] 1 at real tokens, 0 at pad
          ``diffusion_mask``  – [B, max_seq_length] bool, True at masked positions
          ``timesteps``           – [B] mean ``t`` per row (DiffusionTrainer compatible)
          ``packed_seq_lengths``  – [total_samples] int32 flat sample lengths (Flash Attn varlen)
          ``cu_seqlens``          – list[Tensor] per-batch int32 cumulative seq lengths
          ``seq_lengths``         – list[Tensor] per-batch int32 individual sample lengths
          ``sample_timesteps``    – list[Tensor] per-sample ``t`` values per row
          ``position_ids``        – [B, max_seq_length] 0-based position within sample
        """
        # 1. Prepare each sample
        prepared: list[tuple[list[int], list[int], list[bool]]] = []
        for feat in features:
            result = self._prepare_sample(feat)
            if result is not None:
                prepared.append(result)

        if not prepared:
            raise ValueError("All samples were dropped during preparation.")

        # 2. Pack samples greedily
        packed_groups = self._pack_samples(prepared)
        B = len(packed_groups)
        L = self.max_seq_length

        # 3. Build batch tensors
        out_input_ids = torch.full((B, L), self.pad_token_id, dtype=torch.long)
        out_labels = torch.full((B, L), -100, dtype=torch.long)
        out_attn_mask = torch.zeros(B, L, dtype=torch.long)
        out_diffusion_mask = torch.zeros(B, L, dtype=torch.bool)
        out_position_ids = torch.zeros(B, L, dtype=torch.long)

        # per-batch-element metadata (variable number of samples per slot)
        all_cu_seqlens: list[torch.Tensor] = []
        all_seq_lengths: list[torch.Tensor] = []
        all_timesteps: list[torch.Tensor] = []

        for b, (group, _total) in enumerate(packed_groups):
            offset = 0
            seq_lens: list[int] = []
            ts: list[float] = []

            for ids, clean_labels, maskable in group:
                slen = len(ids)
                end = offset + slen

                # Sample per-sample diffusion timestep
                t_val = float(
                    self.time_epsilon
                    + (1.0 - self.time_epsilon) * torch.rand(1).item()
                )
                ts.append(t_val)

                # Compute masking probability and Bernoulli mask
                alpha_t = self.scheduler.alpha(
                    torch.tensor([t_val])
                ).item()
                p_mask = 1.0 - alpha_t

                rand = torch.rand(slen)
                maskable_t = torch.tensor(maskable, dtype=torch.bool)
                diff_mask = (rand < p_mask) & maskable_t  # [slen] bool

                # Apply noising
                noised = torch.tensor(ids, dtype=torch.long)
                noised[diff_mask] = self.mask_token_id

                # Build labels for this sample: clean at masked, -100 elsewhere
                lbl = torch.tensor(clean_labels, dtype=torch.long)
                lbl[~diff_mask] = -100

                # Position ids restart at 0 for each sample in the packed seq
                pos = torch.arange(slen, dtype=torch.long)

                out_input_ids[b, offset:end] = noised
                out_labels[b, offset:end] = lbl
                out_attn_mask[b, offset:end] = 1
                out_diffusion_mask[b, offset:end] = diff_mask
                out_position_ids[b, offset:end] = pos

                seq_lens.append(slen)
                offset = end

            # cu_seqlens: [0, len0, len0+len1, ...]
            cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
            for i, sl in enumerate(seq_lens):
                cu[i + 1] = cu[i] + sl
            all_cu_seqlens.append(cu)
            all_seq_lengths.append(torch.tensor(seq_lens, dtype=torch.int32))
            all_timesteps.append(torch.tensor(ts, dtype=torch.float32))

        # Build dense (B,) timesteps tensor for DiffusionTrainer compatibility.
        # Each packed row may contain multiple samples with different t values;
        # we use the mean t per row as a representative value for loss weighting.
        # The per-sample list is preserved as ``sample_timesteps`` for
        # custom trainers that need per-sample granularity.
        dense_timesteps = torch.stack(
            [t.mean() for t in all_timesteps]
        )  # [B], float32

        # Build flat packed_seq_lengths tensor for get_packed_info_from_kwargs().
        # Concatenates all per-batch seq_lengths into a single 1-D int32 tensor.
        # Example: B=2, row0=[6,6], row1=[12] → packed_seq_lengths=[6,6,12]
        packed_seq_lengths = torch.cat(all_seq_lengths)  # [total_samples_in_batch], int32

        batch: dict[str, Any] = {
            "input_ids": out_input_ids,
            "labels": out_labels,
            "attention_mask": out_attn_mask,
            "diffusion_mask": out_diffusion_mask,
            "position_ids": out_position_ids,
            "timesteps": dense_timesteps,          # [B] — DiffusionTrainer compatible
            "packed_seq_lengths": packed_seq_lengths,  # [total_samples] — for A2DAttention_fast_forward
            "cu_seqlens": all_cu_seqlens,          # list[Tensor], one per batch elem
            "seq_lengths": all_seq_lengths,        # list[Tensor], one per batch elem
            "sample_timesteps": all_timesteps,     # list[Tensor] — per-sample granularity
        }

        # 4. Build attention bias for the model forward pass
        if _FLASH_ATTN_AVAILABLE:
            # Flash Attention varlen path: just pass cu_seqlens.
            # The model's A2DAttention_fast_forward reads cu_seqlens from
            # get_packed_info_from_kwargs when attention_mask shape triggers it.
            # We keep attention_mask as the standard 2D mask; cu_seqlens is the
            # authoritative source for Flash Attention block boundaries.
            pass  # cu_seqlens already in batch
        else:
            # SDPA fallback: build a dense block-diagonal mask for each batch element.
            # Shape: [B, 1, L, L] — True where attention is allowed.
            device = out_input_ids.device
            sdpa_masks = []
            for b, (group, _) in enumerate(packed_groups):
                seq_lens_b = [len(s[0]) for s in group]
                total = sum(seq_lens_b)
                m = _build_block_diagonal_mask(seq_lens_b, total, device)
                # Pad to [1, 1, L, L]
                padded = torch.zeros(1, 1, L, L, dtype=torch.bool, device=device)
                padded[:, :, :total, :total] = m
                sdpa_masks.append(padded)
            batch["block_attention_mask"] = torch.cat(sdpa_masks, dim=0)  # [B, 1, L, L]
            logger.debug(
                "PackedMaskedDiffusionDataCollator: Flash Attention not available, "
                "using dense block-diagonal SDPA mask."
            )

        return batch
