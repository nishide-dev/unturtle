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

"""Block Diffusion data collator with block-size aligned padding.

``BlockDiffusionDataCollator`` extends :class:`~.collator.MaskedDiffusionDataCollator`
by first padding each sequence to the nearest multiple of ``block_size`` using
EOS tokens, then delegating to the parent for standard collation and forward
noising.

This ensures that every sequence length is divisible by ``block_size``, which
is required by the BD3LM block-diagonal attention mask.

Reference: BD3LM — Block Diffusion Discrete Denoising Language Models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .collator import MaskedDiffusionDataCollator
from .schedulers import BaseAlphaScheduler, LinearAlphaScheduler


@dataclass
class BlockDiffusionDataCollator(MaskedDiffusionDataCollator):
    """Collate and noise a batch with sequences padded to a ``block_size`` multiple.

    Before delegating to :class:`~.collator.MaskedDiffusionDataCollator`, each
    sequence is right-padded with EOS tokens so that its length is a multiple of
    ``block_size``.  This guarantees the batch is compatible with the block-diagonal
    attention mask used in BD3LM training.

    Args:
        block_size:  Block size for BD3LM.  Each sequence length will be rounded
                     up to the nearest multiple of this value.  Defaults to 32.

    All other arguments are forwarded to
    :class:`~.collator.MaskedDiffusionDataCollator`.
    """

    # Override scheduler default so the field order stays compatible with the
    # parent dataclass (fields without defaults must come before those with defaults).
    scheduler: BaseAlphaScheduler = field(default_factory=LinearAlphaScheduler)
    block_size: int = 32

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad features to block_size multiple, then apply forward noising.

        Each feature dict is mutated in place before being passed to the parent
        collator.  The following keys are extended with EOS tokens:

        - ``input_ids`` — always padded with ``eos_token_id``
        - ``labels``    — if present, padded with ``eos_token_id`` (so EOS positions
                          are maskable by the diffusion process)
        - ``attention_mask`` — if present, padded with 1s (EOS tokens are real
                               tokens, not padding)

        Args:
            features: List of dataset items.  Each must have at least ``input_ids``.

        Returns:
            A batch dict as produced by
            :class:`~.collator.MaskedDiffusionDataCollator.__call__`, with sequence
            length guaranteed to be a multiple of ``block_size``.
        """
        eos_id: int = self.tokenizer.eos_token_id  # type: ignore[assignment]
        if eos_id is None:
            raise ValueError(
                "Tokenizer has no eos_token_id.  BlockDiffusionDataCollator requires "
                "an EOS token for block-size alignment padding."
            )

        padded: list[dict[str, Any]] = []
        for feature in features:
            feat = dict(feature)  # shallow copy — do not mutate the original
            ids = list(feat["input_ids"])
            current_len = len(ids)
            target_len = math.ceil(current_len / self.block_size) * self.block_size
            pad_len = target_len - current_len

            if pad_len > 0:
                feat["input_ids"] = ids + [eos_id] * pad_len

                if "labels" in feat:
                    feat["labels"] = list(feat["labels"]) + [eos_id] * pad_len

                if "attention_mask" in feat:
                    feat["attention_mask"] = (
                        list(feat["attention_mask"]) + [1] * pad_len
                    )

            padded.append(feat)

        return super().__call__(padded)
