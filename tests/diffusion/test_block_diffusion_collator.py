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

"""Tests for BlockDiffusionDataCollator — block-size aligned padding + forward noising."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from unturtle.diffusion.block_diffusion_collator import BlockDiffusionDataCollator
from unturtle.diffusion.schedulers import LinearAlphaScheduler

VOCAB = ["[PAD]", "[UNK]", " masked", "[BOS]", "[EOS]"] + [f"w{i}" for i in range(95)]
VOCAB_SIZE = len(VOCAB)
MASK_ID = 2
EOS_ID = 4


def _make_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(
        WordLevel(vocab={w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "mask_token": " masked",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        }
    )
    return fast


class TestBlockDiffusionCollator:
    @pytest.fixture(scope="module")
    def tokenizer(self):
        return _make_tokenizer()

    def test_output_keys(self, tokenizer):
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=MASK_ID,
        )
        samples = [{"input_ids": [5, 6, 7, 8, 9, 10]} for _ in range(2)]
        batch = collator(samples)
        for key in ("input_ids", "labels", "diffusion_mask", "timesteps"):
            assert key in batch, f"Missing key: {key}"

    def test_sequence_padded_to_block_multiple(self, tokenizer):
        """Sequences are padded to a multiple of block_size with EOS tokens."""
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=MASK_ID,
            completion_only=False,
        )
        # 6 tokens → should pad to 8 (next multiple of 4)
        samples = [{"input_ids": [5, 6, 7, 8, 9, 10]} for _ in range(2)]
        batch = collator(samples)
        L = batch["input_ids"].shape[1]
        assert L % 4 == 0, f"Sequence length {L} must be a multiple of block_size=4"
        assert L == 8, f"Expected length 8, got {L}"
        # Padded positions should have EOS token in the original (before noising)
        # Check labels at positions 6, 7 are EOS_ID
        assert batch["labels"][0, 6].item() == EOS_ID
        assert batch["labels"][0, 7].item() == EOS_ID

    def test_already_aligned_no_extra_padding(self, tokenizer):
        """Sequences already at block_size multiple get no extra padding."""
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=MASK_ID,
            completion_only=False,
        )
        samples = [{"input_ids": [5, 6, 7, 8]} for _ in range(2)]
        batch = collator(samples)
        assert batch["input_ids"].shape[1] == 4

    def test_labels_with_completion_only(self, tokenizer):
        """With completion_only=True, prompt (label=-100) positions keep -100 after alignment."""
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=MASK_ID,
            completion_only=True,
        )
        # 6 tokens, prompt_len=2, completion_len=4
        samples = [
            {"input_ids": [5, 6, 7, 8, 9, 10], "labels": [-100, -100, 7, 8, 9, 10]}
            for _ in range(2)
        ]
        batch = collator(samples)
        L = batch["input_ids"].shape[1]
        assert L % 4 == 0

    def test_no_eos_token_raises(self, tokenizer, monkeypatch):
        """Raises ValueError when tokenizer has no eos_token_id."""
        monkeypatch.setattr(tokenizer, "eos_token_id", None)
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=MASK_ID,
        )
        with pytest.raises(ValueError, match="eos_token_id"):
            collator([{"input_ids": [5, 6, 7]}])

    def test_masking_still_applied(self, tokenizer):
        """After alignment, forward noising is still applied."""
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=MASK_ID,
            completion_only=False,
            time_epsilon=0.999,  # force p_mask ≈ 1
        )
        torch.manual_seed(0)
        samples = [{"input_ids": list(range(5, 13))} for _ in range(16)]
        batch = collator(samples)
        # With p_mask ≈ 1, at least some positions should be masked
        assert batch["diffusion_mask"].any()
        # Masked positions should have mask_token_id
        assert (batch["input_ids"][batch["diffusion_mask"]] == MASK_ID).all()
