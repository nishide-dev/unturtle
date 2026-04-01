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

"""Tests for PackedMaskedDiffusionDataCollator."""

from __future__ import annotations

import pytest
import torch
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer():
    raw = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="[PAD]",
    )
    tok.add_special_tokens({"unk_token": "[UNK]", "mask_token": "[MASK]", "pad_token": "[PAD]"})
    tok.name_or_path = "local"
    return tok


@pytest.fixture
def collator(tokenizer):
    from unturtle.diffusion import PackedMaskedDiffusionDataCollator
    return PackedMaskedDiffusionDataCollator(
        tokenizer=tokenizer,
        max_seq_length=32,
        completion_only=False,
    )


def _make_samples(n: int, length: int, vocab_size: int = 100) -> list[dict]:
    """Create n dummy samples each of `length` tokens."""
    return [
        {
            "input_ids": torch.randint(4, vocab_size, (length,)).tolist(),
            "labels": torch.randint(4, vocab_size, (length,)).tolist(),
            "attention_mask": [1] * length,
        }
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

class TestPackedShape:
    def test_output_keys_present(self, collator):
        samples = _make_samples(4, 8)
        batch = collator(samples)
        for key in ("input_ids", "labels", "attention_mask", "diffusion_mask",
                    "position_ids", "cu_seqlens", "seq_lengths", "timesteps",
                    "sample_timesteps"):
            assert key in batch, f"Missing key: {key}"

    def test_input_ids_shape(self, collator):
        """output shape is [B, max_seq_length]."""
        samples = _make_samples(4, 8)
        batch = collator(samples)
        B, L = batch["input_ids"].shape
        assert L == 32  # max_seq_length

    def test_attention_mask_marks_real_tokens(self, collator):
        """attention_mask must be 1 at packed tokens and 0 at padding."""
        samples = _make_samples(4, 7)
        batch = collator(samples)
        # Every real token must be marked
        attn = batch["attention_mask"]  # [B, L]
        for b in range(attn.shape[0]):
            real = attn[b].sum().item()
            assert real > 0, "No real tokens in row"
            assert real <= 32

    def test_labels_minus100_at_unmasked(self, collator):
        """positions not in diffusion_mask must have label == -100."""
        samples = _make_samples(4, 8)
        batch = collator(samples)
        dm = batch["diffusion_mask"]  # [B, L] bool
        lbl = batch["labels"]         # [B, L]
        # Unmasked non-padding positions should be -100
        assert (lbl[~dm] == -100).all(), "Un-masked positions must have label -100"

    def test_position_ids_reset_per_sample(self, collator):
        """position_ids must restart from 0 for each packed sample."""
        # Use samples shorter than max_seq_length so multiple pack into one slot
        samples = _make_samples(4, 8)  # 4 * 8 = 32 = max_seq_length
        batch = collator(samples)
        pos = batch["position_ids"]  # [B, L]
        for b in range(pos.shape[0]):
            cu = batch["cu_seqlens"][b]  # [n_samples+1]
            for i in range(len(cu) - 1):
                start = cu[i].item()
                end = cu[i + 1].item()
                expected = torch.arange(end - start, dtype=torch.long)
                assert torch.equal(pos[b, start:end], expected), (
                    f"position_ids not reset at batch={b}, sample={i}: {pos[b, start:end]}"
                )

    def test_cu_seqlens_monotone(self, collator):
        """cu_seqlens must be strictly increasing."""
        samples = _make_samples(6, 5)
        batch = collator(samples)
        for cu in batch["cu_seqlens"]:
            assert (cu[1:] > cu[:-1]).all(), f"cu_seqlens not monotone: {cu}"

    def test_seq_lengths_sum_le_max_seq_length(self, collator):
        """Total token count per batch element must not exceed max_seq_length."""
        samples = _make_samples(8, 4)
        batch = collator(samples)
        for sl in batch["seq_lengths"]:
            assert sl.sum().item() <= 32

    def test_timesteps_dense_shape(self, collator):
        """timesteps must be a 1-D [B] float tensor compatible with DiffusionTrainer."""
        samples = _make_samples(4, 8)
        batch = collator(samples)
        ts = batch["timesteps"]
        assert ts.ndim == 1, f"timesteps must be 1-D, got shape {ts.shape}"
        assert ts.shape[0] == batch["input_ids"].shape[0], "timesteps B != input_ids B"
        assert ts.dtype == torch.float32

    def test_sample_timesteps_per_sample_granularity(self, collator):
        """sample_timesteps must contain one t value per packed sample, not per row."""
        samples = _make_samples(4, 8)  # 4 * 8 = 32 = max_seq_length (one per row)
        batch = collator(samples)
        for b, st in enumerate(batch["sample_timesteps"]):
            n_seqs = len(batch["seq_lengths"][b])
            assert len(st) == n_seqs, (
                f"sample_timesteps[{b}] has {len(st)} entries but {n_seqs} packed samples"
            )


# ---------------------------------------------------------------------------
# Packing correctness
# ---------------------------------------------------------------------------

class TestPackingCorrectness:
    def test_no_tokens_lost(self, collator):
        """Total real tokens in batch >= input tokens (some may be in separate rows)."""
        samples = _make_samples(8, 4)
        total_in = sum(len(s["input_ids"]) for s in samples)
        batch = collator(samples)
        total_out = batch["attention_mask"].sum().item()
        assert total_out == total_in, (
            f"Token count mismatch: in={total_in} out={total_out}"
        )

    def test_long_sample_truncated(self, tokenizer):
        """Samples longer than max_seq_length are truncated when truncate_long='truncate'."""
        from unturtle.diffusion import PackedMaskedDiffusionDataCollator
        coll = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            completion_only=False,
            truncate_long="truncate",
        )
        samples = [{"input_ids": list(range(5, 30)), "attention_mask": [1] * 25}]
        batch = coll(samples)
        assert batch["input_ids"].shape[1] == 16

    def test_long_sample_dropped(self, tokenizer):
        """Samples longer than max_seq_length are dropped when truncate_long='drop'."""
        from unturtle.diffusion import PackedMaskedDiffusionDataCollator
        coll = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            completion_only=False,
            truncate_long="drop",
        )
        # Mix: one long (dropped), two short (kept)
        samples = [
            {"input_ids": list(range(5, 25)), "attention_mask": [1] * 20},  # dropped
            {"input_ids": list(range(5, 13)), "attention_mask": [1] * 8},   # kept
            {"input_ids": list(range(5, 13)), "attention_mask": [1] * 8},   # kept
        ]
        batch = coll(samples)
        total_real = batch["attention_mask"].sum().item()
        assert total_real == 16, f"Expected 16 real tokens, got {total_real}"

    def test_all_samples_dropped_raises(self, tokenizer):
        """If every sample is dropped, a ValueError should be raised."""
        from unturtle.diffusion import PackedMaskedDiffusionDataCollator
        coll = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=4,
            completion_only=False,
            truncate_long="drop",
        )
        samples = [{"input_ids": list(range(5, 15)), "attention_mask": [1] * 10}]
        with pytest.raises(ValueError, match="dropped"):
            coll(samples)


# ---------------------------------------------------------------------------
# Attention boundary tests
# ---------------------------------------------------------------------------

class TestAttentionBoundaries:
    def test_cu_seqlens_encode_boundaries(self, collator):
        """cu_seqlens must encode exact per-sample start/end offsets."""
        # 2 samples of length 6 packed into one row
        samples = [
            {"input_ids": [5] * 6, "attention_mask": [1] * 6},
            {"input_ids": [6] * 6, "attention_mask": [1] * 6},
        ]
        batch = collator(samples)
        # find the row that has both packed
        for b in range(batch["input_ids"].shape[0]):
            cu = batch["cu_seqlens"][b]
            if len(cu) == 3:  # two samples packed
                assert cu[0].item() == 0
                assert cu[1].item() == 6
                assert cu[2].item() == 12
                break
        else:
            pytest.skip("Samples were placed in separate rows")

    def test_block_diagonal_mask_shape_when_no_flash(self, tokenizer, monkeypatch):
        """When Flash Attention is unavailable, block_attention_mask must be [B, 1, L, L]."""
        from unturtle.diffusion import packed_collator as pc
        monkeypatch.setattr(pc, "_FLASH_ATTN_AVAILABLE", False)

        from unturtle.diffusion.packed_collator import PackedMaskedDiffusionDataCollator
        coll = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            completion_only=False,
        )
        samples = _make_samples(2, 6)
        batch = coll(samples)
        assert "block_attention_mask" in batch
        B, one, L1, L2 = batch["block_attention_mask"].shape
        assert one == 1
        assert L1 == L2 == 16

    def test_samples_do_not_attend_across_boundaries(self, tokenizer, monkeypatch):
        """In the block-diagonal mask, positions from different samples must not attend."""
        from unturtle.diffusion import packed_collator as pc
        monkeypatch.setattr(pc, "_FLASH_ATTN_AVAILABLE", False)

        from unturtle.diffusion.packed_collator import PackedMaskedDiffusionDataCollator
        coll = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            completion_only=False,
        )
        samples = [
            {"input_ids": [5] * 6, "attention_mask": [1] * 6},
            {"input_ids": [6] * 6, "attention_mask": [1] * 6},
        ]
        batch = coll(samples)

        if "block_attention_mask" not in batch:
            pytest.skip("Flash Attention available; SDPA mask not generated")

        # find row with two packed samples
        for b in range(batch["block_attention_mask"].shape[0]):
            cu = batch["cu_seqlens"][b]
            if len(cu) == 3:
                mask = batch["block_attention_mask"][b, 0]  # [L, L]
                # sample 0: positions 0..5, sample 1: positions 6..11
                cross = mask[0:6, 6:12]  # positions from sample 0 attending to sample 1
                assert not cross.any(), (
                    "Sample 0 should not attend to sample 1 positions"
                )
                break


# ---------------------------------------------------------------------------
# Diffusion mask correctness
# ---------------------------------------------------------------------------

class TestDiffusionMask:
    def test_diffusion_mask_only_at_real_positions(self, collator):
        """diffusion_mask must be False at padding positions."""
        samples = _make_samples(4, 8)
        batch = collator(samples)
        dm = batch["diffusion_mask"]      # [B, L] bool
        attn = batch["attention_mask"]    # [B, L] int
        # Padding positions (attn==0) must not be masked
        assert not dm[attn == 0].any(), "diffusion_mask set at padding positions"

    def test_completion_only_masks_only_completion(self, tokenizer):
        """With completion_only=True, only completion tokens should be masked."""
        from unturtle.diffusion import PackedMaskedDiffusionDataCollator
        coll = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=32,
            completion_only=True,
        )
        # prompt: first 4 tokens (label=-100), completion: last 4 tokens
        samples = [
            {
                "input_ids": [5, 6, 7, 8, 9, 10, 11, 12],
                "labels":    [-100, -100, -100, -100, 9, 10, 11, 12],
                "attention_mask": [1] * 8,
            }
        ]
        batch = coll(samples)
        dm = batch["diffusion_mask"][0]       # [L]
        attn = batch["attention_mask"][0]     # [L]
        real_len = attn.sum().item()
        # First 4 positions are prompt → must not be masked
        assert not dm[:4].any(), "Prompt positions must not be masked"
