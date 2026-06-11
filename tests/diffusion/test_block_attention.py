"""Tests for Block Diffusion attention mask construction.

The Block Diffusion attention mask is composed of three sub-masks applied to a
concatenated [x_t, x_0] sequence of length 2L:

  1. M_BD (Block Diagonal): self-attention within noised blocks and within clean blocks
  2. M_OBC (Offset Block Causal): cross-attention from noised positions to clean (earlier) blocks
  3. M_BC (Block Causal): causal attention within clean positions

Positions 0..L-1 are x_t (noised), positions L..2L-1 are x_0 (clean).
"""

import pytest
import torch

from unturtle.diffusion.block_attention import create_block_diffusion_attention_mask


class TestBlockAttentionMask:
    def test_output_shape(self):
        L, block_size = 8, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        assert mask.shape == (1, 1, 2 * L, 2 * L)

    def test_mask_is_boolean(self):
        L, block_size = 8, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        assert mask.dtype == torch.bool

    def test_block_diagonal_within_noised(self):
        L, block_size = 8, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        block0 = mask[0, 0, 0:4, 0:4]
        assert block0.all(), "Block 0 in x_t should fully self-attend"
        block1 = mask[0, 0, 4:8, 4:8]
        assert block1.all(), "Block 1 in x_t should fully self-attend"
        cross = mask[0, 0, 0:4, 4:8]
        assert not cross.any(), "Block 0 should not attend to block 1 within x_t"

    def test_block_diagonal_within_clean(self):
        L, block_size = 8, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        block0 = mask[0, 0, 8:12, 8:12]
        assert block0.all(), "Block 0 in x_0 should fully self-attend"
        block1 = mask[0, 0, 12:16, 12:16]
        assert block1.all(), "Block 1 in x_0 should fully self-attend"

    def test_block_causal_within_clean(self):
        L, block_size = 8, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        fwd = mask[0, 0, 12:16, 8:12]
        assert fwd.all(), "x_0 block 1 should attend to x_0 block 0 (block causal)"
        bwd = mask[0, 0, 8:12, 12:16]
        assert not bwd.any(), "x_0 block 0 should not attend to x_0 block 1"

    def test_offset_block_causal_cross(self):
        L, block_size = 8, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        cross = mask[0, 0, 4:8, 8:12]
        assert cross.all(), "x_t block 1 should attend to x_0 block 0 (OBC)"
        cross0 = mask[0, 0, 0:4, 8:16]
        assert not cross0.any(), (
            "x_t block 0 should not attend to x_0 (no earlier context)"
        )

    def test_noised_cannot_attend_to_later_noised_blocks(self):
        L, block_size = 8, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        cross = mask[0, 0, 0:4, 4:8]
        assert not cross.any(), "x_t block 0 should not attend to x_t block 1"

    def test_noised_block_k_attends_to_clean_blocks_0_to_k_minus_1(self):
        L, block_size = 12, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        attends_0 = mask[0, 0, 8:12, 12:16]  # x_0 block 0
        attends_1 = mask[0, 0, 8:12, 16:20]  # x_0 block 1
        assert attends_0.all(), "x_t block 2 should attend to x_0 block 0"
        assert attends_1.all(), "x_t block 2 should attend to x_0 block 1"
        attends_2 = mask[0, 0, 8:12, 20:24]  # x_0 block 2
        assert not attends_2.any(), "x_t block 2 should not attend to x_0 block 2"

    def test_matches_dllm_reference(self):
        L, block_size = 16, 4
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)

        q_idx = torch.arange(2 * L)[:, None]
        kv_idx = torch.arange(2 * L)[None, :]
        x0_flag_q = q_idx >= L
        x0_flag_kv = kv_idx >= L
        block_q = torch.where(x0_flag_q, (q_idx - L) // block_size, q_idx // block_size)
        block_kv = torch.where(
            x0_flag_kv, (kv_idx - L) // block_size, kv_idx // block_size
        )
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
        offset_block_causal = (block_q > block_kv) & x0_flag_kv & ~x0_flag_q
        block_causal = (block_q >= block_kv) & x0_flag_kv & x0_flag_q
        ref = block_diagonal | offset_block_causal | block_causal
        ref = ref.unsqueeze(0).unsqueeze(0)

        assert torch.equal(mask, ref), "Mask must match dllm reference exactly"

    def test_block_size_equals_seq_len(self):
        L, block_size = 8, 8
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        cross = mask[0, 0, 0:8, 8:16]
        assert not cross.any(), "Single-block: x_t should not attend to x_0"
        clean_self = mask[0, 0, 8:16, 8:16]
        assert clean_self.all(), "Single-block: x_0 should fully self-attend"

    def test_small_block_size(self):
        L, block_size = 4, 1
        mask = create_block_diffusion_attention_mask(seq_len=L, block_size=block_size)
        assert mask.shape == (1, 1, 8, 8)
        assert mask[0, 0, 0, 0].item()
        assert mask[0, 0, 1, 4].item()  # x_0 block 0 = position 4
        assert not mask[0, 0, 1, 5].item()  # x_0 block 1 = position 5

    def test_raises_on_non_divisible_seq_len(self):
        with pytest.raises(ValueError, match="divisible"):
            create_block_diffusion_attention_mask(seq_len=7, block_size=4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mask_on_cuda(self):
        L, block_size = 8, 4
        mask = create_block_diffusion_attention_mask(
            seq_len=L, block_size=block_size, device="cuda"
        )
        assert mask.device.type == "cuda"
        assert mask.shape == (1, 1, 16, 16)
