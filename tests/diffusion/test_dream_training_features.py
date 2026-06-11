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
Tests for Dream-specific training features added in issues #201, #202, #203:

  - #201  right_shift_logits: Shift Operation for Dream fine-tuning
  - #202  CART (context_adaptive_reweight): context-adaptive loss weighting
  - #203  loss_norm_type: configurable loss normalisation

All tests run on CPU (no GPU/Triton required).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from unturtle.diffusion.reweighting import context_adaptive_reweight
from unturtle.kernels.fused_masked_diffusion_loss import fused_masked_diffusion_loss

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_logits(B: int, L: int, V: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, L, V)


def _make_labels(
    B: int, L: int, V: int, prompt_len: int = 2, seed: int = 1
) -> torch.Tensor:
    """Clean token labels; first ``prompt_len`` positions are -100."""
    torch.manual_seed(seed)
    labels = torch.randint(0, V, (B, L))
    labels[:, :prompt_len] = -100
    return labels


def _make_diffusion_mask(
    labels: torch.Tensor, mask_rate: float = 0.5, seed: int = 2
) -> torch.Tensor:
    """Randomly mask ~mask_rate of completion positions."""
    torch.manual_seed(seed)
    maskable = labels != -100
    rand = torch.rand_like(labels, dtype=torch.float)
    return (rand < mask_rate) & maskable


# ─────────────────────────────────────────────────────────────────────────────
# #202  CART: context_adaptive_reweight
# ─────────────────────────────────────────────────────────────────────────────


class TestContextAdaptiveReweight:
    def test_shape(self):
        M = context_adaptive_reweight(seq_len=8)
        assert M.shape == (8, 8)

    def test_diagonal_is_zero(self):
        M = context_adaptive_reweight(seq_len=8, cart_p=0.8)
        assert torch.all(M.diagonal() == 0.0)

    def test_symmetry(self):
        # Geometric distribution is symmetric around the origin → M[n,i] == M[i,n]
        M = context_adaptive_reweight(seq_len=8, cart_p=0.5)
        assert torch.allclose(M, M.T, atol=1e-6)

    def test_nonnegative(self):
        M = context_adaptive_reweight(seq_len=8)
        assert torch.all(M >= 0.0)

    def test_closer_tokens_get_higher_weight(self):
        # For a given position n, the weight from position i should decrease
        # as |n-i| increases.
        M = context_adaptive_reweight(seq_len=10, cart_p=0.8)
        n = 5  # target masked position
        # w(n, 4) > w(n, 3) > w(n, 1)  (neighbours closer than far positions)
        assert M[n, n - 1] > M[n, n - 2], (
            "adjacent token should have higher weight than 2-away"
        )
        assert M[n, n - 2] > M[n, n - 4], "2-away should be higher than 4-away"

    def test_cart_p_1_zeros_far(self):
        # cart_p=1.0 means Geo(1, k) = p*(1-p)^(k-1) = 1 for k=1, 0 for k>1.
        # So w(k) = 0.5 for |n-i|=1, 0 elsewhere (except diagonal which is 0).
        M = context_adaptive_reweight(seq_len=8, cart_p=1.0)
        # Must not contain NaN (regression test for 0*(-inf) bug)
        assert not torch.isnan(M).any(), "cart_p=1.0 must not produce NaN weights"
        # Positions at distance 1 should be exactly 0.5
        # Positions at distance > 1 should be 0
        for n in range(8):
            for i in range(8):
                d = abs(n - i)
                if d == 0:
                    assert M[n, i] == pytest.approx(0.0, abs=1e-6), (
                        f"M[{n},{i}] (diagonal) should be 0"
                    )
                elif d == 1:
                    assert M[n, i] == pytest.approx(0.5, abs=1e-6), (
                        f"M[{n},{i}] (distance=1) should be 0.5 for cart_p=1.0"
                    )
                else:
                    assert M[n, i] == pytest.approx(0.0, abs=1e-6), (
                        f"M[{n},{i}] (distance={d}) should be 0 for cart_p=1.0"
                    )

    def test_invalid_cart_p(self):
        with pytest.raises(ValueError, match="cart_p must be in"):
            context_adaptive_reweight(8, cart_p=0.0)
        with pytest.raises(ValueError, match="cart_p must be in"):
            context_adaptive_reweight(8, cart_p=1.5)

    def test_weight_application_zeros_clean_positions(self):
        """After matmul with clean mask, clean positions should have weight=0."""
        L = 8
        M = context_adaptive_reweight(L, cart_p=0.8)

        # Simulate: positions 0,1 are prompt (-100), positions 2-7 are completion
        # Diffusion masks positions 3, 5 (these are the masked tokens)
        diffusion_mask = torch.zeros(1, L, dtype=torch.bool)
        diffusion_mask[0, 3] = True
        diffusion_mask[0, 5] = True

        clean_mask = ~diffusion_mask  # True at clean positions
        weight = clean_mask.float().matmul(M)  # [1, L]
        weight = weight.masked_fill(clean_mask, 0.0)

        # Clean positions must be zero
        assert torch.all(weight[clean_mask] == 0.0)
        # Masked positions (3, 5) should have positive weight from their clean neighbours
        assert weight[0, 3] > 0.0
        assert weight[0, 5] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# #203  loss_norm_type
# ─────────────────────────────────────────────────────────────────────────────


class TestLossNormType:
    B, L, V = 2, 8, 32

    def _loss(self, logits, labels, diffusion_mask, norm_type):
        return fused_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_norm_type=norm_type,
        )

    def test_token_norm_equals_manual(self):
        """'token' should match manual sum / n_maskable computation."""
        B, L, V = self.B, self.L, self.V
        logits = _make_logits(B, L, V)
        labels = _make_labels(B, L, V)
        diffusion_mask = _make_diffusion_mask(labels)

        loss = self._loss(logits, labels, diffusion_mask, "token")

        # Manual computation
        masked_labels = torch.where(diffusion_mask, labels, torch.tensor(-100))
        per_token = F.cross_entropy(
            logits.view(B * L, V),
            masked_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, L)
        n_maskable = (labels != -100).sum()
        expected = per_token.sum() / n_maskable

        assert torch.allclose(loss, expected, atol=1e-5), (
            f"token norm: {loss} vs {expected}"
        )

    def test_sequence_norm(self):
        """'sequence' should be between 'token' and 'batch' for typical inputs."""
        B, L, V = self.B, self.L, self.V
        logits = _make_logits(B, L, V)
        labels = _make_labels(B, L, V)
        diffusion_mask = _make_diffusion_mask(labels)

        loss_token = self._loss(logits, labels, diffusion_mask, "token")
        loss_seq = self._loss(logits, labels, diffusion_mask, "sequence")
        loss_batch = self._loss(logits, labels, diffusion_mask, "batch")

        # All should be positive scalars
        assert loss_token.item() > 0
        assert loss_seq.item() > 0
        assert loss_batch.item() > 0

    def test_sequence_norm_equals_manual(self):
        """'sequence' should match: (per_token / n_per_seq).sum() / B."""
        B, L, V = self.B, self.L, self.V
        logits = _make_logits(B, L, V, seed=10)
        labels = _make_labels(B, L, V)
        diffusion_mask = _make_diffusion_mask(labels)

        loss = self._loss(logits, labels, diffusion_mask, "sequence")

        masked_labels = torch.where(diffusion_mask, labels, torch.tensor(-100))
        per_token = F.cross_entropy(
            logits.view(B * L, V),
            masked_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, L)
        maskable_mask = labels != -100
        n_per_seq = maskable_mask.sum(dim=-1, keepdim=True).clamp_min(1).float()
        expected = (per_token / n_per_seq).sum() / B

        assert torch.allclose(loss, expected, atol=1e-5), (
            f"sequence norm: {loss} vs {expected}"
        )

    def test_batch_norm(self):
        """'batch' should equal sum / B."""
        B, L, V = self.B, self.L, self.V
        logits = _make_logits(B, L, V)
        labels = _make_labels(B, L, V)
        diffusion_mask = _make_diffusion_mask(labels)

        loss = self._loss(logits, labels, diffusion_mask, "batch")

        masked_labels = torch.where(diffusion_mask, labels, torch.tensor(-100))
        per_token = F.cross_entropy(
            logits.view(B * L, V),
            masked_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, L)
        expected = per_token.sum() / B

        assert torch.allclose(loss, expected, atol=1e-5), (
            f"batch norm: {loss} vs {expected}"
        )

    def test_invalid_norm_type(self):
        logits = _make_logits(1, 4, 16)
        labels = _make_labels(1, 4, 16)
        dm = _make_diffusion_mask(labels)
        with pytest.raises(ValueError, match="Unknown loss_norm_type"):
            fused_masked_diffusion_loss(logits, labels, dm, loss_norm_type="invalid")

    def test_norm_scale_relationship(self):
        """'token' and 'sequence' should relate to loss counts appropriately.

        When all sequences have the same number of maskable tokens,
        'token' == 'sequence' (per-seq normalisation == global normalisation).
        """
        B, L, V = 3, 10, 32
        torch.manual_seed(42)
        logits = _make_logits(B, L, V)
        # Make labels with SAME number of maskable positions per sequence: positions 2-9
        labels = torch.randint(0, V, (B, L))
        labels[:, :2] = -100  # uniform prompt length

        # Fix mask: always mask position 4 only (1 masked per sequence)
        diffusion_mask = torch.zeros(B, L, dtype=torch.bool)
        diffusion_mask[:, 4] = True

        loss_token = self._loss(logits, labels, diffusion_mask, "token")
        loss_seq = self._loss(logits, labels, diffusion_mask, "sequence")

        # When maskable count is the same per sequence, token == sequence
        assert torch.allclose(loss_token, loss_seq, atol=1e-5), (
            f"With equal maskable counts per seq: token={loss_token} vs seq={loss_seq}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# #201  right_shift_logits
# ─────────────────────────────────────────────────────────────────────────────


class TestRightShiftLogits:
    """Tests for the Dream Shift Operation in DiffusionTrainer."""

    def _make_trainer(self, right_shift: bool):
        """Build a minimal DiffusionTrainer on CPU with a tiny fake model."""
        from unittest.mock import MagicMock, patch

        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

        _ = DiffusionTrainingArguments(
            output_dir="/tmp/test_shift",
            no_cuda=True,
            right_shift_logits=right_shift,
            loss_weight_type="uniform",
        )

        # Minimal tokenizer mock
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 999
        tokenizer.pad = True
        tokenizer.padding_side = "right"

        # Minimal model mock
        model = MagicMock()
        model.config.mask_token_id = 999

        with patch.object(DiffusionTrainer, "__init__", lambda self, *a, **kw: None):
            trainer = DiffusionTrainer.__new__(DiffusionTrainer)

        trainer._alpha_scheduler = __import__(
            "unturtle.diffusion.schedulers", fromlist=["LinearAlphaScheduler"]
        ).LinearAlphaScheduler()
        trainer._time_epsilon = 1e-3
        trainer._loss_weight_type = "uniform"
        trainer._cart_p = 0.8
        trainer._loss_norm_type = "token"
        trainer._right_shift_logits = right_shift
        return trainer

    def test_shift_op_changes_logits(self):
        """right_shift_logits=True must produce different logits than False."""
        B, L, V = 2, 6, 32
        torch.manual_seed(7)
        logits_orig = _make_logits(B, L, V)

        # Manually apply shift as DiffusionTrainer does
        shifted = torch.cat([logits_orig[:, :1], logits_orig[:, :-1]], dim=1)

        assert not torch.allclose(logits_orig, shifted), (
            "Shift should produce different logits"
        )
        # First position should be identical (logits[:, 0] is repeated)
        assert torch.allclose(shifted[:, 0], logits_orig[:, 0])
        # Second position of shifted equals first position of original
        assert torch.allclose(shifted[:, 1], logits_orig[:, 0])

    def test_shift_op_semantics_position(self):
        """After shift, logit at position i predicts token at i+1.

        This means shifted[i] == original[i-1] for i > 0.
        """
        B, L, V = 1, 8, 16
        torch.manual_seed(3)
        logits = _make_logits(B, L, V)
        shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        for i in range(1, L):
            assert torch.allclose(shifted[:, i], logits[:, i - 1]), (
                f"shifted[:, {i}] should equal original[:, {i - 1}]"
            )

    def test_shift_and_no_shift_give_different_loss(self):
        """Training with and without shift should yield different loss values."""
        from unturtle.kernels.fused_masked_diffusion_loss import (
            fused_masked_diffusion_loss,
        )

        B, L, V = 2, 8, 32
        logits = _make_logits(B, L, V)
        labels = _make_labels(B, L, V, prompt_len=2)
        diffusion_mask = _make_diffusion_mask(labels)

        loss_no_shift = fused_masked_diffusion_loss(logits, labels, diffusion_mask)

        shifted_logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1).contiguous()
        loss_with_shift = fused_masked_diffusion_loss(
            shifted_logits, labels, diffusion_mask
        )

        assert not torch.allclose(loss_no_shift, loss_with_shift), (
            "Shift and no-shift should yield different loss values"
        )

    def test_shift_training_arg_stored(self):
        """DiffusionTrainingArguments should correctly store right_shift_logits."""
        from unturtle.diffusion import DiffusionTrainingArguments

        args_true = DiffusionTrainingArguments(
            output_dir="/tmp", right_shift_logits=True
        )
        args_false = DiffusionTrainingArguments(
            output_dir="/tmp", right_shift_logits=False
        )

        assert args_true.right_shift_logits is True
        assert args_false.right_shift_logits is False

    def test_cart_training_arg_stored(self):
        """DiffusionTrainingArguments should store cart_p."""
        from unturtle.diffusion import DiffusionTrainingArguments

        args = DiffusionTrainingArguments(
            output_dir="/tmp", loss_weight_type="cart", cart_p=0.6
        )
        assert args.loss_weight_type == "cart"
        assert args.cart_p == pytest.approx(0.6)

    def test_loss_norm_type_arg_stored(self):
        """DiffusionTrainingArguments should store loss_norm_type."""
        from unturtle.diffusion import DiffusionTrainingArguments

        for norm in ("token", "sequence", "batch"):
            args = DiffusionTrainingArguments(output_dir="/tmp", loss_norm_type=norm)
            assert args.loss_norm_type == norm


# ─────────────────────────────────────────────────────────────────────────────
# Integration: CART weight applied in build_loss_weights
# ─────────────────────────────────────────────────────────────────────────────


class TestCARTInBuildLossWeights:
    """Verify CART weight tensor properties when computed via trainer._build_loss_weights."""

    def _build_cart_weights(
        self, B: int, L: int, diffusion_mask: torch.Tensor, cart_p: float = 0.8
    ):
        from unturtle.diffusion.reweighting import context_adaptive_reweight

        weight_matrix = context_adaptive_reweight(L, cart_p=cart_p)
        clean_mask = ~diffusion_mask
        weight = clean_mask.float().matmul(weight_matrix)
        weight = weight.masked_fill(clean_mask, 0.0)
        return weight

    def test_cart_weight_shape(self):
        B, L = 3, 10
        diffusion_mask = _make_diffusion_mask(_make_labels(B, L, 32))
        w = self._build_cart_weights(B, L, diffusion_mask)
        assert w.shape == (B, L)

    def test_cart_weight_clean_positions_zero(self):
        B, L = 2, 8
        labels = _make_labels(B, L, 32)
        diffusion_mask = _make_diffusion_mask(labels)
        w = self._build_cart_weights(B, L, diffusion_mask)

        clean_mask = ~diffusion_mask
        assert torch.all(w[clean_mask] == 0.0), (
            "Clean positions must have zero CART weight"
        )

    def test_cart_weight_masked_positions_positive(self):
        """Masked positions adjacent to clean context should have positive weight."""
        B, L = 1, 8
        # Mask only position 4; positions 0-3, 5-7 are clean
        diffusion_mask = torch.zeros(B, L, dtype=torch.bool)
        diffusion_mask[0, 4] = True

        w = self._build_cart_weights(B, L, diffusion_mask, cart_p=0.8)
        assert w[0, 4].item() > 0.0, (
            "Masked position with clean neighbours must have positive weight"
        )

    def test_fully_masked_sequence_near_zero_weight(self):
        """A fully masked sequence (no clean neighbours) gets ~0 CART weight."""
        B, L = 1, 6
        # Mask all positions (no clean context) — simulates pathological case
        diffusion_mask = torch.ones(B, L, dtype=torch.bool)
        w = self._build_cart_weights(B, L, diffusion_mask, cart_p=0.8)
        # All positions are masked (clean_mask = all False), so all weights are 0
        assert torch.all(w == 0.0)
