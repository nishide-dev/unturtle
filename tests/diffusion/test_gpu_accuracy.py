"""GPU numerical accuracy tests: fast_masked_diffusion_loss vs F.cross_entropy.

Verifies that the Triton-based kernel produces numerically equivalent results
to PyTorch's reference F.cross_entropy on CUDA across a range of shapes,
masking ratios, vocab sizes, and weighting modes.

All tests are skipped on CPU-only machines.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from unturtle import (
    fast_masked_diffusion_loss,
    masked_diffusion_loss_from_timesteps,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPU accuracy tests"
)

# Tolerance: Triton FP32 vs PyTorch FP32 reference
ATOL = 1e-3
RTOL = 1e-3


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _reference_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure PyTorch reference: masked CE on CUDA."""
    B, L, V = logits.shape
    masked_labels = labels.clone()
    masked_labels[~diffusion_mask] = -100

    per_token = F.cross_entropy(
        logits.reshape(B * L, V).float(),
        masked_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, L)

    n_masked = diffusion_mask.sum().clamp_min(1)

    if loss_weights is None:
        return per_token.sum() / n_masked

    if loss_weights.shape == (B,):
        loss_weights = loss_weights.unsqueeze(1)
    return (per_token * loss_weights).sum() / n_masked


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_inputs(
    B: int = 4,
    L: int = 32,
    V: int = 256,
    mask_ratio: float = 0.5,
    seed: int = 42,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (logits, labels, diffusion_mask) on the given device."""
    torch.manual_seed(seed)
    logits = torch.randn(B, L, V, device=device, requires_grad=False)
    labels = torch.randint(0, V, (B, L), device=device)
    diffusion_mask = torch.rand(B, L, device=device) < mask_ratio
    # Ensure at least one masked token per batch
    diffusion_mask[:, 0] = True
    return logits, labels, diffusion_mask


# ---------------------------------------------------------------------------
# B-1: Basic equivalence
# ---------------------------------------------------------------------------


class TestGPUNumericalAccuracy:

    @pytest.mark.parametrize("mask_ratio", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    def test_uniform_loss_vs_reference(self, mask_ratio):
        """No loss_weights: Triton result matches PyTorch reference."""
        logits, labels, mask = _make_inputs(mask_ratio=mask_ratio)
        ref = _reference_loss(logits, labels, mask)
        got = fast_masked_diffusion_loss(logits, labels, mask)
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (
            f"mask_ratio={mask_ratio}: ref={ref.item():.6f} got={got.item():.6f}"
        )

    @pytest.mark.parametrize("B,L,V", [
        (1, 16, 128),
        (4, 32, 256),
        (8, 64, 512),
        (2, 128, 1024),
    ])
    def test_shapes(self, B, L, V):
        """Various batch/seq/vocab shapes produce matching results."""
        logits, labels, mask = _make_inputs(B=B, L=L, V=V)
        ref = _reference_loss(logits, labels, mask)
        got = fast_masked_diffusion_loss(logits, labels, mask)
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (
            f"[{B},{L},{V}]: ref={ref.item():.6f} got={got.item():.6f}"
        )

    def test_large_vocab(self):
        """Vocab > 65536 triggers chunked path; result must still match."""
        logits, labels, mask = _make_inputs(B=2, L=16, V=65537)
        ref = _reference_loss(logits, labels, mask)
        got = fast_masked_diffusion_loss(logits, labels, mask)
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (
            f"large vocab: ref={ref.item():.6f} got={got.item():.6f}"
        )

    def test_all_masked(self):
        """All positions masked: equivalent to plain CE over all tokens."""
        B, L, V = 4, 32, 256
        torch.manual_seed(0)
        logits = torch.randn(B, L, V, device="cuda")
        labels = torch.randint(0, V, (B, L), device="cuda")
        mask = torch.ones(B, L, dtype=torch.bool, device="cuda")

        ref = _reference_loss(logits, labels, mask)
        got = fast_masked_diffusion_loss(logits, labels, mask)
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL)

    def test_no_masked_tokens_returns_zero(self):
        """No masked positions → loss = 0."""
        logits, labels, _ = _make_inputs()
        mask = torch.zeros_like(labels, dtype=torch.bool)
        got = fast_masked_diffusion_loss(logits, labels, mask)
        assert got.item() == 0.0, f"Expected 0.0, got {got.item()}"


# ---------------------------------------------------------------------------
# B-2: Timestep weighting
# ---------------------------------------------------------------------------


class TestTimestepWeighting:

    def test_per_sequence_weights_vs_reference(self):
        """Per-sequence weights (B,): Triton matches reference."""
        logits, labels, mask = _make_inputs(B=4, L=32, V=256)
        torch.manual_seed(1)
        weights = torch.rand(4, device="cuda") + 0.1  # positive weights

        ref = _reference_loss(logits, labels, mask, loss_weights=weights)
        got = fast_masked_diffusion_loss(logits, labels, mask, loss_weights=weights)
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (
            f"per-seq weights: ref={ref.item():.6f} got={got.item():.6f}"
        )

    def test_timestep_weighting_convenience(self):
        """masked_diffusion_loss_from_timesteps applies 1/t weighting correctly."""
        logits, labels, mask = _make_inputs(B=4, L=32, V=256)
        torch.manual_seed(2)
        t = torch.rand(4, device="cuda").clamp_min(1e-3)

        ref = _reference_loss(logits, labels, mask, loss_weights=1.0 / t)
        got = masked_diffusion_loss_from_timesteps(logits, labels, mask, timesteps=t)
        assert torch.allclose(got, ref, atol=ATOL, rtol=RTOL), (
            f"1/t weighting: ref={ref.item():.6f} got={got.item():.6f}"
        )

    @pytest.mark.parametrize("t_val", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_uniform_timestep_weighting(self, t_val):
        """Uniform t → constant 1/t scaling of the plain CE loss."""
        B, L, V = 4, 32, 256
        logits, labels, mask = _make_inputs(B=B, L=L, V=V)
        t = torch.full((B,), t_val, device="cuda")

        plain_loss = _reference_loss(logits, labels, mask)
        weighted = masked_diffusion_loss_from_timesteps(logits, labels, mask, t)
        ref_weighted = _reference_loss(logits, labels, mask, loss_weights=1.0 / t)

        assert torch.allclose(weighted, ref_weighted, atol=ATOL, rtol=RTOL)
        # 1/t weighted loss should be approximately plain_loss / t_val
        assert torch.allclose(weighted, plain_loss / t_val, atol=ATOL * 5, rtol=RTOL * 5)


# ---------------------------------------------------------------------------
# B-3: Gradient accuracy
# ---------------------------------------------------------------------------


class TestGradientAccuracy:

    def test_gradients_match_reference(self):
        """Triton kernel gradients (w.r.t. logits) match PyTorch autograd."""
        B, L, V = 4, 32, 256
        torch.manual_seed(7)
        logits_ref = torch.randn(B, L, V, device="cuda", requires_grad=True)
        logits_tri = logits_ref.detach().clone().requires_grad_(True)
        labels = torch.randint(0, V, (B, L), device="cuda")
        mask = torch.rand(B, L, device="cuda") < 0.5
        mask[:, 0] = True

        loss_ref = _reference_loss(logits_ref, labels, mask)
        loss_tri = fast_masked_diffusion_loss(logits_tri, labels, mask)

        loss_ref.backward()
        loss_tri.backward()

        assert torch.allclose(logits_ref.grad, logits_tri.grad, atol=1e-4, rtol=1e-4), (
            f"Max grad diff: {(logits_ref.grad - logits_tri.grad).abs().max().item():.2e}"
        )

    def test_gradient_norm_finite(self):
        """Gradients should be finite and non-zero after backward."""
        logits, labels, mask = _make_inputs()
        logits = logits.requires_grad_(True)
        loss = fast_masked_diffusion_loss(logits, labels, mask)
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all(), "Gradients contain inf/nan"
        assert logits.grad.abs().sum() > 0, "All-zero gradients"


# ---------------------------------------------------------------------------
# B-4: Numerical stability
# ---------------------------------------------------------------------------


class TestNumericalStability:

    def test_large_logits(self):
        """Very large logit magnitudes should not produce NaN."""
        B, L, V = 4, 32, 256
        torch.manual_seed(3)
        logits = torch.randn(B, L, V, device="cuda") * 100.0
        labels = torch.randint(0, V, (B, L), device="cuda")
        mask = torch.ones(B, L, dtype=torch.bool, device="cuda")

        loss = fast_masked_diffusion_loss(logits, labels, mask)
        assert torch.isfinite(loss), f"Non-finite loss with large logits: {loss.item()}"

    def test_small_logits(self):
        """Very small logit magnitudes should not produce NaN."""
        B, L, V = 4, 32, 256
        torch.manual_seed(4)
        logits = torch.randn(B, L, V, device="cuda") * 1e-4
        labels = torch.randint(0, V, (B, L), device="cuda")
        mask = torch.ones(B, L, dtype=torch.bool, device="cuda")

        loss = fast_masked_diffusion_loss(logits, labels, mask)
        assert torch.isfinite(loss), f"Non-finite loss with small logits: {loss.item()}"

    def test_half_precision(self):
        """FP16 logits: loss should still be finite (kernel upcasts internally)."""
        B, L, V = 4, 32, 256
        torch.manual_seed(5)
        logits = torch.randn(B, L, V, device="cuda", dtype=torch.float16)
        labels = torch.randint(0, V, (B, L), device="cuda")
        mask = torch.ones(B, L, dtype=torch.bool, device="cuda")

        loss = fast_masked_diffusion_loss(logits, labels, mask)
        assert torch.isfinite(loss), f"Non-finite loss with FP16: {loss.item()}"
