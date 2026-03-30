"""
Tests for masked diffusion language model components.

Validates numerical correctness of:
  - fast_masked_diffusion_loss   (Phase 1 Triton kernel wrapper)
  - masked_diffusion_loss_from_timesteps  (d1-style 1/t weighting)
  - LinearAlphaScheduler / CosineAlphaScheduler  (Phase 3 schedulers)
  - MaskedDiffusionDataCollator  (Phase 2 data collator)

Run with:
    pytest tests/test_masked_diffusion_loss.py -v
    pytest tests/test_masked_diffusion_loss.py -v -k "not cuda"  # CPU-only
"""

import math
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reference_masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation."""
    B, L, V = logits.shape
    masked_labels = labels.clone()
    masked_labels[~diffusion_mask] = -100

    per_token = F.cross_entropy(
        logits.view(B * L, V),
        masked_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, L)  # 0 at unmasked positions

    n_masked = diffusion_mask.sum().clamp_min(1)

    if loss_weights is None:
        return per_token.sum() / n_masked

    if loss_weights.dim() == 1:
        loss_weights = loss_weights.unsqueeze(1)

    return (per_token * loss_weights).sum() / n_masked


# ---------------------------------------------------------------------------
# fast_masked_diffusion_loss
# ---------------------------------------------------------------------------

class TestFastMaskedDiffusionLoss:

    @pytest.fixture(autouse=True)
    def _import(self):
        from unturtle import (
            fast_masked_diffusion_loss,
            masked_diffusion_loss_from_timesteps,
        )
        self.fast_masked_diffusion_loss = fast_masked_diffusion_loss
        self.masked_diffusion_loss_from_timesteps = masked_diffusion_loss_from_timesteps

    def _run(self, device: str, B=2, L=16, V=500):
        torch.manual_seed(42)
        logits = torch.randn(B, L, V, device=device)
        labels = torch.randint(0, V, (B, L), device=device)
        mask = torch.rand(B, L) > 0.4
        mask = mask.to(device)

        ref = _reference_masked_ce(logits.cpu(), labels.cpu(), mask.cpu()).to(device)
        got = self.fast_masked_diffusion_loss(logits, labels, mask)

        assert torch.allclose(ref, got, atol=1e-4), (
            f"[{device}] uniform: ref={ref.item():.6f} got={got.item():.6f}"
        )

    def test_uniform_cpu(self):
        self._run("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_uniform_cuda(self):
        self._run("cuda")

    def test_all_masked(self):
        """When every token is masked the loss should equal plain CE."""
        torch.manual_seed(0)
        B, L, V = 2, 8, 100
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)

        ref = F.cross_entropy(logits.view(B * L, V), labels.view(-1))
        got = self.fast_masked_diffusion_loss(logits, labels, mask)

        assert torch.allclose(ref, got, atol=1e-4), (
            f"all_masked: ref={ref.item():.6f} got={got.item():.6f}"
        )

    def test_no_masked_tokens_returns_zero(self):
        """When no token is masked the loss must be 0."""
        B, L, V = 2, 8, 50
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        mask = torch.zeros(B, L, dtype=torch.bool)

        got = self.fast_masked_diffusion_loss(logits, labels, mask)
        assert got.item() == pytest.approx(0.0, abs=1e-6), f"got={got.item()}"

    def test_timestep_weighting_cpu(self):
        """1/t weighting should match the reference."""
        torch.manual_seed(7)
        B, L, V = 3, 12, 200
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        mask = torch.rand(B, L) > 0.5
        t = torch.rand(B) * 0.9 + 0.1  # (0.1, 1.0)

        weights = (1.0 / t)  # [B]
        ref = _reference_masked_ce(logits, labels, mask, loss_weights=weights)
        got = self.masked_diffusion_loss_from_timesteps(logits, labels, mask, t)

        assert torch.allclose(ref, got, atol=1e-4), (
            f"timestep: ref={ref.item():.6f} got={got.item():.6f}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_timestep_weighting_cuda(self):
        torch.manual_seed(7)
        B, L, V = 3, 12, 200
        logits = torch.randn(B, L, V, device="cuda")
        labels = torch.randint(0, V, (B, L), device="cuda")
        mask = (torch.rand(B, L) > 0.5).to("cuda")
        t = (torch.rand(B) * 0.9 + 0.1).to("cuda")

        weights = 1.0 / t
        ref = _reference_masked_ce(
            logits.cpu(), labels.cpu(), mask.cpu(), loss_weights=weights.cpu()
        ).to("cuda")
        got = self.masked_diffusion_loss_from_timesteps(logits, labels, mask, t)

        assert torch.allclose(ref, got, atol=1e-4), (
            f"cuda timestep: ref={ref.item():.6f} got={got.item():.6f}"
        )

    def test_large_vocab_cpu(self):
        """Vocab > 65536 triggers the chunked forward path."""
        torch.manual_seed(99)
        B, L, V = 1, 4, 65537  # one more than MAX_FUSED_SIZE
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)

        ref = _reference_masked_ce(logits, labels, mask)
        got = self.fast_masked_diffusion_loss(logits, labels, mask)

        assert torch.allclose(ref, got, atol=1e-3), (
            f"large_vocab: ref={ref.item():.6f} got={got.item():.6f}"
        )

    def test_gradient_flows(self):
        """Backward pass must produce non-zero gradients on logits."""
        B, L, V = 2, 6, 50
        logits = torch.randn(B, L, V, requires_grad=True)
        labels = torch.randint(0, V, (B, L))
        mask = torch.rand(B, L) > 0.5

        loss = self.fast_masked_diffusion_loss(logits, labels, mask)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Alpha schedulers
# ---------------------------------------------------------------------------

class TestAlphaSchedulers:

    @pytest.fixture(autouse=True)
    def _import(self):
        from unturtle.diffusion import (
            LinearAlphaScheduler,
            CosineAlphaScheduler,
            make_alpha_scheduler,
        )
        self.LinearAlphaScheduler = LinearAlphaScheduler
        self.CosineAlphaScheduler = CosineAlphaScheduler
        self.make_alpha_scheduler = make_alpha_scheduler

    def test_linear_boundary_values(self):
        sched = self.LinearAlphaScheduler()
        assert sched.alpha(0.0) == pytest.approx(1.0)
        assert sched.alpha(1.0) == pytest.approx(0.0)
        assert sched.alpha(0.5) == pytest.approx(0.5)

    def test_cosine_boundary_values(self):
        sched = self.CosineAlphaScheduler()
        # alpha(0) = 1 - cos(pi/2 * 1) = 1 - 0 = 1
        assert sched.alpha(0.0) == pytest.approx(1.0, abs=1e-6)
        # alpha(1) = 1 - cos(0) = 1 - 1 = 0
        assert sched.alpha(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_masking_prob_is_complement_of_alpha(self):
        sched = self.LinearAlphaScheduler()
        t = torch.linspace(0.01, 0.99, 10)
        assert torch.allclose(sched.masking_prob(t), 1.0 - sched.alpha(t), atol=1e-6)

    def test_weight_positive(self):
        """MDLM weight w(t) = -α'(t) / (1-α(t)) should be > 0."""
        sched = self.LinearAlphaScheduler()
        t = torch.linspace(0.05, 0.95, 20)
        w = sched.weight(t)
        assert (w > 0).all(), f"Non-positive weights found: {w}"

    def test_make_alpha_scheduler_linear(self):
        sched = self.make_alpha_scheduler("linear")
        assert isinstance(sched, self.LinearAlphaScheduler)

    def test_make_alpha_scheduler_cosine(self):
        sched = self.make_alpha_scheduler("cosine")
        assert isinstance(sched, self.CosineAlphaScheduler)

    def test_make_alpha_scheduler_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown alpha scheduler"):
            self.make_alpha_scheduler("unknown_sched")

    def test_tensor_input(self):
        sched = self.LinearAlphaScheduler()
        t = torch.tensor([0.1, 0.5, 0.9])
        alpha = sched.alpha(t)
        expected = torch.tensor([0.9, 0.5, 0.1])
        assert torch.allclose(alpha, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# MaskedDiffusionDataCollator
# ---------------------------------------------------------------------------

class TestMaskedDiffusionDataCollator:

    @pytest.fixture(autouse=True)
    def _import(self):
        from unturtle.diffusion import MaskedDiffusionDataCollator
        from unturtle.diffusion import LinearAlphaScheduler
        self.MaskedDiffusionDataCollator = MaskedDiffusionDataCollator
        self.LinearAlphaScheduler = LinearAlphaScheduler

    def _make_tokenizer(self, mask_token_id: int = 999):
        """Minimal mock tokenizer."""
        class FakeTokenizer:
            mask_token_id = None
            padding_side = "right"

        tok = FakeTokenizer()
        tok.mask_token_id = mask_token_id
        return tok

    def _make_features(self, B: int = 4, L: int = 16, V: int = 100, seed: int = 0):
        torch.manual_seed(seed)
        features = []
        for _ in range(B):
            input_ids = torch.randint(0, V - 1, (L,))  # avoid mask_token_id=999
            labels = input_ids.clone()
            labels[: L // 2] = -100  # first half = prompt (not maskable)
            features.append({"input_ids": input_ids, "labels": labels})
        return features

    def test_output_keys(self):
        collator = self.MaskedDiffusionDataCollator(
            tokenizer=self._make_tokenizer(),
            scheduler=self.LinearAlphaScheduler(),
        )
        batch = collator(self._make_features())
        for key in ("input_ids", "labels", "diffusion_mask", "timesteps"):
            assert key in batch, f"Missing key: {key}"

    def test_timesteps_in_range(self):
        eps = 1e-3
        collator = self.MaskedDiffusionDataCollator(
            tokenizer=self._make_tokenizer(),
            scheduler=self.LinearAlphaScheduler(),
            time_epsilon=eps,
        )
        batch = collator(self._make_features(B=32))
        t = batch["timesteps"]
        assert (t >= eps).all(), f"t below epsilon: {t.min()}"
        assert (t <= 1.0).all(), f"t above 1: {t.max()}"

    def test_completion_only_masking(self):
        """Prompt positions (initial label == -100) must never be masked."""
        collator = self.MaskedDiffusionDataCollator(
            tokenizer=self._make_tokenizer(),
            scheduler=self.LinearAlphaScheduler(),
            completion_only=True,
        )
        features = self._make_features(B=8, L=20)
        batch = collator(features)

        # Reconstruct the original prompt mask from features
        original_labels = torch.stack([f["labels"] for f in features])
        prompt_positions = original_labels == -100  # [B, L]

        # No masked position should be in the prompt
        assert not (batch["diffusion_mask"] & prompt_positions).any(), (
            "Prompt tokens were masked despite completion_only=True"
        )

    def test_noised_input_ids_have_mask_token(self):
        """All positions in diffusion_mask should hold mask_token_id."""
        mask_id = 999
        collator = self.MaskedDiffusionDataCollator(
            tokenizer=self._make_tokenizer(mask_token_id=mask_id),
            scheduler=self.LinearAlphaScheduler(),
        )
        batch = collator(self._make_features())
        masked_vals = batch["input_ids"][batch["diffusion_mask"]]
        assert (masked_vals == mask_id).all(), (
            "Some masked positions do not contain mask_token_id"
        )

    def test_labels_minus100_at_unmasked(self):
        """Unmasked positions must have label == -100."""
        collator = self.MaskedDiffusionDataCollator(
            tokenizer=self._make_tokenizer(),
            scheduler=self.LinearAlphaScheduler(),
        )
        batch = collator(self._make_features())
        unmasked_labels = batch["labels"][~batch["diffusion_mask"]]
        assert (unmasked_labels == -100).all()

    def test_no_mask_token_raises(self):
        class NoMaskTok:
            mask_token_id = None
            padding_side = "right"

        with pytest.raises(ValueError, match="mask_token_id"):
            self.MaskedDiffusionDataCollator(tokenizer=NoMaskTok())
