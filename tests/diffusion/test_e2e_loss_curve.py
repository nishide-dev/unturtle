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

"""Single-step loss equivalence tests: unturtle implementations vs d1/MDLM reference.

Verifies that ``fast_masked_diffusion_loss`` and ``DiffusionTrainer.compute_loss``
return numerically identical results to the d1/MDLM reference implementation
for the same forward-pass inputs (logits, labels, diffusion_mask, timesteps).

The full training loop (loss decrease, optimizer step) is covered by::

    tests/test_e2e_integration.py  (CPU fast E2E)
    tests/test_e2e_real_checkpoint.py  (GPU slow E2E, real HF checkpoints)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Reference implementations matching d1 SFT / MDLM
# ---------------------------------------------------------------------------


def _d1_reference_loss_fused(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
) -> torch.Tensor:
    """d1/MDLM reference: n_maskable normalized CE via F.cross_entropy.

    Source alignment:
    - dev/repos/dllm/dllm/core/trainers/mdlm.py L202
    - dev/repos/d1/SFT/sft_trainer.py L25
    """
    B, L, V = logits.shape
    masked_labels = labels.clone()
    masked_labels[~diffusion_mask] = -100
    per_token = F.cross_entropy(
        logits.view(B * L, V),
        masked_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    n_maskable = (labels != -100).sum().clamp_min(1)
    return per_token.sum() / n_maskable


def _d1_reference_loss_weighted(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    loss_weights: torch.Tensor,  # (B,) or (B, L)
) -> torch.Tensor:
    """d1-style weighted reference with per-sequence weighting.

    Source: d1 SFT uses weight = 1/t per sequence, broadcast over L.
    """
    B, L, V = logits.shape
    masked_labels = labels.clone()
    masked_labels[~diffusion_mask] = -100
    per_token = F.cross_entropy(
        logits.view(B * L, V),
        masked_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, L)
    n_maskable = (labels != -100).sum().clamp_min(1)

    if loss_weights.shape == (B,):
        loss_weights = loss_weights.unsqueeze(1)
    weighted = per_token * loss_weights
    return weighted.sum() / n_maskable


# ---------------------------------------------------------------------------
# Helpers: generate diffusion-style inputs
# ---------------------------------------------------------------------------


def _make_diffusion_inputs(
    B: int,
    L: int,
    V: int,
    mask_rate: float,
    device: str = "cuda",
    seed: int = 42,
    prompt_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create logits, labels, diffusion_mask, timesteps for testing.

    Args:
        B, L, V: batch, sequence length, vocab size.
        mask_rate: fraction of completion tokens to mask.
        prompt_ratio: fraction of leading tokens set to -100 (prompt exclusion).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(B, L, V, generator=g, device=device, dtype=torch.float32)

    g = torch.Generator(device=device).manual_seed(seed + 1)
    labels = torch.randint(0, V, (B, L), generator=g, device=device)

    # Apply prompt exclusion: set leading tokens to -100
    if prompt_ratio > 0:
        n_prompt = int(L * prompt_ratio)
        if n_prompt > 0:
            labels[:, :n_prompt] = -100

    g = torch.Generator(device=device).manual_seed(seed + 2)
    diffusion_mask = torch.rand(B, L, generator=g, device=device) < mask_rate
    # Never mask prompt positions (-100 labels don't contribute anyway, but be explicit)
    if prompt_ratio > 0:
        n_prompt = int(L * prompt_ratio)
        if n_prompt > 0:
            diffusion_mask[:, :n_prompt] = False
    diffusion_mask[:, 0] = True  # always mask at least the first token if not prompt

    g = torch.Generator(device=device).manual_seed(seed + 3)
    timesteps = (
        torch.rand(B, generator=g, device=device) * (1.0 - 1e-3) + 1e-3
    )  # [eps, 1)

    return logits, labels, diffusion_mask, timesteps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mask_rate", [0.15, 0.50])
@pytest.mark.parametrize(
    "B,L,V",
    [
        (2, 32, 256),
        (4, 64, 512),
        (2, 128, 1024),
    ],
)
def test_uniform_loss_matches_d1_reference(mask_rate, B, L, V):
    """uniform loss_weight_type: unturtle Triton path vs d1 n_maskable reference."""
    from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

    logits, labels, diffusion_mask, _ = _make_diffusion_inputs(
        B, L, V, mask_rate, device="cuda"
    )

    # unturtle path (uniform = no loss_weights)
    unturtle_loss = fast_masked_diffusion_loss(
        logits=logits,
        labels=labels,
        diffusion_mask=diffusion_mask,
    )

    # d1 reference
    ref_loss = _d1_reference_loss_fused(logits, labels, diffusion_mask)

    assert unturtle_loss.shape == (), (
        f"Expected scalar, got shape {unturtle_loss.shape}"
    )
    assert torch.isfinite(unturtle_loss), (
        f"unturtle loss is not finite: {unturtle_loss}"
    )
    assert torch.allclose(
        unturtle_loss.cpu(),
        ref_loss.cpu(),
        atol=1e-4,
        rtol=1e-3,
    ), (
        f"unturtle={unturtle_loss.item():.6f} vs d1_ref={ref_loss.item():.6f} "
        f"(abs_diff={abs(unturtle_loss.item() - ref_loss.item()):.2e}) "
        f"mask_rate={mask_rate} B={B} L={L} V={V}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mask_rate", [0.15, 0.50])
@pytest.mark.parametrize(
    "B,L,V",
    [
        (2, 32, 256),
        (4, 64, 512),
        (2, 128, 1024),
    ],
)
def test_timestep_weighted_loss_matches_d1(mask_rate, B, L, V):
    """timestep loss_weight_type: weight = 1/t matches d1 SFT reference."""
    from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

    logits, labels, diffusion_mask, timesteps = _make_diffusion_inputs(
        B, L, V, mask_rate, device="cuda"
    )

    # unturtle path: 1/t weights broadcast to (B, L)
    weights = (1.0 / timesteps.clamp_min(1e-6)).unsqueeze(1).expand(B, L)
    unturtle_loss = fast_masked_diffusion_loss(
        logits=logits,
        labels=labels,
        diffusion_mask=diffusion_mask,
        loss_weights=weights,
    )

    # d1 reference
    ref_weights = 1.0 / timesteps.clamp_min(1e-6)  # (B,)
    ref_loss = _d1_reference_loss_weighted(
        logits,
        labels,
        diffusion_mask,
        loss_weights=ref_weights,
    )

    assert torch.allclose(
        unturtle_loss.cpu(),
        ref_loss.cpu(),
        atol=1e-4,
        rtol=1e-3,
    ), (
        f"unturtle={unturtle_loss.item():.6f} vs d1_ref={ref_loss.item():.6f} "
        f"mask_rate={mask_rate} B={B} L={L} V={V}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prompt_exclusion_loss_matches_reference():
    """labels with some -100 positions (prompt/padding) handled correctly."""
    from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

    B, L, V = 2, 32, 256
    # 50% of leading tokens are "prompt" (set to -100)
    logits, labels, diffusion_mask, _ = _make_diffusion_inputs(
        B,
        L,
        V,
        mask_rate=0.5,
        device="cuda",
        prompt_ratio=0.5,
    )

    unturtle_loss = fast_masked_diffusion_loss(
        logits=logits,
        labels=labels,
        diffusion_mask=diffusion_mask,
    )
    ref_loss = _d1_reference_loss_fused(logits, labels, diffusion_mask)

    assert torch.allclose(
        unturtle_loss.cpu(),
        ref_loss.cpu(),
        atol=1e-4,
        rtol=1e-3,
    ), f"unturtle={unturtle_loss.item():.6f} vs d1_ref={ref_loss.item():.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_cuda_paths_numerically_identical():
    """CPU fallback and CUDA path in the library produce identical loss values."""
    logits, labels, diffusion_mask, _ = _make_diffusion_inputs(
        B=2, L=32, V=128, mask_rate=0.5, device="cpu"
    )

    from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

    # Exercise the library's CPU fallback path
    cpu_loss = fast_masked_diffusion_loss(
        logits=logits,
        labels=labels,
        diffusion_mask=diffusion_mask,
    )

    cuda_logits = logits.cuda()
    cuda_labels = labels.cuda()
    cuda_mask = diffusion_mask.cuda()

    cuda_loss = fast_masked_diffusion_loss(
        logits=cuda_logits,
        labels=cuda_labels,
        diffusion_mask=cuda_mask,
    )

    assert torch.allclose(
        cpu_loss,
        cuda_loss.cpu(),
        atol=1e-5,
        rtol=1e-4,
    ), f"CPU={cpu_loss.item():.8f} vs CUDA={cuda_loss.item():.8f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compute_loss_wiring_matches_reference():
    """DiffusionTrainer.compute_loss wiring: args forwarded correctly to loss kernel.

    Validates that:
    1. model.forward(input_ids) produces logits
    2. logits + labels + diffusion_mask go to fast_masked_diffusion_loss correctly
    3. loss_weights is None for the "uniform" path
    4. loss matches the d1 reference for the same inputs
    """
    from unittest.mock import MagicMock, patch

    import torch.nn as nn

    B, L, V = 2, 16, 30

    class DummyModel(nn.Module):
        def forward(self, input_ids):
            B_i, L_i = input_ids.shape
            logits = torch.randn(B_i, L_i, V, device=input_ids.device)
            return MagicMock(logits=logits)

    batch = {
        "input_ids": torch.randint(0, V, (B, L)),
        "labels": torch.randint(0, V, (B, L)),
        "diffusion_mask": torch.rand(B, L) < 0.5,
        "timesteps": torch.rand(B) * 0.9 + 0.1,
    }

    from unturtle.diffusion import DiffusionTrainingArguments

    _ = DiffusionTrainingArguments(output_dir="/tmp")
    model = DummyModel()

    # Patch the loss to capture arguments
    captured = {}

    def capture_loss(logits, labels, diffusion_mask, loss_weights=None, **kw):
        captured["logits"] = logits
        captured["labels"] = labels
        captured["diffusion_mask"] = diffusion_mask
        captured["loss_weights"] = loss_weights
        from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

        return fast_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_weights=loss_weights,
        )

    # Keep references before compute_loss (which calls inputs.pop)
    orig_labels = batch["labels"]
    orig_mask = batch["diffusion_mask"]

    with patch("unturtle.diffusion.trainer.fast_masked_diffusion_loss", capture_loss):
        from unturtle.diffusion.trainer import DiffusionTrainer

        trainer = DiffusionTrainer.__new__(DiffusionTrainer)
        trainer._alpha_scheduler = None
        trainer._time_epsilon = 1e-3
        trainer._loss_weight_type = "uniform"
        trainer._cart_p = 0.8
        trainer._loss_norm_type = "token"
        trainer._right_shift_logits = False
        trainer.model = model
        trainer.model_accepts_loss_kwargs = False

        loss = trainer.compute_loss(model, batch)

    # Verify captured arguments match originals exactly
    assert "logits" in captured, "loss kernel was not called"
    assert torch.equal(captured["labels"], orig_labels), (
        "labels were modified: device mismatch or value change"
    )
    assert torch.equal(captured["diffusion_mask"], orig_mask), (
        "diffusion_mask was modified"
    )
    assert captured["loss_weights"] is None, (
        f"Expected uniform (no weights), got {captured['loss_weights']}"
    )
    # Loss must be finite
    assert torch.isfinite(loss), f"loss is not finite: {loss.item():.6f}"
    # Loss must match the d1 reference for the captured logits/labels/mask
    ref_loss = _d1_reference_loss_fused(
        captured["logits"],
        captured["labels"],
        captured["diffusion_mask"],
    )
    assert torch.allclose(
        loss.cpu(),
        ref_loss.cpu(),
        atol=1e-4,
        rtol=1e-3,
    ), (
        f"compute_loss={loss.item():.6f} vs d1_ref={ref_loss.item():.6f} "
        f"(abs_diff={abs(loss.item() - ref_loss.item()):.2e})"
    )


# -----------------------------------------------------------------------
# Trainer-level: DiffusionTrainer.compute_loss returns same value as kernel
# -----------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("loss_weight_type", ["uniform", "timestep"])
def test_diffusion_trainer_compute_loss_matches_kernel(loss_weight_type):
    """DiffusionTrainer.compute_loss should return the same loss as calling
    fast_masked_diffusion_loss directly with the extracted inputs.

    This validates the trainer's wiring: model forward → loss kernel call
    with correct arguments and loss_weights.
    """
    from unittest.mock import patch

    import torch.nn as nn

    B, L, V = 2, 16, 64

    class DummyModel(nn.Module):
        def forward(self, input_ids):
            B_i, L_i = input_ids.shape
            logits = torch.randn(B_i, L_i, V, device="cuda")
            return type("Outputs", (), {"logits": logits})()

    from unturtle.diffusion import DiffusionTrainingArguments
    from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

    _ = DiffusionTrainingArguments(output_dir="/tmp", loss_weight_type=loss_weight_type)
    model = DummyModel().cuda()

    captured = {}

    def capture_loss(logits, labels, diffusion_mask, loss_weights=None, **kw):
        captured["logits"] = logits
        captured["labels"] = labels
        captured["diffusion_mask"] = diffusion_mask
        captured["loss_weights"] = loss_weights
        return fast_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_weights=loss_weights,
        )

    torch.manual_seed(42)
    batch = {
        "input_ids": torch.randint(0, V, (B, L), device="cuda"),
        "labels": torch.randint(0, V, (B, L), device="cuda"),
        "diffusion_mask": torch.rand(B, L, device="cuda") < 0.5,
        "timesteps": torch.rand(B, device="cuda") * 0.9 + 0.1,
    }
    orig_labels = batch["labels"].clone()
    orig_mask = batch["diffusion_mask"].clone()
    orig_timesteps = batch["timesteps"].clone()

    with patch("unturtle.diffusion.trainer.fast_masked_diffusion_loss", capture_loss):
        from unturtle.diffusion.trainer import DiffusionTrainer

        trainer = DiffusionTrainer.__new__(DiffusionTrainer)
        trainer._alpha_scheduler = None
        trainer._time_epsilon = 1e-3
        trainer._loss_weight_type = loss_weight_type
        trainer._cart_p = 0.8
        trainer._loss_norm_type = "token"
        trainer._right_shift_logits = False
        trainer.model = model
        trainer.model_accepts_loss_kwargs = False

        trainer_loss = trainer.compute_loss(model, batch)

    # Verify wiring
    assert torch.equal(captured["labels"], orig_labels), "labels were modified"
    assert torch.equal(captured["diffusion_mask"], orig_mask), (
        "diffusion_mask was modified"
    )
    assert torch.isfinite(trainer_loss), f"loss not finite: {trainer_loss.item()}"

    # Recompute reference loss directly
    if loss_weight_type == "timestep":
        weights = 1.0 / orig_timesteps.clamp_min(1e-6)
        ref_loss = _d1_reference_loss_weighted(
            captured["logits"],
            captured["labels"],
            captured["diffusion_mask"],
            loss_weights=weights,
        )
    else:
        ref_loss = _d1_reference_loss_fused(
            captured["logits"],
            captured["labels"],
            captured["diffusion_mask"],
        )

    assert torch.allclose(
        trainer_loss.cpu(),
        ref_loss.cpu(),
        atol=1e-4,
        rtol=1e-3,
    ), (
        f"compute_loss={trainer_loss.item():.6f} vs d1_ref={ref_loss.item():.6f} "
        f"(loss_weight_type={loss_weight_type})"
    )
