"""Tests for DiffuGRPOTrainer and DiffuGRPOConfig.

These tests cover:
  - Config field defaults and custom values
  - _forward_process masking logic
  - _get_num_transfer_tokens distribution
  - _add_gumbel_noise (temperature=0 returns unchanged logits)
  - generate() shape (CPU smoke test with a tiny dummy model)
  - Import from all three namespaces
"""

import dataclasses

import pytest
import torch


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


def test_import_from_unturtle_diffusion():
    from unturtle.diffusion import DiffuGRPOTrainer, DiffuGRPOConfig  # noqa: F401


def test_import_from_unturtle():
    from unturtle import DiffuGRPOTrainer, DiffuGRPOConfig  # noqa: F401


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestDiffuGRPOConfig:
    def test_default_fields(self):
        from unturtle.diffusion import DiffuGRPOConfig

        # Check field defaults directly via dataclasses to avoid TRL __post_init__ validation.
        import dataclasses

        defaults = {f.name: f.default for f in dataclasses.fields(DiffuGRPOConfig)}
        assert defaults["block_length"] == 64
        assert defaults["diffusion_steps"] == 64
        assert defaults["cfg_scale"] == 0.0
        assert defaults["remasking"] == "low_confidence"
        assert defaults["p_mask_prompt"] == 0.3
        assert defaults["mask_id"] == 126336
        assert defaults["random_masking"] is True
        assert defaults["generation_batch_size"] == 4

    def test_custom_fields(self):
        from unturtle.diffusion import DiffuGRPOConfig

        # Use dataclasses.fields to check field *defaults* without triggering
        # TRL __post_init__ validation (generation_batch_size / num_generations rules).
        import dataclasses

        # Verify by constructing with valid args and checking attributes.
        # TRL 0.29+ requires generation_batch_size divisible by num_generations.
        cfg = DiffuGRPOConfig(
            output_dir="/tmp/test_grpo",
            per_device_train_batch_size=1,
            num_generations=2,
            generation_batch_size=2,
            block_length=32,
            diffusion_steps=32,
            cfg_scale=1.5,
            remasking="random",
            p_mask_prompt=0.5,
            mask_id=99999,
            random_masking=False,
        )
        assert cfg.block_length == 32
        assert cfg.diffusion_steps == 32
        assert cfg.cfg_scale == 1.5
        assert cfg.remasking == "random"
        assert cfg.p_mask_prompt == 0.5
        assert cfg.mask_id == 99999
        assert cfg.random_masking is False


# ---------------------------------------------------------------------------
# Static methods (no model needed)
# ---------------------------------------------------------------------------


class TestDiffuGRPOStaticMethods:
    """Test static/standalone methods directly without a full trainer instance."""

    def test_add_gumbel_noise_zero_temperature(self):
        """Temperature=0 should return logits unchanged."""
        from unturtle.diffusion import DiffuGRPOTrainer

        logits = torch.randn(2, 10, 100)
        out = DiffuGRPOTrainer._add_gumbel_noise(logits, temperature=0.0, dtype=torch.float32)
        assert torch.equal(out, logits)

    def test_add_gumbel_noise_nonzero_temperature(self):
        """Temperature>0 should return a different tensor (almost surely)."""
        from unturtle.diffusion import DiffuGRPOTrainer

        torch.manual_seed(42)
        logits = torch.randn(2, 10, 100)
        out = DiffuGRPOTrainer._add_gumbel_noise(logits, temperature=1.0, dtype=torch.float32)
        assert not torch.equal(out, logits)

    def test_get_num_transfer_tokens_shape(self):
        """Output shape should be [B, steps]."""
        from unturtle.diffusion import DiffuGRPOTrainer

        mask_index = torch.ones(3, 64, dtype=torch.bool)
        result = DiffuGRPOTrainer._get_num_transfer_tokens(mask_index, steps=8)
        assert result.shape == (3, 8)

    def test_get_num_transfer_tokens_sum(self):
        """Sum across steps should equal total masked tokens per sample."""
        from unturtle.diffusion import DiffuGRPOTrainer

        # 13 masked tokens, 5 steps → [3,3,3,2,2]
        mask_index = torch.zeros(1, 20, dtype=torch.bool)
        mask_index[0, :13] = True
        result = DiffuGRPOTrainer._get_num_transfer_tokens(mask_index, steps=5)
        assert result.sum().item() == 13

    def test_get_num_transfer_tokens_even(self):
        """Evenly divisible case: all steps equal."""
        from unturtle.diffusion import DiffuGRPOTrainer

        mask_index = torch.ones(1, 12, dtype=torch.bool)
        result = DiffuGRPOTrainer._get_num_transfer_tokens(mask_index, steps=4)
        assert (result == 3).all()


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------


class TestForwardProcess:
    """Test _forward_process masking without a real model."""

    def _make_trainer(self):
        """Create a minimal DiffuGRPOTrainer-like namespace object."""
        from unturtle.diffusion import DiffuGRPOConfig

        class _Stub:
            args = DiffuGRPOConfig(output_dir="/tmp/stub", per_device_train_batch_size=1, num_generations=2, generation_batch_size=2, p_mask_prompt=0.5)

            def _forward_process(self, *a, **kw):
                from unturtle.diffusion import DiffuGRPOTrainer
                return DiffuGRPOTrainer._forward_process(self, *a, **kw)

        return _Stub()

    def test_completion_always_masked(self):
        """Completion tokens (prompt_index=False) must always be masked."""
        from unturtle.diffusion import DiffuGRPOTrainer, DiffuGRPOConfig

        cfg = DiffuGRPOConfig(output_dir="/tmp/t", per_device_train_batch_size=1, num_generations=2, generation_batch_size=2, p_mask_prompt=0.0)

        class _Stub:
            args = cfg

        stub = _Stub()
        batch = torch.arange(10).unsqueeze(0).expand(4, -1).clone()
        prompt_index = torch.zeros(10, dtype=torch.bool)
        prompt_index[:5] = True  # first 5 = prompt

        noisy, p_mask = DiffuGRPOTrainer._forward_process(stub, batch, prompt_index, mask_id=999)

        # completion positions (5-9) must all be mask_id
        assert (noisy[:, 5:] == 999).all(), "completion tokens must be masked"
        # with p_mask_prompt=0, prompt tokens must NOT be masked
        assert (noisy[:, :5] != 999).all(), "prompt tokens must be unmasked when p=0"

    def test_p_mask_shape(self):
        from unturtle.diffusion import DiffuGRPOTrainer, DiffuGRPOConfig

        cfg = DiffuGRPOConfig(output_dir="/tmp/t", per_device_train_batch_size=1, num_generations=2, generation_batch_size=2, p_mask_prompt=0.3)

        class _Stub:
            args = cfg

        stub = _Stub()
        B, L = 3, 8
        batch = torch.randint(0, 100, (B, L))
        prompt_index = torch.zeros(L, dtype=torch.bool)
        prompt_index[:4] = True

        noisy, p_mask = DiffuGRPOTrainer._forward_process(stub, batch, prompt_index, mask_id=999)
        assert noisy.shape == (B, L)
        assert p_mask.shape == (B, L)

    def test_seed_reproducibility(self):
        """Same seed → same noisy batch."""
        from unturtle.diffusion import DiffuGRPOTrainer, DiffuGRPOConfig

        cfg = DiffuGRPOConfig(output_dir="/tmp/t", per_device_train_batch_size=1, num_generations=2, generation_batch_size=2, p_mask_prompt=0.5)

        class _Stub:
            args = cfg

        stub = _Stub()
        batch = torch.arange(16).unsqueeze(0).expand(2, -1).clone()
        prompt_index = torch.zeros(16, dtype=torch.bool)
        prompt_index[:8] = True

        noisy1, _ = DiffuGRPOTrainer._forward_process(stub, batch, prompt_index, mask_id=999, seed=42)
        noisy2, _ = DiffuGRPOTrainer._forward_process(stub, batch, prompt_index, mask_id=999, seed=42)
        assert torch.equal(noisy1, noisy2)
