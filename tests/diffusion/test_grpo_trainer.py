"""Tests for DiffuGRPOTrainer and DiffuGRPOConfig.

These tests cover:
  - Config field defaults and custom values
  - _forward_process masking logic
  - _get_num_transfer_tokens distribution
  - _add_gumbel_noise (temperature=0 returns unchanged logits)
  - generate() shape (CPU smoke test with a tiny dummy model)
  - generate(..., denoise_trajectory=) for wd1++ snapshots
  - Import from all three namespaces
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


def test_import_from_unturtle_diffusion():
    from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer  # noqa: F401


def test_import_from_unturtle():
    from unturtle import DiffuGRPOConfig, DiffuGRPOTrainer  # noqa: F401


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestDiffuGRPOConfig:
    def test_default_fields(self):
        # Check field defaults directly via dataclasses to avoid TRL __post_init__ validation.

        from unturtle.diffusion import DiffuGRPOConfig

        defaults = {f.name: f.default for f in dataclasses.fields(DiffuGRPOConfig)}
        assert defaults["block_length"] == 64
        assert defaults["diffusion_steps"] == 64
        assert defaults["cfg_scale"] == 0.0
        assert defaults["remasking"] == "low_confidence"
        assert defaults["p_mask_prompt"] == 0.3
        assert defaults["mask_id"] == 126336
        assert defaults["random_masking"] is True
        assert defaults["generation_batch_size"] is None
        assert defaults["diffu_policy_objective"] == "grpo"
        assert defaults["wd1_psi"] == 1.0

    def test_custom_fields(self):
        # Use dataclasses.fields to check field *defaults* without triggering
        # TRL __post_init__ validation (generation_batch_size / num_generations rules).

        from unturtle.diffusion import DiffuGRPOConfig

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

    def test_invalid_diffu_policy_objective_rejected(self):
        from unturtle.diffusion import DiffuGRPOConfig

        with pytest.raises(ValueError, match="diffu_policy_objective"):
            DiffuGRPOConfig(
                output_dir="/tmp/test_grpo",
                per_device_train_batch_size=1,
                num_generations=2,
                generation_batch_size=2,
                diffu_policy_objective="invalid",
            )

    def test_wd1plusplus_objective_allowed_on_config(self):
        from unturtle.diffusion import DiffuGRPOConfig

        cfg = DiffuGRPOConfig(
            output_dir="/tmp/test_grpo",
            per_device_train_batch_size=1,
            num_generations=2,
            generation_batch_size=2,
            diffu_policy_objective="wd1++",
        )
        assert cfg.diffu_policy_objective == "wd1++"


# ---------------------------------------------------------------------------
# Static methods (no model needed)
# ---------------------------------------------------------------------------


class TestDiffuGRPOStaticMethods:
    """Test static/standalone methods directly without a full trainer instance."""

    def test_add_gumbel_noise_zero_temperature(self):
        """Temperature=0 should return logits unchanged."""
        from unturtle.diffusion import DiffuGRPOTrainer

        logits = torch.randn(2, 10, 100)
        out = DiffuGRPOTrainer._add_gumbel_noise(
            logits, temperature=0.0, dtype=torch.float32
        )
        assert torch.equal(out, logits)

    def test_add_gumbel_noise_nonzero_temperature(self):
        """Temperature>0 should return a different tensor (almost surely)."""
        from unturtle.diffusion import DiffuGRPOTrainer

        torch.manual_seed(42)
        logits = torch.randn(2, 10, 100)
        out = DiffuGRPOTrainer._add_gumbel_noise(
            logits, temperature=1.0, dtype=torch.float32
        )
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
# generate() — denoise trajectory (wd1++ snapshots)
# ---------------------------------------------------------------------------


def test_generate_records_denoise_trajectory():
    pytest.importorskip("trl")
    from unturtle.diffusion import DiffuGRPOTrainer

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.d = nn.Parameter(torch.tensor(0.0))

        @property
        def device(self) -> torch.device:
            return self.d.device

        @property
        def dtype(self) -> torch.dtype:
            return self.d.dtype

        def forward(self, input_ids: torch.Tensor):
            b, s = input_ids.shape
            return SimpleNamespace(
                logits=torch.zeros(b, s, 32, device=self.device, dtype=self.dtype)
            )

    t = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    model = _M()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    traj: list[tuple[torch.Tensor, torch.Tensor]] = []
    out = DiffuGRPOTrainer.generate(
        t,
        model,
        prompt,
        steps=4,
        gen_length=8,
        block_length=4,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=99,
        denoise_trajectory=traj,
    )
    assert out.shape == (1, 3 + 8)
    assert len(traj) == 4
    x_l, x0p = traj[0]
    assert x_l.shape == x0p.shape == (1, 11)


def test_wd1pp_completion_logp_sum_at_masked_positions():
    pytest.importorskip("trl")
    from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer

    cfg = DiffuGRPOConfig(
        output_dir="/tmp/t",
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        p_mask_prompt=0.0,
        mask_id=99,
    )
    tr = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    tr.args = cfg
    x_l = torch.tensor([[10, 11, 99, 99, 5, 6]], dtype=torch.long)
    x0 = torch.tensor([[10, 11, 3, 7, 5, 6]], dtype=torch.long)

    class _Peak(nn.Module):
        def __init__(self, targets: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("d", torch.tensor(0.0))
            self.targets = targets

        def forward(self, input_ids: torch.Tensor):
            b, s, v = input_ids.shape[0], input_ids.shape[1], 32
            logits = torch.full(
                (b, s, v), -50.0, device=input_ids.device, dtype=torch.float32
            )
            for bi in range(b):
                for si in range(s):
                    tid = int(self.targets[bi, si].item())
                    if 0 <= tid < v:
                        logits[bi, si, tid] = 50.0
            return SimpleNamespace(logits=logits)

    model = _Peak(x0)
    out = DiffuGRPOTrainer._wd1pp_completion_logp_sum(
        tr, model, x_l, x0, logits_to_keep=4, seed=0
    )
    assert out.shape == (1,)
    assert out.item() > -0.05


def test_wd1plusplus_microbatch_slice_matches_buffer(tmp_path):
    """``compute_loss`` must slice trajectory with ``_step - 1`` (post-``_prepare_inputs`` increment)."""
    pytest.importorskip("trl")
    from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer

    cfg = DiffuGRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        diffu_policy_objective="wd1++",
    )
    tr = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    tr.args = cfg
    tr.num_generations = 2
    tr.num_iterations = 1
    tr.beta = 0.0
    tr.control = SimpleNamespace(should_evaluate=False)
    tr._diffu_mask_seeds = [99]
    tr._metrics = {"train": {"kl": [], "clip_ratio": []}}
    tr.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)
    xl = torch.cat(
        [torch.zeros(2, 6, dtype=torch.long), torch.ones(2, 6, dtype=torch.long)], dim=0
    )
    tr._wd1pp_trajectory = [(xl, xl.clone())]
    inputs = {
        "prompt_ids": torch.zeros(2, 2, dtype=torch.long),
        "completion_ids": torch.zeros(2, 4, dtype=torch.long),
        "completion_mask": torch.ones(2, 4, dtype=torch.float),
        "advantages": torch.tensor([1.0, -1.0]),
    }
    markers: list[float] = []

    def _capture(_m, xl_s, _x0, _ltk, _seed):
        markers.append(float(xl_s[:, 0].float().mean().item()))
        return torch.ones(xl_s.size(0), device=xl_s.device)

    tr._wd1pp_completion_logp_sum = _capture  # type: ignore[method-assign]

    tr._step = 1
    DiffuGRPOTrainer.compute_loss(tr, nn.Identity(), inputs)
    tr._step = 2
    DiffuGRPOTrainer.compute_loss(tr, nn.Identity(), inputs)
    assert markers == [0.0, 1.0]


def test_compute_loss_wd1plusplus_with_trajectory():
    pytest.importorskip("trl")
    from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer

    cfg = DiffuGRPOConfig(
        output_dir="/tmp/t",
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        diffu_policy_objective="wd1++",
    )
    tr = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    tr.args = cfg
    tr._step = 0
    tr.num_generations = 2
    tr.num_iterations = 1
    tr.beta = 0.0
    tr.control = SimpleNamespace(should_evaluate=False)
    tr._diffu_mask_seeds = [42]
    tr._metrics = {"train": {"kl": [], "clip_ratio": []}}
    tr.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)
    tr._wd1pp_trajectory = [
        (torch.zeros(2, 6, dtype=torch.long), torch.zeros(2, 6, dtype=torch.long)),
    ]
    tr.args.steps_per_generation = 1

    def _fake_wd1pp(_m, xl, x0, _ltk, _seed):
        assert xl.shape[0] == x0.shape[0] == 2
        return torch.tensor([0.5, -0.25], device=xl.device)

    tr._wd1pp_completion_logp_sum = _fake_wd1pp  # type: ignore[method-assign]

    def _fail_logps(*_a, **_kw):
        raise AssertionError("_get_per_token_logps must not run when beta=0 for wd1++")

    tr._get_per_token_logps = _fail_logps  # type: ignore[method-assign]

    inputs = {
        "prompt_ids": torch.zeros(2, 2, dtype=torch.long),
        "completion_ids": torch.zeros(2, 4, dtype=torch.long),
        "completion_mask": torch.ones(2, 4, dtype=torch.float),
        "advantages": torch.tensor([1.0, -1.0]),
    }
    loss = DiffuGRPOTrainer.compute_loss(tr, nn.Identity(), inputs)
    assert loss.shape == ()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Forward process
# ---------------------------------------------------------------------------


class TestForwardProcess:
    """Test _forward_process masking without a real model."""

    def _make_trainer(self):
        """Create a minimal DiffuGRPOTrainer-like namespace object."""
        from unturtle.diffusion import DiffuGRPOConfig

        class _Stub:
            args = DiffuGRPOConfig(
                output_dir="/tmp/stub",
                per_device_train_batch_size=1,
                num_generations=2,
                generation_batch_size=2,
                p_mask_prompt=0.5,
            )

            def _forward_process(self, *a, **kw):
                from unturtle.diffusion import DiffuGRPOTrainer

                return DiffuGRPOTrainer._forward_process(self, *a, **kw)

        return _Stub()

    def test_completion_always_masked(self):
        """Completion tokens (prompt_index=False) must always be masked (p_mask=1.0).

        This matches the d1 reference implementation where completion tokens
        are fully masked so the model must predict them from scratch.
        """
        from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer

        cfg = DiffuGRPOConfig(
            output_dir="/tmp/t",
            per_device_train_batch_size=1,
            num_generations=2,
            generation_batch_size=2,
            p_mask_prompt=0.0,
        )

        class _Stub:
            args = cfg

        stub = _Stub()
        batch = torch.arange(10).unsqueeze(0).expand(4, -1).clone()
        prompt_index = torch.zeros(10, dtype=torch.bool)
        prompt_index[:5] = True  # first 5 = prompt

        noisy, p_mask = DiffuGRPOTrainer._forward_process(
            stub, batch, prompt_index, mask_id=999
        )

        # completion positions (5-9) must all be mask_id
        assert (noisy[:, 5:] == 999).all(), "completion tokens must be masked"
        # with p_mask_prompt=0, prompt tokens must NOT be masked
        assert (noisy[:, :5] != 999).all(), "prompt tokens must be unmasked when p=0"
        # completion p_mask must be 1.0
        assert (p_mask[:, 5:] == 1.0).all(), "completion p_mask must be 1.0"

    def test_p_mask_shape(self):
        from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer

        cfg = DiffuGRPOConfig(
            output_dir="/tmp/t",
            per_device_train_batch_size=1,
            num_generations=2,
            generation_batch_size=2,
            p_mask_prompt=0.3,
        )

        class _Stub:
            args = cfg

        stub = _Stub()
        B, L = 3, 8
        batch = torch.randint(0, 100, (B, L))
        prompt_index = torch.zeros(L, dtype=torch.bool)
        prompt_index[:4] = True

        noisy, p_mask = DiffuGRPOTrainer._forward_process(
            stub, batch, prompt_index, mask_id=999
        )
        assert noisy.shape == (B, L)
        assert p_mask.shape == (B, L)

    def test_seed_reproducibility(self):
        """Same seed → same noisy batch."""
        from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer

        cfg = DiffuGRPOConfig(
            output_dir="/tmp/t",
            per_device_train_batch_size=1,
            num_generations=2,
            generation_batch_size=2,
            p_mask_prompt=0.5,
        )

        class _Stub:
            args = cfg

        stub = _Stub()
        batch = torch.arange(16).unsqueeze(0).expand(2, -1).clone()
        prompt_index = torch.zeros(16, dtype=torch.bool)
        prompt_index[:8] = True

        noisy1, _ = DiffuGRPOTrainer._forward_process(
            stub, batch, prompt_index, mask_id=999, seed=42
        )
        noisy2, _ = DiffuGRPOTrainer._forward_process(
            stub, batch, prompt_index, mask_id=999, seed=42
        )
        assert torch.equal(noisy1, noisy2)


# ---------------------------------------------------------------------------
# wd1 coefficients
# ---------------------------------------------------------------------------


def test_wd1_completion_coef_sums_to_zero_within_each_group():
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    advantages = torch.tensor([1.0, -0.5, -0.5, 2.0, -1.0, -1.0])
    coef = DiffuGRPOTrainer._wd1_completion_coef(advantages, num_generations=3, psi=1.0)
    assert coef.shape == advantages.shape
    assert torch.allclose(coef.view(2, 3).sum(dim=-1), torch.zeros(2), atol=1e-5)


# ---------------------------------------------------------------------------
# TRL buffering / mask seeds (regression)
# ---------------------------------------------------------------------------


def test_diffu_prepare_inputs_buffers_microbatches_and_regenerates_rollouts():
    """``steps_per_generation > 1`` and ``num_iterations > 1``: buffer slices + refresh cadence.

    Uses a bare instance (``__new__``) so we only exercise TRL-style buffering in
    :meth:`~unturtle.diffusion.DiffuGRPOTrainer._prepare_inputs` without constructing
    a full GRPO stack.
    """
    pytest.importorskip("trl")
    from types import SimpleNamespace

    import torch.nn as nn

    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    rollout_generation_count = 0

    def minimal_rollout_batch(batch_size: int) -> dict:
        return {
            "prompt_ids": torch.zeros(batch_size, 4, dtype=torch.long),
            "prompt_mask": torch.ones(batch_size, 4, dtype=torch.long),
            "completion_ids": torch.zeros(batch_size, 4, dtype=torch.long),
            "completion_mask": torch.ones(batch_size, 4, dtype=torch.long),
            "old_per_token_logps": torch.zeros(2, batch_size, 4),
            "advantages": torch.zeros(batch_size),
        }

    def fake_generate_and_score(_inputs: object) -> dict:
        nonlocal rollout_generation_count
        rollout_generation_count += 1
        v = rollout_generation_count
        tr._diffu_mask_seeds = [100 * v + 1, 100 * v + 2]
        return minimal_rollout_batch(2)

    tr = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    m = nn.Module()
    m.train()
    tr.model = m
    tr._step = 0
    tr._buffered_inputs = None
    tr.args = SimpleNamespace(steps_per_generation=2)
    tr.num_iterations = 2
    tr._generate_and_score_completions = fake_generate_and_score  # type: ignore[method-assign]
    tr._diffu_mask_seeds = []

    # One full GRPO "outer" cycle: steps_per_generation * num_iterations micro-steps.
    for _ in range(4):
        DiffuGRPOTrainer._prepare_inputs(tr, {})
    assert rollout_generation_count == 1
    assert tr._diffu_mask_seeds == [101, 102]
    assert tr._buffered_inputs is not None
    assert len(tr._buffered_inputs) == 2
    assert tr._buffered_inputs[0]["prompt_ids"].shape[0] == 1
    assert tr._buffered_inputs[1]["prompt_ids"].shape[0] == 1

    # Seeds set at rollout time stay on the trainer until the next regeneration
    # (this path does not put ``mask_seeds`` in the split batch dict; see the
    # ``split_tensor_dict`` regression test below).
    assert tr._diffu_mask_seeds == [101, 102]

    # Next cycle regenerates completions and new mask seeds.
    for _ in range(4):
        DiffuGRPOTrainer._prepare_inputs(tr, {})
    assert rollout_generation_count == 2
    assert tr._diffu_mask_seeds == [201, 202]


def _make_buffering_trainer(steps_per_generation: int, num_iterations: int):
    """Bare trainer wired for ``_prepare_inputs`` + ``compute_loss`` buffering tests."""
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    tr = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    m = nn.Module()
    m.train()
    tr.model = m
    tr._step = 0
    tr._buffered_inputs = None
    tr.args = SimpleNamespace(
        steps_per_generation=steps_per_generation,
        num_iterations=num_iterations,
        diffu_policy_objective="grpo",
    )
    tr.num_iterations = num_iterations
    tr.num_generations = 2
    tr.beta = 0.0
    tr.control = SimpleNamespace(should_evaluate=False)
    tr.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)
    tr._metrics = {"train": {"kl": [], "clip_ratio": []}}
    tr._diffu_mask_seeds = []
    return tr


@pytest.mark.parametrize("beta", [0.0, 0.04])
def test_prepare_inputs_slices_cached_logps_along_batch_dim_and_compute_loss_runs(beta):
    """Bug regression: ``old/ref_per_token_logps`` are ``[num_iterations, B_gen, Lc]``.

    ``split_tensor_dict`` slices dim 0 (the iterations dim for these tensors), so
    ``_prepare_inputs`` must instead attach the dim-1 (batch) slice per micro-batch.
    Also checks ``compute_loss`` consumes each micro-batch with the correct GRPO
    iteration index (one advance per full pass over the buffer, not per micro-batch)
    and threads the buffer row window into ``_get_per_token_logps``.

    ``beta=0.04`` additionally exercises the KL path over the sliced
    ``ref_per_token_logps[this_itr_idx]``. Also regression for eval-rollout
    clobbering: ``mask_seeds`` travel with the buffered inputs, so overwriting
    the trainer attribute (as an eval rollout does) must not change the seeds
    used by ``compute_loss``.
    """
    pytest.importorskip("trl")
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    S, num_iterations, B_gen, Lc = 2, 2, 4, 4
    cs = B_gen // S  # micro-batch size
    expected_seeds = [11, 22]

    # Distinctive per-(iteration, row) values so slices are checkable.
    full_old = (
        torch.arange(num_iterations * B_gen * Lc, dtype=torch.float32).view(
            num_iterations, B_gen, Lc
        )
        * 1e-3
    )
    full_ref = full_old + 0.5  # small offset: keeps exp(ref - logps) finite

    rollout_calls = 0

    def fake_generate_and_score(_inputs: object) -> dict:
        nonlocal rollout_calls
        rollout_calls += 1
        tr._diffu_mask_seeds = list(expected_seeds)
        return {
            "prompt_ids": torch.zeros(B_gen, 4, dtype=torch.long),
            "prompt_mask": torch.ones(B_gen, 4, dtype=torch.long),
            "completion_ids": torch.zeros(B_gen, Lc, dtype=torch.long),
            "completion_mask": torch.ones(B_gen, Lc, dtype=torch.long),
            "old_per_token_logps": full_old.clone(),
            "ref_per_token_logps": full_ref.clone(),
            "advantages": torch.zeros(B_gen),
            "mask_seeds": list(expected_seeds),
        }

    tr = _make_buffering_trainer(S, num_iterations)
    tr.beta = beta
    tr._generate_and_score_completions = fake_generate_and_score  # type: ignore[method-assign]

    logps_calls: list[dict] = []

    def spy_get_per_token_logps(
        _model,
        input_ids,
        logits_to_keep,
        mask_seeds,
        buffer_total_rows=None,
        buffer_row_offset=0,
    ):
        logps_calls.append(
            {
                "seeds": list(mask_seeds),
                "total_rows": buffer_total_rows,
                "row_offset": buffer_row_offset,
            }
        )
        n_it, b, _l = input_ids.shape
        return torch.zeros(n_it, b, logits_to_keep)

    tr._get_per_token_logps = spy_get_per_token_logps  # type: ignore[method-assign]

    # One full generate_every cycle = S * num_iterations micro-steps.
    for t in range(S * num_iterations):
        inputs = DiffuGRPOTrainer._prepare_inputs(tr, {})
        micro_idx = t % S
        expected_itr = t // S

        # Eval-rollout clobber simulation: an eval `_generate_and_score_completions`
        # overwrites the trainer attribute mid-cycle. Buffered inputs must be immune.
        tr._diffu_mask_seeds = [999, 998]
        assert inputs["mask_seeds"] == expected_seeds

        # Cached log-probs: iterations preserved on dim 0, batch sliced on dim 1.
        for key, full in (
            ("old_per_token_logps", full_old),
            ("ref_per_token_logps", full_ref),
        ):
            got = inputs[key]
            assert got.shape == (num_iterations, cs, Lc), key
            assert torch.equal(got, full[:, micro_idx * cs : (micro_idx + 1) * cs]), key

        # Buffer row window for train-time mask reproduction.
        assert inputs["buffer_total_rows"] == B_gen
        assert inputs["buffer_row_offset"] == micro_idx * cs

        loss = DiffuGRPOTrainer.compute_loss(tr, tr.model, inputs)
        assert torch.isfinite(loss)

        # compute_loss must use the seed of the current GRPO iteration
        # (advancing once per full pass over the buffer) and pass the window —
        # sourced from the buffered inputs, not the (clobbered) trainer attribute.
        call = logps_calls[-1]
        assert call["seeds"] == [expected_seeds[expected_itr]], f"micro-step {t}"
        assert call["total_rows"] == B_gen
        assert call["row_offset"] == micro_idx * cs

    assert rollout_calls == 1
    assert len(logps_calls) == S * num_iterations


def test_prepare_inputs_keeps_empty_cached_logps_lists():
    """``old/ref_per_token_logps`` are ``[]`` when unused — chunks must keep them ``[]``."""
    pytest.importorskip("trl")
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    def fake_generate_and_score(_inputs: object) -> dict:
        tr._diffu_mask_seeds = [7]
        return {
            "prompt_ids": torch.zeros(4, 4, dtype=torch.long),
            "completion_ids": torch.zeros(4, 4, dtype=torch.long),
            "completion_mask": torch.ones(4, 4, dtype=torch.long),
            "old_per_token_logps": [],
            "ref_per_token_logps": [],
            "advantages": torch.zeros(4),
        }

    tr = _make_buffering_trainer(steps_per_generation=2, num_iterations=1)
    tr._generate_and_score_completions = fake_generate_and_score  # type: ignore[method-assign]

    for _t in range(2):
        inputs = DiffuGRPOTrainer._prepare_inputs(tr, {})
        assert inputs["old_per_token_logps"] == []
        assert inputs["ref_per_token_logps"] == []


def test_forward_process_window_reproduces_rollout_mask():
    """Windowed draw: rand((total_rows, L)) sliced at the row offset matches the
    rows of the full-batch draw under the same seed (Bug: micro-batches after the
    first were re-masked differently than their cached old/ref log-probs)."""
    from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer

    cfg = DiffuGRPOConfig(
        output_dir="/tmp/t",
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        p_mask_prompt=0.5,
    )

    class _Stub:
        args = cfg

    stub = _Stub()
    L = 16
    batch = torch.arange(1, 4 * L + 1).view(4, L)
    prompt_index = torch.zeros(L, dtype=torch.bool)
    prompt_index[:8] = True

    # Rollout-time draw over the full [4, L] batch.
    noisy_full, p_full = DiffuGRPOTrainer._forward_process(
        stub, batch, prompt_index, mask_id=999, seed=42
    )
    # Train-time draw over the second micro-batch (rows 2:4) with the window.
    noisy_win, p_win = DiffuGRPOTrainer._forward_process(
        stub,
        batch[2:4],
        prompt_index,
        mask_id=999,
        seed=42,
        total_rows=4,
        row_offset=2,
    )
    assert torch.equal(noisy_win, noisy_full[2:4])
    assert torch.equal(p_win, p_full[2:4])

    # Without the window (legacy behavior) rows 0:2 of the full draw are
    # reproduced instead — i.e. the wrong mask for rows 2:4.
    noisy_plain, _ = DiffuGRPOTrainer._forward_process(
        stub, batch[2:4], prompt_index, mask_id=999, seed=42
    )
    masked_pattern_plain = noisy_plain == 999
    masked_pattern_rows01 = (noisy_full == 999)[0:2]
    assert torch.equal(masked_pattern_plain, masked_pattern_rows01)

    # S=1 bit-identity: total_rows == b, offset 0 must equal the plain draw.
    noisy_a, _ = DiffuGRPOTrainer._forward_process(
        stub, batch, prompt_index, mask_id=999, seed=42
    )
    noisy_b, _ = DiffuGRPOTrainer._forward_process(
        stub, batch, prompt_index, mask_id=999, seed=42, total_rows=4, row_offset=0
    )
    assert torch.equal(noisy_a, noisy_b)


def test_trl_split_tensor_dict_empty_second_chunk_for_iteration_sized_list():
    """TRL slices every sequence-like value by batch chunk — not ``num_iterations``.

    ``mask_seeds`` has length ``num_iterations`` (one seed per inner GRPO step), which
    is unrelated to micro-batch size. If it lived in the post-generation dict,
    ``split_tensor_dict(..., steps_per_generation)`` would slice the list along the
    batch dimension and later chunks would see an empty list — the bug fixed by
    storing seeds on :attr:`DiffuGRPOTrainer._diffu_mask_seeds` instead.
    """
    pytest.importorskip("trl")
    from trl.trainer.utils import split_tensor_dict

    batch = {
        "input_ids": torch.zeros(4, 8, dtype=torch.long),
        "mask_seeds": [101, 202],
    }
    chunks = split_tensor_dict(batch, num_chunks=2)
    assert chunks[0]["mask_seeds"] == [101, 202]
    assert chunks[1]["mask_seeds"] == []


def test_wd1pp_buffered_chunks_survive_eval_rollout_clobber():
    """wd1++: ``mask_seeds`` and the denoise trajectory travel with the buffered
    chunks, so an eval rollout that resets/overwrites the trainer attributes
    mid-generation-cycle must not affect train-time ``compute_loss``."""
    pytest.importorskip("trl")
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    B_gen, S = 4, 2
    xl = torch.cat(
        [torch.zeros(2, 6, dtype=torch.long), torch.ones(2, 6, dtype=torch.long)], dim=0
    )
    traj = [(xl, xl.clone())]

    def fake_generate_and_score(_inputs: object) -> dict:
        tr._diffu_mask_seeds = [7]
        tr._wd1pp_trajectory = traj
        return {
            "prompt_ids": torch.zeros(B_gen, 2, dtype=torch.long),
            "completion_ids": torch.zeros(B_gen, 4, dtype=torch.long),
            "completion_mask": torch.ones(B_gen, 4, dtype=torch.float),
            "old_per_token_logps": [],
            "ref_per_token_logps": [],
            "advantages": torch.tensor([1.0, -1.0, 0.5, -0.5]),
            "mask_seeds": [7],
            "wd1pp_trajectory": traj,
        }

    tr = _make_buffering_trainer(steps_per_generation=S, num_iterations=1)
    tr.args.diffu_policy_objective = "wd1++"
    tr.args.wd1_psi = 1.0
    tr._generate_and_score_completions = fake_generate_and_score  # type: ignore[method-assign]
    tr._wd1pp_trajectory = None

    seen: list[tuple[float, int]] = []

    def _capture(_m, xl_s, _x0, _ltk, seed):
        seen.append((float(xl_s[:, 0].float().mean().item()), int(seed)))
        return torch.ones(xl_s.size(0))

    tr._wd1pp_completion_logp_sum = _capture  # type: ignore[method-assign]

    for _t in range(S):
        inputs = DiffuGRPOTrainer._prepare_inputs(tr, {})
        # Full (unsliced) per-rollout artifacts on every chunk.
        assert inputs["mask_seeds"] == [7]
        assert inputs["wd1pp_trajectory"] is traj
        # Eval rollout lands mid-cycle: trainer attributes clobbered.
        tr._diffu_mask_seeds = [999]
        tr._wd1pp_trajectory = None
        loss = DiffuGRPOTrainer.compute_loss(tr, nn.Identity(), inputs)
        assert torch.isfinite(loss)

    # Chunk 0 saw trajectory rows 0:2 (zeros), chunk 1 rows 2:4 (ones); both
    # used the buffered seed 7, not the clobbered trainer attribute.
    assert seen == [(0.0, 7), (1.0, 7)]


# ---------------------------------------------------------------------------
# Reference log-probs: PEFT adapter vs TRL ref_model (beta != 0)
# ---------------------------------------------------------------------------


def _make_ref_logps_trainer(model: nn.Module):
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    tr = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    tr.model = model
    tr.num_iterations = 1
    tr.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    calls: list[nn.Module] = []

    def spy_get_per_token_logps(model_arg, input_ids, logits_to_keep, mask_seeds):
        calls.append(model_arg)
        n_it, b, _l = input_ids.shape
        return torch.zeros(n_it, b, logits_to_keep)

    tr._get_per_token_logps = spy_get_per_token_logps  # type: ignore[method-assign]
    return tr, calls


def test_compute_ref_logps_prefers_peft_disable_adapter():
    """A policy with ``disable_adapter`` acts as its own reference (d1/TRL PEFT path)."""
    pytest.importorskip("trl")
    from contextlib import contextmanager

    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    entered: list[bool] = []

    class _PeftLike(nn.Module):
        @contextmanager
        def disable_adapter(self):
            entered.append(True)
            yield

    model = _PeftLike()
    tr, calls = _make_ref_logps_trainer(model)
    tr.ref_model = None
    ids = torch.zeros(2, 6, dtype=torch.long)
    out = DiffuGRPOTrainer._compute_ref_per_token_logps(tr, ids, 4, [7])
    assert out.shape == (1, 2, 4)
    assert entered == [True]
    assert calls == [model]


def test_compute_ref_logps_uses_trl_ref_model_for_full_finetune():
    """Regression (#48): full-finetune policy (no ``disable_adapter``) must use
    TRL's ``self.ref_model`` instead of raising ``AttributeError``."""
    pytest.importorskip("trl")
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    model = nn.Identity()
    ref_model = nn.Identity()
    tr, calls = _make_ref_logps_trainer(model)
    tr.ref_model = ref_model
    ids = torch.zeros(2, 6, dtype=torch.long)
    out = DiffuGRPOTrainer._compute_ref_per_token_logps(tr, ids, 4, [7])
    assert out.shape == (1, 2, 4)
    assert len(calls) == 1
    assert calls[0] is ref_model


def test_compute_ref_logps_raises_without_adapter_or_ref_model():
    pytest.importorskip("trl")
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    tr, calls = _make_ref_logps_trainer(nn.Identity())
    tr.ref_model = None
    ids = torch.zeros(2, 6, dtype=torch.long)
    with pytest.raises(ValueError, match="reference policy"):
        DiffuGRPOTrainer._compute_ref_per_token_logps(tr, ids, 4, [7])
    assert calls == []


# ---------------------------------------------------------------------------
# wd1 / wd1++ group alignment guard (#48)
# ---------------------------------------------------------------------------


def test_wd1_compute_loss_rejects_partial_reward_groups():
    """Micro-batch not divisible by ``num_generations`` must fail fast with an
    actionable error (not a raw reshape error / silent partial-group softmax)."""
    pytest.importorskip("trl")
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    tr = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    tr.args = SimpleNamespace(
        steps_per_generation=1, diffu_policy_objective="wd1", wd1_psi=1.0
    )
    tr.num_iterations = 1
    tr.num_generations = 2
    tr.beta = 0.0
    tr._step = 1
    tr._diffu_mask_seeds = [3]
    tr.control = SimpleNamespace(should_evaluate=False)
    tr._metrics = {"train": {"kl": [], "clip_ratio": []}}
    tr.accelerator = SimpleNamespace(gather_for_metrics=lambda x: x)

    inputs = {
        "prompt_ids": torch.zeros(3, 2, dtype=torch.long),
        "completion_ids": torch.zeros(3, 4, dtype=torch.long),
        "completion_mask": torch.ones(3, 4, dtype=torch.float),
        "advantages": torch.tensor([1.0, -1.0, 0.5]),  # 3 % num_generations != 0
    }
    with pytest.raises(ValueError, match="whole reward groups"):
        DiffuGRPOTrainer.compute_loss(tr, nn.Identity(), inputs)


# ---------------------------------------------------------------------------
# Advantage scaling (TRL scale_rewards semantics, #48)
# ---------------------------------------------------------------------------


def _make_advantage_trainer(scale_rewards):
    from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

    tr = DiffuGRPOTrainer.__new__(DiffuGRPOTrainer)
    tr.num_generations = 2
    tr.args = SimpleNamespace(scale_rewards=scale_rewards)
    return tr


class TestComputeGroupAdvantages:
    rewards = torch.tensor([1.0, 3.0, 2.0, 2.0])

    def test_none_is_d1_mean_centering_only(self):
        from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

        tr = _make_advantage_trainer("none")
        adv, std = DiffuGRPOTrainer._compute_group_advantages(tr, self.rewards)
        assert torch.allclose(adv, torch.tensor([-1.0, 1.0, 0.0, 0.0]))
        # std still computed for logging (group std)
        assert torch.allclose(std, torch.tensor([2.0, 2.0, 0.0, 0.0]).sqrt())

    def test_false_maps_to_none(self):
        from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

        tr = _make_advantage_trainer(False)
        adv, _ = DiffuGRPOTrainer._compute_group_advantages(tr, self.rewards)
        assert torch.allclose(adv, torch.tensor([-1.0, 1.0, 0.0, 0.0]))

    def test_group_scaling_divides_by_group_std(self):
        from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

        tr = _make_advantage_trainer("group")
        adv, std = DiffuGRPOTrainer._compute_group_advantages(tr, self.rewards)
        g_std = torch.tensor([2.0]).sqrt().item()
        expected = torch.tensor([-1.0 / (g_std + 1e-4), 1.0 / (g_std + 1e-4), 0.0, 0.0])
        assert torch.allclose(adv, expected)

    def test_batch_scaling_divides_by_batch_std(self):
        from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

        tr = _make_advantage_trainer("batch")
        adv, std = DiffuGRPOTrainer._compute_group_advantages(tr, self.rewards)
        b_std = self.rewards.std().item()
        expected = torch.tensor([-1.0, 1.0, 0.0, 0.0]) / (b_std + 1e-4)
        assert torch.allclose(adv, expected)
        assert torch.allclose(std, torch.full((4,), b_std))

    def test_invalid_value_rejected(self):
        from unturtle.diffusion.grpo_trainer import DiffuGRPOTrainer

        tr = _make_advantage_trainer("bogus")
        with pytest.raises(ValueError, match="scale_rewards"):
            DiffuGRPOTrainer._compute_group_advantages(tr, self.rewards)
