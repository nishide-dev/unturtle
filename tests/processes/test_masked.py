"""
Tests for the pure masked-diffusion forward process (#70, #62 PR1).

Validates that ``MaskedDiffusionProcess`` reproduces the masking / label
semantics of the current non-packed ``MaskedDiffusionDataCollator`` when
given an equivalent pre-collated clean batch, plus the new contracts the
process layer introduces (non-mutation, explicit-generator reproducibility,
model-inputs vs objective-inputs split).

Run with:
    pytest tests/processes/test_masked.py -v
"""

import pytest
import torch

from unturtle.processes import MaskedDiffusionProcess, ProcessOutput

MASK_ID = 99

# ---------------------------------------------------------------------------
# Fake schedulers (deterministic extremes — no probabilistic assertions)
# ---------------------------------------------------------------------------


class KeepAll:
    """alpha(t) = 1 → p_mask = 0 → nothing is masked."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)


class MaskAll:
    """alpha(t) = 0 → p_mask = 1 → every eligible position is masked."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)


class FloatMaskAll:
    """Scheduler returning a plain Python float (scalar normalization path)."""

    def alpha(self, t: torch.Tensor) -> float:
        return 0.0


class HalfLinear:
    """A normal vectorized schedule: alpha(t) = 1 - t."""

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def make_batch(with_labels: bool = True, with_attention_mask: bool = True):
    """Tiny hand-constructed clean batch.

    Row 0: 2 prompt tokens + 2 completion tokens (no padding)
    Row 1: 1 prompt token + 1 completion token + 2 padding positions
    """
    batch = {
        "input_ids": torch.tensor([[5, 6, 7, 8], [9, 10, 0, 0]], dtype=torch.long),
    }
    if with_attention_mask:
        batch["attention_mask"] = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.long
        )
    if with_labels:
        batch["labels"] = torch.tensor(
            [[-100, -100, 7, 8], [-100, 10, -100, -100]], dtype=torch.long
        )
    return batch


def make_process(scheduler=None, **kwargs) -> MaskedDiffusionProcess:
    return MaskedDiffusionProcess(
        scheduler=scheduler if scheduler is not None else MaskAll(),
        mask_token_id=MASK_ID,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# A. Shape and key contract
# ---------------------------------------------------------------------------


class TestShapeAndKeyContract:
    def test_output_shapes(self):
        batch = make_batch()
        out = make_process()(batch)

        assert isinstance(out, ProcessOutput)
        assert out.model_inputs["input_ids"].shape == (2, 4)
        assert out.objective_inputs["labels"].shape == (2, 4)
        assert out.objective_inputs["diffusion_mask"].shape == (2, 4)
        assert out.objective_inputs["timesteps"].shape == (2,)

    def test_unrelated_fields_survive_in_model_inputs(self):
        batch = make_batch()
        batch["position_ids"] = torch.arange(4).expand(2, 4)
        batch["packed_seq_lengths"] = [[4], [2]]
        batch["block_attention_mask"] = torch.ones(2, 4, 4)
        batch["custom_kwarg"] = "keep-me"

        out = make_process()(batch)

        assert torch.equal(out.model_inputs["position_ids"], batch["position_ids"])
        assert out.model_inputs["packed_seq_lengths"] == [[4], [2]]
        assert torch.equal(
            out.model_inputs["block_attention_mask"], batch["block_attention_mask"]
        )
        assert out.model_inputs["custom_kwarg"] == "keep-me"
        assert torch.equal(out.model_inputs["attention_mask"], batch["attention_mask"])

    def test_supervision_fields_not_in_model_inputs(self):
        out = make_process()(make_batch())

        assert "labels" not in out.model_inputs
        assert "diffusion_mask" not in out.model_inputs
        assert "timesteps" not in out.model_inputs


# ---------------------------------------------------------------------------
# B. Input is not mutated
# ---------------------------------------------------------------------------


class TestNonMutation:
    def test_input_batch_and_tensors_unchanged(self):
        batch = make_batch()
        snapshot = {k: v.clone() for k, v in batch.items()}
        keys_before = set(batch.keys())

        make_process()(batch)

        assert set(batch.keys()) == keys_before
        for k, v in snapshot.items():
            assert torch.equal(batch[k], v), f"input field {k!r} was mutated"

    def test_noised_input_ids_is_not_the_input_tensor(self):
        batch = make_batch()
        out = make_process(scheduler=KeepAll())(batch)

        # Even when nothing is masked, the model input must be a fresh tensor.
        assert out.model_inputs["input_ids"] is not batch["input_ids"]

    @pytest.mark.parametrize("completion_only", [True, False])
    def test_returned_labels_do_not_alias_the_input_labels(self, completion_only):
        # PR2 hands these to a Trainer that pops/edits `labels` on a batch the
        # caller may still hold, so aliasing must be impossible in either mode.
        batch = make_batch()
        out = make_process(scheduler=KeepAll(), completion_only=completion_only)(batch)

        labels = out.objective_inputs["labels"]
        assert labels is not batch["labels"]
        assert labels is not batch["input_ids"]

        original = batch["labels"].clone()
        labels[:] = -100
        assert torch.equal(batch["labels"], original)

    def test_returned_diffusion_mask_does_not_alias_attention_mask(self):
        batch = make_batch(with_labels=False)
        out = make_process(scheduler=MaskAll(), completion_only=False)(batch)

        original = batch["attention_mask"].clone()
        out.objective_inputs["diffusion_mask"][:] = False
        assert torch.equal(batch["attention_mask"], original)


# ---------------------------------------------------------------------------
# C. Completion-only semantics
# ---------------------------------------------------------------------------


class TestCompletionOnly:
    def test_prompt_never_masked_and_completion_fully_masked(self):
        batch = make_batch()
        out = make_process(scheduler=MaskAll())(batch)

        noised = out.model_inputs["input_ids"]
        mask = out.objective_inputs["diffusion_mask"]
        expected_maskable = batch["labels"] != -100

        assert torch.equal(mask, expected_maskable)
        assert torch.equal(
            noised[~expected_maskable], batch["input_ids"][~expected_maskable]
        )
        assert (noised[expected_maskable] == MASK_ID).all()

    def test_labels_keep_prompt_minus_100_and_completion_targets(self):
        batch = make_batch()
        out = make_process(scheduler=MaskAll())(batch)

        assert torch.equal(out.objective_inputs["labels"], batch["labels"])


# ---------------------------------------------------------------------------
# D. Full-sequence semantics
# ---------------------------------------------------------------------------


class TestFullSequence:
    def test_all_attended_positions_eligible_padding_untouched(self):
        batch = make_batch()
        out = make_process(scheduler=MaskAll(), completion_only=False)(batch)

        attended = batch["attention_mask"].bool()
        mask = out.objective_inputs["diffusion_mask"]
        noised = out.model_inputs["input_ids"]

        assert torch.equal(mask, attended)
        assert (noised[attended] == MASK_ID).all()
        assert torch.equal(noised[~attended], batch["input_ids"][~attended])

    def test_labels_are_clean_ids_with_padding_minus_100(self):
        batch = make_batch()
        out = make_process(scheduler=MaskAll(), completion_only=False)(batch)

        labels = out.objective_inputs["labels"]
        attended = batch["attention_mask"].bool()

        assert torch.equal(labels[attended], batch["input_ids"][attended])
        assert (labels[~attended] == -100).all()

    def test_no_attention_mask_all_positions_eligible(self):
        batch = make_batch(with_labels=False, with_attention_mask=False)
        out = make_process(scheduler=MaskAll(), completion_only=False)(batch)

        assert out.objective_inputs["diffusion_mask"].all()
        assert (out.model_inputs["input_ids"] == MASK_ID).all()
        assert torch.equal(out.objective_inputs["labels"], batch["input_ids"])


# ---------------------------------------------------------------------------
# E. Completion-only without labels (legacy fallback)
# ---------------------------------------------------------------------------


class TestCompletionOnlyWithoutLabels:
    def test_falls_back_to_attention_mask_eligibility(self):
        batch = make_batch(with_labels=False)
        out = make_process(scheduler=MaskAll(), completion_only=True)(batch)

        attended = batch["attention_mask"].bool()
        assert torch.equal(out.objective_inputs["diffusion_mask"], attended)
        labels = out.objective_inputs["labels"]
        assert torch.equal(labels[attended], batch["input_ids"][attended])
        assert (labels[~attended] == -100).all()

    def test_falls_back_to_all_positions_without_attention_mask(self):
        batch = make_batch(with_labels=False, with_attention_mask=False)
        out = make_process(scheduler=MaskAll(), completion_only=True)(batch)

        assert out.objective_inputs["diffusion_mask"].all()
        assert torch.equal(out.objective_inputs["labels"], batch["input_ids"])


# ---------------------------------------------------------------------------
# F. Deterministic extremes
# ---------------------------------------------------------------------------


class TestDeterministicExtremes:
    def test_keep_all_masks_nothing(self):
        batch = make_batch()
        out = make_process(scheduler=KeepAll())(batch)

        assert not out.objective_inputs["diffusion_mask"].any()
        assert torch.equal(out.model_inputs["input_ids"], batch["input_ids"])

    def test_mask_all_masks_every_eligible_position_only(self):
        batch = make_batch()
        out = make_process(scheduler=MaskAll())(batch)

        eligible = batch["labels"] != -100
        assert torch.equal(out.objective_inputs["diffusion_mask"], eligible)


# ---------------------------------------------------------------------------
# G. Explicit generator reproducibility
# ---------------------------------------------------------------------------


class TestGeneratorReproducibility:
    def test_same_seed_same_draws(self):
        batch = make_batch()
        process = make_process(scheduler=HalfLinear())

        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        out1 = process(batch, generator=g1)
        out2 = process(batch, generator=g2)

        assert torch.equal(
            out1.objective_inputs["timesteps"], out2.objective_inputs["timesteps"]
        )
        assert torch.equal(
            out1.objective_inputs["diffusion_mask"],
            out2.objective_inputs["diffusion_mask"],
        )
        assert torch.equal(
            out1.model_inputs["input_ids"], out2.model_inputs["input_ids"]
        )

    def test_different_seeds_differ(self):
        # Nontrivial batch so seed collisions are effectively impossible for t.
        batch = {"input_ids": torch.randint(1, 50, (8, 16))}
        process = make_process(scheduler=HalfLinear())

        out1 = process(batch, generator=torch.Generator().manual_seed(1))
        out2 = process(batch, generator=torch.Generator().manual_seed(2))

        assert not torch.equal(
            out1.objective_inputs["timesteps"], out2.objective_inputs["timesteps"]
        )


# ---------------------------------------------------------------------------
# H. Time epsilon bounds
# ---------------------------------------------------------------------------


class TestTimeEpsilonBounds:
    def test_timesteps_within_eps_and_one(self):
        eps = 0.3
        batch = {"input_ids": torch.randint(1, 50, (512, 2))}
        process = make_process(scheduler=HalfLinear(), time_epsilon=eps)

        t = process(batch, generator=torch.Generator().manual_seed(0)).objective_inputs[
            "timesteps"
        ]

        # torch.rand is [0, 1) so t is [eps, 1)
        assert (t >= eps).all()
        assert (t < 1.0).all()

    def test_zero_epsilon_is_valid(self):
        # 0 <= time_epsilon < 1 per spec, so 0.0 must be accepted, not rejected.
        batch = {"input_ids": torch.randint(1, 50, (64, 2))}
        process = make_process(scheduler=HalfLinear(), time_epsilon=0.0)

        t = process(batch, generator=torch.Generator().manual_seed(0)).objective_inputs[
            "timesteps"
        ]
        assert (t >= 0.0).all()
        assert (t < 1.0).all()


# ---------------------------------------------------------------------------
# I. Scheduler return normalization
# ---------------------------------------------------------------------------


class TestSchedulerNormalization:
    def test_tensor_scheduler(self):
        batch = make_batch()
        out = make_process(scheduler=MaskAll())(batch)
        assert out.objective_inputs["diffusion_mask"].any()

    def test_wrong_length_schedule_is_rejected(self):
        # A [1] result for B=2 would silently broadcast to every row, applying
        # one row's masking rate to the whole batch. Fail loudly instead.
        class WrongLength:
            def alpha(self, t):
                return torch.zeros(1)

        with pytest.raises(ValueError, match="alpha"):
            make_process(scheduler=WrongLength())(make_batch())

    def test_float_returning_scheduler(self):
        batch = make_batch()
        out = make_process(scheduler=FloatMaskAll())(batch)

        eligible = batch["labels"] != -100
        assert torch.equal(out.objective_inputs["diffusion_mask"], eligible)


# ---------------------------------------------------------------------------
# J. Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("eps", [-0.1, 1.0, 1.5])
    def test_invalid_time_epsilon(self, eps):
        with pytest.raises(ValueError):
            MaskedDiffusionProcess(
                scheduler=MaskAll(), mask_token_id=MASK_ID, time_epsilon=eps
            )

    def test_negative_mask_token_id(self):
        with pytest.raises(ValueError):
            MaskedDiffusionProcess(scheduler=MaskAll(), mask_token_id=-1)


# ---------------------------------------------------------------------------
# K. Dtype / device sanity
# ---------------------------------------------------------------------------


class TestDtypeDevice:
    def test_dtypes(self):
        batch = make_batch()
        out = make_process(scheduler=HalfLinear())(batch)

        assert out.model_inputs["input_ids"].dtype == torch.long
        assert out.objective_inputs["labels"].dtype == torch.long
        assert out.objective_inputs["diffusion_mask"].dtype == torch.bool
        assert out.objective_inputs["timesteps"].is_floating_point()

    def test_outputs_on_input_device(self):
        batch = make_batch()
        device = batch["input_ids"].device
        out = make_process(scheduler=HalfLinear())(batch)

        assert out.model_inputs["input_ids"].device == device
        assert out.objective_inputs["diffusion_mask"].device == device
        assert out.objective_inputs["timesteps"].device == device


# ---------------------------------------------------------------------------
# Legacy parity — deterministic comparison against MaskedDiffusionDataCollator
# ---------------------------------------------------------------------------


class _NoPadTokenizer:
    """Minimal stand-in without .pad → legacy collator uses default_data_collator."""

    mask_token_id = MASK_ID


def _legacy_collate(features, scheduler, completion_only):
    from unturtle.diffusion.collator import MaskedDiffusionDataCollator

    collator = MaskedDiffusionDataCollator(
        tokenizer=_NoPadTokenizer(),
        scheduler=scheduler,
        mask_token_id=MASK_ID,
        completion_only=completion_only,
    )
    return collator(features)


def _features_from_batch(batch):
    keys = list(batch.keys())
    B = batch["input_ids"].shape[0]
    return [{k: batch[k][i].tolist() for k in keys} for i in range(B)]


class TestLegacyCollatorParity:
    @pytest.mark.parametrize("completion_only", [True, False])
    @pytest.mark.parametrize("fake_scheduler", [KeepAll, MaskAll])
    def test_deterministic_parity(self, completion_only, fake_scheduler):
        batch = make_batch()
        legacy = _legacy_collate(
            _features_from_batch(batch), fake_scheduler(), completion_only
        )
        out = make_process(scheduler=fake_scheduler(), completion_only=completion_only)(
            batch
        )

        assert torch.equal(out.model_inputs["input_ids"], legacy["input_ids"])
        assert torch.equal(out.objective_inputs["labels"], legacy["labels"])
        assert torch.equal(
            out.objective_inputs["diffusion_mask"], legacy["diffusion_mask"]
        )

    @pytest.mark.parametrize("fake_scheduler", [KeepAll, MaskAll])
    def test_deterministic_parity_without_labels(self, fake_scheduler):
        batch = make_batch(with_labels=False)
        legacy = _legacy_collate(_features_from_batch(batch), fake_scheduler(), True)
        out = make_process(scheduler=fake_scheduler(), completion_only=True)(batch)

        assert torch.equal(out.model_inputs["input_ids"], legacy["input_ids"])
        assert torch.equal(out.objective_inputs["labels"], legacy["labels"])
        assert torch.equal(
            out.objective_inputs["diffusion_mask"], legacy["diffusion_mask"]
        )
