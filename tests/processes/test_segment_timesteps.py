"""
Per-segment timesteps for packed batches (#62 PR3).

The packed collator sampled one `t` per original sample, then collapsed them to
a row mean for `DiffusionTrainer` compatibility.  That mean is why both the
trainer and the evaluator hard-reject packed batches for `timestep` and
`scheduler` weighting: a row mean is the wrong `t` for every sample in the row.

With `segment_ids` the process samples per segment and broadcasts, so
`timesteps` is `[B, L]` and each position carries its own sample's `t`.
"""

import pytest
import torch

MASK_ID = 99


class Linear:
    def alpha(self, t):
        return 1.0 - t


class MaskAll:
    def alpha(self, t):
        return torch.zeros_like(t)


def _process(scheduler=None, **kwargs):
    from unturtle.processes import MaskedDiffusionProcess

    return MaskedDiffusionProcess(
        scheduler=scheduler if scheduler is not None else Linear(),
        mask_token_id=MASK_ID,
        completion_only=False,
        **kwargs,
    )


def _packed_batch():
    """One row holding three packed samples of lengths 2, 3, 1."""
    return {
        "input_ids": torch.tensor([[5, 6, 7, 8, 9, 10]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "segment_ids": torch.tensor([[0, 0, 1, 1, 1, 2]]),
    }


class TestPerSegmentSampling:
    def test_timesteps_are_per_position(self):
        out = _process()(_packed_batch())
        timesteps = out.objective_inputs["timesteps"]

        assert timesteps.shape == (1, 6), (
            "packed timesteps must be [B, L]; a [B] tensor cannot carry a "
            "different t per packed sample"
        )

    def test_each_segment_gets_one_shared_timestep(self):
        out = _process()(_packed_batch())
        t = out.objective_inputs["timesteps"][0]

        # Segments are [0,0], [1,1,1], [2].
        assert t[0] == t[1]
        assert t[2] == t[3] == t[4]

    def test_segments_get_different_timesteps(self):
        """The whole point: not one t for the row."""
        batch = {
            "input_ids": torch.arange(1, 41).reshape(1, 40),
            "attention_mask": torch.ones(1, 40, dtype=torch.long),
            "segment_ids": torch.arange(40).reshape(1, 40) // 4,
        }
        out = _process()(batch)
        t = out.objective_inputs["timesteps"][0]

        per_segment = {
            int(s): float(t[i]) for i, s in enumerate(batch["segment_ids"][0])
        }
        assert len(set(per_segment.values())) > 1, (
            "every segment drew the same t; sampling is not per-segment"
        )

    def test_masking_rate_follows_each_segments_own_timestep(self):
        """A segment's corruption must use its own t, not the row's mean."""

        class PerSegment:
            """alpha = 1 for even t-index, 0 for odd — separable outcomes."""

            def alpha(self, t):
                return (t < 0.5).to(t.dtype)

        batch = {
            "input_ids": torch.arange(1, 9).reshape(1, 8),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "segment_ids": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        }
        # Force one segment below 0.5 and one above by pinning the draw.
        process = _process(scheduler=PerSegment(), time_epsilon=0.0)
        generator = torch.Generator().manual_seed(0)
        out = process(batch, generator=generator)

        t = out.objective_inputs["timesteps"][0]
        mask = out.objective_inputs["diffusion_mask"][0]
        for segment in (0, 1):
            span = batch["segment_ids"][0] == segment
            # alpha=1 -> p_mask=0 -> nothing masked; alpha=0 -> all masked.
            expected = bool(t[span][0] >= 0.5)
            assert bool(mask[span].all()) == expected
            assert bool(mask[span].any()) == expected


class TestUnpackedIsUnchanged:
    def test_without_segment_ids_timesteps_stay_per_row(self):
        """#62 PR3 must not change the unpacked contract."""
        batch = {
            "input_ids": torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]]),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        }
        out = _process()(batch)

        assert out.objective_inputs["timesteps"].shape == (2,)

    def test_segment_ids_is_not_forwarded_to_the_model(self):
        """It is packing topology, not a model input."""
        out = _process()(_packed_batch())

        assert "segment_ids" not in out.model_inputs


class TestSegmentIdsValidation:
    def test_shape_mismatch_is_rejected(self):
        batch = _packed_batch()
        batch["segment_ids"] = torch.zeros(1, 3, dtype=torch.long)

        with pytest.raises(ValueError, match="segment_ids"):
            _process()(batch)


class TestWeightingBecomesCorrect:
    def test_timestep_weights_differ_per_segment(self):
        """`1/t` weighting was wrong on packed rows; now it is per-sample."""
        out = _process()(_packed_batch())
        t = out.objective_inputs["timesteps"]

        weights = 1.0 / t.clamp_min(1e-6)
        assert weights.shape == (1, 6)
        # Segment 0 and segment 2 drew different t, so different weights.
        assert weights[0, 0] != weights[0, 5]

    def test_a_row_mean_would_have_lost_this(self):
        """Pins why the change matters, not just that shapes changed."""
        out = _process()(_packed_batch())
        t = out.objective_inputs["timesteps"][0]

        row_mean = t.mean()
        assert not torch.allclose(t, row_mean.expand_as(t)), (
            "segments drew indistinguishable timesteps; the fixture cannot "
            "show what the mean discarded"
        )


class TestPackedCollatorIntegration:
    """The collator emits topology; the process owns corruption (#62 PR3)."""

    class _Tok:
        mask_token_id = MASK_ID
        pad_token_id = 0
        eos_token_id = 2

    def _collator(self, noise):
        from unturtle.diffusion.packed_collator import (
            PackedMaskedDiffusionDataCollator,
        )

        return PackedMaskedDiffusionDataCollator(
            tokenizer=self._Tok(),
            max_seq_length=16,
            mask_token_id=MASK_ID,
            completion_only=False,
            noise=noise,
        )

    def _features(self):
        return [{"input_ids": [5, 6, 7]}, {"input_ids": [8, 9]}]

    def test_clean_collator_emits_topology_not_supervision(self):
        batch = self._collator(noise=False)(self._features())

        assert "segment_ids" in batch
        # Both supervision keys or neither: one alone is the half-noised shape
        # `classify_batch` rejects.
        assert "diffusion_mask" not in batch
        assert "timesteps" not in batch

    def test_noising_collator_is_unchanged(self):
        batch = self._collator(noise=True)(self._features())

        assert "diffusion_mask" in batch
        assert "timesteps" in batch
        assert batch["timesteps"].shape == (1,), "the legacy row mean is [B]"

    def test_segment_ids_identify_the_original_samples(self):
        batch = self._collator(noise=False)(self._features())
        segments = batch["segment_ids"][0]

        # Two samples of length 3 and 2, then padding.
        assert segments[:3].tolist() == [0, 0, 0]
        assert segments[3:5].tolist() == [1, 1]
        assert (segments[5:] == -1).all(), "padding must own no sample"

    def test_clean_packed_batch_flows_through_the_process(self):
        batch = self._collator(noise=False)(self._features())
        out = _process()(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "segment_ids": batch["segment_ids"].clamp_min(0),
            }
        )

        t = out.objective_inputs["timesteps"]
        assert t.shape == batch["input_ids"].shape
        # Each original sample shares one t.
        assert t[0, 0] == t[0, 1] == t[0, 2]
        assert t[0, 3] == t[0, 4]


class TestPackedWeightingGuards:
    """`timestep`/`scheduler` were barred on packed input; only the row mean was the reason."""

    class _Tok:
        mask_token_id = MASK_ID
        pad_token_id = 0
        eos_token_id = 2

    def _packed(self, noise):
        from unturtle.diffusion.packed_collator import (
            PackedMaskedDiffusionDataCollator,
        )

        return PackedMaskedDiffusionDataCollator(
            tokenizer=self._Tok(), max_seq_length=16, mask_token_id=MASK_ID, noise=noise
        )

    def _evaluator(self, collator, weighting):
        import torch.nn as nn

        from unturtle.eval import MaskedDiffusionEvaluator

        class _Cfg:
            model_type = "tiny-a2d-llama"
            mask_token_id = MASK_ID

        model = nn.Linear(2, 2)
        model.config = _Cfg()
        return MaskedDiffusionEvaluator(
            model=model,
            tokenizer=self._Tok(),
            data_collator=collator,
            loss_weight_type=weighting,
        )

    @pytest.mark.parametrize("weighting", ["timestep", "scheduler"])
    def test_noising_packed_still_rejected(self, weighting):
        with pytest.raises(ValueError, match="noising"):
            self._evaluator(self._packed(noise=True), weighting)

    @pytest.mark.parametrize("weighting", ["timestep", "scheduler"])
    def test_clean_packed_is_now_allowed(self, weighting):
        """The process supplies a per-segment [B, L] t, so it is well-defined."""
        assert self._evaluator(self._packed(noise=False), weighting) is not None
