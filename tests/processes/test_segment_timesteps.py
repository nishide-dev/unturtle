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


def _real_tokenizer():
    """A real fast tokenizer with a mask token, for trainer-level tests."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.add_special_tokens(
        {
            "unk_token": "[UNK]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "eos_token": "[EOS]",
        }
    )
    tokenizer.name_or_path = "local"
    return tokenizer


def _real_model(vocab_size=128):
    from transformers import BertConfig, BertForMaskedLM

    return BertForMaskedLM(
        BertConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )


def _real_packed_collator(tokenizer, *, noise):
    from unturtle.diffusion.packed_collator import PackedMaskedDiffusionDataCollator

    return PackedMaskedDiffusionDataCollator(
        tokenizer=tokenizer,
        max_seq_length=16,
        mask_token_id=tokenizer.mask_token_id,
        completion_only=False,
        noise=noise,
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
        """Raw collator output, padding sentinel and all.

        An earlier version of this test applied `.clamp_min(0)` — the exact
        fix the production path was missing — so it passed while every real
        packed batch raised `index -1 is out of bounds`.
        """
        batch = self._collator(noise=False)(self._features())
        assert (batch["segment_ids"] == -1).any(), (
            "fixture has no padding, so it cannot exercise the sentinel"
        )
        out = _process()(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "segment_ids": batch["segment_ids"],
            }
        )

        t = out.objective_inputs["timesteps"]
        assert t.shape == batch["input_ids"].shape
        # Each original sample shares one t.
        assert t[0, 0] == t[0, 1] == t[0, 2]
        assert t[0, 3] == t[0, 4]
        # Padding owns no sample, so it gets no meaningful timestep.
        unowned = batch["segment_ids"] < 0
        assert (t[unowned] == 0).all()
        # And nothing unowned is ever masked.
        assert not out.objective_inputs["diffusion_mask"][unowned].any()


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

    @pytest.mark.parametrize("weighting", ["timestep", "scheduler"])
    def test_trainer_build_evaluator_agrees_with_the_evaluator(
        self, weighting, tmp_path
    ):
        """`build_diffusion_evaluator` must apply the same narrowed guard.

        It carried its own copy of the pre-#62 guard, rejecting *any* packed
        collator regardless of `noise`.  A clean packed setup therefore trained
        without complaint and then raised when asked for its evaluator.
        """
        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

        tokenizer = _real_tokenizer()
        collator = _real_packed_collator(tokenizer, noise=False)
        args = DiffusionTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            max_steps=1,
            use_cpu=True,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
            loss_weight_type=weighting,
        )
        trainer = DiffusionTrainer(
            model=_real_model(),
            args=args,
            train_dataset=[{"input_ids": [5, 6, 7]}],
            processing_class=tokenizer,
            data_collator=collator,
        )

        assert trainer.build_diffusion_evaluator() is not None

    @pytest.mark.parametrize("weighting", ["timestep", "scheduler"])
    def test_trainer_build_evaluator_still_rejects_noising_packed(
        self, weighting, tmp_path
    ):
        """Narrowing must not open the case that is genuinely ill-defined."""
        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

        tokenizer = _real_tokenizer()
        args = DiffusionTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            max_steps=1,
            use_cpu=True,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
            loss_weight_type=weighting,
        )
        # The trainer's own __init__ guard already rejects this pairing, which
        # is the behaviour under test: the two guards must agree.
        with pytest.raises(ValueError, match="noising"):
            trainer = DiffusionTrainer(
                model=_real_model(),
                args=args,
                train_dataset=[{"input_ids": [5, 6, 7]}],
                processing_class=tokenizer,
                data_collator=_real_packed_collator(tokenizer, noise=True),
            )
            trainer.build_diffusion_evaluator()


class TestCleanPackedActuallyRuns:
    """Constructing the evaluator is not evidence the path works.

    The guards were relaxed for clean-packed + timestep/scheduler weighting,
    but nothing ran behind them — and behind them the gather was raising
    `index -1 is out of bounds` on every real packed batch.
    """

    def _tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        raw = Tokenizer(models.BPE(unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw,
            unk_token="[UNK]",
            mask_token="[MASK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
        )
        tokenizer.add_special_tokens(
            {
                "unk_token": "[UNK]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
                "eos_token": "[EOS]",
            }
        )
        tokenizer.name_or_path = "local"
        return tokenizer

    def _model(self, vocab_size=128):
        from transformers import BertConfig, BertForMaskedLM

        return BertForMaskedLM(
            BertConfig(
                vocab_size=vocab_size,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                max_position_embeddings=64,
            )
        )

    @pytest.mark.parametrize("weighting", ["uniform", "timestep", "scheduler", "cart"])
    def test_compute_loss_on_a_clean_packed_batch(self, weighting, tmp_path):
        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments
        from unturtle.diffusion.packed_collator import (
            PackedMaskedDiffusionDataCollator,
        )

        tokenizer = self._tokenizer()
        model = self._model()
        collator = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            mask_token_id=tokenizer.mask_token_id,
            completion_only=False,
            noise=False,
        )
        args = DiffusionTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            max_steps=1,
            use_cpu=True,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
            loss_weight_type=weighting,
        )
        trainer = DiffusionTrainer(
            model=model,
            args=args,
            train_dataset=[{"input_ids": [5, 6, 7]}],
            processing_class=tokenizer,
            data_collator=collator,
        )

        batch = collator([{"input_ids": [5, 6, 7]}, {"input_ids": [8, 9]}])
        loss = trainer.compute_loss(model, dict(batch))

        assert torch.isfinite(loss), f"{weighting} produced {loss}"
        assert loss.item() >= 0.0

    @pytest.mark.parametrize("weighting", ["timestep", "scheduler"])
    def test_per_segment_timesteps_actually_reach_the_loss(self, weighting, tmp_path):
        """The packed row's per-segment `t` must survive all the way to the loss.

        `test_compute_loss_on_a_clean_packed_batch` only asserts the loss is
        finite, which is not enough: collapsing `(B, L)` timesteps back to a
        per-row mean — the exact pre-#62 bug #62 was filed to remove — still
        produces a perfectly finite loss.  Mutation-verified: mean-reducing the
        weights in `DiffusionTrainer._build_loss_weights` passes the entire
        1009-test suite.  Only `uniform` and `cart` are excluded here, because
        neither reads `timesteps` at all.

        Feeds one fixed batch through `compute_loss` twice, changing *only* the
        timesteps: once with genuine per-segment values, once with each row's
        mean.  Any implementation that reduces `(B, L)` to a row summary makes
        the two identical.
        """
        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments
        from unturtle.diffusion.packed_collator import (
            PackedMaskedDiffusionDataCollator,
        )
        from unturtle.kernels.masked_diffusion_loss import (
            fast_masked_diffusion_loss,
        )

        tokenizer = self._tokenizer()
        model = self._model()
        collator = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=16,
            mask_token_id=tokenizer.mask_token_id,
            completion_only=False,
            noise=False,
        )
        args = DiffusionTrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            max_steps=1,
            use_cpu=True,
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
            loss_weight_type=weighting,
        )
        trainer = DiffusionTrainer(
            model=model,
            args=args,
            train_dataset=[{"input_ids": [5, 6, 7]}],
            processing_class=tokenizer,
            data_collator=collator,
        )

        clean = collator([{"input_ids": [5, 6, 7]}, {"input_ids": [8, 9]}])
        noised = trainer._apply_forward_process(dict(clean))
        per_segment = noised["timesteps"]

        assert per_segment.dim() == 2, (
            f"expected [B, L] per-segment timesteps, got {tuple(per_segment.shape)}"
        )

        # Segments must carry genuinely different t, or collapsing to the row
        # mean would be a no-op and the assertion below could not fail.
        owned = per_segment > 0
        distinct = torch.unique(per_segment[owned])
        assert distinct.numel() > 1, (
            "packed row carries a single timestep; this batch cannot "
            f"distinguish per-segment from row-mean weighting (t={distinct})"
        )

        model.eval()  # dropout would perturb the comparison with its own noise
        with torch.no_grad():
            actual = trainer.compute_loss(model, {**noised, "timesteps": per_segment})

            # Reference: the same loss computed from the per-position weights
            # this weighting is *defined* to use.  Asserting merely that the
            # per-segment loss differs from the row-mean loss is not enough — a
            # trainer that mangles the weights also produces a different number,
            # and "different" would accept it.  Verified: an implementation that
            # mean-reduces `(B, L)` weights satisfies the difference check while
            # returning a loss five orders of magnitude off.
            labels = noised["labels"]
            diffusion_mask = noised["diffusion_mask"]
            forward_inputs = {
                k: v
                for k, v in noised.items()
                if k not in ("labels", "diffusion_mask", "timesteps")
            }
            logits = model(**forward_inputs).logits
            if weighting == "timestep":
                expected_w = 1.0 / per_segment.clamp_min(1e-6)
            else:
                expected_w = trainer._alpha_scheduler.weight(per_segment)
            expected = fast_masked_diffusion_loss(
                logits=logits,
                labels=labels,
                diffusion_mask=diffusion_mask,
                loss_weights=expected_w,
            )

        assert expected_w.shape == per_segment.shape, (
            f"{weighting}: weights collapsed from {tuple(per_segment.shape)} to "
            f"{tuple(expected_w.shape)} before reaching the loss"
        )
        assert torch.allclose(actual, expected, atol=1e-5), (
            f"{weighting}: trainer loss {actual.item():.6f} does not match the "
            f"per-segment reference {expected.item():.6f}; the [B, L] timesteps "
            "are being altered between the process and the loss"
        )
