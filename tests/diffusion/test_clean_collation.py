"""
Tests for clean collation + device-side process integration (#62 PR2).

PR1 (#70) added `unturtle.processes`; this covers wiring it into the training
and evaluation paths:

  - `MaskedDiffusionDataCollator(noise=False)` pads and builds supervision but
    performs no corruption.
  - `DiffusionTrainer` applies the process inside `compute_loss`, and skips it
    when the batch already carries noised keys (packed/legacy collators).
  - `MaskedDiffusionEvaluator` applies the process after device transfer.

Run with:
    pytest tests/diffusion/test_clean_collation.py -v
"""

import pytest
import torch

from unturtle.diffusion import LinearAlphaScheduler, MaskedDiffusionDataCollator

MASK_ID = 103
SEQ_LEN = 8


class _Tokenizer:
    """Minimal tokenizer stub without `.pad` → default_data_collator path."""

    mask_token_id = MASK_ID
    eos_token_id = 102
    pad_token_id = 0
    # Non-empty so the Trainer does not try to fetch processor_config.json.
    name_or_path = "local"


def _real_tokenizer():
    """A genuine PreTrainedTokenizerFast built offline (no hub access).

    SFTTrainer type-checks `processing_class`, and the tokenizer-padding
    branch of the collator needs a real `.pad`, so the stub is not enough.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="[PAD]",
    )
    tokenizer.add_special_tokens(
        {"unk_token": "[UNK]", "mask_token": "[MASK]", "pad_token": "[PAD]"}
    )
    return tokenizer


def _make_collator(**kwargs):
    return MaskedDiffusionDataCollator(
        tokenizer=_Tokenizer(),
        scheduler=LinearAlphaScheduler(),
        mask_token_id=MASK_ID,
        **kwargs,
    )


def _samples(n=4, prompt_len=3):
    out = []
    for _ in range(n):
        ids = list(range(5, 5 + SEQ_LEN))
        labels = [-100] * prompt_len + ids[prompt_len:]
        out.append(
            {"input_ids": ids, "labels": labels, "attention_mask": [1] * SEQ_LEN}
        )
    return out


# ---------------------------------------------------------------------------
# Clean collation contract
# ---------------------------------------------------------------------------


class TestCleanCollation:
    def test_emits_no_noised_keys(self):
        batch = _make_collator(noise=False)(_samples())

        assert "diffusion_mask" not in batch
        assert "timesteps" not in batch

    def test_input_ids_are_untouched(self):
        samples = _samples()
        batch = _make_collator(noise=False)(samples)

        expected = torch.tensor([s["input_ids"] for s in samples], dtype=torch.long)
        assert torch.equal(batch["input_ids"], expected)
        assert (batch["input_ids"] != MASK_ID).all()

    def test_supervision_is_still_built(self):
        samples = _samples()
        batch = _make_collator(noise=False)(samples)

        expected = torch.tensor([s["labels"] for s in samples], dtype=torch.long)
        assert torch.equal(batch["labels"], expected)
        assert batch["attention_mask"].shape == (len(samples), SEQ_LEN)

    def test_padding_still_happens(self):
        # Ragged input needs the tokenizer-padding branch, so this case uses a
        # real tokenizer rather than the `.pad`-less stub.
        tokenizer = _real_tokenizer()

        collator = MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=tokenizer.mask_token_id,
            noise=False,
        )
        batch = collator(
            [
                {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]},
                {"input_ids": [8, 9, 10, 11, 12], "attention_mask": [1] * 5},
            ]
        )
        assert batch["input_ids"].shape == (2, 5)
        assert batch["attention_mask"].shape == (2, 5)
        assert "diffusion_mask" not in batch

    def test_noising_remains_the_default(self):
        # PR2 must not change behavior for callers that never opt in — the
        # packed collator and every existing consumer rely on this.
        batch = _make_collator()(_samples())

        assert "diffusion_mask" in batch
        assert "timesteps" in batch

    def test_clean_batch_feeds_the_process_to_equivalent_output(self):
        """The split must be lossless: clean collation + process == old collator."""
        from unturtle.processes import MaskedDiffusionProcess

        class MaskAll:
            def alpha(self, t):
                return torch.zeros_like(t)

        samples = _samples()
        legacy = MaskedDiffusionDataCollator(
            tokenizer=_Tokenizer(),
            scheduler=MaskAll(),
            mask_token_id=MASK_ID,
        )(samples)

        clean = _make_collator(noise=False)(samples)
        out = MaskedDiffusionProcess(scheduler=MaskAll(), mask_token_id=MASK_ID)(clean)

        assert torch.equal(out.model_inputs["input_ids"], legacy["input_ids"])
        assert torch.equal(out.objective_inputs["labels"], legacy["labels"])
        assert torch.equal(
            out.objective_inputs["diffusion_mask"], legacy["diffusion_mask"]
        )


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


def _tiny_model(vocab_size=128, hidden=16, seq_len=SEQ_LEN):
    """A minimal bidirectional LM good enough to exercise compute_loss."""
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=hidden * 2,
        max_position_embeddings=max(seq_len * 4, 64),
    )
    return BertForMaskedLM(config)


def _make_trainer(tmp_path, **arg_overrides):
    from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

    model = _tiny_model()
    tokenizer = _real_tokenizer()
    # Non-empty so the Trainer does not fetch processor_config.json from the hub.
    tokenizer.name_or_path = "local"
    args = DiffusionTrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=1,
        use_cpu=True,
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to=[],
        **arg_overrides,
    )
    trainer = DiffusionTrainer(
        model=model,
        args=args,
        train_dataset=_samples(n=4),
        processing_class=tokenizer,
    )
    return trainer, model, tokenizer


class TestTrainerProcessIntegration:
    def test_default_collator_is_clean(self, tmp_path):
        trainer, _, _ = _make_trainer(tmp_path)

        assert trainer.data_collator.noise is False, (
            "DiffusionTrainer must inject a clean collator; the process now owns noising"
        )

    def test_trainer_exposes_a_process(self, tmp_path):
        from unturtle.processes import MaskedDiffusionProcess

        trainer, _, tokenizer = _make_trainer(tmp_path)

        assert isinstance(trainer.forward_process, MaskedDiffusionProcess)
        assert trainer.forward_process.mask_token_id == tokenizer.mask_token_id

    def test_compute_loss_on_a_clean_batch(self, tmp_path):
        trainer, model, _ = _make_trainer(tmp_path)
        batch = trainer.data_collator(_samples())

        loss = trainer.compute_loss(model, dict(batch))

        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_compute_loss_does_not_renoise_an_already_noised_batch(self, tmp_path):
        """Packed/legacy collators still noise; the trainer must not do it twice."""
        trainer, model, _ = _make_trainer(tmp_path)
        noised = _make_collator()(_samples())

        before = noised["input_ids"].clone()
        mask_before = noised["diffusion_mask"].clone()
        inputs = dict(noised)
        loss = trainer.compute_loss(model, inputs)

        assert torch.isfinite(loss)
        # The pre-noised ids must have been used as-is, not corrupted again.
        assert torch.equal(noised["input_ids"], before)
        assert torch.equal(noised["diffusion_mask"], mask_before)

    def test_process_honors_completion_only_from_args(self, tmp_path):
        trainer, _, _ = _make_trainer(tmp_path, completion_only=False)

        assert trainer.forward_process.completion_only is False

    def test_process_honors_time_epsilon_and_scheduler_from_args(self, tmp_path):
        from unturtle.diffusion.schedulers import CosineAlphaScheduler

        trainer, _, _ = _make_trainer(
            tmp_path, alpha_scheduler="cosine", time_epsilon=0.05
        )

        assert trainer.forward_process.time_epsilon == 0.05
        assert isinstance(trainer.forward_process.scheduler, CosineAlphaScheduler)

    def test_objective_matches_the_legacy_noising_path(self, tmp_path):
        """#62's core claim: the process reproduces the collator's objective.

        Under a deterministic all-mask schedule both paths must corrupt the
        same positions and yield the same loss — otherwise the refactor
        silently changed what the model trains on.
        """
        from unturtle.processes import MaskedDiffusionProcess

        class MaskAll:
            def alpha(self, t):
                return torch.zeros_like(t)

        trainer, model, tokenizer = _make_trainer(tmp_path)
        mask_id = tokenizer.mask_token_id
        samples = _samples()

        # eval() disables dropout: otherwise the two forward passes differ by
        # their dropout masks and the comparison measures noise, not the
        # objective.
        model.eval()

        legacy_batch = MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            scheduler=MaskAll(),
            mask_token_id=mask_id,
        )(samples)
        legacy_loss = trainer.compute_loss(model, dict(legacy_batch))

        trainer.forward_process = MaskedDiffusionProcess(
            scheduler=MaskAll(), mask_token_id=mask_id
        )
        clean_loss = trainer.compute_loss(model, dict(trainer.data_collator(samples)))

        assert torch.allclose(legacy_loss, clean_loss), (
            f"objective drifted: legacy={legacy_loss.item()} clean={clean_loss.item()}"
        )

    def test_fixed_seed_reproduces_the_loss(self, tmp_path):
        """#62 requires fixed-seed reproducibility, not CPU-RNG bit parity."""
        trainer, model, _ = _make_trainer(tmp_path)
        batch = trainer.data_collator(_samples())

        torch.manual_seed(1234)
        loss1 = trainer.compute_loss(model, dict(batch))
        torch.manual_seed(1234)
        loss2 = trainer.compute_loss(model, dict(batch))

        assert torch.equal(loss1, loss2)


# ---------------------------------------------------------------------------
# BD3LM integration — x_0 is now exact rather than reconstructed
# ---------------------------------------------------------------------------


class TestBlockDiffusionProcessIntegration:
    def _trainer(self, tmp_path, tokenizer, model, collator=None):
        from unturtle.diffusion.block_diffusion_trainer import (
            BlockDiffusionTrainer,
            BlockDiffusionTrainingArguments,
        )

        args = BlockDiffusionTrainingArguments(
            output_dir=str(tmp_path / "bd3lm"),
            per_device_train_batch_size=2,
            remove_unused_columns=False,
            report_to="none",
            use_cpu=True,
            bf16=False,
            fp16=False,
            max_steps=1,
            block_size=4,
            completion_only=False,
        )
        dataset = [
            {"input_ids": list(range(5, 5 + SEQ_LEN)), "attention_mask": [1] * SEQ_LEN}
            for _ in range(4)
        ]
        kwargs = {"data_collator": collator} if collator is not None else {}
        return BlockDiffusionTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            **kwargs,
        )

    def test_compute_loss_on_a_clean_batch(self, tmp_path):
        tokenizer = _real_tokenizer()
        tokenizer.name_or_path = "local"
        model = _tiny_model()
        trainer = self._trainer(tmp_path, tokenizer, model)

        batch = trainer.data_collator(
            [
                {
                    "input_ids": list(range(5, 5 + SEQ_LEN)),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(2)
            ]
        )
        loss = trainer.compute_loss(model, dict(batch))

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("completion_only", [True, False])
    def test_x0_half_holds_the_true_clean_ids(self, tmp_path, completion_only):
        """The x_0 half of the [x_t, x_0] concat must be the real clean ids.

        Fully masking x_t makes this observable: if x_0 were derived from x_t
        (rather than held directly) the second half would contain mask tokens.
        Both supervision modes are checked because they build labels
        differently.
        """
        from unturtle.processes import MaskedDiffusionProcess

        class MaskAll:
            def alpha(self, t):
                return torch.zeros_like(t)

        tokenizer = _real_tokenizer()
        tokenizer.name_or_path = "local"
        model = _tiny_model()
        trainer = self._trainer(tmp_path, tokenizer, model)
        trainer.forward_process = MaskedDiffusionProcess(
            scheduler=MaskAll(),
            mask_token_id=tokenizer.mask_token_id,
            completion_only=completion_only,
        )

        prompt_len = 4
        clean_ids = torch.tensor([list(range(5, 5 + SEQ_LEN))], dtype=torch.long)
        labels = clean_ids.clone()
        labels[:, :prompt_len] = -100

        captured = {}
        original_forward = model.forward

        def capture(*args, **kwargs):
            captured["input_ids"] = kwargs.get("input_ids")
            return original_forward(*args, **kwargs)

        model.forward = capture
        try:
            trainer.compute_loss(
                model,
                {
                    "input_ids": clean_ids.clone(),
                    "labels": labels.clone(),
                    "attention_mask": torch.ones_like(clean_ids),
                },
            )
        finally:
            model.forward = original_forward

        concat = captured["input_ids"]
        L = clean_ids.shape[1]
        assert torch.equal(concat[:, L:], clean_ids), (
            f"x_0 half must hold the exact clean ids: got "
            f"{concat[:, L:].tolist()} expected {clean_ids.tolist()}"
        )
        # Everything eligible under this mode was masked in the x_t half.
        maskable = slice(prompt_len, L) if completion_only else slice(0, L)
        assert (concat[:, maskable] == tokenizer.mask_token_id).all()


# ---------------------------------------------------------------------------
# Evaluator integration — the silent-degradation path (CLI eval.py:284)
# ---------------------------------------------------------------------------


class TestEvaluatorProcessIntegration:
    def _evaluator(self, **kwargs):
        from unturtle.eval import MaskedDiffusionEvaluator

        return MaskedDiffusionEvaluator(
            model=_tiny_model(),
            tokenizer=_Tokenizer(),
            **kwargs,
        )

    def test_default_collator_is_clean(self):
        evaluator = self._evaluator()

        assert evaluator.data_collator.noise is False

    def test_evaluate_runs_on_clean_collation(self):
        evaluator = self._evaluator()
        dataset = _samples(n=4)

        metrics = evaluator.evaluate(dataset, batch_size=2)

        assert metrics, "evaluator returned no metrics"
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())

    def test_evaluate_still_works_with_a_noising_collator(self):
        """An explicitly-passed legacy collator must keep working (CLI passes one)."""
        evaluator = self._evaluator(data_collator=_make_collator())

        metrics = evaluator.evaluate(_samples(n=4), batch_size=2)

        assert metrics
