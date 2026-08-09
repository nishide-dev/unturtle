"""
Hybrid-attention conversion trains end-to-end (#63).

The conversion recipe existed (#101, #112) but nothing proved the full chain:
convert a real AR model → collate SFT-style batches carrying the prompt
boundary → `DiffusionTrainer.train()` — a real optimizer loop, not a single
`model(**inputs)` (MDLM-DiT taught us those diverge: gradient checkpointing,
column pruning and accelerator moves only bite inside `train()`).

Two acceptance criteria of #63 are pinned here:

- one supported AR backbone converts and trains end-to-end on the existing
  masked-diffusion objective stack, in BOTH topologies (hybrid and uniform
  bidirectional), with no trainer-side model-private branches — the trainer
  ships the batch via ``model(**inputs)`` and the model consumes or ignores
  ``prompt_lengths`` by contract;
- LoRA wrapping does not silently restore causal-only attention.

The missing plumbing this file drives out (TDD): nothing supplied
``prompt_lengths`` at training time.  `prompt_lengths_from_labels` derives the
boundary from the SFT convention already in every batch (first supervised
position), and `HybridPromptCollator` rides it on the collated batch.
"""

from __future__ import annotations

import pytest
import torch


def _tiny_ar(vocab=64, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )


def _tokenizer():
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    raw_tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw_tok.normalizer = normalizers.Lowercase()
    raw_tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="[PAD]",
    )
    tokenizer.add_special_tokens(
        {"unk_token": "[UNK]", "mask_token": "[MASK]", "pad_token": "[PAD]"}
    )
    tokenizer.name_or_path = "local"
    return tokenizer


def _sft_rows(vocab, rows=16, length=16, prompt=5, seed=0):
    """SFT convention: labels are -100 on the prompt, clean ids on the target."""
    generator = torch.Generator().manual_seed(seed)
    dataset = []
    for _ in range(rows):
        ids = torch.randint(3, vocab, (length,), generator=generator).tolist()
        dataset.append(
            {
                "input_ids": ids,
                "labels": [-100] * prompt + ids[prompt:],
                "attention_mask": [1] * length,
            }
        )
    return dataset


class TestPromptLengthsFromLabels:
    def test_the_boundary_is_the_first_supervised_position(self):
        from unturtle.models.conversion.a2d.tiny_a2d import prompt_lengths_from_labels

        labels = torch.tensor(
            [
                [-100, -100, -100, 7, 8, 9],
                [5, 6, 7, 8, 9, 10],
                [-100, -100, -100, -100, -100, 4],
            ]
        )

        lengths = prompt_lengths_from_labels(labels)

        assert lengths.tolist() == [3, 0, 5]
        assert lengths.dtype == torch.long
        assert lengths.device == labels.device

    def test_a_fully_unsupervised_row_is_all_prompt(self):
        """No supervised position → the whole row is observed context.

        Such a row contributes no loss anyway; what matters is that it maps to
        a *valid* boundary (p = L) rather than 0, which would silently flip a
        dead row to fully bidirectional."""
        from unturtle.models.conversion.a2d.tiny_a2d import prompt_lengths_from_labels

        labels = torch.full((2, 6), -100)

        assert prompt_lengths_from_labels(labels).tolist() == [6, 6]

    def test_later_ignore_positions_do_not_move_the_boundary(self):
        """-100 holes inside the target (padding, ignored spans) are a labels
        concern, not a topology concern: the prompt/target split is the first
        supervised position, full stop."""
        from unturtle.models.conversion.a2d.tiny_a2d import prompt_lengths_from_labels

        labels = torch.tensor([[-100, -100, 7, -100, 9, -100]])

        assert prompt_lengths_from_labels(labels).tolist() == [2]

    def test_a_non_2d_input_is_rejected(self):
        from unturtle.models.conversion.a2d.tiny_a2d import prompt_lengths_from_labels

        with pytest.raises(ValueError, match="2-D"):
            prompt_lengths_from_labels(torch.tensor([-100, 3, 4]))


class TestHybridPromptCollator:
    def test_it_adds_the_boundary_and_changes_nothing_else(self):
        """The wrapper is additive: same keys, same tensors, plus
        ``prompt_lengths`` consistent with the *padded* labels (the boundary
        must be computed after padding, or left-padding shifts it)."""
        from unturtle.diffusion import MaskedDiffusionDataCollator
        from unturtle.models.conversion.a2d.tiny_a2d import HybridPromptCollator

        tokenizer = _tokenizer()
        base = MaskedDiffusionDataCollator(
            tokenizer=tokenizer, mask_token_id=1, noise=False
        )
        wrapped = HybridPromptCollator(
            MaskedDiffusionDataCollator(
                tokenizer=tokenizer, mask_token_id=1, noise=False
            )
        )
        rows = _sft_rows(vocab=64, rows=4, prompt=5)

        reference = base(rows)
        batch = wrapped(rows)

        assert batch["prompt_lengths"].tolist() == [5, 5, 5, 5]
        assert set(batch) == set(reference) | {"prompt_lengths"}
        for key in reference:
            assert torch.equal(batch[key], reference[key]), f"{key} changed"

    def test_the_boundary_is_computed_on_padded_labels(self):
        """Ragged rows: the collator pads labels with -100; a boundary taken
        from the raw feature would be wrong for every padded row under
        left-padding.  Right-padding keeps leading runs intact, so here the
        check is simply that ragged batches collate and each row keeps its own
        prompt length."""
        from unturtle.diffusion import MaskedDiffusionDataCollator
        from unturtle.models.conversion.a2d.tiny_a2d import HybridPromptCollator

        wrapped = HybridPromptCollator(
            MaskedDiffusionDataCollator(
                tokenizer=_tokenizer(), mask_token_id=1, noise=False
            )
        )
        short = _sft_rows(vocab=64, rows=1, length=10, prompt=3)[0]
        long = _sft_rows(vocab=64, rows=1, length=16, prompt=7)[0]

        batch = wrapped([short, long])

        assert batch["prompt_lengths"].tolist() == [3, 7]

    def test_a_base_collator_without_labels_is_rejected(self):
        """No labels, no boundary — refusing loudly beats emitting a batch
        whose hybrid flag silently never activates (`prompt_lengths` absent →
        uniform bidirectional, the exact degradation this wrapper exists to
        prevent going unnoticed)."""
        from unturtle.models.conversion.a2d.tiny_a2d import HybridPromptCollator

        wrapped = HybridPromptCollator(
            lambda features: {"input_ids": torch.ones(2, 4, dtype=torch.long)}
        )

        with pytest.raises(ValueError, match="labels"):
            wrapped([{}, {}])


class TestHybridTrainingEndToEnd:
    def _train(self, model, collator, tmp_path, steps=24):
        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

        tokenizer = _tokenizer()
        args = DiffusionTrainingArguments(
            output_dir=str(tmp_path / "out"),
            max_steps=steps,
            per_device_train_batch_size=4,
            learning_rate=5e-3,
            logging_steps=1,
            save_steps=10_000,
            use_cpu=True,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to="none",
            seed=7,
        )
        losses: list[float] = []
        original_log = DiffusionTrainer.log

        def capturing_log(self_inner, logs, start_time=None, **kw):
            if "loss" in logs:
                losses.append(float(logs["loss"]))
            original_log(self_inner, logs, start_time=start_time, **kw)

        DiffusionTrainer.log = capturing_log
        try:
            trainer = DiffusionTrainer(
                model=model,
                args=args,
                train_dataset=_sft_rows(vocab=64, rows=16),
                data_collator=collator,
                processing_class=tokenizer,
            )
            trainer.train()
        finally:
            DiffusionTrainer.log = original_log
        return losses

    def _hybrid_collator(self):
        from unturtle.diffusion import MaskedDiffusionDataCollator
        from unturtle.models.conversion.a2d.tiny_a2d import HybridPromptCollator

        return HybridPromptCollator(
            MaskedDiffusionDataCollator(
                tokenizer=_tokenizer(), mask_token_id=1, completion_only=True
            )
        )

    def test_a_converted_hybrid_model_trains_and_the_loss_decreases(self, tmp_path):
        """The real `train()` loop, not a bare forward (MDLM-DiT lesson)."""
        from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model

        model = convert_ar_model(
            _tiny_ar(), mask_token_id=1, hybrid_attention=True
        ).train()

        losses = self._train(model, self._hybrid_collator(), tmp_path)

        assert len(losses) >= 20
        early = sum(losses[:3]) / 3
        late = sum(losses[-3:]) / 3
        assert late < early, f"loss did not decrease: {early:.3f} -> {late:.3f}"

    def test_the_bidirectional_control_trains_on_the_same_pipeline(self, tmp_path):
        """Matched-arm property for the #63 benchmark: the SAME collator and
        batch (including `prompt_lengths`) drives an unconverted-topology
        model, which ignores the boundary by contract instead of crashing or
        silently changing semantics."""
        from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model

        model = convert_ar_model(
            _tiny_ar(), mask_token_id=1, hybrid_attention=False
        ).train()

        losses = self._train(model, self._hybrid_collator(), tmp_path)

        early = sum(losses[:3]) / 3
        late = sum(losses[-3:]) / 3
        assert late < early, f"loss did not decrease: {early:.3f} -> {late:.3f}"

    def test_prompt_lengths_actually_reach_the_attention_topology(self, tmp_path):
        """The threading proof: identical noised batch, with and without the
        boundary key, must yield DIFFERENT losses on a hybrid model.  If any
        link in collator → trainer → `model(**inputs)` → mask build drops the
        key, both calls see uniform bidirectional attention and this fails —
        the exact silent degradation the fail-safe design would otherwise
        hide."""
        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments
        from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model

        model = convert_ar_model(
            _tiny_ar(), mask_token_id=1, hybrid_attention=True
        ).eval()

        torch.manual_seed(3)
        batch = self._hybrid_collator()(_sft_rows(vocab=64, rows=4))
        assert "diffusion_mask" in batch, "the noising collator must pre-noise"

        args = DiffusionTrainingArguments(
            output_dir=str(tmp_path / "out"),
            use_cpu=True,
            report_to="none",
            remove_unused_columns=False,
        )
        trainer = DiffusionTrainer(
            model=model,
            args=args,
            train_dataset=_sft_rows(vocab=64, rows=4),
            data_collator=self._hybrid_collator(),
            processing_class=_tokenizer(),
        )

        with torch.no_grad():
            with_boundary = trainer.compute_loss(
                model, {k: v.clone() for k, v in batch.items()}
            )
            without_boundary = trainer.compute_loss(
                model,
                {k: v.clone() for k, v in batch.items() if k != "prompt_lengths"},
            )

        assert not torch.isclose(with_boundary, without_boundary), (
            "removing prompt_lengths did not change the loss; the boundary "
            "never reached the attention mask"
        )


class TestLoRADoesNotRestoreCausality:
    """#63 acceptance: LoRA/QLoRA must not silently bring back the causal mask.

    The probe is behavioural (a suffix edit must move prefix logits through
    the PEFT wrapper), because the failure mode is a fast-path patch or
    wrapper forward quietly rebuilding upstream's causal default."""

    def _peft(self, model):
        from unturtle.fast_diffusion_model import FastDiffusionModel

        return FastDiffusionModel.get_peft_model(
            model,
            r=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=4,
            lora_dropout=0,
            use_gradient_checkpointing=False,
        )

    def test_the_wrapped_model_is_still_bidirectional(self):
        from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model

        peft_model = self._peft(convert_ar_model(_tiny_ar(), mask_token_id=1)).eval()

        ids = torch.randint(3, 64, (1, 8))
        edited = ids.clone()
        edited[0, -1] = (edited[0, -1] + 1) % 61 + 3

        with torch.no_grad():
            moved = not torch.allclose(
                peft_model(input_ids=ids).logits[0, 0],
                peft_model(input_ids=edited).logits[0, 0],
            )

        assert moved, "LoRA wrapping restored causal-only attention"

    def test_the_hybrid_topology_survives_the_peft_wrapper(self):
        """`prompt_lengths` must still change the logits through the wrapper —
        PEFT forwards kwargs to the base model, and the eq.-(3) build must be
        reachable there."""
        from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model

        peft_model = self._peft(
            convert_ar_model(_tiny_ar(), mask_token_id=1, hybrid_attention=True)
        ).eval()

        ids = torch.randint(3, 64, (2, 8))
        boundary = torch.tensor([4, 4])

        with torch.no_grad():
            hybrid = peft_model(input_ids=ids, prompt_lengths=boundary).logits
            uniform = peft_model(input_ids=ids).logits

        assert not torch.allclose(hybrid, uniform), (
            "prompt_lengths stopped reaching the mask build under PEFT"
        )

    def test_a_lora_hybrid_model_trains(self, tmp_path):
        from unturtle.models.conversion.a2d.tiny_a2d import convert_ar_model

        peft_model = self._peft(
            convert_ar_model(_tiny_ar(), mask_token_id=1, hybrid_attention=True)
        ).train()

        e2e = TestHybridTrainingEndToEnd()
        losses = e2e._train(peft_model, e2e._hybrid_collator(), tmp_path, steps=12)

        assert losses and all(torch.isfinite(torch.tensor(loss)) for loss in losses), (
            f"non-finite LoRA training losses: {losses}"
        )
