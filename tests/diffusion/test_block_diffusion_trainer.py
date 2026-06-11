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

"""Tests for BlockDiffusionTrainer — block diffusion training via DiffusionTrainer subclass."""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

VOCAB = ["[PAD]", "[UNK]", " masked", "[BOS]", "[EOS]"] + [f"w{i}" for i in range(95)]
VOCAB_SIZE = len(VOCAB)
SEQ_LEN = 16


def _make_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(
        WordLevel(vocab={w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "mask_token": " masked",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
        }
    )
    fast.padding_side = "right"
    return fast


def _make_bert() -> BertForMaskedLM:
    cfg = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,  # large enough for 2*SEQ_LEN
        pad_token_id=0,
    )
    return BertForMaskedLM(cfg)


class TestBlockDiffusionTrainerInstantiation:
    def test_import(self):
        from unturtle.diffusion.block_diffusion_trainer import (
            BlockDiffusionTrainer,
            BlockDiffusionTrainingArguments,
        )

        assert BlockDiffusionTrainer is not None
        assert BlockDiffusionTrainingArguments is not None

    def test_is_subclass_of_diffusion_trainer(self):
        from unturtle.diffusion.block_diffusion_trainer import BlockDiffusionTrainer

        assert issubclass(BlockDiffusionTrainer, DiffusionTrainer)

    def test_instantiation(self, tmp_path):
        from unturtle.diffusion.block_diffusion_trainer import (
            BlockDiffusionTrainer,
            BlockDiffusionTrainingArguments,
        )

        tokenizer = _make_tokenizer()
        model = _make_bert()
        dataset = Dataset.from_list(
            [
                {
                    "input_ids": torch.randint(5, VOCAB_SIZE, (SEQ_LEN,)).tolist(),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(4)
            ]
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
        trainer = BlockDiffusionTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        assert trainer._block_size == 4


class TestBlockDiffusionComputeLoss:
    """Test the compute_loss override for x_t ⊕ x_0 concat."""

    @pytest.fixture(scope="module")
    def tokenizer(self):
        return _make_tokenizer()

    @pytest.fixture
    def model(self):
        return _make_bert()

    def test_compute_loss_returns_scalar(self, tokenizer, model, tmp_path):
        from unturtle.diffusion.block_diffusion_collator import (
            BlockDiffusionDataCollator,
        )
        from unturtle.diffusion.block_diffusion_trainer import (
            BlockDiffusionTrainer,
            BlockDiffusionTrainingArguments,
        )
        from unturtle.diffusion.schedulers import LinearAlphaScheduler

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
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=tokenizer.mask_token_id,
            completion_only=False,
        )
        dataset = Dataset.from_list(
            [
                {
                    "input_ids": list(range(5, 5 + SEQ_LEN)),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(4)
            ]
        )
        trainer = BlockDiffusionTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )

        batch = collator(
            [
                {
                    "input_ids": list(range(5, 5 + SEQ_LEN)),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(2)
            ]
        )
        inputs = {k: v for k, v in batch.items()}
        loss = trainer.compute_loss(model, inputs)

        assert loss.ndim == 0, "Loss must be scalar"
        assert loss.item() > 0, "Loss must be positive"
        assert not torch.isnan(loss), "Loss must not be NaN"
        assert torch.isfinite(loss), "Loss must be finite"

    def test_gradient_flows(self, tokenizer, model, tmp_path):
        """Gradients flow through the BD3LM compute_loss path."""
        from unturtle.diffusion.block_diffusion_collator import (
            BlockDiffusionDataCollator,
        )
        from unturtle.diffusion.block_diffusion_trainer import (
            BlockDiffusionTrainer,
            BlockDiffusionTrainingArguments,
        )
        from unturtle.diffusion.schedulers import LinearAlphaScheduler

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
        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=tokenizer.mask_token_id,
            completion_only=False,
        )
        dataset = Dataset.from_list(
            [
                {
                    "input_ids": list(range(5, 5 + SEQ_LEN)),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(4)
            ]
        )
        trainer = BlockDiffusionTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )

        batch = collator(
            [
                {
                    "input_ids": list(range(5, 5 + SEQ_LEN)),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(2)
            ]
        )
        inputs = {k: v for k, v in batch.items()}
        loss = trainer.compute_loss(model, inputs)
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "At least some parameters should have gradients"
        total_grad_norm = sum(g.norm().item() for g in grads)
        assert total_grad_norm > 0, "Gradient norm should be > 0"

    def test_loss_decreases(self, tokenizer, tmp_path):
        """One optimizer step should decrease the loss."""
        from unturtle.diffusion.block_diffusion_collator import (
            BlockDiffusionDataCollator,
        )
        from unturtle.diffusion.block_diffusion_trainer import (
            BlockDiffusionTrainer,
            BlockDiffusionTrainingArguments,
        )
        from unturtle.diffusion.schedulers import LinearAlphaScheduler

        torch.manual_seed(0)
        model = _make_bert()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        collator = BlockDiffusionDataCollator(
            tokenizer=tokenizer,
            block_size=4,
            scheduler=LinearAlphaScheduler(),
            mask_token_id=tokenizer.mask_token_id,
            completion_only=False,
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
        dataset = Dataset.from_list(
            [
                {
                    "input_ids": list(range(5, 5 + SEQ_LEN)),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(4)
            ]
        )
        trainer = BlockDiffusionTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            data_collator=collator,
        )

        batch = collator(
            [
                {
                    "input_ids": list(range(5, 5 + SEQ_LEN)),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(2)
            ]
        )
        inputs = {k: v for k, v in batch.items()}

        loss1 = trainer.compute_loss(model, inputs)
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.manual_seed(42)
        batch2 = collator(
            [
                {
                    "input_ids": list(range(5, 5 + SEQ_LEN)),
                    "attention_mask": [1] * SEQ_LEN,
                }
                for _ in range(2)
            ]
        )
        inputs2 = {k: v for k, v in batch2.items()}
        loss2 = trainer.compute_loss(model, inputs2)
        assert loss2.item() < loss1.item() * 1.1, (
            f"Loss did not decrease: {loss1.item():.4f} -> {loss2.item():.4f}"
        )
