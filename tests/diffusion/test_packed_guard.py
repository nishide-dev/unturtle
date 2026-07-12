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

"""Tests for the DiffusionTrainer unpatched-model packed-metadata guard (#57).

An unpatched model silently ignores ``block_attention_mask`` /
``packed_seq_lengths`` (they ride through ``**kwargs``), producing
cross-sample attention while the loss still decreases.  The trainer must
warn exactly once on the first packed batch when the model does not appear
to consume the metadata — and stay silent when the FastDiffusionModel
fast-forward patch (or a natively packed-aware forward) is detected.
"""

from __future__ import annotations

import logging
import types

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast

from unturtle.diffusion import (
    DiffusionTrainer,
    DiffusionTrainingArguments,
    PackedMaskedDiffusionDataCollator,
)
from unturtle.diffusion.trainer import _model_consumes_packed_metadata

TRAINER_LOGGER = "unturtle.diffusion.trainer"

VOCAB = ["[PAD]", "[UNK]", "[MASK]", "[BOS]", "[EOS]"] + [f"w{i}" for i in range(95)]
VOCAB_SIZE = len(VOCAB)
MAX_SEQ_LEN = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(
        WordLevel(vocab={w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.add_special_tokens(
        {"pad_token": "[PAD]", "unk_token": "[UNK]", "mask_token": "[MASK]"}
    )
    return fast


def _make_bert() -> BertForMaskedLM:
    cfg = BertConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=MAX_SEQ_LEN + 4,
        pad_token_id=0,
    )
    return BertForMaskedLM(cfg)


class _KwargsSinkLM(torch.nn.Module):
    """Tiny LM whose forward silently swallows packed metadata via **kwargs.

    This is exactly the failure mode of an unpatched TinyA2D: the packed
    kwargs are accepted without error and never reach any attention module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inner = _make_bert()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.inner(input_ids=input_ids, attention_mask=attention_mask)


def _mark_as_unturtle_patched(model: torch.nn.Module) -> None:
    """Install an instance-level forward like FastDiffusionModel patching does.

    FastDiffusionModel installs its fast forwards via ``types.MethodType``;
    only the packed-aware TinyA2D fast forward carries the explicit
    ``_consumes_packed_metadata`` marker the guard keys on. The dummy
    submodule is never invoked during the actual forward pass.
    """

    def _fake_fast_forward(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("marker forward must never be called")

    _fake_fast_forward._consumes_packed_metadata = True
    marker = torch.nn.Identity()
    marker.forward = types.MethodType(_fake_fast_forward, marker)
    model._unturtle_marker = marker


def _mark_as_non_packed_unturtle_patched(model: torch.nn.Module) -> None:
    """Instance-level unturtle fast forward WITHOUT the packed marker
    (Dream / ModernBERT / LLaDA patching) — must NOT count as packed-aware."""

    def _fake_fast_forward(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("marker forward must never be called")

    _fake_fast_forward.__module__ = "unturtle.models.backbones.dream.modeling_dream"
    marker = torch.nn.Identity()
    marker.forward = types.MethodType(_fake_fast_forward, marker)
    model._non_packed_marker = marker


def _make_trainer(tmp_path, model, tokenizer):
    """Build a DiffusionTrainer.

    ``model`` should be an HF PreTrainedModel (trainer init requirement); the
    guard itself is exercised through ``compute_loss(sink_model, ...)`` which
    checks the model passed to it.
    """
    collator = PackedMaskedDiffusionDataCollator(
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        completion_only=False,
    )
    args = DiffusionTrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        remove_unused_columns=False,
        report_to="none",
        use_cpu=True,
        bf16=False,
        fp16=False,
        max_steps=1,
        completion_only=False,
        loss_weight_type="uniform",
    )
    dataset = Dataset.from_list(
        [{"input_ids": torch.randint(5, VOCAB_SIZE, (8,)).tolist()} for _ in range(4)]
    )
    return DiffusionTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )


def _packed_batch(trainer) -> dict:
    torch.manual_seed(0)
    samples = [
        {"input_ids": torch.randint(5, VOCAB_SIZE, (6,)).tolist()} for _ in range(4)
    ]
    return dict(trainer.data_collator(samples))


def _guard_warnings(caplog) -> list[logging.LogRecord]:
    return [
        rec
        for rec in caplog.records
        if rec.name == TRAINER_LOGGER
        and rec.levelno == logging.WARNING
        and "packed-attention metadata" in rec.getMessage()
    ]


# ---------------------------------------------------------------------------
# _model_consumes_packed_metadata unit tests
# ---------------------------------------------------------------------------


class TestModelConsumesPackedMetadata:
    def test_plain_model_does_not_consume(self):
        assert not _model_consumes_packed_metadata(_make_bert())
        assert not _model_consumes_packed_metadata(_KwargsSinkLM())

    def test_unturtle_instance_forward_counts_as_patched(self):
        model = _KwargsSinkLM()
        _mark_as_unturtle_patched(model)
        assert _model_consumes_packed_metadata(model)

    def test_unturtle_non_packed_instance_forward_does_not_count(self):
        """Dream/ModernBERT/LLaDA fast forwards never read packed metadata —
        being patched from unturtle code alone is not a signal (review fix)."""
        model = _make_bert()
        _mark_as_non_packed_unturtle_patched(model)
        assert not _model_consumes_packed_metadata(model)

    def test_non_unturtle_instance_forward_does_not_count(self):
        model = _KwargsSinkLM()

        def _other_forward(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError

        # __module__ is this test module, not an unturtle one.
        marker = torch.nn.Identity()
        marker.forward = types.MethodType(_other_forward, marker)
        model._marker = marker
        assert not _model_consumes_packed_metadata(model)

    def test_explicit_signature_declaration_counts_as_native(self):
        class _NativePacked(torch.nn.Module):
            def forward(self, hidden_states, block_attention_mask=None):
                return hidden_states

        model = _KwargsSinkLM()
        model._native = _NativePacked()
        assert _model_consumes_packed_metadata(model)


# ---------------------------------------------------------------------------
# DiffusionTrainer.compute_loss guard behaviour
# ---------------------------------------------------------------------------


class TestPackedMetadataGuard:
    def test_unpatched_model_packed_batch_warns_once(self, tmp_path, caplog):
        tokenizer = _make_tokenizer()
        model = _KwargsSinkLM()
        trainer = _make_trainer(tmp_path, model.inner, tokenizer)

        with caplog.at_level(logging.WARNING, logger=TRAINER_LOGGER):
            loss1 = trainer.compute_loss(model, _packed_batch(trainer))
            loss2 = trainer.compute_loss(model, _packed_batch(trainer))

        warnings = _guard_warnings(caplog)
        assert len(warnings) == 1, "guard must warn exactly once per trainer"
        message = warnings[0].getMessage()
        assert "block_attention_mask" in message
        assert "packed_seq_lengths" in message
        # Warning only — the loss must still be computed.
        assert torch.isfinite(loss1) and torch.isfinite(loss2)

    def test_patched_marker_model_does_not_warn(self, tmp_path, caplog):
        tokenizer = _make_tokenizer()
        model = _KwargsSinkLM()
        _mark_as_unturtle_patched(model)
        trainer = _make_trainer(tmp_path, model.inner, tokenizer)

        with caplog.at_level(logging.WARNING, logger=TRAINER_LOGGER):
            loss = trainer.compute_loss(model, _packed_batch(trainer))

        assert not _guard_warnings(caplog)
        assert torch.isfinite(loss)

    def test_unpacked_batches_never_trigger_the_check(self, tmp_path, caplog):
        tokenizer = _make_tokenizer()
        model = _KwargsSinkLM()
        trainer = _make_trainer(tmp_path, model.inner, tokenizer)

        batch = _packed_batch(trainer)
        for key in ("block_attention_mask", "packed_seq_lengths"):
            batch.pop(key, None)

        with caplog.at_level(logging.WARNING, logger=TRAINER_LOGGER):
            loss = trainer.compute_loss(model, batch)

        assert not _guard_warnings(caplog)
        # The one-shot flag must remain unarmed so a later packed batch warns.
        assert trainer._packed_metadata_checked is False
        assert torch.isfinite(loss)
