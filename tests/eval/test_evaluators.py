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

from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from unturtle.diffusion import PackedMaskedDiffusionDataCollator
from unturtle.eval import GenerationEvaluator, MaskedDiffusionEvaluator

from tests.diffusion.test_integration import _make_tokenizer, _make_bert, VOCAB_SIZE, SEQ_LEN


def _make_diffusion_dataset(num_rows: int = 6, prompt_len: int = 4) -> Dataset:
    rows = []
    for _ in range(num_rows):
        ids = torch.randint(5, VOCAB_SIZE, (SEQ_LEN,)).tolist()
        labels = [-100] * prompt_len + ids[prompt_len:]
        rows.append(
            {
                "input_ids": ids,
                "labels": labels,
                "attention_mask": [1] * SEQ_LEN,
            }
        )
    return Dataset.from_list(rows)


class TinyDiffusionModel(torch.nn.Module):
    def __init__(self, mask_token_id: int | None = None):
        super().__init__()
        self.calls = 0
        self.last_attention_mask = None
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.config = type("Config", (), {"mask_token_id": mask_token_id})()

    def diffusion_generate(self, input_ids, max_length=None, attention_mask=None, **_kwargs):
        self.calls += 1
        self.last_attention_mask = attention_mask.clone() if attention_mask is not None else None
        out = input_ids.clone()
        target_length = max_length or out.shape[1]
        if target_length > out.shape[1]:
            pad = torch.full(
                (out.shape[0], target_length - out.shape[1]),
                7,
                dtype=out.dtype,
                device=out.device,
            )
            return torch.cat([out, pad], dim=1)
        if out.shape[1] > 0:
            out[:, -1] = 7
        return out


class TestMaskedDiffusionEvaluator:
    @pytest.fixture
    def tokenizer(self):
        return _make_tokenizer()

    def test_evaluate_returns_expected_keys(self, tokenizer):
        model = _make_bert("cpu")
        evaluator = MaskedDiffusionEvaluator(model=model, tokenizer=tokenizer)
        metrics = evaluator.evaluate(_make_diffusion_dataset(), batch_size=2)
        assert set(metrics) == {
            "eval_loss",
            "eval_masked_token_nll",
            "eval_perplexity",
            "eval_mask_rate",
            "eval_num_batches",
        }
        assert metrics["eval_num_batches"] == 3.0
        assert metrics["eval_loss"] >= 0.0

    def test_evaluate_respects_max_batches(self, tokenizer):
        model = _make_bert("cpu")
        evaluator = MaskedDiffusionEvaluator(model=model, tokenizer=tokenizer)
        metrics = evaluator.evaluate(_make_diffusion_dataset(num_rows=8), batch_size=2, max_batches=2)
        assert metrics["eval_num_batches"] == 2.0

    def test_evaluate_uses_completion_only_masking(self, tokenizer):
        model = _make_bert("cpu")
        evaluator = MaskedDiffusionEvaluator(
            model=model,
            tokenizer=tokenizer,
            completion_only=True,
        )
        metrics = evaluator.evaluate(_make_diffusion_dataset(num_rows=4, prompt_len=6), batch_size=2)
        assert 0.0 <= metrics["eval_mask_rate"] <= 1.0

    def test_evaluator_uses_model_config_mask_token_id_fallback(self, tokenizer):
        tokenizer.mask_token = None
        tokenizer.mask_token_id = None
        model = _make_bert("cpu")
        model.config.mask_token_id = 2
        evaluator = MaskedDiffusionEvaluator(model=model, tokenizer=tokenizer)
        assert evaluator.data_collator.mask_token_id == 2

    def test_mask_rate_uses_maskable_tokens(self, tokenizer):
        model = _make_bert("cpu")
        evaluator = MaskedDiffusionEvaluator(model=model, tokenizer=tokenizer)
        metrics = evaluator.evaluate(_make_diffusion_dataset(num_rows=4, prompt_len=8), batch_size=2)
        assert 0.0 <= metrics["eval_mask_rate"] < 1.0

    def test_weighted_eval_reports_unweighted_nll_and_perplexity(self, tokenizer):
        model = _make_bert("cpu")
        evaluator = MaskedDiffusionEvaluator(
            model=model,
            tokenizer=tokenizer,
            loss_weight_type="timestep",
        )
        metrics = evaluator.evaluate(_make_diffusion_dataset(num_rows=4), batch_size=2)
        assert metrics["eval_loss"] != metrics["eval_masked_token_nll"]
        assert metrics["eval_perplexity"] >= 1.0

    def test_evaluator_rejects_packed_non_uniform_weighting(self, tokenizer):
        model = _make_bert("cpu")
        collator = PackedMaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            max_seq_length=SEQ_LEN,
            scheduler=None,
            mask_token_id=tokenizer.mask_token_id,
        )
        with pytest.raises(ValueError, match="PackedMaskedDiffusionDataCollator"):
            MaskedDiffusionEvaluator(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                loss_weight_type="timestep",
            )


class TestGenerationEvaluator:
    def test_generation_metrics_return_expected_keys(self):
        model = TinyDiffusionModel()
        evaluator = GenerationEvaluator(model=model, tokenizer=None)
        dataset = [{"input_ids": [1, 2, 3, 0], "references": [7]}]
        metrics = evaluator.evaluate(dataset)
        assert set(metrics) == {"gen_exact_match", "gen_token_accuracy", "gen_num_examples"}
        assert metrics["gen_num_examples"] == 1.0

    def test_generation_uses_diffusion_generate(self):
        model = TinyDiffusionModel()
        evaluator = GenerationEvaluator(model=model, tokenizer=None)
        dataset = [{"input_ids": [1, 2, 3, 0], "references": [7]}]
        evaluator.evaluate(dataset)
        assert model.calls == 1

    def test_generation_exact_match_and_token_accuracy(self):
        model = TinyDiffusionModel()
        evaluator = GenerationEvaluator(model=model, tokenizer=None)
        dataset = [
            {"input_ids": [1, 2, 3, 0], "references": [7]},
            {"input_ids": [4, 5, 6, 0], "references": [9]},
        ]
        metrics = evaluator.evaluate(dataset)
        assert metrics["gen_exact_match"] == 0.5
        assert metrics["gen_token_accuracy"] == 0.5

    def test_generation_uses_prompt_prefix_from_labels(self):
        model = TinyDiffusionModel()
        evaluator = GenerationEvaluator(model=model, tokenizer=None)
        dataset = [
            {
                "input_ids": [1, 2, 3, 4],
                "labels": [-100, -100, 7, 7],
            }
        ]
        metrics = evaluator.evaluate(dataset)
        assert metrics["gen_exact_match"] == 1.0
        assert metrics["gen_token_accuracy"] == 1.0

    def test_generation_trims_padding_and_passes_attention_mask(self):
        model = TinyDiffusionModel()
        evaluator = GenerationEvaluator(model=model, tokenizer=None)
        dataset = [
            {
                "input_ids": [1, 2, 7, 7, 0, 0],
                "labels": [-100, -100, 7, 7, -100, -100],
                "attention_mask": [1, 1, 1, 1, 0, 0],
            }
        ]
        metrics = evaluator.evaluate(dataset)
        assert metrics["gen_exact_match"] == 1.0
        assert metrics["gen_token_accuracy"] == 1.0
        assert model.last_attention_mask is not None
        assert model.last_attention_mask.tolist() == [[1, 1]]

    def test_generation_handles_left_padding(self):
        model = TinyDiffusionModel()
        evaluator = GenerationEvaluator(model=model, tokenizer=None)
        dataset = [
            {
                "input_ids": [0, 0, 1, 2, 7, 7],
                "labels": [-100, -100, -100, -100, 7, 7],
                "attention_mask": [0, 0, 1, 1, 1, 1],
            }
        ]
        metrics = evaluator.evaluate(dataset)
        assert metrics["gen_exact_match"] == 1.0
        assert metrics["gen_token_accuracy"] == 1.0
        assert model.last_attention_mask is not None
        assert model.last_attention_mask.tolist() == [[1, 1]]
