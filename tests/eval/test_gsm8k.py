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

import torch

from unturtle.eval._answer_parser import extract_numeric_answer
from unturtle.eval.gsm8k import GSM8KEvaluator


class TestExtractNumericAnswer:
    def test_extract_boxed_answer(self):
        assert extract_numeric_answer(r"therefore \boxed{42}") == 42.0

    def test_extract_bare_number_fallback(self):
        assert extract_numeric_answer("the answer is 17") == 17.0

    def test_extract_returns_none_on_empty(self):
        assert extract_numeric_answer("no numbers here at all!") is None

    def test_extract_returns_none_on_empty_string(self):
        assert extract_numeric_answer("") is None

    def test_numeric_normalisation_float(self):
        assert extract_numeric_answer(r"\boxed{42.0}") == 42.0

    def test_numeric_normalisation_comma(self):
        assert extract_numeric_answer(r"\boxed{1,234}") == 1234.0

    def test_numeric_normalisation_negative(self):
        assert extract_numeric_answer(r"\boxed{-3}") == -3.0

    def test_last_boxed_wins(self):
        # When multiple \boxed{} appear, last one wins
        assert extract_numeric_answer(r"\boxed{10} then \boxed{99}") == 99.0

    def test_last_bare_number_wins(self):
        assert extract_numeric_answer("first 10 then 99") == 99.0

    def test_boxed_takes_priority_over_bare(self):
        assert extract_numeric_answer(r"99 and \boxed{42}") == 42.0


# ---------------------------------------------------------------------------
# Stub model — mimics generate behaviour without real weights
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal tokenizer stub for GSM8KEvaluator tests."""

    eos_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return messages[-1]["content"]

    def encode(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(c) % 100 + 3 for c in text[:8]]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        return self._answer

    _answer: str = r"\boxed{42}"


class _CorrectAnswerModel(torch.nn.Module):
    """Always generates a sequence whose decode yields the gold answer."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0
        self.last_algorithm: str | None = None
        self.last_mask_token_id: object = "<unset>"

    def generate(
        self,
        input_ids,
        *,
        algorithm="auto",
        max_length=None,
        steps=None,
        temperature=None,
        mask_token_id="<unset>",
        **_kw,
    ):
        self.calls += 1
        self.last_algorithm = algorithm
        self.last_mask_token_id = mask_token_id
        pad = torch.full((input_ids.shape[0], 1), 7, dtype=input_ids.dtype)
        return torch.cat([input_ids, pad], dim=1)


def test_gsm8k_evaluator_docstring_marks_it_as_smoke_only() -> None:
    assert "smoke" in (GSM8KEvaluator.__doc__ or "").lower()
    assert "formal" in (GSM8KEvaluator.__doc__ or "").lower()


class TestGSM8KEvaluator:
    """Tests for GSM8KEvaluator using stub model + tokenizer (no GPU, no real dataset)."""

    def _make_tiny_dataset(self, n: int = 4):
        return [
            {"question": f"What is {i} + {i}?", "answer": f"#### {i * 2}"}
            for i in range(1, n + 1)
        ]

    def test_evaluate_returns_expected_keys(self, monkeypatch):
        tok = _StubTokenizer()
        tok._answer = r"\boxed{42}"
        model = _CorrectAnswerModel()

        from unturtle.eval.gsm8k import GSM8KEvaluator

        evaluator = GSM8KEvaluator(
            model=model, tokenizer=tok, num_steps=4, max_new_tokens=8
        )
        dataset = self._make_tiny_dataset(3)
        monkeypatch.setattr(
            evaluator, "_load_dataset", lambda split, seed, num_examples: dataset
        )

        metrics = evaluator.evaluate()
        assert set(metrics) == {
            "gsm8k_accuracy",
            "gsm8k_num_correct",
            "gsm8k_num_examples",
            "gsm8k_parse_failures",
            "gsm8k_gold_parse_failures",
        }

    def test_evaluate_accuracy_all_correct(self, monkeypatch):
        tok = _StubTokenizer()
        tok._answer = r"\boxed{2}"
        model = _CorrectAnswerModel()

        from unturtle.eval.gsm8k import GSM8KEvaluator

        evaluator = GSM8KEvaluator(
            model=model, tokenizer=tok, num_steps=4, max_new_tokens=8
        )
        dataset = [{"question": "What is 1 + 1?", "answer": "#### 2"}]
        monkeypatch.setattr(
            evaluator, "_load_dataset", lambda split, seed, num_examples: dataset
        )

        metrics = evaluator.evaluate()
        assert metrics["gsm8k_accuracy"] == 1.0
        assert metrics["gsm8k_num_correct"] == 1.0
        assert metrics["gsm8k_num_examples"] == 1.0

    def test_evaluate_accuracy_none_correct(self, monkeypatch):
        tok = _StubTokenizer()
        tok._answer = r"\boxed{999}"
        model = _CorrectAnswerModel()

        from unturtle.eval.gsm8k import GSM8KEvaluator

        evaluator = GSM8KEvaluator(
            model=model, tokenizer=tok, num_steps=4, max_new_tokens=8
        )
        dataset = [{"question": "What is 1 + 1?", "answer": "#### 2"}]
        monkeypatch.setattr(
            evaluator, "_load_dataset", lambda split, seed, num_examples: dataset
        )

        metrics = evaluator.evaluate()
        assert metrics["gsm8k_accuracy"] == 0.0
        assert metrics["gsm8k_num_correct"] == 0.0

    def test_evaluate_parse_failure_counted(self, monkeypatch):
        tok = _StubTokenizer()
        tok._answer = "I have no idea"
        model = _CorrectAnswerModel()

        from unturtle.eval.gsm8k import GSM8KEvaluator

        evaluator = GSM8KEvaluator(
            model=model, tokenizer=tok, num_steps=4, max_new_tokens=8
        )
        dataset = [{"question": "What is 1 + 1?", "answer": "#### 2"}]
        monkeypatch.setattr(
            evaluator, "_load_dataset", lambda split, seed, num_examples: dataset
        )

        metrics = evaluator.evaluate()
        assert metrics["gsm8k_parse_failures"] == 1.0
        assert metrics["gsm8k_accuracy"] == 0.0

    def test_evaluate_num_examples_limit(self, monkeypatch):
        tok = _StubTokenizer()
        tok._answer = r"\boxed{0}"
        model = _CorrectAnswerModel()

        from unturtle.eval.gsm8k import GSM8KEvaluator

        evaluator = GSM8KEvaluator(
            model=model, tokenizer=tok, num_steps=4, max_new_tokens=8
        )
        dataset = self._make_tiny_dataset(10)

        def _limited(split, seed, num_examples):
            return dataset[:num_examples] if num_examples else dataset

        monkeypatch.setattr(evaluator, "_load_dataset", _limited)

        metrics = evaluator.evaluate(num_examples=3)
        assert metrics["gsm8k_num_examples"] == 3.0

    def test_evaluate_uses_generate(self, monkeypatch):
        tok = _StubTokenizer()
        tok._answer = r"\boxed{2}"
        model = _CorrectAnswerModel()

        from unturtle.eval.gsm8k import GSM8KEvaluator

        evaluator = GSM8KEvaluator(
            model=model, tokenizer=tok, num_steps=4, max_new_tokens=8
        )
        dataset = [{"question": "What is 1 + 1?", "answer": "#### 2"}]
        monkeypatch.setattr(
            evaluator, "_load_dataset", lambda split, seed, num_examples: dataset
        )

        evaluator.evaluate()
        assert model.calls == 1
        assert model.last_algorithm == "mdlm"

    def test_mask_token_id_resolved_from_tokenizer(self, monkeypatch):
        tok = _StubTokenizer()
        tok._answer = r"\boxed{2}"
        tok.mask_token_id = 99
        model = _CorrectAnswerModel()
        # config fallback present but should be shadowed by the tokenizer value.
        model.config = type("Cfg", (), {"mask_token_id": 7})()

        from unturtle.eval.gsm8k import GSM8KEvaluator

        evaluator = GSM8KEvaluator(
            model=model, tokenizer=tok, num_steps=4, max_new_tokens=8
        )
        assert evaluator.mask_token_id == 99

        dataset = [{"question": "What is 1 + 1?", "answer": "#### 2"}]
        monkeypatch.setattr(
            evaluator, "_load_dataset", lambda split, seed, num_examples: dataset
        )
        evaluator.evaluate()
        assert model.last_mask_token_id == 99

    def test_mask_token_id_falls_back_to_model_config(self, monkeypatch):
        tok = _StubTokenizer()
        tok._answer = r"\boxed{2}"
        tok.mask_token_id = None  # tokenizer cannot supply it
        model = _CorrectAnswerModel()
        model.config = type("Cfg", (), {"mask_token_id": 7})()

        from unturtle.eval.gsm8k import GSM8KEvaluator

        evaluator = GSM8KEvaluator(
            model=model, tokenizer=tok, num_steps=4, max_new_tokens=8
        )
        assert evaluator.mask_token_id == 7

        dataset = [{"question": "What is 1 + 1?", "answer": "#### 2"}]
        monkeypatch.setattr(
            evaluator, "_load_dataset", lambda split, seed, num_examples: dataset
        )
        evaluator.evaluate()
        assert model.last_mask_token_id == 7


def test_gsm8k_evaluator_docstring_marks_it_as_dllm_only() -> None:
    assert "dllm-only" in (GSM8KEvaluator.__doc__ or "").lower()
