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

import re
from typing import Any

import torch

from ._answer_parser import extract_numeric_answer
from .base import BaseEvaluator

DEFAULT_SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step. "
    "Put your final answer in \\boxed{}."
)


def _extract_gold_answer(answer_text: str) -> float | None:
    """Extract the numeric gold answer from a GSM8K answer string.

    GSM8K answers end with '#### <number>'.
    """
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_text)
    if m:
        raw = m.group(1).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            pass
    return extract_numeric_answer(answer_text)


class GSM8KEvaluator(BaseEvaluator):
    """Smoke-only GSM8K evaluator for masked-diffusion models.

    dLLM-only by design: it pins ``algorithm="mdlm"`` and forwards a
    masked-diffusion ``mask_token_id``, so it cannot drive non-masked
    backbones (e.g. DiffusionGemma block-AR).

    This evaluator is kept for fast local sanity checks and benchmark-harness
    debugging. It is not the authoritative formal benchmark path for Unturtle.

    Loads the HuggingFace ``gsm8k`` dataset, formats each question as a
    zero-shot chat prompt via ``apply_chat_template``, generates with
    ``generate``, extracts the numeric answer, and compares to
    the gold answer.

    Metrics returned by :meth:`evaluate`:

    - ``gsm8k_accuracy``: fraction of examples answered correctly.
    - ``gsm8k_num_correct``: raw count of correct answers.
    - ``gsm8k_num_examples``: total examples evaluated.
    - ``gsm8k_parse_failures``: examples where no number could be extracted.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        num_steps: int = 128,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        metric_key_prefix: str = "gsm8k",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.num_steps = num_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.metric_key_prefix = metric_key_prefix
        # Two-stage lookup mirroring the harness adapter: tokenizer first, then
        # the model config (real checkpoints may carry mask_token_id only on
        # model.config, not the tokenizer).
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(
                getattr(model, "config", None), "mask_token_id", None
            )
        self.mask_token_id = mask_token_id

    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def _generate(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        prompt_len = input_ids.shape[1]

        max_length = prompt_len + self.max_new_tokens
        # algorithm="mdlm": pins pre-unification no-cache MDLM so recorded
        # DecodingConfigs keep describing the real decode path.
        sequences = self.model.generate(
            input_ids,
            algorithm="mdlm",
            max_length=max_length,
            mask_token_id=self.mask_token_id,
            steps=self.num_steps,
            temperature=self.temperature,
        )

        if hasattr(sequences, "sequences"):
            sequences = sequences.sequences
        generated_ids = sequences[0, prompt_len:].detach().cpu()
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _load_dataset(
        self,
        split: str,
        seed: int,
        num_examples: int | None,
    ) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split=split)
        ds = ds.shuffle(seed=seed)
        if num_examples is not None:
            ds = ds.select(range(min(num_examples, len(ds))))
        return list(ds)

    def evaluate(
        self,
        split: str = "test",
        num_examples: int | None = None,
        seed: int = 42,
    ) -> dict[str, float]:
        # Generation runs one example at a time: generate is
        # memory-intensive and does not support batch_size > 1 here.
        dataset = self._load_dataset(split, seed, num_examples)

        num_correct = 0
        parse_failures = 0
        gold_parse_failures = 0
        total = 0

        with self.evaluation_mode():
            for example in dataset:
                question: str = example["question"]
                gold = _extract_gold_answer(example["answer"])

                if gold is None:
                    gold_parse_failures += 1
                    total += 1
                    continue

                prompt = self._build_prompt(question)
                generated = self._generate(prompt)
                predicted = extract_numeric_answer(generated)

                if predicted is None:
                    parse_failures += 1
                elif abs(predicted - gold) < 1e-6:
                    num_correct += 1

                total += 1

        if gold_parse_failures > 0:
            import warnings

            warnings.warn(
                f"{gold_parse_failures}/{total} gold answers could not be parsed. "
                "Check the dataset format and split.",
                stacklevel=2,
            )

        prefix = self.metric_key_prefix
        denom = max(total, 1)
        return {
            self._metric_key(prefix, "accuracy"): num_correct / denom,
            self._metric_key(prefix, "num_correct"): float(num_correct),
            self._metric_key(prefix, "num_examples"): float(total),
            self._metric_key(prefix, "parse_failures"): float(parse_failures),
            self._metric_key(prefix, "gold_parse_failures"): float(gold_parse_failures),
        }
