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

from typing import Any

import torch

from .base import BaseEvaluator


class GenerationEvaluator(BaseEvaluator):
    """Evaluate diffusion generation against token-level references."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        metric_key_prefix: str = "gen",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(model=model, tokenizer=tokenizer, device=device)
        self.metric_key_prefix = metric_key_prefix

    @staticmethod
    def _to_long_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.long)
        return torch.tensor(value, dtype=torch.long)

    def _extract_prompt_and_reference(
        self,
        example: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        input_ids = self._to_long_tensor(example["input_ids"])
        attention_mask = None
        valid_positions = None
        if "attention_mask" in example and example["attention_mask"] is not None:
            attention_mask = self._to_long_tensor(example["attention_mask"])
            valid_positions = attention_mask.bool()
            input_ids = input_ids[valid_positions]
            attention_mask = attention_mask[valid_positions]

        if "references" in example and example["references"] is not None:
            return input_ids, self._to_long_tensor(example["references"]), attention_mask

        if "labels" in example and example["labels"] is not None:
            labels = self._to_long_tensor(example["labels"])
            if valid_positions is not None:
                labels = labels[valid_positions]

            supervised_positions = (labels != -100).nonzero(as_tuple=False).flatten()
            if supervised_positions.numel() == 0:
                return input_ids, None, attention_mask

            start = int(supervised_positions[0].item())
            end = int(supervised_positions[-1].item()) + 1
            return input_ids[:start], labels[start:end], attention_mask[:start] if attention_mask is not None else None

        return input_ids, None, attention_mask

    def _generate_one(
        self,
        prompt_ids: torch.Tensor,
        generation_config: Any | None,
        reference_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        prompt_ids = prompt_ids.unsqueeze(0).to(self.device)
        generation_kwargs: dict[str, Any] = {}
        if generation_config is not None:
            if isinstance(generation_config, dict):
                generation_kwargs.update(generation_config)
            else:
                generation_kwargs["generation_config"] = generation_config

        if "max_length" not in generation_kwargs and "generation_config" not in generation_kwargs:
            target_len = int(reference_ids.numel()) if reference_ids is not None else 1
            generation_kwargs["max_length"] = prompt_ids.shape[1] + target_len
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask.unsqueeze(0).to(self.device)

        diffusion_generate = getattr(self.model, "diffusion_generate", None)
        if callable(diffusion_generate):
            sequences = diffusion_generate(prompt_ids, **generation_kwargs)
        else:
            generate = getattr(self.model, "generate")
            sequences = generate(prompt_ids, **generation_kwargs)

        if hasattr(sequences, "sequences"):
            sequences = sequences.sequences
        return sequences[0].detach().cpu()

    def evaluate(
        self,
        dataset: Any,
        generation_config: Any | None = None,
        batch_size: int = 1,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        del batch_size  # current implementation evaluates one prompt at a time

        total_examples = 0
        exact_matches = 0
        correct_tokens = 0
        total_reference_tokens = 0

        with self.evaluation_mode():
            for idx, example in enumerate(dataset):
                if max_batches is not None and idx >= max_batches:
                    break

                prompt_ids, reference_ids, attention_mask = self._extract_prompt_and_reference(example)
                generated = self._generate_one(prompt_ids, generation_config, reference_ids, attention_mask)

                if reference_ids is None or reference_ids.numel() == 0:
                    total_examples += 1
                    continue

                prompt_len = int(prompt_ids.numel())
                target_len = int(reference_ids.numel())
                generated_suffix = generated[prompt_len : prompt_len + target_len]

                if generated_suffix.numel() < target_len:
                    padded = torch.full((target_len,), -1, dtype=torch.long)
                    padded[: generated_suffix.numel()] = generated_suffix
                    generated_suffix = padded

                matches = generated_suffix.eq(reference_ids.cpu())
                exact_matches += int(matches.all().item())
                correct_tokens += int(matches.sum().item())
                total_reference_tokens += target_len
                total_examples += 1

        prefix = self.metric_key_prefix
        denom_examples = max(total_examples, 1)
        denom_tokens = max(total_reference_tokens, 1)
        return {
            self._metric_key(prefix, "exact_match"): exact_matches / denom_examples,
            self._metric_key(prefix, "token_accuracy"): correct_tokens / denom_tokens,
            self._metric_key(prefix, "num_examples"): float(total_examples),
        }
