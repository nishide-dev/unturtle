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

"""lm-evaluation-harness adapter for Unturtle diffusion models.

``lm_eval`` is optional, so the ``lm_eval.api.model.LM`` subclass is defined inside a lazy
factory (``build_harness_lm``) rather than at module top level. Importing this module never
requires ``lm_eval``.
"""

from __future__ import annotations

from typing import Any


def _import_lm_base() -> type:
    try:
        from lm_eval.api.model import LM
    except Exception as exc:
        raise ImportError(
            "lm-evaluation-harness is required for the canonical harness adapter. "
            "Install it (optional 'eval' extra) to run canonical benchmarks."
        ) from exc
    return LM


def _normalize_until(until: Any) -> list[str]:
    """Normalize lm-eval's ``until`` (str | list[str] | None) to a list of stops.

    lm-eval passes a task's ``generation_kwargs["until"]`` through verbatim, and it is
    legitimately either a bare string (e.g. ``until: "Question:"``) or a list. ``list()``
    on a string would split it into characters and silently truncate generations at the
    first matching character — so a string must be wrapped, not iterated. Mirrors
    lm-eval's own ``handle_stop_sequences``.
    """
    if until is None:
        return []
    if isinstance(until, str):
        return [until]
    return [s for s in until if isinstance(s, str)]


def _apply_stop_sequences(text: str, stops: list[str]) -> str:
    idxs = [text.find(s) for s in stops if s and s in text]
    return text[: min(idxs)] if idxs else text


def build_harness_lm(
    *,
    model: Any,
    tokenizer: Any,
    num_steps: int,
    max_new_tokens: int,
    temperature: float,
    use_chat_template: bool,
) -> Any:
    """Build an lm_eval LM wrapping an Unturtle diffusion model.

    Defined as a factory because ``lm_eval.api.model.LM`` (the base class) is an optional
    dependency that must not be imported at module load time.
    """
    lm_base = _import_lm_base()

    class UnturtleHarnessLM(lm_base):  # type: ignore[misc, valid-type]
        """Routes lm-eval ``generate_until`` through ``generate``."""

        def __init__(self) -> None:
            super().__init__()
            self._model = model
            self._tokenizer = tokenizer
            self._num_steps = num_steps
            self._max_new_tokens = max_new_tokens
            self._temperature = temperature
            self._use_chat_template = use_chat_template

        def _build_prompt(self, context: str) -> str:
            apply = getattr(self._tokenizer, "apply_chat_template", None)
            if self._use_chat_template and callable(apply):
                return apply(
                    [{"role": "user", "content": context}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return context

        def _generate_one(self, context: str, gen_kwargs: dict[str, Any]) -> str:
            prompt = self._build_prompt(context)
            input_ids = self._tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=False
            )
            device = next(iter(self._model.parameters())).device
            input_ids = input_ids.to(device)
            prompt_len = input_ids.shape[1]
            max_new = int(gen_kwargs.get("max_gen_toks", self._max_new_tokens))
            max_length = prompt_len + max_new

            mask_token_id = getattr(self._tokenizer, "mask_token_id", None)
            if mask_token_id is None:
                mask_token_id = getattr(
                    getattr(self._model, "config", None), "mask_token_id", None
                )

            sequences = self._model.generate(
                input_ids,
                max_length=max_length,
                mask_token_id=mask_token_id,
                steps=self._num_steps,
                temperature=self._temperature,
            )

            if hasattr(sequences, "sequences"):
                sequences = sequences.sequences
            generated_ids = sequences[0, prompt_len:].detach().cpu()
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            return _apply_stop_sequences(
                text, _normalize_until(gen_kwargs.get("until"))
            )

        def generate_until(self, requests: list[Any]) -> list[str]:
            outputs: list[str] = []
            for req in requests:
                context, gen_kwargs = req.args
                outputs.append(self._generate_one(context, dict(gen_kwargs or {})))
            return outputs

        def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
            raise NotImplementedError(
                "loglikelihood is not supported for diffusion models in this adapter"
            )

        def loglikelihood_rolling(
            self, requests: list[Any]
        ) -> list[tuple[float, bool]]:
            raise NotImplementedError(
                "loglikelihood_rolling is not supported for diffusion models"
            )

    return UnturtleHarnessLM()
