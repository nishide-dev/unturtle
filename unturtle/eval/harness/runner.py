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

"""Canonical harness evaluation runner.

Loads an Unturtle diffusion model via the canonical loader, wraps it in the lm-eval
adapter, runs ``lm_eval.simple_evaluate``, and returns a summary that embeds the exact
``DecodingConfig`` used — so every score is reproducible.
"""

from __future__ import annotations

from typing import Any

from unturtle.fast_diffusion_model import FastDiffusionModel

from .configs import get_decoding_config
from .model_adapter import build_harness_lm


def _import_simple_evaluate():  # noqa: ANN202
    try:
        from lm_eval import simple_evaluate
    except Exception as exc:
        raise ImportError(
            "lm-evaluation-harness is required to run canonical evaluations. "
            "Install it (optional 'eval' extra)."
        ) from exc
    return simple_evaluate


def run_harness_evaluation(
    *,
    model_name: str,
    model_family: str,
    task: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run a canonical lm-eval-harness evaluation for one (model_family, task).

    Returns a summary embedding the recorded decoding config and raw results.
    """
    config = get_decoding_config(model_family, task)
    simple_evaluate = _import_simple_evaluate()

    model, tokenizer = FastDiffusionModel.from_pretrained(model_name)
    FastDiffusionModel.for_inference(model)

    lm = build_harness_lm(
        model=model,
        tokenizer=tokenizer,
        num_steps=config.num_steps,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        use_chat_template=config.use_chat_template,
        algorithm=config.algorithm,
    )

    results = simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=config.fewshot,
        limit=limit,
    )

    return {
        "task": task,
        "model": model_name,
        "model_family": model_family,
        "decoding_config": config.as_dict(),
        "results": results,
    }
