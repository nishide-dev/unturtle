# unturtle/eval/harness/configs.py
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

"""Per-(model_family, task) decoding configs for canonical dLLM evaluation.

The dLLM paper shows benchmark scores swing sharply with inference hyperparameters
(max_new_tokens, eos-suppression, parallel-decode steps, temperature). To make scores
reproducible and comparable, every canonical evaluation pins an explicit, versioned
``DecodingConfig`` and records it alongside the score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DecodingConfig:
    """Explicit decoding hyperparameters for one (model_family, task) pair."""

    model_family: str
    task: str
    max_new_tokens: int
    num_steps: int
    temperature: float
    use_chat_template: bool
    fewshot: int
    algorithm: str = "mdlm"

    def as_dict(self) -> dict[str, Any]:
        """Serialize for recording alongside benchmark results."""
        return asdict(self)


# NOTE: only hyperparameters that actually take effect in the adapter/generation path are
# recorded here, so a recorded config never misrepresents the real decoding. The
# ``algorithm`` field selects the decode path: "mdlm" (default) uses the no-cache MDLM
# path with steps/temperature/mask_token_id; "block_ar" uses the DiffusionGemma
# block-autoregressive path with max_denoising_steps (temperature is recorded in the
# config for documentation but is NOT forwarded on the block_ar path — entropy knobs
# stay at upstream defaults). Each recorded config is stored alongside its score so the
# decode path is always unambiguous. Knobs the dLLM paper highlights but not yet wired
# into generation (e.g. eos-suppression, parallel-decode width) will be added when the
# generation path honors them.


# Canonical decoding configs. Keyed by (model_family, task).
# Values follow dLLM / d1 reasoning-eval conventions; tune as reproductions are verified.
_DECODING_CONFIGS: dict[tuple[str, str], DecodingConfig] = {
    ("a2d_qwen3", "gsm8k"): DecodingConfig(
        model_family="a2d_qwen3",
        task="gsm8k",
        max_new_tokens=256,
        num_steps=256,
        temperature=0.0,
        use_chat_template=True,
        fewshot=0,
    ),
    ("a2d_qwen3", "gsm8k_cot"): DecodingConfig(
        model_family="a2d_qwen3",
        task="gsm8k_cot",
        max_new_tokens=512,
        num_steps=512,
        temperature=0.0,
        use_chat_template=True,
        fewshot=8,
    ),
    ("diffusion_gemma", "gsm8k"): DecodingConfig(
        model_family="diffusion_gemma",
        task="gsm8k",
        max_new_tokens=256,
        num_steps=48,
        # temperature recorded for documentation; NOT forwarded on block_ar path
        temperature=0.0,
        use_chat_template=True,
        fewshot=0,
        algorithm="block_ar",
    ),
}


def get_decoding_config(model_family: str, task: str) -> DecodingConfig:
    """Return the canonical decoding config for (model_family, task)."""
    try:
        return _DECODING_CONFIGS[(model_family, task)]
    except KeyError as exc:
        available = sorted(_DECODING_CONFIGS)
        raise KeyError(
            f"No decoding config for {(model_family, task)!r}. Available: {available}"
        ) from exc


def list_decoding_configs() -> list[tuple[str, str]]:
    """List all registered (model_family, task) config keys."""
    return sorted(_DECODING_CONFIGS)
