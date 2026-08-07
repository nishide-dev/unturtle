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

"""Shared ``mask_token_id`` resolution for the masked-diffusion stack.

The tokenizer→model-config fallback chain was previously duplicated across
``DiffusionTrainer``, ``MaskedDiffusionEvaluator``, and the CLI collator
factory.  ``MaskedDiffusionProcess`` takes a required ``int``, so the
resolution has to happen at exactly one place in orchestration (#62).

For real checkpoints the id often lives on ``model.config`` rather than the
tokenizer (see CLAUDE.md's gotcha list), so the config fallback is not
optional.
"""

from __future__ import annotations

from typing import Any


def resolve_mask_token_id(
    tokenizer: Any = None,
    model: Any = None,
    explicit: int | None = None,
) -> int | None:
    """Resolve the ``[MASK]`` token id, or return ``None`` if unavailable.

    Precedence: ``explicit`` → ``tokenizer.mask_token_id`` →
    ``model.config.mask_token_id``.

    Returns ``None`` rather than raising so callers can attach their own
    context; use :func:`require_mask_token_id` when the id is mandatory.
    """
    if explicit is not None:
        return int(explicit)

    from_tokenizer = getattr(tokenizer, "mask_token_id", None)
    if from_tokenizer is not None:
        return int(from_tokenizer)

    from_config = getattr(getattr(model, "config", None), "mask_token_id", None)
    if from_config is not None:
        return int(from_config)

    return None


def require_mask_token_id(
    tokenizer: Any = None,
    model: Any = None,
    explicit: int | None = None,
    *,
    context: str = "masked diffusion",
) -> int:
    """Like :func:`resolve_mask_token_id`, but raise when nothing resolves.

    Raises:
        ValueError: if neither the tokenizer nor the model config carries a
            ``mask_token_id`` and none was passed explicitly.
    """
    mask_token_id = resolve_mask_token_id(tokenizer, model, explicit)
    if mask_token_id is None:
        raise ValueError(
            f"Could not resolve mask_token_id for {context}: neither the "
            "tokenizer nor model.config defines one.  Pass mask_token_id "
            "explicitly."
        )
    return mask_token_id
