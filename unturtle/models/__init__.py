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

"""unturtle.models — Diffusion Language Model architectures.

Public API (generation infrastructure + backbones — available now)::

    from unturtle.models.generation.cache import BlockKVCache
    from unturtle.models.generation.diffusion_generation_utils import (
        MaskedDiffusionGenerationConfig,
        MaskedDiffusionGenerationMixin,
        MaskedDiffusionModelOutput,
        prepare_for_sampling,
    )
    from unturtle.models.backbones.llada import LLaDAConfig, LLaDAModelLM
    from unturtle.models.backbones.dream import DreamConfig, DreamModel
    from unturtle.models.backbones.modernbert import (
        DiffusionModernBertConfig,
        DiffusionModernBertForMaskedLM,
    )

Conversion methods will be exported here once Task 5 lands:

    from unturtle.models.conversion.a2d import (
        TinyA2DLlamaConfig, TinyA2DLlamaLMHeadModel,
        TinyA2DQwen2Config, TinyA2DQwen2LMHeadModel,
        TinyA2DQwen3Config, TinyA2DQwen3LMHeadModel,
    )
"""

from .backbones.dream import (
    DreamConfig,
    DreamGenerationConfig,
    DreamGenerationMixin,
    DreamModel,
)
from .backbones.llada import (
    LLaDAConfig,
    LLaDAGenerationConfig,
    LLaDAGenerationMixin,
    LLaDAModel,
    LLaDAModelLM,
)
from .backbones.modernbert import (
    # Backward compat aliases
    A2DModernBertConfig,
    A2DModernBertForMaskedLM,
    A2DModernBertModel,
    DiffusionModernBertConfig,
    DiffusionModernBertForMaskedLM,
    DiffusionModernBertModel,
)
from .generation.cache import BlockKVCache
from .generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
    MaskedDiffusionGenerationMixin,
    MaskedDiffusionModelOutput,
    prepare_for_sampling,
)

__all__ = [
    # generation infrastructure
    "BlockKVCache",
    "MaskedDiffusionGenerationConfig",
    "MaskedDiffusionGenerationMixin",
    "MaskedDiffusionModelOutput",
    "prepare_for_sampling",
    # LLaDA backbone
    "LLaDAConfig",
    "LLaDAGenerationConfig",
    "LLaDAGenerationMixin",
    "LLaDAModel",
    "LLaDAModelLM",
    # Dream backbone
    "DreamConfig",
    "DreamGenerationConfig",
    "DreamGenerationMixin",
    "DreamModel",
    # ModernBERT diffusion backbone
    "DiffusionModernBertConfig",
    "DiffusionModernBertModel",
    "DiffusionModernBertForMaskedLM",
    # ModernBERT backward compat aliases (A2D-prefixed)
    "A2DModernBertConfig",
    "A2DModernBertModel",
    "A2DModernBertForMaskedLM",
    # Conversion (Task 5) exports will be added here.
]
