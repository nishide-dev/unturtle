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

Public API (generation infrastructure — available now)::

    from unturtle.models.generation.cache import BlockKVCache
    from unturtle.models.generation.diffusion_generation_utils import (
        MaskedDiffusionGenerationConfig,
        MaskedDiffusionGenerationMixin,
        MaskedDiffusionModelOutput,
        prepare_for_sampling,
    )

Backbones and conversion methods will be exported here once Task 4 / Task 5 land:

    from unturtle.models.backbones.llada import LLaDAConfig, LLaDAModelLM
    from unturtle.models.backbones.dream import DreamConfig, DreamModel
    from unturtle.models.conversion.a2d import (
        TinyA2DLlamaConfig, TinyA2DLlamaLMHeadModel,
        TinyA2DQwen2Config, TinyA2DQwen2LMHeadModel,
        TinyA2DQwen3Config, TinyA2DQwen3LMHeadModel,
    )
"""

from .generation.cache import BlockKVCache
from .generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
    MaskedDiffusionGenerationMixin,
    MaskedDiffusionModelOutput,
    prepare_for_sampling,
)

__all__ = [
    "BlockKVCache",
    "MaskedDiffusionGenerationConfig",
    "MaskedDiffusionGenerationMixin",
    "MaskedDiffusionModelOutput",
    "prepare_for_sampling",
    # Backbones (Task 4) and conversion (Task 5) exports will be added here.
]
