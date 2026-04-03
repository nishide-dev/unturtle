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

Public API::

    from unturtle.models.a2d import (
        A2DLlamaConfig, A2DLlamaLMHeadModel,
        A2DQwen2Config, A2DQwen2LMHeadModel,
        A2DQwen3Config, A2DQwen3LMHeadModel,
    )
    from unturtle.models.llada import LLaDAConfig, LLaDAModelLM
    from unturtle.models.dream import DreamConfig, DreamModel
"""

from .diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
    MaskedDiffusionGenerationMixin,
    MaskedDiffusionModelOutput,
)
from .a2d import (
    A2DGenerationConfig,
    A2DGenerationMixin,
    A2DLlamaConfig,
    A2DLlamaModel,
    A2DLlamaLMHeadModel,
    A2DModernBertConfig,
    A2DModernBertModel,
    A2DModernBertForMaskedLM,
    A2DQwen2Config,
    A2DQwen2Model,
    A2DQwen2LMHeadModel,
    A2DQwen3Config,
    A2DQwen3Model,
    A2DQwen3LMHeadModel,
)
from .llada import (
    LLaDAGenerationConfig,
    LLaDAGenerationMixin,
    LLaDAConfig,
    LLaDAModel,
    LLaDAModelLM,
)
from .dream import (
    DreamConfig,
    DreamModel,
    DreamGenerationMixin,
    DreamGenerationConfig,
)

__all__ = [
    "MaskedDiffusionGenerationConfig",
    "MaskedDiffusionGenerationMixin",
    "MaskedDiffusionModelOutput",
    "A2DGenerationConfig",
    "A2DGenerationMixin",
    "A2DLlamaConfig",
    "A2DLlamaModel",
    "A2DLlamaLMHeadModel",
    "A2DModernBertConfig",
    "A2DModernBertModel",
    "A2DModernBertForMaskedLM",
    "A2DQwen2Config",
    "A2DQwen2Model",
    "A2DQwen2LMHeadModel",
    "A2DQwen3Config",
    "A2DQwen3Model",
    "A2DQwen3LMHeadModel",
    "LLaDAGenerationConfig",
    "LLaDAGenerationMixin",
    "LLaDAConfig",
    "LLaDAModel",
    "LLaDAModelLM",
    "DreamConfig",
    "DreamModel",
    "DreamGenerationMixin",
    "DreamGenerationConfig",
]
