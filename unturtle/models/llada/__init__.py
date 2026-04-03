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

"""LLaDA (Large Language Diffusion with mAsking) models.

Native bidirectional diffusion language model architecture.
Reference: https://arxiv.org/abs/2502.09992

Usage::

    from unturtle.models.llada import LLaDAConfig, LLaDAModelLM

    config = LLaDAConfig(d_model=256, n_heads=4, n_layers=4, vocab_size=50257)
    model = LLaDAModelLM(config)
    # or load pretrained:
    # config = LLaDAConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    # model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
"""

from .configuration_llada import (
    LLaDAConfig,
    ModelConfig,
    LayerNormType,
    ActivationType,
    BlockType,
    InitFnType,
    ActivationCheckpointingStrategy,
)
from .generation_utils import LLaDAGenerationConfig, LLaDAGenerationMixin
from .modeling_llada import (
    LLaDAPreTrainedModel,
    LLaDAModel,
    LLaDAModelLM,
)

__all__ = [
    "LLaDAGenerationConfig",
    "LLaDAGenerationMixin",
    "LLaDAConfig",
    "ModelConfig",
    "LayerNormType",
    "ActivationType",
    "BlockType",
    "InitFnType",
    "ActivationCheckpointingStrategy",
    "LLaDAPreTrainedModel",
    "LLaDAModel",
    "LLaDAModelLM",
]
