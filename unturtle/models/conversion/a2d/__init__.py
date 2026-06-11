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

"""unturtle.models.conversion.a2d — the AR→diffusion conversion method family.

``a2d`` is the *family* (AutoRegressive→Diffusion conversion). The current implementation
is the **Tiny-A2D** recipe (dLLM paper section 4.2.2: the small Qwen3-0.6B mask-replacement
variant), under ``tiny_a2d``. Future recipes (DiffuLLaMA, RND1) would be sibling
subpackages here. These are thin adapters over ``transformers`` backbones — the backbone
(Qwen/Llama) lives upstream; this code is the *method*.
"""

from .tiny_a2d import (
    TinyA2DGenerationConfig,
    TinyA2DGenerationMixin,
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
    TinyA2DLlamaModel,
    TinyA2DQwen2Config,
    TinyA2DQwen2LMHeadModel,
    TinyA2DQwen2Model,
    TinyA2DQwen3Config,
    TinyA2DQwen3LMHeadModel,
    TinyA2DQwen3Model,
)

#: Supported AR backbones for the Tiny-A2D recipe -> their LM-head classes.
TINY_A2D_MODEL_CLASSES = {
    "llama": TinyA2DLlamaLMHeadModel,
    "qwen2": TinyA2DQwen2LMHeadModel,
    "qwen3": TinyA2DQwen3LMHeadModel,
}

TINY_A2D_CONFIG_CLASSES = {
    "llama": TinyA2DLlamaConfig,
    "qwen2": TinyA2DQwen2Config,
    "qwen3": TinyA2DQwen3Config,
}

__all__ = [
    "TINY_A2D_MODEL_CLASSES",
    "TINY_A2D_CONFIG_CLASSES",
    "TinyA2DGenerationConfig",
    "TinyA2DGenerationMixin",
    "TinyA2DLlamaConfig",
    "TinyA2DLlamaModel",
    "TinyA2DLlamaLMHeadModel",
    "TinyA2DQwen2Config",
    "TinyA2DQwen2Model",
    "TinyA2DQwen2LMHeadModel",
    "TinyA2DQwen3Config",
    "TinyA2DQwen3Model",
    "TinyA2DQwen3LMHeadModel",
]
