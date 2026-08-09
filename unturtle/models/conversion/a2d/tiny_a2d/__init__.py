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

"""Tiny-A2D (AutoRegressive→Diffusion) model adapters.

Lightweight adapters that convert causal LLMs to bidirectional masked
diffusion LMs by removing causal attention masking. All pretrained weights
are preserved without modification. This is the **Tiny-A2D** recipe of the
``a2d`` conversion family (dLLM paper section 4.2.2).

Supported base architectures:
  - LLaMA (Meta-Llama-3, Meta-Llama-3.1, …)
  - Qwen2 (Qwen/Qwen2.5-*)
  - Qwen3 (Qwen/Qwen3-*)

Usage::

    from unturtle.models.conversion.a2d.tiny_a2d import (
        TinyA2DLlamaConfig, TinyA2DLlamaLMHeadModel,
        TinyA2DQwen2Config, TinyA2DQwen2LMHeadModel,
        TinyA2DQwen3Config, TinyA2DQwen3LMHeadModel,
    )
"""

from .generation_utils import TinyA2DGenerationConfig, TinyA2DGenerationMixin
from .loading import ar_head_classes, convert_ar_model, load_tiny_a2d_from_ar
from .modeling_llama import (
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
    TinyA2DLlamaModel,
)
from .modeling_qwen2 import (
    TinyA2DQwen2Config,
    TinyA2DQwen2LMHeadModel,
    TinyA2DQwen2Model,
)
from .modeling_qwen3 import (
    TinyA2DQwen3Config,
    TinyA2DQwen3LMHeadModel,
    TinyA2DQwen3Model,
)

__all__ = [
    "ar_head_classes",
    "convert_ar_model",
    "load_tiny_a2d_from_ar",
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
