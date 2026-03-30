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

"""A2D (AutoRegressive→Diffusion) model adapters.

Lightweight adapters that convert causal LLMs to bidirectional masked
diffusion LMs by removing causal attention masking. All pretrained weights
are preserved without modification.

Supported base architectures:
  - LLaMA (Meta-Llama-3, Meta-Llama-3.1, …)
  - Qwen2 (Qwen/Qwen2.5-*)
  - Qwen3 (Qwen/Qwen3-*)

Usage::

    from unturtle.models.a2d import A2DLlamaConfig, A2DLlamaLMHeadModel
    from unturtle.models.a2d import A2DQwen2Config, A2DQwen2LMHeadModel
    from unturtle.models.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
"""

from .modeling_llama import A2DLlamaConfig, A2DLlamaModel, A2DLlamaLMHeadModel
from .modeling_qwen2 import A2DQwen2Config, A2DQwen2Model, A2DQwen2LMHeadModel
from .modeling_qwen3 import A2DQwen3Config, A2DQwen3Model, A2DQwen3LMHeadModel

__all__ = [
    "A2DLlamaConfig",
    "A2DLlamaModel",
    "A2DLlamaLMHeadModel",
    "A2DQwen2Config",
    "A2DQwen2Model",
    "A2DQwen2LMHeadModel",
    "A2DQwen3Config",
    "A2DQwen3Model",
    "A2DQwen3LMHeadModel",
]
