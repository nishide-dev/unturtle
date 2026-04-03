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

"""A2D (AutoRegressive→Diffusion) adapter for ModernBERT models.

ModernBERT is already a bidirectional encoder, so no attention mask surgery is
needed. This adapter adds ``A2DGenerationMixin`` to enable dLLM generation and
registers a distinct ``model_type`` so that checkpoints fine-tuned with unturtle
can be round-tripped via AutoModel without colliding with the upstream
``"modernbert"`` type.

Usage::

    from unturtle.models.a2d import A2DModernBertConfig, A2DModernBertForMaskedLM

    config = A2DModernBertConfig(
        vocab_size=50368,
        hidden_size=768,
        intermediate_size=1152,
        num_hidden_layers=4,
        num_attention_heads=12,
    )
    model = A2DModernBertForMaskedLM(config)
    # fine-tune with DiffusionTrainer
"""

import transformers
from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertModel

from .generation_utils import A2DGenerationMixin


class A2DModernBertConfig(ModernBertConfig):
    model_type = "a2d-modernbert"


class A2DModernBertModel(ModernBertModel):
    """ModernBertModel with A2D model_type.

    ModernBERT is already bidirectional — no forward override is needed.
    This subclass exists only to expose the correct ``config_class``.
    """

    config_class = A2DModernBertConfig


class A2DModernBertForMaskedLM(A2DGenerationMixin, ModernBertForMaskedLM):
    """ModernBERT masked-LM head wrapped for dLLM use.

    Inherits the full ``ModernBertForMaskedLM`` implementation plus
    ``A2DGenerationMixin`` for MDLM denoising generation.
    """

    config_class = A2DModernBertConfig

    def __init__(self, config: A2DModernBertConfig):
        super().__init__(config)
        # Replace ModernBertModel with A2DModernBertModel so config_class is correct.
        self.model = A2DModernBertModel(config)


transformers.AutoConfig.register("a2d-modernbert", A2DModernBertConfig)
transformers.AutoModel.register(A2DModernBertConfig, A2DModernBertForMaskedLM)
transformers.AutoModelForMaskedLM.register(A2DModernBertConfig, A2DModernBertForMaskedLM)
