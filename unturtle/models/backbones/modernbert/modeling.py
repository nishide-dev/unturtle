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

"""ModernBERT diffusion model.

ModernBERT is already a bidirectional encoder, so no attention mask surgery is
needed. This module adds ``MaskedDiffusionBlockGenerationMixin`` to enable dLLM generation and
registers a distinct ``model_type`` so that checkpoints fine-tuned with unturtle
can be round-tripped via AutoModel without colliding with the upstream
``"modernbert"`` type.

Usage::

    from unturtle.models.backbones.modernbert import (
        DiffusionModernBertConfig,
        DiffusionModernBertForMaskedLM,
    )

    config = DiffusionModernBertConfig(
        vocab_size=50368,
        hidden_size=768,
        intermediate_size=1152,
        num_hidden_layers=4,
        num_attention_heads=12,
    )
    model = DiffusionModernBertForMaskedLM(config)
    # fine-tune with DiffusionTrainer
"""

import contextlib

import transformers
from transformers import ModernBertConfig, ModernBertForMaskedLM, ModernBertModel

from ...generation.masked_diffusion_block_mixin import (
    MaskedDiffusionBlockGenerationMixin,
)
from .configuration import DiffusionModernBertConfig


class DiffusionModernBertModel(ModernBertModel):
    """ModernBertModel with diffusion model_type.

    ModernBERT is already bidirectional — no forward override is needed.
    This subclass exists only to expose the correct ``config_class``.
    """

    config_class = DiffusionModernBertConfig


class DiffusionModernBertForMaskedLM(
    MaskedDiffusionBlockGenerationMixin, ModernBertForMaskedLM
):
    """ModernBERT masked-LM head wrapped for dLLM use.

    Inherits the full ``ModernBertForMaskedLM`` implementation plus
    ``MaskedDiffusionBlockGenerationMixin`` for MDLM denoising generation.
    """

    config_class = DiffusionModernBertConfig
    # Encoder backbone returns no past_key_values; block-decode cache hook unusable.
    supports_block_decode = False

    def __init__(self, config: DiffusionModernBertConfig):
        super().__init__(config)
        # Replace ModernBertModel with DiffusionModernBertModel so config_class is correct.
        # tie_weights() must be called after the swap to restore the decoder↔embedding tie
        # that post_init() established against the original ModernBertModel instance.
        self.model = DiffusionModernBertModel(config)
        self.tie_weights()


# Backward compat aliases for the old A2D-prefixed names.
A2DModernBertConfig = DiffusionModernBertConfig
A2DModernBertModel = DiffusionModernBertModel
A2DModernBertForMaskedLM = DiffusionModernBertForMaskedLM


# Register the diffusion-specific ModernBERT model_type.
with contextlib.suppress(ValueError):
    transformers.AutoConfig.register("modernbert-diffusion", DiffusionModernBertConfig)

# Guard AutoModel registrations against re-import.
with contextlib.suppress(ValueError):
    transformers.AutoModel.register(
        DiffusionModernBertConfig, DiffusionModernBertForMaskedLM
    )
with contextlib.suppress(ValueError):
    transformers.AutoModelForMaskedLM.register(
        DiffusionModernBertConfig, DiffusionModernBertForMaskedLM
    )
