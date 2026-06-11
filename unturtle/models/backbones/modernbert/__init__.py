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

"""ModernBERT diffusion model adapters.

ModernBERT is a native bidirectional encoder (BERT-family), so no
AR→Diffusion attention surgery is needed. This package adds
``MaskedDiffusionBlockGenerationMixin`` for dLLM generation utilities and registers a
distinct ``model_type`` so that fine-tuned checkpoints round-trip via
AutoModel without colliding with the upstream ``"modernbert"`` type.

Usage::

    from unturtle.models.backbones.modernbert import (
        DiffusionModernBertConfig,
        DiffusionModernBertModel,
        DiffusionModernBertForMaskedLM,
    )
"""

from .modeling import (
    A2DModernBertConfig,
    A2DModernBertForMaskedLM,
    A2DModernBertModel,
    DiffusionModernBertConfig,
    DiffusionModernBertForMaskedLM,
    DiffusionModernBertModel,
)

__all__ = [
    "A2DModernBertConfig",
    "A2DModernBertForMaskedLM",
    "A2DModernBertModel",
    "DiffusionModernBertConfig",
    "DiffusionModernBertForMaskedLM",
    "DiffusionModernBertModel",
]
