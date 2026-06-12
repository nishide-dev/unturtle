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

"""unturtle.models.backbones — the *architecture* axis of a dLLM.

A concrete dLLM is a point in three orthogonal axes::

    backbone architecture  ×  conversion method  ×  training objective
    (this package)            (unturtle.models.conversion)  (unturtle.diffusion)

This package is the canonical home for backbone architectures. The native
bidirectional backbones physically live here as subpackages; their heavy
implementations (and their one-time ``transformers.AutoConfig.register`` side
effects) run exactly once on import.

Native bidirectional backbones:
  - LLaDA  (unturtle.models.backbones.llada)
  - Dream  (unturtle.models.backbones.dream)
  - ModernBERT diffusion  (unturtle.models.backbones.modernbert)

Canvas block-diffusion backbones (self-conditioned, not masked-diffusion):
  - DiffusionGemma  (unturtle.models.backbones.diffusion_gemma)

AR backbones reachable via the A2D conversion method live under
``unturtle.models.conversion`` (see unturtle.models.conversion.a2d) — they are a
conversion *method*, not a peer architecture, so they are not re-exported here.
"""

from .diffusion_gemma import UnturtleDiffusionGemmaForBlockDiffusion
from .dream import DreamConfig, DreamModel
from .llada import LLaDAConfig, LLaDAModel, LLaDAModelLM
from .modernbert import (
    DiffusionModernBertConfig,
    DiffusionModernBertForMaskedLM,
    DiffusionModernBertModel,
)

__all__ = [
    "UnturtleDiffusionGemmaForBlockDiffusion",
    "DreamConfig",
    "DreamModel",
    "LLaDAConfig",
    "LLaDAModel",
    "LLaDAModelLM",
    "DiffusionModernBertConfig",
    "DiffusionModernBertModel",
    "DiffusionModernBertForMaskedLM",
]
