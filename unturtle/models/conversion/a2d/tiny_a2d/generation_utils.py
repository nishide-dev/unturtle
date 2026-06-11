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

"""Generation mixin for Tiny-A2D (AutoRegressive→Diffusion) models.

The actual generation logic (block-decode + MDLM + BD3LM) is generic masked-diffusion
behavior shared with native diffusion backbones (e.g. ModernBERT), so it lives in
``unturtle.models.generation.masked_diffusion_block_mixin``. These thin subclasses give
the Tiny-A2D models their own named generation mixin/config.
"""

from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionModelOutput,
)
from unturtle.models.generation.masked_diffusion_block_mixin import (
    MaskedDiffusionBlockGenerationConfig,
    MaskedDiffusionBlockGenerationMixin,
)


class TinyA2DGenerationConfig(MaskedDiffusionBlockGenerationConfig):
    """Generation config for Tiny-A2D models (currently identical to the shared config)."""

    pass


class TinyA2DGenerationMixin(MaskedDiffusionBlockGenerationMixin):
    """Generation mixin for Tiny-A2D models.

    All behavior is inherited from
    :class:`~unturtle.models.generation.masked_diffusion_block_mixin.MaskedDiffusionBlockGenerationMixin`
    (block-decode KV-cache, MDLM no-cache loop, and BD3LM block-diffusion). This subclass
    exists so Tiny-A2D models have a recipe-named mixin.
    """

    pass


__all__ = [
    "TinyA2DGenerationConfig",
    "TinyA2DGenerationMixin",
    "MaskedDiffusionModelOutput",
]
