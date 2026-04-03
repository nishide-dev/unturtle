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

"""Generation utilities for LLaDA models.

Thin wrappers around the shared MDLM generation kernel.
"""

from unturtle.models.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
    MaskedDiffusionGenerationMixin,
    MaskedDiffusionModelOutput,
)


class LLaDAGenerationConfig(MaskedDiffusionGenerationConfig):
    """Generation config for LLaDA models (currently identical to the shared config)."""
    pass


class LLaDAGenerationMixin(MaskedDiffusionGenerationMixin):
    """Generation mixin for LLaDA models.

    Delegates entirely to :class:`MaskedDiffusionGenerationMixin`.
    Subclass and override :meth:`_sample` here for LLaDA-specific behaviour.
    """
    pass


__all__ = [
    "LLaDAGenerationConfig",
    "LLaDAGenerationMixin",
    "MaskedDiffusionModelOutput",
]
