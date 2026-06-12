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

"""DiffusionGemma backbone wrapper.

DiffusionGemma is NOT a masked diffusion LM: it denoises a per-block "canvas"
with self-conditioning under entropy/confidence acceptance (no mask token).
This subpackage wraps the upstream ``DiffusionGemmaForBlockDiffusion`` with a
unified ``generate(algorithm=...)`` shim so the Unturtle contract holds.

Usage::

    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )
"""

from .modeling import UnturtleDiffusionGemmaForBlockDiffusion

__all__ = [
    "UnturtleDiffusionGemmaForBlockDiffusion",
]
