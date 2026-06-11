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

"""unturtle.models.conversion — the *method* axis of a dLLM.

Conversion methods turn a non-diffusion backbone into a dLLM. Currently:
  - a2d: AutoRegressive→Diffusion family; the implemented recipe is Tiny-A2D
    (see unturtle.models.conversion.a2d.tiny_a2d)
"""

from .a2d import TINY_A2D_CONFIG_CLASSES, TINY_A2D_MODEL_CLASSES

__all__ = ["TINY_A2D_MODEL_CLASSES", "TINY_A2D_CONFIG_CLASSES"]
