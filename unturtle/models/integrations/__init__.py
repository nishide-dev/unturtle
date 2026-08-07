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

"""unturtle.models.integrations — per-model-family loading knowledge (#68).

Internal.  Keeps ``model_type``-specific loading decisions out of the central
``FastDiffusionModel`` orchestration path so a new family is a registration
rather than another branch.  Not re-exported from top-level ``unturtle``.
"""

from .base import BackboneIntegration
from .registry import (
    find_integration,
    iter_integrations,
    native_model_classes,
    post_load_class_swaps,
    register_integration,
    resolve_native_class,
    resolve_post_load_wrapper,
)

__all__ = [
    "BackboneIntegration",
    "find_integration",
    "iter_integrations",
    "native_model_classes",
    "post_load_class_swaps",
    "register_integration",
    "resolve_native_class",
    "resolve_post_load_wrapper",
]
