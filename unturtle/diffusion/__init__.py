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

"""unturtle.diffusion — Masked Diffusion Language Model training stack.

This is the canonical package. ``unsloth.diffusion`` is a compatibility
shim that re-exports everything from here.

Public API::

    from unturtle.diffusion import (
        BaseAlphaScheduler, LinearAlphaScheduler, CosineAlphaScheduler,
        make_alpha_scheduler,
        MaskedDiffusionDataCollator,
        DiffusionTrainer, DiffusionTrainingArguments,
        DiffuGRPOTrainer, DiffuGRPOConfig,
    )
"""

from .schedulers import (
    BaseAlphaScheduler,
    LinearAlphaScheduler,
    CosineAlphaScheduler,
    make_alpha_scheduler,
)
from .collator import MaskedDiffusionDataCollator
from .trainer import DiffusionTrainer, DiffusionTrainingArguments
from .grpo_trainer import DiffuGRPOTrainer, DiffuGRPOConfig

__all__ = [
    "BaseAlphaScheduler",
    "LinearAlphaScheduler",
    "CosineAlphaScheduler",
    "make_alpha_scheduler",
    "MaskedDiffusionDataCollator",
    "DiffusionTrainer",
    "DiffusionTrainingArguments",
    "DiffuGRPOTrainer",
    "DiffuGRPOConfig",
]
