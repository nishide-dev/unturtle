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

"""
unturtle — the public-facing package name for this project.

Phase B: dLLM-specific modules now live in ``unturtle/`` as the canonical
source.  ``unsloth.diffusion`` and ``unsloth.kernels.masked_diffusion_loss``
are compatibility shims that re-export from here.

``from unsloth import *`` is kept for all upstream unsloth symbols.
"""

from unsloth import *  # noqa: F401, F403
from unsloth import __version__  # noqa: F401

# dLLM additions — canonical source is unturtle.*
from unturtle.diffusion import (  # noqa: F401
    BaseAlphaScheduler,
    CosineAlphaScheduler,
    DiffusionTrainer,
    DiffusionTrainingArguments,
    DiffuGRPOTrainer,
    DiffuGRPOConfig,
    LinearAlphaScheduler,
    MaskedDiffusionDataCollator,
    PackedMaskedDiffusionDataCollator,
    make_alpha_scheduler,
)
from unturtle.kernels.masked_diffusion_loss import (  # noqa: F401
    fast_masked_diffusion_loss,
    masked_diffusion_loss_from_timesteps,
)
from unturtle.fast_diffusion_model import FastDiffusionModel  # noqa: F401
