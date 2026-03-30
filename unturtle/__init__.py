# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

The internal implementation lives in ``unsloth/`` to keep upstream merges
from unslothai/unsloth conflict-free.  This shim re-exports everything so
that ``import unturtle`` works identically to ``import unsloth``.

Migration path
--------------
Phase A (current): ``unturtle`` is a thin re-export layer over ``unsloth``.
Phase B (future):  dLLM-specific modules will be moved into ``unturtle/``
                   proper as they diverge sufficiently from upstream.
"""

from unsloth import *  # noqa: F401, F403
from unsloth import __version__  # noqa: F401

# Re-export dLLM additions explicitly so IDEs can resolve them
from unsloth.diffusion import (  # noqa: F401
    BaseAlphaScheduler,
    CosineAlphaScheduler,
    DiffusionTrainer,
    DiffusionTrainingArguments,
    DiffuGRPOTrainer,
    DiffuGRPOConfig,
    LinearAlphaScheduler,
    MaskedDiffusionDataCollator,
    make_alpha_scheduler,
)
from unsloth.kernels.masked_diffusion_loss import (  # noqa: F401
    fast_masked_diffusion_loss,
    masked_diffusion_loss_from_timesteps,
)
