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

"""unturtle.kernels — Triton-optimised kernels for dLLM training.

Canonical home for unturtle-specific kernels.  The underlying CE kernel
(``Fast_CrossEntropyLoss``) remains in ``unsloth.kernels`` to stay in sync
with upstream unslothai/unsloth.

Public API::

    from unturtle.kernels.masked_diffusion_loss import (
        fast_masked_diffusion_loss,
        masked_diffusion_loss_from_timesteps,
    )
"""

from .masked_diffusion_loss import (
    fast_masked_diffusion_loss,
    masked_diffusion_loss_from_timesteps,
)
from .fast_lora import (
    LoRA_QKV_Bias,
    apply_lora_qkv_with_bias,
)

__all__ = [
    "fast_masked_diffusion_loss",
    "masked_diffusion_loss_from_timesteps",
    "LoRA_QKV_Bias",
    "apply_lora_qkv_with_bias",
]
