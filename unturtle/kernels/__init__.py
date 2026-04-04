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

Canonical home for unturtle-specific kernels vendored or extended for unturtle.

Public API::

    from unturtle.kernels import (
        Fast_CrossEntropyLoss,
        LoRA_QKV,
        LoRA_QKV_Bias,
        apply_lora_o,
        apply_lora_qkv,
        apply_lora_qkv_with_bias,
        apply_lora_mlp_swiglu,
        fast_cross_entropy_loss,
        fast_masked_diffusion_loss,
        fast_rope_embedding,
        fused_masked_diffusion_loss,
        masked_diffusion_loss_from_timesteps,
    )
"""

from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss, fast_cross_entropy_loss
from .masked_diffusion_loss import (
    fast_masked_diffusion_loss,
    masked_diffusion_loss_from_timesteps,
)
from .fused_masked_diffusion_loss import fused_masked_diffusion_loss
from unsloth.kernels.rope_embedding import fast_rope_embedding

__all__ = [
    "Fast_CrossEntropyLoss",
    "fast_cross_entropy_loss",
    "fast_masked_diffusion_loss",
    "masked_diffusion_loss_from_timesteps",
    "fused_masked_diffusion_loss",
    "fast_rope_embedding",
]

try:
    from .fast_lora import (
        LoRA_QKV,
        LoRA_QKV_Bias,
        apply_lora_mlp_swiglu,
        apply_lora_o,
        apply_lora_qkv,
        apply_lora_qkv_with_bias,
    )
except (ImportError, OSError, AttributeError):
    # fast_lora has optional bitsandbytes-backed dependencies; keep the rest of
    # unturtle.kernels importable when they are unavailable or partially linked.
    pass
else:
    __all__.extend(
        [
            "LoRA_QKV",
            "LoRA_QKV_Bias",
            "apply_lora_qkv",
            "apply_lora_qkv_with_bias",
            "apply_lora_o",
            "apply_lora_mlp_swiglu",
        ]
    )
