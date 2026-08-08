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

Import from ``unturtle.diffusion`` or re-exports on ``unturtle``.

Public API::

    from unturtle.diffusion import (
        BaseAlphaScheduler, LinearAlphaScheduler, CosineAlphaScheduler,
        make_alpha_scheduler,
        MaskedDiffusionDataCollator,
        DiffusionTrainer, DiffusionTrainingArguments,
        DiffuGRPOTrainer, DiffuGRPOConfig,
        create_block_diffusion_attention_mask,
        BlockDiffusionDataCollator,
        BlockDiffusionTrainer, BlockDiffusionTrainingArguments,
    )
"""

from .collator import MaskedDiffusionDataCollator
from .dfm_loss import discrete_flow_matching_loss
from .packed_collator import PackedMaskedDiffusionDataCollator
from .reweighting import context_adaptive_reweight
from .schedulers import (
    BaseAlphaScheduler,
    CosineAlphaScheduler,
    LinearAlphaScheduler,
    make_alpha_scheduler,
)
from .trainer import DiffusionTrainer, DiffusionTrainingArguments

try:
    from .grpo_trainer import DiffuGRPOConfig, DiffuGRPOTrainer
except ModuleNotFoundError as exc:
    missing_root = (exc.name or "").split(".", 1)[0]
    optional_roots = {"trl", "mergekit", "vllm"}
    if missing_root not in optional_roots:
        raise
    missing_exc = exc

    def _missing_grpo_dependency(name: str):
        class _MissingGRPODependency:
            def __init__(self, *_args, **_kwargs):
                missing_name = (
                    getattr(missing_exc, "name", None) or "an optional dependency"
                )
                raise ModuleNotFoundError(
                    "unturtle DiffuGRPO requires additional optional dependencies. "
                    f"Missing module: {missing_name}. Start from the Hugging Face stack "
                    "(`pip install -e '.[huggingface]'`) and install the GRPO-only "
                    "packages needed for your environment before using "
                    "DiffuGRPOTrainer or DiffuGRPOConfig."
                ) from missing_exc

        _MissingGRPODependency.__name__ = name
        return _MissingGRPODependency

    DiffuGRPOTrainer = _missing_grpo_dependency("DiffuGRPOTrainer")
    DiffuGRPOConfig = _missing_grpo_dependency("DiffuGRPOConfig")
from .block_attention import create_block_diffusion_attention_mask
from .block_diffusion_collator import BlockDiffusionDataCollator
from .block_diffusion_trainer import (
    BlockDiffusionTrainer,
    BlockDiffusionTrainingArguments,
)

__all__ = [
    "discrete_flow_matching_loss",
    "BaseAlphaScheduler",
    "LinearAlphaScheduler",
    "CosineAlphaScheduler",
    "make_alpha_scheduler",
    "MaskedDiffusionDataCollator",
    "PackedMaskedDiffusionDataCollator",
    "context_adaptive_reweight",
    "DiffusionTrainer",
    "DiffusionTrainingArguments",
    "DiffuGRPOTrainer",
    "DiffuGRPOConfig",
    "create_block_diffusion_attention_mask",
    "BlockDiffusionDataCollator",
    "BlockDiffusionTrainer",
    "BlockDiffusionTrainingArguments",
]
