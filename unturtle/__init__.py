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

Current state (Phase B/C):
  dLLM-specific modules live in ``unturtle/`` as the canonical source.
  ``unsloth.diffusion`` and ``unsloth.kernels.masked_diffusion_loss`` are
  compatibility shims that re-export from here.
  ``from unsloth import *`` is kept so upstream unsloth symbols (FastLanguageModel,
  UnslothTrainer, etc.) remain accessible without an extra import.

Target state (Phase Z — full migration, no unsloth dependency):
  The long-term goal is to remove the ``from unsloth import *`` wildcard and
  re-implement (or vendor) everything unturtle needs directly:
    - Triton kernels (cross_entropy_loss, rope_embedding, fast_lora, …)
    - FastLanguageModel → replaced by FastDiffusionModel
    - UnslothTrainer / UnslothTrainingArguments → DiffusionTrainer already wraps these;
      eventually DiffusionTrainer should stand alone without the unsloth base.
    - Attention dispatch utilities (run_attention, select_attention_backend, …)
    - Packing utilities (get_packed_info_from_kwargs, …)
  Until that migration is complete, unsloth remains a required dependency and all
  upstream symbols are re-exported here for backwards compatibility.
  Track progress: each file that no longer needs unsloth should be noted in
  CLAUDE.md §「unsloth 完全移行チェックリスト」when it is decoupled.
"""

from unsloth import *  # noqa: F401, F403  # TODO(Phase Z): remove after full migration
from unsloth import __version__  # noqa: F401

# ---------------------------------------------------------------------------
# dLLM training stack — canonical source is unturtle.*
# ---------------------------------------------------------------------------
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
from unturtle.kernels.fused_masked_diffusion_loss import (  # noqa: F401
    fused_masked_diffusion_loss,
)
from unturtle.fast_diffusion_model import FastDiffusionModel  # noqa: F401

# ---------------------------------------------------------------------------
# dLLM model classes
# ---------------------------------------------------------------------------
from unturtle.models import (  # noqa: F401
    A2DLlamaConfig,
    A2DLlamaModel,
    A2DLlamaLMHeadModel,
    A2DQwen2Config,
    A2DQwen2Model,
    A2DQwen2LMHeadModel,
    A2DQwen3Config,
    A2DQwen3Model,
    A2DQwen3LMHeadModel,
    LLaDAConfig,
    LLaDAModel,
    LLaDAModelLM,
    DreamConfig,
    DreamModel,
    DreamGenerationMixin,
    DreamGenerationConfig,
)
# Alias for discoverability — DreamModel is the full masked-diffusion model
DreamForDiffusionLM = DreamModel  # noqa: F401

# ---------------------------------------------------------------------------
# Optimizers
# Re-export unsloth optimizers when available so users can write
#   from unturtle import UnslothAdamW
# instead of reaching into unsloth directly.
# TODO(Phase Z): once unsloth dependency is removed, vendor or reimplement
#   these optimizers inside unturtle.optimizers and update these imports.
# ---------------------------------------------------------------------------
try:
    from unsloth.optimizers import (  # noqa: F401
        UnslothAdamW,
        UnslothAdamW8bit,
        UnslothAdamWScheduleFree,
    )
except (ImportError, AttributeError):
    pass  # Older unsloth versions that do not expose these symbols
