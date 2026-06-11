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

"""Unturtle — dLLM (Diffusion Language Model) method layer on top of unsloth.

Unturtle is built on top of unsloth and depends on it permanently. We reuse
unsloth's Triton kernels, model loading infrastructure, and optimizers, and
concentrate Unturtle-specific value on diffusion-specific behavior:

- bidirectional attention for Tiny-A2D / Dream / LLaDA / ModernBERT models
- Triton-optimized masked diffusion loss
- completion-only masking for SFT
- dLLM generation utilities (MDLM denoising / block decode)
"""

# ============================================================================
# Re-exported from unsloth (external dependency)
# ============================================================================
from unsloth import (
    FastLanguageModel,
    FastLlamaModel,
    FastMistralModel,
    FastQwen2Model,
    FastQwen3Model,
    PatchDPOTrainer,
    PatchKTOTrainer,
    UnslothTrainer,
    UnslothTrainingArguments,
    is_bf16_supported,
    is_bfloat16_supported,
)
from unsloth.chat_templates import (
    apply_chat_template,
    get_chat_template,
)
from unsloth.save import (
    patch_saving_functions,
    save_to_gguf,
    unsloth_save_model,
)
from unsloth.tokenizer_utils import (
    check_tokenizer,
    fix_sentencepiece_tokenizer,
    get_tokenizer_info,
    load_correct_tokenizer,
)
from unsloth.trainer import (
    QGaloreConfig,
    unsloth_train,
)

from unturtle._version import __version__

# ============================================================================
# Optimizers (re-exported from unsloth)
# ============================================================================
try:
    from unsloth.optimizers import (
        GaLoreProjector,
        QGaLoreAdamW8bit,
        UnslothAdamW,
        UnslothAdamW8bit,
        UnslothAdamWScheduleFree,
    )
except (ImportError, AttributeError):
    # Older unsloth versions may not expose all optimizer symbols.
    GaLoreProjector = None
    QGaLoreAdamW8bit = None
    UnslothAdamW = None
    UnslothAdamW8bit = None
    UnslothAdamWScheduleFree = None

# ============================================================================
# Unturtle-specific dLLM APIs
# ============================================================================
from unturtle.diffusion import (
    BaseAlphaScheduler,
    CosineAlphaScheduler,
    DiffuGRPOConfig,
    DiffuGRPOTrainer,
    DiffusionTrainer,
    DiffusionTrainingArguments,
    LinearAlphaScheduler,
    MaskedDiffusionDataCollator,
    PackedMaskedDiffusionDataCollator,
    make_alpha_scheduler,
)
from unturtle.eval import (
    BaseEvaluator,
    GenerationEvaluator,
    MaskedDiffusionEvaluator,
)
from unturtle.fast_diffusion_model import FastDiffusionModel
from unturtle.models import (
    A2DModernBertConfig,
    A2DModernBertForMaskedLM,
    A2DModernBertModel,
    DiffusionModernBertConfig,
    DiffusionModernBertForMaskedLM,
    DiffusionModernBertModel,
    DreamConfig,
    DreamGenerationConfig,
    DreamGenerationMixin,
    DreamModel,
    LLaDAConfig,
    LLaDAModel,
    LLaDAModelLM,
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
    TinyA2DLlamaModel,
    TinyA2DQwen2Config,
    TinyA2DQwen2LMHeadModel,
    TinyA2DQwen2Model,
    TinyA2DQwen3Config,
    TinyA2DQwen3LMHeadModel,
    TinyA2DQwen3Model,
)
from unturtle.trainer import (
    UnturtleTrainer,
    UnturtleTrainingArguments,
    unturtle_train,
)

DreamForDiffusionLM = DreamModel  # Alias for backward compatibility

# Kernels (advanced users may import directly)
from unturtle.kernels.fused_masked_diffusion_loss import (  # noqa: E402
    fused_masked_diffusion_loss,
)
from unturtle.kernels.masked_diffusion_loss import (  # noqa: E402
    fast_masked_diffusion_loss,
    masked_diffusion_loss_from_timesteps,
)

__all__ = [
    # Version
    "__version__",
    # Re-exported from unsloth (core)
    "FastLanguageModel",
    "UnslothTrainer",
    "UnslothTrainingArguments",
    "is_bfloat16_supported",
    "is_bf16_supported",
    # Re-exported from unsloth (models)
    "FastLlamaModel",
    "FastMistralModel",
    "FastQwen2Model",
    "FastQwen3Model",
    # Re-exported from unsloth (trainer patches)
    "PatchDPOTrainer",
    "PatchKTOTrainer",
    # Re-exported from unsloth (tokenizer)
    "check_tokenizer",
    "fix_sentencepiece_tokenizer",
    "load_correct_tokenizer",
    "get_tokenizer_info",
    # Re-exported from unsloth (chat templates)
    "apply_chat_template",
    "get_chat_template",
    # Re-exported from unsloth (save/export)
    "patch_saving_functions",
    "unsloth_save_model",
    "save_to_gguf",
    # Re-exported from unsloth (training)
    "unsloth_train",
    "QGaloreConfig",
    # Re-exported from unsloth (optimizers)
    "GaLoreProjector",
    "QGaLoreAdamW8bit",
    "UnslothAdamW",
    "UnslothAdamW8bit",
    "UnslothAdamWScheduleFree",
    # dLLM training
    "DiffusionTrainer",
    "DiffusionTrainingArguments",
    "MaskedDiffusionDataCollator",
    "PackedMaskedDiffusionDataCollator",
    "DiffuGRPOConfig",
    "DiffuGRPOTrainer",
    # alpha schedulers
    "LinearAlphaScheduler",
    "CosineAlphaScheduler",
    "BaseAlphaScheduler",
    "make_alpha_scheduler",
    # trainers
    "UnturtleTrainer",
    "UnturtleTrainingArguments",
    "unturtle_train",
    # evaluators
    "BaseEvaluator",
    "GenerationEvaluator",
    "MaskedDiffusionEvaluator",
    # fast model wrapper
    "FastDiffusionModel",
    # dLLM models
    "TinyA2DLlamaConfig",
    "TinyA2DLlamaModel",
    "TinyA2DLlamaLMHeadModel",
    "TinyA2DQwen2Config",
    "TinyA2DQwen2Model",
    "TinyA2DQwen2LMHeadModel",
    "TinyA2DQwen3Config",
    "TinyA2DQwen3Model",
    "TinyA2DQwen3LMHeadModel",
    "A2DModernBertConfig",
    "A2DModernBertForMaskedLM",
    "A2DModernBertModel",
    "DiffusionModernBertConfig",
    "DiffusionModernBertForMaskedLM",
    "DiffusionModernBertModel",
    "LLaDAConfig",
    "LLaDAModel",
    "LLaDAModelLM",
    "DreamConfig",
    "DreamModel",
    "DreamGenerationMixin",
    "DreamGenerationConfig",
    "DreamForDiffusionLM",
    # kernels
    "fast_masked_diffusion_loss",
    "masked_diffusion_loss_from_timesteps",
    "fused_masked_diffusion_loss",
]
