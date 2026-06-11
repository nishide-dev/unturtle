"""Unturtle — dLLM method layer on top of unsloth."""

from unturtle._version import __version__
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
from unturtle.kernels.fused_masked_diffusion_loss import fused_masked_diffusion_loss
from unturtle.kernels.masked_diffusion_loss import (
    fast_masked_diffusion_loss,
    masked_diffusion_loss_from_timesteps,
)
from unturtle.trainer import (
    UnturtleTrainer,
    UnturtleTrainingArguments,
    unturtle_train,
)

__all__ = [
    "__version__",
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
    # kernels
    "fast_masked_diffusion_loss",
    "masked_diffusion_loss_from_timesteps",
    "fused_masked_diffusion_loss",
]
