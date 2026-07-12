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

"""Pydantic configuration models for unturtle.cli.

Relationship to ``dev/repos/unsloth`` ``unsloth_cli``:

- Upstream uses a flatter root config (e.g. ``model: str``, ``training.max_seq_length``).
- Unturtle nests model loading under ``model:`` (``model.model``, ``model_type``, …) for
  :class:`~unturtle.fast_diffusion_model.FastDiffusionModel`, adds ``diffusion:`` for dLLM
  SFT, and may include GRPO-oriented sections in example YAMLs under ``examples/configs/``.
- ``unturtle train`` uses ``FastDiffusionModel`` / ``DiffusionTrainer`` (and optionally
  ``DiffuGRPOTrainer``); it does **not** call Studio's ``UnslothTrainer`` from upstream CLI.

Config hierarchy::

    Config
    ├── model:    ModelConfig      — model loading / quantization
    ├── data:     DataConfig       — dataset sources
    ├── training: TrainingConfig   — task (sft/grpo), LR, batch, checkpoints
    ├── diffusion: DiffusionConfig — dLLM-specific training options
    ├── lora:     LoraConfig       — LoRA adapter settings
    ├── logging:  LoggingConfig    — W&B / TensorBoard / HF tokens
    └── grpo:     GRPOConfig       — DiffuGRPO settings (optional)

YAML/JSON config files use the nested key names (e.g. ``training.num_epochs``).
CLI flags are auto-generated from the flattened field names by
:func:`~unturtle.cli.options.add_options_from_config`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


def _grpo_effective_world_size() -> int:
    """Process count for TRL ``generation_batch_size`` (per-rank batch × world)."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return max(1, dist.get_world_size())
    except Exception:
        pass
    for key in ("WORLD_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        v = os.environ.get(key)
        if v:
            try:
                w = int(v)
                if w >= 1:
                    return w
            except ValueError:
                continue
    return 1


# ---------------------------------------------------------------------------
# Sub-config sections
# ---------------------------------------------------------------------------


class _SectionConfig(BaseModel):
    """Base for config sections: assignments are validated immediately.

    Without ``validate_assignment``, CLI overrides applied via ``setattr`` in
    :meth:`Config.apply_overrides` would bypass pydantic validation entirely —
    an invalid ``--task`` / ``--model-type`` would only fail late (or never).
    """

    model_config = ConfigDict(validate_assignment=True)


class ModelConfig(_SectionConfig):
    model: Optional[str] = Field(
        default=None,
        description="HuggingFace model ID or local path.",
    )
    model_type: Literal["auto", "a2d", "llada", "dream"] = Field(
        default="auto",
        description=(
            "Hint for FastDiffusionModel auto-detection: "
            "'auto' uses HF AutoConfig; 'a2d'/'llada'/'dream' are explicit."
        ),
    )
    max_seq_length: int = Field(default=2048, description="Maximum sequence length.")
    load_in_4bit: bool = Field(
        default=True, description="Load model in 4-bit quantisation."
    )


class DataConfig(_SectionConfig):
    dataset: Optional[str] = Field(default=None, description="HuggingFace dataset ID.")
    local_dataset: Optional[List[str]] = Field(
        default=None, description="Paths to local JSONL/JSON dataset files."
    )
    dataset_text_field: str = Field(
        default="text", description="Column name that holds the training text."
    )


class TrainingConfig(_SectionConfig):
    task: Literal["sft", "grpo"] = Field(
        default="sft",
        description="Training task: masked diffusion SFT or Diffu-GRPO / wd1 RL.",
    )
    training_type: Literal["lora", "full"] = Field(
        default="lora", description="Training mode: 'lora' (adapter) or 'full'."
    )
    output_dir: str = Field(
        default="./outputs", description="Directory to save checkpoints."
    )
    num_epochs: int = Field(default=3, description="Number of training epochs.")
    learning_rate: float = Field(default=2e-4, description="Peak learning rate.")
    batch_size: int = Field(default=2, description="Per-device training batch size.")
    gradient_accumulation_steps: int = Field(
        default=4, description="Gradient accumulation steps."
    )
    warmup_steps: int = Field(
        default=5, description="Number of warmup steps for the LR scheduler."
    )
    max_steps: int = Field(
        default=0, description="Maximum training steps; 0 means use num_epochs."
    )
    save_steps: int = Field(
        default=0, description="Save checkpoint every N steps; 0 to disable."
    )
    weight_decay: float = Field(default=0.01, description="AdamW weight decay.")
    random_seed: int = Field(default=3407, description="Random seed.")
    packing: bool = Field(
        default=False, description="Enable sample packing for efficiency."
    )
    gradient_checkpointing: Literal["unsloth", "true", "none"] = Field(
        default="unsloth",
        description="Gradient checkpointing mode ('unsloth' recommended).",
    )


class DiffusionConfig(_SectionConfig):
    """dLLM-specific training options — maps to DiffusionTrainingArguments."""

    alpha_scheduler: Literal["linear", "cosine"] = Field(
        default="linear",
        description="Alpha scheduler for masked diffusion noise: 'linear' or 'cosine'.",
    )
    time_epsilon: float = Field(
        default=1e-3,
        description="Minimum sampled timestep (avoids t→0 degenerate case).",
    )
    loss_weight_type: Literal["uniform", "timestep", "scheduler"] = Field(
        default="uniform",
        description=(
            "Per-token loss weighting: "
            "'uniform' (LLaDA/MDLM default), "
            "'timestep' (d1 SFT 1/t weighting), "
            "'scheduler' (MDLM paper w(t))."
        ),
    )
    completion_only: bool = Field(
        default=True,
        description="Only mask completion tokens; skip the prompt.",
    )


class LoraConfig(_SectionConfig):
    lora_r: int = Field(default=64, description="LoRA rank.")
    lora_alpha: int = Field(default=16, description="LoRA alpha scaling factor.")
    lora_dropout: float = Field(default=0.0, description="LoRA dropout probability.")
    target_modules: str = Field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        description="Comma-separated list of module names to apply LoRA to.",
    )
    use_rslora: bool = Field(
        default=False, description="Use Rank-Stabilised LoRA (rsLoRA)."
    )


class LoggingConfig(_SectionConfig):
    enable_wandb: bool = Field(
        default=False, description="Enable Weights & Biases logging."
    )
    wandb_project: str = Field(
        default="unturtle-training", description="W&B project name."
    )
    wandb_token: Optional[str] = Field(
        default=None,
        description="W&B API key (overrides WANDB_API_KEY env var).",
    )
    enable_tensorboard: bool = Field(
        default=False, description="Enable TensorBoard logging."
    )
    tensorboard_dir: str = Field(
        default="runs", description="TensorBoard log directory."
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace token (overrides HF_TOKEN env var).",
    )


class GRPOConfig(_SectionConfig):
    """DiffuGRPO / wd1 settings — used when ``training.task == \"grpo\"``."""

    diffusion_steps: int = Field(
        default=128, description="Number of diffusion denoising steps during rollout."
    )
    block_length: int = Field(
        default=32, description="Block decode length for DiffuGRPO rollouts."
    )
    mask_id: Optional[int] = Field(
        default=None,
        description="Mask token ID override; resolved from tokenizer if None.",
    )
    cfg_scale: float = Field(
        default=0.0,
        description="Classifier-free guidance scale (0 = disabled).",
    )
    remasking: Literal["random", "low_confidence"] = Field(
        default="random",
        description="Remasking strategy for DiffuGRPO rollouts.",
    )
    num_generations: int = Field(
        default=8,
        ge=2,
        description="GRPO group size (completions per prompt).",
    )
    max_completion_length: int = Field(
        default=128,
        description="Maximum generated completion length (tokens).",
    )
    max_prompt_length: Optional[int] = Field(
        default=None,
        description="Trim prompts to this many tokens from the left; None = no trim.",
    )
    num_iterations: int = Field(
        default=1,
        ge=1,
        description="Inner GRPO iterations (μ); >1 reuses rollouts with TRL buffering.",
    )
    beta: float = Field(
        default=0.0,
        description="KL coefficient vs reference policy (0 disables).",
    )
    p_mask_prompt: float = Field(
        default=0.3,
        description="Probability of masking prompt tokens in log-prob computation.",
    )
    random_masking: bool = Field(
        default=True,
        description="Random mask seeds per GRPO iteration.",
    )
    diffu_policy_objective: Literal["grpo", "wd1", "wd1++"] = Field(
        default="grpo",
        description="Loss: grpo, wd1 (Eq. 8–9), or wd1++ (Eq. 10 MC at denoise snapshots).",
    )
    wd1_psi: float = Field(
        default=1.0, description="ψ for wd1 softmax weights (Eq. 9)."
    )
    scale_rewards: bool = Field(
        default=False,
        description="If True, scale advantages by per-group std (TRL 'group'); False = d1-style unscaled.",
    )
    generation_batch_size: Optional[int] = Field(
        default=None,
        description="Rollout micro-batch size; None = derive from batch size and grad accumulation (TRL).",
    )
    builtin_reward: Literal["length", "constant_one"] = Field(
        default="length",
        description="Built-in reward for CLI runs (no custom Python reward).",
    )


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)

    def apply_overrides(self, **kwargs) -> None:
        """Merge CLI flag overrides into the config.

        Each key is matched against the config sections in order.
        Values that are ``None`` are skipped (flag not provided).
        """
        sections = [
            self.model,
            self.data,
            self.training,
            self.diffusion,
            self.lora,
            self.logging,
            self.grpo,
        ]

        for key, value in kwargs.items():
            if value is None:
                continue
            for section in sections:
                if hasattr(section, key):
                    try:
                        # _SectionConfig has validate_assignment=True, so bad
                        # values raise here instead of failing late/silently.
                        setattr(section, key, value)
                    except ValidationError as e:
                        flag = "--" + key.replace("_", "-")
                        raise ValueError(
                            f"Invalid value for {flag}: {value!r}\n{e}"
                        ) from e
                    break

    def training_args_kwargs(self) -> dict:
        """Return kwargs suitable for DiffusionTrainingArguments."""
        return {
            "output_dir": self.training.output_dir,
            "num_train_epochs": self.training.num_epochs,
            "learning_rate": self.training.learning_rate,
            "per_device_train_batch_size": self.training.batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "warmup_steps": self.training.warmup_steps,
            "max_steps": self.training.max_steps if self.training.max_steps > 0 else -1,
            "save_steps": self.training.save_steps
            if self.training.save_steps > 0
            else 500,
            "weight_decay": self.training.weight_decay,
            "seed": self.training.random_seed,
            "gradient_checkpointing": self.training.gradient_checkpointing != "none",
            # dLLM-specific
            "alpha_scheduler": self.diffusion.alpha_scheduler,
            "time_epsilon": self.diffusion.time_epsilon,
            "loss_weight_type": self.diffusion.loss_weight_type,
            "completion_only": self.diffusion.completion_only,
        }

    def lora_kwargs(self) -> dict:
        """Return kwargs for FastDiffusionModel.get_peft_model()."""
        target_modules = [
            m.strip() for m in str(self.lora.target_modules).split(",") if m.strip()
        ]
        return {
            "r": self.lora.lora_r,
            "lora_alpha": self.lora.lora_alpha,
            "lora_dropout": self.lora.lora_dropout,
            "target_modules": target_modules,
            "use_rslora": self.lora.use_rslora,
        }

    def build_diffu_grpo_config(
        self,
        *,
        mask_token_id: int,
        report_to: str,
        logging_dir: Optional[str] = None,
    ):
        """Instantiate :class:`~unturtle.diffusion.DiffuGRPOConfig` from this file + tokenizer mask id.

        When ``grpo.generation_batch_size`` is omitted, derives a TRL-valid batch from
        ``training.batch_size × world_size × gradient_accumulation_steps`` (rounded up to
        a multiple of ``num_generations``). ``world_size`` comes from an initialized
        ``torch.distributed`` process group if available, else ``WORLD_SIZE`` /
        ``SLURM_NTASKS`` / ``OMPI_COMM_WORLD_SIZE``, else ``1``.
        """
        from unturtle.diffusion import DiffuGRPOConfig

        t, g = self.training, self.grpo
        save_strategy = "steps" if t.save_steps > 0 else "no"
        save_steps_val = t.save_steps if t.save_steps > 0 else 500
        max_steps = t.max_steps if t.max_steps > 0 else -1
        logging_steps = max(1, min(50, save_steps_val // 2)) if t.save_steps > 0 else 10

        kwargs: dict = {
            "output_dir": t.output_dir,
            "num_train_epochs": t.num_epochs,
            "max_steps": max_steps,
            "learning_rate": t.learning_rate,
            "per_device_train_batch_size": t.batch_size,
            "gradient_accumulation_steps": t.gradient_accumulation_steps,
            "warmup_steps": t.warmup_steps,
            "save_strategy": save_strategy,
            "save_steps": save_steps_val,
            "weight_decay": t.weight_decay,
            "seed": t.random_seed,
            "logging_steps": logging_steps,
            "report_to": report_to,
            "bf16": False,
            "fp16": False,
            "gradient_checkpointing": t.gradient_checkpointing != "none",
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "num_generations": g.num_generations,
            "max_completion_length": g.max_completion_length,
            "block_length": g.block_length,
            "diffusion_steps": g.diffusion_steps,
            "cfg_scale": g.cfg_scale,
            "remasking": g.remasking,
            "mask_id": int(g.mask_id) if g.mask_id is not None else int(mask_token_id),
            "p_mask_prompt": g.p_mask_prompt,
            "beta": g.beta,
            "num_iterations": g.num_iterations,
            "scale_rewards": g.scale_rewards,
            "diffu_policy_objective": g.diffu_policy_objective,
            "wd1_psi": g.wd1_psi,
            "random_masking": g.random_masking,
        }
        if g.max_prompt_length is not None:
            kwargs["max_prompt_length"] = g.max_prompt_length
        if g.generation_batch_size is not None:
            kwargs["generation_batch_size"] = g.generation_batch_size
        else:
            # Pre-buffer enough rollouts for all grad-accum microbatches in one round
            # (per-device batch × world × gradient_accumulation_steps), then round up to
            # a multiple of num_generations. Not the same as TRL's steps_per_generation.
            raw = (
                t.batch_size
                * _grpo_effective_world_size()
                * t.gradient_accumulation_steps
            )
            gmul = g.num_generations
            kwargs["generation_batch_size"] = max(gmul, math.ceil(raw / gmul) * gmul)
        if logging_dir and report_to == "tensorboard":
            kwargs["logging_dir"] = logging_dir
        return DiffuGRPOConfig(**kwargs)


# ---------------------------------------------------------------------------
# Shared CLI object builders
# ---------------------------------------------------------------------------


def build_masked_diffusion_collator(
    tokenizer,
    model=None,
    *,
    alpha_scheduler: str = "linear",
    time_epsilon: float = 1e-3,
    completion_only: bool = True,
    mask_token_id: Optional[int] = None,
):
    """Build a ``MaskedDiffusionDataCollator`` the way ``DiffusionTrainer`` does.

    Mirrors the collator injection in :class:`~unturtle.diffusion.DiffusionTrainer`
    (``trainer.py``): the alpha scheduler is instantiated from its name via
    :func:`~unturtle.diffusion.make_alpha_scheduler`, ``time_epsilon`` is passed
    through, and ``mask_token_id`` falls back to ``tokenizer.mask_token_id`` then
    ``model.config.mask_token_id`` (real checkpoints may only carry the mask id
    on the model config).

    Used by ``unturtle train`` and ``unturtle eval`` when constructing the
    collator explicitly, so ``--alpha-scheduler`` / ``--time-epsilon`` /
    ``--mask-token-id`` are honored instead of silently using defaults.
    """
    from unturtle.diffusion import MaskedDiffusionDataCollator, make_alpha_scheduler

    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(getattr(model, "config", None), "mask_token_id", None)

    return MaskedDiffusionDataCollator(
        tokenizer=tokenizer,
        scheduler=make_alpha_scheduler(alpha_scheduler),
        mask_token_id=mask_token_id,
        time_epsilon=time_epsilon,
        completion_only=completion_only,
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(path: Optional[Path]) -> Config:
    """Load Config from a YAML or JSON file; return defaults if path is None."""
    if not path:
        return Config()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")

    return Config(**data)
