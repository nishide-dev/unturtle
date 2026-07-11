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

"""unturtle train — masked diffusion language model training."""

from pathlib import Path
from typing import Callable, List, Optional

import typer

from unturtle.cli.config import Config, load_config
from unturtle.cli.options import add_options_from_config


def _builtin_grpo_reward(name: str) -> Callable[..., List[float]]:
    if name == "length":

        def _fn(prompts, completions, **kw):  # noqa: ANN001
            return [float(len(str(c))) for c in completions]

        return _fn
    if name == "constant_one":

        def _fn(prompts, completions, **kw):  # noqa: ANN001
            return [1.0] * len(completions)

        return _fn
    raise ValueError(f"Unknown grpo.builtin_reward: {name!r}")


def _dataset_for_grpo(train_dataset, dataset_text_field: str):
    """Each row must expose a ``prompt`` string for TRL GRPO (or map from *dataset_text_field*)."""
    cols = train_dataset.column_names
    if "prompt" in cols:
        to_drop = [c for c in cols if c != "prompt"]
        return train_dataset.remove_columns(to_drop) if to_drop else train_dataset
    if dataset_text_field not in cols:
        typer.echo(
            f"Error: dataset has no 'prompt' column and no {dataset_text_field!r} "
            f"(columns: {cols})",
            err=True,
        )
        raise typer.Exit(code=2)

    def _row(ex):
        return {"prompt": ex[dataset_text_field]}

    return train_dataset.map(_row, remove_columns=cols)


def _resolve_model_class(model_type: str):
    if model_type == "auto":
        return None

    from unturtle import DreamModel, LLaDAModelLM, TinyA2DLlamaLMHeadModel

    model_classes = {
        "a2d": TinyA2DLlamaLMHeadModel,
        "dream": DreamModel,
        "llada": LLaDAModelLM,
    }
    return model_classes[model_type]


@add_options_from_config(Config)
def train(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML/JSON config file. CLI flags override config values.",
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        "--hf-token",
        envvar="HF_TOKEN",
        help="HuggingFace token.",
    ),
    wandb_token: Optional[str] = typer.Option(
        None,
        "--wandb-token",
        envvar="WANDB_API_KEY",
        help="Weights & Biases API key.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print resolved config as YAML and exit without training.",
    ),
    config_overrides: dict = None,
):
    """Launch dLLM training (SFT via DiffusionTrainer or RL via DiffuGRPOTrainer)."""
    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    cfg.apply_overrides(**config_overrides)

    # CLI/env tokens override config file values
    from typer.models import OptionInfo

    if isinstance(hf_token, OptionInfo):
        hf_token = None
    if isinstance(wandb_token, OptionInfo):
        wandb_token = None
    hf_token = hf_token or cfg.logging.hf_token
    wandb_token = wandb_token or cfg.logging.wandb_token

    if dry_run:
        import yaml

        data = cfg.model_dump()
        typer.echo(yaml.dump(data, default_flow_style=False, sort_keys=False))
        raise typer.Exit(code=0)

    if not cfg.model.model:
        typer.echo("Error: provide --model or set model.model in --config", err=True)
        raise typer.Exit(code=2)

    if not cfg.data.dataset and not cfg.data.local_dataset:
        typer.echo(
            "Error: provide --dataset or --local-dataset (or via --config)",
            err=True,
        )
        raise typer.Exit(code=2)

    use_lora = cfg.training.training_type.lower() == "lora"
    task = cfg.training.task.lower()

    # Import here to avoid slow startup when only --help is requested
    try:
        from unturtle import FastDiffusionModel
        from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments
    except ImportError as e:
        typer.echo(f"Error: failed to import unturtle — {e}", err=True)
        raise typer.Exit(code=1)

    if task == "grpo":
        try:
            import trl.trainer.grpo_trainer  # noqa: F401
        except ImportError as e:
            typer.echo(
                "Error: GRPO requires optional dependencies — "
                'install with `uv pip install -e ".[huggingface,grpo]"` '
                f"(import failed: {e})",
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            from unturtle.diffusion import DiffuGRPOTrainer
        except ImportError as e:
            typer.echo(f"Error: failed to import DiffuGRPOTrainer — {e}", err=True)
            raise typer.Exit(code=1)

    model_class = _resolve_model_class(cfg.model.model_type)

    # --- Load model ---
    typer.echo(f"Loading model: {cfg.model.model}")
    try:
        model, tokenizer = FastDiffusionModel.from_pretrained(
            model_name=cfg.model.model,
            max_seq_length=cfg.model.max_seq_length,
            load_in_4bit=cfg.model.load_in_4bit if use_lora else False,
            model_class=model_class,
            token=hf_token,
        )
    except Exception as e:
        typer.echo(f"Error: model load failed — {e}", err=True)
        raise typer.Exit(code=1)

    # --- Apply LoRA ---
    if use_lora:
        typer.echo("Applying LoRA adapter...")
        try:
            model = FastDiffusionModel.get_peft_model(
                model,
                use_gradient_checkpointing=cfg.training.gradient_checkpointing,
                **cfg.lora_kwargs(),
            )
        except Exception as e:
            typer.echo(f"Error: LoRA setup failed — {e}", err=True)
            raise typer.Exit(code=1)

    # --- Load dataset ---
    typer.echo("Loading dataset...")
    try:
        if cfg.data.dataset:
            from datasets import load_dataset

            dataset = load_dataset(cfg.data.dataset, token=hf_token)
            train_dataset = dataset.get("train", dataset)
        else:
            import json

            from datasets import Dataset

            rows = []
            for path in cfg.data.local_dataset or []:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
            train_dataset = Dataset.from_list(rows)
    except Exception as e:
        typer.echo(f"Error: dataset load failed — {e}", err=True)
        raise typer.Exit(code=1)

    if cfg.logging.enable_wandb:
        report_to = "wandb"
    elif cfg.logging.enable_tensorboard:
        report_to = "tensorboard"
    else:
        report_to = "none"
    tb_dir = cfg.logging.tensorboard_dir if cfg.logging.enable_tensorboard else None

    # --- Train ---
    typer.echo("Starting training...")
    try:
        if cfg.logging.enable_wandb and wandb_token:
            import os

            os.environ.setdefault("WANDB_API_KEY", wandb_token)
        if cfg.logging.enable_wandb and cfg.logging.wandb_project:
            import os

            os.environ.setdefault("WANDB_PROJECT", cfg.logging.wandb_project)

        if task == "grpo":
            train_dataset = _dataset_for_grpo(
                train_dataset, cfg.data.dataset_text_field
            )
            tok_mask = tokenizer.mask_token_id
            if tok_mask is None and cfg.grpo.mask_id is None:
                typer.echo(
                    "Error: tokenizer has no mask_token_id — set grpo.mask_id in config",
                    err=True,
                )
                raise typer.Exit(code=2)
            mask_token_id = (
                int(cfg.grpo.mask_id) if cfg.grpo.mask_id is not None else int(tok_mask)
            )
            grpo_args = cfg.build_diffu_grpo_config(
                mask_token_id=mask_token_id,
                report_to=report_to,
                logging_dir=tb_dir,
            )
            reward_fn = _builtin_grpo_reward(cfg.grpo.builtin_reward)
            typer.echo(
                f"Diffu-GRPO: objective={grpo_args.diffu_policy_objective}, "
                f"reward={cfg.grpo.builtin_reward}"
            )
            trainer = DiffuGRPOTrainer(
                model=model,
                reward_funcs=reward_fn,
                args=grpo_args,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )
            trainer.train()
        else:
            args_kwargs = cfg.training_args_kwargs()
            args_kwargs["report_to"] = report_to
            if report_to == "tensorboard" and tb_dir:
                args_kwargs["logging_dir"] = tb_dir
            try:
                training_args = DiffusionTrainingArguments(**args_kwargs)
            except Exception as e:
                typer.echo(f"Error: invalid training arguments — {e}", err=True)
                raise typer.Exit(code=1)

            from unturtle.cli.config import build_masked_diffusion_collator

            # Mirror DiffusionTrainer's own collator construction so
            # --alpha-scheduler / --time-epsilon reach the noising process.
            collator = build_masked_diffusion_collator(
                tokenizer,
                model=model,
                alpha_scheduler=cfg.diffusion.alpha_scheduler,
                time_epsilon=cfg.diffusion.time_epsilon,
                completion_only=cfg.diffusion.completion_only,
            )
            trainer = DiffusionTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=collator,
                processing_class=tokenizer,
            )
            trainer.train()
    except KeyboardInterrupt:
        typer.echo("\nTraining interrupted.")
        raise typer.Exit(code=0)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: training failed — {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Training complete.")
