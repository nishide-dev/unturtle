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

"""unturtle eval — evaluate a dLLM on masked diffusion loss or generation metrics."""

import json
from pathlib import Path
from typing import Optional

import typer

from unturtle.cli.config import DataConfig

EVAL_TYPES = ["diffusion", "generation", "both"]
ALG_CHOICES = ["origin", "maskgit_plus", "topk_margin", "entropy"]


def _tokenize_for_diffusion(tokenizer, dataset, dataset_text_field: str):
    def _map(example):
        text = example.get(dataset_text_field)
        if text is None:
            raise ValueError(
                f"Dataset example is missing text field '{dataset_text_field}' required for diffusion eval."
            )
        encoded = tokenizer(str(text), add_special_tokens=True, truncation=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.copy(),
        }

    return dataset.map(
        _map,
        remove_columns=getattr(dataset, "column_names", None),
    )


def _tokenize_for_generation(tokenizer, dataset, dataset_text_field: str):
    def _map(example):
        prompt_text = example.get("prompt")
        completion_text = example.get("completion")
        if prompt_text is not None and completion_text is not None:
            prompt_ids = tokenizer(
                str(prompt_text), add_special_tokens=False, truncation=True
            )["input_ids"]
            completion_ids = tokenizer(
                str(completion_text), add_special_tokens=False, truncation=True
            )["input_ids"]
            input_ids = prompt_ids + completion_ids
            attention_mask = [1] * len(input_ids)
            labels = ([-100] * len(prompt_ids)) + completion_ids
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        text = example.get(dataset_text_field)
        if text is None:
            raise ValueError(
                "Generation eval requires either prompt/completion columns or "
                f"a text field '{dataset_text_field}'."
            )
        encoded = tokenizer(str(text), add_special_tokens=True, truncation=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [1] * len(input_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.copy(),
        }

    return dataset.map(
        _map,
        remove_columns=getattr(dataset, "column_names", None),
    )


def _prepare_eval_dataset(dataset, tokenizer, dataset_text_field: str, eval_type: str):
    column_names = set(getattr(dataset, "column_names", []))
    has_diffusion_schema = {"input_ids", "labels"}.issubset(column_names)
    has_generation_schema = "input_ids" in column_names and (
        "references" in column_names or "labels" in column_names
    )

    if eval_type == "both":
        if not has_generation_schema:
            dataset = _tokenize_for_generation(tokenizer, dataset, dataset_text_field)
            column_names = set(getattr(dataset, "column_names", []))
        has_diffusion_schema = {"input_ids", "labels"}.issubset(column_names)
        if not has_diffusion_schema:
            dataset = _tokenize_for_diffusion(tokenizer, dataset, dataset_text_field)
        return dataset

    if eval_type == "diffusion" and not has_diffusion_schema:
        return _tokenize_for_diffusion(tokenizer, dataset, dataset_text_field)

    if eval_type == "generation" and not has_generation_schema:
        return _tokenize_for_generation(tokenizer, dataset, dataset_text_field)

    return dataset


def eval_cmd(
    model: str = typer.Argument(..., help="HuggingFace model ID or local path."),
    dataset: str = typer.Option(..., "--dataset", help="HuggingFace dataset ID."),
    dataset_text_field: str = typer.Option(
        DataConfig.model_fields["dataset_text_field"].default,
        "--dataset-text-field",
        help=DataConfig.model_fields["dataset_text_field"].description,
    ),
    eval_type: str = typer.Option(
        "diffusion",
        "--eval-type",
        help=f"Evaluation type: {', '.join(EVAL_TYPES)}.",
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace token."
    ),
    max_seq_length: int = typer.Option(2048, "--max-seq-length"),
    load_in_4bit: bool = typer.Option(True, "--load-in-4bit/--no-load-in-4bit"),
    batch_size: int = typer.Option(1, "--batch-size", help="Eval batch size."),
    max_batches: Optional[int] = typer.Option(
        None,
        "--max-batches",
        help="Limit number of batches for diffusion eval (default: all).",
    ),
    max_examples: Optional[int] = typer.Option(
        None,
        "--max-examples",
        help="Limit number of examples for generation eval (default: all).",
    ),
    num_steps: int = typer.Option(
        128, "--num-steps", help="Denoising steps for generation eval."
    ),
    mask_token_id: Optional[int] = typer.Option(
        None, "--mask-token-id", help="Mask token ID override."
    ),
    alg: str = typer.Option(
        "origin",
        "--alg",
        help=f"Sampling algorithm for generation eval: {', '.join(ALG_CHOICES)}.",
    ),
    temperature: float = typer.Option(
        0.0, "--temperature", help="Sampling temperature for generation eval."
    ),
    alpha_scheduler: str = typer.Option(
        "linear",
        "--alpha-scheduler",
        help="Alpha scheduler for diffusion eval: 'linear' or 'cosine'.",
    ),
    loss_weight_type: str = typer.Option(
        "uniform",
        "--loss-weight-type",
        help="Loss weighting for diffusion eval: uniform|timestep|scheduler.",
    ),
    completion_only: bool = typer.Option(
        True,
        "--completion-only/--no-completion-only",
        help="For diffusion eval, only mask completion tokens (not the prompt).",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output-file", help="Write metrics to this JSON file."
    ),
):
    """Evaluate a dLLM using masked diffusion loss or generation metrics."""
    if eval_type not in EVAL_TYPES:
        typer.echo(
            f"Error: --eval-type must be one of {EVAL_TYPES}, got '{eval_type}'",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        from unturtle import FastDiffusionModel
        from unturtle.diffusion import MaskedDiffusionDataCollator
        from unturtle.eval import GenerationEvaluator, MaskedDiffusionEvaluator
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationConfig,
        )
    except ImportError as e:
        typer.echo(f"Error: failed to import unturtle — {e}", err=True)
        raise typer.Exit(code=1)

    # --- Load model ---
    typer.echo(f"Loading model: {model}", err=True)
    try:
        loaded_model, tokenizer = FastDiffusionModel.from_pretrained(
            model_name=model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            token=hf_token,
        )
    except Exception as e:
        typer.echo(f"Error: model load failed — {e}", err=True)
        raise typer.Exit(code=1)

    FastDiffusionModel.for_inference(loaded_model)

    # --- Load dataset ---
    typer.echo(f"Loading dataset: {dataset}", err=True)
    try:
        from datasets import load_dataset as hf_load_dataset

        ds = hf_load_dataset(dataset, token=hf_token)
        if hasattr(ds, "keys"):
            split_name = next(
                (name for name in ("validation", "test", "train") if name in ds),
                None,
            )
            if split_name is None:
                raise ValueError("Dataset has no validation, test, or train split.")
            eval_ds = ds[split_name]
        else:
            eval_ds = ds
        eval_ds = _prepare_eval_dataset(
            eval_ds,
            tokenizer=tokenizer,
            dataset_text_field=dataset_text_field,
            eval_type=eval_type,
        )
    except Exception as e:
        typer.echo(f"Error: dataset load failed — {e}", err=True)
        raise typer.Exit(code=1)

    all_metrics: dict = {}
    failures: list[str] = []

    # --- Diffusion eval ---
    if eval_type in ("diffusion", "both"):
        typer.echo("Running diffusion eval...", err=True)
        try:
            collator = MaskedDiffusionDataCollator(
                tokenizer=tokenizer,
                completion_only=completion_only,
            )
            evaluator = MaskedDiffusionEvaluator(
                model=loaded_model,
                tokenizer=tokenizer,
                data_collator=collator,
                loss_weight_type=loss_weight_type,
                alpha_scheduler=alpha_scheduler,
                completion_only=completion_only,
            )
            diff_metrics = evaluator.evaluate(
                eval_ds,
                batch_size=batch_size,
                max_batches=max_batches,
            )
            all_metrics.update(diff_metrics)
        except Exception as e:
            typer.echo(f"Error: diffusion eval failed — {e}", err=True)
            failures.append("diffusion")

    # --- Generation eval ---
    if eval_type in ("generation", "both"):
        typer.echo("Running generation eval...", err=True)

        # Resolve mask token ID
        resolved_mask_id = mask_token_id
        if resolved_mask_id is None:
            resolved_mask_id = getattr(tokenizer, "mask_token_id", None)
        if resolved_mask_id is None:
            resolved_mask_id = getattr(loaded_model.config, "mask_token_id", None)
        if resolved_mask_id is None:
            typer.echo(
                "Error: cannot resolve mask_token_id for generation eval.",
                err=True,
            )
            failures.append("generation")
        else:
            try:
                gen_config = MaskedDiffusionGenerationConfig(
                    steps=num_steps,
                    mask_token_id=resolved_mask_id,
                    temperature=temperature,
                    alg=alg,
                )
                gen_evaluator = GenerationEvaluator(
                    model=loaded_model,
                    tokenizer=tokenizer,
                )
                gen_metrics = gen_evaluator.evaluate(
                    eval_ds,
                    generation_config=gen_config,
                    max_examples=max_examples,
                )
                all_metrics.update(gen_metrics)
            except Exception as e:
                typer.echo(f"Error: generation eval failed — {e}", err=True)
                failures.append("generation")

    if failures:
        raise typer.Exit(code=1)

    if not all_metrics:
        typer.echo("No metrics collected.", err=True)
        raise typer.Exit(code=1)

    # --- Print metrics ---
    import yaml

    typer.echo("\n" + yaml.dump(all_metrics, default_flow_style=False, sort_keys=True))

    # --- Write output file ---
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        typer.echo(f"Metrics written to {output_file}", err=True)
