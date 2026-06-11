#!/usr/bin/env python3
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
"""GSM8K SFT verification: fine-tune a dLLM on GSM8K train, measure before/after accuracy.

Usage (smoke test — ~5 min on one GPU):
    uv run python benchmarks/gsm8k_sft.py \\
        --num-train-samples 200 --num-epochs 1 --eval-examples 20

Usage (full run — ~30–60 min):
    uv run python benchmarks/gsm8k_sft.py

Results saved to benchmarks/results/gsm8k_sft_<slug>_<date>.json.
Checkpoint saved to outputs/gsm8k_sft/<date>/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import peft  # noqa: F401
import torch
import transformers  # noqa: F401
import trl  # noqa: F401
from datasets import load_dataset
from peft import PeftModel

import unturtle  # noqa: F401
from unturtle import FastDiffusionModel
from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments
from unturtle.eval.gsm8k import DEFAULT_SYSTEM_PROMPT, GSM8KEvaluator
from unturtle.models import TinyA2DQwen3LMHeadModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GSM8K SFT verification for dLLM models")
    p.add_argument(
        "--model",
        default="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        help=(
            "HF model ID or local path for an A2D Qwen3 checkpoint "
            "(default: Qwen3-0.6B-diffusion-mdlm-v0.1)"
        ),
    )
    p.add_argument(
        "--num-train-samples",
        type=int,
        default=7473,
        help="GSM8K train examples for SFT (default: 7473 = full dataset)",
    )
    p.add_argument(
        "--num-epochs", type=int, default=3, help="Training epochs (default: 3)"
    )
    p.add_argument(
        "--eval-examples",
        type=int,
        default=100,
        help="GSM8K test examples for before/after eval (default: 100)",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="Diffusion steps for eval (default: 128)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens for eval (default: 256)",
    )
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    p.add_argument(
        "--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)"
    )
    p.add_argument(
        "--load-in-4bit", action="store_true", help="Enable 4-bit quantisation"
    )
    p.add_argument(
        "--output-dir",
        default="outputs/gsm8k_sft",
        help="Parent directory for checkpoints (default: outputs/gsm8k_sft)",
    )
    p.add_argument(
        "--results-dir",
        default="benchmarks/results",
        help="Directory for JSON results (default: benchmarks/results)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_slug(model_id: str) -> str:
    return Path(model_id).name


def _move_model_to_cuda_if_available(model: torch.nn.Module) -> torch.nn.Module:
    if not torch.cuda.is_available():
        return model

    inner_model = model.get_base_model() if isinstance(model, PeftModel) else model
    if getattr(inner_model, "is_loaded_in_4bit", False):
        return model

    return model.to("cuda")


def _boxed_completion(answer: str) -> str:
    reasoning, _, final = answer.partition("####")
    final_value = final.strip() if final else answer.strip()
    number_matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", final_value)
    boxed_value = number_matches[-1].replace(",", "") if number_matches else final_value
    reasoning = reasoning.rstrip()
    if reasoning:
        return f"{reasoning}\n\\boxed{{{boxed_value}}}"
    return f"\\boxed{{{boxed_value}}}"


def main() -> None:
    args = _parse_args()
    _set_seeds(args.seed)
    print(f"Config: {json.dumps(vars(args), indent=2)}", flush=True)
    # ------------------------------------------------------------------
    # [1/5] Load model & tokenizer
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading model: {args.model} …", flush=True)

    model, tokenizer = FastDiffusionModel.from_pretrained(
        args.model,
        model_class=TinyA2DQwen3LMHeadModel,
        dtype=torch.bfloat16,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    mask_token_id = tokenizer.mask_token_id or getattr(
        model.config, "mask_token_id", None
    )
    if mask_token_id is not None:
        model.config.mask_token_id = mask_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.mask_token_id = mask_token_id
    print(f"  Loaded: {n_params / 1e9:.3f}B params", flush=True)
    print(f"  mask_token_id: {mask_token_id}", flush=True)

    # ------------------------------------------------------------------
    # [2/5] Apply LoRA
    # ------------------------------------------------------------------
    print("\n[2/5] Applying LoRA …", flush=True)

    model = FastDiffusionModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  Trainable: {n_trainable / 1e6:.1f}M / {n_params / 1e9:.3f}B "
        f"({n_trainable / n_params * 100:.2f}%)",
        flush=True,
    )

    # ------------------------------------------------------------------
    # [3/5] Evaluate BEFORE fine-tuning
    # ------------------------------------------------------------------
    print(
        f"\n[3/5] Evaluating BEFORE fine-tuning ({args.eval_examples} examples) …",
        flush=True,
    )

    model = _move_model_to_cuda_if_available(model)

    evaluator = GSM8KEvaluator(
        model=model,
        tokenizer=tokenizer,
        num_steps=args.num_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    before_metrics = evaluator.evaluate(
        split="test",
        num_examples=args.eval_examples,
        seed=args.seed,
    )
    before_acc = before_metrics["gsm8k_accuracy"] * 100
    before_correct = int(before_metrics["gsm8k_num_correct"])
    before_total = int(before_metrics["gsm8k_num_examples"])
    before_parse_failures = int(before_metrics["gsm8k_parse_failures"])
    print(
        f"  accuracy : {before_acc:.2f}% ({before_correct}/{before_total})",
        flush=True,
    )
    print(f"  parse failures: {before_parse_failures}", flush=True)

    # ------------------------------------------------------------------
    # [4/5] Fine-tune on GSM8K train
    # ------------------------------------------------------------------
    print(
        f"\n[4/5] Fine-tuning on GSM8K train "
        f"({args.num_train_samples} samples, {args.num_epochs} epochs) …",
        flush=True,
    )

    _SYSTEM = DEFAULT_SYSTEM_PROMPT
    _MAX_LENGTH = 512

    raw_train = cast(
        Any,
        load_dataset(
            "openai/gsm8k",
            "main",
            split=f"train[:{args.num_train_samples}]",
        ),
    )

    def _preprocess(example: dict) -> dict:
        completion = _boxed_completion(example["answer"])
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": completion},
        ]
        # Build full text to find prompt boundary
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            tokenize=False,
        )
        completion_text = completion + (tokenizer.eos_token or "")
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)[
            "input_ids"
        ]
        input_ids = (prompt_ids + completion_ids)[:_MAX_LENGTH]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:_MAX_LENGTH]
        return {"input_ids": input_ids, "labels": labels}

    train_dataset = raw_train.map(
        _preprocess,
        remove_columns=raw_train.column_names,
        desc="Tokenising GSM8K train",
    )
    train_dataset = train_dataset.filter(
        lambda x: sum(1 for lbl in x["labels"] if lbl != -100) > 0
    )
    print(f"  train examples after filter: {len(train_dataset)}", flush=True)

    today_str = date.today().isoformat()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(checkpoint_dir, exist_ok=False)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    training_args = cast(Any, DiffusionTrainingArguments)(
        output_dir=checkpoint_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        logging_steps=10,
        disable_tqdm=True,
        eval_strategy="no",
        save_strategy="epoch",
        report_to="none",
        loss_weight_type="uniform",
        completion_only=True,
        dataloader_num_workers=0,
    )

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0

    train_losses = [
        e["loss"]
        for e in trainer.state.log_history
        if "loss" in e and "eval_loss" not in e
    ]
    train_loss_final = train_losses[-1] if train_losses else float("nan")

    print(f"  train_loss : {train_loss_final:.4f}", flush=True)
    print(f"  steps      : {train_result.global_step}", flush=True)
    print(f"  elapsed    : {elapsed / 60:.1f} min", flush=True)
    print(f"  peak VRAM  : {peak_vram_gb:.2f} GB", flush=True)
    print(f"  checkpoint : {checkpoint_dir}", flush=True)

    # ------------------------------------------------------------------
    # [5/5] Evaluate AFTER fine-tuning
    # ------------------------------------------------------------------
    print(
        f"\n[5/5] Evaluating AFTER fine-tuning ({args.eval_examples} examples) …",
        flush=True,
    )

    evaluator = GSM8KEvaluator(
        model=model,
        tokenizer=tokenizer,
        num_steps=args.num_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    after_metrics = evaluator.evaluate(
        split="test",
        num_examples=args.eval_examples,
        seed=args.seed,
    )
    after_acc = after_metrics["gsm8k_accuracy"] * 100
    after_correct = int(after_metrics["gsm8k_num_correct"])
    after_total = int(after_metrics["gsm8k_num_examples"])
    after_parse_failures = int(after_metrics["gsm8k_parse_failures"])
    delta_pp = after_acc - before_acc

    print(
        f"  accuracy : {after_acc:.2f}% ({after_correct}/{after_total})",
        flush=True,
    )
    print(f"  parse failures: {after_parse_failures}", flush=True)
    print("  " + "─" * 38, flush=True)
    sign = "+" if delta_pp >= 0 else ""
    print(f"  delta    : {sign}{delta_pp:.2f}pp", flush=True)

    # ------------------------------------------------------------------
    # Save JSON result
    # ------------------------------------------------------------------
    result = {
        "benchmark": "gsm8k_sft",
        "model": args.model,
        "date": today_str,
        "config": {
            "num_train_samples": args.num_train_samples,
            "num_epochs": args.num_epochs,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "eval_examples": args.eval_examples,
            "num_steps": args.num_steps,
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": args.load_in_4bit,
            "seed": args.seed,
        },
        "before": {
            "gsm8k_accuracy": before_metrics["gsm8k_accuracy"],
            "gsm8k_num_correct": before_correct,
            "gsm8k_num_examples": before_total,
            "gsm8k_parse_failures": int(before_metrics["gsm8k_parse_failures"]),
        },
        "after": {
            "gsm8k_accuracy": after_metrics["gsm8k_accuracy"],
            "gsm8k_num_correct": after_correct,
            "gsm8k_num_examples": after_total,
            "gsm8k_parse_failures": int(after_metrics["gsm8k_parse_failures"]),
        },
        "delta": {
            "gsm8k_accuracy": after_metrics["gsm8k_accuracy"]
            - before_metrics["gsm8k_accuracy"],
        },
        "training": {
            "train_loss_final": train_loss_final,
            "elapsed_seconds": elapsed,
            "steps": train_result.global_step,
            "peak_vram_gb": peak_vram_gb,
        },
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(args.model)
    output_path = results_dir / f"gsm8k_sft_{slug}_{run_id}.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
