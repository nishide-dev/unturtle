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
"""GSM8K accuracy benchmark for masked-diffusion language models.

Usage:
    uv run python benchmarks/gsm8k.py --model dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \\
        --num-examples 100 --num-steps 128

Results are saved to benchmarks/results/gsm8k_<model_slug>_<date>.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GSM8K benchmark for dLLM models")
    p.add_argument("--model", required=True, help="HF model ID or local path")
    p.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of test examples (default: all)",
    )
    p.add_argument(
        "--num-steps", type=int, default=128, help="Diffusion steps (default: 128)"
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max generated tokens (default: 256)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    p.add_argument(
        "--load-in-4bit", action="store_true", help="Enable 4-bit quantisation"
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "train"],
        help="Dataset split (default: test)",
    )
    p.add_argument(
        "--output-dir", default="benchmarks/results", help="Directory for JSON output"
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


def _load_model_and_tokenizer(model_id: str, load_in_4bit: bool):
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except ImportError:
        import warnings

        warnings.warn(
            "unsloth is not installed; falling back to AutoModelForCausalLM. "
            "diffusion_generate will not be available — benchmark results will use "
            "model.generate and are NOT comparable to diffusion-model baselines.",
            stacklevel=2,
        )
    # Only ImportError triggers the fallback. All other errors (OOM, bad model
    # ID, for_inference failures) propagate so they are not silently swallowed.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _model_slug(model_id: str) -> str:
    return Path(model_id).name


def main() -> None:
    args = _parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = _load_model_and_tokenizer(args.model, args.load_in_4bit)

    from unturtle.eval.gsm8k import GSM8KEvaluator

    evaluator = GSM8KEvaluator(
        model=model,
        tokenizer=tokenizer,
        num_steps=args.num_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"\nGSM8K Evaluation — {args.model}")
    print(f"  examples : {args.num_examples or 'all'}")
    print(f"  steps    : {args.num_steps}")
    print("  " + "─" * 29)

    metrics = evaluator.evaluate(
        split=args.split,
        num_examples=args.num_examples,
        seed=args.seed,
    )

    prefix = "gsm8k"
    n_correct = int(metrics[f"{prefix}_num_correct"])
    n_total = int(metrics[f"{prefix}_num_examples"])
    accuracy = metrics[f"{prefix}_accuracy"] * 100
    parse_failures = int(metrics[f"{prefix}_parse_failures"])
    gold_parse_failures = int(metrics[f"{prefix}_gold_parse_failures"])

    print(f"  accuracy : {accuracy:.2f}% ({n_correct}/{n_total})")
    print(f"  parse_failures: {parse_failures}")
    if gold_parse_failures > 0:
        print(f"  gold_parse_failures: {gold_parse_failures} (check dataset format)")
    print("  " + "─" * 29)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _model_slug(args.model)
    today = date.today().isoformat()
    output_path = output_dir / f"gsm8k_{slug}_{today}.json"

    result = {
        "benchmark": "gsm8k",
        "model": args.model,
        "date": today,
        "config": {
            "num_steps": args.num_steps,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "num_examples": args.num_examples,
            "split": args.split,
            "seed": args.seed,
        },
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
