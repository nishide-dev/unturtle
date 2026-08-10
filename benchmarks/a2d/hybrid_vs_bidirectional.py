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
"""Matched hybrid-vs-bidirectional adaptation benchmark (#63).

Reproduces (or refutes) PreDiff-LM's *directional* claim on a tractable
setup: starting from the SAME pretrained AR initialization and spending the
SAME compute, does hybrid attention (prompt-causal + target-bidirectional,
eq. 3) adapt better than uniform bidirectional attention?

Matched by construction:

- **Init**: both arms call ``load_tiny_a2d_from_ar`` on the same checkpoint;
  the only difference is the ``hybrid_attention`` config flag.
- **Data & noise**: same tokenized rows, same order, same collator type
  (``HybridPromptCollator`` — the bidirectional arm receives and ignores
  ``prompt_lengths`` by contract), same trainer seed, so both arms see
  identical batches and identical mask draws step for step.
- **Compute**: same steps / batch / sequence length / optimizer / dtype;
  arms run sequentially on the same GPU.

Evaluation is a masked-diffusion NLL proxy on held-out rows: per timestep
``t`` in a fixed grid, Bernoulli(``t``) masks over target positions (drawn
ONCE from an explicit seeded generator and reused for every arm/seed/
checkpoint — the reproducibility rule for research benchmarks), mean CE over
masked positions.  Each arm is evaluated under its own inference topology
(the hybrid arm receives ``prompt_lengths`` at eval), because the comparison
is between the *models users would actually run*.

Usage:
    uv run python benchmarks/a2d/hybrid_vs_bidirectional.py --smoke
    uv run python benchmarks/a2d/hybrid_vs_bidirectional.py \
        --steps 400 --seeds 0 1

Results land in benchmarks/results/hybrid_vs_bidirectional_<stamp>.json
(ignored; archive curated tables under dev/local/).

Frozen reference run (2026-08-09, RTX 6000 Ada, defaults above — the
directional claim reproduced; full tables in
dev/local/hybrid_vs_bidirectional_2026-08-09.md):

    final NLL (step 400, mean over t in {0.25, 0.5, 0.75}):
        arm            seed 0   seed 1
        bidirectional  2.5331   2.5327
        hybrid         2.1188   2.1111
    hybrid ahead at EVERY checkpoint, every t, both seeds; step-0 NLL
    7.81 (hybrid) vs 14.84 (bidirectional) — the causal prompt preserves
    the AR pretraining before any gradient step.
    cost parity: ~4.5-4.7 steps/s and 9.1 GiB peak for both arms
    (seq_len 256 < hybrid_fast_min_seq_len, so the hybrid arm ran the
    dense eq.-(3) mask path).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback

from unturtle.diffusion import (
    DiffusionTrainer,
    DiffusionTrainingArguments,
    MaskedDiffusionDataCollator,
)
from unturtle.models.conversion.a2d.tiny_a2d import (
    HybridPromptCollator,
    load_tiny_a2d_from_ar,
    prompt_lengths_from_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--eval-rows", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--t-grid", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--mask-seed", type=int, default=1234)
    parser.add_argument("--gen-eval-prompts", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny sanity run: 20 steps, 128 train rows, 32 eval rows",
    )
    args = parser.parse_args()
    if args.smoke:
        args.steps = 20
        args.eval_every = 10
        args.seeds = [0]
        args.train_rows = 128
        args.eval_rows = 32
        args.gen_eval_prompts = 16
    return args


def tokenize_gsm8k(tokenizer, split, rows, seq_len, seed):
    """Prompt = question (causal side), target = answer (masked side)."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    dataset = dataset.shuffle(seed=seed).select(range(min(rows, len(dataset))))
    out = []
    for example in dataset:
        prompt_ids = tokenizer(
            f"Question: {example['question']}\nAnswer:",
            add_special_tokens=False,
        )["input_ids"]
        target_ids = tokenizer(
            " " + example["answer"] + tokenizer.eos_token,
            add_special_tokens=False,
        )["input_ids"]
        ids = (prompt_ids + target_ids)[:seq_len]
        prompt_len = min(len(prompt_ids), seq_len)
        if prompt_len >= len(ids):
            continue  # answer fully truncated away — no supervision
        out.append(
            {
                "input_ids": ids,
                "labels": [-100] * prompt_len + ids[prompt_len:],
                "attention_mask": [1] * len(ids),
            }
        )
    return out


def build_eval_pack(rows, tokenizer, t_grid, mask_seed, device):
    """Pad the eval rows once and pre-draw every mask from one explicit
    generator, so every arm/seed/checkpoint scores the exact same noise."""
    max_len = max(len(row["input_ids"]) for row in rows)
    pad_id = tokenizer.pad_token_id or 0
    input_ids, labels = [], []
    for row in rows:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    # Length-derived, not id-derived: comparing against pad_id would silently
    # drop a real prompt token that happens to equal it (pad == eos is common),
    # arm-symmetric but wrong in absolute terms.
    lengths = torch.tensor([len(row["input_ids"]) for row in rows])
    attention_mask = (torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)).long()
    supervised = labels != -100

    generator = torch.Generator().manual_seed(mask_seed)
    masks = {}
    for t in t_grid:
        draw = torch.rand(labels.shape, generator=generator) < t
        mask = draw & supervised
        # Force >= 1 masked target position per row so every row scores.
        dead = ~mask.any(dim=1)
        if bool(dead.any()):
            first_target = torch.argmax(supervised.long(), dim=1)
            mask[dead, first_target[dead]] = True
        mask &= supervised
        masks[t] = mask
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
        "prompt_lengths": prompt_lengths_from_labels(labels).to(device),
        "masks": {t: m.to(device) for t, m in masks.items()},
    }


@torch.no_grad()
def eval_nll(model, pack, mask_token_id, hybrid, batch_size):
    """Mean CE over masked target positions, per t and averaged."""
    model.eval()
    rows = pack["input_ids"].shape[0]
    per_t = {}
    for t, mask in pack["masks"].items():
        total, count = 0.0, 0
        for start in range(0, rows, batch_size):
            sl = slice(start, start + batch_size)
            ids = pack["input_ids"][sl].clone()
            mask_b = mask[sl]
            ids[mask_b] = mask_token_id
            kwargs = {"attention_mask": pack["attention_mask"][sl]}
            if hybrid:
                kwargs["prompt_lengths"] = pack["prompt_lengths"][sl]
            logits = model(input_ids=ids, **kwargs).logits
            ce = torch.nn.functional.cross_entropy(
                logits[mask_b].float(),
                pack["labels"][sl][mask_b],
                reduction="sum",
            )
            total += float(ce)
            count += int(mask_b.sum())
        per_t[t] = total / count
    model.train()
    per_t["mean"] = sum(v for k, v in per_t.items() if k != "mean") / len(pack["masks"])
    return per_t


class NLLCurveCallback(TrainerCallback):
    def __init__(self, model, pack, mask_token_id, hybrid, every, batch_size):
        self.model = model
        self.pack = pack
        self.mask_token_id = mask_token_id
        self.hybrid = hybrid
        self.every = every
        self.batch_size = batch_size
        self.curve = []

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            point = eval_nll(
                self.model,
                self.pack,
                self.mask_token_id,
                self.hybrid,
                self.batch_size,
            )
            self.curve.append({"step": state.global_step, **point})
            print(
                f"    step {state.global_step}: eval NLL mean {point['mean']:.4f}",
                flush=True,
            )


def run_arm(args, hybrid, seed, train_rows, eval_pack, output_dir):
    torch.manual_seed(seed)
    model = load_tiny_a2d_from_ar(
        args.model, hybrid_attention=hybrid, torch_dtype=torch.bfloat16
    ).to(args.device)
    mask_token_id = model.config.mask_token_id
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    collator = HybridPromptCollator(
        MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            mask_token_id=mask_token_id,
            completion_only=True,
        )
    )

    initial = eval_nll(model, eval_pack, mask_token_id, hybrid, args.eval_batch_size)
    print(f"    step 0: eval NLL mean {initial['mean']:.4f}", flush=True)
    callback = NLLCurveCallback(
        model, eval_pack, mask_token_id, hybrid, args.eval_every, args.eval_batch_size
    )
    callback.curve.append({"step": 0, **initial})

    training_args = DiffusionTrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        seed=seed,
        data_seed=seed,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_rows,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    torch.cuda.reset_peak_memory_stats(args.device)
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start

    result = {
        "hybrid": hybrid,
        "seed": seed,
        "curve": callback.curve,
        "final_nll_mean": callback.curve[-1]["mean"],
        "train_seconds": elapsed,
        "steps_per_second": args.steps / elapsed,
        # Read BEFORE the generation section below allocates: dict values
        # evaluate top-to-bottom, and that ordering is what keeps the frozen
        # training-memory number unaffected by the additive section.
        "peak_memory_gib": torch.cuda.max_memory_allocated(args.device) / 2**30,
        # #123 consumer 2: free-generation metrics through the canonical
        # `unturtle.eval` surface, via the MASKED family's own path (mdlm) —
        # an independent second consumer proving the surface is not a
        # DFM-only carve-out.  Additive: the frozen training metrics above
        # are untouched.
        "generation_metrics": generation_metrics_section(
            model, tokenizer, args, seed=seed
        ),
    }
    del trainer, model
    torch.cuda.empty_cache()
    return result


def generation_metrics_section(model, tokenizer, args, *, seed):
    """#125 pre-registered readout: prompt-conditioned MAUVE (primary) +
    guard trio + latency, and GSM8K exact match (secondary), all on the
    canonical `unturtle.eval` surface via the masked family's own path.
    Additive: the frozen #114 training metrics above are untouched."""
    from datasets import load_dataset

    from unturtle.eval import (
        diversity_guards,
        generation_record,
        mauve_score,
        measure_generation,
    )
    from unturtle.eval._answer_parser import extract_numeric_answer
    from unturtle.eval.harness.configs import DecodingConfig

    model.eval()
    decoding = DecodingConfig(
        model_family="tiny-a2d-qwen3",
        task="gsm8k-conditioned-generation",
        max_new_tokens=64,
        num_steps=16,
        temperature=1.0,
        use_chat_template=False,
        fewshot=0,
        algorithm="mdlm",
    )
    held_out = load_dataset("openai/gsm8k", "main", split="test").select(
        range(args.gen_eval_prompts)
    )
    reference_answers = [example["answer"] for example in held_out]
    gold = [extract_numeric_answer(answer) for answer in reference_answers]

    generated_texts: list[str] = []
    all_completions = []
    total_seconds = 0.0
    batch = 16
    for start in range(0, len(held_out), batch):
        chunk = held_out.select(range(start, min(start + batch, len(held_out))))
        prompts = tokenizer(
            [f"Question: {q}\nAnswer:" for q in chunk["question"]],
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).input_ids.to(args.device)

        def sample(prompts=prompts):
            with torch.no_grad():
                # The masked config names this field `steps` (#124 lesson);
                # `num_steps=` would be silently discarded and the run would
                # execute the 128-step default while recording nfe=16.
                return model.generate(
                    prompts,
                    algorithm="mdlm",
                    max_new_tokens=decoding.max_new_tokens,
                    steps=decoding.num_steps,
                    temperature=decoding.temperature,
                )

        samples, seconds = measure_generation(sample)
        total_seconds += seconds
        completions = samples[:, prompts.shape[1] :].cpu()
        all_completions.append(completions)
        generated_texts.extend(
            tokenizer.batch_decode(completions, skip_special_tokens=True)
        )

    completions = torch.cat(all_completions)
    guards = diversity_guards(completions)
    score = mauve_score(
        reference_answers,
        generated_texts,
        featurize_model_name="gpt2",
        device_id=torch.device(args.device).index or 0,
        max_text_length=decoding.max_new_tokens,
    )
    predictions = [extract_numeric_answer(text) for text in generated_texts]
    exact = sum(
        1
        for prediction, answer in zip(predictions, gold, strict=True)
        if prediction is not None and answer is not None and prediction == answer
    ) / len(gold)

    record = generation_record(
        metrics={"mauve": score, "gsm8k_exact_match": exact, **guards},
        seed=seed,
        decoding=decoding,
        nfe=decoding.num_steps,
        latency_seconds=total_seconds,
    )
    print(
        f"    generation (mdlm): MAUVE {score:.3f} exact {exact:.3f} guards {guards}",
        flush=True,
    )
    return record


def main() -> None:
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_rows = tokenize_gsm8k(
        tokenizer, "train", args.train_rows, args.seq_len, seed=0
    )
    eval_rows = tokenize_gsm8k(tokenizer, "test", args.eval_rows, args.seq_len, seed=0)
    eval_pack = build_eval_pack(
        eval_rows, tokenizer, args.t_grid, args.mask_seed, args.device
    )
    print(
        f"train rows: {len(train_rows)}  eval rows: {len(eval_rows)}  "
        f"t grid: {args.t_grid}",
        flush=True,
    )

    runs = []
    for seed in args.seeds:
        for hybrid in (False, True):
            arm = "hybrid" if hybrid else "bidirectional"
            print(f"== arm={arm} seed={seed}", flush=True)
            runs.append(
                run_arm(
                    args,
                    hybrid,
                    seed,
                    train_rows,
                    eval_pack,
                    Path(f"outputs/hybrid_vs_bd/{stamp}/{arm}-s{seed}"),
                )
            )

    payload = {
        "config": {
            **{k: v for k, v in vars(args).items() if k not in ("smoke",)},
            "objective": "DiffusionTrainer defaults (MDLM SUBS)",
            "optimizer": "adamw (trainer default), constant lr",
            "eval": "masked CE over target positions, fixed pre-drawn masks",
            "note": (
                "arms share init/data/noise/steps; sequential on one GPU; "
                "each arm evaluated under its own inference topology"
            ),
        },
        "runs": runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"hybrid_vs_bidirectional_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nresults -> {out}", flush=True)

    print("\narm            seed  final NLL (mean over t)")
    for run in runs:
        arm = "hybrid" if run["hybrid"] else "bidirectional"
        print(f"{arm:14s} {run['seed']:4d}  {run['final_nll_mean']:.4f}")


if __name__ == "__main__":
    main()
