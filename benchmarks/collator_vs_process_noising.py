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
CPU-collator vs device-side noising: end-to-end step impact (#62).

#62 asks to "benchmark CPU-collator noising vs device-side noising and report
end-to-end step impact (not only the masking microkernel)".  Both are measured
here through a real ``DiffusionTrainer.compute_loss`` — forward, loss, and
backward — because the masking kernel itself is a rounding error next to the
model, and a microbenchmark would overstate the difference by orders of
magnitude.

**Measured result — device-side noising is not a speedup.**  B=4, V=32000,
60 timed steps small / 40 large, single GPU:

=====================  ==========  ==========  =========
config                  collator     process     delta
=====================  ==========  ==========  =========
L=512  H=512  2 layer     12.61 ms    12.70 ms     +0.7%
L=1024 H=1024 4 layer     90.91 ms    91.97 ms     +1.2%
=====================  ==========  ==========  =========

The device path is consistently *slightly slower*, by around 1%.  Take the
sign seriously and the magnitude lightly: run-to-run stdev is 0.2-1.6 ms, so
a short run swings the ratio a lot (a 15-step run of the small config read
+2.4%).  Three things explain the direction and none of them are a bug:

1. **Every call site in this repo uses ``dataloader_num_workers=0``.**  The
   motivation for moving noising off the DataLoader workers is architectural,
   not a measured win: with no workers, the "CPU collator" already runs in the
   main process, so there is no overlap to reclaim.  The device path instead
   adds kernel launches to a stream that is already the critical path.

2. **The masking work is tiny next to the model.**  Both paths do the same
   Bernoulli draw over ``B x L``; only its location changes.  At this size the
   whole step is ~12 ms, so the delta is a handful of launch overheads.

3. **The payoff #62 bought is correctness, not latency** — per-segment packed
   timesteps, which unblock ``timestep``/``scheduler`` weighting on packed
   batches.  The CPU collator collapsed a packed row to one mean ``t`` and had
   to reject those weightings outright.

Scaling the model 7x (12.6 ms -> 90.9 ms per step) did **not** shrink the
ratio, which argues the cost is not a fixed per-step overhead being amortised
away.  The likeliest reading is that noising ``B x L`` on-device competes for
the same SMs as the forward/backward, so it grows alongside them.  Confirm
with a profiler before treating that as established.

**Pre-#62 and post-#62 runs are not seed-comparable.**  Noising moved from the
collator's CPU RNG stream to the device stream, so identical seeds give
different masks.  #62 explicitly waives bit-parity and requires only
fixed-seed reproducibility, which the trainer's ``set_seed`` provides.  Do not
read a changed loss curve across that boundary as a regression.

Usage::

    uv run python benchmarks/collator_vs_process_noising.py
    uv run python benchmarks/collator_vs_process_noising.py --steps 50 --seq-len 1024
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-collator vs device-side noising, end-to-end per step"
    )
    p.add_argument("--steps", type=int, default=20, help="timed steps per path")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps first")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    return p.parse_args()


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.add_special_tokens(
        {
            "unk_token": "[UNK]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "eos_token": "[EOS]",
        }
    )
    tokenizer.name_or_path = "local"
    return tokenizer


def _model(args):
    from transformers import BertConfig, BertForMaskedLM

    return BertForMaskedLM(
        BertConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.layers,
            num_attention_heads=8,
            intermediate_size=args.hidden_size * 4,
            max_position_embeddings=max(args.seq_len * 2, 512),
        )
    ).to(args.device)


def _features(args):
    torch.manual_seed(0)
    prompt = args.seq_len // 4
    return [
        {
            "input_ids": torch.randint(1, args.vocab_size, (args.seq_len,)).tolist(),
            "labels": (
                [-100] * prompt
                + torch.randint(1, args.vocab_size, (args.seq_len - prompt,)).tolist()
            ),
            "attention_mask": [1] * args.seq_len,
        }
        for _ in range(args.batch_size)
    ]


def _time_path(trainer, model, collator, features, args) -> list[float]:
    """Per-step wall clock for collate + compute_loss + backward."""
    timings: list[float] = []
    for step in range(args.warmup + args.steps):
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        batch = collator(features)
        batch = {
            k: (v.to(args.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        loss = trainer.compute_loss(model, batch)
        loss.backward()
        model.zero_grad(set_to_none=True)

        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if step >= args.warmup:
            timings.append(elapsed)
    return timings


def main() -> None:
    args = _parse_args()

    from unturtle.diffusion import (
        DiffusionTrainer,
        DiffusionTrainingArguments,
        MaskedDiffusionDataCollator,
    )

    tokenizer = _tokenizer()
    features = _features(args)

    print(
        f"device={args.device} B={args.batch_size} L={args.seq_len} "
        f"V={args.vocab_size} H={args.hidden_size} layers={args.layers} "
        f"steps={args.steps} (+{args.warmup} warmup)"
    )
    print()

    results: dict[str, list[float]] = {}
    for label, noise in (("collator (CPU noising)", True), ("process (device)", False)):
        model = _model(args)
        training_args = DiffusionTrainingArguments(
            output_dir="/tmp/unturtle-bench",
            per_device_train_batch_size=args.batch_size,
            max_steps=1,
            use_cpu=(args.device == "cpu"),
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            report_to=[],
        )
        collator = MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            mask_token_id=tokenizer.mask_token_id,
            noise=noise,
        )
        trainer = DiffusionTrainer(
            model=model,
            args=training_args,
            train_dataset=features,
            processing_class=tokenizer,
            data_collator=collator,
        )
        results[label] = _time_path(trainer, model, collator, features, args)
        del model, trainer
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"{'path':<26} {'median ms':>10} {'p90 ms':>9} {'stdev':>8}")
    for label, timings in results.items():
        ms = [t * 1000 for t in timings]
        ms_sorted = sorted(ms)
        p90 = ms_sorted[int(0.9 * (len(ms_sorted) - 1))]
        stdev = statistics.stdev(ms) if len(ms) > 1 else 0.0
        print(f"{label:<26} {statistics.median(ms):>10.2f} {p90:>9.2f} {stdev:>8.2f}")

    baseline, candidate = results.values()
    delta = statistics.median(candidate) / statistics.median(baseline) - 1.0
    print()
    print(f"device-side vs CPU collator: {delta:+.1%} per step")
    print(
        "\nRead this alongside the caveats in the module docstring: with "
        "dataloader_num_workers=0 both paths run in the same process, so this "
        "measures where the work happens, not whether it overlaps."
    )


if __name__ == "__main__":
    main()
