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

**Measured result — no detectable end-to-end difference.**  B=4, L=512,
V=32000, H=512, 2 layers, 5 interleaved trials of 40 timed steps each,
single GPU (transformers 5.14.1, torch 2.10.0+cu128):

=======  ==========  ==========  =========
trial     collator     process     delta
=======  ==========  ==========  =========
1           12.67 ms    12.71 ms     +0.32%
2           12.70 ms    12.75 ms     +0.42%
3           12.69 ms    12.78 ms     +0.71%
4           12.88 ms    12.80 ms     -0.61%
5           12.81 ms    12.95 ms     +1.12%
=======  ==========  ==========  =========

Median **+0.42%**, range -0.61% to +1.12%, device path slower in 4 of 5
trials.  **The sign is not consistent, so this is "no measurable difference",
not a regression.**  Do not quote the median as if it were an effect: at this
size the noising is a handful of kernel launches inside a ~12.7 ms step, and
trial-to-trial drift is larger than the thing being measured.  (An early
single 15-step run read +2.4% and a single 60-step run read +0.7%; both were
drift.  That is why ``--trials`` interleaves the arms and alternates their
order — running one arm to completion and then the other loads all thermal
drift onto whichever goes second.)

Two things this does establish:

1. **There is no latency win to be had here, so do not look for one.**  Every
   call site in this repo uses ``dataloader_num_workers=0``, so the "CPU
   collator" already runs in the main process — there is no worker overlap to
   reclaim.  Moving the work changes where it happens, not whether it overlaps.

2. **#62's payoff is correctness, not speed** — per-segment packed timesteps,
   which unblock ``timestep``/``scheduler`` weighting on packed batches.  The
   CPU collator collapsed each packed row to one mean ``t`` and had to reject
   those weightings outright.

If you do want to attribute a real difference, profile it: the two arms are
not symmetric (the CPU arm's Bernoulli draw happens before the H2D copy, with
no async work in flight; the device arm's kernels queue behind that copy), and
that asymmetry — not SM contention — is the first thing to rule out.

**Pre-#62 and post-#62 runs are not seed-comparable.**  Noising moved from the
collator's CPU RNG stream to the device stream, so identical seeds give
different masks.  #62 explicitly waives bit-parity and requires only
fixed-seed reproducibility, which the trainer's ``set_seed`` provides.  Do not
read a changed loss curve across that boundary as a regression.

Usage::

    uv run python benchmarks/collator_vs_process_noising.py
    uv run python benchmarks/collator_vs_process_noising.py --trials 9 --steps 60
    uv run python benchmarks/collator_vs_process_noising.py --seq-len 1024 --hidden-size 1024
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CPU-collator vs device-side noising, end-to-end per step"
    )
    p.add_argument("--steps", type=int, default=20, help="timed steps per path")
    p.add_argument("--warmup", type=int, default=5, help="untimed steps first")
    p.add_argument(
        "--trials",
        type=int,
        default=5,
        help=(
            "independent A/B trials, interleaved.  A single trial cannot "
            "separate a ~1%% effect from run-to-run drift; the report is a "
            "median over trials plus a sign-consistency count."
        ),
    )
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


def _output_dir() -> Path:
    """A throwaway ``output_dir`` for the Trainer, which requires one.

    Nothing is ever written here — no step is saved and no checkpoint taken —
    but ``TrainingArguments`` creates the directory, so keep it out of the
    working tree.
    """
    path = Path(tempfile.gettempdir()) / "unturtle-bench"
    path.mkdir(parents=True, exist_ok=True)
    return path


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

    def _run_arm(noise: bool) -> list[float]:
        model = _model(args)
        training_args = DiffusionTrainingArguments(
            output_dir=str(_output_dir()),
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
        timings = _time_path(trainer, model, collator, features, args)
        del model, trainer
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return timings

    # Arms are interleaved *within* a trial and their order alternates between
    # trials.  Running one arm to completion and then the other would load all
    # thermal/clock drift onto whichever went second, which at a ~1% effect
    # size is the same order as the thing being measured.
    per_trial: list[tuple[float, float]] = []
    for trial in range(args.trials):
        order = [True, False] if trial % 2 == 0 else [False, True]
        medians: dict[bool, float] = {}
        for noise in order:
            medians[noise] = statistics.median(_run_arm(noise)) * 1000
        per_trial.append((medians[True], medians[False]))
        print(
            f"  trial {trial + 1}/{args.trials}: "
            f"collator {medians[True]:7.2f} ms   process {medians[False]:7.2f} ms   "
            f"{medians[False] / medians[True] - 1:+.2%}"
        )

    collator_ms = [c for c, _ in per_trial]
    process_ms = [p for _, p in per_trial]
    deltas = [p / c - 1.0 for c, p in per_trial]
    slower = sum(d > 0 for d in deltas)

    print()
    print(f"{'path':<26} {'median ms':>10} {'min':>8} {'max':>8}")
    for label, xs in (
        ("collator (CPU noising)", collator_ms),
        ("process (device)", process_ms),
    ):
        print(
            f"{label:<26} {statistics.median(xs):>10.2f} {min(xs):>8.2f} {max(xs):>8.2f}"
        )

    print()
    print(
        f"device-side vs CPU collator: {statistics.median(deltas):+.2%} median "
        f"of {args.trials} trials  (range {min(deltas):+.2%} .. {max(deltas):+.2%})"
    )
    print(f"device path slower in {slower}/{args.trials} trials")
    if slower not in (0, args.trials):
        print(
            "  -> sign is NOT consistent across trials: treat this as "
            "'no measurable difference', not as a regression."
        )
    print(
        "\nRead this alongside the caveats in the module docstring: with "
        "dataloader_num_workers=0 both paths run in the same process, so this "
        "measures where the work happens, not whether it overlaps."
    )


if __name__ == "__main__":
    main()
