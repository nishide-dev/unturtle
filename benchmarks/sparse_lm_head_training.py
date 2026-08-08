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
End-to-end `sparse_lm_head` benchmark through `DiffusionTrainer` (#61).

#77 measured the *kernel*.  This measures the feature through the path users
actually run — `DiffusionTrainer.compute_loss` including forward, loss and
backward — across the axis that decides whether it helps at all.

**Mask ratio is the decisive variable, not vocabulary size.**  The sparse path
projects only masked positions, so its saving scales with `1 - M/(B*L)`.  #77
measured the memory sign flipping around a ~40% mask ratio, and MDLM-style
training samples `t ~ U(0,1)`, averaging ~50% — the regime where this is *not*
a memory win.  A benchmark that only reported a favourable ratio would be
actively misleading, so the ratio sweep is mandatory here rather than optional.

Both arms are handed the **same pre-noised batch** so they cannot accidentally
benchmark different masks, and arms are interleaved with alternating order so
thermal drift does not land entirely on whichever runs second.

Measured (RTX 6000 Ada, fp32, B=2 L=512 H=512, 2 layers, 3 interleaved trials
of 10 timed steps; negative = sparse better):

======  =======  =====================  =====================
vocab   mask     step time              peak memory
======  =======  =====================  =====================
32000   0.15     **-26.1%**             **-19.2%**
32000   0.50     **-13.9%**             +0.2%
32000   0.75     +1.6%                  -6.1%
128256  0.15     **-61.0%**             **-16.5%**
128256  0.50     **-27.0%**             +0.6%
128256  0.75     **-6.8%**              +30.7%
======  =======  =====================  =====================

LoRA, 128256 vocab, same setup:

=======  =====================  =====================
mask     step time              peak memory
=======  =====================  =====================
0.15     **-49.3%**             **-17.4%**
0.50     -4.6%                  +36.8%
0.75     +16.0%                 +53.9%
=======  =====================  =====================

Three things to take from this:

1. **Step time is the real win, and it is much larger than the kernel
   benchmark in #77 suggested** — up to -61% at 128K vocab. End-to-end, the
   ``[B, L, V]`` GEMM dominates a small model's step, so skipping most of it
   matters more than the kernel microbenchmark showed.
2. **Memory is roughly neutral at the ratio MDLM actually trains at** (+0.2% /
   +0.6% at 0.50) and turns clearly negative above it. This is the #77 result
   reproduced end-to-end: sparse projection is *not* automatically a memory
   optimization.
3. **LoRA is the case to be careful with.** Above ~15% masking it is worse on
   memory than full finetune (+36.8% at 0.50), because the frozen backbone's
   activations already dominate and the ``[M, V]`` projection plus its autograd
   graph is close to pure overhead.

So the flag stays **default-off**, but the honest recommendation is narrower
and more useful than "off": enable it for large vocabularies at low mask
ratios, where it is a large step-time win at no memory cost.

Usage::

    uv run python benchmarks/sparse_lm_head_training.py
    uv run python benchmarks/sparse_lm_head_training.py --vocab-size 128256
    uv run python benchmarks/sparse_lm_head_training.py --seq-len 1024 --trials 5
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
        description="dense vs sparse LM head, end-to-end through DiffusionTrainer"
    )
    p.add_argument("--steps", type=int, default=12, help="timed steps per arm")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="interleaved A/B trials; arms alternate order between trials",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="32000 or 128256 are the two #61 asks for",
    )
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        default=[0.15, 0.50, 0.75],
        help="the decisive axis; see the module docstring",
    )
    p.add_argument(
        "--lora",
        action="store_true",
        help="wrap the student in LoRA instead of full finetune",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _output_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "unturtle-sparse-bench"
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


def _model(args, seed: int = 7):
    """Tiny-A2D Llama — the only family declaring `sparse_output_projection`."""
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    torch.manual_seed(seed)
    model = TinyA2DLlamaLMHeadModel(
        TinyA2DLlamaConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 2,
            num_hidden_layers=args.layers,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=max(args.seq_len * 2, 512),
        )
    )
    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0,
            ),
        )
    return model.to(args.device)


def _noised_batch(args, mask_ratio: float, seed: int = 0):
    """One fixed corrupted batch, shared by both arms.

    Built directly rather than via the process so the mask ratio is exact:
    sampling `t` would give a distribution around the target and make the
    decisive axis fuzzy.
    """
    torch.manual_seed(seed)
    B, L, V = args.batch_size, args.seq_len, args.vocab_size
    input_ids = torch.randint(1, V, (B, L))
    labels = input_ids.clone()
    diffusion_mask = torch.rand(B, L) < mask_ratio
    noised = input_ids.clone()
    noised[diffusion_mask] = 1  # any id acts as [MASK] for timing purposes
    return {
        "input_ids": noised.to(args.device),
        "labels": labels.to(args.device),
        "diffusion_mask": diffusion_mask.to(args.device),
        "timesteps": torch.full((B,), mask_ratio, device=args.device),
        "attention_mask": torch.ones(B, L, dtype=torch.long, device=args.device),
    }


def _trainer(args, model, sparse: bool):
    from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

    training_args = DiffusionTrainingArguments(
        output_dir=str(_output_dir()),
        per_device_train_batch_size=args.batch_size,
        max_steps=1,
        use_cpu=(args.device == "cpu"),
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to=[],
        sparse_lm_head=sparse,
    )
    return DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=[{"input_ids": [5, 6, 7]}],
        processing_class=_tokenizer(),
        data_collator=None,
    )


def _measure(args, trainer, model, batch) -> tuple[float, float]:
    """Median step time (ms) and peak allocated memory (MiB)."""
    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    timings: list[float] = []
    for step in range(args.warmup + args.steps):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()

        loss = trainer.compute_loss(model, dict(batch))
        loss.backward()
        model.zero_grad(set_to_none=True)

        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            timings.append(elapsed * 1000)

    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")
    return statistics.median(timings), peak


def main() -> None:
    args = _parse_args()

    print(
        f"device={args.device} B={args.batch_size} L={args.seq_len} "
        f"V={args.vocab_size} H={args.hidden_size} layers={args.layers} "
        f"{'LoRA' if args.lora else 'full-finetune'} "
        f"steps={args.steps} trials={args.trials}"
    )
    print()
    header = (
        f"{'mask':>6} {'dense ms':>9} {'sparse ms':>10} {'dt':>8} "
        f"{'dense MiB':>10} {'sparse MiB':>11} {'dmem':>8}"
    )
    print(header)
    print("-" * len(header))

    for ratio in args.mask_ratios:
        batch = _noised_batch(args, ratio)
        per_trial: list[tuple[float, float, float, float]] = []

        for trial in range(args.trials):
            order = [False, True] if trial % 2 == 0 else [True, False]
            result: dict[bool, tuple[float, float]] = {}
            for sparse in order:
                model = _model(args)
                trainer = _trainer(args, model, sparse)
                result[sparse] = _measure(args, trainer, model, batch)
                del model, trainer
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            per_trial.append((*result[False], *result[True]))

        dense_ms = statistics.median(t[0] for t in per_trial)
        dense_mem = statistics.median(t[1] for t in per_trial)
        sparse_ms = statistics.median(t[2] for t in per_trial)
        sparse_mem = statistics.median(t[3] for t in per_trial)

        d_time = sparse_ms / dense_ms - 1.0
        d_mem = sparse_mem / dense_mem - 1.0 if dense_mem == dense_mem else float("nan")
        print(
            f"{ratio:>6.2f} {dense_ms:>9.2f} {sparse_ms:>10.2f} {d_time:>+7.1%} "
            f"{dense_mem:>10.1f} {sparse_mem:>11.1f} {d_mem:>+7.1%}"
        )

    print()
    print(
        "Negative = sparse is better.  Expect the memory column to change sign\n"
        "as the mask ratio grows: the sparse path only avoids projecting the\n"
        "positions it skips.  MDLM training averages ~50% masking, so read the\n"
        "0.50 row as the default-recipe answer, not the 0.15 one."
    )


if __name__ == "__main__":
    main()
