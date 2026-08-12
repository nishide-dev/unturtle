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

"""#153 Stage 3: ELF-B OWT — official evaluator column + #152 canonical cell.

Two SEPARATE evidence columns (frozen rule: a mismatch between them is not
a parity failure until evaluator/tokenization/EOS semantics are localized):

1. **official** — the oracle's own `Metrics` (gpt2-large bf16, first-EOS
   masking, mean per-sample unigram entropy), imported from the official
   checkout at dev/repos/elf so no transcription can drift;
2. **canonical** — #152 protocol v1: corpus GenPPL via
   `unturtle.eval.frontier.hf_causal_evaluator` (all tokens scored — the
   documented EOS divergence), POOLED text unigram entropy under the
   gpt2-large tokenizer, guard trio over T5 ids, optional MAUVE against
   #130's OWT held-out texts, typed batch 1/8/32 throughput cells.

Usage (single GPU, ~30-60 min at batch 8):
    .venv/bin/python benchmarks/elf/official_owt_cell.py \
        --device cuda:0 --batch-size 8 --num-samples 1000 --steps 32 \
        --sde-gamma 1.5 --out benchmarks/results/elf_b_owt_32
    # 64-step reference cell: --steps 64 --sde-gamma 1.0
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ORACLE_SRC = REPO / "dev" / "repos" / "elf" / "src"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sde-gamma", type=float, default=1.5)
    parser.add_argument("--self-cond-cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--owt-heldout",
        default="dev/local/owt/heldout_1024",
        help="#130 packed held-out rows for the MAUVE reference (optional)",
    )
    parser.add_argument(
        "--skip-throughput",
        action="store_true",
        help="skip batch 1/8/32 cells (record them as missing with reason)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip generation: reuse samples.jsonl + generation_meta.json in "
        "--out (evaluation can then run CPU-only via CUDA_VISIBLE_DEVICES='')",
    )
    parser.add_argument(
        "--eval-device",
        default=None,
        help="device for the canonical evaluator (defaults to --device)",
    )
    return parser.parse_args()


def generate_samples(model, tokenizer, args):
    """Official-config generation, sharded into batches under ONE seed
    stream (eval.py single-rank semantics: one global seed, batches drawn
    sequentially from the same stream)."""
    import torch
    from unturtle_elf.sampler import run_generation_request

    class Request:
        kwargs = {}

    texts, all_ids = [], []
    executed = None
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    remaining = args.num_samples
    while remaining > 0:
        batch = min(args.batch_size, remaining)
        request = Request()
        request.kwargs = {
            "solver": "sde",
            "steps": args.steps,
            "sde_gamma": args.sde_gamma,
            "self_cond_cfg_scale": args.self_cond_cfg_scale,
            "cfg_scale": 1.0,
            "time_schedule": "logit_normal",
            "num_samples": batch,
            # N per-batch derived seeds — a DISCLOSED deviation from the
            # oracle script's single sequential stream (#160 review F4);
            # distributionally equivalent given the reference's own CUDA
            # global-RNG behavior, recorded in extra.per_batch_seed_rule.
            "seed": args.seed + len(all_ids),
        }
        result = run_generation_request(model, request)
        executed = result["executed"]
        ids = result["tokens"].cpu()
        all_ids.append(ids)
        for row in ids:
            texts.append(tokenizer.decode(row.tolist(), skip_special_tokens=True))
        remaining -= batch
        print(
            f"  generated {len(texts)}/{args.num_samples} "
            f"({time.perf_counter() - start:.0f}s)",
            flush=True,
        )
    import torch as _torch

    return texts, _torch.cat(all_ids, dim=0), executed, time.perf_counter() - start


def official_column(texts, max_length, eval_device="cuda"):
    """The oracle's own evaluator, imported from the official checkout.

    The oracle's Metrics hardcodes `cuda if available`; unsloth conversely
    refuses to IMPORT without a visible GPU — so a CPU evaluation cannot use
    CUDA_VISIBLE_DEVICES="".  For eval_device="cpu" we patch
    torch.cuda.is_available to False around the oracle call only (bf16
    semantics preserved; slower, numerically equivalent accumulation)."""
    import torch

    sys.path.insert(0, str(ORACLE_SRC))
    real_is_available = torch.cuda.is_available
    if eval_device == "cpu":
        torch.cuda.is_available = lambda: False
    try:
        from utils.metrics_utils import Metrics

        metrics = Metrics(
            gen_ppl_eval_model_name_or_path="gpt2-large",
            eval_ppl_batch_size=8,
            eval_context_size=1024,
        )
        result = metrics.record_generative_perplexity(texts, max_length=max_length)
        return {
            "genppl_official": float(result["ppl"]),
            "mean_entropy_official": float(result["mean_entropy"]),
            "evaluator": {
                "model": "gpt2-large",
                "dtype": "bfloat16",
                # bf16-on-CPU and bf16-on-CUDA reduce in different orders;
                # two records with the same identity but different devices
                # are distinguishable (#160 review F6).
                "device": eval_device,
                "semantics": "official ELF Metrics (first-EOS masking, "
                "mean per-sample unigram entropy)",
            },
        }
    finally:
        torch.cuda.is_available = real_is_available
        sys.path.remove(str(ORACLE_SRC))
        for name in list(sys.modules):
            if name.split(".")[0] in ("modules", "utils", "configs") and not (
                name.startswith("unturtle")
            ):
                sys.modules.pop(name, None)


def canonical_column(texts, t5_ids, args, device):
    """#152 protocol v1 quality cells."""
    from transformers import AutoTokenizer

    from unturtle.eval import diversity_guards
    from unturtle.eval.frontier import (
        generative_perplexity,
        hf_causal_evaluator,
        text_unigram_entropy,
    )

    evaluator, identity = hf_causal_evaluator(
        "gpt2-large", revision="main", device=device, max_length=1024
    )
    genppl = generative_perplexity(
        texts, evaluator=evaluator, evaluator_identity=identity
    )
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    entropy = text_unigram_entropy(
        texts, tokenize=lambda text: gpt2_tokenizer.encode(text)
    )
    quality = {
        "genppl": genppl["genppl"],
        "genppl_evaluator": {**identity, "revision": identity["revision"]},
        "unigram_entropy": entropy,
        "sample_count": len(texts),
        "collapse_flags": [],
        **diversity_guards(t5_ids),
    }
    heldout = pathlib.Path(args.owt_heldout)
    mauve_note = None
    if heldout.exists():
        import torch as _torch

        from unturtle.eval import mauve_score

        rows = _torch.load(heldout / "rows.pt", weights_only=True)[: len(texts)]
        reference = [
            gpt2_tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in rows
        ]
        quality["mauve"] = mauve_score(
            reference, texts, featurize_model_name="gpt2", max_text_length=256
        )
        quality["mauve_settings"] = {
            "featurize_model_name": "gpt2",
            "max_text_length": 256,
            "num_buckets": "auto",
            "reference": "#130 OWT held-out (gpt2-decoded packed rows)",
        }
    else:
        mauve_note = f"reference texts not found at {heldout}; MAUVE omitted"
    return quality, mauve_note


def throughput_cells(model, args):
    import torch
    from unturtle_elf.sampler import run_generation_request

    from unturtle.eval.frontier import measure_throughput_cells

    class Request:
        kwargs = {}

    def run_batch(batch_size, generator):
        # The CELL's single generator owns the stream: each batch's seed is
        # DRAWN from it, so a per-batch reset cannot masquerade as protocol
        # compliance (#160 review F3).
        derived_seed = int(
            torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
        )
        request = Request()
        request.kwargs = {
            "solver": "sde",
            "steps": args.steps,
            "sde_gamma": args.sde_gamma,
            "self_cond_cfg_scale": args.self_cond_cfg_scale,
            "num_samples": batch_size,
            "seed": derived_seed,
        }
        run_generation_request(model, request)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def warmup():
        run_batch(1, torch.Generator().manual_seed(args.seed))

    return measure_throughput_cells(run_batch, seed=args.seed, warmup=warmup)


def main():
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from unturtle_elf.loader import (
        DEFAULT_CHECKPOINT,
        DEFAULT_REVISION,
        load_elf_model,
    )

    from unturtle.eval.frontier import (
        FRONTIER_PROTOCOL_VERSION,
        cell,
        frontier_record,
        missing_cell,
        write_jsonl,
    )

    if args.resume:
        print("[1-3/5] resuming from saved samples/meta ...", flush=True)
        texts = [
            json.loads(line)["text"]
            for line in (out / "samples.jsonl").read_text().splitlines()
            if line.strip()
        ]
        meta = json.loads((out / "generation_meta.json").read_text())
        executed = meta["executed"]
        cells = meta["cells"]
        peak_memory = meta["peak_memory_bytes"]
        warmup_seconds = meta["warmup_seconds"]
        generation_seconds = meta["generation_wall_seconds"]
        t5_ids = torch.load(out / "t5_ids.pt", weights_only=True)
    else:
        print(f"[1/5] loading ELF-B on {args.device} ...", flush=True)
        from transformers import AutoTokenizer

        model = load_elf_model(device=args.device)
        tokenizer = AutoTokenizer.from_pretrained("t5-small", legacy=True)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        print("[2/5] throughput cells ...", flush=True)
        if args.skip_throughput:
            cells = {
                f"batch_{batch}": missing_cell("missing", "skipped by flag")
                for batch in (1, 8, 32)
            }
            warmup_seconds = None
        else:
            warmup_start = time.perf_counter()
            cells = throughput_cells(model, args)
            warmup_seconds = time.perf_counter() - warmup_start

        print(f"[3/5] generating {args.num_samples} samples ...", flush=True)
        texts, t5_ids, executed, generation_seconds = generate_samples(
            model, tokenizer, args
        )
        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        )
        (out / "samples.jsonl").write_text(
            "\n".join(json.dumps({"text": text}) for text in texts) + "\n"
        )
        torch.save(t5_ids, out / "t5_ids.pt")
        (out / "generation_meta.json").write_text(
            json.dumps(
                {
                    "executed": executed,
                    "cells": cells,
                    "peak_memory_bytes": peak_memory,
                    "warmup_seconds": warmup_seconds,
                    "generation_wall_seconds": generation_seconds,
                }
            )
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("[4/5] official evaluator column ...", flush=True)
    official = official_column(
        texts, max_length=1024, eval_device=args.eval_device or "cuda"
    )
    print(f"  official: {official}", flush=True)

    print("[5/5] canonical #152 column ...", flush=True)
    eval_device = args.eval_device or (
        "cpu" if not torch.cuda.is_available() else args.device
    )
    quality, mauve_note = canonical_column(texts, t5_ids, args, eval_device)

    record = frontier_record(
        family="embedding_flow",
        method="elf",
        checkpoint=f"{DEFAULT_CHECKPOINT}@{DEFAULT_REVISION[:8]}",
        seed=args.seed,
        tier_a_role="embedding_flow",
        provider={
            "distribution": "unturtle-elf",
            "version": "0.0.1",
            "entry_point": "elf",
        },
        quality=quality,
        systems={
            "nfe": executed["nfe"],
            "sequence_length": executed["max_length"],
            "solver": executed["solver"],
            "throughput": cells,
            "peak_memory_bytes": peak_memory,
            "warmup_seconds": warmup_seconds,
            "generation_wall_seconds": generation_seconds,
        },
        decoding=executed,
        steps_requested=executed["steps_requested"],
        steps_executed=executed["steps_executed"],
        extra={
            "official_column": official,
            "mauve_note": mauve_note,
            "per_batch_seed_rule": (
                "N per-batch derived seeds (seed + samples_generated_so_far) "
                "— deviation from the oracle's single sequential stream"
            ),
            "protocol_version": FRONTIER_PROTOCOL_VERSION,
        },
    )
    write_jsonl([record], out / "frontier_record.jsonl")
    print(json.dumps({"official": official, "canonical_quality": quality}, indent=2))
    print(f"records written to {out}")


if __name__ == "__main__":
    main()
