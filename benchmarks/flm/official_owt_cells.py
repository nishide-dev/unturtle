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

"""#155 Stage 4/5: FLM/FMLM OWT — official evaluator column + #152 canonical
cell.  Mirrors benchmarks/elf/official_owt_cell.py (two SEPARATE evidence
columns, resumable eval).

Oracle decode semantics (trainer_base.py:404-418, frozen): plain
`batch_decode` of native gpt2 ids — NO pre-decode EOS masking (unlike ELF);
first-EOS handling lives inside the official evaluator.  Official entropy
is the mean per-sample unigram entropy over NATIVE ids (record_entropy runs
before decoding).

Primary decision cells (Stage-0):
    --algo flm  --steps 1024              (band: GenPPL 62.23 +-15%)
    --algo fmlm --steps 1    --gamma 1.0  (band: 168.30 +-15%)
    --algo fmlm --steps 32   --gamma 1.0  (band:  45.09 +-15%)

Usage:
    .venv/bin/python benchmarks/flm/official_owt_cells.py \
        --algo fmlm --steps 1 --gamma 1.0 --device cuda:0 --batch-size 16 \
        --num-samples 64 --seeds 42,43,44 --out benchmarks/results/fmlm_owt_1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
ORACLE_ROOT = REPO / "dev" / "repos" / "flm"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("flm", "fmlm"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="per seed; official commands use 16 (flm) / 64 (fmlm)",
    )
    parser.add_argument(
        "--seeds",
        default="42,43,44",
        help="comma list; the frozen band is judged on the median",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--owt-heldout", default="dev/local/owt/heldout_1024")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-device", default=None)
    return parser.parse_args()


def load_model(args):
    if args.algo == "flm":
        from unturtle_flm.loader import load_flm_model

        return load_flm_model(device=args.device)
    from unturtle_flm.loader import load_fmlm_model

    return load_fmlm_model(device=args.device)


def run_pack(model, args, *, num_samples, seed):
    from unturtle_flm.sampler import run_flm_request, run_fmlm_request

    class Request:
        kwargs = {
            "steps": args.steps,
            "gamma": args.gamma,
            "num_samples": num_samples,
            "seed": seed,
        }

    runner = run_flm_request if args.algo == "flm" else run_fmlm_request
    return runner(model, Request())


def generate(model, tokenizer, args):
    import torch

    seeds = [int(s) for s in args.seeds.split(",")]
    per_seed = {}
    all_ids = []
    executed = None
    start = time.perf_counter()
    for seed in seeds:
        ids_chunks = []
        remaining = args.num_samples
        offset = 0
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            # Collision-free across seeds at ANY batch size (#161 review
            # F1: seed+offset made 42/43/44 overlap at small batches) —
            # and a DISCLOSED deviation from the oracle's single
            # sequential stream (L.seed_everything once, main.py:415).
            result = run_pack(
                model, args, num_samples=batch, seed=seed * 1_000_003 + offset
            )
            executed = result["executed"]
            ids_chunks.append(result["tokens"].cpu())
            remaining -= batch
            offset += batch
        ids = torch.cat(ids_chunks, dim=0)
        per_seed[seed] = {
            "ids": ids,
            # Oracle decode semantics: plain batch_decode, no masking.
            "texts": tokenizer.batch_decode(ids),
        }
        all_ids.append(ids)
        print(f"  seed {seed}: {ids.shape[0]} samples", flush=True)
    return per_seed, torch.cat(all_ids, dim=0), executed, time.perf_counter() - start


def official_column(per_seed, max_length, eval_device):
    """The oracle's own Metrics (fp32 gpt2-large, first-EOS masking; entropy
    over NATIVE ids) — per seed, plus the median for the band verdict."""
    import torch

    sys.path.insert(0, str(ORACLE_ROOT))
    real_is_available = torch.cuda.is_available
    if eval_device == "cpu":
        torch.cuda.is_available = lambda: False
    try:
        import metrics as oracle_metrics

        rows = {}
        for seed, payload in per_seed.items():
            metric = oracle_metrics.Metrics(
                gen_ppl_eval_model_name_or_path="gpt2-large",
                eval_ppl_batch_size=8,
            )
            # The oracle moves its metrics with TrainerBase.to()
            # (trainer_base.py:136-139); standalone use must do the same or
            # torchmetrics raises a device mismatch on CUDA evaluation.
            metric.to(torch.device(eval_device))
            metric.record_entropy(payload["ids"])
            metric.record_generative_perplexity(
                payload["texts"], max_length, device=eval_device
            )
            rows[seed] = {
                "genppl_official": float(metric.gen_ppl.compute()),
                "entropy_official_native": float(metric.sample_entropy.compute()),
            }
        import statistics

        return {
            "per_seed": rows,
            "median_genppl": statistics.median(
                row["genppl_official"] for row in rows.values()
            ),
            "median_entropy": statistics.median(
                row["entropy_official_native"] for row in rows.values()
            ),
            "evaluator": {
                "model": "gpt2-large",
                "dtype": "float32",
                "device": eval_device,
                "semantics": "official FLM Metrics (first-EOS masking; "
                "entropy = mean per-sample unigram over NATIVE gpt2 ids)",
            },
        }
    finally:
        torch.cuda.is_available = real_is_available
        sys.path.remove(str(ORACLE_ROOT))
        for name in list(sys.modules):
            if name in ("metrics",) and not name.startswith("unturtle"):
                sys.modules.pop(name, None)


def canonical_column(texts, native_ids, args, device):
    """#152 protocol v1 cells over the pooled multi-seed sample set."""
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
        "genppl_evaluator": identity,
        "unigram_entropy": entropy,
        "sample_count": len(texts),
        "collapse_flags": [],
        **diversity_guards(native_ids),
    }
    heldout = pathlib.Path(args.owt_heldout)
    heldout_meta = heldout.parent / f"{heldout.name}.json"
    mauve_note = None
    if heldout.is_file() and heldout_meta.exists():
        import numpy as np

        from unturtle.eval import mauve_score

        meta = json.loads(heldout_meta.read_text())
        memmap = np.memmap(
            heldout,
            dtype=np.uint16,
            mode="r",
            shape=(meta["num_rows"], meta["block_size"]),
        )
        reference = [
            gpt2_tokenizer.decode(
                np.asarray(row, dtype=np.int64).tolist(), skip_special_tokens=True
            )
            for row in memmap[: len(texts)]
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

    from unturtle.eval.frontier import measure_throughput_cells

    def run_batch(batch_size, generator):
        derived_seed = int(
            torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
        )
        run_pack(model, args, num_samples=batch_size, seed=derived_seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def warmup():
        run_batch(1, torch.Generator().manual_seed(0))

    return measure_throughput_cells(run_batch, seed=0, warmup=warmup)


def main():
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from unturtle_flm.loader import (
        FLM_CHECKPOINT,
        FLM_REVISION,
        FMLM_CHECKPOINT,
        FMLM_REVISION,
    )

    from unturtle.eval.frontier import frontier_record, missing_cell, write_jsonl

    if args.resume:
        print("[1-3/5] resuming from saved samples/meta ...", flush=True)
        meta = json.loads((out / "generation_meta.json").read_text())
        executed = meta["executed"]
        cells = meta["cells"]
        peak_memory = meta["peak_memory_bytes"]
        generation_seconds = meta["generation_wall_seconds"]
        native_ids = torch.load(out / "native_ids.pt", weights_only=True)
        per_seed = {}
        for seed, payload in json.loads(
            (out / "per_seed_texts.json").read_text()
        ).items():
            per_seed[int(seed)] = {
                "texts": payload,
                "ids": torch.load(out / f"ids_{seed}.pt", weights_only=True),
            }
        texts = [t for payload in per_seed.values() for t in payload["texts"]]
    else:
        from transformers import AutoTokenizer

        print(f"[1/5] loading {args.algo} on {args.device} ...", flush=True)
        model = load_model(args)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        print("[2/5] throughput cells ...", flush=True)
        if args.skip_throughput:
            cells = {
                f"batch_{batch}": missing_cell("missing", "skipped by flag")
                for batch in (1, 8, 32)
            }
        else:
            cells = throughput_cells(model, args)

        print("[3/5] generating ...", flush=True)
        per_seed, native_ids, executed, generation_seconds = generate(
            model, tokenizer, args
        )
        peak_memory = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        )
        texts = [t for payload in per_seed.values() for t in payload["texts"]]
        torch.save(native_ids, out / "native_ids.pt")
        for seed, payload in per_seed.items():
            torch.save(payload["ids"], out / f"ids_{seed}.pt")
        (out / "per_seed_texts.json").write_text(
            json.dumps({str(seed): p["texts"] for seed, p in per_seed.items()})
        )
        (out / "generation_meta.json").write_text(
            json.dumps(
                {
                    "executed": executed,
                    "cells": cells,
                    "peak_memory_bytes": peak_memory,
                    "generation_wall_seconds": generation_seconds,
                }
            )
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_device = args.eval_device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("[4/5] official evaluator column ...", flush=True)
    official = official_column(per_seed, max_length=1024, eval_device=eval_device)
    print(
        f"  official: median genppl {official['median_genppl']:.3f} "
        f"entropy {official['median_entropy']:.3f}",
        flush=True,
    )

    print("[5/5] canonical #152 column ...", flush=True)
    quality, mauve_note = canonical_column(texts, native_ids, args, eval_device)

    if args.algo == "flm":
        checkpoint = f"{FLM_CHECKPOINT}@{FLM_REVISION[:8]}"
        tier_a_role = None  # one-hot Euclidean control; not a protocol role
    else:
        checkpoint = f"{FMLM_CHECKPOINT}@{FMLM_REVISION[:8]}"
        tier_a_role = "flow_map"

    record = frontier_record(
        family="onehot_flow" if args.algo == "flm" else "flow_map",
        method=args.algo,
        checkpoint=checkpoint,
        seed=int(args.seeds.split(",")[0]),
        tier_a_role=tier_a_role,
        provider={
            "distribution": "unturtle-flm",
            "version": "0.0.1",
            "entry_point": "flm",
        },
        quality=quality,
        systems={
            "nfe": executed["nfe"],
            "sequence_length": executed["max_length"],
            "solver": executed["solver"],
            "throughput": cells,
            "peak_memory_bytes": peak_memory,
            "generation_wall_seconds": generation_seconds,
        },
        decoding=executed,
        steps_requested=executed["steps_requested"],
        steps_executed=executed["steps_executed"],
        extra={
            "official_column": official,
            "mauve_note": mauve_note,
            "seeds": args.seeds,
            "per_seed_seed_rule": (
                "seed*1_000_003 + samples_generated_so_far — N derived seeds, "
                "a DISCLOSED deviation from the oracle's single sequential "
                "stream (official scripts: seed_everything(1) once)"
            ),
        },
    )
    write_jsonl([record], out / "frontier_record.jsonl")
    print(json.dumps({"official": official, "canonical_quality": quality}, indent=2))
    print(f"records written to {out}")


if __name__ == "__main__":
    main()
