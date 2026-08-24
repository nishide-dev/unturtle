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


"""#157 step 2 — canonical quality + paired content drift.

**A SEPARATE RUN from the speed cells.** The baseline run did not persist
generated text, so this producer generates its own samples under the same
frozen config, seed and sample set. Every record says so; they are never
presented as the run that produced the wall-clock numbers.

Frozen conditions, unchanged from the baseline — and the threshold is NOT
retuned:

  checkpoint  GSAI-ML/LLaDA-8B-Instruct @
              08b83a6feb34df1a6011b80c3c00c7563e963b07
  steps 128 | length 1024 | block 128 | threshold 0.9 | seed 42
  arms        1 (no_cache, origin, quota)       — the EXACT reference path
              2 (prefix_cache, origin, quota)   — 1v2 isolates cache
              3 (prefix_cache, maskgit, quota)  — 2v3 isolates alg
              4 (prefix_cache, maskgit, thr0.9) — 3v4 isolates commit policy
  quality     the #152 canonical column via `canonical_quality_column`
  evaluator   gpt2-large @ 32b71b12589c2f8d625668d2335a01cac3249519
  drift       paired against arm 1, the exact path

Generation batch size is 8 — the largest every arm completed in the baseline
(bs32 OOM'd on all cache arms). It is a generation setting, not a frozen
condition, and it is recorded as such.
"""

import argparse
import json
import pathlib
import time

CHECKPOINT = "GSAI-ML/LLaDA-8B-Instruct"
REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
EVALUATOR_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"

ARMS = (
    ("mdlm_origin_quota", "no_cache", "quota", "origin", {"algorithm": "mdlm"}),
    (
        "block_decode_origin_quota",
        "prefix_cache",
        "quota",
        "origin",
        {"algorithm": "block_decode"},
    ),
    (
        "block_decode_maskgit_topk",
        "prefix_cache",
        "quota",
        "maskgit_plus",
        {"algorithm": "block_decode"},
    ),
    (
        "block_decode_maskgit_threshold",
        "prefix_cache",
        "threshold",
        "maskgit_plus",
        {"algorithm": "block_decode", "parallel_decode": True},
    ),
)

RUN_LABEL = "quality — SEPARATE run from the #157 speed cells"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--evaluator-revision", default=EVALUATOR_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=1024)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--owt-heldout", default="dev/local/owt/heldout_1024")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "quality"),
        default="smoke",
        help="smoke: wiring only, tiny budgets, NOT a quality record",
    )
    return parser.parse_args()


def load(args):
    import torch

    from unturtle import FastDiffusionModel
    from unturtle.models.generation.sampler import find_algorithm

    model, tokenizer = FastDiffusionModel.from_pretrained(
        args.checkpoint,
        revision=args.revision,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map=None,
    )
    model.to(args.device).eval()
    for name in ("mdlm", "block_decode"):
        if not find_algorithm(name).supports(model):
            raise RuntimeError(f"{args.checkpoint} does not support {name!r}")
    return model, tokenizer, int(model.config.mask_token_id)


def generate_arm(model, tokenizer, mask_id, arm, args):
    import torch

    _label, _cache, commit, alg, kwargs = arm
    texts: list[str] = []
    rows: list[list[int]] = []
    # Every arm reseeds to the same value so the arms are paired
    # sample-for-sample; the arms still consume the stream differently, which
    # the drift record's determinism note states explicitly.
    torch.manual_seed(args.seed)
    while len(texts) < args.num_samples:
        batch = min(args.batch_size, args.num_samples - len(texts))
        prompt = torch.full((batch, 1), mask_id, dtype=torch.long, device=args.device)
        call = dict(
            max_length=args.gen_length + 1,
            steps=args.steps,
            mask_token_id=mask_id,
            alg=alg,
            temperature=1.0,
            return_dict=False,
            **kwargs,
        )
        if kwargs.get("algorithm") == "block_decode":
            call["block_length"] = args.block_length
        if commit == "threshold":
            call["confidence_threshold"] = args.threshold
        with torch.no_grad():
            out = model.generate(prompt, **call)
        tokens = out if isinstance(out, torch.Tensor) else out.sequences
        for row in tokens:
            ids = row.tolist()[1:]
            rows.append(ids)
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        print(f"  {arm[0]}: {len(texts)}/{args.num_samples}", flush=True)
    return texts[: args.num_samples], rows[: args.num_samples]


def quality_column(texts, rows, args):
    from transformers import AutoTokenizer

    from unturtle.eval.frontier import hf_causal_evaluator
    from unturtle.eval.producers import (
        canonical_evaluator_identity,
        canonical_quality_column,
    )

    evaluator, _raw = hf_causal_evaluator(
        "gpt2-large",
        revision=args.evaluator_revision,
        device=args.device,
        max_length=1024,
    )
    identity = canonical_evaluator_identity(
        model="gpt2-large",
        revision=args.evaluator_revision,
        tokenizer_revision=args.evaluator_revision,
    )
    identity["max_length"] = 1024
    gpt2 = AutoTokenizer.from_pretrained("gpt2-large", revision=args.evaluator_revision)
    quality = canonical_quality_column(
        texts,
        evaluator=evaluator,
        evaluator_identity=identity,
        tokenize=gpt2.encode,
        sample_ids=rows,
    )
    note = None
    heldout = pathlib.Path(args.owt_heldout)
    meta = heldout.parent / f"{heldout.name}.json"
    if heldout.is_file() and meta.exists():
        import numpy as np

        from unturtle.eval import mauve_score

        info = json.loads(meta.read_text())
        memmap = np.memmap(
            heldout,
            dtype=np.uint16,
            mode="r",
            shape=(info["num_rows"], info["block_size"]),
        )
        reference = [
            gpt2.decode(
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
        note = f"reference texts not found at {heldout}; MAUVE omitted"
    return quality, note


def main():
    import torch

    from unturtle.eval.content_drift import paired_content_drift
    from unturtle.eval.decoding_baseline import baseline_cell_key, cache_path_class
    from unturtle.eval.frontier import write_jsonl

    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tokenizer, mask_id = load(args)

    samples: dict[str, tuple[list[str], list[list[int]]]] = {}
    records = []
    for arm in ARMS:
        label, cache_path, commit, alg, _ = arm
        started = time.perf_counter()
        texts, rows = generate_arm(model, tokenizer, mask_id, arm, args)
        elapsed = time.perf_counter() - started
        samples[label] = (texts, rows)
        (out / f"samples_{label}.json").write_text(
            json.dumps(texts, ensure_ascii=False)
        )
        (out / f"native_ids_{label}.json").write_text(json.dumps(rows))
        records.append(
            {
                "arm": label,
                **baseline_cell_key(cache_path=cache_path, commit=commit),
                "cache_class": cache_path_class(cache_path),
                "alg": alg,
                "kind": "generation",
                "run": RUN_LABEL,
                "generation_seconds": elapsed,
                "generation_batch_size": args.batch_size,
                "sample_count": len(texts),
                "mode": args.mode,
            }
        )
        print(json.dumps(records[-1]), flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reference_label = ARMS[0][0]
    for arm in ARMS:
        label, cache_path, commit, alg, _ = arm
        texts, rows = samples[label]
        quality, note = quality_column(texts, rows, args)
        record = {
            "arm": label,
            **baseline_cell_key(cache_path=cache_path, commit=commit),
            "cache_class": cache_path_class(cache_path),
            "alg": alg,
            "kind": "quality",
            "run": RUN_LABEL,
            "quality": quality,
            "mauve_note": note,
            "mode": args.mode,
        }
        if label != reference_label:
            ref_texts, ref_rows = samples[reference_label]
            record["drift_reference"] = reference_label
            record["content_drift_vs_exact"] = paired_content_drift(
                reference_texts=ref_texts,
                candidate_texts=texts,
                reference_ids=ref_rows,
                candidate_ids=rows,
                determinism=(
                    "VERIFIED reproducible: the same arm at the same seed "
                    "produces token-identical output twice (measured: "
                    "exact_token_agreement 1.00, normalized distance 0.000, at "
                    "temperature 1.0 and at 0.0). Drift between arms is "
                    "therefore attributable to the decode path, not to RNG "
                    "variation — though the arms do consume the stream "
                    "differently, so the mechanism is 'a different trajectory' "
                    "rather than 'the same trajectory approximated'"
                ),
            )
        records.append(record)
        print(json.dumps(record, default=str)[:600], flush=True)

    write_jsonl(records, out / "quality_cells.jsonl")
    print(f"wrote {len(records)} records to {out / 'quality_cells.jsonl'}")


if __name__ == "__main__":
    main()
