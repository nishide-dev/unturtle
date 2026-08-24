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


"""#157 baseline — AR / MDLM / block_decode over one LLaDA checkpoint.

Runs ONLY existing Unturtle paths. No candidate code, no threshold tuning,
no winner selection. The Part-0 audit
(`docs/parallel-decoding-reference-audit.md` §7) froze these cells before any
number existed:

  checkpoint  GSAI-ML/LLaDA-8B-Instruct @
              08b83a6feb34df1a6011b80c3c00c7563e963b07 (MIT)
              — chosen because `mdlm` AND `block_decode` are both
              capability-valid on it (verified: supports() True for both),
              which mdlm-owt is not (no `_model_forward_with_cache`)
  cells       batch 1 / 8 / 32, output length 1024, plus a long-output cell
  speed       steady-state WALL-CLOCK latency and throughput; executed NFE
              rides along as an explanatory variable, never a denominator
  warmup      outside every timed region
  typed       CUDA OOM and unsupported become data, not omissions
  commit      commitment order from the loop's own stream_callback
  dependency  `dependency_slice` copy / reverse / kv_recall

The cache axis and the commit axis are recorded separately: a gain visible
only on the diagonal is a commit gain wearing a cache label.
"""

import argparse
import json
import pathlib
import time

CHECKPOINT = "GSAI-ML/LLaDA-8B-Instruct"
REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=1024)
    parser.add_argument(
        "--long-gen-length",
        type=int,
        default=2048,
        help="the long-output cell required by the Part-0 audit",
    )
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="confidence threshold for the parallel-commit arm",
    )
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "baseline"),
        default="smoke",
        help="smoke: wiring only, tiny budgets, NOT a baseline record",
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
    # The audit's central capability fact, asserted rather than assumed: this
    # baseline is only meaningful on a checkpoint where BOTH paths run.
    for name in ("mdlm", "block_decode"):
        algorithm = find_algorithm(name)
        if not algorithm.supports(model):
            raise RuntimeError(
                f"{args.checkpoint} does not support {name!r}; the #157 "
                "baseline needs a checkpoint where mdlm and block_decode are "
                "both capability-valid (mdlm-owt is not — it has no "
                "_model_forward_with_cache)"
            )
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        raise RuntimeError("no mask_token_id on the model config")
    return model, tokenizer, int(mask_id)


# The 2-D grid the Part-0 audit froze (cache x commit) is NOT fully
# realizable on Unturtle's implementation, and the smoke run found out why:
# `parallel_decode=True` refuses `alg='origin'` (the quota policy) and
# requires `use_cache=True`. Measured against the config validator:
#
#   alg            use_cache  parallel_decode   legal
#   origin         False      False             yes
#   origin         True       False             yes
#   origin         False      True              NO  (parallel needs cache)
#   origin         True       True              NO  (parallel refuses origin)
#   maskgit_plus   True       True              yes
#
# So the commit axis is entangled with `alg`: quota commit only exists on
# `origin`, and threshold commit only on a confidence-ordered alg. A cell that
# changed both at once would confound them, so the grid is realized as FOUR
# arms with `alg` recorded explicitly, and the missing corner is a typed
# `unsupported` cell rather than a silently omitted one.
ARMS = (
    # (label, cache_path, commit, alg, generate kwargs)
    ("mdlm_origin_quota", "no_cache", "quota", "origin", {"algorithm": "mdlm"}),
    (
        "block_decode_origin_quota",
        "prefix_cache",
        "quota",
        "origin",
        {"algorithm": "block_decode"},
    ),
    # Confidence-ordered commit WITHOUT the threshold, so the alg change is
    # separable from the parallel-commit change.
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

#: The corner the implementation forbids, recorded as data (#152).
UNSUPPORTED_CELLS = (
    {
        "arm": "mdlm_origin_threshold",
        "cache_path": "no_cache",
        "commit": "threshold",
        "alg": "origin",
        "status": "unsupported",
        "reason": "parallel_decode=True requires use_cache=True and refuses "
        "alg='origin'; threshold commit does not exist on the no-cache quota "
        "path in this implementation",
    },
)


def run_arm(model, mask_id, arm, batch_size, gen_length, args, capture=False):
    """One timed generation, returning (wall_seconds, executed_steps, trajectory)."""
    import torch

    label, _cache, commit, alg, kwargs = arm
    prompt = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=args.device)
    trajectory: list = []
    observed: list[int] = []

    def stream(step, total, x):
        observed.append(int(step))
        if capture:
            trajectory.append(x.detach().cpu().clone())

    # The block loop requires (max_length - prompt_len) % block_length == 0,
    # so max_length is set from the prompt width to make the generated span
    # exactly `gen_length` — the smoke run failed with
    # "gen_length (15) must be divisible by block_length (8)" when this was
    # hardcoded to gen_length + 1.
    call = dict(
        max_length=gen_length + prompt.shape[1],
        steps=args.steps,
        mask_token_id=mask_id,
        alg=alg,
        temperature=1.0,
        return_dict=False,
        step_callback=None,
        stream_callback=stream,
        **kwargs,
    )
    if kwargs.get("algorithm") == "block_decode":
        call["block_length"] = args.block_length
    if commit == "threshold":
        call["confidence_threshold"] = args.threshold

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        model.generate(prompt, **call)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - started
    return wall, (max(observed) if observed else 0), trajectory


def main():
    import torch

    from unturtle.eval.decoding_baseline import (
        baseline_cell_key,
        cache_path_class,
        commit_order_metrics,
        run_typed_cell,
        speed_cell,
    )
    from unturtle.eval.frontier import write_jsonl

    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    model, _tokenizer, mask_id = load(args)

    # Warmup outside every timed region: the #165 producers showed a fresh
    # process paying ~160 s of first-call compile.
    warm = argparse.Namespace(**vars(args))
    warm.steps = 2
    run_arm(model, mask_id, ARMS[0], 1, 64, warm)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    records = []
    for arm in ARMS:
        label, cache_path, commit, alg, _ = arm
        for gen_length in (args.gen_length, args.long_gen_length):
            for batch_size in batch_sizes:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                def run(bs, arm=arm, gen_length=gen_length):
                    return run_arm(model, mask_id, arm, bs, gen_length, args)

                cell = run_typed_cell(run, batch_size=batch_size)
                record = {
                    "arm": label,
                    **baseline_cell_key(cache_path=cache_path, commit=commit),
                    "cache_class": cache_path_class(cache_path),
                    "alg": alg,
                    "gen_length": gen_length,
                    "checkpoint": f"{args.checkpoint}@{args.revision}",
                    "mode": args.mode,
                    "status": cell["status"],
                }
                if cell["status"] == "ok":
                    wall, executed_steps, _ = cell["value"]
                    record["speed"] = speed_cell(
                        wall_seconds=wall,
                        batch_size=batch_size,
                        executed_nfe=executed_steps,
                        sequence_length=gen_length,
                    )
                    record["peak_memory_bytes"] = (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else None
                    )
                else:
                    record["reason"] = cell["reason"]
                records.append(record)
                print(json.dumps(record), flush=True)

    # Commitment order: one captured trajectory per arm at batch 1, because a
    # full trajectory is steps x batch x length.
    for arm in ARMS:
        label, cache_path, commit, alg, _ = arm
        try:
            _wall, _steps, trajectory = run_arm(
                model, mask_id, arm, 1, args.gen_length, args, capture=True
            )
        except torch.cuda.OutOfMemoryError as error:
            records.append(
                {
                    "arm": label,
                    "commit_order": {"status": "oom", "reason": str(error)},
                }
            )
            continue
        commit_metrics = (
            commit_order_metrics(trajectory, mask_id=mask_id)
            if len(trajectory) >= 2
            else {"status": "not_captured"}
        )
        records.append(
            {
                "arm": label,
                **baseline_cell_key(cache_path=cache_path, commit=commit),
                "alg": alg,
                "commit_order": commit_metrics,
                # No span metric here: this fixture declares no output spans,
                # so `answer_before_reasoning_rate` is unsupported (#157 B4).
                "answer_before_reasoning": {
                    "status": "unsupported",
                    "reason": "unconditional generation declares no "
                    "reasoning/answer spans",
                },
                "mode": args.mode,
            }
        )
        print(json.dumps(records[-1], default=str)[:400], flush=True)

    # The forbidden corner is data, not an omission (#152).
    records.extend(dict(cell, mode=args.mode) for cell in UNSUPPORTED_CELLS)
    write_jsonl(records, out / "baseline_cells.jsonl")
    print(f"wrote {len(records)} records to {out / 'baseline_cells.jsonl'}")


if __name__ == "__main__":
    main()
