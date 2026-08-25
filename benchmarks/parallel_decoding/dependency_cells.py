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

"""#157 step 3 — dependency correctness across the four decode arms.

Runs after the prompted preflight cleared its gate (immediate EOS 62/64 -> 0/9,
prefix agreement 1.00, residual masks 0), which established that step 2's
collapse was the mask-as-prompt fixture mismatch. Precise scope carried
forward: that explains the OBSERVED collapse; it does not establish that
cache / alg / commit policy are free of semantic failure in general. This
producer is where that question is actually measured.

Frozen conditions (all five pinned by review before implementation):

1. Scoring stops at the first EOS. The fixed-width raw suffix is retained for
   diagnostics, but the semantic answer is the pre-EOS span only: post-EOS
   tokens are NEVER reattached to the answer. Decoding the whole canvas with
   ``skip_special_tokens=True`` would splice EOS-separated fragments into one
   answer — the exact mistake that made step 2's column unreadable.
2. Generation length is reported PER KIND. ``no_eos_fraction`` is its own
   column; first-EOS mean/median are computed over EOS-bearing rows only and
   never folded into a single ``None = 1024`` average. The preflight found the
   maskgit arms polarize by kind (copy fills the canvas while reverse and
   kv_recall stop in single digits); one mean hides that by cancellation.
3. bs1 is the primary semantic cell, bs8 a batching guard over the same task
   set. Task-local RNG is not aligned across batch shapes, so a per-item
   difference is NOT called a batching semantics change; only per-kind
   aggregates are compared.
4. Correctness and commitment order are reported together for bs1.
   ``answer_before_reasoning`` stays typed ``unsupported`` — dependency_slice
   declares no in-output spans, and the shared helper already returns that.
5. A kind where the EXACT arm itself carries no target signal is typed
   ``reference_floor / undecidable`` for every arm: a cache arm that is merely
   as bad as the reference is not evidence of preservation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
from typing import Any

from unturtle.eval.decoding_baseline import (
    CommitReducer,
    answer_before_reasoning,
    baseline_cell_key,
    cache_path_class,
)
from unturtle.eval.dependency_slice import (
    answer_span,
    dependency_length_diagnostics,
    dependency_tasks,
    score_extraction_pair,
)

RUN_LABEL = (
    "dependency correctness — SEPARATE run from the #157 speed cells, the "
    "(fixture-invalid) step-2 quality cells, and the prompted preflight"
)

FIXTURE_SEED = 0
N_PER_KIND = 8
TASK_LENGTH = 8

# A kind is at the reference floor when the exact arm scores at or below this
# on coupled accuracy: there is no signal for a cache arm to preserve.
REFERENCE_FLOOR_ACCURACY = 0.05


def _load_arms() -> tuple:
    """Load ARMS from the baseline producer by path (standalone scripts)."""
    path = pathlib.Path(__file__).with_name("baseline_cells.py")
    spec = importlib.util.spec_from_file_location("_pd_baseline_cells", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the baseline producer at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.ARMS)


ARMS = _load_arms()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--revision", default="08b83a6feb34df1a6011b80c3c00c7563e963b07"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=1024)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--out", default="benchmarks/results/pd_dependency")
    return parser.parse_args()


def load(args: argparse.Namespace):
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


def generate(model, tokenizer, mask_id, arm, tasks, args, *, batch_size, capture=False):
    """Generate for every task. Prompts are batched only within one kind-order
    slice so the task set is identical across batch sizes."""
    import torch

    _label, _cache, commit, alg, kwargs = arm
    suffixes: list[list[int]] = []
    reducer = CommitReducer(mask_id=mask_id) if capture else None
    prefix_ok = True

    torch.manual_seed(args.seed)
    for start in range(0, len(tasks), batch_size):
        chunk = tasks[start : start + batch_size]
        # The checkpoint is INSTRUCT-tuned and ships a chat template. Passing
        # the bare prompt string leaves the model outside the format it was
        # tuned for, and it rambles instead of answering: measured on this
        # fixture, raw prompting scores ~0 coupled accuracy on all three kinds
        # while the same prompts under the template reproduce `copy` exactly.
        # A reference arm with no task signal makes every cache comparison
        # undecidable (condition 5), so the template is part of the fixture.
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task.prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for task in chunk
        ]
        encoded = tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(args.device)
        if mask_id in input_ids.reshape(-1).tolist():
            raise RuntimeError("a prompt contains the mask sentinel")
        prompt_len = input_ids.shape[1]
        call: dict[str, Any] = {
            "max_length": prompt_len + args.gen_length,
            "steps": args.steps,
            "mask_token_id": mask_id,
            "alg": alg,
            "temperature": 1.0,
            "return_dict": False,
            **kwargs,
        }
        if kwargs.get("algorithm") == "block_decode":
            call["block_length"] = args.block_length
        if commit == "threshold":
            call["confidence_threshold"] = args.threshold
        if reducer is not None:
            call["stream_callback"] = lambda step, _total, state: reducer.update(
                step, state
            )
        with torch.no_grad():
            generated = model.generate(input_ids, **call)
        for row_index, row in enumerate(generated.tolist()):
            if row[:prompt_len] != input_ids[row_index].tolist():
                prefix_ok = False
            suffixes.append(row[prompt_len:])
    return suffixes, reducer, prefix_ok


def main() -> None:
    args = parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    tasks = dependency_tasks(
        n_per_kind=N_PER_KIND, seed=FIXTURE_SEED, length=TASK_LENGTH
    )
    model, tokenizer, mask_id = load(args)
    eos_id = int(tokenizer.eos_token_id)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    floor_kinds: set[str] = set()

    for batch_size in batch_sizes:
        primary = batch_size == batch_sizes[0]
        for arm in ARMS:
            label, cache_path, commit, alg, _kwargs = arm
            suffixes, reducer, prefix_ok = generate(
                model,
                tokenizer,
                mask_id,
                arm,
                tasks,
                args,
                batch_size=batch_size,
                capture=primary,
            )
            # Condition 1: score the pre-EOS span only.
            answers = [answer_span(s, eos_id=eos_id) for s in suffixes]
            texts = [tokenizer.decode(a, skip_special_tokens=True) for a in answers]
            # The frozen extraction rule, shared with the re-scorer so the two
            # cannot diverge. `str.split()` scored ~0 on answers the model had
            # largely right, because an instruct model prefixes prose and joins
            # values with commas.
            scores = score_extraction_pair(tasks, texts)

            # Condition 2: per-kind generation length, never folded together.
            per_kind: dict[str, Any] = {}
            for kind in sorted({task.kind for task in tasks}):
                rows = [
                    s
                    for s, task in zip(suffixes, tasks, strict=True)
                    if task.kind == kind
                ]
                kind_tasks = [t for t in tasks if t.kind == kind]
                kind_texts = [
                    x for x, t in zip(texts, tasks, strict=True) if t.kind == kind
                ]
                kind_scores = score_extraction_pair(kind_tasks, kind_texts)
                per_kind[kind] = {
                    "length": dependency_length_diagnostics(
                        rows, eos_id=eos_id, mask_id=mask_id
                    ),
                    "exact_match": kind_scores["exact_match"],
                    "coupled_token_accuracy": kind_scores["coupled_token_accuracy"],
                    "length_mismatch_fraction": kind_scores["length_mismatch_fraction"],
                }

            record: dict[str, Any] = {
                "arm": label,
                **baseline_cell_key(cache_path=cache_path, commit=commit),
                "cache_class": cache_path_class(cache_path),
                "alg": alg,
                "kind": "dependency",
                "run": RUN_LABEL,
                "batch_size": batch_size,
                "cell_role": "primary_semantic" if primary else "batching_guard",
                "prompt_prefix_agreement": 1.0 if prefix_ok else 0.0,
                "prompt_rendering": "chat_template(add_generation_prompt=True)",
                "scores": scores,
                "per_kind": per_kind,
                "task_count": len(tasks),
            }
            if not primary:
                record["batching_note"] = (
                    "task-local RNG is NOT aligned across batch shapes; a "
                    "per-item difference here is not a batching semantics "
                    "change. Compare per-kind aggregates only (condition 3)."
                )
            # Condition 5: the exact arm defines the floor, per kind.
            if label == ARMS[0][0]:
                floor_kinds = {
                    kind
                    for kind, cell in per_kind.items()
                    if cell["primary"]["coupled_token_accuracy"]
                    <= REFERENCE_FLOOR_ACCURACY
                }
            record["reference_floor_kinds"] = sorted(floor_kinds)
            for kind in floor_kinds:
                record["per_kind"][kind]["measurement_status"] = (
                    "reference_floor / undecidable"
                )
            # Condition 4: commitment order beside correctness, bs1 only.
            if reducer is not None:
                commit_stats = reducer.result()
                record["commit_order"] = {
                    "steps_observed": commit_stats["steps_observed"],
                    "tokens_committed_per_step": commit_stats[
                        "tokens_committed_per_step"
                    ],
                    "revision_events": commit_stats["revision_events"],
                    "uncommitted_positions": commit_stats["uncommitted_positions"],
                }
                record["answer_before_reasoning"] = answer_before_reasoning(
                    normalized_commit_step=commit_stats["normalized_commit_step"],
                    spans=None,
                )
            records.append(record)
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in ("per_kind", "commit_order")
                    }
                ),
                flush=True,
            )
            (out / f"suffixes_{label}_bs{batch_size}.json").write_text(
                json.dumps(suffixes)
            )

    (out / "dependency_cells.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    print(f"wrote {len(records)} records to {out / 'dependency_cells.jsonl'}")


if __name__ == "__main__":
    main()
