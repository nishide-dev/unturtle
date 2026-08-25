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

"""#157 — MANDATORY preflight for the step-3 dependency slice.

Not an independent study. Step 2's quality run is typed FIXTURE INVALID /
PROTOCOL DEVIATION for cross-arm comparison, because its "unconditional"
fixture does not mean the same thing to the two paths:

- the plain LLaDA loop computes ``mask_index = x == mask_token_id`` over the
  WHOLE sequence, so the single leading mask sentinel handed in as the prompt
  is itself denoised (``generation_utils.py`` L120);
- block decode sets ``current_block_start = prompt_len + block_idx *
  block_length`` and scopes ``mask_index_block`` to the current block only
  (``block_decode_mixin.py`` L201, L258), so that same leading mask stays
  FIXED as prompt / cache context and is never denoised.

Same token tensor, different conditioning. Dropping the first column after the
fact does not undo the divergent generation history, and this is a leading
candidate cause of the cached arms collapsing to EOS. Note also that the block
loop has NO EOS early-stop branch (its only ``break`` is block completion), so
"the handler stopped early" was never the explanation.

This preflight removes the fixture mismatch: prompts come from the real step-3
``dependency_tasks`` fixture and contain ZERO mask sentinels, so every arm
receives a token-identical prompt prefix under the same conditioning.

The primary diagnostic is the raw generated suffix IDS, not decoded text —
``skip_special_tokens=True`` deletes exactly the evidence in question.

Frozen conditions are inherited unchanged from the step-2 run (checkpoint,
seed, temperature, steps, gen/block length, threshold 0.9). No new threshold
selection. The prompt set is fixed BEFORE any output is inspected.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from unturtle.eval.decoding_baseline import baseline_cell_key, cache_path_class


def _load_arms() -> tuple:
    """Load ARMS from the baseline producer by path.

    These producers are standalone scripts (no package ``__init__``), so a
    relative import fails. Loading by path keeps ONE definition of the arms
    instead of a copy that can silently drift from the baseline's.
    """
    import importlib.util

    path = pathlib.Path(__file__).with_name("baseline_cells.py")
    spec = importlib.util.spec_from_file_location("_pd_baseline_cells", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the baseline producer at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.ARMS)


ARMS = _load_arms()

RUN_LABEL = (
    "prompted preflight — step-3 gate; SEPARATE run from the #157 speed cells "
    "and from the (fixture-invalid) step-2 quality cells"
)

# Frozen BEFORE any output was inspected: the first `PER_KIND` tasks of each
# kind from the step-3 fixture at its pinned seed.
FIXTURE_SEED = 0
PER_KIND = 3


def frozen_prompts() -> list[dict[str, str]]:
    """The frozen prompt set: representative copy / reverse / kv_recall."""
    from unturtle.eval.dependency_slice import dependency_tasks

    tasks = dependency_tasks(n_per_kind=8, seed=FIXTURE_SEED, length=8)
    chosen: list[dict[str, str]] = []
    seen: dict[str, int] = {}
    for task in tasks:
        if seen.get(task.kind, 0) >= PER_KIND:
            continue
        seen[task.kind] = seen.get(task.kind, 0) + 1
        chosen.append(
            {
                "name": task.name,
                "kind": task.kind,
                "prompt": task.prompt,
                "target": " ".join(task.target),
            }
        )
    return chosen


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
    parser.add_argument("--out", default="benchmarks/results/pd_preflight")
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
    mask_id = int(model.config.mask_token_id)
    return model, tokenizer, mask_id


def suffix_diagnostics(
    suffix: list[int], *, mask_id: int, eos_id: int
) -> dict[str, object]:
    """Everything the gate decides on, measured on RAW ids."""
    first_eos = suffix.index(eos_id) if eos_id in suffix else None
    specials = {mask_id, eos_id}
    return {
        "first_generated_token_is_eos": bool(suffix and suffix[0] == eos_id),
        "first_eos_position": first_eos,
        "eos_token_share": (
            suffix.count(eos_id) / len(suffix) if suffix else float("nan")
        ),
        "residual_mask_count": suffix.count(mask_id),
        "non_special_token_count": sum(1 for t in suffix if t not in specials),
        "generated_width": len(suffix),
    }


def main() -> None:
    import torch

    from unturtle.eval.content_drift import token_edit_distance

    args = parse_args()
    prompts = frozen_prompts()
    model, tokenizer, mask_id = load(args)
    eos_id = int(tokenizer.eos_token_id)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen_prompts.json").write_text(json.dumps(prompts, indent=2))

    records: list[dict[str, object]] = []
    per_arm_suffixes: dict[str, list[list[int]]] = {}

    for label, cache_path, commit, alg, kwargs in ARMS:
        torch.manual_seed(args.seed)
        suffixes: list[list[int]] = []
        prefix_ok = True
        for item in prompts:
            encoded = tokenizer(item["prompt"], return_tensors="pt")
            input_ids = encoded["input_ids"].to(args.device)
            if mask_id in input_ids[0].tolist():
                raise RuntimeError(
                    f"prompt {item['name']!r} contains the mask sentinel — the "
                    "preflight exists to remove exactly this confound"
                )
            prompt_len = input_ids.shape[1]
            call: dict[str, object] = {
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
            with torch.no_grad():
                generated = model.generate(input_ids, **call)
            row = generated[0].tolist()
            if row[:prompt_len] != input_ids[0].tolist():
                prefix_ok = False
            suffixes.append(row[prompt_len:])
        per_arm_suffixes[label] = suffixes

        diagnostics = [
            suffix_diagnostics(s, mask_id=mask_id, eos_id=eos_id) for s in suffixes
        ]
        reference = ARMS[0][0]
        record: dict[str, object] = {
            "arm": label,
            **baseline_cell_key(cache_path=cache_path, commit=commit),
            "cache_class": cache_path_class(cache_path),
            "alg": alg,
            "kind": "prompted_preflight",
            "run": RUN_LABEL,
            "prompt_prefix_agreement": 1.0 if prefix_ok else 0.0,
            "prompt_count": len(prompts),
            "per_prompt": [
                {"name": p["name"], "task_kind": p["kind"], **d}
                for p, d in zip(prompts, diagnostics, strict=True)
            ],
            "rows_with_immediate_eos": sum(
                1 for d in diagnostics if d["first_generated_token_is_eos"]
            ),
            "mean_non_special_tokens": sum(
                int(d["non_special_token_count"]) for d in diagnostics
            )
            / len(diagnostics),
            "mean_eos_share": sum(float(d["eos_token_share"]) for d in diagnostics)
            / len(diagnostics),
            "total_residual_masks": sum(
                int(d["residual_mask_count"]) for d in diagnostics
            ),
        }
        if label != reference:
            ref = per_arm_suffixes[reference]
            record["token_distance_vs_exact"] = sum(
                token_edit_distance(a, b) for a, b in zip(ref, suffixes, strict=True)
            ) / len(suffixes)
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "per_prompt"}))

    (out / "preflight_cells.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    print(f"wrote {len(records)} records to {out / 'preflight_cells.jsonl'}")


if __name__ == "__main__":
    main()
