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

"""#157 step 3 — re-score saved suffixes under the frozen extraction rule.

Generation is NOT repeated: the step-3 run saved every raw suffix, and the
correction was to the scoring definition, not to the samples. Re-scoring reads
those files so the numbers move only because the frozen rule replaced
`str.split()`.

Every cell reports, per the frozen conditions, the primary and secondary
extractions together with the generation-length diagnostics — coupled accuracy
alone must never be read as an arm ranking, because a near-empty output that
matches one position keeps 1/8 while a verbose wrong answer is divided by more
than the target length.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from unturtle.eval.dependency_slice import (
    answer_span,
    dependency_length_diagnostics,
    dependency_tasks,
    score_extraction_pair,
)

FIXTURE_SEED = 0
N_PER_KIND = 8
TASK_LENGTH = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="benchmarks/results/pd_dependency")
    parser.add_argument("--checkpoint", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument(
        "--revision", default="08b83a6feb34df1a6011b80c3c00c7563e963b07"
    )
    return parser.parse_args()


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    results = pathlib.Path(args.results)
    cells = [
        json.loads(line)
        for line in (results / "dependency_cells.jsonl").read_text().splitlines()
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, revision=args.revision, trust_remote_code=True
    )
    eos_id = int(tokenizer.eos_token_id)
    mask_id = 126336
    tasks = dependency_tasks(
        n_per_kind=N_PER_KIND, seed=FIXTURE_SEED, length=TASK_LENGTH
    )

    rescored = []
    for cell in cells:
        path = results / f"suffixes_{cell['arm']}_bs{cell['batch_size']}.json"
        suffixes = json.loads(path.read_text())
        texts = [
            tokenizer.decode(
                answer_span(suffix, eos_id=eos_id), skip_special_tokens=True
            )
            for suffix in suffixes
        ]
        cell["scores_frozen_extraction"] = score_extraction_pair(tasks, texts)
        per_kind = {}
        for kind in sorted({task.kind for task in tasks}):
            keep = [index for index, t in enumerate(tasks) if t.kind == kind]
            per_kind[kind] = {
                **score_extraction_pair(
                    [tasks[i] for i in keep], [texts[i] for i in keep]
                ),
                "length": dependency_length_diagnostics(
                    [suffixes[i] for i in keep], eos_id=eos_id, mask_id=mask_id
                ),
            }
        cell["per_kind_frozen_extraction"] = per_kind
        cell["superseded"] = (
            "the `scores` / `per_kind` fields used str.split() and are "
            "EXTRACTION INVALID; use the *_frozen_extraction fields"
        )
        rescored.append(cell)

    out = results / "dependency_cells_rescored.jsonl"
    out.write_text("".join(json.dumps(cell) + "\n" for cell in rescored))
    print(f"wrote {len(rescored)} rescored cells to {out}")


if __name__ == "__main__":
    main()
