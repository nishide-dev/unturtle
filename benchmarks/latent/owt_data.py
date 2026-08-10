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

"""OpenWebText packing driver for the #130 real-text LaDiff gate.

Reproduces the mdlm data conventions exactly (dev/repos/mdlm/dataloader.py):

- splits: train = ``train[:-100000]``, held-out = ``train[-100000:]`` — the
  held-out documents are the ones the published mdlm-owt checkpoint never
  trained on (its own validation split);
- tokenization: gpt2, ``add_special_tokens=False``, one EOS appended per
  document, no detokenizer for OWT;
- packing: ``[BOS] + 1022 content tokens + [EOS]`` per row via
  ``unturtle.utils.packed_text`` (gpt2: BOS == EOS == 50256), final partial
  chunk dropped once at end of corpus — a deliberate, documented divergence
  from mdlm's realized rows, which drop a remainder per 1000-document
  ``datasets.map`` batch (see unturtle/utils/packed_text.py).

Provenance note: mdlm loaded the script-based ``openwebtext`` dataset;
``datasets>=3`` resolves ``Skylion007/openwebtext`` to the hub's parquet
conversion, which preserves example order.  Document-level identity of the
two split views is therefore expected but not re-verifiable against the
retired script loader; what the #130 gate requires is determinism plus a
checkpoint-unseen held-out, and both hold either way.

Usage:
    .venv/bin/python benchmarks/latent/owt_data.py \
        --split heldout --out dev/local/owt/heldout_1024
    .venv/bin/python benchmarks/latent/owt_data.py \
        --split train --max-rows 2929688 --out dev/local/owt/train_3b_1024
        # 2,929,688 rows x 1024 tokens = 3.0B tokens

The token budget cap (--max-rows) takes a deterministic PREFIX of the split.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

BLOCK_SIZE = 1024
HELD_OUT_DOCS = 100_000
SPLITS = {
    "train": f"train[:-{HELD_OUT_DOCS}]",
    "heldout": f"train[-{HELD_OUT_DOCS}:]",
}
DATASET = "Skylion007/openwebtext"
TOKENIZER = "gpt2"
TOKENIZE_BATCH = 1_000


def iter_tokenized_docs(dataset, tokenizer, eos_id: int):
    """gpt2-tokenize documents batch-wise, appending the per-document EOS
    separator (mdlm: ``tokens + [EOS]``)."""
    for start in range(0, len(dataset), TOKENIZE_BATCH):
        batch = dataset[start : start + TOKENIZE_BATCH]["text"]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        for doc in encoded:
            yield doc + [eos_id]


def build_split(split: str, out: Path, max_rows: int | None) -> dict:
    import glob

    import datasets
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from unturtle.utils.packed_text import iter_wrapped_blocks, write_packed

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    eos_id = tokenizer.eos_token_id  # gpt2: BOS == EOS == 50256
    # Read the locally cached parquet shards directly: the hub loader's
    # per-file lock/verify pass hangs on stale NFS SoftFileLocks (observed
    # twice on this box), and sorted shard filenames preserve the split's
    # canonical row order exactly as the hub loader would.
    snapshot = snapshot_download(DATASET, repo_type="dataset", local_files_only=True)
    shards = sorted(glob.glob(f"{snapshot}/plain_text/train-*.parquet"))
    if len(shards) != 80:
        raise RuntimeError(f"expected 80 OWT parquet shards, found {len(shards)}")
    dataset = datasets.load_dataset("parquet", data_files=shards, split=SPLITS[split])

    rows = iter_wrapped_blocks(
        iter_tokenized_docs(dataset, tokenizer, eos_id),
        BLOCK_SIZE,
        bos=eos_id,
        eos=eos_id,
    )
    if max_rows is not None:
        rows = itertools.islice(rows, max_rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": DATASET,
        "snapshot_revision": Path(snapshot).name,
        "split": SPLITS[split],
        "tokenizer": TOKENIZER,
        "bos_id": eos_id,
        "eos_id": eos_id,
        "num_documents_in_split": len(dataset),
        "max_rows": max_rows,
        "packing": "mdlm _group_texts: [BOS] + (block-2) + [EOS], remainder dropped",
    }
    num_rows = write_packed(out, rows, BLOCK_SIZE, metadata=metadata)
    return {**metadata, "num_rows": num_rows, "out": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="deterministic prefix cap in packed rows (tokens = rows * 1024)",
    )
    args = parser.parse_args()
    result = build_split(args.split, args.out, args.max_rows)
    print(result)


if __name__ == "__main__":
    main()
