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

"""mdlm-style wrapped text packing (#130 data slice).

Reproduces kuleshov-group/mdlm ``dataloader._group_texts`` exactly, but as a
stream: document token streams (each already carrying its trailing EOS
separator — that is the tokenize step's contract, mdlm's ``tokens + [EOS]``)
are concatenated, chunked into ``block_size - 2`` content tokens, and each
chunk is wrapped ``[BOS] + chunk + [EOS]``.  The final partial chunk is
dropped, matching mdlm's single end-of-corpus truncation.

Rows are uint16 (gpt2 + mask = 50258 ids fits); a corpus that does not is
refused rather than silently wrapped.

Storage is a flat uint16 memmap plus a JSON sidecar (``<path>.json``)
recording at least ``block_size`` and ``num_rows`` — experiment scripts add
split spec / tokenizer / counts so packed corpora stay auditable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

UINT16_MAX = np.iinfo(np.uint16).max


def iter_wrapped_blocks(
    token_docs: Iterable[list[int]], block_size: int, *, bos: int, eos: int
) -> Iterator[np.ndarray]:
    """Yield ``[BOS] + chunk + [EOS]`` rows (uint16, len ``block_size``) from a
    stream of document token lists, equivalently to concatenating the whole
    corpus and chunking once.  The trailing partial chunk is dropped."""
    content = block_size - 2
    if content <= 0:
        raise ValueError(f"block_size must exceed 2, got {block_size}")
    carry: list[int] = []
    for doc in token_docs:
        carry.extend(doc)
        while len(carry) >= content:
            chunk, carry = carry[:content], carry[content:]
            ids = [bos] + chunk + [eos]
            hi, lo = max(ids), min(ids)
            if hi > UINT16_MAX or lo < 0:
                raise ValueError(
                    f"token id outside uint16 range in packed row "
                    f"(max {hi}, min {lo})"
                )
            yield np.array(ids, dtype=np.uint16)


def wrap_documents(
    token_docs: Iterable[list[int]], block_size: int, *, bos: int, eos: int
) -> np.ndarray:
    """Materialized form of :func:`iter_wrapped_blocks` (tests / small sets)."""
    rows = list(iter_wrapped_blocks(token_docs, block_size, bos=bos, eos=eos))
    if not rows:
        return np.empty((0, block_size), dtype=np.uint16)
    return np.stack(rows)


def write_packed(
    path: str | Path,
    rows: Iterable[np.ndarray],
    block_size: int,
    *,
    metadata: dict,
) -> int:
    """Stream rows into ``<path>`` (flat uint16) and write the ``<path>.json``
    sidecar.  Returns the number of rows written."""
    path = Path(path)
    num_rows = 0
    with open(path, "wb") as f:
        for row in rows:
            row = np.asarray(row)
            if row.dtype != np.uint16 or row.shape != (block_size,):
                raise ValueError(
                    f"expected uint16 rows of shape ({block_size},), got "
                    f"{row.dtype} {row.shape}"
                )
            f.write(row.tobytes())
            num_rows += 1
    sidecar = {**metadata, "block_size": block_size, "num_rows": num_rows}
    Path(f"{path}.json").write_text(json.dumps(sidecar, indent=2))
    return num_rows


def read_packed(path: str | Path) -> tuple[np.memmap, dict]:
    """Memory-map a packed corpus written by :func:`write_packed`."""
    path = Path(path)
    metadata = json.loads(Path(f"{path}.json").read_text())
    rows = np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
        shape=(metadata["num_rows"], metadata["block_size"]),
    )
    return rows, metadata
