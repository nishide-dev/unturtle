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

"""mdlm-style wrapped packing (#130 data slice).

Contract (mdlm dataloader.py ``_group_texts``): document token streams —
which must already carry their trailing EOS separators, that is the
tokenizer step's job — are concatenated, chunked into ``block_size - 2``
content tokens, and each chunk is wrapped ``[BOS] + chunk + [EOS]``.  The
final partial chunk is dropped.  The streaming packer must be exactly
equivalent to that global concat-then-chunk definition, because the real
corpus never fits in memory at once.
"""

import json

import numpy as np
import pytest

from unturtle.utils.packed_text import (
    iter_wrapped_blocks,
    read_packed,
    wrap_documents,
    write_packed,
)

BOS = 90
EOS = 91
BLOCK = 8  # content per row = 6


def docs_of(*lengths, start=0):
    """Deterministic docs of the given token counts (EOS already appended,
    mirroring the tokenize step's `tokens + [EOS]` contract)."""
    out, v = [], start
    for n in lengths:
        out.append(list(range(v, v + n)) + [EOS])
        v += n
    return out


class TestWrapSemantics:
    def test_rows_are_bos_plus_content_plus_eos(self):
        rows = wrap_documents(docs_of(11), BLOCK, bos=BOS, eos=EOS)
        assert rows.shape == (2, BLOCK)
        for row in rows:
            assert row[0] == BOS and row[-1] == EOS
        # 11 tokens + 1 doc-EOS = 12 content tokens = exactly two rows of 6.
        assert rows[0, 1:-1].tolist() == [0, 1, 2, 3, 4, 5]
        assert rows[1, 1:-1].tolist() == [6, 7, 8, 9, 10, EOS]

    def test_documents_flow_across_row_boundaries(self):
        """No per-document row alignment: doc 2 starts mid-row right after
        doc 1's separator, as in mdlm's global concatenation."""
        rows = wrap_documents(docs_of(3, 4), BLOCK, bos=BOS, eos=EOS)
        flat = rows[:, 1:-1].flatten().tolist()
        assert flat[:9] == [0, 1, 2, EOS, 3, 4, 5, 6, EOS][: len(flat)]

    def test_the_final_partial_chunk_is_dropped(self):
        # 3 + 1 doc-EOS = 4 content tokens < 6: nothing survives.
        assert wrap_documents(docs_of(3), BLOCK, bos=BOS, eos=EOS).shape == (0, BLOCK)
        # 13 + 1 = 14 = 2*6 + 2: the 2-token remainder is dropped.
        rows = wrap_documents(docs_of(13), BLOCK, bos=BOS, eos=EOS)
        assert rows.shape == (2, BLOCK)

    def test_streaming_packer_equals_global_concat_then_chunk(self):
        """Differential: the generator must reproduce the mdlm definition
        computed naively in one shot."""
        docs = docs_of(7, 1, 19, 4, 30)
        naive_stream = [t for d in docs for t in d]
        content = BLOCK - 2
        n_rows = len(naive_stream) // content
        expected = [
            [BOS] + naive_stream[i * content : (i + 1) * content] + [EOS]
            for i in range(n_rows)
        ]
        streamed = list(iter_wrapped_blocks(iter(docs), BLOCK, bos=BOS, eos=EOS))
        assert [r.tolist() for r in streamed] == expected

    def test_token_ids_beyond_uint16_raise(self):
        with pytest.raises(ValueError, match="uint16"):
            wrap_documents([[70000] * 12], BLOCK, bos=BOS, eos=EOS)

    def test_deterministic(self):
        docs = docs_of(25, 9)
        a = wrap_documents(docs, BLOCK, bos=BOS, eos=EOS)
        b = wrap_documents(docs, BLOCK, bos=BOS, eos=EOS)
        assert np.array_equal(a, b)


class TestPackedIO:
    def test_roundtrip_is_bitwise_with_metadata(self, tmp_path):
        rows = wrap_documents(docs_of(40), BLOCK, bos=BOS, eos=EOS)
        meta = {"tokenizer": "gpt2", "split": "train[:-100000]", "block_size": BLOCK}
        path = tmp_path / "packed"
        written = write_packed(path, iter(rows), BLOCK, metadata=meta)
        assert written == rows.shape[0]

        loaded, loaded_meta = read_packed(path)
        assert loaded.dtype == np.uint16
        assert np.array_equal(np.asarray(loaded), rows)
        for key, value in meta.items():
            assert loaded_meta[key] == value
        assert loaded_meta["num_rows"] == rows.shape[0]

    def test_sidecar_is_json_on_disk(self, tmp_path):
        rows = wrap_documents(docs_of(20), BLOCK, bos=BOS, eos=EOS)
        path = tmp_path / "packed"
        write_packed(path, iter(rows), BLOCK, metadata={"split": "x"})
        sidecar = json.loads((tmp_path / "packed.json").read_text())
        assert sidecar["split"] == "x"
        assert sidecar["block_size"] == BLOCK

    def test_reader_memory_maps(self, tmp_path):
        rows = wrap_documents(docs_of(40), BLOCK, bos=BOS, eos=EOS)
        path = tmp_path / "packed"
        write_packed(path, iter(rows), BLOCK, metadata={})
        loaded, _ = read_packed(path)
        assert isinstance(loaded, np.memmap)
