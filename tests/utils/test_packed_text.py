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
        doc 1's separator, as in mdlm's global concatenation.  Sized so two
        full rows survive — every asserted token is real (#132 review: the
        earlier `[: len(flat)]` truncation made this partly self-satisfying)."""
        rows = wrap_documents(docs_of(3, 8), BLOCK, bos=BOS, eos=EOS)
        flat = rows[:, 1:-1].flatten().tolist()
        assert flat == [0, 1, 2, EOS, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_empty_and_all_empty_corpora_yield_zero_rows(self):
        assert wrap_documents([], BLOCK, bos=BOS, eos=EOS).shape == (0, BLOCK)
        assert wrap_documents([[], [], []], BLOCK, bos=BOS, eos=EOS).shape == (
            0,
            BLOCK,
        )

    def test_block_size_guard_fires_at_call_time(self):
        """A generator that only raises on first next() attributes the error
        to whoever consumes it (e.g. deep inside write_packed, after the
        output file exists) — the guard must fire eagerly."""
        with pytest.raises(ValueError, match="block_size"):
            iter_wrapped_blocks(iter([[1, 2, 3]]), 2, bos=BOS, eos=EOS)

    def test_out_of_range_frame_ids_raise_too(self):
        """bos/eos are checked, not just content tokens."""
        with pytest.raises(ValueError, match="uint16"):
            wrap_documents([[1] * 12], BLOCK, bos=70000, eos=EOS)
        with pytest.raises(ValueError, match="uint16"):
            wrap_documents([[1] * 12], BLOCK, bos=BOS, eos=-1)

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

    def test_writer_refuses_malformed_rows(self, tmp_path):
        """A caller streaming raw int64 rows (or the wrong width) must be
        stopped at write time — a silently reinterpreted memmap corrupts the
        corpus for every consumer downstream."""
        with pytest.raises(ValueError, match="uint16"):
            write_packed(
                tmp_path / "bad",
                iter([np.zeros(BLOCK, dtype=np.int64)]),
                BLOCK,
                metadata={},
            )
        with pytest.raises(ValueError, match="shape"):
            write_packed(
                tmp_path / "bad2",
                iter([np.zeros(BLOCK + 1, dtype=np.uint16)]),
                BLOCK,
                metadata={},
            )

    def test_reader_memory_maps(self, tmp_path):
        rows = wrap_documents(docs_of(40), BLOCK, bos=BOS, eos=EOS)
        path = tmp_path / "packed"
        write_packed(path, iter(rows), BLOCK, metadata={})
        loaded, _ = read_packed(path)
        assert isinstance(loaded, np.memmap)

    def test_interrupted_write_leaves_no_file_at_the_final_path(self, tmp_path):
        """#132 review Important: a mid-stream failure must not leave a
        plausibly-sized data file with no sidecar at the corpus path — that
        is the exact artifact an operator is tempted to hand-write a sidecar
        for.  The write is atomic: data lands at the final path only on
        success."""

        def rows_then_boom():
            yield np.zeros(BLOCK, dtype=np.uint16)
            raise RuntimeError("stream died")

        path = tmp_path / "packed"
        with pytest.raises(RuntimeError, match="stream died"):
            write_packed(path, rows_then_boom(), BLOCK, metadata={})
        assert not path.exists(), "truncated corpus left at the final path"
        assert not (tmp_path / "packed.json").exists()
        assert not (tmp_path / "packed.tmp").exists(), (
            "interrupted run left its staging file (a dead multi-GB tmp on "
            "real corpora)"
        )

    def test_a_crash_at_any_single_point_never_yields_data_without_sidecar(
        self, tmp_path, monkeypatch
    ):
        """The ordering claim itself: simulate dying immediately after the
        data lands at the final path — the sidecar must already be there,
        because sidecar-less data is the one artifact the atomic design
        exists to rule out (a crash the other side of the boundary leaves a
        sidecar whose data is missing, which read_packed rejects loudly)."""
        import os as os_module

        real_replace = os_module.replace

        def replace_then_die(src, dst):
            real_replace(src, dst)
            raise RuntimeError("crashed right after the data landed")

        monkeypatch.setattr("unturtle.utils.packed_text.os.replace", replace_then_die)
        rows = wrap_documents(docs_of(40), BLOCK, bos=BOS, eos=EOS)
        path = tmp_path / "packed"
        with pytest.raises(RuntimeError, match="crashed right after"):
            write_packed(path, iter(rows), BLOCK, metadata={})
        if path.exists():
            assert (tmp_path / "packed.json").exists(), (
                "data landed at the final path without its sidecar"
            )

    def test_metadata_colliding_with_computed_fields_is_refused(self, tmp_path):
        """#132 review: computed truth wins over a caller's guess — but a
        DIFFERING caller value is a bug upstream and must not be silently
        overwritten."""
        rows = wrap_documents(docs_of(40), BLOCK, bos=BOS, eos=EOS)
        with pytest.raises(ValueError, match="num_rows"):
            write_packed(
                tmp_path / "p",
                iter(rows),
                BLOCK,
                metadata={"num_rows": 999999},
            )
        # An AGREEING value is fine (e.g. block_size passed through).
        written = write_packed(
            tmp_path / "q", iter(rows), BLOCK, metadata={"block_size": BLOCK}
        )
        assert written == rows.shape[0]
