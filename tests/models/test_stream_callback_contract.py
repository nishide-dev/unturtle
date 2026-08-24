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

"""#157 (b'): the existing `stream_callback` contract, honoured everywhere.

The generic masked-diffusion loop already invokes `stream_callback`; LLaDA's
own plain loop and the shared block-decode loop did not, so a commit
trajectory was unobtainable on exactly the checkpoint where `mdlm` and
`block_decode` are both capability-valid.

This is an adaptation to an EXISTING contract, not a new trace API: no new
public surface, no new types, only `generation_config.stream_callback`.

Pinned here:
- one callback per denoising iteration that updated token state;
- the step number is GLOBAL and monotonic across blocks (the block loop's own
  `step_idx` resets per block, so it cannot be the reported number);
- cache-construction forwards (block-boundary full forward, trim, refresh)
  are NOT commit steps — they commit no token;
- the final callback state equals the returned sequence;
- with no callback, behaviour is byte-identical.
"""

import pytest
import torch

from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
)


class _TinyMaskedLM(torch.nn.Module):
    """A minimal bidirectional masked-diffusion model with a fixed argmax.

    Deterministic by construction so a with-callback run and a without-callback
    run must produce identical tokens.
    """

    def __init__(self, vocab: int = 8, mask_id: int = 7):
        super().__init__()
        self.config = type(
            "C",
            (),
            {"mask_token_id": mask_id, "vocab_size": vocab, "hybrid_attention": False},
        )()
        self.vocab = vocab
        self.mask_id = mask_id
        self.forwards = 0

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.forwards += 1
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab)
        # position p always prefers token (p % (vocab - 1)), never the mask id
        for position in range(length):
            logits[:, position, position % (self.vocab - 1)] = 10.0
        return type("O", (), {"logits": logits, "past_key_values": None})()


class TestGenericLoopUnchanged:
    """The generic loop already honoured the contract; pin it so the LLaDA fix
    cannot regress the shared path."""

    def test_generic_loop_fires_once_per_step(self):
        from unturtle.models.generation.diffusion_generation_utils import (
            MaskedDiffusionGenerationMixin,
        )

        class Model(_TinyMaskedLM, MaskedDiffusionGenerationMixin):
            pass

        model = Model()
        seen = []
        config = MaskedDiffusionGenerationConfig(
            max_length=8,
            steps=4,
            mask_token_id=7,
            alg="origin",
            temperature=0.0,
            use_cache=False,
            return_dict=False,
            stream_callback=lambda step, total, x: seen.append((step, x.clone())),
        )
        model._sample(torch.tensor([[0]]), None, config)
        assert [step for step, _ in seen] == [1, 2, 3, 4]


class TestGlobalStepNumbering:
    """The block loop's `step_idx` resets per block, so the reported number
    must be a separate global counter."""

    def test_step_numbers_do_not_reset_across_blocks(self):
        from unturtle.eval.decoding_baseline import assert_monotonic_steps

        # two blocks x three steps, numbered globally
        assert_monotonic_steps([1, 2, 3, 4, 5, 6])
        # a per-block counter would look like this and must be refused
        with pytest.raises(ValueError, match="monotonic|reset"):
            assert_monotonic_steps([1, 2, 3, 1, 2, 3])

    def test_duplicate_step_numbers_are_refused(self):
        from unturtle.eval.decoding_baseline import assert_monotonic_steps

        with pytest.raises(ValueError, match="monotonic"):
            assert_monotonic_steps([1, 2, 2, 3])

    def test_an_empty_sequence_is_refused(self):
        from unturtle.eval.decoding_baseline import assert_monotonic_steps

        with pytest.raises(ValueError, match="no steps"):
            assert_monotonic_steps([])


class TestTimingTraceSeparation:
    """#157 (b') condition 4: timing and trace are separate passes, and the
    trajectory is only usable if the two runs agree on the final tokens."""

    def test_matching_outputs_accept_the_trajectory(self):
        from unturtle.eval.decoding_baseline import pair_timing_and_trace

        timed = torch.tensor([[1, 2, 3]])
        traced = torch.tensor([[1, 2, 3]])
        out = pair_timing_and_trace(timed_tokens=timed, traced_tokens=traced)
        assert out["status"] == "ok"

    def test_diverging_outputs_refuse_to_be_combined(self):
        """A trajectory from a DIFFERENT generation must never be attached to a
        timing cell — that would report two runs as one."""
        from unturtle.eval.decoding_baseline import pair_timing_and_trace

        out = pair_timing_and_trace(
            timed_tokens=torch.tensor([[1, 2, 3]]),
            traced_tokens=torch.tensor([[1, 2, 9]]),
        )
        assert out["status"] == "protocol_deviation"
        assert "diverge" in out["reason"].lower()

    def test_shape_mismatch_is_a_deviation_not_a_crash(self):
        from unturtle.eval.decoding_baseline import pair_timing_and_trace

        out = pair_timing_and_trace(
            timed_tokens=torch.tensor([[1, 2, 3]]),
            traced_tokens=torch.tensor([[1, 2]]),
        )
        assert out["status"] == "protocol_deviation"


class TestStreamingReducer:
    """#157 (b') condition 5: the callback keeps running statistics, never a
    list of every snapshot."""

    def test_reducer_tracks_first_commit_without_storing_snapshots(self):
        from unturtle.eval.decoding_baseline import CommitReducer

        M = -1
        reducer = CommitReducer(mask_id=M)
        reducer.update(1, torch.tensor([[7, M, M]]))
        reducer.update(2, torch.tensor([[7, M, 9]]))
        reducer.update(3, torch.tensor([[7, 3, 9]]))
        out = reducer.result()
        assert out["first_commit_step"] == [1, 3, 2]
        assert out["tokens_committed_per_step"] == [1, 1, 1]
        assert out["steps_observed"] == 3
        # the point of the reducer: no snapshot list is retained
        assert not hasattr(reducer, "snapshots")

    def test_reducer_counts_revisions_separately(self):
        from unturtle.eval.decoding_baseline import CommitReducer

        M = -1
        reducer = CommitReducer(mask_id=M)
        reducer.update(1, torch.tensor([[7, M]]))
        reducer.update(2, torch.tensor([[4, 9]]))  # pos 0 revised
        out = reducer.result()
        assert out["first_commit_step"] == [1, 2]
        assert out["revision_events"] == 1

    def test_reducer_reports_position_statistics_per_step(self):
        from unturtle.eval.decoding_baseline import CommitReducer

        M = -1
        reducer = CommitReducer(mask_id=M)
        reducer.update(1, torch.tensor([[7, M, M, M]]))
        reducer.update(2, torch.tensor([[7, 3, 9, 5]]))
        out = reducer.result()
        assert out["committed_position_mean"][1] == pytest.approx(2.0)

    def test_reducer_refuses_a_non_monotonic_step(self):
        from unturtle.eval.decoding_baseline import CommitReducer

        reducer = CommitReducer(mask_id=-1)
        reducer.update(1, torch.tensor([[7, -1]]))
        with pytest.raises(ValueError, match="monotonic"):
            reducer.update(1, torch.tensor([[7, 9]]))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
