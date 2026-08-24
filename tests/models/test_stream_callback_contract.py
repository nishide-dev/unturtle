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


class TestLladaHonoursTheContract:
    """The two loops that ignored `stream_callback`: LLaDA's own plain
    `_sample` and the shared `_block_decode_loop`.  Measured before the fix:
    zero invocations on either, on a checkpoint where both `mdlm` and
    `block_decode` are capability-valid — so no commit trajectory was
    obtainable exactly where the paired comparison lives.
    """

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_llada_plain_loop_fires_per_step(self):
        model, mask_id = _load_llada()
        seen = []
        _generate(
            model,
            mask_id,
            algorithm="mdlm",
            steps=4,
            gen_length=8,
            stream_callback=lambda step, total, x: seen.append(step),
        )
        assert seen == [1, 2, 3, 4]

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_block_decode_loop_fires_with_global_steps(self):
        """Two blocks must produce globally increasing numbers, not two
        restarts of a per-block counter."""
        from unturtle.eval.decoding_baseline import assert_monotonic_steps

        model, mask_id = _load_llada()
        seen = []
        _generate(
            model,
            mask_id,
            algorithm="block_decode",
            steps=4,
            gen_length=16,
            block_length=8,
            stream_callback=lambda step, total, x: seen.append(step),
        )
        assert len(seen) >= 2
        assert_monotonic_steps(seen)  # would raise on [1,2,1,2]

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_callback_does_not_change_the_output(self):
        """Condition: with no callback, behaviour is unchanged."""
        model, mask_id = _load_llada()
        without = _generate(
            model,
            mask_id,
            algorithm="block_decode",
            steps=4,
            gen_length=16,
            block_length=8,
        )
        seen = []
        with_cb = _generate(
            model,
            mask_id,
            algorithm="block_decode",
            steps=4,
            gen_length=16,
            block_length=8,
            stream_callback=lambda step, total, x: seen.append(x.clone()),
        )
        assert torch.equal(without, with_cb)
        # and the final callback state IS the returned sequence
        assert torch.equal(seen[-1], with_cb)

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_cache_construction_forwards_are_not_commit_steps(self):
        """A block-boundary full forward, a trim and a cache refresh commit no
        token, so they must not appear as commit steps.  With `steps=4` over
        two blocks the loop runs at most 8 commit iterations — a count that
        also included cache forwards would exceed that."""
        model, mask_id = _load_llada()
        seen = []
        _generate(
            model,
            mask_id,
            algorithm="block_decode",
            steps=4,
            gen_length=16,
            block_length=8,
            stream_callback=lambda step, total, x: seen.append(step),
        )
        # Lower bound too: this assertion passed while the callback fired ZERO
        # times, which is exactly the vacuous-test failure mode. The count must
        # be positive AND bounded, and it must not include the two
        # block-boundary cache forwards.
        assert len(seen) >= 2, "callback never fired — the contract is not honoured"
        assert len(seen) <= 8, f"{len(seen)} callbacks for at most 8 commit steps"
        # exactly one callback per committing iteration, none for cache work
        assert len(seen) == len(set(seen)), "duplicate step numbers"


def _load_llada():
    import torch

    from unturtle import FastDiffusionModel

    model, _ = FastDiffusionModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        revision="08b83a6feb34df1a6011b80c3c00c7563e963b07",
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map=None,
    )
    model.to("cuda:0").eval()
    return model, int(model.config.mask_token_id)


def _generate(model, mask_id, *, algorithm, steps, gen_length, block_length=None, **kw):
    import torch

    prompt = torch.full((1, 1), mask_id, dtype=torch.long, device="cuda:0")
    call = dict(
        algorithm=algorithm,
        steps=steps,
        max_length=gen_length + 1,
        mask_token_id=mask_id,
        alg="origin",
        temperature=0.0,
        return_dict=False,
        **kw,
    )
    if block_length is not None:
        call["block_length"] = block_length
    torch.manual_seed(42)
    with torch.no_grad():
        return model.generate(prompt, **call)


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
