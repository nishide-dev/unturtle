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

"""#152 canonical cross-family frontier surface — RED-first.

The decision surface every later #151 method issue must use.  Measurement
only: the surface produces frontier POINTS and typed cells, never a winner.
Frozen protocol facts pinned here:

- protocol version 1: OpenWebText, context length 1024, batch sizes
  (1, 8, 32), Tier-A roles (ar_control / masked_discrete / uniform_state /
  embedding_flow / flow_map);
- GenPPL never travels without its evaluator identity AND unigram entropy
  (GenPPL alone is entropy-sensitive — arXiv:2604.02718);
- executed steps are recorded, never requested steps alone;
- every protocol batch size has a typed cell — OOM/unsupported cells are
  data, not omissions;
- DFM is NOT accepted as the uniform_state Tier-A role (issue #152 rule;
  the lead external anchor is Sumi, scale-confound labeled);
- warmup/compile cost is measured once, outside every timed cell;
- one RNG generator owns the full evaluation cell — never per-batch resets;
- the existing #123/#124 generation-record surface stays byte-identical.
"""

import json
import math

import pytest

from unturtle.eval import generation_record


class TestProtocolFreeze:
    def test_protocol_version_1_facts(self):
        from unturtle.eval.frontier import FRONTIER_PROTOCOL

        assert FRONTIER_PROTOCOL["version"] == 1
        assert FRONTIER_PROTOCOL["dataset"] == "openwebtext"
        assert FRONTIER_PROTOCOL["context_length"] == 1024
        assert FRONTIER_PROTOCOL["batch_sizes"] == (1, 8, 32)
        assert FRONTIER_PROTOCOL["tier_a_roles"] == (
            "ar_control",
            "masked_discrete",
            "uniform_state",
            "embedding_flow",
            "flow_map",
        )

    def test_protocol_is_read_only(self):
        """A frozen protocol that can be mutated in place is not frozen."""
        from unturtle.eval.frontier import FRONTIER_PROTOCOL

        with pytest.raises(TypeError):
            FRONTIER_PROTOCOL["context_length"] = 512


class TestTypedCells:
    def test_ok_cell_carries_the_value(self):
        from unturtle.eval.frontier import cell

        assert cell(12.5) == {"status": "ok", "value": 12.5}

    def test_missing_cells_require_a_reason(self):
        from unturtle.eval.frontier import missing_cell

        oom = missing_cell("oom", "CUDA out of memory at batch 32")
        assert oom["status"] == "oom"
        assert "batch 32" in oom["reason"]
        with pytest.raises(ValueError, match="reason"):
            missing_cell("oom", "")

    def test_missing_cell_status_vocabulary_is_closed(self):
        """A typo'd status would silently create a fourth category that no
        consumer filters on."""
        from unturtle.eval.frontier import missing_cell

        with pytest.raises(ValueError, match="status"):
            missing_cell("out-of-memory", "typo status")


class TestFrontierRecord:
    def _systems(self, **overrides):
        from unturtle.eval.frontier import cell, missing_cell

        systems = {
            "nfe": 32,
            "sequence_length": 1024,
            "solver": "sde",
            "throughput": {
                "batch_1": cell({"wall_seconds": 2.0, "samples_per_second": 0.5}),
                "batch_8": cell({"wall_seconds": 8.0, "samples_per_second": 1.0}),
                "batch_32": missing_cell("oom", "24GiB card"),
            },
            "peak_memory_bytes": 8_000_000_000,
            "warmup_seconds": 1.5,
        }
        systems.update(overrides)
        return systems

    def _quality(self, **overrides):
        quality = {
            "genppl": 24.1,
            "genppl_evaluator": {"model": "gpt2-large", "revision": "abc123"},
            "unigram_entropy": 5.15,
            "mauve": 0.9,
            "sample_count": 512,
            "collapse_flags": [],
        }
        quality.update(overrides)
        return quality

    def _record(self, **overrides):
        from unturtle.eval.frontier import frontier_record

        kwargs = dict(
            family="embedding_flow",
            method="elf",
            checkpoint="elf-b@rev0",
            seed=7,
            quality=self._quality(),
            systems=self._systems(),
            steps_requested=32,
            steps_executed=32,
            decoding={"algorithm": "elf_sde", "steps": 32},
        )
        kwargs.update(overrides)
        return frontier_record(**kwargs)

    def test_record_is_versioned_json_and_composes_the_generation_record(self):
        record = self._record(
            generation=generation_record(
                metrics={"distinct_fraction": 0.9}, seed=7, nfe=32
            )
        )
        payload = json.loads(json.dumps(record))
        assert payload["frontier_schema_version"] == 1
        assert payload["protocol_version"] == 1
        assert payload["family"] == "embedding_flow"
        assert payload["generation"]["schema_version"] == 1  # v1 rides inside, intact
        assert payload["quality"]["genppl"] == 24.1
        assert payload["systems"]["throughput"]["batch_32"]["status"] == "oom"

    def test_genppl_without_evaluator_identity_is_rejected(self):
        with pytest.raises(ValueError, match="evaluator"):
            self._record(quality=self._quality(genppl_evaluator=None))

    def test_genppl_without_entropy_is_rejected(self):
        """GenPPL alone is entropy-sensitive; the frontier point needs both
        coordinates (arXiv:2604.02718)."""
        quality = self._quality()
        del quality["unigram_entropy"]
        with pytest.raises(ValueError, match="entropy"):
            self._record(quality=quality)

    def test_every_protocol_batch_size_needs_a_typed_cell(self):
        systems = self._systems()
        del systems["throughput"]["batch_8"]
        with pytest.raises(ValueError, match="batch_8"):
            self._record(systems=systems)

    def test_requested_steps_without_executed_steps_is_rejected(self):
        with pytest.raises(ValueError, match="executed"):
            self._record(steps_executed=None)

    def test_provider_provenance_flows_through_verbatim(self):
        provider = {
            "distribution": "elf-pack",
            "version": "0.1",
            "entry_point": "elf",
        }
        record = self._record(provider=provider)
        assert record["provider"] == provider
        assert self._record()["provider"] is None  # builtin/direct stays honest

    def test_dfm_cannot_claim_the_uniform_state_tier_a_role(self):
        """Issue rule: DFM is not a substitute for a real non-masked
        discrete reference (the lead external anchor is Sumi)."""
        with pytest.raises(ValueError, match="[Ss]ubstitute|uniform"):
            self._record(family="dfm", tier_a_role="uniform_state")

    def test_a_legitimate_tier_a_role_is_recorded(self):
        record = self._record(tier_a_role="embedding_flow")
        assert record["tier_a_role"] == "embedding_flow"
        with pytest.raises(ValueError, match="role"):
            self._record(tier_a_role="not_a_role")


class TestTierAGaps:
    def test_gaps_report_roles_without_valid_records(self):
        from unturtle.eval.frontier import FRONTIER_PROTOCOL, tier_a_gaps

        record = TestFrontierRecord()._record(tier_a_role="embedding_flow")
        gaps = tier_a_gaps([record])
        assert "embedding_flow" not in gaps
        assert set(gaps) == set(FRONTIER_PROTOCOL["tier_a_roles"]) - {"embedding_flow"}

    def test_no_verdict_readiness_until_all_roles_are_covered(self):
        from unturtle.eval.frontier import tier_a_gaps

        assert tier_a_gaps([]) == (
            "ar_control",
            "masked_discrete",
            "uniform_state",
            "embedding_flow",
            "flow_map",
        )


class TestFrontierEmitters:
    def _records(self):
        maker = TestFrontierRecord()
        return [
            maker._record(),
            maker._record(
                family="masked_discrete",
                method="mdlm",
                checkpoint="mdlm-owt@r1",
                seed=8,
                quality=maker._quality(genppl=30.0, unigram_entropy=5.4, mauve=0.8),
            ),
        ]

    def test_genppl_entropy_points_preserve_both_coordinates(self):
        from unturtle.eval.frontier import genppl_entropy_points

        points = genppl_entropy_points(self._records())
        assert [(p["method"], p["genppl"], p["unigram_entropy"]) for p in points] == [
            ("elf", 24.1, 5.15),
            ("mdlm", 30.0, 5.4),
        ]
        for point in points:
            assert point["family"]
            assert point["checkpoint"]
            assert point["seed"] is not None

    def test_points_are_deterministically_ordered(self):
        from unturtle.eval.frontier import genppl_entropy_points

        records = self._records()
        assert genppl_entropy_points(records) == genppl_entropy_points(
            list(reversed(records))
        )

    def test_no_winner_no_scalar_aggregate(self):
        """Tripwire, not proof: a flat dir() substring scan catches the
        obvious verdict-shaped exports (rank/winner/aggregate/score_*);
        semantic absence of judgment is carried by the emitters' contracts
        and review, not by this scan alone (#159 review)."""
        import unturtle.eval.frontier as frontier

        for banned in ("winner", "rank", "score_frontier", "aggregate"):
            assert not any(banned in name.lower() for name in dir(frontier)), (
                f"frontier module exports a verdict-shaped name matching {banned!r}"
            )

    def test_speed_quality_points_pair_throughput_cells_with_quality(self):
        from unturtle.eval.frontier import speed_quality_points

        points = speed_quality_points(self._records(), quality_key="mauve")
        elf_points = [p for p in points if p["method"] == "elf"]
        assert {p["batch_size"] for p in elf_points} == {1, 8, 32}
        oom = next(p for p in elf_points if p["batch_size"] == 32)
        assert oom["status"] == "oom"  # carried, not dropped
        ok = next(p for p in elf_points if p["batch_size"] == 8)
        assert ok["samples_per_second"] == 1.0
        assert ok["mauve"] == 0.9
        assert ok["nfe"] == 32

    def test_jsonl_round_trip_is_lossless_and_ordered(self, tmp_path):
        from unturtle.eval.frontier import read_jsonl, write_jsonl

        records = self._records()
        path = tmp_path / "frontier.jsonl"
        write_jsonl(records, path)
        assert read_jsonl(path) == records


class TestGenerativePerplexity:
    def test_genppl_matches_the_hand_computation_and_carries_identity(self):
        from unturtle.eval.frontier import generative_perplexity

        def evaluator(text):
            # deterministic fake: nll = len(text) * 0.5, tokens = len(text)
            return len(text) * 0.5, len(text)

        result = generative_perplexity(
            ["ab", "abcd"],
            evaluator=evaluator,
            evaluator_identity={"model": "fake", "revision": "r0"},
        )
        expected = math.exp((1.0 + 2.0) / 6)
        assert result["genppl"] == pytest.approx(expected)
        assert result["token_count"] == 6
        assert result["evaluator"] == {"model": "fake", "revision": "r0"}

    def test_identity_is_mandatory(self):
        from unturtle.eval.frontier import generative_perplexity

        with pytest.raises(ValueError, match="model.*revision|identity"):
            generative_perplexity(
                ["x"], evaluator=lambda t: (1.0, 1), evaluator_identity={}
            )

    def test_text_unigram_entropy_under_a_common_tokenizer(self):
        """Cross-family comparability: entropy over DECODED text under one
        shared tokenization, not native token ids."""
        from unturtle.eval.frontier import text_unigram_entropy

        entropy = text_unigram_entropy(
            ["a b", "a c"], tokenize=lambda text: text.split()
        )
        # pooled tokens: a,b,a,c -> p = (1/2, 1/4, 1/4)
        expected = -(
            0.5 * math.log(0.5) + 0.25 * math.log(0.25) + 0.25 * math.log(0.25)
        )
        assert entropy == pytest.approx(expected)


class TestThroughputCells:
    def test_cells_cover_all_protocol_batches_with_one_generator(self):
        """One RNG generator owns the WHOLE evaluation cell: the same
        generator object must arrive at every batch call, never a fresh
        per-batch reset."""
        import torch

        from unturtle.eval.frontier import measure_throughput_cells

        seen = []

        def run_batch(batch_size, generator):
            seen.append((batch_size, id(generator), generator.get_state().sum()))
            return list(range(batch_size))

        cells = measure_throughput_cells(run_batch, seed=3)
        assert set(cells) == {"batch_1", "batch_8", "batch_32"}
        assert all(cells[key]["status"] == "ok" for key in cells)
        assert len({generator_id for _, generator_id, _ in seen}) == 1
        assert isinstance(seen[0][2], torch.Tensor)

    def test_warmup_runs_once_outside_every_timed_cell(self):
        """Compile/build cost must not leak into any arm: warmup sleeps
        50ms; every per-cell wall time must exclude it."""
        import time

        from unturtle.eval.frontier import measure_throughput_cells

        warmups = []

        def warmup():
            warmups.append(1)
            time.sleep(0.05)

        cells = measure_throughput_cells(
            lambda batch_size, generator: None, seed=0, warmup=warmup
        )
        assert warmups == [1]
        for key, value in cells.items():
            assert value["value"]["wall_seconds"] < 0.04, (key, value)

    def test_oom_and_unsupported_become_typed_cells_not_omissions(self):
        import torch

        from unturtle.eval.frontier import measure_throughput_cells

        calls = []

        def run_batch(batch_size, generator):
            calls.append(batch_size)
            if batch_size == 32:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return None

        cells = measure_throughput_cells(
            run_batch, seed=0, unsupported={8: "no batching in this sampler"}
        )
        assert cells["batch_8"]["status"] == "unsupported"
        assert "sampler" in cells["batch_8"]["reason"]
        assert 8 not in calls  # unsupported is declared, not attempted
        assert cells["batch_32"]["status"] == "oom"
        assert cells["batch_1"]["status"] == "ok"

    def test_unexpected_errors_are_not_swallowed_into_cells(self):
        """Only OOM becomes a typed cell automatically; a real bug must
        raise, not be recorded as data."""
        from unturtle.eval.frontier import measure_throughput_cells

        def run_batch(batch_size, generator):
            raise RuntimeError("shape mismatch — a bug, not a capacity limit")

        with pytest.raises(RuntimeError, match="shape mismatch"):
            measure_throughput_cells(run_batch, seed=0)


class TestReviewPins159:
    """Pins for the #159 review findings, RED-first."""

    def _maker(self):
        return TestFrontierRecord()

    def test_an_all_oom_role_claim_does_not_cover_the_gap(self):
        """Review F1 (HIGH): tier_a_gaps is the sole machine gate on the
        #151 verdict.  A record that merely CLAIMS a role — empty quality,
        every throughput cell OOM — must not count as coverage."""
        from unturtle.eval.frontier import frontier_record, missing_cell, tier_a_gaps

        stub = frontier_record(
            family="uniform",
            method="sumi",
            checkpoint="sumi-7b@r0",
            seed=0,
            tier_a_role="uniform_state",
            quality={},
            systems={
                "throughput": {
                    "batch_1": missing_cell("oom", "7B on 24GiB"),
                    "batch_8": missing_cell("oom", "7B on 24GiB"),
                    "batch_32": missing_cell("oom", "7B on 24GiB"),
                }
            },
        )
        assert "uniform_state" in tier_a_gaps([stub])

    def test_quality_without_any_ok_compute_cell_is_not_coverage(self):
        """The frontier is quality–diversity–COMPUTE: a role with no valid
        compute cell cannot sit on it."""
        from unturtle.eval.frontier import frontier_record, missing_cell, tier_a_gaps

        maker = self._maker()
        no_compute = frontier_record(
            family="uniform",
            method="sumi",
            checkpoint="sumi-7b@r0",
            seed=0,
            tier_a_role="uniform_state",
            quality=maker._quality(),
            systems={
                "throughput": {
                    "batch_1": missing_cell("oom", "x"),
                    "batch_8": missing_cell("oom", "x"),
                    "batch_32": missing_cell("oom", "x"),
                }
            },
        )
        assert "uniform_state" in tier_a_gaps([no_compute])
        # ...and the full record from the shared maker (quality + ok cells)
        # DOES cover its role.
        real = maker._record(tier_a_role="embedding_flow")
        assert "embedding_flow" not in tier_a_gaps([real])

    def test_explicit_undecidable_reason_resolves_a_role(self):
        """The issue's escape hatch: 'valid cells OR an explicit reason the
        protocol is undecidable' — with the reason mandatory and the role
        name checked."""
        from unturtle.eval.frontier import tier_a_gaps

        gaps = tier_a_gaps(
            [], undecidable={"uniform_state": "no runnable open checkpoint"}
        )
        assert "uniform_state" not in gaps
        assert "ar_control" in gaps
        with pytest.raises(ValueError, match="reason"):
            tier_a_gaps([], undecidable={"uniform_state": ""})
        with pytest.raises(ValueError, match="role"):
            tier_a_gaps([], undecidable={"not_a_role": "x"})

    def test_unknown_quality_keys_are_rejected_not_silently_unvalidated(self):
        """Review F2: quality={'gen_ppl': ...} (typo) previously dodged the
        evaluator-identity and entropy-pairing rules entirely.  Quality keys
        are a closed vocabulary; method-local fields belong in extra."""
        maker = self._maker()
        with pytest.raises(ValueError, match="gen_ppl"):
            maker._record(quality={"gen_ppl": 24.1})

    def test_unknown_throughput_keys_are_rejected_not_silently_dropped(self):
        """Review F2: a batch_64 (or batch_08 typo) cell was accepted and
        then silently dropped by the emitters — recorded data must never
        vanish without a sound."""
        from unturtle.eval.frontier import cell

        maker = self._maker()
        systems = maker._systems()
        systems["throughput"]["batch_64"] = cell({"wall_seconds": 1.0})
        with pytest.raises(ValueError, match="batch_64"):
            maker._record(systems=systems)

    def test_ok_throughput_cells_must_carry_a_mapping(self):
        """Review F6: cell(1.5) passed validation and blew up later inside
        the emitter, far from the producer.  Fail at record time instead."""
        from unturtle.eval.frontier import cell

        maker = self._maker()
        systems = maker._systems()
        systems["throughput"]["batch_1"] = cell(1.5)
        with pytest.raises(ValueError, match="mapping"):
            maker._record(systems=systems)

    def test_generative_perplexity_rejects_empty_inputs_actionably(self):
        """Review F5: an empty generation run (producer OOMed, no samples)
        must raise something actionable, not ZeroDivisionError."""
        from unturtle.eval.frontier import generative_perplexity

        identity = {"model": "fake", "revision": "r0"}
        # The two failure modes carry DISTINCT diagnoses (empty corpus vs
        # all-degenerate corpus) — matched exactly, because the zero-token
        # guard also fires on an empty list and a loose match let the
        # empty-list guard's removal survive the battery.
        with pytest.raises(ValueError, match="zero texts"):
            generative_perplexity(
                [], evaluator=lambda t: (1.0, 1), evaluator_identity=identity
            )
        with pytest.raises(ValueError, match="zero tokens"):
            generative_perplexity(
                ["x"], evaluator=lambda t: (0.0, 0), evaluator_identity=identity
            )

    def test_causal_evaluator_core_matches_the_hand_computation(self):
        """Review F4: the NLL core is now injectable and unit-tested — a
        fake two-token model with known logits, checked against the closed
        form (shift alignment included)."""
        import torch

        from unturtle.eval.frontier import causal_evaluator_from

        class FakeTokenizer:
            def __call__(self, text, **kwargs):
                class Enc:
                    input_ids = torch.tensor([[0, 1, 2]])

                assert kwargs.get("truncation") is True  # F4: bounded input
                return Enc()

        class FakeModel:
            def __call__(self, input_ids):
                class Out:
                    # vocab 3; uniform logits => each target NLL = ln(3)
                    logits = torch.zeros(1, 3, 3)

                return Out()

        evaluator = causal_evaluator_from(FakeModel(), FakeTokenizer())
        nll, tokens = evaluator("abc")
        assert tokens == 2  # 3 ids -> 2 shifted targets
        assert nll == pytest.approx(2 * math.log(3))


class TestExistingSurfaceIsUntouched:
    def test_generation_record_schema_v1_unchanged(self):
        record = generation_record(
            metrics={"mauve": 0.5}, seed=7, decoding={"algorithm": "dfm"}, nfe=8
        )
        assert record["schema_version"] == 1
        assert set(record) == {
            "schema_version",
            "seed",
            "decoding",
            "metrics",
            "nfe",
            "latency_seconds",
            "extra",
        }


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
