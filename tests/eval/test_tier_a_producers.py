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

"""#165 Tier-A control producers — RED-first.

The producers own the `ar_control`, `masked_discrete` and `uniform_state`
roles that `tier_a_gaps()` currently blocks the #151 verdict on.  Pinned
here (Stage-0 freeze + the issue's mutation targets), all on tiny fakes so
the semantics are testable without downloading a checkpoint:

- the AR control is CACHED and competent — a no-cache config is refused,
  not silently accepted;
- AR NFE == generated tokens (one forward per token), executed not requested;
- DFM can never claim `uniform_state` (protocol v1, re-pinned at the
  producer layer);
- every producer emits all three protocol batch cells, typed;
- one cell-owned generator; no per-batch reset;
- official and canonical evaluator columns stay separate;
- scale/data confounds are recorded, never dropped;
- observable net-revision is measured for iterative samplers rather than
  claimed from theory.
"""

import pytest
import torch

from unturtle.eval.frontier import FRONTIER_PROTOCOL, tier_a_gaps


class FakeCausalLM(torch.nn.Module):
    """Minimal AR stand-in that COUNTS forwards, so NFE claims are testable."""

    def __init__(self, vocab=32, use_cache=True):
        super().__init__()
        self.vocab = vocab
        self.forwards = 0
        self.config = type("C", (), {"use_cache": use_cache, "vocab_size": vocab})()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.head = torch.nn.Linear(8, vocab)

    def forward(self, input_ids, **kwargs):
        self.forwards += 1
        return self.head(self.embed(input_ids))


class TestArProducerSemantics:
    def test_no_cache_configuration_is_refused(self):
        """Mutation target: 'naive/no-cache AR presented as the control'."""
        from unturtle.eval.producers import ar_generation_config

        with pytest.raises(ValueError, match="cache"):
            ar_generation_config(use_cache=False)

    def test_config_records_the_competence_settings(self):
        from unturtle.eval.producers import ar_generation_config

        config = ar_generation_config()
        assert config["use_cache"] is True
        assert config["attn_implementation"] in ("sdpa", "flash_attention_2")
        assert config["do_sample"] is True
        assert config["temperature"] == 1.0
        # No truncation knobs: the diffusion anchors do not use them either.
        assert config["top_k"] is None and config["top_p"] is None

    def test_ar_nfe_equals_generated_tokens_and_is_executed(self):
        """AR is one forward per token; the record must carry the EXECUTED
        count, and it must not be silently inherited from the request."""
        from unturtle.eval.producers import ar_nfe

        assert ar_nfe(generated_tokens=1024) == 1024
        with pytest.raises(ValueError, match="executed"):
            ar_nfe(generated_tokens=None)


class TestRoleGuards:
    def test_dfm_cannot_claim_uniform_state_through_the_producer(self):
        from unturtle.eval.producers import build_control_record

        with pytest.raises(ValueError, match="[Ss]ubstitute|uniform"):
            build_control_record(
                role="uniform_state",
                family="dfm",
                method="dfm",
                checkpoint="x@1",
                seed=42,
                quality=_quality(),
                systems=_systems(),
                confounds=["none"],
                official={"genppl_official": 1.0},
            )

    def test_role_must_be_a_protocol_role(self):
        from unturtle.eval.producers import build_control_record

        with pytest.raises(ValueError, match="role"):
            build_control_record(
                role="not_a_role",
                family="ar",
                method="gpt2",
                checkpoint="x@1",
                seed=42,
                quality=_quality(),
                systems=_systems(),
                confounds=["none"],
                official={},
            )

    def test_confounds_are_mandatory(self):
        """Mutation target: 'scale/training-data confound omitted'."""
        from unturtle.eval.producers import build_control_record

        with pytest.raises(ValueError, match="confound"):
            build_control_record(
                role="ar_control",
                family="ar",
                method="gpt2-medium",
                checkpoint="x@1",
                seed=42,
                quality=_quality(),
                systems=_systems(),
                confounds=[],
                official={},
            )

    def test_official_and_canonical_columns_stay_separate(self):
        from unturtle.eval.producers import build_control_record

        record = build_control_record(
            role="ar_control",
            family="ar",
            method="gpt2-medium",
            checkpoint="openai-community/gpt2-medium@6dcaa7a9",
            seed=42,
            quality=_quality(),
            systems=_systems(),
            confounds=["scale: 355M vs 105M anchors"],
            official={"genppl_official": 21.5, "evaluator": {"model": "gpt2-large"}},
        )
        assert record["quality"]["genppl"] == 24.0  # canonical
        assert record["extra"]["official_column"]["genppl_official"] == 21.5
        assert "genppl_official" not in record["quality"]
        assert record["tier_a_role"] == "ar_control"
        assert record["extra"]["confounds"]

    def test_all_three_batch_cells_are_required(self):
        from unturtle.eval.producers import build_control_record

        systems = _systems()
        del systems["throughput"]["batch_8"]
        with pytest.raises(ValueError, match="batch_8"):
            build_control_record(
                role="ar_control",
                family="ar",
                method="gpt2-medium",
                checkpoint="x@1",
                seed=42,
                quality=_quality(),
                systems=systems,
                confounds=["none"],
                official={},
            )


class TestNetRevision:
    def test_net_revision_is_measured_not_assumed(self):
        """Iterative samplers must report how many committed tokens actually
        CHANGE, not merely that revision is theoretically possible."""
        from unturtle.eval.producers import net_revision_stats

        # A trajectory of committed-token snapshots: token 1 flips once.
        trajectory = [
            torch.tensor([[5, 7, 9]]),
            torch.tensor([[5, 7, 9]]),
            torch.tensor([[5, 3, 9]]),
        ]
        stats = net_revision_stats(trajectory)
        assert stats["revised_positions"] == 1
        assert stats["total_positions"] == 3
        assert stats["revision_fraction"] == pytest.approx(1 / 3)
        assert stats["steps_observed"] == 3

    def test_a_frozen_trajectory_reports_zero_revision(self):
        from unturtle.eval.producers import net_revision_stats

        same = torch.tensor([[1, 2]])
        stats = net_revision_stats([same, same.clone(), same.clone()])
        assert stats["revised_positions"] == 0
        assert stats["revision_fraction"] == 0.0

    def test_a_token_that_returns_to_its_original_value_still_counts(self):
        """Mutation target: comparing each snapshot to the FIRST one instead
        of to its predecessor.  A token that flips away and back has been
        revised twice; a first-vs-last diff would score it as untouched and
        under-report how much work the sampler redid."""
        from unturtle.eval.producers import net_revision_stats

        trajectory = [
            torch.tensor([[5, 7]]),
            torch.tensor([[5, 3]]),
            torch.tensor([[5, 7]]),  # position 1 returns to its first value
        ]
        stats = net_revision_stats(trajectory)
        assert stats["revised_positions"] == 1
        assert stats["revision_fraction"] == pytest.approx(0.5)
        # `revised_positions` alone cannot see the round trip: cumulative
        # "differs from predecessor" and "differs from the first state" are
        # provably the same set of positions.  The redone work shows up only
        # in the event count — two flips here, one for a monotone change.
        assert stats["revision_events"] == 2
        monotone = net_revision_stats(
            [torch.tensor([[5, 7]]), torch.tensor([[5, 3]]), torch.tensor([[5, 3]])]
        )
        assert monotone["revised_positions"] == 1
        assert monotone["revision_events"] == 1

    def test_a_single_snapshot_cannot_claim_revision(self):
        from unturtle.eval.producers import net_revision_stats

        with pytest.raises(ValueError, match="at least two"):
            net_revision_stats([torch.tensor([[1, 2]])])


class TestGeneratorOwnership:
    def test_one_generator_reaches_every_batch(self):
        """Mutation target: 'generator reset per batch'.  The producer helper
        must thread the cell's generator through, not create its own."""
        from unturtle.eval.producers import measure_control_throughput

        seen = []

        def run_batch(batch_size, generator):
            seen.append((batch_size, id(generator)))

        cells = measure_control_throughput(run_batch, seed=42)
        assert set(cells) == {f"batch_{b}" for b in FRONTIER_PROTOCOL["batch_sizes"]}
        assert len({generator_id for _, generator_id in seen}) == 1


class TestCoverageAccounting:
    def test_a_valid_control_record_closes_its_gap(self):
        from unturtle.eval.producers import build_control_record

        record = build_control_record(
            role="ar_control",
            family="ar",
            method="gpt2-medium",
            checkpoint="x@1",
            seed=42,
            quality=_quality(),
            systems=_systems(),
            confounds=["scale"],
            official={},
        )
        assert "ar_control" not in tier_a_gaps([record])

    def test_an_all_oom_control_does_not_close_its_gap(self):
        """Mutation target: 'all-OOM role counted as covered'."""
        from unturtle.eval.frontier import missing_cell
        from unturtle.eval.producers import build_control_record

        systems = _systems()
        systems["throughput"] = {
            f"batch_{b}": missing_cell("oom", "too big")
            for b in FRONTIER_PROTOCOL["batch_sizes"]
        }
        record = build_control_record(
            role="uniform_state",
            family="uniform",
            method="sumi",
            checkpoint="x@1",
            seed=42,
            quality=_quality(),
            systems=systems,
            confounds=["scale: 7B"],
            official={},
        )
        assert "uniform_state" in tier_a_gaps([record])


def _quality():
    return {
        "genppl": 24.0,
        "genppl_evaluator": {"model": "gpt2-large", "revision": "main"},
        "unigram_entropy": 5.2,
        "mauve": 0.9,
        "sample_count": 1000,
        "collapse_flags": [],
    }


def _systems():
    from unturtle.eval.frontier import cell

    return {
        "nfe": 1024,
        "sequence_length": 1024,
        "solver": "ar",
        "throughput": {
            f"batch_{b}": cell({"wall_seconds": 1.0, "samples_per_second": b})
            for b in FRONTIER_PROTOCOL["batch_sizes"]
        },
        "peak_memory_bytes": 1_000_000,
    }


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
