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

import json
import math

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

    def test_ar_batch_forwards_is_the_generated_width(self):
        """Forwards executed is the WIDTH of the generated tensor.

        Content lengths (EOS excluded) are off by one whenever a row stops:
        measured on gpt2 with EOS forced on the first generated token, one
        forward ran while every content length was 0 — NFE 0 against 1
        executed (#167 review 2).  `generated.shape[1]` is the executed
        count directly, with no reconstruction from lengths.
        """
        from unturtle.eval.producers import ar_batch_forwards

        assert ar_batch_forwards(generated_width=32) == 32
        assert ar_batch_forwards(generated_width=1) == 1
        with pytest.raises(ValueError, match="no forwards|width"):
            ar_batch_forwards(generated_width=0)

    def test_ar_nfe_sums_the_per_batch_forward_counts(self):
        """Across batches the cell's NFE per sample is the mean of the
        per-batch forward counts — each sample in a batch paid for every
        forward that batch executed."""
        from unturtle.eval.producers import ar_nfe_from_batches

        # Batch A ran 32 forwards for 4 samples, batch B ran 10 for 2.
        assert ar_nfe_from_batches([(32, 4), (10, 2)]) == pytest.approx(
            (32 * 4 + 10 * 2) / 6
        )
        with pytest.raises(ValueError, match="empty|no batches"):
            ar_nfe_from_batches([])

    def test_ar_nfe_equals_generated_tokens_and_is_executed(self):
        """AR is one forward per token; the record must carry the EXECUTED
        count, and it must not be silently inherited from the request."""
        from unturtle.eval.producers import ar_nfe

        assert ar_nfe(generated_tokens=1024) == 1024
        with pytest.raises(ValueError, match="executed"):
            ar_nfe(generated_tokens=None)


class TestProviderProvenance:
    """#167 review 2, blocker 3: each record must name HOW it ran.

    Without it a reader cannot tell a `transformers.generate` KV-cache path
    from a native loop, or know which library versions produced the number.
    """

    def test_ar_provenance_names_the_execution_path(self):
        from unturtle.eval.producers import control_provider

        prov = control_provider(
            "ar_control",
            details={"attn_implementation": "sdpa", "use_cache": True},
        )
        assert "transformers" in prov["engine"].lower()
        assert prov["details"]["use_cache"] is True
        assert prov["transformers_version"]
        assert prov["unturtle_version"]

    def test_mdlm_provenance_names_the_conversion_and_the_local_step(self):
        from unturtle.eval.producers import control_provider

        prov = control_provider("masked_discrete", details={})
        blob = json.dumps(prov).lower()
        assert "convert_mdlm_owt" in blob or "native conversion" in blob
        assert "noise" in blob  # the producer-local noise-removal step

    def test_sumi_provenance_names_the_remote_code_class(self):
        from unturtle.eval.producers import control_provider

        prov = control_provider("uniform_state", details={})
        blob = json.dumps(prov).lower()
        assert "remote" in blob and "sumi" in blob

    def test_an_unknown_role_has_no_invented_provenance(self):
        from unturtle.eval.producers import control_provider

        with pytest.raises(ValueError, match="provenance|role"):
            control_provider("embedding_flow", details={})

    def test_versions_are_read_not_hardcoded(self):
        """Mutation target: a literal version string.  The record must name
        the libraries that actually ran."""
        import transformers

        from unturtle.eval.producers import control_provider

        prov = control_provider("ar_control", details={})
        assert prov["transformers_version"] == transformers.__version__


class TestGenerationTokenizerPin:
    """#167 review 2, blocker 3: MDLM decodes with the gpt2 tokenizer, and
    decoded text IS the quality surface — so that tokenizer needs a pinned
    revision in the record, not an unpinned `from_pretrained("gpt2")`."""

    def test_the_generation_tokenizer_identity_is_pinned(self):
        from unturtle.eval.producers import generation_tokenizer_identity

        identity = generation_tokenizer_identity(
            name="openai-community/gpt2",
            revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
        )
        assert identity["name"] == "openai-community/gpt2"
        assert identity["revision"] == "607a30d783dfa663caf39e06633721c8d4cfcd7e"

    def test_a_floating_generation_tokenizer_is_refused(self):
        from unturtle.eval.producers import generation_tokenizer_identity

        with pytest.raises(ValueError, match="main|commit|revision"):
            generation_tokenizer_identity(name="openai-community/gpt2", revision="main")


class TestRoleSpecificPreflight:
    """#167 review 2, blocker 1: the preflight must enforce the FROZEN
    conditions per role, not just the shared ones.

    Before this, a decision run could produce a role-bearing record at 512
    tokens, 64 steps, canvas 2048, temperature 0.7, a different checkpoint,
    or a different (still immutable) evaluator SHA.  The downstream review
    script refusing it is too late — the record already exists and already
    closes `tier_a_gaps()`.
    """

    def _ar(self, **over):
        cfg = dict(
            model="openai-community/gpt2-medium",
            revision="6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
            max_new_tokens=1024,
            temperature=1.0,
            use_cache=True,
            top_k=None,
            top_p=None,
        )
        cfg.update(over)
        return cfg

    def _mdlm(self, **over):
        cfg = dict(
            repo="kuleshov-group/mdlm-owt",
            revision="d0958fa851335ece6c15260ce0025f030673c0fb",
            steps=128,
            sequence_length=1024,
            noise_removal=True,
            alg="origin",
        )
        cfg.update(over)
        return cfg

    def _sumi(self, **over):
        cfg = dict(
            repo="tohoku-nlp/sumi-7b",
            revision="0d20f7becf84340b8a8d71a8dda577a502a5c8dd",
            steps=128,
            canvas_length=1024,
            sampler="ancestral",
            schedule="linear",
            temperature=1.0,
            min_log_snr=-9.0,
            max_log_snr=9.0,
        )
        cfg.update(over)
        return cfg

    def test_conforming_configs_pass_for_every_role(self):
        from unturtle.eval.producers import assert_frozen_role_config

        assert_frozen_role_config("ar_control", self._ar())
        assert_frozen_role_config("masked_discrete", self._mdlm())
        assert_frozen_role_config("uniform_state", self._sumi())

    @pytest.mark.parametrize(
        "override",
        [
            {"max_new_tokens": 512},
            {"temperature": 0.7},
            {"use_cache": False},
            {"top_k": 50},
            {"model": "openai-community/gpt2"},
            {"revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"},
        ],
    )
    def test_ar_deviations_are_refused(self, override):
        from unturtle.eval.producers import assert_frozen_role_config

        with pytest.raises(ValueError, match="frozen"):
            assert_frozen_role_config("ar_control", self._ar(**override))

    @pytest.mark.parametrize(
        "override",
        [
            {"steps": 64},
            {"sequence_length": 512},
            {"noise_removal": False},
            {"alg": "maskgit_plus"},
            {"revision": "main"},
        ],
    )
    def test_mdlm_deviations_are_refused(self, override):
        from unturtle.eval.producers import assert_frozen_role_config

        with pytest.raises(ValueError, match="frozen"):
            assert_frozen_role_config("masked_discrete", self._mdlm(**override))

    @pytest.mark.parametrize(
        "override",
        [
            {"canvas_length": 2048},
            {"steps": 64},
            {"temperature": 0.7},
            {"sampler": "adaptive"},
            {"schedule": "cosine"},
            {"min_log_snr": -6.0},
            {"max_log_snr": 6.0},
        ],
    )
    def test_sumi_deviations_are_refused(self, override):
        from unturtle.eval.producers import assert_frozen_role_config

        with pytest.raises(ValueError, match="frozen"):
            assert_frozen_role_config("uniform_state", self._sumi(**override))

    def test_an_unknown_role_config_is_refused(self):
        from unturtle.eval.producers import assert_frozen_role_config

        with pytest.raises(ValueError, match="no frozen config|unknown role"):
            assert_frozen_role_config("embedding_flow", {})

    def test_a_missing_key_is_refused_not_skipped(self):
        """Mutation target: only checking the keys the caller happens to
        pass.  An omitted knob must fail, not pass by absence."""
        from unturtle.eval.producers import assert_frozen_role_config

        cfg = self._mdlm()
        del cfg["noise_removal"]
        with pytest.raises(ValueError, match="missing|frozen"):
            assert_frozen_role_config("masked_discrete", cfg)

    def test_preflight_refuses_a_different_immutable_evaluator_sha(self):
        """A pinned-but-different evaluator commit is still not the frozen
        one: GenPPL is not comparable across evaluator identities."""
        from unturtle.eval.producers import decision_preflight

        with pytest.raises(ValueError, match="frozen|canonical"):
            decision_preflight(
                mode="decision",
                role="ar_control",
                num_samples=1000,
                seed=42,
                mauve_available=True,
                evaluator_revision="0" * 40,
                role_config=self._ar(),
            )

    def test_preflight_threads_the_role_config_through(self):
        from unturtle.eval.producers import decision_preflight

        assert (
            decision_preflight(
                mode="decision",
                role="uniform_state",
                num_samples=1000,
                seed=42,
                mauve_available=True,
                evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
                role_config=self._sumi(),
            )
            == "uniform_state"
        )
        with pytest.raises(ValueError, match="frozen"):
            decision_preflight(
                mode="decision",
                role="uniform_state",
                num_samples=1000,
                seed=42,
                mauve_available=True,
                evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
                role_config=self._sumi(canvas_length=2048),
            )

    def test_smoke_mode_does_not_require_the_frozen_config(self):
        """A smoke run may use any settings — it claims no role."""
        from unturtle.eval.producers import decision_preflight

        assert (
            decision_preflight(
                mode="smoke",
                role="uniform_state",
                num_samples=2,
                seed=1,
                mauve_available=False,
                evaluator_revision="main",
                role_config=self._sumi(canvas_length=256, steps=8),
            )
            is None
        )


class TestDecisionPreflight:
    """#167 review 1: a tiny smoke must not be able to close a Tier-A gap.

    `--num-samples 4` still produced a record carrying `tier_a_role`, and a
    missing MAUVE reference only left a note — so a wiring smoke satisfied
    `tier_a_gaps()` exactly like a decision run.  A record may claim a role
    ONLY in decision mode, which verifies the frozen conditions.
    """

    def test_smoke_mode_produces_no_role_claim(self):
        from unturtle.eval.producers import decision_preflight

        role = decision_preflight(
            mode="smoke",
            role="ar_control",
            num_samples=4,
            seed=42,
            mauve_available=False,
            evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
        )
        assert role is None, "a smoke run must not claim a Tier-A role"

    def test_decision_mode_requires_the_frozen_sample_budget(self):
        from unturtle.eval.producers import decision_preflight

        with pytest.raises(ValueError, match="1000|sample"):
            decision_preflight(
                mode="decision",
                role="ar_control",
                num_samples=4,
                seed=42,
                mauve_available=True,
                evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
            )

    def test_decision_mode_requires_the_frozen_seed(self):
        from unturtle.eval.producers import decision_preflight

        with pytest.raises(ValueError, match="seed"):
            decision_preflight(
                mode="decision",
                role="ar_control",
                num_samples=1000,
                seed=7,
                mauve_available=True,
                evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
            )

    def test_decision_mode_requires_the_mauve_reference(self):
        """A missing reference used to degrade to a note while the role
        claim stood."""
        from unturtle.eval.producers import decision_preflight

        with pytest.raises(ValueError, match="MAUVE|reference"):
            decision_preflight(
                mode="decision",
                role="ar_control",
                num_samples=1000,
                seed=42,
                mauve_available=False,
                evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
            )

    def test_decision_mode_refuses_a_floating_evaluator_revision(self):
        """#167 review 4: `main` is not an identity — it moves."""
        from unturtle.eval.producers import decision_preflight

        with pytest.raises(ValueError, match="main|commit|revision"):
            decision_preflight(
                mode="decision",
                role="ar_control",
                num_samples=1000,
                seed=42,
                mauve_available=True,
                evaluator_revision="main",
            )

    def test_a_conforming_decision_run_returns_the_role(self):
        from unturtle.eval.producers import (
            FROZEN_ROLE_CONFIGS,
            decision_preflight,
        )

        assert (
            decision_preflight(
                mode="decision",
                role="masked_discrete",
                num_samples=1000,
                seed=42,
                mauve_available=True,
                evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
                role_config=dict(FROZEN_ROLE_CONFIGS["masked_discrete"]),
            )
            == "masked_discrete"
        )

    def test_decision_mode_without_a_role_config_is_refused(self):
        """The frozen decoding conditions cannot be checked against a config
        that was never supplied (#167 review 2)."""
        from unturtle.eval.producers import decision_preflight

        with pytest.raises(ValueError, match="config"):
            decision_preflight(
                mode="decision",
                role="masked_discrete",
                num_samples=1000,
                seed=42,
                mauve_available=True,
                evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
            )

    def test_an_unknown_mode_is_refused(self):
        from unturtle.eval.producers import decision_preflight

        with pytest.raises(ValueError, match="mode"):
            decision_preflight(
                mode="whatever",
                role="ar_control",
                num_samples=1000,
                seed=42,
                mauve_available=True,
                evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
            )


class TestSmokeCannotCloseAGap:
    """The end-to-end property #167 review 1 asks for: a smoke-mode record
    must not satisfy `tier_a_gaps()`."""

    def test_a_roleless_record_leaves_the_gap_open(self):
        from unturtle.eval.producers import build_control_record, decision_preflight

        role = decision_preflight(
            mode="smoke",
            role="ar_control",
            num_samples=4,
            seed=42,
            mauve_available=False,
            evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
        )
        record = build_control_record(
            role=role,
            family="ar",
            method="gpt2-medium",
            checkpoint="x@1",
            seed=42,
            quality=_quality(),
            systems=_systems(),
            confounds=["scale"],
            official={},
        )
        assert record["tier_a_role"] is None
        assert "ar_control" in tier_a_gaps([record])

    def test_a_decision_record_closes_it(self):
        from unturtle.eval.producers import (
            FROZEN_ROLE_CONFIGS,
            build_control_record,
            decision_preflight,
        )

        role = decision_preflight(
            mode="decision",
            role="ar_control",
            num_samples=1000,
            seed=42,
            mauve_available=True,
            evaluator_revision="32b71b12589c2f8d625668d2335a01cac3249519",
            role_config=dict(FROZEN_ROLE_CONFIGS["ar_control"]),
        )
        record = build_control_record(
            role=role,
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


class TestEvaluatorIdentity:
    """#167 review 4: the canonical evaluator must be pinned to a commit,
    and the record must carry the provider it was resolved with."""

    def test_a_floating_revision_is_refused(self):
        from unturtle.eval.producers import canonical_evaluator_identity

        with pytest.raises(ValueError, match="main|commit"):
            canonical_evaluator_identity(
                model="gpt2-large", revision="main", tokenizer_revision="main"
            )

    def test_identity_carries_model_tokenizer_and_transformers_version(self):
        from unturtle.eval.producers import canonical_evaluator_identity

        sha = "32b71b12589c2f8d625668d2335a01cac3249519"
        identity = canonical_evaluator_identity(
            model="gpt2-large", revision=sha, tokenizer_revision=sha
        )
        assert identity["model"] == "gpt2-large"
        assert identity["revision"] == sha
        assert identity["tokenizer_revision"] == sha
        assert identity["transformers_version"]

    def test_the_tokenizer_revision_must_also_be_pinned(self):
        """The entropy tokenizer moves independently of the scorer."""
        from unturtle.eval.producers import canonical_evaluator_identity

        sha = "32b71b12589c2f8d625668d2335a01cac3249519"
        with pytest.raises(ValueError, match="tokenizer"):
            canonical_evaluator_identity(
                model="gpt2-large", revision=sha, tokenizer_revision=None
            )


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

    def test_step_fields_reach_the_protocol_validator(self):
        """`frontier_record` rejects a requested step count with no executed
        one.  Stashing the pair in `extra` made that check unreachable, so
        `build_control_record` forwards the protocol fields (#165 review
        F3)."""
        from unturtle.eval.producers import build_control_record

        record = build_control_record(
            role="masked_discrete",
            family="mdlm",
            method="mdlm-owt",
            checkpoint="x@1",
            seed=42,
            steps_requested=128,
            steps_executed=128,
            quality=_quality(),
            systems=_systems(),
            confounds=["scale"],
            official={},
        )
        assert record["steps_requested"] == 128
        assert record["steps_executed"] == 128

        with pytest.raises(ValueError, match="executed"):
            build_control_record(
                role="masked_discrete",
                family="mdlm",
                method="mdlm-owt",
                checkpoint="x@1",
                seed=42,
                steps_requested=128,
                quality=_quality(),
                systems=_systems(),
                confounds=["scale"],
                official={},
            )

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


class TestGlobalRngSeam:
    """`transformers.generate()` has NO `generator=` parameter — passing one
    raises `ValueError: model_kwargs are not used by the model`, and its
    sampling reads the GLOBAL torch RNG.  The protocol still requires one
    cell-owned generator, so the producers derive a seed from the cell's
    generator and pin the global RNG for the duration, restoring whatever
    was there before.  Verified against transformers 5.x, torch 2.10.
    """

    def test_seed_is_derived_from_the_cell_generator(self):
        from unturtle.eval.producers import global_rng_from

        cell = torch.Generator().manual_seed(42)
        first = global_rng_from(cell)
        second = global_rng_from(cell)
        # Consecutive draws from the SAME generator differ: the stream
        # advances, so a per-batch reset cannot mimic it.
        assert first != second
        fresh = torch.Generator().manual_seed(42)
        assert global_rng_from(fresh) == first

    def test_the_surrounding_global_rng_is_restored(self):
        from unturtle.eval.producers import global_rng_from, pinned_global_rng

        torch.manual_seed(1234)
        before = torch.rand(3)
        torch.manual_seed(1234)
        with pinned_global_rng(global_rng_from(torch.Generator().manual_seed(7))):
            torch.rand(5)  # burn the pinned stream
        after = torch.rand(3)
        assert torch.equal(before, after), (
            "generation must not advance the caller's global RNG stream"
        )

    def test_the_pinned_stream_is_reproducible(self):
        from unturtle.eval.producers import pinned_global_rng

        with pinned_global_rng(99):
            a = torch.rand(4)
        with pinned_global_rng(99):
            b = torch.rand(4)
        assert torch.equal(a, b)


class TestDeviceGeneratorDerivation:
    """Some native samplers require a generator on the SAME device as the
    model — Sumi's `_ancestral_step` calls `torch.multinomial(...,
    generator=...)` on CUDA tensors and raises `RuntimeError: Expected a
    'cuda' device type for generator but found 'cpu'`.

    `measure_throughput_cells` (#152) owns one CPU generator by design, so
    the producer DERIVES a device generator from it — the cell generator
    still advances once per batch, exactly as it would if handed over
    directly.  Deriving, not replacing: a fresh per-batch generator would
    break the protocol's single-stream requirement.
    """

    def test_derived_generator_lands_on_the_requested_device(self):
        from unturtle.eval.producers import derive_device_generator

        cell = torch.Generator().manual_seed(42)
        derived = derive_device_generator(cell, device="cpu")
        assert derived.device.type == "cpu"

    def test_the_cell_stream_advances_once_per_derivation(self):
        """Mutation target: seeding the derived generator from a constant.
        Two batches would then share an identical RNG stream and the
        throughput cells would silently measure the same sample twice."""
        from unturtle.eval.producers import derive_device_generator

        cell = torch.Generator().manual_seed(42)
        first = derive_device_generator(cell, device="cpu").initial_seed()
        second = derive_device_generator(cell, device="cpu").initial_seed()
        assert first != second
        # Reproducible from the same cell seed.
        fresh = torch.Generator().manual_seed(42)
        assert derive_device_generator(fresh, device="cpu").initial_seed() == first

    def test_derivation_accepts_a_generator_on_any_device(self):
        """A cell generator may itself live on CUDA (a caller that seeded it
        device-side).  `torch.randint` requires the generator's device to
        match the output tensor's, so the draw must follow the GENERATOR's
        device, not assume CPU."""
        from unturtle.eval.producers import derive_device_generator, global_rng_from

        cpu_cell = torch.Generator().manual_seed(3)
        assert isinstance(global_rng_from(cpu_cell), int)
        assert derive_device_generator(cpu_cell, device="cpu").device.type == "cpu"

        if torch.cuda.is_available():
            cuda_cell = torch.Generator(device="cuda").manual_seed(3)
            # Neither call may raise "Expected a 'cpu' device type for
            # generator but found 'cuda'".
            assert isinstance(global_rng_from(cuda_cell), int)
            derived = derive_device_generator(cuda_cell, device="cuda")
            assert derived.device.type == "cuda"

    def test_a_generator_already_on_the_device_is_still_derived_from(self):
        """The derivation must not short-circuit when devices happen to
        match: skipping it would stop advancing the cell stream."""
        from unturtle.eval.producers import derive_device_generator

        cell = torch.Generator().manual_seed(7)
        before = cell.get_state().clone()
        derive_device_generator(cell, device="cpu")
        assert not torch.equal(cell.get_state(), before)


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


class TestCanonicalQualityColumn:
    """The canonical column helper shared by the AR and MDLM producers.

    #153/#155 each hand-rolled this; a third copy for #165 would be a third
    chance to drift, so the shared version is pinned here.  The evaluator is
    injected, so these tests need no download.
    """

    def test_quality_carries_evaluator_identity_with_the_score(self):
        from unturtle.eval.producers import canonical_quality_column

        quality = canonical_quality_column(
            ["a b a", "b b c"],
            evaluator=_fake_evaluator,
            evaluator_identity={"model": "fake", "revision": "r1"},
            tokenize=lambda text: [ord(c) for c in text if c != " "],
            sample_ids=[[1, 2, 1], [2, 2, 3]],
        )
        assert quality["genppl"] == pytest.approx(3.0)
        assert quality["genppl_evaluator"] == {"model": "fake", "revision": "r1"}
        assert quality["unigram_entropy"] > 0
        assert quality["sample_count"] == 2
        assert quality["collapse_flags"] == []

    def test_ragged_batches_are_padded_before_the_guards(self):
        """Batches can end at different widths (a batch whose rows all stop
        early is narrower), so stacking raw id lists raises a ragged-tensor
        error.  The producer pads to a rectangle with an explicit filler and
        records that it did (#167 review 2)."""
        from unturtle.eval.producers import stack_sample_ids

        stacked, meta = stack_sample_ids([[1, 2, 3], [4, 5]], pad_id=0)
        assert stacked.shape == (2, 3)
        assert stacked.tolist() == [[1, 2, 3], [4, 5, 0]]
        assert meta["padded_rows"] == 1
        assert meta["pad_id"] == 0
        assert meta["width"] == 3

    def test_uniform_width_batches_report_no_padding(self):
        from unturtle.eval.producers import stack_sample_ids

        stacked, meta = stack_sample_ids([[1, 2], [3, 4]], pad_id=0)
        assert stacked.shape == (2, 2)
        assert meta["padded_rows"] == 0

    def test_stacking_nothing_is_refused(self):
        from unturtle.eval.producers import stack_sample_ids

        with pytest.raises(ValueError, match="no rows|empty"):
            stack_sample_ids([], pad_id=0)

    def test_the_column_takes_ragged_rows_and_reports_no_padding(self):
        """#167 review 2: the canonical column must score content rows
        directly, so no padding can reach the guards."""
        from unturtle.eval.producers import canonical_quality_column

        quality = canonical_quality_column(
            ["a b a", "b b"],
            evaluator=_fake_evaluator,
            evaluator_identity={"model": "fake", "revision": "r1"},
            tokenize=lambda text: [ord(c) for c in text if c != " "],
            sample_ids=[[1, 2, 1], [2, 2]],
        )
        # row 0: 2/3 distinct, row 1: 1/2 -> mean 0.5833...
        assert quality["distinct_fraction"] == pytest.approx((2 / 3 + 0.5) / 2)
        assert quality["row_count"] == 2
        assert quality["empty_rows"] == 0

    def test_a_padded_tensor_is_refused(self):
        """Mutation target: accepting a rectangle again.  Padding is what
        made the guards padding-dependent in the first place."""
        from unturtle.eval.producers import canonical_quality_column

        with pytest.raises(TypeError, match="ragged content rows"):
            canonical_quality_column(
                ["a b"],
                evaluator=_fake_evaluator,
                evaluator_identity={"model": "fake", "revision": "r1"},
                tokenize=lambda text: [1, 2],
                sample_ids=torch.tensor([[1, 2]]),
            )

    def test_diversity_guards_are_part_of_the_canonical_column(self):
        """#153/#155 both splat `diversity_guards(...)` into the canonical
        column, and `_KNOWN_QUALITY_KEYS` reserves the three slots.  A
        control record without them is the one arm where a collapsed but
        low-GenPPL sample set would go unflagged (#165 review F2)."""
        from unturtle.eval.producers import canonical_quality_column

        quality = canonical_quality_column(
            ["a b a", "b b c"],
            evaluator=_fake_evaluator,
            evaluator_identity={"model": "fake", "revision": "r1"},
            tokenize=lambda text: [ord(c) for c in text if c != " "],
            sample_ids=[[1, 2, 1], [2, 2, 3]],
        )
        for key in (
            "distinct_fraction",
            "pooled_unigram_entropy",
            "unique_rows_fraction",
        ):
            assert key in quality, f"{key} missing from the canonical column"

    def test_omitting_sample_ids_is_refused_not_silently_skipped(self):
        """Mutation target: making the guards optional.  Silently dropping
        them is exactly the drift the shared helper exists to prevent.

        Omitting the argument entirely is a TypeError (it is a required
        keyword); passing None reaches the explicit refusal below.  Both
        fail loudly, which is the property under test."""
        from unturtle.eval.producers import canonical_quality_column

        with pytest.raises(TypeError, match="sample_ids"):
            canonical_quality_column(
                ["a b"],
                evaluator=_fake_evaluator,
                evaluator_identity={"model": "fake", "revision": "r1"},
                tokenize=lambda text: [1, 2],
            )
        with pytest.raises(ValueError, match="sample_ids|diversity"):
            canonical_quality_column(
                ["a b"],
                evaluator=_fake_evaluator,
                evaluator_identity={"model": "fake", "revision": "r1"},
                tokenize=lambda text: [1, 2],
                sample_ids=None,
            )

    def test_an_unidentified_evaluator_is_refused(self):
        """Protocol v1: GenPPL never travels without evaluator identity."""
        from unturtle.eval.producers import canonical_quality_column

        with pytest.raises(ValueError, match="identity"):
            canonical_quality_column(
                ["a b"],
                evaluator=_fake_evaluator,
                evaluator_identity={},
                tokenize=lambda text: [1, 2],
                sample_ids=[[1, 2]],
            )

    def test_empty_texts_are_refused_rather_than_scored(self):
        """The refusal lives in `generative_perplexity`; pinned here because
        the producers pass through it and must not add a fallback."""
        from unturtle.eval.producers import canonical_quality_column

        with pytest.raises(ValueError, match="zero texts"):
            canonical_quality_column(
                [],
                evaluator=_fake_evaluator,
                evaluator_identity={"model": "fake", "revision": "r1"},
                tokenize=lambda text: [1],
                sample_ids=[[1]],
            )

    def test_genppl_is_corpus_pooled_not_a_per_text_mean(self):
        """Long and short texts must be TOKEN-weighted.  A per-text mean of
        the two perplexities below is 5.5; the corpus value is ~3.06 —
        wildly different, and only the pooled one matches the MDLM
        reference (#152)."""
        import math

        from unturtle.eval.producers import canonical_quality_column

        def evaluator(text):
            # "long" carries 100 tokens at ppl 3; "short" 2 tokens at ppl 8.
            if text == "long":
                return math.log(3.0) * 100, 100
            return math.log(8.0) * 2, 2

        quality = canonical_quality_column(
            ["long", "short"],
            evaluator=evaluator,
            evaluator_identity={"model": "fake", "revision": "r1"},
            tokenize=lambda text: [ord(c) for c in text],
            sample_ids=[[1, 2], [3, 4]],
        )
        pooled = math.exp((math.log(3.0) * 100 + math.log(8.0) * 2) / 102)
        assert quality["genppl"] == pytest.approx(pooled)
        assert quality["genppl"] == pytest.approx(3.06, abs=0.02)
        assert quality["genppl"] != pytest.approx((3.0 + 8.0) / 2)


class TestUniformStateAccounting:
    """Sumi (`uniform_state`) audited verbatim from its native
    `generation_sumi.py` @ 0d20f7becf84 (#165):

    - true uniform state: the canvas starts as `randint(0, vocab_size)`,
      there is no mask token, and the ancestral posterior lives on the
      one-hot simplex — so the role is genuinely filled, not substituted;
    - NFE == num_denoising_steps (one forward per step, no extra tail);
    - it DOES accept `generator=`, unlike `transformers.generate()`;
    - it denoises a full `canvas_length` (default 2048, ceiling 4864) every
      step, while `max_new_tokens` is only a content budget.  The #152
      protocol pins context 1024, so the compute a cell reports must be
      the CANVAS it actually forwarded, not the content budget.
    """

    def test_nfe_is_the_step_count_with_no_hidden_tail(self):
        from unturtle.eval.producers import uniform_state_nfe

        assert uniform_state_nfe(steps_executed=128) == 128
        with pytest.raises(ValueError, match="executed"):
            uniform_state_nfe(steps_executed=None)

    def test_canvas_length_is_recorded_not_the_content_budget(self):
        """Mutation target: recording `max_new_tokens` as the sequence
        length.  Sumi forwards the whole canvas each step, so a cell that
        claims 1024 while the model forwarded 2048 understates its compute
        by 2x and makes the frontier point look cheaper than it is."""
        from unturtle.eval.producers import uniform_state_compute_scope

        scope = uniform_state_compute_scope(
            canvas_length=2048, content_budget=1024, prompt_length=1
        )
        assert scope["forwarded_tokens"] == 2048
        assert scope["content_budget"] == 1024
        assert scope["sequence_length"] == 2048
        assert "canvas" in scope["note"].lower()

    def test_a_content_budget_larger_than_the_canvas_is_refused(self):
        from unturtle.eval.producers import uniform_state_compute_scope

        with pytest.raises(ValueError, match="canvas"):
            uniform_state_compute_scope(
                canvas_length=1024, content_budget=2048, prompt_length=1
            )

    def test_protocol_context_mismatch_is_surfaced(self):
        """#152 pins context 1024.  A 2048 canvas is not that condition, so
        the producer must label the deviation rather than let the record
        read as protocol-conformant."""
        from unturtle.eval.producers import uniform_state_compute_scope

        matched = uniform_state_compute_scope(
            canvas_length=1024, content_budget=1022, prompt_length=1
        )
        assert matched["protocol_context_match"] is True
        deviating = uniform_state_compute_scope(
            canvas_length=2048, content_budget=1024, prompt_length=1
        )
        assert deviating["protocol_context_match"] is False


class TestRaggedGuards:
    """#167 review 2, blocker 2: the guards must see only real content.

    Padding to a rectangle and calling the fixed-length guards leaves the
    values padding-dependent — filler enters the pooled entropy, the
    distinct denominator becomes the widest row, and exact-row identity
    compares padded tuples.  Recording `padded_rows` documented the problem
    without fixing it, and the MDLM EOS incident showed the guard scope is
    load-bearing, not a footnote.
    """

    def test_distinct_is_the_mean_of_per_row_ratios(self):
        from unturtle.eval.producers import ragged_diversity_guards

        # row 0: 3 distinct / 3 tokens = 1.0 ; row 1: 1 distinct / 2 = 0.5
        guards = ragged_diversity_guards([[1, 2, 3], [7, 7]])
        assert guards["distinct_fraction"] == pytest.approx((1.0 + 0.5) / 2)

    def test_padding_cannot_change_the_values(self):
        """The point of the primitive: a row that is shorter must not be
        scored against the widest row's length, and the filler token must
        not enter the pooled distribution."""
        import torch

        from unturtle.eval.generation_metrics import diversity_guards
        from unturtle.eval.producers import ragged_diversity_guards

        rows = [[1, 2, 3, 4], [5, 6]]
        ragged = ragged_diversity_guards(rows)
        padded = diversity_guards(torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]]))
        assert ragged["distinct_fraction"] != pytest.approx(padded["distinct_fraction"])
        assert ragged["pooled_unigram_entropy"] != pytest.approx(
            padded["pooled_unigram_entropy"]
        )
        # Six real tokens, all distinct -> ln(6)
        assert ragged["pooled_unigram_entropy"] == pytest.approx(math.log(6))

    def test_pooled_entropy_uses_only_content_tokens(self):
        from unturtle.eval.producers import ragged_diversity_guards

        # Pooled content = [1,1,2] -> H = -(2/3 ln 2/3 + 1/3 ln 1/3)
        guards = ragged_diversity_guards([[1, 1], [2]])
        expected = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3))
        assert guards["pooled_unigram_entropy"] == pytest.approx(expected)

    def test_unique_rows_compares_unpadded_tuples(self):
        """Mutation target: two rows that differ only in padding are NOT
        distinct samples, and two identical rows of different length are."""
        from unturtle.eval.producers import ragged_diversity_guards

        assert ragged_diversity_guards([[1, 2], [1, 2]])[
            "unique_rows_fraction"
        ] == pytest.approx(0.5)
        assert ragged_diversity_guards([[1, 2], [1, 2, 3]])[
            "unique_rows_fraction"
        ] == pytest.approx(1.0)

    def test_empty_rows_are_counted_and_reported(self):
        """A row with no content contributes nothing to distinct/entropy but
        must not silently vanish from the denominators."""
        from unturtle.eval.producers import ragged_diversity_guards

        guards = ragged_diversity_guards([[1, 2], []])
        assert guards["empty_rows"] == 1
        assert guards["row_count"] == 2
        # The empty row scores 0 distinct-fraction, so the mean halves.
        assert guards["distinct_fraction"] == pytest.approx(0.5)
        # ...and it contributes no tokens to the pooled distribution.
        assert guards["pooled_unigram_entropy"] == pytest.approx(math.log(2))
        # Two rows, one empty: still two distinct rows.
        assert guards["unique_rows_fraction"] == pytest.approx(1.0)

    def test_all_empty_is_refused_rather_than_scored(self):
        from unturtle.eval.producers import ragged_diversity_guards

        with pytest.raises(ValueError, match="no content|empty"):
            ragged_diversity_guards([[], []])

    def test_no_rows_at_all_is_refused(self):
        from unturtle.eval.producers import ragged_diversity_guards

        with pytest.raises(ValueError, match="no rows"):
            ragged_diversity_guards([])

    def test_it_agrees_with_the_fixed_length_guards_on_equal_widths(self):
        """When every row is the same length there is no padding, so the
        ragged primitive must reproduce the frozen fixed-length semantics
        exactly — otherwise it silently redefines the metric."""
        import torch

        from unturtle.eval.generation_metrics import diversity_guards
        from unturtle.eval.producers import ragged_diversity_guards

        rows = [[1, 2, 2, 4], [5, 5, 5, 5], [1, 2, 3, 4]]
        ragged = ragged_diversity_guards(rows)
        fixed = diversity_guards(torch.tensor(rows))
        for key in (
            "distinct_fraction",
            "pooled_unigram_entropy",
            "unique_rows_fraction",
        ):
            assert ragged[key] == pytest.approx(fixed[key]), key


class TestGuardScopePerFamily:
    """The guard input must match what the EVALUATOR scored (#165 run 2).

    The first decision run exposed a real bug: the MDLM producer cut guard
    rows at the first gpt2 EOS, but MDLM was trained on packed OWT where
    EOS is a DOCUMENT DELIMITER, so an early EOS is ordinary content, not a
    stop signal.  Its 1000-sample record came back with
    `distinct_fraction 0.0047` and `pooled_unigram_entropy 0.086` from an
    average of 6.9 tokens per row, while GenPPL/entropy scored the full
    ~1024-token decoded canvas (122.84 / 7.56).  The frozen ELF/FMLM
    precedent passes ALL generated ids to `diversity_guards`.

    So the scope is per-family, not universal:
      * AR: EOS ends generation -> guards see the pre-EOS content;
      * masked/uniform diffusion on a fixed canvas: the canvas IS the
        output (that is what gets decoded) -> guards see the canvas.
    """

    def test_ar_scope_cuts_at_eos(self):
        from unturtle.eval.producers import guard_rows

        rows = guard_rows([[5, 7, 99, 1, 2]], eos_id=99, eos_means="end_of_generation")
        assert rows == [[5, 7]]

    def test_canvas_scope_keeps_the_whole_row(self):
        """Mutation target: applying the AR rule to a canvas family, which
        is exactly the bug the first run hit."""
        from unturtle.eval.producers import guard_rows

        rows = guard_rows([[5, 7, 99, 1, 2]], eos_id=99, eos_means="document_delimiter")
        assert rows == [[5, 7, 99, 1, 2]]

    def test_an_unknown_eos_semantics_is_refused(self):
        from unturtle.eval.producers import guard_rows

        with pytest.raises(ValueError, match="eos_means"):
            guard_rows([[1, 2]], eos_id=99, eos_means="whatever")

    def test_the_guard_scope_is_recorded(self):
        """A record must say which rule it used, so a 0.0047 can be told
        apart from a real collapse."""
        from unturtle.eval.producers import guard_scope_note

        note = guard_scope_note(eos_means="document_delimiter")
        assert "canvas" in note.lower() or "delimiter" in note.lower()
        assert "document_delimiter" in note


class TestContentVsCanvasScope:
    """#167 review 3: canonical guards must see the CONTENT tokens.

    Running them over the full canvas lets the untouched tail's diversity
    hide a collapsed decoded region — the tail is denoised context nobody
    reads.  Full-canvas entropy and revision stay as secondary diagnostics
    under their own names.
    """

    def test_content_rows_stop_at_the_first_eos(self):
        from unturtle.eval.producers import content_rows

        rows = content_rows([[5, 7, 9, 99, 1, 2], [5, 99, 3, 4, 5, 6]], eos_id=99)
        assert rows == [[5, 7, 9], [5]]

    def test_a_row_without_eos_keeps_its_whole_length(self):
        from unturtle.eval.producers import content_rows

        assert content_rows([[1, 2, 3]], eos_id=99) == [[1, 2, 3]]

    def test_an_empty_content_row_is_surfaced_not_silently_dropped(self):
        """A row whose first token is EOS has no content; dropping it would
        shrink the guard denominator without saying so."""
        from unturtle.eval.producers import content_rows

        rows = content_rows([[99, 1, 2], [3, 4]], eos_id=99)
        assert rows == [[], [3, 4]]

    def test_canvas_diagnostics_stay_under_separate_names(self):
        from unturtle.eval.producers import canvas_diagnostics

        canvas = torch.tensor([[1, 2, 3, 3], [4, 4, 4, 4]])
        diag = canvas_diagnostics(canvas, content_widths=[2, 1])
        assert "canvas_pooled_unigram_entropy" in diag
        assert "canvas_distinct_fraction" in diag
        assert diag["canvas_width"] == 4
        assert diag["content_width_mean"] == pytest.approx(1.5)
        # These are NOT the canonical guard names, so they cannot be mistaken
        # for the canonical column's collapse detection.
        assert "pooled_unigram_entropy" not in diag
        assert "distinct_fraction" not in diag

    def test_revision_stats_ride_the_record_when_a_trajectory_exists(self):
        """`net_revision_stats` was tested but never wired: the Sumi
        producer only kept step NUMBERS, so no record carried measured
        revision (#167 review 3)."""
        from unturtle.eval.producers import revision_diagnostics

        trajectory = [
            torch.tensor([[5, 7]]),
            torch.tensor([[5, 3]]),
            torch.tensor([[5, 3]]),
        ]
        diag = revision_diagnostics(trajectory)
        assert diag["revised_positions"] == 1
        assert diag["revision_events"] == 1
        assert diag["steps_observed"] == 3

    def test_revision_diagnostics_says_so_when_nothing_was_captured(self):
        from unturtle.eval.producers import revision_diagnostics

        diag = revision_diagnostics([])
        assert diag["status"] == "not_captured"
        assert "revised_positions" not in diag


class TestThroughputWork:
    """#167 review 5: each throughput cell needs its own executed work.

    Natural EOS makes forward counts differ by batch size, and the
    top-level NFE comes from the quality run's batch — so without per-cell
    work, batch scaling and generation-length differences are inseparable.
    """

    def test_each_cell_carries_its_own_executed_work(self):
        from unturtle.eval.producers import measure_control_throughput

        widths = {1: 1024, 8: 512, 32: 128}

        def run_batch(batch_size, generator):
            return {
                "forwards_executed": widths[batch_size],
                "content_length_mean": widths[batch_size] / 2,
            }

        cells = measure_control_throughput(run_batch, seed=42)
        for batch_size, width in widths.items():
            value = cells[f"batch_{batch_size}"]["value"]
            assert value["forwards_executed"] == width
            assert value["content_length_mean"] == width / 2
            # token-work = forwards x batch, the work the cell actually did
            assert value["token_work"] == width * batch_size

    def test_a_cell_that_reports_no_work_is_still_typed(self):
        """A `run_batch` that returns nothing must not fabricate work."""
        from unturtle.eval.producers import measure_control_throughput

        cells = measure_control_throughput(lambda b, g: None, seed=42)
        for cell_value in cells.values():
            assert cell_value["status"] == "ok"
            assert "forwards_executed" not in cell_value["value"]
            assert cell_value["value"]["samples_per_second"] > 0


class TestMdlmNoiseRemoval:
    """Upstream MDLM's `sampling.noise_removal=True` (config default) runs
    ONE extra forward after the loop and overwrites every position with the
    argmax of the SUBS-parameterized logits.  Unturtle's `alg="origin"`
    loop has no equivalent, so the producer supplies it — verbatim, at the
    producer layer, with zero core edits (#153/#155 precedent).

    Audited against dev/repos/mdlm/diffusion.py:658-696 (`_sample`) and
    :261-277 (`_subs_parameterization`).
    """

    def test_subs_pins_unmasked_positions_to_themselves(self):
        from unturtle.eval.producers import subs_parameterization

        mask_id = 3
        xt = torch.tensor([[1, mask_id, 2]])
        logits = torch.zeros(1, 3, 4)
        logits[0, 0, 2] = 10.0  # would argmax to token 2 at an UNMASKED slot
        out = subs_parameterization(logits, xt, mask_index=mask_id)
        # Position 0 is unmasked and holds token 1: SUBS forces it there.
        assert out[0, 0].argmax().item() == 1
        assert out[0, 0, 1].item() == 0.0
        assert out[0, 0, 2].item() <= -1e6
        # The masked position keeps a real distribution...
        assert out[0, 1, mask_id].item() <= -1e6  # ...minus the mask token
        assert torch.allclose(out[0, 1].exp().sum(), torch.tensor(1.0), atol=1e-5)

    def test_noise_removal_replaces_the_loop_output_by_argmax(self):
        """The extra forward is DETERMINISTIC: no sampling, no temperature."""
        from unturtle.eval.producers import mdlm_noise_removal

        mask_id = 3
        x = torch.tensor([[1, mask_id, 2]])
        calls = []

        def forward(input_ids):
            calls.append(input_ids.clone())
            logits = torch.zeros(1, 3, 4)
            logits[0, 1, 0] = 5.0  # the masked slot should commit token 0
            return logits

        out = mdlm_noise_removal(x, forward=forward, mask_index=mask_id)
        assert len(calls) == 1, "noise removal is exactly one extra forward"
        assert torch.equal(calls[0], x)
        assert out.tolist() == [[1, 0, 2]]

    def test_noise_removal_is_deterministic_not_sampled(self):
        """Mutation target: committing by multinomial draw instead of argmax.

        The masked slot below is a NEAR-TIE (0.5 / 0.3 / 0.2 after SUBS), so
        a sampler would disagree with itself across runs and across RNG
        states, while the upstream argmax always commits the same token.
        """
        from unturtle.eval.producers import mdlm_noise_removal

        mask_id = 3
        x = torch.full((64, 1), mask_id, dtype=torch.long)

        def forward(input_ids):
            import math

            logits = torch.zeros(input_ids.shape[0], 1, 4)
            logits[:, :, 0] = math.log(0.5)
            logits[:, :, 1] = math.log(0.3)
            logits[:, :, 2] = math.log(0.2)
            logits[:, :, mask_id] = math.log(1e-9)
            return logits

        runs = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            runs.append(mdlm_noise_removal(x, forward=forward, mask_index=mask_id))
        # Every position, every seed: the same token.  A multinomial draw
        # over (0.5, 0.3, 0.2) would put ~50% of the 64 rows elsewhere.
        for run in runs:
            assert run.unique().tolist() == [0], "noise removal must argmax, not sample"
        assert torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2])

    def test_noise_removal_never_emits_the_mask_token(self):
        """Mutation target: skipping SUBS inside noise removal.  A raw argmax
        could commit the mask id itself and the sample would carry a literal
        mask token into the evaluator."""
        from unturtle.eval.producers import mdlm_noise_removal

        mask_id = 3
        x = torch.tensor([[mask_id, mask_id]])

        def forward(input_ids):
            logits = torch.zeros(1, 2, 4)
            logits[..., mask_id] = 100.0  # mask is the raw argmax everywhere
            return logits

        out = mdlm_noise_removal(x, forward=forward, mask_index=mask_id)
        assert mask_id not in out.flatten().tolist()

    def test_nfe_accounting_includes_the_extra_forward(self):
        from unturtle.eval.producers import mdlm_nfe

        assert mdlm_nfe(steps_executed=128, noise_removal=True) == 129
        assert mdlm_nfe(steps_executed=128, noise_removal=False) == 128


class TestMdlmRevisionPin:
    """#165 needs the published MDLM checkpoint pinned to
    d0958fa851335ece6c15260ce0025f030673c0fb.  Before this, `load_mdlm_owt`
    had no `revision` parameter at all, so a record naming a revision was
    describing whatever `main` happened to be."""

    def test_the_first_download_is_pinned(self, monkeypatch):
        """Pins only the FIRST download (the fake stops there).  The
        every-download property is covered by the next test, which lets the
        config through — this one would not see a one-sided pin (#165
        review F6)."""
        import huggingface_hub

        from unturtle.models.backbones.mdlm_dit import convert_mdlm_owt

        seen = []

        def fake_download(repo_id, filename, **kwargs):
            seen.append((filename, kwargs.get("revision")))
            raise _StopDownload

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        with pytest.raises(_StopDownload):
            convert_mdlm_owt.load_mdlm_owt(revision="pinned-sha")
        assert seen == [("config.json", "pinned-sha")]

    def test_config_and_weights_cannot_come_from_different_revisions(self, monkeypatch):
        """Mutation target: threading `revision` into only one of the two
        downloads.  A config/weights mismatch would load silently."""
        import huggingface_hub

        from unturtle.models.backbones.mdlm_dit import convert_mdlm_owt

        seen = []
        real = huggingface_hub.hf_hub_download

        def fake_download(repo_id, filename, **kwargs):
            seen.append((filename, kwargs.get("revision")))
            if filename == "model.safetensors":
                raise _StopDownload
            return real(repo_id, filename, **kwargs)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        with pytest.raises(_StopDownload):
            convert_mdlm_owt.load_mdlm_owt(
                revision="d0958fa851335ece6c15260ce0025f030673c0fb"
            )
        assert len(seen) == 2
        assert {revision for _, revision in seen} == {
            "d0958fa851335ece6c15260ce0025f030673c0fb"
        }


class _StopDownload(Exception):
    """Stops the loader once the download arguments have been observed."""


def _fake_evaluator(text):
    """One text -> (total_nll_nats, token_count), the frontier contract.

    Constant per-token NLL of log(3), so corpus GenPPL is exactly 3.0.
    """
    import math

    tokens = len(text.replace(" ", ""))
    return math.log(3.0) * tokens, tokens


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
