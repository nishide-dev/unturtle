"""
Canonical generation evaluation surface (#123).

Promotes the free-generation measurement layer #121/#122 built
benchmark-locally into `unturtle.eval`: the diversity guard trio, MAUVE
(lazy), latency, and a JSON record schema composing with `DecodingConfig`.

The guard trio's tests pin the DISCRIMINATING cases the #121/#122 reviews
established — each guard is blind alone:

- a batch of IDENTICAL diverse-looking rows: per-row distinct and pooled
  entropy both read healthy; only `unique_rows_fraction` sees the collapse;
- per-row degenerate but different-per-row: unique rows reads healthy; only
  `distinct_fraction` sees it.

Decision rules (margins, cutoffs, gate definitions) are deliberately NOT
here — those are each experiment's pre-registered hypotheses (#123 scope).
"""

import json
import math
import subprocess
import sys

import pytest
import torch


class TestTheGuardTrio:
    def test_distinct_fraction_is_the_mean_per_row_unique_share(self):
        from unturtle.eval import distinct_fraction

        samples = torch.tensor([[1, 1, 1, 1], [1, 2, 3, 4]])

        assert distinct_fraction(samples) == pytest.approx((1 / 4 + 4 / 4) / 2)

    def test_pooled_unigram_entropy_matches_the_hand_computation(self):
        """Corpus-POOLED over the batch's tokens (the #121 wording lesson) —
        pinned against an independent closed form, not the implementation."""
        from unturtle.eval import pooled_unigram_entropy

        samples = torch.tensor([[1, 1, 1], [1, 1, 2]])
        # pooled counts: {1: 5, 2: 1}; p = (5/6, 1/6)
        expected = -(5 / 6 * math.log(5 / 6) + 1 / 6 * math.log(1 / 6))

        assert pooled_unigram_entropy(samples) == pytest.approx(expected)

    def test_unique_rows_fraction_counts_exact_row_duplicates(self):
        from unturtle.eval import unique_rows_fraction

        samples = torch.tensor([[1, 2], [1, 2], [3, 4]])

        assert unique_rows_fraction(samples) == pytest.approx(2 / 3)

    def test_identical_diverse_rows_fool_everything_but_unique_rows(self):
        """The #121 review's dangerous case: 64 copies of one diverse row.
        Per-row distinct = 1.0 and pooled entropy is high — both 'healthy' —
        and only the unique-rows guard reports the collapse."""
        from unturtle.eval import (
            distinct_fraction,
            pooled_unigram_entropy,
            unique_rows_fraction,
        )

        row = torch.arange(16)
        samples = row.unsqueeze(0).expand(64, -1)

        assert distinct_fraction(samples) == pytest.approx(1.0)
        assert pooled_unigram_entropy(samples) == pytest.approx(math.log(16))
        assert unique_rows_fraction(samples) == pytest.approx(1 / 64)

    def test_per_row_degenerate_rows_fool_everything_but_distinct(self):
        """The dual case: every row is one repeated token, all different —
        unique rows reads 1.0; distinct is what catches it."""
        from unturtle.eval import distinct_fraction, unique_rows_fraction

        samples = torch.arange(8).unsqueeze(1).expand(-1, 16)

        assert unique_rows_fraction(samples) == pytest.approx(1.0)
        assert distinct_fraction(samples) == pytest.approx(1 / 16)

    def test_diversity_guards_bundles_the_trio_under_the_canonical_names(self):
        """The names are the schema consumers key on (#122's JSONs use
        them); the bundle must agree with the individual functions."""
        from unturtle.eval import (
            distinct_fraction,
            diversity_guards,
            pooled_unigram_entropy,
            unique_rows_fraction,
        )

        samples = torch.randint(
            0, 32, (8, 16), generator=torch.Generator().manual_seed(0)
        )

        guards = diversity_guards(samples)

        assert guards == {
            "distinct_fraction": distinct_fraction(samples),
            "pooled_unigram_entropy": pooled_unigram_entropy(samples),
            "unique_rows_fraction": unique_rows_fraction(samples),
        }


class TestLatency:
    def test_measure_generation_returns_the_result_and_positive_seconds(self):
        from unturtle.eval import measure_generation

        sentinel = torch.ones(2, 3)

        result, seconds = measure_generation(lambda: sentinel)

        assert result is sentinel
        assert seconds > 0


class TestTheRecordSchema:
    def test_the_record_is_json_serializable_and_carries_the_contract(self):
        """Seed and decoding config ride with every result — the harness'
        recording convention, extended to free generation."""
        from unturtle.eval import generation_record

        record = generation_record(
            metrics={"mauve": 0.5, "distinct_fraction": 0.9},
            seed=7,
            decoding={"algorithm": "dfm", "steps": 8},
            nfe=8,
            latency_seconds=1.25,
        )

        payload = json.loads(json.dumps(record))
        assert payload["schema_version"] == 1
        assert payload["seed"] == 7
        assert payload["decoding"] == {"algorithm": "dfm", "steps": 8}
        assert payload["metrics"]["mauve"] == 0.5
        assert payload["nfe"] == 8
        assert payload["latency_seconds"] == 1.25

    def test_a_harness_decoding_config_composes_directly(self):
        """`DecodingConfig` is the harness' recording unit; the record must
        accept it without the caller hand-serializing."""
        from unturtle.eval import generation_record
        from unturtle.eval.harness.configs import DecodingConfig

        config = DecodingConfig(
            model_family="tiny-a2d",
            task="free-generation",
            max_new_tokens=64,
            num_steps=8,
            temperature=1.0,
            use_chat_template=False,
            fewshot=0,
            algorithm="dfm",
        )

        record = generation_record(metrics={}, seed=0, decoding=config)

        assert record["decoding"]["model_family"] == "tiny-a2d"
        assert record["decoding"]["algorithm"] == "dfm"
        json.dumps(record)

    def test_seed_is_mandatory_context_not_an_afterthought(self):
        from unturtle.eval import generation_record

        with pytest.raises(TypeError):
            generation_record(metrics={})  # no seed


class TestMauveIsLazy:
    def test_importing_unturtle_eval_does_not_import_mauve(self):
        """The lm_eval rule extended: the eval package must import without
        the optional dependency ever loading."""
        code = (
            "import sys; import unturtle.eval; "
            "assert 'mauve' not in sys.modules, 'mauve imported eagerly'; "
            "print('lazy-ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert "lazy-ok" in result.stdout, result.stderr[-500:]

    def test_a_missing_mauve_fails_with_an_actionable_message(self, monkeypatch):
        from unturtle.eval import mauve_score

        monkeypatch.setitem(sys.modules, "mauve", None)

        with pytest.raises(ImportError, match="mauve-text"):
            mauve_score(["reference"], ["generated"])


@pytest.mark.slow
def test_mauve_score_discriminates_on_real_features():
    """The #122 sanity, as a regression: same-distribution text scores far
    above token noise.  Tiny sizes keep this CPU-feasible; the absolute
    values are not pinned (they depend on the feature model), only the
    separation that makes MAUVE usable as a primary metric."""
    from unturtle.eval import mauve_score

    # Same-distribution halves vs token noise.  Measured while writing this
    # test: two disjoint TEMPLATE families also score ~0 (gpt2 features
    # cluster them apart), and below ~500 pooled points the k-means
    # quantization crushes everything toward 0 — so the regression uses a
    # half-split of one pool at 256+256, where the measured separation is
    # 0.816 vs 0.004.
    pool = [
        f"The number {i} plus {j} equals {i + j}, which is easy to check."
        for i in range(32)
        for j in range(16)
    ]
    reference = pool[0::2][:256]
    same_distribution = pool[1::2][:256]
    noise = [f"zx qv jkl pfff woq {i} {j} " * 4 for i in range(16) for j in range(16)]

    good = mauve_score(reference, same_distribution, device_id=-1)
    bad = mauve_score(reference, noise, device_id=-1)

    assert good > bad + 0.3, f"MAUVE failed to separate: {good:.3f} vs {bad:.3f}"
