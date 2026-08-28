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

"""#166 Stage-2 selection evidence: the agreement harness's own discipline.

The harness needs a GPU and the real checkpoint; these tests pin the properties
that decide whether its evidence means anything — that it discriminates a wrong
candidate, that it never claims an outer-wall result, and that its batch-32
digest mode reports what it actually measured.
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import pathlib
import tokenize

import pytest

pytest.importorskip("unturtle_flm", reason="FLM pack not installed")

import torch  # noqa: E402


def _code_only(source: str) -> str:
    """Source with comments and string literals removed: asserting a forbidden
    construct is absent otherwise fails on the docstring that forbids it."""
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def _producer():
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "flm"
        / "state_update_agreement.py"
    )
    spec = importlib.util.spec_from_file_location("_agreement", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(batch=2, length=8, vocab=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(batch, length, vocab, generator=g)
    d = torch.randn(batch, length, vocab, generator=g).abs()
    eps = torch.randn(batch, length, vocab, generator=g)
    wz = torch.zeros(batch, 1, 1)
    wd = torch.ones(batch, 1, 1)
    madj = torch.full((batch, 1, 1), -0.37)
    noise_std = torch.full((batch, 1, 1), 0.37)
    return z, d, eps, wz, wd, madj, noise_std


class TestCandidateSemantics:
    def test_the_selected_candidate_is_NOT_bit_identical_on_cpu(self):
        """MEASURED, and the reason the scope is not boilerplate.

        On CPU `addcmul` contracts to an FMA — the product is never rounded to
        fp32 — so it does NOT reproduce mul-then-add and differs by ~2.4e-07.
        The bit-identity claim holds on the recorded CUDA path only, which is
        why anything outside that scope must fall back to the reference.
        """
        producer = _producer()
        z, d, eps, wz, wd, madj, noise_std = _inputs()
        ref = producer.reference_update(z, d, wz, wd, madj, noise_std, eps)
        got = producer.addcmul_update(z, d, wz, wd, madj, noise_std, eps)
        assert not torch.equal(ref, got), (
            "if this now matches, CPU FMA contraction changed and the scope "
            "note should be revisited"
        )
        # Small enough to confirm the arithmetic is right; the ISSUE is that it
        # is not bit-exact, and ~1 ULP was already shown to amplify.
        assert float((ref - got).abs().max()) < 1e-5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_the_selected_candidate_is_bit_identical_on_cuda(self):
        """The claim's actual platform."""
        producer = _producer()
        args = [
            a.cuda() if torch.is_tensor(a) else a for a in _inputs(batch=2, vocab=64)
        ]
        z, d, eps, wz, wd, madj, noise_std = args
        ref = producer.reference_update(z, d, wz, wd, madj, noise_std, eps)
        got = producer.addcmul_update(z, d, wz, wd, madj, noise_std, eps)
        assert torch.equal(ref, got)

    def test_the_rejected_collapse_is_not_bit_identical(self):
        """The harness must DISCRIMINATE. A gate that cannot fail proves
        nothing, and this is the candidate it had to reject."""
        producer = _producer()
        z, d, eps, wz, wd, madj, noise_std = _inputs()
        ref = producer.reference_update(z, d, wz, wd, madj, noise_std, eps)
        got = producer.collapsed_update(z, d, wz, wd, madj, noise_std, eps)
        assert not torch.equal(ref, got)
        # ...but it IS algebraically correct, so the difference is tiny.
        assert float((ref - got).abs().max()) < 1e-5

    def test_a_deliberately_wrong_update_is_caught(self):
        """Guards the guard: the difference from a genuinely wrong coefficient
        must dwarf fp32 reassociation noise."""
        producer = _producer()
        z, d, eps, wz, wd, madj, noise_std = _inputs()
        ref = producer.reference_update(z, d, wz, wd, madj, noise_std, eps)
        wrong = (1.0 - noise_std) * d + (noise_std * 1.01) * eps
        collapsed = producer.collapsed_update(z, d, wz, wd, madj, noise_std, eps)
        assert float((ref - wrong).abs().max()) > 100 * float(
            (ref - collapsed).abs().max()
        )

    def test_no_candidate_mutates_its_inputs(self):
        """MEASURED, and it invalidated an earlier negative result.

        `in_place_update` accumulates into a freshly allocated `acc`; z, d and
        eps are read-only. The benchmark previously cloned d and eps for that
        arm and charged the copies as mandatory, producing a spurious 0.82x.
        Since nothing is mutated, no arm needs defensive copies and all timings
        are directly comparable.
        """
        producer = _producer()
        for name, fn in producer.CANDIDATES.items():
            z, d, eps, wz, wd, madj, noise_std = _inputs()
            z0, d0, eps0 = z.clone(), d.clone(), eps.clone()
            out = fn(z, d, wz, wd, madj, noise_std, eps)
            assert torch.equal(z, z0), f"{name} mutated z"
            assert torch.equal(d, d0), f"{name} mutated d"
            assert torch.equal(eps, eps0), f"{name} mutated eps"
            for held, label in ((z, "z"), (d, "d"), (eps, "eps")):
                assert out.data_ptr() != held.data_ptr(), (
                    f"{name} returned a view of {label}"
                )

    def test_the_benchmark_clones_nothing(self):
        """A clone in one arm would tax that arm alone and bias the ratio."""
        producer = _producer()
        code = _code_only(inspect.getsource(producer.local_benchmark))
        assert ". clone ( )" not in code
        assert "mutates" not in code

    def test_the_selected_candidate_is_named(self):
        producer = _producer()
        assert producer.SELECTED == "addcmul"
        assert set(producer.CANDIDATES) == {"addcmul", "collapsed", "in_place"}


class TestCorrectnessOnlyDiscipline:
    def test_the_artifact_declares_it_records_no_latency_claim(self):
        """The two latency kinds are separated: this producer DOES record a
        local microbenchmark, so a bare `records_latency: false` was
        misleading."""
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert '"correctness_only": True' in source
        assert '"records_end_to_end_latency": False' in source
        assert '"records_local_microbenchmark_latency": True' in source
        # Fragments that do not cross a source-line split: the literal is
        # concatenated across lines, so the joined text still carries quotes.
        assert "No outer-wall " in source
        assert "measurement is performed here" in source
        assert "end-to-end gain" in source

    def test_the_local_benchmark_is_labelled_local(self):
        producer = _producer()
        source = inspect.getsource(producer.local_benchmark)
        assert "LOCAL only" in source

    def test_no_production_source_is_modified(self):
        """Benchmark-local only: the harness must not patch the sampler."""
        producer = _producer()
        code = _code_only(pathlib.Path(producer.__file__).read_text())
        for forbidden in ("_install_observer", "run_fmlm_request", "sampler."):
            assert forbidden not in code, forbidden


class TestArtifactProseMatchesMeasurement:
    """The committed artifact's descriptions must agree with its own numbers.

    Hand-copied rounded speedups drifted from the record once already —
    `1.00x / 1.03x / 0.99x` written against `0.9981 / 1.0240 / 0.9953` measured —
    and a peak-memory note kept describing a skip that no longer happened. These
    tests read the committed artifact, so the two cannot diverge silently again.
    """

    @staticmethod
    def _artifact():
        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "docs"
            / "artifacts"
            / "166-fmlm-state-update-agreement.json"
        )
        if not path.exists():
            pytest.skip("artifact not present")
        import json

        return json.loads(path.read_text())

    def test_the_rejection_reasons_quote_no_figures(self):
        """Prose that carries numbers has to be kept in sync by hand, and it
        was not. The numbers belong in `local_microbenchmark`."""
        import re

        rejected = self._artifact()["run"]["rejected_candidates"]
        pattern = re.compile(r"\b\d+\.\d+\s*x\b")
        for name, entry in rejected.items():
            found = pattern.findall(entry["reason"])
            assert not found, f"{name} reason quotes {found}; cite the record"

    def test_the_speedup_summary_is_derived_from_the_raw_timings(self):
        """Recomputed INDEPENDENTLY from `reference_ms` and each candidate's
        `ms`, not from the producer's own `local_speedup` field.

        Comparing the summary against `local_speedup` would only prove the
        producer agrees with itself: if its derivation were wrong, both sides
        would carry the same error and the test would pass. The expected value
        is therefore rebuilt from the raw milliseconds here.
        """
        artifact = self._artifact()
        summary = artifact["local_speedup_summary"]
        for row in artifact["local_microbenchmark"]:
            batch = str(row["batch"])
            reference_ms = row["reference_ms"]
            assert reference_ms > 0
            for name, measured in row["candidates"].items():
                recorded = summary[name][batch]
                if measured.get("skipped"):
                    assert recorded is None
                    continue
                expected = round(reference_ms / measured["ms"], 4)
                assert recorded == pytest.approx(expected, abs=1e-9), (
                    f"{name} b={batch}: summary {recorded} against "
                    f"{reference_ms} / {measured['ms']} = {expected}"
                )

    def test_the_rounding_rule_is_fixed_at_four_places(self):
        """Pinned so a change to the producer's rounding is a visible decision
        rather than a silent drift in what the artifact reports."""
        artifact = self._artifact()
        for name, batches in artifact["local_speedup_summary"].items():
            for batch, value in batches.items():
                if value is None:
                    continue
                assert value == round(value, 4), f"{name} b={batch}: {value}"

    def test_the_candidate_speedup_field_agrees_with_its_own_timings(self):
        """The `local_speedup` field itself must equal reference_ms / ms."""
        for row in self._artifact()["local_microbenchmark"]:
            for name, measured in row["candidates"].items():
                if measured.get("skipped"):
                    continue
                assert measured["local_speedup"] == pytest.approx(
                    row["reference_ms"] / measured["ms"], rel=1e-9
                ), f"{name} b={row['batch']}"

    def test_the_peak_note_does_not_claim_a_skip_that_did_not_happen(self):
        artifact = self._artifact()
        for row in artifact["local_microbenchmark"]:
            peak = row.get("measured_peak_allocated_gib")
            if not peak:
                continue
            skipped = [
                name for name, v in row["candidates"].items() if v.get("skipped")
            ]
            if not skipped:
                assert "nothing is skipped" in peak["note"], (
                    "the note describes a skip, but every candidate was measured"
                )

    def test_every_candidate_was_measured_at_every_batch(self):
        artifact = self._artifact()
        for row in artifact["local_microbenchmark"]:
            for name, v in row["candidates"].items():
                assert not v.get("skipped"), f"{name} skipped at b={row['batch']}"
                assert v["ms"] is not None

    def test_the_erratum_records_the_withdrawn_figure(self):
        """The retraction stays visible; only the live claim is corrected."""
        entry = self._artifact()["run"]["rejected_candidates"]["in_place"]
        assert "0.82x" in entry["erratum"]
        assert "clones were never required" in entry["erratum"]
        assert "0.82" not in entry["reason"]


class TestSequentialDigestReporting:
    @staticmethod
    def _side(digests, tokens):
        return {
            "steps": [
                {"digest": v, "shape": [1, 4, 4], "dtype": "torch.float32"}
                for v in digests
            ],
            "tokens": tokens,
            "final_latent": {
                "digest": "f",
                "shape": [1, 4, 4],
                "dtype": "torch.float32",
            },
            "rng_cpu": torch.zeros(8, dtype=torch.uint8),
            "rng_cuda": torch.zeros(8, dtype=torch.uint8),
            "executed_metadata": {"steps_executed": 32},
        }

    def test_the_delta_is_null_not_zero(self):
        """No subtraction is performed in this mode, so a measured-looking 0.0
        would misstate the method."""
        producer = _producer()
        tokens = torch.zeros(1, 4, dtype=torch.long)
        record = producer.sequential_compare(
            self._side(["a", "b"], tokens), self._side(["a", "b"], tokens), "order"
        )
        assert record["per_step_max_abs_delta"] is None
        assert "not concurrently resident" in record["per_step_max_abs_delta_reason"]
        assert record["per_step_bit_equal_inferred_from_raw_digest"] is True

    def test_a_digest_mismatch_is_reported_with_its_index(self):
        producer = _producer()
        tokens = torch.zeros(1, 4, dtype=torch.long)
        record = producer.sequential_compare(
            self._side(["a", "b", "c"], tokens),
            self._side(["a", "X", "c"], tokens),
            "order",
        )
        assert record["first_mismatch_step"] == 1
        assert record["per_step_digest_equal_count"] == 2
        assert record["all_identical"] is False

    def test_a_shape_change_is_not_hidden_by_an_equal_digest(self):
        """Identical bytes under a different shape would otherwise read as
        equal."""
        producer = _producer()
        tokens = torch.zeros(1, 4, dtype=torch.long)
        a = self._side(["a"], tokens)
        b = self._side(["a"], tokens)
        b["steps"][0]["shape"] = [4, 1, 4]
        record = producer.sequential_compare(a, b, "order")
        assert record["per_step_digest_equal_count"] == 0

    def test_the_mode_and_its_reason_are_recorded(self):
        producer = _producer()
        tokens = torch.zeros(1, 4, dtype=torch.long)
        record = producer.sequential_compare(
            self._side(["a"], tokens), self._side(["a"], tokens), "order"
        )
        assert record["comparison_mode"] == "sequential_raw_digest"
        assert "exceeds device memory" in record["comparison_reason"]
        assert record["digest_algorithm"].startswith("sha256")


class TestFmlmEndpointContract:
    def test_no_post_decode_masking_is_implied(self):
        """FMLM returns `z.argmax(-1)` directly; `mask_after_eos` is
        ELF-specific and must not be represented as an FMLM stage."""
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert '"public_tokens_are_raw_endpoint_tokens": True' in source
        assert '"post_decode_masking": "none"' in source
        assert "mask_after_eos" not in _code_only(
            pathlib.Path(producer.__file__).read_text()
        )

    def test_the_sampler_really_has_no_masking_stage(self):
        """Verified against the source, not assumed: if FMLM ever gains a
        masking stage, this contract must be revisited."""
        sampler = pathlib.Path(
            importlib.util.find_spec("unturtle_flm.sampler").origin
        ).read_text()
        fmlm = sampler.split("def run_fmlm_request")[1]
        assert "mask_after_eos" not in fmlm


class TestScopeAndRejections:
    def test_the_bit_identity_claim_is_scoped(self):
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        for key in ("torch", "cuda", "dtype", "autocast", "execution", "layout"):
            assert f'"{key}"' in source, key
        assert "a claim that addcmul rounds like separate mul+add" in source

    def test_both_rejections_are_recorded_with_reasons(self):
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert '"collapsed"' in source and '"in_place"' in source
        assert "476/1024" in source
        # The figure is whatever the clone-free benchmark measures; what must
        # be recorded is that the rejection rests on a measured non-material
        # result, not on an artifact of unnecessary copies.
        assert "non-material" in source or "SLOWER" in source

    def test_the_amplification_evidence_is_retained(self):
        """The rejection rests on it: ~1 ULP locally, O(1) after feedback."""
        producer = _producer()
        source = inspect.getsource(producer.provenance)
        assert "1.19e-07" in source
        assert "8.61e-01" in source
        assert "RNG states" in source and "stayed identical" in " ".join(source.split())

    def test_the_diagnostic_seeds_also_check_terminal_rng(self):
        """Lockstep shares one RNG stream, so it cannot show terminal RNG
        equality. Omitting the independent check while reporting
        `all_identical` overstated what had been verified."""
        producer = _producer()
        source = inspect.getsource(producer.main)
        diagnostic = source.split("for seed in DIAGNOSTIC_SEEDS:")[1]
        assert "terminal_rng_equality(" in diagnostic
        assert '"terminal_cpu_rng_equal": cpu_eq' in diagnostic
        assert '"terminal_cuda_rng_equal": cuda_eq' in diagnostic
        # ...and they must feed the verdict, not merely be recorded.
        verdict = diagnostic.split('record["all_identical"] = all(')[1]
        assert "cpu_eq" in verdict and "cuda_eq" in verdict

    def test_the_selected_candidate_is_not_described_as_preserving_op_count(self):
        """`addcmul` issues one mul and three addcmul, not the reference's seven
        full-size ops. Preserving the RESULT is the claim; reducing
        materialization is the mechanism."""
        producer = _producer()
        doc = inspect.getdoc(producer.addcmul_update)
        assert "REDUCING full-size tensor materialization" in doc
        assert "does NOT execute the reference's seven" in doc

    def test_the_diagnostic_seeds_are_excluded_from_the_formal_claim(self):
        producer = _producer()
        source = inspect.getsource(producer.main)
        assert '"excluded_from_formal_claim": True' in source
        assert producer.DIAGNOSTIC_SEEDS == (101, 102, 103, 104, 105)
        assert producer.DIAGNOSTIC_BATCH == 1

    def test_the_frozen_cells_are_pinned(self):
        producer = _producer()
        assert producer.STEPS == 32
        assert producer.GAMMA == 1.0
        assert producer.SEED == 100
        assert producer.MAX_LENGTH == 1024
        assert producer.FORMAL_BATCHES == (1, 8, 32)

    def test_lockstep_is_used_where_memory_allows(self):
        """Batch 32 falls back to digests only because both arms cannot be
        resident; 1 and 8 get the stricter comparison."""
        producer = _producer()
        assert producer.LOCKSTEP_MAX_BATCH == 8
        source = inspect.getsource(producer.main)
        assert "batch <= LOCKSTEP_MAX_BATCH" in source
