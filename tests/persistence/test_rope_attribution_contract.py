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

"""#174 PR 0 contract tests.

1. The decision gate is a pure function over MEASURED fields; the six
   required causal mutants must never yield a CAUSAL verdict.
2. The committed artifact classifies every case and is digest-consistent
   (independent canonicalizer).
3. Live reproduction: a fresh-process probe under the deterministic poison and
   a pinned MATH backend reproduces the divergence and its removal by buffer
   restoration; the FLASH backend masks it — the tolerance-free explanation
   of "passes in isolation, fails in the suite".
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "174-persistence-attribution-v1.json"
PROBE = REPO_ROOT / "benchmarks" / "persistence" / "attribution_probe.py"


# ---------------------------------------------------------------------------
# 1. decision gate — pure-function mutants (CPU, no torch import needed)
# ---------------------------------------------------------------------------


def _arm(load_path, object_id, digest, buffers, equal, max_abs=0.0, any_nan=False):
    return {
        "load_path": load_path,
        "object_id": object_id,
        "persistent_digest": digest,
        "buffers": buffers,
        "output_vs_original": {
            "equal": equal,
            "max_abs_delta": max_abs,
            "any_nan": any_nan,
        },
    }


def _causal_arms():
    """The measured shape of a genuine CAUSAL case."""
    return {
        "original": _arm(
            "original", 1, "W", {"rot": {"digest": "F", "equals_reference": True}}, True
        ),
        "direct_state_dict": _arm(
            "direct_state_dict",
            2,
            "W",
            {"rot": {"digest": "F", "equals_reference": True}},
            True,
        ),
        "reload": _arm(
            "from_pretrained",
            3,
            "W",
            {"rot": {"digest": "G", "equals_reference": False}},
            False,
            max_abs=float("inf"),
            any_nan=True,
        ),
        "reload_restored": _arm(
            "from_pretrained+restored_buffers",
            4,
            "W",
            {"rot": {"digest": "F", "equals_reference": True}},
            True,
        ),
    }


def _classify(arms, sdpa="MATH"):
    from unturtle.diagnostics.persistence import classify_rope_attribution

    return classify_rope_attribution(arms=arms, sdpa_backend=sdpa)["verdict"]


def test_genuine_case_is_causal():
    assert _classify(_causal_arms()) == "ROPE LOAD-PATH CAUSAL"


def test_mutant_skip_rewrite_but_report_it_ran():
    """Reported flags are not inputs: if the buffers were NOT rewritten (digests
    equal the reference) the verdict cannot be CAUSAL, whatever the probe says."""
    arms = _causal_arms()
    arms["reload"]["buffers"]["rot"] = {"digest": "F", "equals_reference": True}
    arms["reload"]["reported_load_path_rewrite"] = True
    assert _classify(arms) != "ROPE LOAD-PATH CAUSAL"


def test_mutant_rewrite_to_original_digest_but_report_changed():
    arms = _causal_arms()
    arms["reload"]["buffers"]["rot"]["equals_reference"] = True
    arms["reload"]["buffers"]["rot"]["reported_changed"] = True
    assert _classify(arms) == "ROPE NOT CAUSAL"


def test_mutant_unrelated_persistent_weight_changed():
    arms = _causal_arms()
    arms["reload"]["persistent_digest"] = "W-prime"
    assert _classify(arms) == "PERSISTENT WEIGHTS DIFFER"


def test_mutant_restore_one_layer_claim_full():
    arms = _causal_arms()
    for name in ("reload", "reload_restored"):
        arms[name]["buffers"]["rot2"] = {"digest": "G2", "equals_reference": False}
    arms["reload_restored"]["restored_buffer_names"] = ["rot", "rot2"]  # the CLAIM
    arms["reload_restored"]["output_vs_original"]["equal"] = False
    arms["reload_restored"]["output_vs_original"]["max_abs_delta"] = 0.5
    verdict = _classify(arms)
    assert verdict != "ROPE LOAD-PATH CAUSAL"
    assert verdict == "ROPE CONTRIBUTORY, NOT SUFFICIENT"


def test_mutant_sdpa_backend_drift_is_inadmissible():
    assert _classify(_causal_arms(), sdpa=None) == "INADMISSIBLE COMPARISON"


def test_mutant_arm1_vs_arm1():
    arms = _causal_arms()
    arms["reload"] = copy.deepcopy(arms["original"])
    arms["reload"]["load_path"] = "from_pretrained"  # lies about the path…
    # …but is the same object; also try the honest object id with a wrong path
    assert _classify(arms) == "INADMISSIBLE COMPARISON"
    arms = _causal_arms()
    arms["reload"]["load_path"] = "original"
    assert _classify(arms) == "INADMISSIBLE COMPARISON"


# ---------------------------------------------------------------------------
# 2. artifact
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT_PATH.exists(), "artifact not generated/committed"
    return json.loads(ARTIFACT_PATH.read_text())


def _prune(node, top=True):
    if isinstance(node, dict):
        return {
            k: _prune(v, top=False)
            for k, v in node.items()
            if not (top and k in ("producer", "volatile")) and k != "volatile"
        }
    if isinstance(node, list):
        return [_prune(v, top=False) for v in node]
    return node


def test_artifact_digest_recomputes(artifact):
    pruned = _prune(dict(artifact))
    pruned.pop("semantic_digest", None)
    canonical = json.dumps(
        pruned, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    assert hashlib.sha256(canonical.encode()).hexdigest() == artifact["semantic_digest"]


def test_artifact_classifies_all_cases(artifact):
    for case in ("mdlm_dit_plain", "mdlm_dit_latent_conditioned", "dream_native"):
        row = artifact["cases"][case]["classification"]
        assert row["attribution"] == "ROPE LOAD-PATH CAUSAL", (case, row)
        # MATH exposes, FLASH masks, on CPU
        assert "cpu/MATH/empty_like_nan" in row["causal_under"], (case, row)
        assert (
            row["per_condition"]["cpu/MATH/empty_like_nan"] == "ROPE LOAD-PATH CAUSAL"
        )
    # the plain DiT cell is exactly the "zero gates hide finite garbage,
    # flash hides NaN" story: FLASH must mask it
    plain = artifact["cases"]["mdlm_dit_plain"]["classification"]
    assert plain["per_condition"]["cpu/FLASH/empty_like_nan"] == (
        "NO DIVERGENCE UNDER THIS CONDITION"
    )


def test_artifact_unpinned_backend_is_inadmissible(artifact):
    assert (
        artifact["admissibility_probe"]["verdict"]["verdict"]
        == "INADMISSIBLE COMPARISON"
    )


def test_artifact_persistent_weights_identical_everywhere(artifact):
    for case, row in artifact["cases"].items():
        for key, cond in row["conditions"].items():
            reload = cond["arms"]["reload"]
            assert reload["first_persistent_mismatch"] is None, (case, key)
            assert (
                reload["persistent_digest"]
                == cond["arms"]["original"]["persistent_digest"]
            )


def test_artifact_collection_deterministic(artifact):
    coll = artifact["collection"]
    assert coll["identical_across_two_runs"] is True
    assert coll["run_1"]["count"] == coll["run_2"]["count"] > 0


# ---------------------------------------------------------------------------
# 3. live reproduction (fresh processes; unsloth chain needs a GPU-visible box)
# ---------------------------------------------------------------------------


def _probe(case: str, sdpa: str, poison: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = pathlib.Path(handle.name)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["UNTURTLE_EXPECTED_ROOT"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--case",
            case,
            "--device",
            "cpu",
            "--sdpa",
            sdpa,
            "--poison",
            poison,
            "--out",
            str(out),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    return json.loads(out.read_text())


@pytest.mark.gpu
@pytest.mark.parametrize(
    "case", ["mdlm_dit_plain", "mdlm_dit_latent_conditioned", "dream_native"]
)
def test_live_math_backend_reproduces_and_restoration_removes(case):
    result = _probe(case, "MATH", "empty_like_nan")
    reload = result["arms"]["reload"]
    restored = result["arms"]["reload_restored"]
    assert reload["first_persistent_mismatch"] is None
    assert all(e["equals_reference"] is False for e in reload["buffers"].values())
    assert all(e["finite"] is False for e in reload["buffers"].values())
    assert reload["output_vs_original"]["equal"] is False
    assert all(e["equals_reference"] is True for e in restored["buffers"].values())
    assert restored["output_vs_original"]["equal"] is True
    assert result["verdict"]["verdict"] == "ROPE LOAD-PATH CAUSAL"


@pytest.mark.gpu
def test_live_flash_backend_masks_the_same_defect():
    result = _probe("mdlm_dit_plain", "FLASH", "empty_like_nan")
    reload = result["arms"]["reload"]
    assert all(
        e["finite"] is False for e in reload["buffers"].values()
    )  # still garbage
    assert reload["output_vs_original"]["equal"] is True  # …but invisible
    assert result["verdict"]["verdict"] == "NO DIVERGENCE UNDER THIS CONDITION"
