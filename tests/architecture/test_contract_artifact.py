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

"""Schema, digest and dependency-direction checks for the #184 artifact.

The digest is recomputed with an INDEPENDENT implementation of the pruning
and canonicalization spec — deliberately not the producer's helper, so a
producer bug cannot validate itself.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "184-architecture-contract-v1.json"

ROW_STATUSES = {"observed", "blocked", "unsupported", "unverified"}
TOP_SECTIONS = {
    "imports",
    "public_api",
    "models",
    "integrations",
    "runtime_mutations",
    "generation",
    "persistence",
    "process_global_state",
    "registry_hub",
    "verdicts",
}


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT_PATH.exists(), "artifact not generated/committed"
    return json.loads(ARTIFACT_PATH.read_text())


def _independent_prune(node, top=True):
    """Independent implementation of the volatile-pruning spec:
    drop top-level 'producer', drop any dict key named 'volatile'."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if top and key in ("producer", "volatile"):
                continue
            if key == "volatile":
                continue
            out[key] = _independent_prune(value, top=False)
        return out
    if isinstance(node, list):
        return [_independent_prune(v, top=False) for v in node]
    return node


def test_schema_and_sections(artifact):
    assert artifact["schema_version"] == 1
    assert set(artifact) >= TOP_SECTIONS
    producer = artifact["producer"]
    assert re.fullmatch(r"[0-9a-f]{40}", producer["commit"])
    assert producer["worktree_clean"] is True
    for key in ("python", "torch", "transformers", "peft"):
        assert producer[key], f"producer.{key} missing"


def test_every_status_is_typed(artifact):
    """No null-as-unmeasured: every dict carrying a 'status' uses the enum,
    and non-observed rows carry a reason."""
    violations = []

    def walk(node, path):
        if isinstance(node, dict):
            if "status" in node and isinstance(node["status"], str):
                status = node["status"]
                if status not in ROW_STATUSES:
                    violations.append((path, f"invalid status {status!r}"))
                elif status != "observed" and not node.get("reason"):
                    violations.append((path, f"{status} without reason"))
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(artifact, "$")
    assert not violations, violations[:10]


def test_semantic_digest_recomputes(artifact):
    pruned = _independent_prune(dict(artifact))
    pruned.pop("semantic_digest", None)
    canonical = json.dumps(
        pruned, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    assert digest == artifact["semantic_digest"], (
        "independent digest recomputation disagrees with the recorded digest"
    )


def test_no_volatile_paths_in_semantic_content(artifact):
    """Absolute paths and /tmp fragments must not leak into semantic content."""
    pruned = _independent_prune(dict(artifact))
    text = json.dumps(pruned)
    assert "/grouper/" not in text
    assert "/tmp/" not in text


def test_mutation_ledger_complete(artifact):
    ledger = artifact["runtime_mutations"]
    assert ledger["unclaimed_hits"] == [], (
        "mutation-shaped production sites without a ledger row: "
        f"{ledger['unclaimed_hits'][:5]}"
    )
    assert ledger["scanned_hits"] >= len(ledger["rows"]) - 5
    for row in ledger["rows"]:
        assert row["row"]["status"] == "observed", (
            f"ledger row {row['mutation_id']} did not anchor to the tree: {row['row']}"
        )
        assert row["scope"] in ("object-local", "process-global")


def test_required_verdicts_present(artifact):
    verdicts = artifact["verdicts"]
    required = {
        "transformers_native_model_inheritance": "KEEP",
        "fast_diffusion_model_internal_ownership": "EXTRACT -> #185",
        "installation_only_fast_path_success": "REPLACE -> #185",
        "signature_guessing_generation": "REPLACE -> #186",
        "diffusion_gemma_class_swap": "REPLACE -> #186",
        "root_export_growth": "DEPRECATE",
        "universal_hierarchy": "DO NOT CREATE",
        "get_peft_model_random_state": "linked defect -> #188",
        "save_reload_global_state_instability": "RESOLVED -> #174 (uninitialized non-persistent rotary buffers; fixed)",
        "registry_hub_explicit_contract": "KEEP",
    }
    for key, value in required.items():
        assert key in verdicts, key
        assert verdicts[key]["verdict"] == value, (key, verdicts[key]["verdict"])
    # #177 carry-forward: the dtype gate's fail-open branch is UNVERIFIED,
    # never compatible.
    gate = verdicts["dtype_gate_fail_open"]
    assert gate["row"]["status"] == "unverified"
    assert gate["row"]["reason"] == "input_embedding_unresolvable"


def test_diagnostics_never_imported_by_production():
    """unturtle.diagnostics is observation-only; the production tree must not
    import it (one-directional dependency)."""
    offenders = []
    for path in (REPO_ROOT / "unturtle").rglob("*.py"):
        if "diagnostics" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+unturtle\.diagnostics", text, re.M):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, offenders
