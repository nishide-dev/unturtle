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

"""The mutation ledger must cover the production tree — verified by an
INDEPENDENT scan (own pattern literals, own file walk), so a producer-side
scanner bug or a deleted ledger row cannot pass unnoticed."""

from __future__ import annotations

import json
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "184-architecture-contract-v1.json"

# Deliberately a LITERAL copy of the mutation-shape spec, not an import from
# the producer: shared constants would let one bug hide in both places.
PATTERNS = re.compile(
    r"(?:__class__ =)|(?:types\.MethodType\()|(?:\.forward = )"
    r"|(?:\.apply_qkv = )|(?:\.apply_o = )|(?:\.apply_mlp = )|(?:\.apply_wo = )"
    r"|(?:__dict__\.pop\(\"generate\")|(?:os\.environ\[)"
    r"|(?:\.generation_config = )|(?:extend_rope_embedding\()"
    r"|(?:\.max_seq_length = )|(?:\.gradient_checkpointing = )"
    r"|(?:AutoConfig\.register\()|(?:\.o_proj = attn\.Wo)|(?:\.gate_proj = ff_proj)"
    r"|(?:prepare_model_for_kbit_training\()"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text())


def _own_scan() -> list[dict]:
    hits = []
    for path in sorted((REPO_ROOT / "unturtle").rglob("*.py")):
        if "diagnostics" in path.parts:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "``" in stripped:
                continue
            code = stripped.split("#", 1)[0]
            if not PATTERNS.search(code):
                continue
            if code.startswith(("def ", "class ")) and "ensure_default_hub" not in code:
                continue
            hits.append({"file": rel, "code": code.strip()})
    return hits


def test_every_claim_anchors_to_the_tree(artifact):
    """Each ledger claim must still exist in production code (no drift)."""
    file_cache: dict[str, str] = {}
    missing = []
    for row in artifact["runtime_mutations"]["rows"]:
        for claim in row["claims"]:
            matched = False
            for path in (REPO_ROOT / "unturtle").rglob("*.py"):
                rel = path.relative_to(REPO_ROOT).as_posix()
                if not rel.endswith(claim["file"]):
                    continue
                text = file_cache.setdefault(rel, path.read_text(encoding="utf-8"))
                if claim["contains"] in text:
                    matched = True
                    break
            if not matched:
                missing.append((row["mutation_id"], claim))
    assert not missing, f"ledger claims no longer anchored: {missing[:5]}"


def test_every_mutation_site_is_claimed(artifact):
    """Own scan of the production tree: every mutation-shaped line must be
    claimed by some ledger row — a deleted row fails here."""
    rows = artifact["runtime_mutations"]["rows"]
    unclaimed = []
    for hit in _own_scan():
        claimed = any(
            hit["file"].endswith(claim["file"]) and claim["contains"] in hit["code"]
            for row in rows
            for claim in row["claims"]
        )
        if not claimed:
            unclaimed.append(hit)
    assert not unclaimed, f"mutation sites without a ledger row: {unclaimed[:5]}"


def test_ledger_row_typing(artifact):
    for row in artifact["runtime_mutations"]["rows"]:
        for field in (
            "mutation_id",
            "owner",
            "target",
            "applicability",
            "before_identity",
            "after_identity",
            "idempotent",
            "reversible",
            "scope",
            "success_signal",
            "liveness_evidence",
            "classification",
        ):
            assert row.get(field), (row["mutation_id"], field)
    # warning-only success signals must SAY so — that fact carries the
    # "installation-only fast-path success -> REPLACE #185" verdict.
    fast_hooks = [
        row
        for row in artifact["runtime_mutations"]["rows"]
        if "fast_hook" in row["mutation_id"] or "fast_forward" in row["mutation_id"]
    ]
    assert fast_hooks
    for row in fast_hooks:
        assert "warning-only" in row["success_signal"], row["mutation_id"]


def test_process_global_rows_flagged(artifact):
    rows = {r["mutation_id"]: r for r in artifact["runtime_mutations"]["rows"]}
    assert rows["kbit_preparation_env"]["scope"] == "process-global"
    assert rows["autoclass_registration"]["scope"] == "process-global"
    assert rows["default_registry_bootstrap"]["scope"] == "process-global"
    assert rows["rope_extension"]["classification"] == "UNDECIDABLE -> #174"
    assert rows["rope_extension"]["linked_issue"] == 174
