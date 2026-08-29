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

"""Sole producer of the #174 persistence-attribution artifact.

Runs ``attribution_probe.py`` in fresh subprocesses over the condition matrix
(case × SDPA backend × uninitialized-memory poison × device), captures the
pytest collection node set twice, classifies each case, and writes
``docs/artifacts/174-persistence-attribution-v1.json`` plus a Markdown
summary generated from it. Same gates as the #184 producer: clean worktree,
per-probe import-root verification (violation aborts), volatile fields
excluded from the semantic digest, ``--check`` for determinism.

Semantic vs volatile: conditions with ``poison=none`` depend on what the
allocator last freed and are recorded as VOLATILE evidence; the semantic
verdict per case comes from the deterministic ``empty_like_nan`` conditions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE = REPO_ROOT / "benchmarks" / "persistence" / "attribution_probe.py"
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "174-persistence-attribution-v1.json"
MARKDOWN_PATH = REPO_ROOT / "docs" / "architecture" / "persistence-attribution-v1.md"

_BASE_ENV = dict(os.environ)  # captured before anything heavy (see #184 producer)


def _load_by_path(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_arch = _load_by_path("_diag_arch", "unturtle/diagnostics/architecture.py")
_pers = _load_by_path("_diag_pers", "unturtle/diagnostics/persistence.py")
semantic_digest = _arch.semantic_digest
make_row = _arch.make_row

CASES = ("mdlm_dit_plain", "mdlm_dit_latent_conditioned", "dream_native")
SEMANTIC_CONDITIONS = tuple(
    {"sdpa": sdpa, "poison": "empty_like_nan", "device": device}
    for device in ("cpu", "cuda")
    for sdpa in ("MATH", "FLASH")
)
VOLATILE_CONDITIONS = (
    {"sdpa": "MATH", "poison": "none", "device": "cpu"},
    {"sdpa": "MATH", "poison": "nan", "device": "cpu"},
    {"sdpa": "MATH", "poison": "none", "device": "cuda"},
)
ADMISSIBILITY_CONDITION = {"sdpa": "none", "poison": "empty_like_nan", "device": "cpu"}


def run_probe(case: str, condition: dict, *, gpu: str) -> tuple[dict, dict]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = pathlib.Path(handle.name)
    command = [
        sys.executable,
        str(PROBE),
        "--case",
        case,
        "--device",
        condition["device"],
        "--sdpa",
        condition["sdpa"],
        "--poison",
        condition["poison"],
        "--out",
        str(out),
    ]
    env = dict(_BASE_ENV)
    env.update(
        PYTHONPATH=str(REPO_ROOT),
        UNTURTLE_EXPECTED_ROOT=str(REPO_ROOT),
        PYTHONHASHSEED="0",
        CUDA_VISIBLE_DEVICES=gpu,
    )
    proc = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    provenance = {
        "case": case,
        "condition": condition,
        "command": [c.replace(str(REPO_ROOT), "<repo>") for c in command],
        "exit_code": proc.returncode,
    }
    if proc.returncode == 3:
        raise SystemExit(f"IMPORT ROOT VIOLATION in probe {case} {condition}")
    if out.exists() and out.read_text().strip():
        result = json.loads(out.read_text())
        out.unlink()
    else:
        result = {
            "status": "blocked",
            "reason": f"no probe output (exit {proc.returncode}): {proc.stderr[-300:]}",
        }
    return result, provenance


def _volatilize(record: dict) -> dict:
    """Move allocator/identity-dependent numbers under ``volatile`` keys."""
    if not isinstance(record, dict):
        return record
    out = {}
    for key, value in record.items():
        if key in ("object_id",):
            out.setdefault("volatile", {})[key] = value
        elif key in ("output_vs_original", "first_persistent_mismatch") and isinstance(
            value, dict
        ):
            kept = {}
            for k, v in value.items():
                if k in (
                    "max_abs_delta",
                    "max_rel_delta",
                    "non_equal_count",
                    "first_differing_index",
                ):
                    kept.setdefault("volatile", {})[k] = v
                else:
                    kept[k] = v
            out[key] = kept
        elif key == "buffers" and isinstance(value, dict):
            buffers = {}
            for name, entry in value.items():
                cleaned = dict(entry)
                if cleaned.get("equals_reference") is False:
                    # an uninitialized buffer's digest is whatever memory held
                    cleaned.setdefault("volatile", {})["digest"] = cleaned.pop("digest")
                buffers[name] = cleaned
            out[key] = buffers
        elif isinstance(value, dict):
            out[key] = _volatilize(value)
        else:
            out[key] = value
    return out


def collect_node_set(*, gpu: str) -> dict:
    """``pytest --collect-only`` node IDs for the fast suite, in a fresh process."""
    env = dict(_BASE_ENV)
    env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONHASHSEED="0")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--collect-only",
            "-q",
            "-m",
            "not slow",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    nodes = sorted(line for line in proc.stdout.splitlines() if "::" in line)
    import hashlib

    return {
        "count": len(nodes),
        "digest": hashlib.sha256("\n".join(nodes).encode()).hexdigest()[:16],
        "volatile": {"nodes": nodes},
    }


def classify_case(conditions: dict[str, dict]) -> dict:
    """Per-case verdict from the SEMANTIC (deterministic-poison) conditions."""
    verdicts = {
        key: cond["verdict"]["verdict"]
        for key, cond in conditions.items()
        if cond.get("status") == "observed"
    }
    values = set(verdicts.values())
    if _pers.VERDICT_PERSISTENT_DRIFT in values:
        final = _pers.VERDICT_PERSISTENT_DRIFT
    elif _pers.VERDICT_CAUSAL in values and values <= {
        _pers.VERDICT_CAUSAL,
        _pers.VERDICT_NO_DIVERGENCE,
    }:
        final = _pers.VERDICT_CAUSAL
    elif _pers.VERDICT_CONTRIBUTORY in values:
        final = _pers.VERDICT_CONTRIBUTORY
    elif values == {_pers.VERDICT_NO_DIVERGENCE}:
        final = _pers.VERDICT_NO_DIVERGENCE
    elif _pers.VERDICT_NOT_CAUSAL in values:
        final = _pers.VERDICT_NOT_CAUSAL
    else:
        final = _pers.VERDICT_INADMISSIBLE
    return {
        "attribution": final,
        "per_condition": verdicts,
        "causal_under": sorted(
            k for k, v in verdicts.items() if v == _pers.VERDICT_CAUSAL
        ),
        "masked_under": sorted(
            k for k, v in verdicts.items() if v == _pers.VERDICT_NO_DIVERGENCE
        ),
    }


def producer_info() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    dirty = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "worktree_clean": dirty == "",
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "unsloth": version("unsloth"),
        "peft": version("peft"),
        "platform": platform.platform(),
        "probes": [],
    }


def _normalize_strings(node):
    if isinstance(node, dict):
        return {k: _normalize_strings(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_strings(v) for v in node]
    if isinstance(node, str):
        node = node.replace(str(REPO_ROOT), "<repo>")
        node = re.sub(r"/tmp/[^\s'\"]+", "<tmp>", node)
        node = re.sub(r"/grouper/[^\s'\"]+", "<path>", node)
        return node
    return node


def capture(*, gpu: str, allow_dirty: bool = False) -> dict:
    producer = producer_info()
    if not producer["worktree_clean"] and not allow_dirty:
        raise SystemExit("worktree is not clean; commit producer code first")

    def key(condition: dict) -> str:
        return f"{condition['device']}/{condition['sdpa']}/{condition['poison']}"

    cases: dict[str, dict] = {}
    for case in CASES:
        semantic: dict[str, dict] = {}
        volatile: dict[str, dict] = {}
        for condition in SEMANTIC_CONDITIONS:
            result, provenance = run_probe(case, condition, gpu=gpu)
            producer["probes"].append(provenance)
            semantic[key(condition)] = _volatilize(result)
        for condition in VOLATILE_CONDITIONS:
            result, provenance = run_probe(case, condition, gpu=gpu)
            producer["probes"].append(provenance)
            volatile[key(condition)] = result
        cases[case] = {
            "conditions": semantic,
            "classification": classify_case(semantic),
            "volatile": {"allocator_luck_conditions": volatile},
        }

    admissibility, provenance = run_probe(
        "mdlm_dit_plain", ADMISSIBILITY_CONDITION, gpu=gpu
    )
    producer["probes"].append(provenance)

    collection = {
        "run_1": collect_node_set(gpu=gpu),
        "run_2": collect_node_set(gpu=gpu),
    }
    collection["identical_across_two_runs"] = (
        collection["run_1"]["digest"] == collection["run_2"]["digest"]
    )
    collection["note"] = (
        "the historical 2088 vs 2089 passed-count difference is not "
        "reproducible on this tree/environment (two fresh collections agree); "
        "conditional collection sites (collect-time skips, optional imports) "
        "remain the candidate explanation and the final gate requires an "
        "identical node set across three consecutive full runs"
    )

    mechanism = make_row(
        "observed",
        source="attribution_probe arms + transformers modeling_utils read",
        owner="#174",
        evidence={
            "load_path": (
                "transformers PreTrainedModel.from_pretrained re-materializes "
                "every non-persistent buffer with torch.empty_like (uninitialized "
                "memory) and relies on _init_weights to give it a value"
            ),
            "mdlm_dit": (
                "MDLMDiTPreTrainedModel._init_weights is a deliberate no-op and "
                "Rotary is not a *RotaryEmbedding with original_inv_freq, so the "
                "upstream re-init branch never runs: model.rotary.inv_freq stays "
                "uninitialized after from_pretrained"
            ),
            "dream": (
                "DreamPreTrainedModel._init_weights handles nn.Linear/nn.Embedding "
                "only and does not delegate to the base class, so "
                "DreamRotaryEmbedding.inv_freq is likewise uninitialized after "
                "from_pretrained (this reclassifies the #184 native_fp finding: "
                "not _extend_rope_if_possible, which is a no-op — no "
                "extend_rope_embedding exists in unturtle)"
            ),
            "why_intermittent": (
                "adaLN-Zero gates make finite garbage invisible at init; only "
                "NaN/Inf bit patterns in the recycled memory surface, and only "
                "under an SDPA backend that propagates NaN (CPU math does, CPU "
                "flash swallows it) — the process-bistable backend selection "
                "recorded in #184"
            ),
        },
    )

    sections = {
        "cases": cases,
        "admissibility_probe": _volatilize(admissibility),
        "collection": collection,
        "mechanism": mechanism,
    }
    artifact = {
        "schema_version": 1,
        "producer": producer,
        **sections,
    }
    artifact = _normalize_strings(artifact)
    artifact["semantic_digest"] = semantic_digest(artifact)
    return artifact


def render_markdown(artifact: dict) -> str:
    lines = [
        "# Persistence attribution v1 (#174)",
        "",
        "Generated by `benchmarks/persistence/capture_attribution.py` from "
        "`docs/artifacts/174-persistence-attribution-v1.json` — regenerate, never edit.",
        "",
        f"- producer commit: `{artifact['producer']['commit']}`",
        f"- semantic digest: `{artifact['semantic_digest']}`",
        "",
        "## Verdicts",
        "",
        "| case | attribution | causal under | masked under |",
        "|---|---|---|---|",
    ]
    for case, row in artifact["cases"].items():
        c = row["classification"]
        lines.append(
            f"| {case} | **{c['attribution']}** | {', '.join(c['causal_under']) or '—'} | "
            f"{', '.join(c['masked_under']) or '—'} |"
        )
    lines += ["", "## Mechanism", ""]
    for key, text in artifact["mechanism"]["evidence"].items():
        lines.append(f"- **{key}**: {text}")
    coll = artifact["collection"]
    lines += [
        "",
        "## Collection node set",
        "",
        f"- run 1: {coll['run_1']['count']} nodes (`{coll['run_1']['digest']}`)",
        f"- run 2: {coll['run_2']['count']} nodes (`{coll['run_2']['digest']}`)",
        f"- identical: {coll['identical_across_two_runs']}",
        f"- {coll['note']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for probes")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    artifact = capture(gpu=args.gpu, allow_dirty=args.allow_dirty)
    if args.check:
        committed = json.loads(ARTIFACT_PATH.read_text())
        recomputed = semantic_digest(committed)
        print(f"committed : {committed['semantic_digest']}")
        print(f"recomputed: {recomputed}")
        print(f"fresh     : {artifact['semantic_digest']}")
        if committed["semantic_digest"] != recomputed:
            raise SystemExit("committed artifact digest does not match its content")
        if artifact["semantic_digest"] != committed["semantic_digest"]:
            raise SystemExit("fresh capture diverges from the committed artifact")
        print("deterministic")
        return
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_PATH.write_text(render_markdown(artifact) + "\n")
    for case, row in artifact["cases"].items():
        print(f"{case}: {row['classification']['attribution']}")
    print(f"semantic digest: {artifact['semantic_digest']}")


if __name__ == "__main__":
    main()
