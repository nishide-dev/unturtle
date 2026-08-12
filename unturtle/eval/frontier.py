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

"""Canonical cross-family quality–diversity–compute frontier surface (#152).

The decision surface every #151 method issue must use.  **Measurement only**:
this module produces frontier POINTS and typed cells; it deliberately exports
no ranking, no single-scalar composite, and no winner function — a frontier
that needs an aggregate score to look stable is not stable (#152 stop
condition).

Frozen protocol (version 1 — see :data:`FRONTIER_PROTOCOL`; recorded on
issue #152 before any new-family quality flowed through this surface):

- OpenWebText free generation at context length 1024;
- throughput at batch 1 / 8 / 32, every cell typed — an OOM or unsupported
  batch is DATA (:func:`missing_cell`), never an omission;
- GenPPL never travels without its evaluator identity and its unigram-
  entropy partner (GenPPL alone is entropy-sensitive, arXiv:2604.02718);
- executed steps are recorded — a record carrying only requested steps is
  rejected (samplers may terminate early or round step counts);
- Tier-A verdict roles: ar_control / masked_discrete / uniform_state /
  embedding_flow / flow_map.  DFM is NOT accepted for uniform_state (the
  lead external anchor is Sumi, whose scale confound must be labeled);
- one RNG generator owns the whole evaluation cell
  (:func:`measure_throughput_cells` enforces this structurally);
- warmup/compile cost is spent once, outside every timed cell;
- decoded TEXT is compared under a common evaluator; native token spaces
  are never forced into one vocabulary.

Composition, not replacement: the #123/#124 ``generation_record`` (schema
version 1) rides inside :func:`frontier_record` unchanged; its consumers
remain behavior-identical.
"""

from __future__ import annotations

import json
import math
import time
from types import MappingProxyType
from typing import Any, Callable

FRONTIER_PROTOCOL_VERSION = 1

FRONTIER_PROTOCOL = MappingProxyType(
    {
        "version": FRONTIER_PROTOCOL_VERSION,
        "dataset": "openwebtext",
        "context_length": 1024,
        "batch_sizes": (1, 8, 32),
        "tier_a_roles": (
            "ar_control",
            "masked_discrete",
            "uniform_state",
            "embedding_flow",
            "flow_map",
        ),
    }
)

_MISSING_CELL_STATUSES = ("oom", "unsupported", "missing")

__all__ = [
    "FRONTIER_PROTOCOL",
    "FRONTIER_PROTOCOL_VERSION",
    "cell",
    "frontier_record",
    "generative_perplexity",
    "genppl_entropy_points",
    "measure_throughput_cells",
    "missing_cell",
    "read_jsonl",
    "speed_quality_points",
    "text_unigram_entropy",
    "tier_a_gaps",
    "write_jsonl",
]


def cell(value: Any) -> dict[str, Any]:
    """A valid measurement cell."""
    return {"status": "ok", "value": value}


def missing_cell(status: str, reason: str) -> dict[str, Any]:
    """A typed invalid/absent cell — recorded, never dropped.

    ``status`` is a closed vocabulary (``oom`` / ``unsupported`` /
    ``missing``) so consumers can filter without guessing; ``reason`` is
    mandatory because an unexplained hole in a decision table is an
    omission wearing a type."""
    if status not in _MISSING_CELL_STATUSES:
        raise ValueError(
            f"unknown cell status {status!r}; expected one of {_MISSING_CELL_STATUSES}"
        )
    if not reason:
        raise ValueError("a missing/invalid cell requires a non-empty reason")
    return {"status": status, "reason": reason}


def _validate_quality(quality: dict[str, Any]) -> None:
    if "genppl" not in quality:
        return
    evaluator = quality.get("genppl_evaluator")
    if not isinstance(evaluator, dict) or not evaluator.get("model"):
        raise ValueError(
            "a GenPPL value must carry its evaluator identity "
            "(quality['genppl_evaluator'] = {'model': ..., 'revision': ...}); "
            "GenPPL numbers are not comparable across evaluators"
        )
    if "unigram_entropy" not in quality:
        raise ValueError(
            "GenPPL without unigram entropy is not a frontier point — GenPPL "
            "alone is entropy-sensitive (arXiv:2604.02718); record both "
            "coordinates"
        )


def _validate_systems(systems: dict[str, Any]) -> None:
    throughput = systems.get("throughput")
    if throughput is None:
        return
    absent = [
        f"batch_{batch_size}"
        for batch_size in FRONTIER_PROTOCOL["batch_sizes"]
        if f"batch_{batch_size}" not in throughput
    ]
    if absent:
        raise ValueError(
            f"throughput is missing typed cells for {absent}; an OOM or "
            "unsupported batch must be a missing_cell(...), not an omission"
        )


def _validate_tier_a_role(family: str, tier_a_role: str | None) -> None:
    if tier_a_role is None:
        return
    if tier_a_role not in FRONTIER_PROTOCOL["tier_a_roles"]:
        raise ValueError(
            f"unknown Tier-A role {tier_a_role!r}; protocol roles: "
            f"{FRONTIER_PROTOCOL['tier_a_roles']}"
        )
    if tier_a_role == "uniform_state" and family == "dfm":
        raise ValueError(
            "DFM is not a substitute for the uniform_state Tier-A role "
            "(#152 rule); use a real non-masked discrete reference — the "
            "lead external anchor is Sumi (label its scale confound)"
        )


def frontier_record(
    *,
    family: str,
    method: str,
    checkpoint: str,
    seed: int,
    quality: dict[str, Any] | None = None,
    systems: dict[str, Any] | None = None,
    decoding: Any = None,
    generation: dict[str, Any] | None = None,
    provider: dict[str, Any] | None = None,
    tier_a_role: str | None = None,
    steps_requested: int | None = None,
    steps_executed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One versioned cross-family measurement record.

    Composes over (never replaces) the #123 ``generation_record`` via the
    ``generation`` field.  Method-specific settings belong in ``decoding`` /
    ``extra`` as tagged fields — there is deliberately no universal
    GenerationConfig.  ``provider`` is plugin provenance (#145) and stays
    ``None`` for builtin/direct methods.

    Raises:
        ValueError: GenPPL without evaluator identity or entropy; a
            protocol batch size without a typed cell; requested steps
            without executed steps; DFM claiming uniform_state; an unknown
            Tier-A role.
    """
    quality = dict(quality or {})
    systems = dict(systems or {})
    _validate_quality(quality)
    _validate_systems(systems)
    _validate_tier_a_role(family, tier_a_role)
    if steps_requested is not None and steps_executed is None:
        raise ValueError(
            "record executed steps, not requested steps alone — samplers "
            "may terminate early or round step counts (#152 rule)"
        )
    return {
        "frontier_schema_version": 1,
        "protocol_version": FRONTIER_PROTOCOL_VERSION,
        "family": family,
        "method": method,
        "checkpoint": checkpoint,
        "seed": seed,
        "tier_a_role": tier_a_role,
        "provider": provider,
        "quality": quality,
        "systems": systems,
        "decoding": decoding,
        "generation": generation,
        "steps_requested": steps_requested,
        "steps_executed": steps_executed,
        "extra": extra or {},
    }


def tier_a_gaps(records: list[dict[str, Any]]) -> tuple[str, ...]:
    """Protocol Tier-A roles with no record claiming them — the machine
    check behind "no cross-family verdict until every role has valid cells
    or an explicit undecidable reason"."""
    covered = {
        record.get("tier_a_role")
        for record in records
        if record.get("tier_a_role") is not None
    }
    return tuple(
        role for role in FRONTIER_PROTOCOL["tier_a_roles"] if role not in covered
    )


def _point_sort_key(record: dict[str, Any]) -> tuple:
    return (
        record.get("family", ""),
        record.get("method", ""),
        record.get("checkpoint", ""),
        record.get("seed", 0),
    )


def genppl_entropy_points(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """GenPPL–entropy frontier points, deterministically ordered, judged by
    nobody: both coordinates travel together and no point is dropped or
    ranked."""
    points = []
    for record in sorted(records, key=_point_sort_key):
        quality = record.get("quality") or {}
        if "genppl" not in quality:
            continue
        points.append(
            {
                "family": record["family"],
                "method": record["method"],
                "checkpoint": record["checkpoint"],
                "seed": record["seed"],
                "genppl": quality["genppl"],
                "unigram_entropy": quality["unigram_entropy"],
                "genppl_evaluator": quality["genppl_evaluator"],
            }
        )
    return points


def speed_quality_points(
    records: list[dict[str, Any]], *, quality_key: str
) -> list[dict[str, Any]]:
    """Speed–quality coordinates per (record, batch size).  Typed non-ok
    cells are carried with their status — a hole in the table is data."""
    points = []
    for record in sorted(records, key=_point_sort_key):
        systems = record.get("systems") or {}
        throughput = systems.get("throughput") or {}
        quality = record.get("quality") or {}
        for batch_size in FRONTIER_PROTOCOL["batch_sizes"]:
            throughput_cell = throughput.get(f"batch_{batch_size}")
            if throughput_cell is None:
                continue
            point = {
                "family": record["family"],
                "method": record["method"],
                "checkpoint": record["checkpoint"],
                "seed": record["seed"],
                "batch_size": batch_size,
                "status": throughput_cell["status"],
                "nfe": systems.get("nfe"),
                quality_key: quality.get(quality_key),
            }
            if throughput_cell["status"] == "ok":
                point.update(throughput_cell["value"])
            else:
                point["reason"] = throughput_cell["reason"]
            points.append(point)
    return points


def write_jsonl(records: list[dict[str, Any]], path: Any) -> None:
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Any) -> list[dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generative_perplexity(
    texts: list[str],
    *,
    evaluator: Callable[[str], tuple[float, int]],
    evaluator_identity: dict[str, str],
) -> dict[str, Any]:
    """Corpus GenPPL under an explicit evaluator: ``exp(total_nll / total_tokens)``.

    ``evaluator`` maps one text to ``(total_nll_nats, token_count)`` under
    the evaluator model; ``evaluator_identity`` (``model`` + ``revision``)
    is mandatory because GenPPL is meaningless without it.  Use
    :func:`hf_causal_evaluator` for a real Hugging Face evaluator.
    """
    if not evaluator_identity.get("model") or not evaluator_identity.get("revision"):
        raise ValueError(
            "evaluator_identity must carry 'model' and 'revision' — GenPPL "
            "values are not comparable across evaluator identities"
        )
    total_nll = 0.0
    total_tokens = 0
    for text in texts:
        nll, token_count = evaluator(text)
        total_nll += nll
        total_tokens += token_count
    return {
        "genppl": math.exp(total_nll / total_tokens),
        "token_count": total_tokens,
        "evaluator": dict(evaluator_identity),
    }


def text_unigram_entropy(
    texts: list[str], *, tokenize: Callable[[str], list[Any]]
) -> float:
    """Unigram entropy (nats) of decoded TEXT under a COMMON tokenization —
    the cross-family entropy coordinate.  Never computed on native token
    ids, which would privilege one vocabulary."""
    counts: dict[Any, int] = {}
    for text in texts:
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
    total = sum(counts.values())
    return -sum(count / total * math.log(count / total) for count in counts.values())


def hf_causal_evaluator(
    model_name: str, *, revision: str, device: str = "cpu"
) -> tuple[Callable[[str], tuple[float, int]], dict[str, str]]:
    """A real GenPPL evaluator over a Hugging Face causal LM (lazy imports;
    heavyweight — for actual runs, not unit tests).  Returns
    ``(evaluator, evaluator_identity)`` ready for
    :func:`generative_perplexity`."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        .to(device)
        .eval()
    )

    def evaluator(text: str) -> tuple[float, int]:
        encoded = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = model(encoded).logits
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = encoded[:, 1:]
        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(token_nll.sum()), int(targets.numel())

    return evaluator, {"model": model_name, "revision": revision}


def measure_throughput_cells(
    run_batch: Callable[[int, Any], Any],
    *,
    seed: int,
    batch_sizes: tuple[int, ...] = FRONTIER_PROTOCOL["batch_sizes"],
    warmup: Callable[[], Any] | None = None,
    unsupported: dict[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Timed throughput cells over the protocol batch sizes.

    Structural protocol enforcement:

    - ONE ``torch.Generator`` (seeded once) is passed to every
      ``run_batch(batch_size, generator)`` call — the RNG belongs to the
      evaluation cell, never reset per batch;
    - ``warmup`` (compile/build/cache) runs exactly once, before any timed
      cell, so its cost cannot leak into one arm;
    - a declared-``unsupported`` batch size is never attempted and becomes a
      typed cell; CUDA OOM becomes a typed ``oom`` cell; any other exception
      is a bug and propagates.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    if warmup is not None:
        warmup()

    cells: dict[str, dict[str, Any]] = {}
    unsupported = unsupported or {}
    for batch_size in batch_sizes:
        key = f"batch_{batch_size}"
        if batch_size in unsupported:
            cells[key] = missing_cell("unsupported", unsupported[batch_size])
            continue
        start = time.perf_counter()
        try:
            run_batch(batch_size, generator)
        except torch.cuda.OutOfMemoryError as error:
            cells[key] = missing_cell("oom", str(error) or "CUDA out of memory")
            continue
        wall_seconds = time.perf_counter() - start
        cells[key] = cell(
            {
                "wall_seconds": wall_seconds,
                "per_sample_latency": wall_seconds / batch_size,
                "samples_per_second": (
                    batch_size / wall_seconds if wall_seconds > 0 else float("inf")
                ),
            }
        )
    return cells
