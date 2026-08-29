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

"""Observation helpers for the #184 architecture-contract artifact.

Observation-only: this module must never be imported from production code
(loaders, trainers, generation). It is imported by the artifact producer
(``benchmarks/architecture/capture_contract.py``) and by probe subprocesses
AFTER the measured import has happened.

Tests deliberately do NOT reuse these helpers for their comparison values —
independent computation is a #184 requirement so the tests cannot share a
producer bug.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

#: The only statuses an artifact row may carry. ``None`` never means
#: "not measured" — absence of measurement is ``unverified`` with a reason.
ROW_STATUSES = ("observed", "blocked", "unsupported", "unverified")

#: Artifact paths (dot-joined) excluded from the semantic digest. Everything
#: else must be byte-stable across regenerations at the same commit.
VOLATILE_PATHS = ("producer", "volatile")


def make_row(
    status: str,
    *,
    reason: str | None = None,
    source: str,
    owner: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A typed artifact row. ``status`` must be one of :data:`ROW_STATUSES`."""
    if status not in ROW_STATUSES:
        raise ValueError(f"invalid row status {status!r}; expected {ROW_STATUSES}")
    if status != "observed" and reason is None:
        raise ValueError(f"status {status!r} requires a reason")
    return {
        "status": status,
        "reason": reason,
        "source": source,
        "owner": owner,
        "evidence": evidence or {},
    }


def class_fqn(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def mro_fqns(cls: type) -> list[str]:
    return [class_fqn(base) for base in cls.__mro__]


def describe_method_owner(cls: type, name: str) -> dict[str, Any]:
    """Resolve which class in ``cls.__mro__`` actually provides ``name``.

    Uses ``inspect.getattr_static`` (no descriptor protocol, no instance
    ``__getattr__`` games) and then locates the defining class by scanning
    ``vars(base)`` along the MRO — a plain ``hasattr`` says nothing about
    ownership.
    """
    try:
        static = inspect.getattr_static(cls, name)
    except AttributeError:
        return {"status": "absent"}
    defined_in = None
    for base in cls.__mro__:
        if name in vars(base):
            defined_in = class_fqn(base)
            break
    func = getattr(static, "__func__", static)
    return {
        "status": "present",
        "defined_in": defined_in,
        "qualname": getattr(func, "__qualname__", None),
        "module": getattr(func, "__module__", None),
        "kind": type(static).__name__,
    }


def canonical_json(payload: Any) -> str:
    """Deterministic JSON serialization (sorted keys, no whitespace drift)."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _prune_volatile(node: Any, path: tuple[str, ...] = ()) -> Any:
    """Drop volatile subtrees: the top-level ``producer`` section and any
    dict key named ``volatile`` at any depth."""
    if isinstance(node, dict):
        pruned = {}
        for key, value in node.items():
            child_path = path + (key,)
            if not path and key in VOLATILE_PATHS:
                continue
            if key == "volatile":
                continue
            pruned[key] = _prune_volatile(value, child_path)
        return pruned
    if isinstance(node, list):
        return [_prune_volatile(item, path) for item in node]
    return node


def semantic_digest(artifact: dict[str, Any]) -> str:
    """sha256 over the canonical JSON of the non-volatile artifact content.

    The digest field itself is excluded, so the artifact can embed it.
    """
    pruned = _prune_volatile(artifact)
    pruned.pop("semantic_digest", None)
    return hashlib.sha256(canonical_json(pruned).encode()).hexdigest()


def tensor_digest(tensor: Any) -> str:
    """Content digest of a tensor (fp32 bytes on CPU, shape-tagged)."""
    import torch

    with torch.no_grad():
        data = tensor.detach().to("cpu", torch.float32).contiguous()
    payload = str(tuple(data.shape)).encode() + data.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]
