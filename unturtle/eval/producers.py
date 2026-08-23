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

"""Tier-A control-producer helpers (#165).

The #152 protocol owns measurement; #153/#155 own the flow families.  This
module owns only the small, SHARED discipline the three remaining control
roles (`ar_control`, `masked_discrete`, `uniform_state`) need, so each
producer script cannot re-invent it differently:

- an AR control must be COMPETENT — a no-cache configuration is refused
  rather than quietly measured (the issue's headline mutation target);
- AR NFE is one forward per generated token, and must be the EXECUTED
  count;
- a control record carries its role, its official column kept separate
  from the canonical one, and its **mandatory** confound labels (scale,
  training data, tokenizer) — the protocol forbids calling an unmatched
  control "matched", so the record cannot omit them;
- the cell's single generator threads through every batch;
- iterative samplers report OBSERVED net revision, never revision
  capability inferred from theory.

Deliberately NOT here: any universal AR/discrete model abstraction, or any
model loading.  Producers stay thin scripts under `benchmarks/`.
"""

from __future__ import annotations

from typing import Any, Callable

from unturtle.eval.frontier import (
    FRONTIER_PROTOCOL,
    frontier_record,
    measure_throughput_cells,
)

__all__ = [
    "ar_generation_config",
    "ar_nfe",
    "build_control_record",
    "measure_control_throughput",
    "net_revision_stats",
]


def ar_generation_config(
    *,
    use_cache: bool = True,
    attn_implementation: str = "sdpa",
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """The frozen competent-AR settings (Stage-0 freeze).

    KV cache is mandatory: #152 requires the AR control to be a competent
    optimized path, and the protocol explicitly forbids comparing a
    compiled/cached diffusion path against a naive AR loop.  Sampling
    carries NO truncation (top-k/top-p) because the diffusion anchors use
    none either.
    """
    if not use_cache:
        raise ValueError(
            "the AR control must run with the KV cache enabled — a no-cache "
            "loop is not a competent optimized control (#152 protocol / "
            "#165 mutation target)"
        )
    return {
        "use_cache": True,
        "attn_implementation": attn_implementation,
        "do_sample": True,
        "temperature": temperature,
        "top_k": None,
        "top_p": None,
        "max_new_tokens": max_new_tokens,
    }


def ar_nfe(*, generated_tokens: int | None) -> int:
    """AR denoiser-call accounting: one forward per generated token.

    Recorded from what was GENERATED, never from a requested length — an
    early EOS shortens the run and the record must follow.  (This number is
    not comparable to a diffusion step count; the producer notes that in
    the record.)
    """
    if generated_tokens is None:
        raise ValueError(
            "AR NFE requires the executed generated-token count; a requested "
            "length is not evidence of what ran (#165 mutation target)"
        )
    return int(generated_tokens)


def build_control_record(
    *,
    role: str,
    family: str,
    method: str,
    checkpoint: str,
    seed: int,
    quality: dict[str, Any],
    systems: dict[str, Any],
    confounds: list[str],
    official: dict[str, Any],
    decoding: Any = None,
    provider: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A protocol-v1 record for one Tier-A control role.

    Adds three producer-level guards on top of `frontier_record`: the role
    must be a protocol role, `confounds` must be non-empty, and the
    official column is forced into `extra` so it can never be conflated
    with the canonical `quality` fields.  DFM-as-uniform_state is rejected
    by `frontier_record` itself.
    """
    if role not in FRONTIER_PROTOCOL["tier_a_roles"]:
        raise ValueError(
            f"unknown Tier-A role {role!r}; protocol roles: "
            f"{FRONTIER_PROTOCOL['tier_a_roles']}"
        )
    if not confounds:
        raise ValueError(
            "confounds must be recorded explicitly (scale / training data / "
            "tokenizer): the protocol forbids presenting an unmatched "
            "control as matched (#165). Pass ['none'] only when the control "
            "is genuinely matched on every axis."
        )
    overlap = set(quality) & {"genppl_official", "entropy_official_native"}
    if overlap:
        raise ValueError(
            f"official-evaluator keys {sorted(overlap)} must not appear in "
            "the canonical quality column — the two evaluator columns stay "
            "separate (#152/#165)"
        )
    merged_extra = dict(extra or {})
    merged_extra["official_column"] = official
    merged_extra["confounds"] = list(confounds)
    return frontier_record(
        family=family,
        method=method,
        checkpoint=checkpoint,
        seed=seed,
        tier_a_role=role,
        provider=provider,
        quality=quality,
        systems=systems,
        decoding=decoding,
        extra=merged_extra,
    )


def measure_control_throughput(
    run_batch: Callable[[int, Any], Any],
    *,
    seed: int,
    warmup: Callable[[], Any] | None = None,
    unsupported: dict[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Protocol throughput cells for a control producer — a thin pass-through
    to `measure_throughput_cells` so every producer inherits the same
    one-generator / warmup-outside / typed-cell discipline."""
    return measure_throughput_cells(
        run_batch, seed=seed, warmup=warmup, unsupported=unsupported
    )


def net_revision_stats(trajectory: list[Any]) -> dict[str, Any]:
    """Measure how much an iterative sampler ACTUALLY revises.

    ``trajectory`` is a list of committed-token snapshots (identical shape,
    one per observed step).  Reports the number of positions whose value
    changed at least once after its first snapshot — evidence about real
    revision, as opposed to the theoretical claim that a uniform/masked
    sampler "can" revise (#152 Sumi note, #165 mutation target).
    """
    if len(trajectory) < 2:
        raise ValueError(
            "net revision needs at least two snapshots; a single state "
            "cannot show whether any token changed"
        )
    import torch

    first = trajectory[0]
    changed = torch.zeros_like(first, dtype=torch.bool)
    previous = first
    for snapshot in trajectory[1:]:
        if snapshot.shape != first.shape:
            raise ValueError(
                f"snapshot shape {tuple(snapshot.shape)} != "
                f"{tuple(first.shape)}; net revision compares aligned states"
            )
        changed |= snapshot != previous
        previous = snapshot
    total = int(changed.numel())
    revised = int(changed.sum())
    return {
        "revised_positions": revised,
        "total_positions": total,
        "revision_fraction": revised / total if total else 0.0,
        "steps_observed": len(trajectory),
    }
