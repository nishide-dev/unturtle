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

"""
Canonical free-generation measurement surface (#123).

The metrics #121/#122 built benchmark-locally, promoted so promotion-grade
judgments rest on shared, regression-tested code.  Deliberately
**measurement only**: margins, collapse cutoffs and gate definitions are
each experiment's pre-registered hypotheses and stay in the experiment.

The diversity guard trio — each is blind alone (#121/#122 reviews):

- :func:`distinct_fraction`   per-row intra-sample repetition;
- :func:`pooled_unigram_entropy`  CORPUS-pooled token skew (ceiling
  ``ln(B*L)`` pooled tokens, not ``ln(vocab)``); cannot see a batch that
  collapsed to one diverse-looking row;
- :func:`unique_rows_fraction`  inter-sample mode collapse — the case the
  other two cannot see.

MAUVE is the primary free-generation quality metric (#122: it is not
rewarded by repetitive low-entropy text, unlike likelihood judges).  It is
an OPTIONAL dependency: ``import unturtle.eval`` never imports it (the
``lm_eval`` rule), and :func:`mauve_score` fails actionably when absent.

The computations are kept operation-for-operation identical to the #122
benchmark's inline versions so that porting a frozen benchmark onto this
surface is bit-neutral.
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from typing import Any, Callable

import torch


def distinct_fraction(samples: torch.Tensor) -> float:
    """Mean per-row share of distinct tokens, in ``[1/L, 1]``."""
    return (
        float(
            torch.tensor(
                [row.unique().numel() for row in samples], dtype=torch.float32
            ).mean()
        )
        / samples.shape[1]
    )


def pooled_unigram_entropy(samples: torch.Tensor) -> float:
    """Unigram entropy of the batch's POOLED tokens, in nats.

    Pooled, not per-sample: the ceiling is ``ln(B * L)`` (every pooled token
    distinct), and a batch that collapsed onto one diverse row still scores
    high — pair with :func:`unique_rows_fraction`.
    """
    counts = torch.bincount(samples.reshape(-1)).float()
    frequencies = counts[counts > 0] / counts.sum()
    return float(-(frequencies * frequencies.log()).sum())


def unique_rows_fraction(samples: torch.Tensor) -> float:
    """Share of exactly-distinct rows — the inter-sample collapse guard."""
    return len({tuple(row.tolist()) for row in samples}) / samples.shape[0]


def diversity_guards(samples: torch.Tensor) -> dict[str, float]:
    """The trio under the canonical names #122's records established."""
    return {
        "distinct_fraction": distinct_fraction(samples),
        "pooled_unigram_entropy": pooled_unigram_entropy(samples),
        "unique_rows_fraction": unique_rows_fraction(samples),
    }


def mauve_score(
    reference_texts: list[str],
    generated_texts: list[str],
    *,
    featurize_model_name: str = "gpt2",
    device_id: int = 0,
    max_text_length: int = 256,
    verbose: bool = False,
) -> float:
    """MAUVE between a reference and a generated text distribution.

    Optional dependency: install with ``uv pip install mauve-text``.  The
    feature model defaults to ``gpt2`` (base); record the choice with the
    score — MAUVE values are not comparable across feature models.
    """
    try:
        import mauve
    except ImportError as error:
        raise ImportError(
            "mauve_score requires the optional dependency `mauve-text` "
            "(uv pip install mauve-text); `import unturtle.eval` itself "
            "deliberately does not."
        ) from error
    return float(
        mauve.compute_mauve(
            p_text=reference_texts,
            q_text=generated_texts,
            featurize_model_name=featurize_model_name,
            device_id=device_id,
            max_text_length=max_text_length,
            verbose=verbose,
        ).mauve
    )


def measure_generation(generate_fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run one generation call and return ``(result, wall_seconds)``."""
    start = time.perf_counter()
    result = generate_fn()
    return result, time.perf_counter() - start


def generation_record(
    *,
    metrics: dict[str, Any],
    seed: int,
    decoding: Any = None,
    nfe: int | None = None,
    latency_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-serializable result record: seed and decoding config ride with
    every score (the harness' recording convention, extended to free
    generation).  ``decoding`` accepts a harness ``DecodingConfig`` (or any
    dataclass — serialized via ``asdict``) or a plain dict.
    """
    if is_dataclass(decoding) and not isinstance(decoding, type):
        decoding = asdict(decoding)
    return {
        "schema_version": 1,
        "seed": seed,
        "decoding": decoding,
        "metrics": metrics,
        "nfe": nfe,
        "latency_seconds": latency_seconds,
        "extra": extra or {},
    }


__all__ = [
    "distinct_fraction",
    "diversity_guards",
    "generation_record",
    "mauve_score",
    "measure_generation",
    "pooled_unigram_entropy",
    "unique_rows_fraction",
]
