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

"""Paired content drift between two decode paths on one checkpoint (#157).

A distribution metric answers "is this arm's output still good"; it cannot
answer "did the approximate path change what the model would have said". For a
**training-free change on identical weights** the second question is the sharp
one, and it is answerable directly: compare the two runs sample by sample.

Reported as separate columns, deliberately:

- ``exact_token_agreement`` / ``exact_text_agreement`` — different token
  sequences can decode to the same text, so a retokenization is visible as
  text agreement without token agreement. Merging them would hide that;
- edit distances in raw and length-normalized form, since a 5-token edit means
  something different at length 8 and at length 1024;
- ``mean_text_similarity`` for the near-agreement case that exact comparison
  reports as a flat zero;
- ``length_mismatch_fraction``, because a path that truncates is drifting even
  when its shared prefix is identical;
- ``determinism`` — a drift number is only attributable to the decode path if
  the two runs were supposed to agree. Under temperature-1.0 sampling they are
  not, and the record says so rather than letting the reader assume the path
  caused it.

A zero is always reported as a measured zero, never by omitting the field.
"""

from __future__ import annotations

import difflib
from typing import Any

__all__ = ["paired_content_drift", "token_edit_distance"]


def token_edit_distance(reference: list[int], candidate: list[int]) -> int:
    """Levenshtein distance over token ids.

    Used rather than a position-wise mismatch count because a single insertion
    would otherwise report every later position as changed, which would make an
    off-by-one look like total drift.
    """
    if not reference:
        return len(candidate)
    if not candidate:
        return len(reference)
    previous = list(range(len(candidate) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, cand_token in enumerate(candidate, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_token != cand_token),
                )
            )
        previous = current
    return previous[-1]


def paired_content_drift(
    *,
    reference_texts: list[str],
    candidate_texts: list[str],
    reference_ids: list[list[int]],
    candidate_ids: list[list[int]],
    determinism: str = "assumed: same checkpoint, config, seed and sample set; "
    "any disagreement is attributed to the decode path only if this holds",
) -> dict[str, Any]:
    """Drift of `candidate` relative to `reference`, paired per sample.

    `reference` is the exact path (no cache); `candidate` is the
    approximate-reuse path. Refuses mismatched sample counts rather than
    truncating to the shorter run — a silently truncated comparison would
    report drift over a subset while looking like a full one.
    """
    if not reference_texts and not candidate_texts:
        raise ValueError("no samples to compare")
    if len(reference_texts) != len(candidate_texts):
        raise ValueError(
            f"sample count differs: reference {len(reference_texts)} vs "
            f"candidate {len(candidate_texts)}; the runs are not paired and "
            "are not truncated to match"
        )
    if len(reference_ids) != len(reference_texts) or len(candidate_ids) != len(
        candidate_texts
    ):
        raise ValueError(
            f"ids do not match the texts in count: "
            f"{len(reference_ids)}/{len(reference_texts)} reference, "
            f"{len(candidate_ids)}/{len(candidate_texts)} candidate"
        )

    n = len(reference_texts)
    text_matches = 0
    token_matches = 0
    length_mismatches = 0
    edit_total = 0
    normalized_total = 0.0
    similarity_total = 0.0

    for ref_text, cand_text, ref_row, cand_row in zip(
        reference_texts, candidate_texts, reference_ids, candidate_ids, strict=True
    ):
        if ref_text == cand_text:
            text_matches += 1
        if list(ref_row) == list(cand_row):
            token_matches += 1
        if len(ref_row) != len(cand_row):
            length_mismatches += 1
        distance = token_edit_distance(list(ref_row), list(cand_row))
        edit_total += distance
        denominator = max(len(ref_row), len(cand_row), 1)
        normalized_total += distance / denominator
        similarity_total += difflib.SequenceMatcher(None, ref_text, cand_text).ratio()

    return {
        "sample_count": n,
        "exact_text_agreement": text_matches / n,
        "exact_token_agreement": token_matches / n,
        "length_mismatch_fraction": length_mismatches / n,
        "mean_token_edit_distance": edit_total / n,
        "mean_normalized_token_distance": normalized_total / n,
        "mean_text_similarity": similarity_total / n,
        "determinism": determinism,
    }
