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

from __future__ import annotations

import re


def _extract_last_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} in *text*, or None."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return text[start : pos - 1]


def _to_float(raw: str) -> float | None:
    """Normalise a raw number string to float, or return None."""
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_numeric_answer(text: str) -> float | None:
    """Extract a numeric answer from model output.

    Priority:
    1. Last \\boxed{...} occurrence (supports nested braces), if its content is numeric.
    2. If \\boxed{...} is present but its content is not numeric: last bare number
       in the text strictly before the \\boxed{ marker.
    3. If no \\boxed{...} is present: last bare number anywhere in the text.

    Returns None if no number can be extracted.
    """
    boxed_idx = text.rfind(r"\boxed{")
    if boxed_idx != -1:
        boxed = _extract_last_boxed(text)
        if boxed is not None:
            result = _to_float(boxed)
            if result is not None:
                return result
        # Non-numeric boxed content: search only the text before the \boxed{ marker
        search_text = text[:boxed_idx]
    else:
        search_text = text

    # Fallback: last number (integer or decimal, optional leading minus)
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", search_text)
    if matches:
        return _to_float(matches[-1])
    return None
