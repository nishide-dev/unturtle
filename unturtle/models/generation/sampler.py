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

"""Decoding-algorithm selection for Unturtle dLLM generation.

Makes the decoding algorithm an explicit, first-class choice instead of an implicit
combination of ``MaskedDiffusionGenerationConfig`` flags. Each named algorithm maps to the
flag set that the existing ``diffusion_generate`` dispatch already understands — so this is
pure *selection*, with no generation logic of its own.

Algorithms (discrete masked diffusion):
  - ``mdlm``         : plain MDLM denoising loop
  - ``block_decode`` : Fast-dLLM KV-cache block decode (parallel decode is an option within)
  - ``bd3lm``        : BD3LM block diffusion

The registry is intentionally open: continuous-diffusion algorithms (e.g. ``continuous_*``)
would be added to a separate table when continuous diffusion LMs land. The current loops are
discrete-masked-only.
"""

from __future__ import annotations

from typing import Any

#: Discrete masked-diffusion algorithms -> the diffusion_generate flag set each selects.
DISCRETE_ALGORITHMS: dict[str, dict[str, bool]] = {
    "mdlm": {"use_cache": False, "use_block_diffusion": False},
    "block_decode": {"use_cache": True, "use_block_diffusion": False},
    "bd3lm": {"use_cache": False, "use_block_diffusion": True},
}


def algorithm_to_flags(algorithm: str) -> dict[str, bool]:
    """Return the diffusion_generate flag set for a named discrete algorithm."""
    try:
        return dict(DISCRETE_ALGORITHMS[algorithm])
    except KeyError as exc:
        raise ValueError(
            f"Unknown decoding algorithm {algorithm!r}. "
            f"Supported: {sorted(DISCRETE_ALGORITHMS)}."
        ) from exc


def _supports_block_decode(model: Any) -> bool:
    """True if the model implements the Fast-dLLM block-decode cache hook."""
    return callable(getattr(model, "_model_forward_with_cache", None))


def resolve_algorithm(algorithm: str, model: Any, *, bd3lm_requested: bool) -> str:
    """Resolve ``algorithm`` to a concrete discrete algorithm name.

    ``auto`` picks the fastest discrete path the model supports:
      - BD3LM if requested,
      - else block-decode (Fast-dLLM) when the model supports the cache hook,
      - else plain MDLM.
    An explicit algorithm name is validated and returned as-is.
    """
    if algorithm == "auto":
        if bd3lm_requested:
            return "bd3lm"
        if _supports_block_decode(model):
            return "block_decode"
        return "mdlm"
    if algorithm not in DISCRETE_ALGORITHMS:
        raise ValueError(
            f"Unknown decoding algorithm {algorithm!r}. "
            f"Supported: {sorted(DISCRETE_ALGORITHMS)} (or 'auto')."
        )
    return algorithm
