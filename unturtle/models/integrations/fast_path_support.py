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

"""Family-agnostic fast-path primitives shared by the façade and the per-family
providers (#185).

Nothing here knows a model's structure. A provider (e.g.
``unturtle.models.conversion.a2d.tiny_a2d.fast_paths``) owns *where* the
projections live; this module owns the guarded kernel import and the
per-projection eligibility predicates every family applies identically.
Kept outside ``fast_diffusion_model`` so a provider never has to import the
façade (that would invert the dependency the extraction exists to remove).
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from unturtle.kernels.fast_lora import (
        apply_lora_mlp_swiglu,
        apply_lora_o,
        apply_lora_qkv,
        apply_lora_qkv_with_bias,
    )
except (ImportError, OSError, AttributeError) as exc:
    apply_lora_mlp_swiglu = None
    apply_lora_o = None
    apply_lora_qkv = None
    apply_lora_qkv_with_bias = None
    FAST_LORA_IMPORT_ERROR: BaseException | None = exc
else:
    FAST_LORA_IMPORT_ERROR = None

_logger = logging.getLogger(__name__)


def require_fast_lora() -> None:
    """Raise the deferred kernel import error when the fused kernels are needed."""
    if FAST_LORA_IMPORT_ERROR is not None:
        raise ImportError(
            "FastDiffusionModel requires unturtle.kernels.fast_lora and its optional "
            "bitsandbytes-backed dependencies to be importable."
        ) from FAST_LORA_IMPORT_ERROR


def warn_once(msg: str) -> None:
    """Log a warning that won't repeat (uses transformers if available)."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logger = hf_logging.get_logger(__name__)
        hf_logger.warning_once(msg)
    except Exception:  # noqa: BLE001
        _logger.warning(msg)


def no_bias(proj: Any) -> bool:
    """The fused kernels take bias-free projections (PEFT-wrapped or plain)."""
    return getattr(proj, "base_layer", proj).bias is None


def no_lora_magnitude(proj: Any) -> bool:
    """DoRA magnitude vectors are not supported by the fused kernels."""
    return len(getattr(proj, "lora_magnitude_vector", []) or []) == 0


def has_lora(*projs: Any) -> bool:
    return all(hasattr(p, "lora_A") for p in projs)


__all__ = [
    "FAST_LORA_IMPORT_ERROR",
    "apply_lora_mlp_swiglu",
    "apply_lora_o",
    "apply_lora_qkv",
    "apply_lora_qkv_with_bias",
    "has_lora",
    "no_bias",
    "no_lora_magnitude",
    "require_fast_lora",
    "warn_once",
]
