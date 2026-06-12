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
flag set that the model's ``generate`` dispatch understands — so this is
pure *selection*, with no generation logic of its own.

Algorithms (discrete masked diffusion):
  - ``mdlm``         : plain MDLM denoising loop
  - ``block_decode`` : Fast-dLLM KV-cache block decode (parallel decode is an option within);
                       requires the model to implement ``_model_forward_with_cache`` and opt in
                       via ``supports_block_decode`` (defaults to ``True`` when absent).
  - ``bd3lm``        : Unturtle's masked block diffusion (BD3LM); requires
                       ``_sample_block_diffusion`` (TinyA2D family today).

Algorithm (self-conditioned canvas block diffusion):
  - ``block_ar``     : upstream native canvas block diffusion for the DiffusionGemma family;
                       requires ``_denoising_step`` (the DiffusionGemmaGenerationMixin probe).
                       No mask token is used — the upstream ``GenerationConfig`` governs the
                       generation loop entirely, so ``algorithm_to_flags("block_ar")`` returns
                       ``{}`` (no ``use_cache``/``use_block_diffusion`` injection).

Key distinction — ``bd3lm`` vs ``block_ar``:
  - ``bd3lm``        : Unturtle's *masked* block diffusion; requires a mask token; uses
                       ``_sample_block_diffusion``; flag ``use_block_diffusion=True`` injected.
  - ``block_ar``     : upstream *self-conditioned* canvas block diffusion (DiffusionGemma);
                       no mask token; governed by the upstream generation config; no flags
                       injected.

Explicit algorithm choices are capability-checked: passing an algorithm the model cannot
execute raises ``ValueError`` immediately rather than silently falling back or crashing
mid-generation.

The registry is intentionally open: continuous-diffusion algorithms (e.g. ``continuous_*``)
would be added to a separate table when continuous diffusion LMs land. The masked-diffusion
loops (mdlm/block_decode/bd3lm) are discrete-masked-only; block_ar is for the canvas family.
"""

from __future__ import annotations

from typing import Any

#: Discrete masked-diffusion algorithms -> the generate() flag set each selects.
DISCRETE_ALGORITHMS: dict[str, dict[str, bool]] = {
    "mdlm": {"use_cache": False, "use_block_diffusion": False},
    "block_decode": {"use_cache": True, "use_block_diffusion": False},
    "bd3lm": {"use_cache": False, "use_block_diffusion": True},
}

#: Canvas block-diffusion algorithms -> the generate() flag set each selects.
#: ``block_ar`` injects no flags; the upstream GenerationConfig governs the loop.
CANVAS_ALGORITHMS: dict[str, dict[str, bool]] = {
    "block_ar": {},
}

#: All registered algorithms.
ALL_ALGORITHMS: dict[str, dict[str, bool]] = {
    **DISCRETE_ALGORITHMS,
    **CANVAS_ALGORITHMS,
}


def algorithm_to_flags(algorithm: str) -> dict[str, bool]:
    """Return the generate() flag set for a named algorithm.

    For discrete masked algorithms (mdlm / block_decode / bd3lm) this returns the
    ``use_cache`` / ``use_block_diffusion`` flags that the model's generate dispatch
    understands.

    For ``block_ar`` (DiffusionGemma canvas block diffusion) this returns ``{}`` —
    no flags are injected because the upstream ``GenerationConfig`` governs the loop
    entirely; Unturtle only selects the algorithm, it does not override the config.
    """
    try:
        return dict(ALL_ALGORITHMS[algorithm])
    except KeyError as exc:
        raise ValueError(
            f"Unknown decoding algorithm {algorithm!r}. "
            f"Supported: {sorted(ALL_ALGORITHMS)}."
        ) from exc


def _supports_block_ar(model: Any) -> bool:
    """True if the model is a DiffusionGemma-family canvas block-diffusion model.

    The presence of ``_denoising_step`` is the canonical probe for
    ``DiffusionGemmaGenerationMixin``.  These models use self-conditioned canvas
    block diffusion (no mask token) and their generation loop is governed by the
    upstream ``GenerationConfig`` rather than Unturtle flags.
    """
    return callable(getattr(model, "_denoising_step", None))


def _supports_mdlm(model: Any) -> bool:
    """True if the model implements the masked-diffusion sampling loop.

    The presence of ``_sample`` is the canonical probe for the masked-diffusion
    generation mixin (LLaDA / Dream / TinyA2D / ModernBERT all define it).
    Models without ``_sample`` have no mask-token semantics and cannot run
    mdlm / block_decode / bd3lm algorithms.
    """
    return callable(getattr(model, "_sample", None))


def _supports_block_decode(model: Any) -> bool:
    """True if the model implements the block-decode cache hook AND opts in.

    ``supports_block_decode = False`` lets a backbone that inherits the mixin
    generically (e.g. encoder backbones without KV cache) opt out of the
    block-decode fast path.
    """
    if not getattr(model, "supports_block_decode", True):
        return False
    return callable(getattr(model, "_model_forward_with_cache", None))


def _supports_bd3lm(model: Any) -> bool:
    """True if the model implements BD3LM block-diffusion sampling."""
    return callable(getattr(model, "_sample_block_diffusion", None))


def resolve_algorithm(algorithm: str, model: Any, *, bd3lm_requested: bool) -> str:
    """Resolve ``algorithm`` to a concrete algorithm name.

    ``auto`` picks the fastest path the model supports:
      - ``block_ar`` first, when the model is a DiffusionGemma-family canvas model
        (implements ``_denoising_step``); this takes priority over masked algorithms.
      - Else BD3LM if requested (and the model implements ``_sample_block_diffusion``),
      - else block-decode (Fast-dLLM) when the model supports the cache hook,
      - else plain MDLM.

    Explicit algorithm names are capability-checked:
      - ``"block_ar"`` requires ``_supports_block_ar(model)`` (DiffusionGemma family);
        raises ``ValueError`` mentioning "block_ar" otherwise.
      - ``"mdlm"`` requires ``_supports_mdlm(model)`` (masked-diffusion loop); raises
        ``ValueError`` mentioning "masked" when called on a canvas-block model.
      - ``"block_decode"`` requires ``_supports_block_decode(model)``.
      - ``"bd3lm"`` (explicit or via auto + bd3lm_requested) requires
        ``_supports_bd3lm(model)``; BD3LM is implemented on the TinyA2D family today.
    """
    if algorithm == "auto":
        if _supports_block_ar(model):
            return "block_ar"
        if bd3lm_requested:
            if not _supports_bd3lm(model):
                raise ValueError(
                    f"{type(model).__name__} does not implement BD3LM "
                    f"(_sample_block_diffusion); supported on the TinyA2D family. "
                    f"Use algorithm='mdlm' or 'block_decode'."
                )
            return "bd3lm"
        if _supports_block_decode(model):
            return "block_decode"
        if _supports_mdlm(model):
            return "mdlm"
        raise ValueError(
            f"{type(model).__name__} does not implement any known decoding algorithm "
            "(no _denoising_step, _model_forward_with_cache, or _sample). "
            "Ensure the model is a supported dLLM backbone."
        )
    if algorithm not in ALL_ALGORITHMS:
        raise ValueError(
            f"Unknown decoding algorithm {algorithm!r}. "
            f"Supported: {sorted(ALL_ALGORITHMS)} (or 'auto')."
        )
    if algorithm == "block_ar" and not _supports_block_ar(model):
        raise ValueError(
            f"{type(model).__name__} does not support block_ar "
            f"(native canvas block diffusion, DiffusionGemma family); "
            f"use algorithm='mdlm' or 'block_decode' for masked models."
        )
    if algorithm == "mdlm" and not _supports_mdlm(model):
        raise ValueError(
            f"{type(model).__name__} has no masked-diffusion sampling loop "
            f"(no mask-token semantics); use algorithm='block_ar'."
        )
    if algorithm == "block_decode" and not _supports_block_decode(model):
        raise ValueError(
            f"{type(model).__name__} does not support block-decode "
            f"(no usable KV-cache forward); use algorithm='mdlm'."
        )
    if algorithm == "bd3lm" and not _supports_bd3lm(model):
        raise ValueError(
            f"{type(model).__name__} does not implement BD3LM "
            f"(_sample_block_diffusion); supported on the TinyA2D family. "
            f"Use algorithm='mdlm' or 'block_decode'."
        )
    return algorithm
