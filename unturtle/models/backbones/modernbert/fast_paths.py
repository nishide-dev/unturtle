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

"""ModernBERT-diffusion fast-path provider (#185): this family's own knowledge
of *where* its projections live and *which* fused kernels apply.

Extracted from ``FastDiffusionModel._patch_modernbert_peft`` behavior-for-behavior.
ModernBERT uses fused ``Wqkv`` / ``Wo`` (attention) and ``Wi`` / ``Wo`` (MLP);
only the ``Wo`` output projection takes ``apply_lora_o`` — the fused Wqkv/Wi
shapes differ from what ``apply_lora_qkv`` / ``apply_lora_mlp_swiglu`` expect
(#59 Phase 2), so this provider deliberately never widens beyond ``o``.

Family-specific behaviors preserved verbatim:

- when the structure resolves, ``apply_wo`` stubs are installed on CPU *and*
  CUDA, before the device gate, so ``ModernBertAttention_fast_forward`` can
  always dispatch through them (on an untraversable structure stubs are
  skipped — exactly as on ``main``, where the lookup failed first);
- an untraversable structure warns and installs nothing (the family's
  historical fail-open), rather than raising;
- ``Wo`` failing the bias/DoRA gates is silently left standard — only the
  missing-LoRA case warns (historical asymmetry, kept for parity).

Nothing here imports ``unturtle.fast_diffusion_model``.
"""

from __future__ import annotations

import types
from typing import Any, Literal

from unturtle.models.integrations.fast_path_support import (
    apply_lora_o,
    no_bias,
    no_lora_magnitude,
    require_fast_lora,
    warn_once,
)
from unturtle.models.integrations.reports import SupportResult

from ._fast_forward import ModernBertAttention_fast_forward, _install_modernbert_stubs

FAMILY = "modernbert-diffusion"

#: PEFT ``target_modules`` names this family maps onto each fast kind.
#: Wqkv / Wi are named so a request is *recorded* (and then reported as not
#: applied); only ``o`` is installable today.
QKV_TARGETS = frozenset({"Wqkv"})
O_TARGETS = frozenset({"Wo"})
MLP_TARGETS = frozenset({"Wi"})

#: The fast callables this provider installs, by kind (for identity checks).
FAST_CALLABLES = {
    "o": apply_lora_o,
    "attention_forward": ModernBertAttention_fast_forward,
}

_LAYERS_PATH = ("base_model", "model", "model", "layers")


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def decoder_layers(model: Any) -> Any | None:
    """Resolve the encoder layer list of a PEFT-wrapped ModernBERT-diffusion model.

    Path: ``PeftModel → base_model → model (LM) → model (encoder) → layers``.
    The one deep attribute path of this family lives here and nowhere else.
    Returns ``None`` (never raises) when the model is not shaped this way.
    """
    node = model
    for attr in _LAYERS_PATH:
        node = getattr(node, attr, None)
        if node is None:
            return None
    return node


def check_structure(model: Any) -> SupportResult:
    """Typed structural applicability: can this provider reach its layers?"""
    node = model
    reached: list[str] = []
    for attr in _LAYERS_PATH:
        node = getattr(node, attr, None)
        if node is None:
            return SupportResult(
                status="unsupported",
                reason="structure_mismatch",
                details={
                    "expected": ".".join(_LAYERS_PATH),
                    "missing": attr,
                    "reached": ".".join(reached) or "<model>",
                    "family": FAMILY,
                },
            )
        reached.append(attr)
    return SupportResult(
        status="supported",
        details={"layers": len(node), "path": ".".join(_LAYERS_PATH)},
    )


def layer_targets(layer: Any) -> dict[str, Any]:
    """Discover the projection modules of one encoder layer (``None`` if absent)."""
    attn = getattr(layer, "attn", None)
    return {
        "attn": attn,
        "Wqkv": getattr(attn, "Wqkv", None),
        "Wo": getattr(attn, "Wo", None),
    }


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


def wo_applicable(targets: dict[str, Any]) -> bool:
    wo = targets["Wo"]
    return (
        wo is not None
        and hasattr(wo, "lora_A")
        and no_bias(wo)
        and no_lora_magnitude(wo)
    )


def requested_kinds(target_modules: Any, on_cuda: bool) -> tuple[str, ...]:
    """Which fast kinds the PEFT ``target_modules`` ask this family for."""
    names = set(target_modules or ())
    requested: list[str] = []
    if names & QKV_TARGETS:
        requested.append("qkv")
    if names & O_TARGETS:
        requested.append("o")
    if names & MLP_TARGETS:
        requested.append("mlp")
    if on_cuda:
        requested.append("attention_forward")
    return tuple(requested)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def patch_peft(
    model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]
) -> tuple[int, int, int]:
    """Install the ModernBERT fast paths on a PEFT-wrapped model.

    Returns ``(0, n_o, 0)`` — only ``Wo`` takes a fused kernel today.
    Installation is not liveness: the façade's ``probe_liveness`` proves
    execution. An untraversable structure warns and installs nothing (the
    family's historical direct-call behavior; the façade reports it as a typed
    ``structure_mismatch`` fallback before ever calling this).
    """
    n_o = 0

    first_param = next(iter(model.parameters()), None)
    on_cuda = first_param is not None and first_param.device.type == "cuda"

    layers = decoder_layers(model)
    if layers is None:
        warn_once(
            "FastDiffusionModel (ModernBERT): could not locate model.layers — "
            "is this a valid A2DModernBertForMaskedLM PEFT model?"
        )
        return 0, 0, 0

    # Install apply_wo stubs unconditionally (CPU + CUDA) so fast_forward
    # and downstream code can dispatch through apply_wo regardless of device.
    _install_modernbert_stubs(model)

    if not on_cuda:
        return 0, 0, 0

    if lora_dropout == 0 and bias == "none":
        require_fast_lora()

    for layer in layers:
        targets = layer_targets(layer)
        attn = targets["attn"]
        if attn is None:
            continue

        # Always inject bidirectional fast-forward on CUDA
        attn.forward = types.MethodType(ModernBertAttention_fast_forward, attn)

        if lora_dropout != 0 or bias != "none":
            continue

        # Wo output projection — apply Triton apply_lora_o when conditions met
        if wo_applicable(targets):
            # Redirect apply_wo to Triton apply_lora_o.
            # apply_lora_o reads self.o_proj — we alias Wo as o_proj for compatibility.
            attn.o_proj = attn.Wo
            attn.apply_wo = apply_lora_o
            n_o += 1
        elif targets["Wo"] is not None and not hasattr(targets["Wo"], "lora_A"):
            warn_once(
                "FastDiffusionModel (ModernBERT): Wo has no LoRA adapter — "
                "is 'Wo' in target_modules?"
            )

    return 0, n_o, 0


def report(model: Any, counts: tuple[int, int, int]) -> str:
    """The family's post-patch log line."""
    _n_qkv, n_o, _n_mlp = counts
    layers = decoder_layers(model)
    n_layers = len(layers) if layers is not None else 0
    return (
        f"FastDiffusionModel (ModernBERT) patched {n_layers} layers with "
        f"{n_o} Wo (output proj) layers. "
        "Wqkv/MLP Triton kernels not yet supported for ModernBERT — "
        "see issue #59 Phase 2."
    )


__all__ = [
    "FAMILY",
    "FAST_CALLABLES",
    "MLP_TARGETS",
    "O_TARGETS",
    "QKV_TARGETS",
    "check_structure",
    "decoder_layers",
    "layer_targets",
    "patch_peft",
    "report",
    "requested_kinds",
    "wo_applicable",
]
