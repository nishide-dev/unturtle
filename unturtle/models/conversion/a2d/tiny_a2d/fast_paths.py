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

"""Tiny-A2D fast-path provider (#185): the Llama/Qwen-shaped family's own
knowledge of *where* its projections live and *which* fused kernels apply.

Extracted from ``FastDiffusionModel._patch_a2d_peft`` behavior-for-behavior.
The façade keeps PEFT wrapping, the forked-RNG seed (#188), the shared report
types, the liveness probe and the dtype gate (#177); this module owns

- deep structure traversal (``PeftModel → base_model.model.model.layers``),
- target-module discovery per decoder layer,
- QKV / O / MLP applicability (LoRA present, bias-free, no DoRA),
- the installation itself, and the family's post-patch report line,
- a *typed* structure-mismatch reason instead of an ``AttributeError``.

Nothing here imports ``unturtle.fast_diffusion_model``.
"""

from __future__ import annotations

import types
from typing import Any, Literal

from unturtle.models.integrations.fast_path_support import (
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
    has_lora,
    no_bias,
    no_lora_magnitude,
    require_fast_lora,
    warn_once,
)
from unturtle.models.integrations.reports import SupportResult

from ._fast_forward import TinyA2DAttention_fast_forward

FAMILY = "tiny-a2d"

#: PEFT ``target_modules`` names this family maps onto each fast kind.
QKV_TARGETS = frozenset({"q_proj", "k_proj", "v_proj"})
O_TARGETS = frozenset({"o_proj"})
MLP_TARGETS = frozenset({"gate_proj", "up_proj", "down_proj"})

#: The fast callables this provider installs, by kind (for identity checks).
FAST_CALLABLES = {
    "qkv": apply_lora_qkv,
    "o": apply_lora_o,
    "mlp": apply_lora_mlp_swiglu,
    "attention_forward": TinyA2DAttention_fast_forward,
}

_LAYERS_PATH = ("base_model", "model", "model", "layers")


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def decoder_layers(model: Any) -> Any | None:
    """Resolve the decoder layer list of a PEFT-wrapped Tiny-A2D model.

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
    """Discover the projection modules of one decoder layer (``None`` if absent)."""
    attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    return {
        "self_attn": attn,
        "mlp": mlp,
        "q_proj": getattr(attn, "q_proj", None),
        "k_proj": getattr(attn, "k_proj", None),
        "v_proj": getattr(attn, "v_proj", None),
        "o_proj": getattr(attn, "o_proj", None),
        "gate_proj": getattr(mlp, "gate_proj", None),
        "up_proj": getattr(mlp, "up_proj", None),
        "down_proj": getattr(mlp, "down_proj", None),
    }


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------


def _eligible(*projs: Any) -> bool:
    return (
        all(p is not None for p in projs)
        and has_lora(*projs)
        and all(no_bias(p) for p in projs)
        and all(no_lora_magnitude(p) for p in projs)
    )


def qkv_applicable(targets: dict[str, Any]) -> bool:
    return _eligible(targets["q_proj"], targets["k_proj"], targets["v_proj"])


def o_applicable(targets: dict[str, Any]) -> bool:
    return _eligible(targets["o_proj"])


def mlp_applicable(targets: dict[str, Any]) -> bool:
    return _eligible(targets["gate_proj"], targets["up_proj"], targets["down_proj"])


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
    """Install the Tiny-A2D fast paths on a PEFT-wrapped model.

    Returns ``(n_qkv, n_o, n_mlp)`` — the number of layers whose QKV / O / MLP
    received a fused kernel. Installation is not liveness: the façade's
    ``probe_liveness`` proves execution.
    """
    n_qkv = n_o = n_mlp = 0

    layers = decoder_layers(model)
    if layers is None:
        structure = check_structure(model)
        raise AttributeError(
            f"Tiny-A2D fast paths: {structure.reason} — expected "
            f"{structure.details['expected']}, missing {structure.details['missing']!r} "
            f"after {structure.details['reached']}"
        )

    # Triton kernels and flash attention require the model to be on CUDA.
    first_param = next(iter(model.parameters()), None)
    on_cuda = first_param is not None and first_param.device.type == "cuda"

    if on_cuda and lora_dropout == 0 and bias == "none":
        require_fast_lora()

    for layer in layers:
        targets = layer_targets(layer)

        # --- fast attention (bidirectional) — GPU only ---
        if on_cuda:
            targets["self_attn"].forward = types.MethodType(
                TinyA2DAttention_fast_forward, targets["self_attn"]
            )

        if not on_cuda or lora_dropout != 0 or bias != "none":
            # Triton custom autograd does not support dropout or bias in LoRA
            continue

        # --- MLP patching ---
        if mlp_applicable(targets):
            mlp = targets["mlp"]
            mlp.forward = types.MethodType(apply_lora_mlp_swiglu, mlp)
            n_mlp += 1
        else:
            warn_once(
                "FastDiffusionModel: cannot patch MLP layer with Triton LoRA kernel "
                "(LoRA adapters not enabled or bias present)."
            )

        # --- QKV patching ---
        if qkv_applicable(targets):
            targets["self_attn"].apply_qkv = apply_lora_qkv
            n_qkv += 1
        else:
            warn_once(
                "FastDiffusionModel: cannot patch QKV with Triton kernel "
                "(LoRA adapters not enabled or bias present — e.g. Dream q/k/v_proj)."
            )

        # --- O projection patching ---
        if o_applicable(targets):
            targets["self_attn"].apply_o = apply_lora_o
            n_o += 1
        else:
            warn_once(
                "FastDiffusionModel: cannot patch O projection with Triton kernel."
            )

    return n_qkv, n_o, n_mlp


def report(model: Any, counts: tuple[int, int, int]) -> str:
    """The family's post-patch log line."""
    n_qkv, n_o, n_mlp = counts
    layers = decoder_layers(model)
    n_layers = len(layers) if layers is not None else 0
    return (
        f"FastDiffusionModel patched {n_layers} layers with "
        f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers "
        f"(bidirectional, causal=False)."
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
    "mlp_applicable",
    "o_applicable",
    "patch_peft",
    "qkv_applicable",
    "report",
    "requested_kinds",
]
