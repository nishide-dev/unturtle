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

"""Dream fast-path provider (#185): this family's own knowledge of *where* its
projections live and *which* fused kernels apply.

Extracted from ``FastDiffusionModel._patch_dream_peft`` behavior-for-behavior.
Dream's q/k/v_proj carry ``bias=True``, so the QKV kernel is
``apply_lora_qkv_with_bias`` and the QKV gate deliberately does **not** require
bias-free projections (#177's complete fused-path contract for Dream); o_proj
(bias=False) and the swiglu MLP use the standard kernels with the full
bias-free gate.

Family-specific behaviors preserved verbatim:

- the CUDA gate runs FIRST — on CPU the provider returns zeros without ever
  touching the model's structure (historically a CPU model was never traversed);
- only the QKV gate warns when ineligible; O and MLP stay silently standard;
- per-layer tolerance: a layer without ``self_attn`` / ``mlp`` is skipped;
- the injected ``DreamAttention_fast_forward`` covers the non-cache path only;
  cache-enabled block decode delegates internally to the class forward, so
  ``model.generate(..., use_cache=True)`` keeps working on a patched model.

Nothing here imports ``unturtle.fast_diffusion_model``, and nothing here
touches generation defaults (#189) or the #174 RoPE reload fix
(``_install_inv_freq`` lives in ``modeling_dream``).
"""

from __future__ import annotations

import types
from typing import Any, Literal

from unturtle.models.integrations.fast_path_support import (
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv_with_bias,
    has_lora,
    no_bias,
    no_lora_magnitude,
    require_fast_lora,
    warn_once,
)
from unturtle.models.integrations.reports import SupportResult

from .modeling_dream import DreamAttention_fast_forward

FAMILY = "dream"

#: PEFT ``target_modules`` names this family maps onto each fast kind.
QKV_TARGETS = frozenset({"q_proj", "k_proj", "v_proj"})
O_TARGETS = frozenset({"o_proj"})
MLP_TARGETS = frozenset({"gate_proj", "up_proj", "down_proj"})

#: The fast callables this provider installs, by kind (for identity checks).
FAST_CALLABLES = {
    "qkv": apply_lora_qkv_with_bias,
    "o": apply_lora_o,
    "mlp": apply_lora_mlp_swiglu,
    "attention_forward": DreamAttention_fast_forward,
}

_LAYERS_PATH = ("base_model", "model", "model", "layers")


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def decoder_layers(model: Any) -> Any | None:
    """Resolve the decoder layer list of a PEFT-wrapped Dream model.

    Path: ``PeftModel → base_model → model → model.layers`` (Dream wraps
    DreamBaseModel as ``self.model``, same depth as LLaMA). The one deep
    attribute path of this family lives here and nowhere else. Returns
    ``None`` (never raises) when the model is not shaped this way.
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


def qkv_applicable(targets: dict[str, Any]) -> bool:
    """Dream q/k/v have bias=True: the bias kernel is used, so the gate checks
    LoRA presence and no-DoRA but deliberately NOT bias-freeness."""
    projs = (targets["q_proj"], targets["k_proj"], targets["v_proj"])
    return (
        all(p is not None for p in projs)
        and has_lora(*projs)
        and all(no_lora_magnitude(p) for p in projs)
    )


def o_applicable(targets: dict[str, Any]) -> bool:
    o_proj = targets["o_proj"]
    return (
        o_proj is not None
        and has_lora(o_proj)
        and no_bias(o_proj)
        and no_lora_magnitude(o_proj)
    )


def mlp_applicable(targets: dict[str, Any]) -> bool:
    projs = (targets["gate_proj"], targets["up_proj"], targets["down_proj"])
    return (
        targets["mlp"] is not None
        and all(p is not None for p in projs)
        and has_lora(*projs)
        and all(no_bias(p) for p in projs)
        and all(no_lora_magnitude(p) for p in projs)
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
    """Install the Dream fast paths on a PEFT-wrapped model.

    Returns ``(n_qkv, n_o, n_mlp)``. Installation is not liveness: the façade's
    ``probe_liveness`` proves execution. The CUDA gate runs before any
    structure access (the family's historical ordering); an untraversable
    structure on CUDA raises ``AttributeError`` for direct callers — the
    façade reports it as a typed ``structure_mismatch`` fallback before ever
    calling this.
    """
    n_qkv = n_o = n_mlp = 0

    # Triton kernels require the model to be on CUDA.
    first_param = next(iter(model.parameters()), None)
    if first_param is None or first_param.device.type != "cuda":
        return n_qkv, n_o, n_mlp

    layers = decoder_layers(model)
    if layers is None:
        structure = check_structure(model)
        raise AttributeError(
            f"Dream fast paths: {structure.reason} — expected "
            f"{structure.details['expected']}, missing {structure.details['missing']!r} "
            f"after {structure.details['reached']}"
        )

    if lora_dropout == 0 and bias == "none":
        require_fast_lora()

    for layer in layers:
        targets = layer_targets(layer)
        self_attn = targets["self_attn"]

        # Inject Triton RoPE fast forward unconditionally (CUDA already checked above)
        if self_attn is not None:
            self_attn.forward = types.MethodType(DreamAttention_fast_forward, self_attn)

        if lora_dropout != 0 or bias != "none":
            continue

        if self_attn is None:
            continue

        # --- QKV: Dream has bias=True → use apply_lora_qkv_with_bias ---
        if qkv_applicable(targets):
            self_attn.apply_qkv = apply_lora_qkv_with_bias
            n_qkv += 1
        else:
            warn_once(
                "FastDiffusionModel (Dream): cannot patch QKV with Triton kernel "
                "(LoRA adapters not enabled or lora_magnitude_vector present)."
            )

        # --- O projection (bias=False in Dream) ---
        if o_applicable(targets):
            self_attn.apply_o = apply_lora_o
            n_o += 1

        # --- MLP: Dream uses gate_proj/up_proj/down_proj (bias=False) ---
        if mlp_applicable(targets):
            mlp = targets["mlp"]
            mlp.forward = types.MethodType(apply_lora_mlp_swiglu, mlp)
            n_mlp += 1

    return n_qkv, n_o, n_mlp


def report(model: Any, counts: tuple[int, int, int]) -> str:
    """The family's post-patch log line."""
    n_qkv, n_o, n_mlp = counts
    layers = decoder_layers(model)
    n_layers = len(layers) if layers is not None else 0
    return (
        f"FastDiffusionModel (Dream) patched {n_layers} layers with "
        f"{n_qkv} QKV layers (bias kernel), {n_o} O layers and {n_mlp} MLP layers."
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
