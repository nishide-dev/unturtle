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

"""LLaDA fast-path provider (#185): this family's own knowledge of *where* its
projections live and *which* fused kernels apply.

Extracted from ``FastDiffusionModel._patch_llada_peft``. LLaDA uses a
non-standard hierarchy — ``transformer.blocks`` (not ``model.layers``) at a
depth that varies with how the model was wrapped — and per-kind names
``q/k/v_proj`` / ``attn_out`` / ``ff_proj``+``up_proj``+``ff_out``.

Together with this extraction, ``LLaDALlamaBlock.forward`` and
``LLaDABlock.attention`` now dispatch through ``apply_qkv`` / ``apply_o``
(#185 intended behavior change): before, those hooks were installed but never
called (installed-not-live, #184 ledger). ``apply_lora_o`` reads
``self.o_proj``, so the O install sets the ``o_proj -> attn_out`` alias.

Family-specific behaviors preserved verbatim:

- the CUDA gate runs FIRST — on CPU the provider returns zeros without ever
  touching the model's structure;
- an unresolvable transformer / missing ``blocks`` warns and installs nothing
  (the family's historical fail-open), rather than raising;
- non-``LLaDALlamaBlock`` block types are skipped with a warning;
- Triton RoPE fast forward installs once per rotary module (idempotent);
- the MLP kernel installs only for SiLU activation (SwiGLU halves the gate
  width via ``chunk(2)`` and would shape-mismatch the kernel).

Nothing here imports ``unturtle.fast_diffusion_model``.
"""

from __future__ import annotations

import types
from typing import Any, Literal

import torch

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

FAMILY = "llada"

#: PEFT ``target_modules`` names this family maps onto each fast kind.
QKV_TARGETS = frozenset({"q_proj", "k_proj", "v_proj"})
O_TARGETS = frozenset({"attn_out"})
MLP_TARGETS = frozenset({"ff_proj", "up_proj", "ff_out"})

#: The fast callables this provider installs, by kind (for identity checks).
#: The rope fast forward is built per-module (closure over the class forward),
#: so it is identified by the ``_fast_rope_patched`` marker, not by identity.
FAST_CALLABLES = {
    "qkv": apply_lora_qkv,
    "o": apply_lora_o,
    "mlp": apply_lora_mlp_swiglu,
}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def transformer_of(model: Any) -> Any | None:
    """Resolve the LLaDA transformer of a PEFT-wrapped model.

    ``PeftModel → base_model → model (LLaDAModelLM) → model (LLaDAModel) →
    transformer`` — with a runtime fallback to ``base_model.model.transformer``
    for directly-wrapped backbones. The family's deep attribute paths live
    here and nowhere else. Returns ``None`` (never raises).
    """
    inner = getattr(model, "base_model", None)
    inner = getattr(inner, "model", None)
    if inner is None:
        return None
    if hasattr(inner, "model") and hasattr(inner.model, "transformer"):
        return inner.model.transformer
    return getattr(inner, "transformer", None)


def decoder_blocks(model: Any) -> Any | None:
    """The transformer's block list, or ``None`` when unresolvable."""
    transformer = transformer_of(model)
    return getattr(transformer, "blocks", None)


def check_structure(model: Any) -> SupportResult:
    """Typed structural applicability: can this provider reach its blocks?"""
    transformer = transformer_of(model)
    if transformer is None:
        return SupportResult(
            status="unsupported",
            reason="structure_mismatch",
            details={
                "expected": "base_model.model[.model].transformer.blocks",
                "missing": "transformer",
                "family": FAMILY,
            },
        )
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        return SupportResult(
            status="unsupported",
            reason="structure_mismatch",
            details={
                "expected": "base_model.model[.model].transformer.blocks",
                "missing": "blocks",
                "family": FAMILY,
            },
        )
    return SupportResult(status="supported", details={"blocks": len(blocks)})


def block_targets(block: Any) -> dict[str, Any]:
    """Discover the projection modules of one block (``None`` if absent)."""
    return {
        "q_proj": getattr(block, "q_proj", None),
        "k_proj": getattr(block, "k_proj", None),
        "v_proj": getattr(block, "v_proj", None),
        "attn_out": getattr(block, "attn_out", None),
        "ff_proj": getattr(block, "ff_proj", None),
        "up_proj": getattr(block, "up_proj", None),
        "ff_out": getattr(block, "ff_out", None),
        "act": getattr(block, "act", None),
        "rotary_emb": getattr(block, "rotary_emb", None),
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
    return _eligible(targets["attn_out"])


def mlp_applicable(targets: dict[str, Any]) -> bool:
    """SiLU-only: with SwiGLU the gate output is halved by ``chunk(2)`` while
    ``up_proj`` stays full-width — a shape mismatch in the Triton kernel."""
    act = targets["act"]
    return (
        act is not None
        and isinstance(act, torch.nn.SiLU)
        and _eligible(targets["ff_proj"], targets["up_proj"], targets["ff_out"])
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
        requested.append("rope")
    return tuple(requested)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def patch_peft(
    model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]
) -> tuple[int, int, int]:
    """Install the LLaDA fast paths on a PEFT-wrapped model.

    Returns ``(n_qkv, n_o, n_mlp)``. Installation is not liveness — but since
    the #185 wiring, ``LLaDALlamaBlock.forward`` / ``LLaDABlock.attention``
    dispatch through the installed hooks, and the façade's ``probe_liveness``
    proves execution. The CUDA gate runs before any structure access; an
    unresolvable structure warns and installs nothing (the family's historical
    fail-open for direct calls — the façade reports it as a typed
    ``structure_mismatch`` fallback before ever calling this).
    """
    from unturtle.models.backbones.llada.modeling_llada import (
        LLaDALlamaBlock,
        _make_llada_fast_rope_forward,
    )

    n_qkv = n_o = n_mlp = 0

    # Triton kernels require the model to be on CUDA.
    first_param = next(iter(model.parameters()), None)
    if first_param is None or first_param.device.type != "cuda":
        return n_qkv, n_o, n_mlp

    transformer = transformer_of(model)
    if transformer is None:
        warn_once(
            "FastDiffusionModel (LLaDA): could not locate transformer — "
            "cannot patch LoRA kernels. Is this a supported LLaDA checkpoint?"
        )
        return n_qkv, n_o, n_mlp

    if not hasattr(transformer, "blocks"):
        warn_once(
            "FastDiffusionModel (LLaDA): transformer.blocks not found — "
            "cannot patch LoRA kernels. Is this a supported LLaDA checkpoint?"
        )
        return n_qkv, n_o, n_mlp

    blocks = transformer.blocks

    if lora_dropout == 0 and bias == "none":
        require_fast_lora()

    for block in blocks:
        if not isinstance(block, LLaDALlamaBlock):
            warn_once(
                f"FastDiffusionModel (LLaDA): skipping block type {type(block).__name__} "
                "(only LLaDALlamaBlock is supported for Triton LoRA patching)."
            )
            continue

        targets = block_targets(block)

        # Inject Triton RoPE fast forward unconditionally (CUDA already checked above).
        rotary_emb = targets["rotary_emb"]
        if rotary_emb is not None and not getattr(
            rotary_emb, "_fast_rope_patched", False
        ):
            rotary_emb.forward = types.MethodType(
                _make_llada_fast_rope_forward(type(rotary_emb).forward), rotary_emb
            )
            rotary_emb._fast_rope_patched = True

        if lora_dropout != 0 or bias != "none":
            continue

        # LLaDALlamaBlock: q_proj / k_proj / v_proj (bias depends on config)
        if qkv_applicable(targets):
            block.apply_qkv = apply_lora_qkv
            n_qkv += 1
        else:
            warn_once(
                "FastDiffusionModel (LLaDA): cannot patch QKV with Triton kernel "
                "(LoRA not enabled or bias present — config.include_qkv_bias=True)."
            )

        # attn_out (o_proj equivalent)
        if o_applicable(targets):
            # apply_lora_o reads self.o_proj — alias attn_out for the kernel.
            # Instance-__dict__ assignment, NOT nn.Module registration: a
            # registered alias would duplicate every attn_out entry in
            # state_dict()/save_pretrained (save-format change).
            block.__dict__["o_proj"] = block.attn_out
            block.apply_o = apply_lora_o
            n_o += 1
        else:
            warn_once(
                "FastDiffusionModel (LLaDA): cannot patch attn_out with Triton kernel."
            )

        # ff_proj / up_proj / ff_out — gated MLP (gate/up/down).
        if not (
            targets["act"] is not None and isinstance(targets["act"], torch.nn.SiLU)
        ):
            warn_once(
                f"FastDiffusionModel (LLaDA): skipping Triton MLP patch for "
                f"{type(targets['act']).__name__} activation — only SiLU is supported. "
                "MLP LoRA will use PEFT default path."
            )
        elif mlp_applicable(targets):
            # Set gate_proj/down_proj aliases for apply_lora_mlp_swiglu compatibility.
            block.gate_proj = block.ff_proj
            block.down_proj = block.ff_out
            block.apply_mlp = apply_lora_mlp_swiglu
            n_mlp += 1
        else:
            warn_once(
                "FastDiffusionModel (LLaDA): cannot patch MLP with Triton kernel "
                "(LoRA not enabled, bias present, or magnitude scaling active)."
            )

    return n_qkv, n_o, n_mlp


def report(model: Any, counts: tuple[int, int, int]) -> str:
    """The family's post-patch log line."""
    n_qkv, n_o, _n_mlp = counts
    blocks = decoder_blocks(model)
    n_blocks = len(blocks) if blocks is not None else 0
    return (
        f"FastDiffusionModel (LLaDA) patched {n_blocks} blocks with "
        f"{n_qkv} QKV blocks and {n_o} O (attn_out) blocks."
    )


__all__ = [
    "FAMILY",
    "FAST_CALLABLES",
    "MLP_TARGETS",
    "O_TARGETS",
    "QKV_TARGETS",
    "block_targets",
    "check_structure",
    "decoder_blocks",
    "mlp_applicable",
    "o_applicable",
    "patch_peft",
    "qkv_applicable",
    "report",
    "requested_kinds",
    "transformer_of",
]
