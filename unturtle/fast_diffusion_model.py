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
FastDiffusionModel — analogous to unsloth's FastLanguageModel but for
Diffusion Language Models (dLLMs).

Applies unsloth's Triton-fused LoRA kernels and Flash Attention with
bidirectional (non-causal) masking to A2D / LLaDA / Dream models.

Usage::

    from unturtle import FastDiffusionModel
    from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaLMHeadModel

    model, tokenizer = FastDiffusionModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastDiffusionModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
    )

    # Switch to inference mode (eval + no_grad context manager)
    FastDiffusionModel.for_inference(model)

    # Save LoRA-merged weights
    FastDiffusionModel.save_pretrained_merged(model, "output/merged", tokenizer)
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import logging
import types
from typing import Any, Literal, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from transformers import AutoConfig, AutoTokenizer

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
    _FAST_LORA_IMPORT_ERROR = exc
else:
    _FAST_LORA_IMPORT_ERROR = None

from unturtle.models.backbones.dream.modeling_dream import (
    DreamAttention_fast_forward,
)
from unturtle.models.backbones.modernbert._fast_forward import (
    ModernBertAttention_fast_forward,
    _install_modernbert_stubs,
)
from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
    TinyA2DAttention_fast_forward,
)
from unturtle.save import patch_saving_functions, prepare_model_for_kbit_training

_logger = logging.getLogger(__name__)


def _require_fast_lora() -> None:
    if _FAST_LORA_IMPORT_ERROR is not None:
        raise ImportError(
            "FastDiffusionModel requires unturtle.kernels.fast_lora and its optional "
            "bitsandbytes-backed dependencies to be importable."
        ) from _FAST_LORA_IMPORT_ERROR


def _warn_once(msg: str) -> None:
    """Log a warning that won't repeat (uses transformers if available)."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logger = hf_logging.get_logger(__name__)
        hf_logger.warning_once(msg)
    except Exception:  # noqa: BLE001
        _logger.warning(msg)


# Model types that follow the standard LLaMA/Qwen2 layer hierarchy:
# model.model.model.layers (through PeftModel → base_model → model)
_TINY_A2D_MODEL_TYPES = frozenset(
    [
        "tiny-a2d-llama",
        "tiny-a2d-qwen2",
        "tiny-a2d-qwen3",
        "llama",
        "qwen2",
        "qwen3",
    ]
)

# Dream model_type (note: Dream uses "Dream" with capital D)
_DREAM_MODEL_TYPES = frozenset(["dream", "Dream"])

# LLaDA model_type
_LLADA_MODEL_TYPES = frozenset(["llada"])

# ModernBERT model_type(s) — diffusion wrapper around native bidirectional encoder
_MODERNBERT_A2D_MODEL_TYPES = frozenset(["modernbert-diffusion"])


# ---------------------------------------------------------------------------
# Internal patching helpers
# ---------------------------------------------------------------------------


def _patch_a2d_peft(
    model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]
) -> tuple[int, int, int]:
    """Patch A2D model (standard LLaMA/Qwen2/3 layer layout) with Triton LoRA kernels
    and inject bidirectional fast attention forward.

    Returns (n_qkv, n_o, n_mlp) — number of patched layer types.
    """
    n_qkv = n_o = n_mlp = 0

    # Standard path: PeftModel → base_model → model → model.layers
    layers = model.base_model.model.model.layers

    # Triton kernels and flash attention require the model to be on CUDA.
    first_param = next(iter(model.parameters()), None)
    on_cuda = first_param is not None and first_param.device.type == "cuda"

    if on_cuda and lora_dropout == 0 and bias == "none":
        _require_fast_lora()

    for layer in layers:
        # --- fast attention (bidirectional) — GPU only ---
        if on_cuda:
            layer.self_attn.forward = types.MethodType(
                TinyA2DAttention_fast_forward, layer.self_attn
            )

        if not on_cuda or lora_dropout != 0 or bias != "none":
            # Triton custom autograd does not support dropout or bias in LoRA
            continue

        # --- MLP patching ---
        mlp = layer.mlp
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj
        if (
            hasattr(gate_proj, "lora_A")
            and hasattr(up_proj, "lora_A")
            and hasattr(down_proj, "lora_A")
            and _no_bias(gate_proj)
            and _no_bias(up_proj)
            and _no_bias(down_proj)
            and _no_lora_mag(gate_proj)
            and _no_lora_mag(up_proj)
            and _no_lora_mag(down_proj)
        ):
            mlp.forward = types.MethodType(apply_lora_mlp_swiglu, mlp)
            n_mlp += 1
        else:
            _warn_once(
                "FastDiffusionModel: cannot patch MLP layer with Triton LoRA kernel "
                "(LoRA adapters not enabled or bias present)."
            )

        # --- QKV patching ---
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        if (
            hasattr(q_proj, "lora_A")
            and hasattr(k_proj, "lora_A")
            and hasattr(v_proj, "lora_A")
            and _no_bias(q_proj)
            and _no_bias(k_proj)
            and _no_bias(v_proj)
            and _no_lora_mag(q_proj)
            and _no_lora_mag(k_proj)
            and _no_lora_mag(v_proj)
        ):
            layer.self_attn.apply_qkv = apply_lora_qkv
            n_qkv += 1
        else:
            _warn_once(
                "FastDiffusionModel: cannot patch QKV with Triton kernel "
                "(LoRA adapters not enabled or bias present — e.g. Dream q/k/v_proj)."
            )

        # --- O projection patching ---
        o_proj = layer.self_attn.o_proj
        if hasattr(o_proj, "lora_A") and _no_bias(o_proj) and _no_lora_mag(o_proj):
            layer.self_attn.apply_o = apply_lora_o
            n_o += 1
        else:
            _warn_once(
                "FastDiffusionModel: cannot patch O projection with Triton kernel."
            )

    return n_qkv, n_o, n_mlp


def _patch_dream_peft(
    model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]
) -> tuple[int, int, int]:
    """Patch Dream model with Triton LoRA kernels.

    Dream's q/k/v_proj have ``bias=True``, so the standard ``apply_lora_qkv``
    is replaced with ``apply_lora_qkv_with_bias`` (``LoRA_QKV_Bias`` kernel).
    o_proj (bias=False) uses the standard ``apply_lora_o``.
    MLP (gate/up/down, all bias=False) uses ``apply_lora_mlp_swiglu``.

    Layer layout: ``model.base_model.model.model.layers``
    (Dream wraps DreamBaseModel as ``self.model``, same depth as LLaMA).

    The injected ``DreamAttention_fast_forward`` covers the non-cache path only;
    cache-enabled block decode (tuple KV caches, ``dual_cache`` /
    ``replace_position``) delegates internally to the standard class forward, so
    ``model.generate(..., use_cache=True)`` keeps working on a patched model.
    """
    n_qkv = n_o = n_mlp = 0

    # Triton kernels require the model to be on CUDA.
    first_param = next(iter(model.parameters()), None)
    if first_param is None or first_param.device.type != "cuda":
        return n_qkv, n_o, n_mlp

    layers = model.base_model.model.model.layers

    if lora_dropout == 0 and bias == "none":
        _require_fast_lora()

    for layer in layers:
        self_attn = layer.self_attn if hasattr(layer, "self_attn") else None

        # Inject Triton RoPE fast forward unconditionally (CUDA already checked above)
        if self_attn is not None:
            self_attn.forward = types.MethodType(DreamAttention_fast_forward, self_attn)

        if lora_dropout != 0 or bias != "none":
            continue

        if self_attn is None:
            continue

        # --- QKV: Dream has bias=True → use apply_lora_qkv_with_bias ---
        q_proj = getattr(self_attn, "q_proj", None)
        k_proj = getattr(self_attn, "k_proj", None)
        v_proj = getattr(self_attn, "v_proj", None)
        if (
            q_proj is not None
            and k_proj is not None
            and v_proj is not None
            and hasattr(q_proj, "lora_A")
            and hasattr(k_proj, "lora_A")
            and hasattr(v_proj, "lora_A")
            and _no_lora_mag(q_proj)
            and _no_lora_mag(k_proj)
            and _no_lora_mag(v_proj)
        ):
            self_attn.apply_qkv = apply_lora_qkv_with_bias
            n_qkv += 1
        else:
            _warn_once(
                "FastDiffusionModel (Dream): cannot patch QKV with Triton kernel "
                "(LoRA adapters not enabled or lora_magnitude_vector present)."
            )

        # --- O projection (bias=False in Dream) ---
        o_proj = getattr(self_attn, "o_proj", None)
        if (
            o_proj is not None
            and hasattr(o_proj, "lora_A")
            and _no_bias(o_proj)
            and _no_lora_mag(o_proj)
        ):
            self_attn.apply_o = apply_lora_o
            n_o += 1

        # --- MLP: Dream uses gate_proj/up_proj/down_proj (bias=False) ---
        mlp = layer.mlp if hasattr(layer, "mlp") else None
        if mlp is not None:
            gate_proj = getattr(mlp, "gate_proj", None)
            up_proj = getattr(mlp, "up_proj", None)
            down_proj = getattr(mlp, "down_proj", None)
            if (
                gate_proj is not None
                and up_proj is not None
                and down_proj is not None
                and hasattr(gate_proj, "lora_A")
                and hasattr(up_proj, "lora_A")
                and hasattr(down_proj, "lora_A")
                and _no_bias(gate_proj)
                and _no_bias(up_proj)
                and _no_bias(down_proj)
                and _no_lora_mag(gate_proj)
                and _no_lora_mag(up_proj)
                and _no_lora_mag(down_proj)
            ):
                mlp.forward = types.MethodType(apply_lora_mlp_swiglu, mlp)
                n_mlp += 1

    return n_qkv, n_o, n_mlp


def _patch_llada_peft(
    model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]
) -> tuple[int, int, int]:
    """Patch LLaDA model with Triton LoRA kernels.

    LLaDA uses a non-standard layer hierarchy:
      ``model.base_model.model.transformer.blocks`` (list of ``LLaDABlock``).

    ``LLaDALlamaBlock`` has ``q_proj/k_proj/v_proj/attn_out/ff_proj/up_proj``.
    Other block types (``LLaDASequentialBlock``) use ``att_proj`` (fused QKV)
    and are not supported by the split QKV kernel — they are skipped with a
    warning.
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

    # LLaDAModelLM wraps LLaDAModel in self.model, so the path differs:
    # PeftModel → base_model → model (LLaDAModelLM) → model (LLaDAModel) → transformer
    inner = model.base_model.model
    if hasattr(inner, "model") and hasattr(inner.model, "transformer"):
        transformer = inner.model.transformer
    elif hasattr(inner, "transformer"):
        transformer = inner.transformer
    else:
        _warn_once(
            "FastDiffusionModel (LLaDA): could not locate transformer — "
            "cannot patch LoRA kernels. Is this a supported LLaDA checkpoint?"
        )
        return n_qkv, n_o, n_mlp

    if not hasattr(transformer, "blocks"):
        _warn_once(
            "FastDiffusionModel (LLaDA): transformer.blocks not found — "
            "cannot patch LoRA kernels. Is this a supported LLaDA checkpoint?"
        )
        return n_qkv, n_o, n_mlp

    blocks = transformer.blocks

    if lora_dropout == 0 and bias == "none":
        _require_fast_lora()

    for block in blocks:
        if not isinstance(block, LLaDALlamaBlock):
            _warn_once(
                f"FastDiffusionModel (LLaDA): skipping block type {type(block).__name__} "
                "(only LLaDALlamaBlock is supported for Triton LoRA patching)."
            )
            continue

        # Inject Triton RoPE fast forward unconditionally (CUDA already checked above).
        rotary_emb = getattr(block, "rotary_emb", None)
        if rotary_emb is not None and not getattr(
            rotary_emb, "_fast_rope_patched", False
        ):
            import types

            rotary_emb.forward = types.MethodType(
                _make_llada_fast_rope_forward(type(rotary_emb).forward), rotary_emb
            )
            rotary_emb._fast_rope_patched = True

        if lora_dropout != 0 or bias != "none":
            continue

        # LLaDALlamaBlock: q_proj / k_proj / v_proj (bias depends on config)
        q_proj = getattr(block, "q_proj", None)
        k_proj = getattr(block, "k_proj", None)
        v_proj = getattr(block, "v_proj", None)
        if (
            q_proj is not None
            and k_proj is not None
            and v_proj is not None
            and hasattr(q_proj, "lora_A")
            and hasattr(k_proj, "lora_A")
            and hasattr(v_proj, "lora_A")
            and _no_bias(q_proj)
            and _no_bias(k_proj)
            and _no_bias(v_proj)
            and _no_lora_mag(q_proj)
            and _no_lora_mag(k_proj)
            and _no_lora_mag(v_proj)
        ):
            block.apply_qkv = apply_lora_qkv
            n_qkv += 1
        else:
            _warn_once(
                "FastDiffusionModel (LLaDA): cannot patch QKV with Triton kernel "
                "(LoRA not enabled or bias present — config.include_qkv_bias=True)."
            )

        # attn_out (o_proj equivalent)
        attn_out = getattr(block, "attn_out", None)
        if (
            attn_out is not None
            and hasattr(attn_out, "lora_A")
            and _no_bias(attn_out)
            and _no_lora_mag(attn_out)
        ):
            block.apply_o = apply_lora_o
            n_o += 1
        else:
            _warn_once(
                "FastDiffusionModel (LLaDA): cannot patch attn_out with Triton kernel."
            )

        # ff_proj / up_proj / ff_out — gated MLP (gate/up/down).
        # apply_lora_mlp_swiglu reads self.gate_proj / self.up_proj / self.down_proj
        # and uses the SiLU-gated SwiGLU Triton kernel.
        # Only patch when activation_type is SiLU (output_multiplier==1); with SwiGLU
        # (output_multiplier==0.5) ff_proj output is halved by chunk(2) while up_proj
        # stays full-width, producing a shape mismatch in the Triton kernel.
        block_act = getattr(block, "act", None)
        act_is_silu = block_act is not None and isinstance(block_act, torch.nn.SiLU)
        ff_proj = getattr(block, "ff_proj", None)
        up_proj = getattr(block, "up_proj", None)
        ff_out = getattr(block, "ff_out", None)
        if not act_is_silu:
            _warn_once(
                f"FastDiffusionModel (LLaDA): skipping Triton MLP patch for "
                f"{type(block_act).__name__} activation — only SiLU is supported. "
                "MLP LoRA will use PEFT default path."
            )
        elif (
            ff_proj is not None
            and up_proj is not None
            and ff_out is not None
            and hasattr(ff_proj, "lora_A")
            and hasattr(up_proj, "lora_A")
            and hasattr(ff_out, "lora_A")
            and _no_bias(ff_proj)
            and _no_bias(up_proj)
            and _no_bias(ff_out)
            and _no_lora_mag(ff_proj)
            and _no_lora_mag(up_proj)
            and _no_lora_mag(ff_out)
        ):
            # Set gate_proj/down_proj aliases for apply_lora_mlp_swiglu compatibility.
            block.gate_proj = ff_proj
            block.down_proj = ff_out
            block.apply_mlp = apply_lora_mlp_swiglu
            n_mlp += 1
        else:
            _warn_once(
                "FastDiffusionModel (LLaDA): cannot patch MLP with Triton kernel "
                "(LoRA not enabled, bias present, or magnitude scaling active)."
            )

    return n_qkv, n_o, n_mlp


def _patch_modernbert_peft(
    model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]
) -> tuple[int, int, int]:
    """Patch ModernBERT diffusion model with bidirectional fast attention and Triton O-projection.

    ModernBERT uses fused ``Wqkv`` and ``Wo`` (attention) and ``Wi`` / ``Wo`` (MLP).
    Unlike the LLaMA/Qwen2 path, QKV and MLP Triton kernels are **not** applied
    in this initial implementation because the fused projection shapes differ from
    what ``apply_lora_qkv`` / ``apply_lora_mlp_swiglu`` expect.

    What IS patched:
    - ``layer.attn.forward`` → ``ModernBertAttention_fast_forward`` (CUDA only)
    - ``layer.attn.Wo``     → ``apply_lora_o`` when conditions allow (CUDA, no dropout, no bias)

    Layer hierarchy:
        PeftModel → base_model → model (DiffusionModernBertForMaskedLM)

    Returns (n_qkv_patched=0, n_o_patched, n_mlp_patched=0).
    """
    n_o = 0

    first_param = next(iter(model.parameters()), None)
    on_cuda = first_param is not None and first_param.device.type == "cuda"

    # A2DModernBertForMaskedLM wraps A2DModernBertModel in self.model
    # Path: PeftModel → base_model → model (LM) → model (encoder) → layers
    try:
        layers = model.base_model.model.model.layers
    except AttributeError:
        _warn_once(
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
        _require_fast_lora()

    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is None:
            continue

        # Always inject bidirectional fast-forward on CUDA
        attn.forward = types.MethodType(ModernBertAttention_fast_forward, attn)

        if lora_dropout != 0 or bias != "none":
            continue

        # Wo output projection — apply Triton apply_lora_o when conditions met
        wo = getattr(attn, "Wo", None)
        if (
            wo is not None
            and hasattr(wo, "lora_A")
            and _no_bias(wo)
            and _no_lora_mag(wo)
        ):
            # Redirect apply_wo to Triton apply_lora_o.
            # apply_lora_o reads self.o_proj — we alias Wo as o_proj for compatibility.
            attn.o_proj = attn.Wo
            attn.apply_wo = apply_lora_o
            n_o += 1
        elif wo is not None and not hasattr(wo, "lora_A"):
            _warn_once(
                "FastDiffusionModel (ModernBERT): Wo has no LoRA adapter — "
                "is 'Wo' in target_modules?"
            )

    return 0, n_o, 0


def _no_bias(proj: Any) -> bool:
    return getattr(proj, "base_layer", proj).bias is None


def _no_lora_mag(proj: Any) -> bool:
    return len(getattr(proj, "lora_magnitude_vector", []) or []) == 0


def _load_model_with_optional_4bit_fallback(
    loader: Any,
    model_name: str,
    load_kwargs: dict[str, Any],
) -> Any:
    try:
        return loader.from_pretrained(model_name, **load_kwargs)
    except torch.cuda.OutOfMemoryError:
        # OOM is not a 4-bit-specific failure — a full-precision retry needs
        # MORE memory and is doomed. Surface the real error to the user.
        raise
    except Exception as exc:  # noqa: BLE001
        if "quantization_config" not in load_kwargs:
            raise

        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs.pop("quantization_config", None)
        fallback_kwargs.pop("device_map", None)
        _warn_once(
            "FastDiffusionModel: 4-bit loading failed "
            f"({type(exc).__name__}: {exc}) — retrying with full-precision loading."
        )
        return loader.from_pretrained(model_name, **fallback_kwargs)


def _import_bitsandbytes() -> Any:
    """Import hook for bitsandbytes (separate function for testability)."""
    import bitsandbytes as bnb

    return bnb


def _find_quantized_linear_modules(model: Any) -> list[tuple[str, Any]]:
    """Return ``(name, module)`` for modules holding bnb-quantized weights.

    Detection is by the ``weight.quant_state`` attribute (present on
    ``bitsandbytes`` ``Params4bit``), not by isinstance, so it works without
    importing bitsandbytes and with test stubs.
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if getattr(getattr(module, "weight", None), "quant_state", None) is not None
    ]


def _dequantize_merged_model_(model: Any) -> Any:
    """Dequantize bnb 4-bit Linear modules in *model* to 16-bit, in place.

    ``merge_and_unload()`` on a 4-bit-loaded PEFT model returns a base model
    whose Linear layers are still ``bnb.nn.Linear4bit`` — saving that as a
    "merged 16-bit" artifact would silently ship nf4 weights plus a
    ``quantization_config``. This mirrors what unsloth's own merged save does
    per layer (``fast_dequantize(W, quant_state)`` in ``unsloth/save.py``),
    using bitsandbytes' public ``functional.dequantize_4bit``.

    Raises:
        RuntimeError: when the model holds quantized weights but they cannot
            be dequantized (bitsandbytes missing or dequantization failed) —
            the caller must NOT save mislabeled 4-bit weights. Re-export from
            a 16-bit load (e.g. ``load_in_4bit=False``) instead.
    """
    quantized = _find_quantized_linear_modules(model)
    if not quantized:
        return model

    error_hint = (
        "cannot save a truthful merged 16-bit artifact from 4-bit weights. "
        "Re-load the checkpoint with load_in_4bit=False (CLI: "
        "`unturtle export --no-load-in-4bit`) and export again."
    )
    try:
        bnb = _import_bitsandbytes()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"FastDiffusionModel: model has {len(quantized)} bnb-quantized "
            f"module(s) but bitsandbytes is unavailable ({exc}); {error_hint}"
        ) from exc

    named_modules = dict(model.named_modules())
    for name, module in quantized:
        weight = module.weight
        try:
            dequant = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"FastDiffusionModel: failed to dequantize module {name!r} "
                f"({type(exc).__name__}: {exc}); {error_hint}"
            ) from exc
        target_dtype = getattr(weight.quant_state, "dtype", dequant.dtype)
        new_linear = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=dequant.device,
            dtype=target_dtype,
        )
        new_linear.weight = torch.nn.Parameter(
            dequant.to(target_dtype), requires_grad=False
        )
        if module.bias is not None:
            new_linear.bias = torch.nn.Parameter(
                module.bias.data.to(target_dtype), requires_grad=False
            )
        parent_name, _, child_name = name.rpartition(".")
        parent = named_modules[parent_name] if parent_name else model
        setattr(parent, child_name, new_linear)

    # The artifact is now 16-bit — drop the stale quantization metadata so the
    # saved config does not claim 4-bit loading.
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "quantization_config"):
        try:
            del config.quantization_config
        except AttributeError:
            config.quantization_config = None
    for attr in ("is_loaded_in_4bit", "is_quantized"):
        if getattr(model, attr, False):
            setattr(model, attr, False)
    hf_quantizer = getattr(model, "hf_quantizer", None)
    if hf_quantizer is not None:
        model.hf_quantizer = None

    _logger.info(
        "FastDiffusionModel: dequantized %d bnb 4-bit module(s) to 16-bit "
        "for the merged save.",
        len(quantized),
    )
    return model


#: model_type → callable returning the wrapper class to swap in after a
#: FastModel load (FastModel loads upstream classes; wrappers add only a
#: ``generate`` shim, so ``__class__`` swap is safe). Filled by backbone modules.
_POST_LOAD_CLASS_SWAPS: dict[str, Any] = {}


def _resolve_diffusion_gemma_wrapper() -> Any:
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    return UnturtleDiffusionGemmaForBlockDiffusion


_POST_LOAD_CLASS_SWAPS["diffusion_gemma"] = _resolve_diffusion_gemma_wrapper


def _apply_post_load_class_swap(model: Any) -> None:
    """Swap model's class to the registered wrapper, if any.

    When ``unsloth.FastModel`` loads a model it returns the upstream
    ``transformers`` class.  Backbone modules can register a resolver in
    :data:`_POST_LOAD_CLASS_SWAPS` so that the thin Unturtle wrapper class is
    installed via ``__class__`` assignment after loading.

    After the class swap (or when the model is already the wrapper class),
    any instance-level ``generate`` attribute is removed.  unsloth FastModel
    installs ``unsloth_base_fast_generate`` as an instance attribute (saving
    the original as ``self._old_generate``), which would shadow the wrapper
    class's unified ``generate`` shim AND forces ``cache_implementation=
    "static"``, crashing DiffusionGemma's flex-attention canvas block loop.
    Dropping the instance attribute lets the class-level shim win.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    resolver = _POST_LOAD_CLASS_SWAPS.get(model_type)
    if resolver is None:
        return
    wrapper_cls = resolver()
    if not isinstance(model, wrapper_cls):
        model.__class__ = wrapper_cls
    # unsloth FastModel installs an instance-level fast-generate wrapper
    # (saving the original as `_old_generate`). It would shadow the wrapper
    # class's unified `generate` shim AND forces cache_implementation="static",
    # which breaks DiffusionGemma's canvas flex-attention block mask. Drop the
    # instance attribute so the class-level shim (verbatim upstream delegation)
    # wins. This runs whether the class was just swapped or was already the
    # wrapper (covers re-entrant / double-swap scenarios).
    model.__dict__.pop("generate", None)


def _native_model_classes() -> dict[str, Any]:
    """Build the ``model_type`` → unturtle native model class map.

    These classes are the from-scratch / wrapper implementations Unturtle owns
    (LLaDA, Dream, Tiny-A2D Llama/Qwen2/Qwen3). Loading through them bypasses any
    ``trust_remote_code`` Hub modeling code, so fixes in the unturtle classes
    (e.g. ``_tied_weights_keys``) always take effect.
    """
    import unturtle.models  # noqa: F401 — registers A2D/LLaDA/Dream AutoConfig entries

    classes: dict[str, Any] = {}
    try:
        from unturtle.models.backbones.llada import LLaDAModelLM

        classes["llada"] = LLaDAModelLM
    except ImportError:
        pass
    try:
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        classes["mdlm-dit"] = MDLMDiTForMaskedDiffusionLM
    except ImportError:
        pass
    try:
        from unturtle.models.backbones.dream import DreamModel

        classes["dream"] = DreamModel
        # DreamConfig.model_type is "Dream" (capital D) — match Hub configs.
        classes["Dream"] = DreamModel
    except ImportError:
        pass
    try:
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
            TinyA2DLlamaLMHeadModel,
        )

        classes["tiny-a2d-llama"] = TinyA2DLlamaLMHeadModel
    except ImportError:
        pass
    try:
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen2 import (
            TinyA2DQwen2LMHeadModel,
        )

        classes["tiny-a2d-qwen2"] = TinyA2DQwen2LMHeadModel
    except ImportError:
        pass
    try:
        from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3 import (
            TinyA2DQwen3LMHeadModel,
        )

        classes["tiny-a2d-qwen3"] = TinyA2DQwen3LMHeadModel
    except ImportError:
        pass
    return classes


def _load_native(
    model_name: str, load_kwargs: dict, trust_remote_code: bool
) -> Any | None:
    """Load via an Unturtle native class when ``model_name``'s ``model_type`` is one.

    Peeks at the config (no weights) and, if the ``model_type`` maps to a native
    Unturtle class, loads through it directly — preserving the ``trust_remote_code``
    bypass contract. Returns ``None`` when the model is *not* a native dLLM, so the
    caller can delegate the load elsewhere. CUDA OOM propagates.
    """
    native_classes = _native_model_classes()

    try:
        peek_kwargs: dict[str, Any] = {}
        if "token" in load_kwargs:
            peek_kwargs["token"] = load_kwargs["token"]
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, **peek_kwargs
        )
        model_type = getattr(config, "model_type", "")
        if model_type in native_classes:
            native_cls = native_classes[model_type]
            _logger.debug(
                "FastDiffusionModel: using native unturtle class %s for model_type=%r",
                native_cls.__name__,
                model_type,
            )
            return _load_model_with_optional_4bit_fallback(
                native_cls, model_name, load_kwargs
            )
    except torch.cuda.OutOfMemoryError:
        raise  # OOM should propagate, not fall through to slower loaders
    except Exception as exc:  # noqa: BLE001
        _logger.debug("FastDiffusionModel: native class lookup failed: %s", exc)

    return None


def _load_via_automodel(model_name: str, load_kwargs: dict) -> Any:
    """Load a non-native (HF-registered) model_type via the AutoModel fallback chain.

    This is the offline / unsloth-unavailable fallback path: loading/quantization is
    handled by ``transformers``' ``Auto*`` loaders.  The diffusion patch is applied
    afterwards by :func:`_patch_for_diffusion`, so the resulting model behaves as a
    bidirectional dLLM regardless of which path produced it.  Raises if every loader
    fails.

    The primary non-native path is :func:`_load_via_fastmodel` (unsloth FastModel);
    this function is only reached when that path is unavailable or raises.
    """
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
    )

    loaders = [
        ("AutoModel", AutoModel),
        ("AutoModelForMaskedLM", AutoModelForMaskedLM),
        ("AutoModelForCausalLM", AutoModelForCausalLM),
    ]
    last_exc: Exception | None = None
    for name, loader_cls in loaders:
        try:
            return _load_model_with_optional_4bit_fallback(
                loader_cls, model_name, load_kwargs
            )
        except Exception as exc:  # noqa: BLE001
            _logger.debug("FastDiffusionModel: %s failed: %s", name, exc)
            last_exc = exc

    raise RuntimeError(
        f"FastDiffusionModel: could not load {model_name!r} via any AutoModel variant. "
        f"Pass model_class= explicitly.\nLast error: {last_exc}"
    ) from last_exc


def _import_fastmodel() -> Any:
    """Import hook for unsloth FastModel (separate function for testability)."""
    from unsloth import FastModel

    return FastModel


def _load_via_fastmodel(
    model_name: str, load_kwargs: dict, *, load_in_4bit: bool
) -> tuple[Any, Any] | None:
    """Load a non-native model_type via ``unsloth.FastModel.from_pretrained``.

    Returns ``(model, tokenizer)`` on success, or ``None`` when unsloth is
    unavailable or the load fails — the caller then falls back to the Auto*
    chain (offline / local-stub paths keep working).

    FastModel owns quantization on this path; ``quantization_config`` is not
    forwarded.  ``load_in_4bit`` is threaded through explicitly from the
    caller so the user's original intent is preserved.
    """
    try:
        fast_model_cls = _import_fastmodel()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("FastDiffusionModel: unsloth FastModel unavailable: %s", exc)
        return None
    try:
        # Forward all user kwargs FastModel.from_pretrained can accept
        # (revision, cache_dir, subfolder, attn_implementation, …) instead of a
        # tiny allowlist. Keys FastModel's signature cannot take are dropped
        # with a warning so nothing disappears silently.
        import inspect

        try:
            sig = inspect.signature(fast_model_cls.from_pretrained)
            accepted = set(sig.parameters)
            accepts_var_kw = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepted = set()
            accepts_var_kw = True

        fm_kwargs: dict[str, Any] = {}
        dropped: list[str] = []
        for key, value in load_kwargs.items():
            if key == "torch_dtype":
                # FastModel uses `dtype=` not `torch_dtype=`
                fm_kwargs["dtype"] = value
            elif key == "quantization_config":
                # FastModel owns quantization on this path (intentional skip;
                # the caller's intent travels via load_in_4bit below).
                continue
            elif accepts_var_kw or key in accepted:
                fm_kwargs[key] = value
            else:
                dropped.append(key)
        if dropped:
            _warn_once(
                "FastDiffusionModel: dropping kwargs not accepted by "
                f"unsloth FastModel.from_pretrained: {sorted(dropped)}"
            )
        # FastModel owns quantization — pass load_in_4bit explicitly from the caller.
        fm_kwargs["load_in_4bit"] = load_in_4bit
        model, tokenizer = fast_model_cls.from_pretrained(model_name, **fm_kwargs)
        return model, tokenizer
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001
        _logger.debug("FastDiffusionModel: FastModel load failed: %s", exc)
        _warn_once(
            "FastDiffusionModel: unsloth FastModel failed to load "
            f"{model_name!r} ({exc}) — falling back to the transformers Auto* "
            "chain (no unsloth quantization/patches on this load)."
        )
        return None


def _load_model_auto(
    model_name: str,
    load_kwargs: dict,
    trust_remote_code: bool,
    *,
    load_in_4bit: bool = False,
) -> tuple[Any, Any | None]:
    """Resolve and load a model: native dLLM class first, FastModel, then HF Auto*.

    Returns ``(model, tokenizer_or_none)``.  The tokenizer is non-``None`` only
    when ``unsloth.FastModel`` was used (it returns a tokenizer alongside the
    model).  On the native or Auto* paths the tokenizer is ``None`` and the
    caller must load it separately.

    Native Unturtle backbones (llada / Dream / tiny-a2d-*) are loaded through their
    own classes (``_load_native``), bypassing ``trust_remote_code`` Hub code.  Any
    other ``model_type`` tries ``unsloth.FastModel.from_pretrained`` first (so
    HF-registered dLLM backbones such as DiffusionGemma get unsloth's loading /
    quantization / patch chain), then falls back to the ``transformers`` ``Auto*``
    loaders when unsloth is unavailable or the load fails.  Kept as a single
    module-level entry point for callers and tests that patch it by name.
    """
    model = _load_native(model_name, load_kwargs, trust_remote_code)
    if model is not None:
        return model, None

    result = _load_via_fastmodel(model_name, load_kwargs, load_in_4bit=load_in_4bit)
    if result is not None:
        return result  # (model, tokenizer)

    return _load_via_automodel(model_name, load_kwargs), None


def _load_tokenizer(
    model_name: str, trust_remote_code: bool, token: Optional[str]
) -> Any:
    """Load tokenizer; warn instead of silently returning None."""
    try:
        tok_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if token is not None:
            tok_kwargs["token"] = token
        return AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    except Exception as exc:  # noqa: BLE001
        import warnings

        warnings.warn(
            f"FastDiffusionModel: tokenizer not found for {model_name!r}: {exc}\n"
            "Pass a tokenizer manually or verify the model path.",
            stacklevel=3,
        )
        return None


def _extend_rope_if_possible(model: Any, max_seq_length: int) -> None:
    """Extend RoPE embeddings to cover ``max_seq_length`` if the model supports it.

    Iterates through all modules looking for a ``rotary_emb`` or
    ``rotary_embedding`` attribute that exposes ``extend_rope_embedding``.
    This mirrors unsloth's ``extend_model_function``.
    """
    for module in model.modules():
        for rope_attr in ("rotary_emb", "rotary_embedding"):
            rope = getattr(module, rope_attr, None)
            if rope is None:
                continue
            if hasattr(rope, "extend_rope_embedding"):
                try:
                    rope.extend_rope_embedding(max_seq_length)
                    _logger.debug(
                        "FastDiffusionModel: extended RoPE to %d via %s",
                        max_seq_length,
                        type(rope).__name__,
                    )
                except Exception as exc:  # noqa: BLE001
                    _logger.debug("FastDiffusionModel: RoPE extension failed: %s", exc)


def _propagate_max_seq_length(model: Any, max_seq_length: int) -> None:
    """Set max_seq_length on every nested model attribute (mirrors unsloth)."""
    internal = model
    while hasattr(internal, "model"):
        internal.max_seq_length = max_seq_length
        internal = internal.model
    internal.max_seq_length = max_seq_length
    for module in model.modules():
        module.max_seq_length = max_seq_length


def _patch_for_diffusion(model: Any, max_seq_length: int) -> Any:
    """Apply the Unturtle diffusion patch shared by every load path.

    Both the native loader and the unsloth/HF delegation path funnel through here,
    as does the PEFT-adapter branch. It installs the apply_qkv/apply_o stubs (so the
    bidirectional fast-forward and LoRA fast paths can attach), records
    ``max_seq_length`` across the nested model, and extends RoPE when supported.
    Returns the (mutated) model for call-site clarity.
    """
    _install_apply_stubs(model)
    model.max_seq_length = max_seq_length
    _propagate_max_seq_length(model, max_seq_length)
    _extend_rope_if_possible(model, max_seq_length)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FastDiffusionModel:
    """Drop-in loader and PEFT helper for diffusion language models."""

    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        model_class: Any = None,
        trust_remote_code: bool = True,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Load a dLLM model (optionally 4-bit quantised).

        Does NOT call unsloth's ``pre_patch()`` — that would inject causal
        fast-forward functions, which is wrong for bidirectional dLLMs.

        Args:
            model_name:         HuggingFace model id or local path.
            max_seq_length:     Maximum sequence length.
            dtype:              Torch dtype.  Defaults to bfloat16 on CUDA GPUs
                                that support it, float16 on other CUDA GPUs, and
                                float32 on CPU.
            load_in_4bit:       Enable 4-bit NF4 quantisation via bitsandbytes.
                                Silently disabled when running on CPU or when
                                bitsandbytes is not installed.
            model_class:        Explicit model class override (e.g.
                                ``TinyA2DLlamaLMHeadModel``).  When *None* the class
                                is resolved via a fallback chain:
                                ``AutoModel`` → ``AutoModelForMaskedLM`` →
                                ``AutoModelForCausalLM``.
            trust_remote_code:  Passed to ``from_pretrained``.
            token:              HuggingFace Hub auth token.
            **kwargs:           Forwarded to ``from_pretrained``.

        Returns:
            ``(model, tokenizer)`` tuple.  ``tokenizer`` may be ``None`` with
            a warning if no tokenizer files are found.
        """
        # --- PEFT adapter checkpoint detection (API Gap G3) ---
        # If the path is a local directory that contains adapter_config.json but
        # not full model weights, load the base model first then wrap with PEFT.
        from pathlib import Path as _Path

        _local = _Path(model_name) if not model_name.startswith("http") else None
        _has_full_weights = False
        if _local is not None and _local.is_dir():
            try:
                _has_full_weights = (
                    any(
                        (_local / filename).exists()
                        for filename in (
                            "model.safetensors",
                            "pytorch_model.bin",
                            "pytorch_model.safetensors",
                            # Sharded checkpoints use an index file instead of a single weight file
                            "model.safetensors.index.json",
                            "pytorch_model.bin.index.json",
                        )
                    )
                    or any(_local.glob("model-*-of-*.safetensors"))
                    or any(
                        _local.glob("pytorch_model-*-of-*.bin")
                    )  # legacy sharded .bin
                )
            except OSError as _e:
                _logger.warning(
                    "FastDiffusionModel: could not scan %r for weight files (%s). "
                    "Assuming full weights present to avoid incorrect adapter path.",
                    str(_local),
                    _e,
                )
                _has_full_weights = True
        if (
            _local is not None
            and _local.is_dir()
            and (_local / "adapter_config.json").exists()
            and not _has_full_weights
        ):
            import json as _json

            from peft import PeftModel as _PeftModel

            try:
                _adapter_cfg = _json.loads(
                    (_local / "adapter_config.json").read_text(encoding="utf-8")
                )
            except _json.JSONDecodeError as _e:
                raise ValueError(
                    f"FastDiffusionModel: adapter_config.json at {_local} is not valid JSON: {_e}"
                ) from _e
            except OSError as _e:
                raise RuntimeError(
                    f"FastDiffusionModel: could not read adapter_config.json at {_local}: {_e}"
                ) from _e
            _base_model_id = _adapter_cfg.get("base_model_name_or_path", "")
            if not _base_model_id:
                raise ValueError(
                    f"adapter_config.json at {_local} has no base_model_name_or_path."
                )
            _logger.info(
                "FastDiffusionModel: detected PEFT adapter checkpoint at %r; "
                "loading base model %r first.",
                str(_local),
                _base_model_id,
            )
            base_model, tokenizer = FastDiffusionModel.from_pretrained(
                model_name=_base_model_id,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                model_class=model_class,
                trust_remote_code=trust_remote_code,
                token=token,
                **kwargs,
            )
            try:
                model = _PeftModel.from_pretrained(base_model, str(_local))
            except Exception as _e:
                raise RuntimeError(
                    f"FastDiffusionModel: failed to load PEFT adapter from {_local!r} "
                    f"onto base model {_base_model_id!r}: {_e}"
                ) from _e
            _patch_for_diffusion(model, max_seq_length)
            _logger.info(
                "FastDiffusionModel: PEFT adapter loaded from %r.", str(_local)
            )
            return model, tokenizer

        # --- dtype auto-detection ---
        if dtype is None:
            if torch.cuda.is_available():
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
            else:
                dtype = torch.float32

        is_on_cpu = not torch.cuda.is_available()

        load_kwargs: dict[str, Any] = dict(
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if token is not None:
            load_kwargs["token"] = token

        # --- 4-bit quantisation (CUDA only) ---
        if load_in_4bit and not is_on_cpu:
            if importlib.util.find_spec("bitsandbytes") is None:
                _warn_once(
                    "bitsandbytes not installed — falling back to full-precision loading."
                )
            else:
                try:
                    from transformers import BitsAndBytesConfig

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                    load_kwargs["quantization_config"] = bnb_config
                    # device_map="auto" is required for multi-GPU or when GPU 0 is partially occupied.
                    if "device_map" not in load_kwargs:
                        load_kwargs["device_map"] = "auto"
                except ImportError:
                    _warn_once(
                        "bitsandbytes not installed — falling back to full-precision loading."
                    )
        elif load_in_4bit and is_on_cpu:
            _warn_once(
                "FastDiffusionModel: load_in_4bit=True requires CUDA — "
                "falling back to full-precision loading on CPU."
            )

        # --- Resolve model class (explicit override → native dLLM → FastModel → Auto*) ---
        fm_tokenizer: Any | None = None
        if model_class is None:
            model, fm_tokenizer = _load_model_auto(
                model_name,
                load_kwargs,
                trust_remote_code,
                load_in_4bit=load_in_4bit and not is_on_cpu,
            )
        else:
            model = model_class.from_pretrained(model_name, **load_kwargs)

        # --- Post-load class swap (e.g. DiffusionGemma wrapper) ---
        _apply_post_load_class_swap(model)

        # --- Diffusion patch (shared across load paths) ---
        _patch_for_diffusion(model, max_seq_length)

        # --- Tokenizer (prefer FastModel's tokenizer; fall back to separate load) ---
        tokenizer = (
            fm_tokenizer
            if fm_tokenizer is not None
            else _load_tokenizer(model_name, trust_remote_code, token)
        )

        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model: Any,
        r: int = 16,
        target_modules: Optional[list[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: Literal["none", "all", "lora_only"] = "none",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int = 3407,
        **kwargs: Any,
    ) -> Any:
        """Apply LoRA and patch with unsloth's Triton kernels.

        Uses ``TaskType.FEATURE_EXTRACTION`` (not CAUSAL_LM) to avoid
        ``PeftModelForCausalLM`` type guards in unsloth's ``patch_peft_model``.

        Args:
            model:                      Base model (output of ``from_pretrained``).
            r:                          LoRA rank.
            target_modules:             Which linear layers to target.
            lora_alpha:                 LoRA scaling factor.
            lora_dropout:               Dropout in LoRA adapters (0 = disabled).
            bias:                       LoRA bias mode (``"none"``, ``"all"``,
                                        ``"lora_only"``).
            use_gradient_checkpointing: ``"unsloth"`` for unsloth-style GC,
                                        ``True`` for standard, ``False`` to disable.
            random_state:               Seed passed to PEFT.
            **kwargs:                   Forwarded to ``LoraConfig``.

        Returns:
            PEFT model with Triton LoRA kernels patched in.
        """
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            **kwargs,
        )

        # Install apply_qkv / apply_o stubs before PEFT wrapping so that
        # fast-forward functions can dispatch even when the model was not
        # loaded via from_pretrained (e.g. tests using random-weight models).
        _install_apply_stubs(model)

        quantization_method = getattr(model, "quantization_method", None)
        is_quantized_model = any(
            getattr(model, attr, False)
            for attr in ("is_loaded_in_4bit", "is_loaded_in_8bit", "hqq_quantized")
        ) or quantization_method in {"gptq", "aqlm", "eetq", "torchao", "hqq"}

        if is_quantized_model:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_reentrant=True,
            )
        elif bool(use_gradient_checkpointing):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):
                input_embeddings = model.get_input_embeddings()
                if input_embeddings is not None:
                    input_embeddings.register_forward_hook(
                        lambda module, inputs, output: output.requires_grad_(True)
                    )
            _apply_gradient_checkpointing_mode(model, use_gradient_checkpointing)

        model = get_peft_model(model, lora_config)
        model._unturtle_gradient_checkpointing_mode = use_gradient_checkpointing

        FastDiffusionModel.patch_peft_model(model, lora_dropout=lora_dropout, bias=bias)
        patch_saving_functions(model)

        return model

    @staticmethod
    def patch_peft_model(
        model: Any,
        lora_dropout: float = 0,
        bias: Literal["none", "all", "lora_only"] = "none",
    ) -> None:
        """Inject Triton LoRA kernels and bidirectional attention into a PEFT model.

        Safe to call again after adding new adapters.

        Args:
            model:          PEFT-wrapped dLLM model.
            lora_dropout:   Must match the LoRA config used when wrapping.
            bias:           Must match the LoRA config used when wrapping.
        """
        model_type = model.config.model_type

        if model_type in _TINY_A2D_MODEL_TYPES:
            n_qkv, n_o, n_mlp = _patch_a2d_peft(model, lora_dropout, bias)
            n_layers = len(model.base_model.model.model.layers)
            _warn_once(
                f"FastDiffusionModel patched {n_layers} layers with "
                f"{n_qkv} QKV layers, {n_o} O layers and {n_mlp} MLP layers "
                f"(bidirectional, causal=False)."
            )
        elif model_type in _DREAM_MODEL_TYPES:
            n_qkv, n_o, n_mlp = _patch_dream_peft(model, lora_dropout, bias)
            n_layers = len(model.base_model.model.model.layers)
            _warn_once(
                f"FastDiffusionModel (Dream) patched {n_layers} layers with "
                f"{n_qkv} QKV layers (bias kernel), {n_o} O layers and {n_mlp} MLP layers."
            )
        elif model_type in _LLADA_MODEL_TYPES:
            n_qkv, n_o, n_mlp = _patch_llada_peft(model, lora_dropout, bias)
            inner = model.base_model.model
            _llada_transformer = (
                inner.model.transformer
                if hasattr(inner, "model") and hasattr(inner.model, "transformer")
                else getattr(inner, "transformer", None)
            )
            n_blocks = (
                len(_llada_transformer.blocks) if _llada_transformer is not None else 0
            )
            _warn_once(
                f"FastDiffusionModel (LLaDA) patched {n_blocks} blocks with "
                f"{n_qkv} QKV blocks and {n_o} O (attn_out) blocks."
            )
        elif model_type in _MODERNBERT_A2D_MODEL_TYPES:
            _n_qkv, n_o, _n_mlp = _patch_modernbert_peft(model, lora_dropout, bias)
            n_layers = len(model.base_model.model.model.layers)
            _warn_once(
                f"FastDiffusionModel (ModernBERT) patched {n_layers} layers with "
                f"{n_o} Wo (output proj) layers. "
                "Wqkv/MLP Triton kernels not yet supported for ModernBERT — "
                "see issue #59 Phase 2."
            )
        else:
            raise NotImplementedError(
                f"FastDiffusionModel does not yet support model_type={model_type!r}. "
                "Supported types: "
                + ", ".join(
                    sorted(
                        _TINY_A2D_MODEL_TYPES
                        | _DREAM_MODEL_TYPES
                        | _LLADA_MODEL_TYPES
                        | _MODERNBERT_A2D_MODEL_TYPES
                    )
                )
            )

        # Propagate max_seq_length through the wrapped model hierarchy
        if hasattr(model, "max_seq_length"):
            _propagate_max_seq_length(model, model.max_seq_length)

    @staticmethod
    def for_inference(model: Any) -> Any:
        """Switch model to inference mode.

        Sets ``model.eval()`` and disables gradient checkpointing so that
        inference is as fast as possible.  Returns the model for convenience.

        Note: plain MDLM does not use KV cache; block-decode models manage their
        own dLLM cache internally.  Either way, no external cache-enabling step
        is needed here, unlike ``FastLanguageModel.for_inference`` for AR models.

        Usage::

            FastDiffusionModel.for_inference(model)
            with torch.no_grad():
                logits = model(**inputs).logits

        Args:
            model: A dLLM model (plain or PEFT-wrapped).

        Returns:
            The same model in eval mode.
        """
        model.eval()
        _apply_gradient_checkpointing_mode(model, False)
        return model

    @staticmethod
    def generate(
        model: Any,
        inputs: Any = None,
        *,
        algorithm: str = "auto",
        **kwargs: Any,
    ) -> Any:
        """Generate from a dLLM via its unified ``generate`` entry point.

        Thin facade that forwards to ``model.generate(inputs, algorithm=...)``.
        Algorithm resolution (auto/mdlm/block_decode/bd3lm) happens inside the
        model's ``generate``. Output is whatever ``model.generate`` returns.

        Args:
            model: A dLLM model exposing ``generate`` (e.g. from
                ``FastDiffusionModel.from_pretrained``).
            inputs: Prompt token IDs (``[B, L]``).
            algorithm: ``"auto"`` | ``"mdlm"`` | ``"block_decode"`` | ``"bd3lm"``.
            **kwargs: Forwarded to ``model.generate`` / the generation config.

        Returns:
            Whatever ``model.generate`` returns (token IDs or model output).
        """
        if not callable(getattr(model, "generate", None)):
            raise TypeError(
                f"{type(model).__name__} has no `generate` method; "
                "FastDiffusionModel.generate requires a dLLM model."
            )
        return model.generate(inputs, algorithm=algorithm, **kwargs)

    @staticmethod
    def for_training(model: Any, use_gradient_checkpointing: bool | str = True) -> Any:
        """Switch model back to training mode and re-enable gradient checkpointing.

        Args:
            model:                      A dLLM model (plain or PEFT-wrapped).
            use_gradient_checkpointing: ``True`` / ``"unsloth"`` to enable GC,
                                        ``False`` to leave it disabled.

        Returns:
            The same model in train mode.
        """
        model.train()
        _apply_gradient_checkpointing_mode(model, use_gradient_checkpointing)
        return model

    @staticmethod
    def save_pretrained_merged(
        model: Any,
        save_directory: str,
        tokenizer: Any = None,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None:
        """Merge LoRA adapters into the base weights and save.

        Calls PEFT's ``merge_and_unload()`` on a copy of the model, then saves
        the merged weights with ``save_pretrained``.  The original model
        (with adapters) is left unchanged.

        Args:
            model:              PEFT-wrapped dLLM model (output of ``get_peft_model``).
            save_directory:     Local directory path to save the merged model.
            tokenizer:          Optional tokenizer to save alongside the model.
            safe_serialization: Use safetensors format (recommended).
            **kwargs:           Forwarded to ``save_pretrained``.
        """
        import copy

        _logger.info("FastDiffusionModel: merging LoRA adapters into base weights …")
        merged = copy.deepcopy(model)
        # merge_and_unload returns the unwrapped base model with adapters merged.
        merged = merged.merge_and_unload()
        # On a 4-bit-loaded model the merged Linear layers are still bnb
        # Linear4bit — dequantize (or fail loudly) so the saved artifact is
        # genuinely 16-bit, never mislabeled nf4 weights.
        merged = _dequantize_merged_model_(merged)
        merged.save_pretrained(
            save_directory,
            safe_serialization=safe_serialization,
            **kwargs,
        )
        _logger.info("FastDiffusionModel: merged model saved to %r", save_directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)
            _logger.info("FastDiffusionModel: tokenizer saved to %r", save_directory)

    @staticmethod
    def push_to_hub_merged(
        model: Any,
        repo_id: str,
        tokenizer: Any = None,
        safe_serialization: bool = True,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Merge LoRA adapters and push merged weights to the HuggingFace Hub.

        Merges adapters via PEFT's ``merge_and_unload()`` then calls
        ``push_to_hub`` on the merged model.  The original model is unchanged.

        Args:
            model:              PEFT-wrapped dLLM model.
            repo_id:            HuggingFace Hub repository id (e.g. ``"user/my-model"``).
            tokenizer:          Optional tokenizer to push alongside the model.
            safe_serialization: Use safetensors format.
            token:              HuggingFace auth token.
            **kwargs:           Forwarded to ``push_to_hub``.
        """
        import copy

        _logger.info("FastDiffusionModel: merging LoRA adapters for Hub push …")
        merged = copy.deepcopy(model)
        merged = merged.merge_and_unload()
        # Same honesty guarantee as save_pretrained_merged: never push nf4
        # weights under a merged-16bit label.
        merged = _dequantize_merged_model_(merged)

        push_kwargs: dict[str, Any] = dict(
            safe_serialization=safe_serialization, **kwargs
        )
        if token is not None:
            push_kwargs["token"] = token

        merged.push_to_hub(repo_id, **push_kwargs)
        _logger.info("FastDiffusionModel: merged model pushed to %r", repo_id)
        if tokenizer is not None:
            tokenizer.push_to_hub(repo_id, **push_kwargs)

    @staticmethod
    def save_pretrained_gguf(
        model: Any,
        save_directory: str,
        tokenizer: Any = None,
        quantization_method: str = "q4_k_m",
        **kwargs: Any,
    ) -> None:
        """Convert and save model weights in GGUF format.

        Ensures ``unturtle.save.patch_saving_functions`` has been applied before
        delegating to the monkey-patched ``model.save_pretrained_gguf``.

        Args:
            model:                A dLLM model (plain or PEFT-wrapped).
            save_directory:       Local directory path for GGUF output.
            tokenizer:            Tokenizer to pass to the GGUF converter.
            quantization_method:  One of ``q4_k_m``, ``q5_k_m``, ``q8_0``, ``f16``.
            **kwargs:             Forwarded to the underlying GGUF save call.
        """
        from unturtle.save import patch_saving_functions

        patch_saving_functions(model)
        if not hasattr(model, "save_pretrained_gguf"):
            raise RuntimeError(
                "save_pretrained_gguf is not available. "
                "Ensure the llama.cpp GGUF toolchain is installed."
            )
        model.save_pretrained_gguf(
            save_directory,
            tokenizer,
            quantization_method=quantization_method,
            **kwargs,
        )
        _logger.info(
            "FastDiffusionModel: GGUF model saved to %r (quant=%s)",
            save_directory,
            quantization_method,
        )

    @staticmethod
    def save_lora_adapter(
        model: Any,
        save_directory: str,
        tokenizer: Any = None,
    ) -> None:
        """Save LoRA adapter weights only (no base model weights).

        Args:
            model:          A PEFT-wrapped dLLM model.
            save_directory: Local directory path for the adapter files.
            tokenizer:      Optional tokenizer to save alongside the adapter.

        Raises:
            ValueError: If ``model`` is not a PEFT model.
        """
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required for save_lora_adapter") from exc

        if not isinstance(model, PeftModel):
            raise ValueError(
                "save_lora_adapter requires a PEFT-wrapped model. "
                "The provided model appears to be a merged (non-PEFT) model."
            )
        model.save_pretrained(save_directory)
        _logger.info("FastDiffusionModel: LoRA adapter saved to %r", save_directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)
            _logger.info("FastDiffusionModel: tokenizer saved to %r", save_directory)

    @staticmethod
    @contextlib.contextmanager
    def inference_context(model: Any):
        """Context manager that temporarily switches to inference mode.

        Restores training mode on exit.

        Usage::

            with FastDiffusionModel.inference_context(model):
                logits = model(**inputs).logits

        Args:
            model: A dLLM model (plain or PEFT-wrapped).
        """
        was_training = model.training
        gc_mode = _get_gradient_checkpointing_mode(model)
        FastDiffusionModel.for_inference(model)
        try:
            with torch.no_grad():
                yield model
        finally:
            if was_training:
                FastDiffusionModel.for_training(
                    model, use_gradient_checkpointing=gc_mode
                )
            else:
                model.eval()
                _apply_gradient_checkpointing_mode(model, gc_mode)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _original_apply_qkv(self: Any, X: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return self.q_proj(X), self.k_proj(X), self.v_proj(X)


def _original_apply_o(self: Any, X: torch.Tensor) -> torch.Tensor:
    return self.o_proj(X)


def _install_apply_stubs(model: Any) -> None:
    """Set apply_qkv / apply_o stubs on all self_attn layers that lack them.

    unsloth's fast-forward dispatch protocol requires these attributes to exist
    even before PEFT is applied, so the fast-forward function can call
    ``self.apply_qkv(self, hidden_states)`` unconditionally.
    """
    for module in model.modules():
        if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
            if not hasattr(module, "apply_qkv"):
                module.apply_qkv = _original_apply_qkv
            if not hasattr(module, "apply_o"):
                module.apply_o = _original_apply_o


def _get_gradient_checkpointing_mode(model: Any) -> bool | str:
    """Return the current gradient-checkpointing mode tracked by unturtle.

    We explicitly track the requested mode because a temporary inference pass
    should be reversible: `True`, `False`, and `"unsloth"` need to round-trip.
    Falling back to module flags loses the distinction between `True` and
    `"unsloth"`.
    """
    if hasattr(model, "_unturtle_gradient_checkpointing_mode"):
        return model._unturtle_gradient_checkpointing_mode

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            return bool(module.gradient_checkpointing)
    return False


def _apply_gradient_checkpointing_mode(model: Any, mode: bool | str) -> None:
    """Apply and persist a gradient-checkpointing mode to all reachable modules."""
    model._unturtle_gradient_checkpointing_mode = mode

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = bool(mode)

    if bool(mode):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
