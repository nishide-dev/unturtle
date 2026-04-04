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
    from unturtle.models.a2d import A2DLlamaLMHeadModel

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

from unturtle.save import patch_saving_functions, prepare_model_for_kbit_training

from unturtle.models.a2d._fast_forward import (
    A2DAttention_fast_forward,
    ModernBertAttention_fast_forward,
    _install_modernbert_stubs,
)
from unturtle.models.dream.modeling_dream import DreamAttention_fast_forward

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
_A2D_MODEL_TYPES = frozenset(
    ["a2d-llama", "a2d-qwen2", "a2d-qwen3", "llama", "qwen2", "qwen3"]
)

# Dream model_type (note: Dream uses "Dream" with capital D)
_DREAM_MODEL_TYPES = frozenset(["dream", "Dream"])

# LLaDA model_type
_LLADA_MODEL_TYPES = frozenset(["llada"])

# ModernBERT A2D model_type
_MODERNBERT_A2D_MODEL_TYPES = frozenset(["a2d-modernbert"])


# ---------------------------------------------------------------------------
# Internal patching helpers
# ---------------------------------------------------------------------------


def _patch_a2d_peft(model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]) -> tuple[int, int, int]:
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
            layer.self_attn.forward = types.MethodType(A2DAttention_fast_forward, layer.self_attn)

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
        if (
            hasattr(o_proj, "lora_A")
            and _no_bias(o_proj)
            and _no_lora_mag(o_proj)
        ):
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
    from unturtle.models.llada.modeling_llada import LLaDALlamaBlock, _make_llada_fast_rope_forward

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
        if rotary_emb is not None and not getattr(rotary_emb, "_fast_rope_patched", False):
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
    """Patch A2DModernBertForMaskedLM with bidirectional fast attention and Triton O-projection.

    ModernBERT uses fused ``Wqkv`` and ``Wo`` (attention) and ``Wi`` / ``Wo`` (MLP).
    Unlike the LLaMA/Qwen2 path, QKV and MLP Triton kernels are **not** applied
    in this initial implementation because the fused projection shapes differ from
    what ``apply_lora_qkv`` / ``apply_lora_mlp_swiglu`` expect.

    What IS patched:
    - ``layer.attn.forward`` → ``ModernBertAttention_fast_forward`` (CUDA only)
    - ``layer.attn.Wo``     → ``apply_lora_o`` when conditions allow (CUDA, no dropout, no bias)

    Layer hierarchy:
        PeftModel → base_model → model (A2DModernBertForMaskedLM)
            → model (A2DModernBertModel) → layers[i].attn / .mlp

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
    except Exception as exc:  # noqa: BLE001
        if "quantization_config" not in load_kwargs:
            raise

        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs.pop("quantization_config", None)
        fallback_kwargs.pop("device_map", None)
        _warn_once(
            "FastDiffusionModel: 4-bit loading failed — retrying with full-precision loading."
        )
        _logger.debug(
            "FastDiffusionModel: retrying without 4-bit quantization after error: %s", exc
        )
        return loader.from_pretrained(model_name, **fallback_kwargs)


def _load_model_auto(model_name: str, load_kwargs: dict, trust_remote_code: bool) -> Any:
    """Try to load a model via AutoModel, AutoModelForMaskedLM, AutoModelForCausalLM.

    Registers unturtle model types so AutoConfig resolves them before trying.
    Falls back through the chain and raises if all attempts fail.

    Strategy: peek at the model_type in AutoConfig, and if it matches an unturtle
    native class (llada, dream, a2d_*), use that class directly — bypassing
    trust_remote_code so the Hub's potentially older modeling code is never loaded.
    This ensures fixes in unturtle model classes (e.g. _tied_weights_keys) take effect.
    """
    import unturtle.models  # noqa: F401 — registers A2D/LLaDA/Dream AutoConfig entries

    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

    # Map model_type → unturtle model class (bypasses trust_remote_code Hub code).
    _UNTURTLE_MODEL_CLASSES: dict[str, Any] = {}
    try:
        from unturtle.models.llada import LLaDAModelLM
        _UNTURTLE_MODEL_CLASSES["llada"] = LLaDAModelLM
    except ImportError:
        pass
    try:
        from unturtle.models.dream import DreamModel
        _UNTURTLE_MODEL_CLASSES["dream"] = DreamModel
    except ImportError:
        pass

    # Peek at model_type without loading weights.
    try:
        peek_kwargs: dict[str, Any] = {}
        if "token" in load_kwargs:
            peek_kwargs["token"] = load_kwargs["token"]
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, **peek_kwargs
        )
        model_type = getattr(config, "model_type", "")
        if model_type in _UNTURTLE_MODEL_CLASSES:
            native_cls = _UNTURTLE_MODEL_CLASSES[model_type]
            _logger.debug(
                "FastDiffusionModel: using native unturtle class %s for model_type=%r",
                native_cls.__name__, model_type,
            )
            return _load_model_with_optional_4bit_fallback(native_cls, model_name, load_kwargs)
    except torch.cuda.OutOfMemoryError:
        raise  # OOM should propagate, not fall through to slower AutoModel loaders
    except Exception as exc:  # noqa: BLE001
        _logger.debug("FastDiffusionModel: native class lookup failed: %s", exc)

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
                    _logger.debug(
                        "FastDiffusionModel: RoPE extension failed: %s", exc
                    )


def _propagate_max_seq_length(model: Any, max_seq_length: int) -> None:
    """Set max_seq_length on every nested model attribute (mirrors unsloth)."""
    internal = model
    while hasattr(internal, "model"):
        internal.max_seq_length = max_seq_length
        internal = internal.model
    internal.max_seq_length = max_seq_length
    for module in model.modules():
        module.max_seq_length = max_seq_length


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
                                ``A2DLlamaLMHeadModel``).  When *None* the class
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
        # --- dtype auto-detection ---
        if dtype is None:
            if torch.cuda.is_available():
                dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
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

        # --- Resolve model class ---
        if model_class is None:
            model = _load_model_auto(model_name, load_kwargs, trust_remote_code)
        else:
            model = model_class.from_pretrained(model_name, **load_kwargs)

        # --- Apply stubs and sequence length ---
        _install_apply_stubs(model)
        model.max_seq_length = max_seq_length
        _propagate_max_seq_length(model, max_seq_length)
        _extend_rope_if_possible(model, max_seq_length)

        # --- Tokenizer ---
        tokenizer = _load_tokenizer(model_name, trust_remote_code, token)

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

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_reentrant=True,
        )
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

        if model_type in _A2D_MODEL_TYPES:
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
            n_blocks = len(_llada_transformer.blocks) if _llada_transformer is not None else 0
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
                + ", ".join(sorted(
                    _A2D_MODEL_TYPES | _DREAM_MODEL_TYPES
                    | _LLADA_MODEL_TYPES | _MODERNBERT_A2D_MODEL_TYPES
                ))
            )

        # Propagate max_seq_length through the wrapped model hierarchy
        if hasattr(model, "max_seq_length"):
            _propagate_max_seq_length(model, model.max_seq_length)

    @staticmethod
    def for_inference(model: Any) -> Any:
        """Switch model to inference mode.

        Sets ``model.eval()`` and disables gradient checkpointing so that
        inference is as fast as possible.  Returns the model for convenience.

        Note: dLLMs do not use KV cache, so there is no cache-enabling step
        unlike ``FastLanguageModel.for_inference`` for AR models.

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

        push_kwargs: dict[str, Any] = dict(safe_serialization=safe_serialization, **kwargs)
        if token is not None:
            push_kwargs["token"] = token

        merged.push_to_hub(repo_id, **push_kwargs)
        _logger.info("FastDiffusionModel: merged model pushed to %r", repo_id)
        if tokenizer is not None:
            tokenizer.push_to_hub(repo_id, **push_kwargs)

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
                FastDiffusionModel.for_training(model, use_gradient_checkpointing=gc_mode)
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
