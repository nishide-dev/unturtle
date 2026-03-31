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
"""

from __future__ import annotations

import functools
import logging
import types
from typing import Any, Literal, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from transformers import AutoConfig, AutoTokenizer

from unsloth.kernels.fast_lora import (
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
)
from unsloth.models._utils import prepare_model_for_kbit_training
from unsloth.save import patch_saving_functions

from unturtle.models.a2d._fast_forward import A2DAttention_fast_forward

_logger = logging.getLogger(__name__)


def _warn_once(msg: str) -> None:
    """Log a warning that won't repeat (uses transformers if available)."""
    try:
        from transformers.utils import logging as hf_logging

        hf_logger = hf_logging.get_logger(__name__)
        hf__warn_once(msg)
    except Exception:  # noqa: BLE001
        _logger.warning(msg)

# Model types that follow the standard LLaMA/Qwen2 layer hierarchy:
# model.model.model.layers (through PeftModel → base_model → model)
_A2D_MODEL_TYPES = frozenset(
    ["a2d-llama", "a2d-qwen2", "a2d-qwen3", "llama", "qwen2", "qwen3"]
)


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

    for layer in layers:
        # --- fast attention (bidirectional) ---
        layer.self_attn.forward = types.MethodType(A2DAttention_fast_forward, layer.self_attn)

        if lora_dropout != 0 or bias != "none":
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


def _no_bias(proj: Any) -> bool:
    return getattr(proj, "base_layer", proj).bias is None


def _no_lora_mag(proj: Any) -> bool:
    return len(getattr(proj, "lora_magnitude_vector", []) or []) == 0


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
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Load a dLLM model (optionally 4-bit quantised).

        Does NOT call unsloth's ``pre_patch()`` — that would inject causal
        fast-forward functions, which is wrong for bidirectional dLLMs.

        Args:
            model_name:         HuggingFace model id or local path.
            max_seq_length:     Maximum sequence length.
            dtype:              Torch dtype (None → bfloat16 when supported).
            load_in_4bit:       Enable 4-bit NF4 quantisation via bitsandbytes.
            model_class:        Explicit model class override (e.g.
                                ``A2DLlamaLMHeadModel``).  When *None* the class
                                is resolved via ``AutoModel``.
            trust_remote_code:  Passed to ``from_pretrained``.
            **kwargs:           Forwarded to ``from_pretrained``.

        Returns:
            ``(model, tokenizer)`` tuple.
        """
        if dtype is None:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

        load_kwargs: dict[str, Any] = dict(
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = bnb_config
            except ImportError:
                _warn_once(
                    "bitsandbytes not installed — falling back to full-precision loading."
                )

        # Resolve model class
        if model_class is None:
            # Import A2D models so AutoModel registry picks them up
            import unturtle.models  # noqa: F401 — registers AutoConfig entries

            from transformers import AutoModelForMaskedLM

            model = AutoModelForMaskedLM.from_pretrained(model_name, **load_kwargs)
        else:
            model = model_class.from_pretrained(model_name, **load_kwargs)

        # Set apply_qkv / apply_o stubs on each attention layer so the
        # dispatch protocol works before PEFT is applied.
        _install_apply_stubs(model)

        model.max_seq_length = max_seq_length
        _propagate_max_seq_length(model, max_seq_length)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
        except Exception:  # noqa: BLE001
            tokenizer = None

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

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_reentrant=True,
        )
        model = get_peft_model(model, lora_config)

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
        else:
            raise NotImplementedError(
                f"FastDiffusionModel does not yet support model_type={model_type!r}. "
                "Supported types: " + ", ".join(sorted(_A2D_MODEL_TYPES))
            )

        # Propagate max_seq_length through the wrapped model hierarchy
        if hasattr(model, "max_seq_length"):
            _propagate_max_seq_length(model, model.max_seq_length)


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
