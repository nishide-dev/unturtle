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

"""Saving utilities for unturtle models.

Adapted from unsloth/save.py — the push_to_hub wrapper is changed to tag
models with "unturtle" instead of "unsloth".
"""

from __future__ import annotations

import inspect
import logging
import types
from typing import Any

_logger = logging.getLogger(__name__)


def patch_saving_functions(model, vision: bool = False):
    """Patch ``push_to_hub`` and related methods on a PEFT model.

    Wraps ``push_to_hub`` so that "unturtle" is appended to the model's tags
    and to the commit message / description when pushing to HuggingFace Hub.

    Also delegates the heavier merged/GGUF/GGML/TorchAO save methods back to
    the upstream ``unsloth.save`` implementation so that those workflows still
    work without requiring a full re-port here.

    Args:
        model:  PEFT-wrapped model returned by ``FastDiffusionModel.get_peft_model``.
        vision: Unused; kept for API compatibility with the unsloth version.
    """
    # Determine the original (un-patched) push_to_hub
    if (
        hasattr(model, "push_to_hub")
        and model.push_to_hub.__name__ == "_unturtle_push_to_hub"
        and hasattr(model, "original_push_to_hub")
    ):
        # Already patched; no-op
        return model

    # Walk the model chain and patch each push_to_hub
    original_model = model
    while True:
        if (
            hasattr(original_model, "push_to_hub")
            and original_model.push_to_hub.__name__ != "_unturtle_push_to_hub"
        ):
            original_model.original_push_to_hub = original_model.push_to_hub
            original_model.push_to_hub = types.MethodType(
                _unturtle_push_to_hub, original_model
            )
            if hasattr(original_model, "add_model_tags"):
                original_model.add_model_tags(["unturtle"])

        if hasattr(original_model, "model"):
            original_model = original_model.model
        else:
            break

    # Delegate heavier save methods to unsloth.save so they remain functional.
    # Only unavailability-shaped failures (unsloth missing / API drift) are
    # tolerated — real bugs inside patch_saving_functions must propagate.
    try:
        from unsloth.save import patch_saving_functions as _unsloth_patch

        _unsloth_patch(model, vision=vision)
    except (ImportError, AttributeError) as exc:
        _logger.warning(
            "unturtle.save: skipping unsloth.save.patch_saving_functions — "
            "merged/GGUF/TorchAO save methods will be unavailable on this model "
            "(%s: %s)",
            type(exc).__name__,
            exc,
        )

    return model


def _unturtle_push_to_hub(self, *args, **kwargs):
    """push_to_hub wrapper that injects the 'unturtle' tag."""
    # Collect all arguments via the original signature
    sig = inspect.signature(self.original_push_to_hub)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)

    # Inject tag
    if (
        "tags" in arguments
        and arguments["tags"] is not None
        and isinstance(arguments["tags"], (list, tuple))
        and "unturtle" not in arguments["tags"]
    ):
        arguments["tags"] = list(arguments["tags"]) + ["unturtle"]
    elif "tags" in arguments:
        arguments["tags"] = ["unturtle"]
    elif hasattr(self, "add_model_tags"):
        self.add_model_tags(["unturtle"])

    # Inject commit_message
    if "commit_message" in arguments:
        msg = arguments["commit_message"]
        if msg is not None:
            if not msg.endswith(" "):
                msg += " "
            if "Unturtle" not in msg:
                msg += "(Trained with Unturtle)"
        else:
            msg = "Upload model trained with Unturtle"
        arguments["commit_message"] = msg

    # Inject commit_description
    if "commit_description" in arguments:
        desc = arguments["commit_description"]
        if desc is not None:
            if not desc.endswith(" "):
                desc += " "
            if "Unturtle" not in desc:
                desc += "(Trained with Unturtle)"
        else:
            desc = "Upload model trained with Unturtle"
        arguments["commit_description"] = desc

    try:
        return self.original_push_to_hub(**arguments)
    except TypeError:
        # Fallback: drop tags if the original method doesn't accept them
        arguments.pop("tags", None)
        return self.original_push_to_hub(**arguments)


def prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    use_reentrant: bool = True,
):
    """Prepare a quantized model for LoRA training with unsloth's semantics.

    peft's ``prepare_model_for_kbit_training`` upcasts every non-quantized
    fp16/bf16 parameter to fp32, so a 4-bit model's real hidden states become
    fp32 while its weights dequantize to the 16-bit compute dtype. The fused
    LoRA paths (unsloth ``matmul_lora`` — behind Unturtle's bias-aware QKV AND
    unsloth's own MLP/O hooks) multiply the activation directly against the
    dequantized weight, so under that preparation no fused path can execute
    (#177). Unsloth's preparation keeps frozen parameters at their loaded
    dtype — only trainable LoRA parameters are upcast to fp32, and
    ``matmul_lora`` casts those to the activation dtype per matmul — so
    activations stay in the compute dtype and every fused path runs.

    Falls back to the peft implementation when unsloth is unavailable; on that
    path no fused kernel can be installed either (they import from unsloth),
    so peft's fp32 upcast cannot conflict with ``matmul_lora``.
    """
    if use_gradient_checkpointing not in (True, False, "unsloth"):
        # unsloth asserts this exact domain; peft accepted any truthy value.
        use_gradient_checkpointing = bool(use_gradient_checkpointing)

    try:
        from unsloth.models._utils import (
            prepare_model_for_kbit_training as _unsloth_prepare,
        )
    except (ImportError, OSError, AttributeError):
        _unsloth_prepare = None

    if _unsloth_prepare is not None:
        return _unsloth_prepare(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_reentrant=use_reentrant,
        )

    from peft import prepare_model_for_kbit_training as _peft_prepare

    return _peft_prepare(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
    )


__all__ = ["patch_saving_functions", "prepare_model_for_kbit_training"]


# ---------------------------------------------------------------------------
# Save / export helpers (#185 PR 3) — moved from the façade; FastDiffusionModel
# delegates here. Heavy imports stay lazy, matching this module's style.
# ---------------------------------------------------------------------------


def import_bitsandbytes() -> Any:
    """Import hook for bitsandbytes (separate function for testability)."""
    import bitsandbytes as bnb

    return bnb


def find_quantized_linear_modules(model: Any) -> list[tuple[str, Any]]:
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


def dequantize_merged_model_(model: Any) -> Any:
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
    import torch

    quantized = find_quantized_linear_modules(model)
    if not quantized:
        return model

    error_hint = (
        "cannot save a truthful merged 16-bit artifact from 4-bit weights. "
        "Re-load the checkpoint with load_in_4bit=False (CLI: "
        "`unturtle export --no-load-in-4bit`) and export again."
    )
    try:
        bnb = import_bitsandbytes()
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


def save_pretrained_merged(
    model,
    save_directory: str,
    tokenizer=None,
    safe_serialization: bool = True,
    **kwargs,
) -> None:
    """Merge LoRA adapters into the base weights and save (façade delegate)."""
    import copy

    _logger.info("FastDiffusionModel: merging LoRA adapters into base weights …")
    merged = copy.deepcopy(model)
    # merge_and_unload returns the unwrapped base model with adapters merged.
    merged = merged.merge_and_unload()
    # On a 4-bit-loaded model the merged Linear layers are still bnb
    # Linear4bit — dequantize (or fail loudly) so the saved artifact is
    # genuinely 16-bit, never mislabeled nf4 weights.
    merged = dequantize_merged_model_(merged)
    merged.save_pretrained(
        save_directory,
        safe_serialization=safe_serialization,
        **kwargs,
    )
    _logger.info("FastDiffusionModel: merged model saved to %r", save_directory)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)
        _logger.info("FastDiffusionModel: tokenizer saved to %r", save_directory)


def push_to_hub_merged(
    model,
    repo_id: str,
    tokenizer=None,
    safe_serialization: bool = True,
    token=None,
    **kwargs,
) -> None:
    """Merge LoRA adapters and push merged weights to the Hub (façade delegate)."""
    import copy

    _logger.info("FastDiffusionModel: merging LoRA adapters for Hub push …")
    merged = copy.deepcopy(model)
    merged = merged.merge_and_unload()
    # Same honesty guarantee as save_pretrained_merged: never push nf4
    # weights under a merged-16bit label.
    merged = dequantize_merged_model_(merged)

    push_kwargs = dict(safe_serialization=safe_serialization, **kwargs)
    if token is not None:
        push_kwargs["token"] = token

    merged.push_to_hub(repo_id, **push_kwargs)
    _logger.info("FastDiffusionModel: merged model pushed to %r", repo_id)
    if tokenizer is not None:
        tokenizer.push_to_hub(repo_id, **push_kwargs)


def save_pretrained_gguf(
    model,
    save_directory: str,
    tokenizer=None,
    quantization_method: str = "q4_k_m",
    **kwargs,
) -> None:
    """Convert and save model weights in GGUF format (façade delegate)."""
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


def save_lora_adapter(
    model,
    save_directory: str,
    tokenizer=None,
) -> None:
    """Save LoRA adapter weights only, no base weights (façade delegate)."""
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
