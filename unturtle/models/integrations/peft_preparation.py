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

"""PEFT preparation (#185 PR 2): everything that turns a base dLLM into a
LoRA-wrapped, training-ready model — SEPARATE from optional fast-path
optimization.

The boundary: :func:`prepare_peft_model` returns a typed
:class:`~unturtle.models.integrations.reports.PreparedPeftModel` whose
``model`` carries **no** fast hooks yet. The façade hands that model to the
family optimization provider (``patch_peft_model_with_report``), which is free
to decline (typed fallback) without ever affecting preparation.

Owned here, extracted from the façade behavior-for-behavior:

- default LoRA target modules and ``LoraConfig`` construction
  (``TaskType.FEATURE_EXTRACTION`` to avoid ``PeftModelForCausalLM`` guards);
- the ``apply_qkv`` / ``apply_o`` dispatch stubs, installed before wrapping;
- quantized-model detection and k-bit preparation (#177 path);
- gradient-checkpointing enablement and the tracked, round-trippable mode;
- adapter creation with the #188 forked-RNG contract: ``random_state`` is
  applied inside ``torch.random.fork_rng`` **immediately around**
  ``peft.get_peft_model`` — deterministic adapters, caller RNG untouched,
  ``None`` = legacy unseeded.

Nothing here installs a Triton kernel or a fast forward.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model

from unturtle.models.integrations.reports import PreparedPeftModel
from unturtle.save import prepare_model_for_kbit_training

#: The default LoRA targets (Llama/Qwen-shaped names; families with other
#: naming pass their own ``target_modules``).
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Quantization markers that require k-bit preparation.
_QUANT_ATTRS = ("is_loaded_in_4bit", "is_loaded_in_8bit", "hqq_quantized")
_QUANT_METHODS = {"gptq", "aqlm", "eetq", "torchao", "hqq"}


def build_lora_config(
    r: int = 16,
    target_modules: Optional[list[str]] = None,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    bias: Literal["none", "all", "lora_only"] = "none",
    **kwargs: Any,
) -> LoraConfig:
    """The façade's LoRA configuration, defaults included."""
    if target_modules is None:
        target_modules = list(DEFAULT_TARGET_MODULES)
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        **kwargs,
    )


def is_quantized_model(model: Any) -> bool:
    quantization_method = getattr(model, "quantization_method", None)
    return (
        any(getattr(model, attr, False) for attr in _QUANT_ATTRS)
        or quantization_method in _QUANT_METHODS
    )


def _original_apply_qkv(self: Any, X: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return self.q_proj(X), self.k_proj(X), self.v_proj(X)


def _original_apply_o(self: Any, X: torch.Tensor) -> torch.Tensor:
    return self.o_proj(X)


def install_apply_stubs(model: Any) -> None:
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


def wrap_with_peft_seeded(
    model: Any, lora_config: LoraConfig, random_state: int | None
) -> Any:
    """``peft.get_peft_model`` with adapter initialization owned by ``random_state``.

    PEFT initializes ``lora_A`` (kaiming) from torch's global generator, so the
    adapters depend on whatever consumed the RNG earlier in the process (#188:
    measured as two bit-stable adapter sets flipping with test ordering).
    Seeding happens inside ``torch.random.fork_rng`` covering the CPU generator
    and every CUDA device, so:

    - the same ``random_state`` yields the same adapters regardless of prior
      RNG consumption (the documented contract of the parameter);
    - the caller's global RNG state is restored on exit — unlike unsloth's
      ``set_seed(random_state)``, this does not reseed the caller's process.

    ``random_state=None`` keeps the legacy unseeded behavior.
    """
    if random_state is None:
        return get_peft_model(model, lora_config)
    devices = (
        list(range(torch.cuda.device_count()))
        if torch.cuda.is_available() and torch.cuda.is_initialized()
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(random_state)
        return get_peft_model(model, lora_config)


def get_gradient_checkpointing_mode(model: Any) -> bool | str:
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


def apply_gradient_checkpointing_mode(model: Any, mode: bool | str) -> None:
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


def prepare_peft_model(
    model: Any,
    lora_config: LoraConfig,
    *,
    use_gradient_checkpointing: str | bool = "unsloth",
    random_state: int | None = 3407,
) -> PreparedPeftModel:
    """PEFT preparation, without optimization.

    Behavior-for-behavior the preparation half of the façade's historical
    ``get_peft_model``: stub install → quantized detection → k-bit preparation
    OR gradient-checkpointing enablement → forked-RNG adapter creation →
    tracked GC mode. The returned :class:`PreparedPeftModel` is the typed
    boundary handed to the optional optimization provider; its ``model``
    carries no fast hooks.
    """
    # Install apply_qkv / apply_o stubs before PEFT wrapping so that
    # fast-forward functions can dispatch even when the model was not
    # loaded via from_pretrained (e.g. tests using random-weight models).
    install_apply_stubs(model)

    quantized = is_quantized_model(model)
    kbit_prepared = False
    if quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_reentrant=True,
        )
        kbit_prepared = True
    elif bool(use_gradient_checkpointing):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):
            input_embeddings = model.get_input_embeddings()
            if input_embeddings is not None:
                input_embeddings.register_forward_hook(
                    lambda module, inputs, output: output.requires_grad_(True)
                )
        apply_gradient_checkpointing_mode(model, use_gradient_checkpointing)

    model = wrap_with_peft_seeded(model, lora_config, random_state)
    model._unturtle_gradient_checkpointing_mode = use_gradient_checkpointing

    return PreparedPeftModel(
        model=model,
        lora_config=lora_config,
        quantized=quantized,
        kbit_prepared=kbit_prepared,
        gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )


__all__ = [
    "DEFAULT_TARGET_MODULES",
    "apply_gradient_checkpointing_mode",
    "build_lora_config",
    "get_gradient_checkpointing_mode",
    "install_apply_stubs",
    "is_quantized_model",
    "prepare_peft_model",
    "wrap_with_peft_seeded",
]
