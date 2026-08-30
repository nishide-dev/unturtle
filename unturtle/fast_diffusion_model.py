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
import contextvars
import functools
import importlib
import logging
import types
from typing import Any, Literal, Optional

import torch
from peft.tuners.lora import Linear as LoraLinear
from transformers import AutoConfig, AutoTokenizer

import unturtle.save as _save
from unturtle.models.backbones.dream.modeling_dream import (
    DreamAttention_fast_forward,
)
from unturtle.models.backbones.modernbert._fast_forward import (
    ModernBertAttention_fast_forward,
)
from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
    TinyA2DAttention_fast_forward,
)

# Model-family loading knowledge (#68).  Safe to import eagerly: the registry
# holds zero-arg resolvers, so nothing under models/backbones/ is imported
# until a lookup actually needs a class.  Nothing under unturtle/models/
# imports this module, so the dependency stays one-directional.
from unturtle.models.integrations import (
    find_peft_integration,
    native_model_classes,
    post_load_class_swaps,
    supported_peft_model_types,
)
from unturtle.models.integrations.fast_path_support import (
    FAST_LORA_IMPORT_ERROR as _FAST_LORA_IMPORT_ERROR,
)
from unturtle.models.integrations.fast_path_support import (
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
    apply_lora_qkv_with_bias,
)
from unturtle.models.integrations.fast_path_support import no_bias as _no_bias
from unturtle.models.integrations.fast_path_support import (
    no_lora_magnitude as _no_lora_mag,
)
from unturtle.models.integrations.fast_path_support import (
    require_fast_lora as _require_fast_lora,
)
from unturtle.models.integrations.fast_path_support import warn_once as _warn_once
from unturtle.models.integrations.peft_preparation import (
    _original_apply_o,
    _original_apply_qkv,
    build_lora_config,
    prepare_peft_model,
)
from unturtle.models.integrations.peft_preparation import (
    apply_gradient_checkpointing_mode as _apply_gradient_checkpointing_mode,
)
from unturtle.models.integrations.peft_preparation import (
    get_gradient_checkpointing_mode as _get_gradient_checkpointing_mode,
)
from unturtle.models.integrations.peft_preparation import (
    inference_mode as _inference_mode,
)
from unturtle.models.integrations.peft_preparation import (
    install_apply_stubs as _install_apply_stubs,
)
from unturtle.models.integrations.peft_preparation import (
    set_inference_mode as _set_inference_mode,
)
from unturtle.models.integrations.peft_preparation import (
    set_training_mode as _set_training_mode,
)
from unturtle.models.integrations.peft_preparation import (
    wrap_with_peft_seeded as _wrap_with_peft_seeded,
)
from unturtle.models.integrations.reports import (
    LivenessReport,
    LoadedModel,
    PatchReport,
    PreparedPeftModel,
    SupportResult,
)
from unturtle.models.loading import (
    _LOAD_PATH_TRACE,
    _automodel_loaders,
    _extend_rope_if_possible,
    _import_fastmodel,
    _integration_name_for,
    _load_model_auto,
    _load_model_auto_traced,
    _load_model_with_optional_4bit_fallback,
    _load_native,
    _load_tokenizer,
    _load_via_automodel,
    _load_via_fastmodel,
    _native_model_classes,
    _patch_for_diffusion,
    _propagate_max_seq_length,
    load_model,
)
from unturtle.save import (
    dequantize_merged_model_ as _dequantize_merged_model_,
)
from unturtle.save import (
    find_quantized_linear_modules as _find_quantized_linear_modules,
)
from unturtle.save import (
    import_bitsandbytes as _import_bitsandbytes,
)
from unturtle.save import patch_saving_functions

_logger = logging.getLogger(__name__)


# The per-family PEFT model_type vocabulary now lives in the
# BackboneIntegration registry (#68) as each integration's
# ``peft_model_types``.  Keeping frozensets here as well would be a second
# copy of the same facts, free to drift from the one dispatch reads.


# ---------------------------------------------------------------------------
# Internal patching helpers
# ---------------------------------------------------------------------------


def _fast_path_support(model: Any) -> SupportResult:
    """Three-valued dtype gate for the fused fast paths (#177, frozen by #184).

    - ``supported``: no quantized weights, or the embedding dtype equals the
      single dtype the quantized weights dequantize to;
    - ``unsupported`` / ``incompatible_compute_dtype``: mixed or mismatched;
    - ``unverified`` / ``input_embedding_unresolvable``: the structure could
      not be inspected. Production stays fail-open here (per-layer gates still
      apply) but the report says so — never ``supported``.
    """
    quantized = _find_quantized_linear_modules(model)
    if not quantized:
        return SupportResult("supported", details={"quantized_modules": 0})
    quant_dtypes = {
        str(getattr(module.weight.quant_state, "dtype", None))
        for _, module in quantized
    }
    quant_dtypes.discard("None")
    if not quant_dtypes:
        return SupportResult(
            "unverified",
            reason="quant_state_dtype_unreadable",
            details={"quantized_modules": len(quantized)},
        )
    try:
        get_embeddings = getattr(model, "get_input_embeddings", None)
        embedding = get_embeddings() if callable(get_embeddings) else None
    except Exception:  # noqa: BLE001 — e.g. NotImplementedError on exotic models
        embedding = None
    weight = getattr(embedding, "weight", None)
    if weight is None:
        return SupportResult(
            "unverified",
            reason="input_embedding_unresolvable",
            details={"quant_dtypes": sorted(quant_dtypes)},
        )
    details = {
        "embedding_dtype": str(weight.dtype),
        "quant_dtypes": sorted(quant_dtypes),
        "quantized_modules": len(quantized),
    }
    if len(quant_dtypes) != 1 or str(weight.dtype) not in quant_dtypes:
        return SupportResult(
            "unsupported", reason="incompatible_compute_dtype", details=details
        )
    return SupportResult("supported", details=details)


def _fast_path_dtype_incompatibility(model: Any) -> str | None:
    """Typed reason when no fast path can execute on this model, else ``None``.

    Compatibility adapter over :func:`_fast_path_support`: only an
    ``unsupported`` verdict blocks patching; ``unverified`` stays fail-open
    (behavior unchanged from #177) and is surfaced by the PatchReport instead.

    The fused LoRA paths (unsloth ``matmul_lora``) multiply the activation
    directly against the dequantized quantized weight, and the fast attention
    forwards assume 16-bit hidden states. When the hidden-state dtype (its
    origin: the input embedding) does not match what the quantized weights
    dequantize to — e.g. after peft's own ``prepare_model_for_kbit_training``
    upcast everything to fp32 — none of them can run (#177). Such a model must
    be left entirely on the standard PEFT path, never partially fast.

    The comparison target is ``quant_state.dtype`` — the dtype the weight
    actually DEQUANTIZES to in the fused path — deliberately NOT
    ``Linear4bit.compute_dtype``: the standard bnb forward casts activations
    to ``compute_dtype``, but ``matmul_lora`` never does. Measured: a bf16
    activation against an fp16-``quant_state`` weight fails in ``matmul_lora``
    while the standard forward succeeds, so gating on ``compute_dtype`` would
    install hooks whose first forward raises — the exact #177 failure shape.
    Mixed quant dtypes are likewise incompatible: no single hidden-state dtype
    can feed them all, and the contract is all-or-nothing.

    Never raises. A model whose structure cannot be resolved (no embedding,
    exotic wrapper) returns ``None`` — fail-open, because the per-layer gates
    in the patchers still apply and a false ``incompatible`` verdict would
    silently disable every fast path on a healthy model.
    """
    support = _fast_path_support(model)
    return support.reason if support.status == "unsupported" else None


# ---------------------------------------------------------------------------
# Observed fast-path installation and liveness (#185 PR 0 — descriptive only)
# ---------------------------------------------------------------------------


def _fast_callables() -> dict[str, tuple[Any, ...]]:
    """The fast callables a report recognises, by kind (None-safe when the
    kernels failed to import)."""
    qkv = tuple(f for f in (apply_lora_qkv, apply_lora_qkv_with_bias) if f is not None)
    o = tuple(f for f in (apply_lora_o,) if f is not None)
    mlp = tuple(f for f in (apply_lora_mlp_swiglu,) if f is not None)
    attention = (
        DreamAttention_fast_forward,
        TinyA2DAttention_fast_forward,
        ModernBertAttention_fast_forward,
    )
    return {"qkv": qkv, "o": o, "mlp": mlp, "attention_forward": attention}


def _has_lora(*modules: Any) -> bool:
    return all(hasattr(m, "lora_A") for m in modules if m is not None)


def _observe_fast_paths(model: Any) -> dict[str, dict[str, tuple[str, ...]]]:
    """Which fast callables are INSTALLED on which modules — by identity.

    Observation, not intention: a callable is ``applied`` only when the module
    attribute *is* one of the known fast implementations. ``skipped`` lists
    modules that carry the standard stub / class implementation although they
    are shaped for the kind (LoRA present on the projections). Installation is
    still not liveness — see :func:`probe_liveness`.
    """
    fast = _fast_callables()
    applied: dict[str, list[str]] = {k: [] for k in fast}
    applied["rope"] = []
    skipped: dict[str, list[str]] = {k: [] for k in applied}
    for name, module in model.named_modules():
        own = module.__dict__
        if "apply_qkv" in own:
            (applied if own["apply_qkv"] in fast["qkv"] else skipped)["qkv"].append(
                name
            )
        if "apply_o" in own:
            (applied if own["apply_o"] in fast["o"] else skipped)["o"].append(name)
        if "apply_wo" in own:
            (applied if own["apply_wo"] in fast["o"] else skipped)["o"].append(name)
        if "apply_mlp" in own:
            (applied if own["apply_mlp"] in fast["mlp"] else skipped)["mlp"].append(
                name
            )
        instance_forward = own.get("forward")
        func = getattr(instance_forward, "__func__", None)
        if func in fast["attention_forward"]:
            applied["attention_forward"].append(name)
        elif "apply_qkv" in own and instance_forward is None:
            skipped["attention_forward"].append(name)
        if func in fast["mlp"]:
            applied["mlp"].append(name)
        elif (
            instance_forward is None
            and hasattr(module, "down_proj")
            and hasattr(module, "up_proj")
            and _has_lora(module.down_proj, module.up_proj)
            and "apply_mlp" not in own
        ):
            skipped["mlp"].append(name)
        if getattr(module, "_fast_rope_patched", False):
            applied["rope"].append(name)
    return {
        "applied": {k: tuple(v) for k, v in applied.items() if v},
        "skipped": {k: tuple(v) for k, v in skipped.items() if v},
    }


def probe_liveness(
    model: Any,
    inputs: dict[str, Any],
    *,
    backward: bool = False,
    applied: dict[str, tuple[str, ...]] | None = None,
) -> LivenessReport:
    """Prove which installed fast callables actually EXECUTE.

    Temporarily wraps every applied target on its own module with a counting
    wrapper (``functools.wraps`` keeps the sampler-visible signature), runs one
    forward with ``inputs`` (and a scalar backward when ``backward=True``),
    then restores the originals. Counters live on the very module/kind the
    report lists, so a counter on the wrong module cannot vouch for another.
    ``live`` requires every applied target to run at least once; backward
    liveness additionally requires every LoRA parameter under an applied
    target to receive a gradient. The model's state (mode, grads) is restored.
    """

    if applied is None:
        applied = _observe_fast_paths(model)["applied"]
    modules = dict(model.named_modules())
    counts: dict[str, int] = {}
    restore: list[tuple[Any, str, Any]] = []

    def install(module: Any, attr: str, key: str, bound: bool) -> None:
        original = module.__dict__[attr]
        counts[key] = 0
        if bound:  # MethodType(func, module)
            func = original.__func__

            @functools.wraps(func)
            def counting(self, *args, **kwargs):
                counts[key] += 1
                return func(self, *args, **kwargs)

            module.__dict__[attr] = types.MethodType(counting, module)
        else:  # plain function called as attr(self, X)

            @functools.wraps(original)
            def counting(*args, **kwargs):
                counts[key] += 1
                return original(*args, **kwargs)

            module.__dict__[attr] = counting
        restore.append((module, attr, original))

    for kind, paths in applied.items():
        for path in paths:
            module = modules.get(path)
            if module is None:
                counts[f"{path}:{kind}"] = 0
                continue
            key = f"{path}:{kind}"
            own = module.__dict__
            if kind == "qkv" and "apply_qkv" in own:
                install(module, "apply_qkv", key, bound=False)
            elif kind == "o" and "apply_o" in own:
                install(module, "apply_o", key, bound=False)
            elif kind == "o" and "apply_wo" in own:
                install(module, "apply_wo", key, bound=False)
            elif kind == "mlp" and "apply_mlp" in own:
                install(module, "apply_mlp", key, bound=False)
            elif kind in ("mlp", "attention_forward", "rope") and "forward" in own:
                # bound fast forwards (Dream/A2D/ModernBERT attention, LLaDA rope)
                install(module, "forward", key, bound=True)
            else:
                counts[key] = 0  # nothing to count: cannot be live

    was_training = model.training
    backward_counts: dict[str, int] | None = None
    try:
        if backward:
            model.train()
            model.zero_grad(set_to_none=True)
            output = model(**inputs)
            logits = output.logits if hasattr(output, "logits") else output
            logits.float().square().mean().backward()
            backward_counts = {}
            for kind, paths in applied.items():
                for path in paths:
                    module = modules.get(path)
                    if module is None:
                        backward_counts[f"{path}:{kind}"] = 0
                        continue
                    lora_params = [
                        param for n, param in module.named_parameters() if "lora_" in n
                    ]
                    if not lora_params:
                        # A parameter-less target (e.g. LLaDA's rotary module)
                        # cannot receive gradients by construction — it must not
                        # gate backward liveness. Forward counters still cover it.
                        continue
                    backward_counts[f"{path}:{kind}"] = sum(
                        1 for param in lora_params if param.grad is not None
                    )
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(**inputs)
    finally:
        # LIFO: a (module, attr) installed twice must end at its pre-probe original
        for module, attr, original in reversed(restore):
            module.__dict__[attr] = original
        model.train(was_training)

    forward_live = bool(counts) and all(v > 0 for v in counts.values())
    backward_live = (
        None
        if backward_counts is None
        else bool(backward_counts) and all(v > 0 for v in backward_counts.values())
    )
    return LivenessReport(
        forward=dict(counts),
        backward=backward_counts,
        forward_live=forward_live,
        backward_live=backward_live,
        live=forward_live and (backward_live is not False),
        probe={
            "input_keys": sorted(inputs),
            "backward_requested": backward,
            "targets": len(counts),
        },
    )


def _requested_kinds(target_modules: Any, on_cuda: bool) -> tuple[str, ...]:
    names = set(target_modules or ())
    requested: list[str] = []
    if names & {"q_proj", "k_proj", "v_proj", "Wqkv", "att_proj"}:
        requested.append("qkv")
    if names & {"o_proj", "attn_out", "Wo"}:
        requested.append("o")
    if names & {"gate_proj", "up_proj", "down_proj", "ff_proj", "ff_out", "Wi"}:
        requested.append("mlp")
    if on_cuda:
        requested.append("attention_forward")
    return tuple(requested)


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
        """Load a dLLM model (optionally 4-bit quantised). Returns
        ``(model, tokenizer)`` — the compatibility shape of
        :meth:`from_pretrained_with_report`, whose objects are returned as-is.
        """
        return FastDiffusionModel.from_pretrained_with_report(
            model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            token=token,
            **kwargs,
        ).as_tuple()

    @staticmethod
    def from_pretrained_with_report(
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        model_class: Any = None,
        trust_remote_code: bool = True,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> LoadedModel:
        """Load a dLLM model (optionally 4-bit quantised) with provenance.

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
        return load_model(
            model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            token=token,
            **kwargs,
        )

    @staticmethod
    def get_peft_model_with_report(
        model: Any,
        r: int = 16,
        target_modules: Optional[list[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: Literal["none", "all", "lora_only"] = "none",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int | None = 3407,
        **kwargs: Any,
    ) -> tuple[Any, PatchReport]:
        """Apply LoRA and patch with unsloth's Triton kernels; return the model
        AND its :class:`PatchReport`.

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
            random_state:               Seed for LoRA adapter initialization. PEFT
                                        draws ``lora_A`` from torch's GLOBAL RNG, so
                                        without this two wraps differ whenever anything
                                        consumed the RNG earlier (#188). The seed is
                                        applied inside a forked torch RNG: adapter init
                                        is deterministic given ``random_state`` AND the
                                        caller's global RNG state is neither consumed nor
                                        changed (unsloth instead calls ``set_seed`` on the
                                        global RNG). ``None`` disables seeding.
            **kwargs:                   Forwarded to ``LoraConfig``.

        Returns:
            PEFT model with Triton LoRA kernels patched in.
        """
        lora_config = build_lora_config(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            **kwargs,
        )
        prepared = prepare_peft_model(
            model,
            lora_config,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
        )
        # Optimization is a SEPARATE, optional step on the prepared model: the
        # family provider may decline with a typed fallback without affecting
        # preparation (#185 PR 2 boundary).
        report = FastDiffusionModel.patch_peft_model_with_report(
            prepared.model, lora_dropout=lora_dropout, bias=bias
        )
        patch_saving_functions(prepared.model)

        return prepared.model, report

    @staticmethod
    def get_peft_model(
        model: Any,
        r: int = 16,
        target_modules: Optional[list[str]] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        bias: Literal["none", "all", "lora_only"] = "none",
        use_gradient_checkpointing: str | bool = "unsloth",
        random_state: int | None = 3407,
        **kwargs: Any,
    ) -> Any:
        """Apply LoRA and patch with unsloth's Triton kernels.

        Compatibility entry point: identical behavior to
        :meth:`get_peft_model_with_report`, report discarded — the returned
        object IS the one the report path built.
        """
        model, _report = FastDiffusionModel.get_peft_model_with_report(
            model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            **kwargs,
        )
        return model

    @staticmethod
    def patch_peft_model(
        model: Any,
        lora_dropout: float = 0,
        bias: Literal["none", "all", "lora_only"] = "none",
    ) -> None:
        """Inject Triton LoRA kernels and bidirectional attention into a PEFT model.

        Safe to call again after adding new adapters. Compatibility entry point:
        identical behavior to :meth:`patch_peft_model_with_report`, report
        discarded.

        Args:
            model:          PEFT-wrapped dLLM model.
            lora_dropout:   Must match the LoRA config used when wrapping.
            bias:           Must match the LoRA config used when wrapping.
        """
        FastDiffusionModel.patch_peft_model_with_report(
            model, lora_dropout=lora_dropout, bias=bias
        )

    @staticmethod
    def patch_peft_model_with_report(
        model: Any,
        lora_dropout: float = 0,
        bias: Literal["none", "all", "lora_only"] = "none",
        *,
        requested: tuple[str, ...] | None = None,
    ) -> PatchReport:
        """:meth:`patch_peft_model` plus a :class:`PatchReport` describing what
        was requested, applied (observed by callable identity), skipped, or
        withheld as a typed fallback. Liveness is NOT claimed here — call
        :func:`probe_liveness` (or ``FastDiffusionModel.probe_liveness``) with
        real inputs to prove execution.
        """
        model_type = model.config.model_type

        integration = find_peft_integration(model_type)
        if integration is None:
            raise NotImplementedError(
                f"FastDiffusionModel does not yet support model_type={model_type!r}. "
                "Supported types: " + ", ".join(supported_peft_model_types())
            )

        # Resolved through the registry rather than a module-level reference so
        # that tests monkeypatching `<provider>.patch_peft` by name still take effect.
        provider = integration.fast_paths
        patcher = integration.peft_patcher
        if patcher is None:
            # Unreachable today (all patchers live in this module), but a
            # patcher behind an optional dependency would resolve to None;
            # report that as unsupported rather than dying on a TypeError.
            raise NotImplementedError(
                f"FastDiffusionModel cannot PEFT-patch model_type={model_type!r}: "
                f"the {integration.name!r} patcher could not be imported."
            )

        support = _fast_path_support(model)
        if provider is not None and support.status != "unsupported":
            # The family knows its own structure; a model it cannot traverse
            # is a typed, whole-set withhold (never a partial patch).
            structure = provider.check_structure(model)
            if structure.status == "unsupported":
                support = structure
        incompatibility = support.reason if support.status == "unsupported" else None
        warnings_seen: list[str] = []
        fallback: str | None = None
        if incompatibility == "structure_mismatch":
            fallback = incompatibility
            message = (
                "FastDiffusionModel: skipping ALL fast-path patching "
                f"(reason={incompatibility}): {support.details}. "
                "The standard PEFT path is retained."
            )
            warnings_seen.append(message)
            _warn_once(message)
        elif incompatibility is not None:
            # All-or-nothing: a model whose hidden states cannot feed the fused
            # kernels must not end up with SOME fast hooks installed (#177) —
            # the pre-fix failure mode was exactly a fully-hooked model whose
            # first forward raised. The standard PEFT path handles the dtype
            # itself (bnb casts activations to the compute dtype internally).
            fallback = incompatibility
            message = (
                "FastDiffusionModel: skipping ALL fast-path patching "
                f"(reason={incompatibility}): the model's hidden-state dtype "
                "does not match its quantization compute dtype, so neither the "
                "fused LoRA kernels nor the fast attention forwards can "
                "execute (#177). The standard PEFT path is retained."
            )
            warnings_seen.append(message)
            _warn_once(message)
        else:
            counts = patcher(model, lora_dropout, bias)
            reporter = (
                provider.report if provider is not None else integration.peft_report
            )
            if reporter is not None:
                message = reporter(model, counts)
                warnings_seen.append(message)
                _warn_once(message)

        # Propagate max_seq_length through the wrapped model hierarchy
        if hasattr(model, "max_seq_length"):
            _propagate_max_seq_length(model, model.max_seq_length)

        first_param = next(iter(model.parameters()), None)
        on_cuda = first_param is not None and first_param.device.type == "cuda"
        observed = _observe_fast_paths(model)
        if requested is None:
            peft_config = getattr(model, "peft_config", {}) or {}
            target_modules: set[str] = set()
            for config in peft_config.values():
                targets = getattr(config, "target_modules", None) or ()
                target_modules |= (
                    set(targets) if not isinstance(targets, str) else set()
                )
            kinds = (
                provider.requested_kinds if provider is not None else _requested_kinds
            )
            requested = kinds(target_modules, on_cuda)
        return PatchReport(
            family=integration.name,
            model_type=str(model_type),
            support=support,
            requested=tuple(requested),
            applied=observed["applied"],
            skipped=observed["skipped"] if fallback is None else {},
            fallback=fallback,
            applicability={
                "on_cuda": on_cuda,
                "lora_dropout": lora_dropout,
                "bias": bias,
                "gate": support.to_dict(),
            },
            liveness=None,
            warnings=tuple(warnings_seen),
        )

    #: Liveness probe re-exported on the facade (see module-level function).
    probe_liveness = staticmethod(probe_liveness)

    #: PEFT-preparation boundary re-exported on the facade (#185 PR 2): returns
    #: a PreparedPeftModel whose model carries NO fast-path optimization yet.
    prepare_peft_model = staticmethod(prepare_peft_model)

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
        return _set_inference_mode(model)

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
        # An UN-wrapped canvas-family instance (e.g. a direct unsloth
        # FastModel load, which our loader no longer produces but a user
        # still can): its upstream `generate` has no unified `algorithm`
        # entry. Route it through the explicit runner (#186) — never restamp
        # the class.
        from unturtle.models.loading import _POST_LOAD_CLASS_SWAPS

        model_type = getattr(getattr(model, "config", None), "model_type", None)
        resolver = _POST_LOAD_CLASS_SWAPS.get(model_type) if model_type else None
        if resolver is not None and not isinstance(model, resolver()):
            from unturtle.models.generation.sampler import (
                GenerationRequest,
                dispatch_generation,
            )

            return dispatch_generation(
                model,
                GenerationRequest(
                    inputs=inputs,
                    generation_config=kwargs.pop("generation_config", None),
                    kwargs=kwargs,
                ),
                algorithm=algorithm,
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
        return _set_training_mode(model, use_gradient_checkpointing)

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
        _save.save_pretrained_merged(
            model,
            save_directory,
            tokenizer=tokenizer,
            safe_serialization=safe_serialization,
            **kwargs,
        )

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
        _save.push_to_hub_merged(
            model,
            repo_id,
            tokenizer=tokenizer,
            safe_serialization=safe_serialization,
            token=token,
            **kwargs,
        )

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
        _save.save_pretrained_gguf(
            model,
            save_directory,
            tokenizer=tokenizer,
            quantization_method=quantization_method,
            **kwargs,
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
        _save.save_lora_adapter(model, save_directory, tokenizer=tokenizer)

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
        with _inference_mode(model) as m:
            yield m


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
