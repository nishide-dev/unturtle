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

"""Model / tokenizer loading (#185 PR 3): the loader boundary behind
``FastDiffusionModel.from_pretrained``.

Owned here, extracted from the façade behavior-for-behavior:

- PEFT adapter checkpoint detection (adapter_config.json without full weights);
- dtype auto-detection and 4-bit quantization kwargs (BitsAndBytesConfig,
  device_map default, CPU fallbacks with the historical warnings);
- the load resolution chain: native Unturtle class → unsloth FastModel →
  transformers Auto* (with the 4-bit full-precision retry), traced through
  ``_LOAD_PATH_TRACE`` for :class:`LoadedModel` provenance;
- tokenizer loading, RoPE extension, ``max_seq_length`` propagation and the
  shared diffusion patch;
- :func:`load_model`, which returns the typed
  :class:`~unturtle.models.integrations.reports.LoadedModel`.

Function names keep their historical underscore spelling because they are
long-standing test seams (tests monkeypatch them by name on this module).

Wrapper-family model types (DiffusionGemma) load THROUGH their registered
wrapper class — wrapper-first on the Auto chain, FastModel skipped for those
types. The former runtime ``__class__`` swap is gone (#186): the load is the
one owner of the model's class.
"""

from __future__ import annotations

import contextvars
import importlib
import logging
from typing import Any, Callable, Optional

import torch
from transformers import AutoConfig, AutoTokenizer

from unturtle.models.integrations import native_model_classes, post_load_class_swaps
from unturtle.models.integrations.fast_path_support import warn_once as _warn_once
from unturtle.models.integrations.peft_preparation import (
    install_apply_stubs as _install_apply_stubs,
)
from unturtle.models.integrations.reports import LoadedModel
from unturtle.save import find_quantized_linear_modules

_logger = logging.getLogger(__name__)

#: model_type -> zero-arg wrapper resolver (used to try a registered wrapper
#: class FIRST on the Auto* chain). The ``__class__`` stamping itself lives in
#: the façade (#186); this map only orders the loaders.
_POST_LOAD_CLASS_SWAPS: dict[str, Any] = post_load_class_swaps()


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


def _native_model_classes() -> dict[str, Any]:
    """Build the ``model_type`` → unturtle native model class map.

    These classes are the from-scratch / wrapper implementations Unturtle owns
    (LLaDA, Dream, MDLM-DiT, Tiny-A2D Llama/Qwen2/Qwen3). Loading through them
    bypasses any ``trust_remote_code`` Hub modeling code, so fixes in the
    unturtle classes (e.g. ``_tied_weights_keys``) always take effect.

    The per-family knowledge lives in the BackboneIntegration registry (#68);
    this stays as the loader's entry point.  A family whose optional
    dependencies are missing drops out individually rather than emptying the
    map.
    """
    import unturtle.models  # noqa: F401 — registers A2D/LLaDA/Dream AutoConfig entries

    return native_model_classes()


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


def _automodel_loaders() -> list[tuple[str, Any]]:
    """The Auto* fallback chain, resolved at call time.

    A separate seam on purpose: unsloth **replaces**
    ``sys.modules["transformers"]`` with a different module object at import
    time, so a test that patches attributes on whatever ``transformers`` name
    it holds never reaches the classes this loader resolves — measured: the
    patched attribute was visible by direct read while the loader's own
    from-import still produced the real classes, and an ordering mutant
    survived a test that looked airtight.  Tests patch THIS function on the
    unturtle module instead, which no third-party import games can bypass.
    """
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
    )

    return [
        ("AutoModel", AutoModel),
        ("AutoModelForMaskedLM", AutoModelForMaskedLM),
        ("AutoModelForCausalLM", AutoModelForCausalLM),
    ]


def _load_via_automodel(
    model_name: str, load_kwargs: dict, model_type: str | None = None
) -> Any:
    """Load a non-native (HF-registered) model_type via the AutoModel fallback chain.

    This is the offline / unsloth-unavailable fallback path: loading/quantization is
    handled by ``transformers``' ``Auto*`` loaders.  The diffusion patch is applied
    afterwards by :func:`_patch_for_diffusion`, so the resulting model behaves as a
    bidirectional dLLM regardless of which path produced it.  Raises if every loader
    fails.

    The primary non-native path is :func:`_load_via_fastmodel` (unsloth FastModel);
    this function is only reached when that path is unavailable or raises.
    """
    loaders = _automodel_loaders()

    # A model_type with a registered wrapper loads through the wrapper class
    # itself first.  AutoModel resolves diffusion_gemma to the bare composite
    # model -- the wrong head for the wrapper contract (#96), and one the
    # swap guard now refuses to stamp.  Loading via the wrapper picks the
    # right head AND runs the normal __init__ postamble, so generation_config
    # is populated the ordinary way.  Any failure falls through to the Auto*
    # chain unchanged.
    if model_type is None:
        try:
            model_type = getattr(
                AutoConfig.from_pretrained(model_name, **load_kwargs),
                "model_type",
                None,
            )
        except Exception:  # noqa: BLE001 -- config fetch is best-effort here
            model_type = None
    resolver = _POST_LOAD_CLASS_SWAPS.get(model_type)
    if resolver is not None:
        wrapper_cls = resolver()
        loaders.insert(0, (wrapper_cls.__name__, wrapper_cls))
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


def _wrapper_model_type(model_name: str, load_kwargs: dict) -> str | None:
    """The model_type when it has a registered post-load wrapper, else None.

    Best-effort config peek, mirroring `_load_via_automodel`'s own: a failed
    peek routes through the normal chain rather than raising here.
    """
    try:
        model_type = getattr(
            AutoConfig.from_pretrained(model_name, **load_kwargs), "model_type", None
        )
    except Exception:  # noqa: BLE001 -- config fetch is best-effort here
        return None
    return model_type if model_type in _POST_LOAD_CLASS_SWAPS else None


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
    model, tokenizer, path = _load_model_auto_traced(
        model_name, load_kwargs, trust_remote_code, load_in_4bit=load_in_4bit
    )
    _LOAD_PATH_TRACE.set(path)
    return model, tokenizer


#: Load path taken by the most recent `_load_model_auto` call in this context.
#: `from_pretrained_with_report` reads it right after calling the module-level
#: seam (which tests monkeypatch by name); a replaced loader that does not
#: report leaves the default, so the report says "unknown" rather than guessing.
_LOAD_PATH_TRACE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "unturtle_load_path", default="unknown"
)


def _load_model_auto_traced(
    model_name: str,
    load_kwargs: dict,
    trust_remote_code: bool,
    *,
    load_in_4bit: bool = False,
) -> tuple[Any, Any | None, str]:
    """:func:`_load_model_auto` plus the load path taken (``native`` /
    ``upstream`` / ``auto``). Same resolution order, same module-level seams."""
    model = _load_native(model_name, load_kwargs, trust_remote_code)
    if model is not None:
        return model, None, "native"

    # Wrapper-family model types (#186): the registered wrapper class IS the
    # load path — `_load_via_automodel` tries it first. FastModel is skipped
    # for these types: it returns the upstream class with an instance-level
    # fast-generate shim and no generation_config, which the removed runtime
    # `__class__` swap used to repair after the fact. One owner now: the load.
    wrapper_type = _wrapper_model_type(model_name, load_kwargs)
    if wrapper_type is not None:
        return (
            _load_via_automodel(model_name, load_kwargs, model_type=wrapper_type),
            None,
            "auto",
        )

    result = _load_via_fastmodel(model_name, load_kwargs, load_in_4bit=load_in_4bit)
    if result is not None:
        model, tokenizer = result
        return model, tokenizer, "upstream"

    return _load_via_automodel(model_name, load_kwargs), None, "auto"


def _integration_name_for(model: Any) -> str | None:
    """Family name for a loaded (possibly PEFT-wrapped) model, read-only."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    from unturtle.models.integrations.registry import iter_integrations

    for integration in iter_integrations():
        if (
            model_type in integration.model_types
            or model_type in integration.peft_model_types
        ):
            return integration.name
    return None


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
# The loader boundary (#185 PR 3)
# ---------------------------------------------------------------------------


def detect_adapter_base(model_name: str) -> str | None:
    """Return the base model id when ``model_name`` is a local PEFT adapter
    checkpoint (adapter_config.json present, full weights absent); else None.

    Raises the historical errors for unreadable / invalid / incomplete
    adapter_config.json.
    """
    from pathlib import Path as _Path

    _local = _Path(model_name) if not model_name.startswith("http") else None
    if _local is None or not _local.is_dir():
        return None
    _has_full_weights = False
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
            or any(_local.glob("pytorch_model-*-of-*.bin"))
        )  # legacy sharded .bin
    except OSError as _e:
        _logger.warning(
            "FastDiffusionModel: could not scan %r for weight files (%s). "
            "Assuming full weights present to avoid incorrect adapter path.",
            str(_local),
            _e,
        )
        _has_full_weights = True
    if not (_local / "adapter_config.json").exists() or _has_full_weights:
        return None

    import json as _json

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
    return _base_model_id


def build_load_kwargs(
    dtype: Optional[torch.dtype],
    load_in_4bit: bool,
    trust_remote_code: bool,
    token: Optional[str],
    **kwargs: Any,
) -> tuple[dict[str, Any], torch.dtype, bool]:
    """dtype auto-detection + 4-bit quantization kwargs (historical behavior).

    Returns ``(load_kwargs, resolved_dtype, effective_load_in_4bit)`` where
    ``effective_load_in_4bit`` is the caller's intent gated on CUDA.
    """
    if dtype is None:
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
    return load_kwargs, dtype, load_in_4bit and not is_on_cpu


def load_model(
    model_name: str,
    max_seq_length: int = 2048,
    dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = True,
    model_class: Any = None,
    trust_remote_code: bool = True,
    token: Optional[str] = None,
    **kwargs: Any,
) -> LoadedModel:
    """The loader boundary: resolve, load and diffusion-patch a dLLM, returning
    a typed :class:`LoadedModel` with real load-path provenance.

    Wrapper-family model types load THROUGH their registered wrapper class
    (wrapper-first Auto chain); no runtime ``__class__`` mutation exists any
    more (#186 replaced the post-load swap).
    """
    # --- PEFT adapter checkpoint detection (API Gap G3) ---
    _base_model_id = detect_adapter_base(model_name)
    if _base_model_id is not None:
        from peft import PeftModel as _PeftModel

        _logger.info(
            "FastDiffusionModel: detected PEFT adapter checkpoint at %r; "
            "loading base model %r first.",
            model_name,
            _base_model_id,
        )
        base = load_model(
            model_name=_base_model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            model_class=model_class,
            trust_remote_code=trust_remote_code,
            token=token,
            **kwargs,
        )
        base_model, tokenizer = base.model, base.tokenizer
        try:
            model = _PeftModel.from_pretrained(base_model, model_name)
        except Exception as _e:
            raise RuntimeError(
                f"FastDiffusionModel: failed to load PEFT adapter from {model_name!r} "
                f"onto base model {_base_model_id!r}: {_e}"
            ) from _e
        _patch_for_diffusion(model, max_seq_length)
        _logger.info("FastDiffusionModel: PEFT adapter loaded from %r.", model_name)
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            integration=_integration_name_for(model),
            load_path="adapter",
            details={"base_model": _base_model_id},
        )

    load_kwargs, dtype, effective_4bit = build_load_kwargs(
        dtype, load_in_4bit, trust_remote_code, token, **kwargs
    )

    # --- Resolve model class (explicit override → native dLLM → FastModel → Auto*) ---
    fm_tokenizer: Any | None = None
    if model_class is None:
        _LOAD_PATH_TRACE.set("unknown")
        model, fm_tokenizer = _load_model_auto(
            model_name,
            load_kwargs,
            trust_remote_code,
            load_in_4bit=effective_4bit,
        )
        load_path = _LOAD_PATH_TRACE.get()
    else:
        model = model_class.from_pretrained(model_name, **load_kwargs)
        load_path = "explicit_class"

    # --- Diffusion patch (shared across load paths) ---
    _patch_for_diffusion(model, max_seq_length)

    # --- Tokenizer (prefer FastModel's tokenizer; fall back to separate load) ---
    tokenizer = (
        fm_tokenizer
        if fm_tokenizer is not None
        else _load_tokenizer(model_name, trust_remote_code, token)
    )

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        integration=_integration_name_for(model),
        load_path=load_path,
        details={
            # No runtime class mutation exists since #186; the key stays for
            # detail-shape compatibility and is False by construction.
            "class_swapped": False,
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "quantized": bool(find_quantized_linear_modules(model)),
            "tokenizer_from_fastmodel": fm_tokenizer is not None,
        },
    )
