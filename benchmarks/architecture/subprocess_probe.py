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

"""One fresh-process observation for the #184 architecture-contract artifact.

Invoked only by ``capture_contract.py`` (and by the contract tests), one case
per process, so every measurement sees a genuinely fresh interpreter:

    python subprocess_probe.py <case> --out <file> [--json '{...}']

Writes exactly one JSON document to ``--out`` (stdout is unusable: unsloth
prints banners to it during import). Exit codes: 0 = observed, 2 = typed
blocked (the JSON explains why), 3 = import-root violation (the orchestrator
must abort artifact generation — the probe read a different checkout than the
one being characterized).

The snapshot helpers below are deliberately self-contained: importing
``unturtle.diagnostics`` here would run ``unturtle/__init__`` and pollute the
"before" state that the import probes exist to measure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import random
import sys
import warnings

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: sys.modules is huge and mostly stdlib; the contract cares about these.
MODULE_PREFIXES = (
    "unturtle",
    "unsloth",
    "unsloth_zoo",
    "transformers",
    "peft",
    "torch",
    "triton",
    "bitsandbytes",
    "trl",
    "accelerate",
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _filtered_modules() -> dict[str, list[str] | int]:
    tracked = sorted(
        name
        for name in sys.modules
        if any(name == p or name.startswith(p + ".") for p in MODULE_PREFIXES)
    )
    top = sorted({name.split(".")[0] for name in tracked})
    return {"top_level": top, "tracked_count": len(tracked)}


def _torch_state() -> dict:
    if "torch" not in sys.modules:
        return {"imported": False}
    import torch

    state: dict = {
        "imported": True,
        "default_dtype": str(torch.get_default_dtype()),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "checkpoint_fn": f"{torch.utils.checkpoint.checkpoint.__module__}."
        f"{torch.utils.checkpoint.checkpoint.__qualname__}",
        "grad_enabled": bool(torch.is_grad_enabled()),
    }
    return state


def _autoclass_extra_registrations() -> dict:
    if "transformers" not in sys.modules:
        return {"transformers_imported": False}
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_MAPPING,
    )

    def extra(mapping) -> list[str]:
        content = getattr(mapping, "_extra_content", {})
        out = []
        for key, value in content.items():
            key_name = (
                key if isinstance(key, str) else getattr(key, "__name__", str(key))
            )
            out.append(f"{key_name}->{getattr(value, '__name__', str(value))}")
        return sorted(out)

    return {
        "transformers_imported": True,
        "config_mapping_extra": extra(CONFIG_MAPPING),
        "auto_model_extra": extra(MODEL_MAPPING),
        "auto_masked_lm_extra": extra(MODEL_FOR_MASKED_LM_MAPPING),
        "auto_causal_lm_extra": extra(MODEL_FOR_CAUSAL_LM_MAPPING),
    }


def _default_registry_hub_state() -> dict:
    """Observe the default RegistryHub WITHOUT creating it."""
    registry_mod = sys.modules.get("unturtle.registry")
    if registry_mod is None:
        return {"module_imported": False}
    hub = getattr(registry_mod, "_default_hub", None)
    if hub is None:
        return {"module_imported": True, "default_hub_created": False}
    axes = {}
    axis_names = (
        "generation_algorithms",
        "backbone_integrations",
        "processes",
        "training_recipes",
        "conversions",
        "post_training_recipes",
        "methods",
    )
    for axis_name in axis_names:
        registry = getattr(hub, axis_name, None)
        if registry is None:
            axes[axis_name] = None
            continue
        try:
            axes[axis_name] = sorted(registry._known_names())
        except Exception as exc:  # noqa: BLE001 — observation only
            axes[axis_name] = f"<unreadable: {type(exc).__name__}>"
    return {
        "module_imported": True,
        "default_hub_created": True,
        "bootstrapped": bool(getattr(hub, "_bootstrapped", False)),
        "axes": axes,
    }


def _normalize_message(message: str) -> str:
    text = message.replace(str(REPO_ROOT), "<repo>")
    for marker in ("site-packages",):
        index = text.find(marker)
        while index > 0:
            start = text.rfind(" ", 0, index)
            text = text[: start + 1] + "<env>/" + text[index:]
            index = text.find(marker, start + len("<env>/") + index)
            break
    return text[:200]


_OUT_PATH: pathlib.Path | None = None


def _emit(payload: dict) -> None:
    assert _OUT_PATH is not None, "--out not parsed yet"
    _OUT_PATH.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _verify_import_root() -> None:
    """Abort (exit 3) if the imported unturtle is not this checkout's."""
    expected = os.environ.get("UNTURTLE_EXPECTED_ROOT")
    if not expected:
        _emit({"probe_error": "UNTURTLE_EXPECTED_ROOT not set"})
        raise SystemExit(3)
    import unturtle

    actual_root = pathlib.Path(unturtle.__file__).resolve().parents[1]
    if actual_root != pathlib.Path(expected).resolve():
        _emit(
            {
                "probe_error": "import_root_mismatch",
                "expected": str(expected),
                "actual": str(actual_root),
            }
        )
        raise SystemExit(3)


# ---------------------------------------------------------------------------
# case: import
# ---------------------------------------------------------------------------


def probe_import(args: dict) -> dict:
    module_name = args["module"]
    random.seed(12345)
    before = {
        "modules": _filtered_modules(),
        "environ": dict(os.environ),
        "py_random_digest": _digest(str(random.getstate())),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module(module_name)
    _verify_import_root()
    after_environ = dict(os.environ)
    env_added = sorted(set(after_environ) - set(before["environ"]))
    env_changed = sorted(
        key
        for key in set(after_environ) & set(before["environ"])
        if after_environ[key] != before["environ"][key]
    )
    result = {
        "module": module_name,
        "modules_before": before["modules"],
        "modules_after": _filtered_modules(),
        "environ_added_keys": env_added,
        "environ_changed_keys": env_changed,
        "unsloth_env": {
            key: after_environ[key]
            for key in sorted(after_environ)
            if key.startswith("UNSLOTH_")
        },
        "python_random_consumed": _digest(str(random.getstate()))
        != before["py_random_digest"],
        "torch": _torch_state(),
        "autoclass": _autoclass_extra_registrations(),
        "default_registry_hub": _default_registry_hub_state(),
        "warnings": sorted(
            {
                f"{w.category.__name__}: {_normalize_message(str(w.message))}"
                for w in caught
            }
        ),
        "volatile": {
            "torch_cpu_rng_digest": (
                _digest(str(sys.modules["torch"].get_rng_state().numpy().tobytes()))
                if "torch" in sys.modules
                else None
            ),
        },
    }
    return result


# ---------------------------------------------------------------------------
# case: model
# ---------------------------------------------------------------------------

FAMILIES: dict[str, dict] = {
    "dream": {
        "config": "unturtle.models.backbones.dream.configuration_dream:DreamConfig",
        "model": "unturtle.models.backbones.dream.modeling_dream:DreamModel",
    },
    "llada": {
        "config": "unturtle.models.backbones.llada.configuration_llada:LLaDAConfig",
        "model": "unturtle.models.backbones.llada.modeling_llada:LLaDAModelLM",
    },
    "mdlm_dit": {
        "config": "unturtle.models.backbones.mdlm_dit.configuration_mdlm_dit:MDLMDiTConfig",
        "model": "unturtle.models.backbones.mdlm_dit.modeling_mdlm_dit:MDLMDiTForMaskedDiffusionLM",
    },
    "tiny_a2d_llama": {
        "config": "unturtle.models.conversion.a2d.tiny_a2d.modeling_llama:TinyA2DLlamaConfig",
        "model": "unturtle.models.conversion.a2d.tiny_a2d.modeling_llama:TinyA2DLlamaLMHeadModel",
    },
    "tiny_a2d_qwen2": {
        "config": "unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen2:TinyA2DQwen2Config",
        "model": "unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen2:TinyA2DQwen2LMHeadModel",
    },
    "tiny_a2d_qwen3": {
        "config": "unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3:TinyA2DQwen3Config",
        "model": "unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3:TinyA2DQwen3LMHeadModel",
    },
    "modernbert_diffusion": {
        "config": "unturtle.models.backbones.modernbert.configuration:DiffusionModernBertConfig",
        "model": "unturtle.models.backbones.modernbert.modeling:DiffusionModernBertForMaskedLM",
    },
    "diffusion_gemma": {
        # Resolved through the integrations registry wrapper resolver, the
        # same seam production loading uses.
        "wrapper": True,
    },
}

#: The methods whose OWNER (not mere presence) is part of the contract.
OWNED_METHODS = (
    "generate",
    "save_pretrained",
    "from_pretrained",
    "post_init",
    "tie_weights",
    "gradient_checkpointing_enable",
    "_set_gradient_checkpointing",
    "get_input_embeddings",
    "forward",
)


def _resolve(spec: str):
    module_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def probe_model(args: dict) -> dict:
    family = args["family"]
    spec = FAMILIES[family]
    import unturtle.models  # noqa: F401 — fires the AutoConfig registrations

    _verify_import_root()
    from unturtle.diagnostics.architecture import (
        class_fqn,
        describe_method_owner,
        mro_fqns,
    )

    if spec.get("wrapper"):
        from unturtle.models.integrations import post_load_class_swaps

        swaps = post_load_class_swaps()
        try:
            model_cls = swaps["diffusion_gemma"]()
        except Exception as exc:  # noqa: BLE001 — optional upstream dependency
            return {
                "family": family,
                "status": "blocked",
                "reason": f"wrapper resolver failed: {type(exc).__name__}: {exc}"[:200],
            }
        config_cls = getattr(model_cls, "config_class", None)
    else:
        model_cls = _resolve(spec["model"])
        config_cls = _resolve(spec["config"])

    model_type = getattr(config_cls, "model_type", None)
    autoclass = _autoclass_extra_registrations()
    config_registered = any(
        entry.startswith(f"{model_type}->")
        for entry in autoclass.get("config_mapping_extra", [])
    )
    return {
        "family": family,
        "status": "observed",
        "model_class": class_fqn(model_cls),
        "config_class": class_fqn(config_cls) if config_cls else None,
        "declared_config_class": (
            class_fqn(model_cls.config_class)
            if getattr(model_cls, "config_class", None)
            else None
        ),
        "model_type": model_type,
        "base_model_prefix": getattr(model_cls, "base_model_prefix", None),
        "mro": mro_fqns(model_cls),
        "method_owners": {
            name: describe_method_owner(model_cls, name) for name in OWNED_METHODS
        },
        "tied_weights_keys": getattr(model_cls, "_tied_weights_keys", None),
        "autoclass_config_registered": config_registered,
        "autoclass_extra": autoclass,
    }


# ---------------------------------------------------------------------------
# case: public-api
# ---------------------------------------------------------------------------


def probe_public_api(args: dict) -> dict:
    import unturtle
    import unturtle.models

    _verify_import_root()

    def describe(module) -> dict:
        declared = list(getattr(module, "__all__", []))
        symbols = {}
        for name in declared:
            value = getattr(module, name, None)
            if value is None:
                symbols[name] = {"resolved": False}
                continue
            symbols[name] = {
                "resolved": True,
                "module": getattr(value, "__module__", type(value).__module__),
                "qualname": getattr(value, "__qualname__", type(value).__qualname__),
                "kind": type(value).__name__,
            }
        return {"all_count": len(declared), "all": sorted(declared), "symbols": symbols}

    return {
        "status": "observed",
        "unturtle": describe(unturtle),
        "unturtle.models": describe(unturtle.models),
    }


# ---------------------------------------------------------------------------
# case: integrations
# ---------------------------------------------------------------------------


def probe_integrations(args: dict) -> dict:
    import unturtle.models  # noqa: F401 — registrations fire here

    _verify_import_root()
    from unturtle.models.integrations.registry import iter_integrations

    rows = {}
    for integration in iter_integrations():

        def resolve(resolver_name: str, integration=integration) -> dict:
            resolver = getattr(integration, resolver_name, None)
            if resolver is None:
                return {"declared": False}
            try:
                resolved = resolver() if callable(resolver) else resolver
            except Exception as exc:  # noqa: BLE001 — optional deps degrade
                return {
                    "declared": True,
                    "resolved": False,
                    "reason": f"{type(exc).__name__}: {str(exc)[:120]}",
                }
            name = getattr(
                resolved, "__qualname__", getattr(resolved, "__name__", None)
            )
            module = getattr(resolved, "__module__", None)
            return {
                "declared": True,
                "resolved": True,
                "target": f"{module}.{name}" if module else str(name),
            }

        rows[integration.name] = {
            "model_types": sorted(integration.model_types),
            "peft_model_types": sorted(getattr(integration, "peft_model_types", ())),
            "native": resolve("_native_resolver"),
            "wrapper": resolve("_wrapper_resolver"),
            "peft_patcher": resolve("_peft_patcher"),
        }
    return {"status": "observed", "integrations": rows}


# ---------------------------------------------------------------------------
# tiny fixtures shared by the execution probes
# ---------------------------------------------------------------------------


def _tiny_dream_model(dtype=None, seed: int = 0):
    import torch

    from unturtle.models.backbones.dream.configuration_dream import DreamConfig
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    torch.manual_seed(seed)
    config = DreamConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        mask_token_id=1,
        pad_token_id=0,
    )
    model = DreamModel(config)
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                proj.bias.normal_(std=0.2)
    # A directly-instantiated DreamModel carries a plain GenerationConfig,
    # whose missing Dream fields (eps, …) make the unified generate raise —
    # a from_pretrained load installs DreamGenerationConfig. Recorded as a
    # note in the generation section; installed here so the probe measures
    # the load-path contract, not the fresh-instantiation gap.
    from unturtle.models.backbones.dream.generation_utils import (
        DreamGenerationConfig,
    )

    model.generation_config = DreamGenerationConfig(
        mask_token_id=config.mask_token_id, pad_token_id=config.pad_token_id
    )
    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    return model


def _tiny_a2d_llama_model(seed: int = 0):
    import torch

    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    torch.manual_seed(seed)
    config = TinyA2DLlamaConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        mask_token_id=511,
    )
    model = TinyA2DLlamaLMHeadModel(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# case: generation
# ---------------------------------------------------------------------------


def probe_generation(args: dict) -> dict:
    family = args["family"]
    import torch

    _builders = {"dream": _tiny_dream_model, "tiny_a2d_llama": _tiny_a2d_llama_model}
    if family not in _builders:
        return {"family": family, "status": "blocked", "reason": "no tiny builder"}
    model = _builders[family]()
    _verify_import_root()

    from unturtle.registry import ensure_default_hub

    algorithms = sorted(ensure_default_hub().generation_algorithms._known_names())

    # Instrument: count top-level forwards (NFE) and record which candidate
    # sampling entry points actually run.
    invoked: list[str] = []
    candidate_names = [
        name
        for name in dir(type(model))
        if any(
            token in name
            for token in ("diffusion_generate", "_sample", "block_decode", "bd3lm")
        )
        and callable(getattr(type(model), name, None))
        and not name.startswith("__")
    ]

    import functools

    def instrument():
        # functools.wraps is load-bearing: the sampler resolves call
        # signatures and per-method attributes via inspection (the #186
        # "signature guessing" seam), so a wrapper that hides the original
        # signature changes the very dispatch being observed.
        originals = {}
        for name in candidate_names:
            original = getattr(type(model), name)

            def make(name, original):
                @functools.wraps(original)
                def wrapped(self, *a, **k):
                    invoked.append(name)
                    return original(self, *a, **k)

                return wrapped

            originals[name] = original
            setattr(type(model), name, make(name, original))
        return originals

    def deinstrument(originals):
        for name, original in originals.items():
            setattr(type(model), name, original)

    nfe = {"count": 0}
    model.register_forward_pre_hook(
        lambda module, inputs: nfe.__setitem__("count", nfe["count"] + 1)
    )

    torch.manual_seed(0)
    prompt = torch.randint(2, 400, (1, 8))

    requested_steps = 4
    base_kwargs = {
        "max_new_tokens": 8,
        "steps": requested_steps,
        "temperature": 0.0,
        "block_length": 4,
    }
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is not None:
        base_kwargs["mask_token_id"] = int(mask_token_id)

    def explicit_generation_config():
        if family == "dream":
            from unturtle.models.backbones.dream.generation_utils import (
                DreamGenerationConfig,
            )

            return DreamGenerationConfig(
                mask_token_id=mask_token_id, pad_token_id=model.config.pad_token_id
            )
        return None

    def attempt(algorithm, generation_config):
        invoked.clear()
        nfe["count"] = 0
        originals = instrument()
        try:
            torch.manual_seed(0)
            kwargs = dict(base_kwargs)
            if generation_config is not None:
                kwargs["generation_config"] = generation_config
            with torch.no_grad():
                output = model.generate(prompt, algorithm=algorithm, **kwargs)
            shape = list(output.shape) if hasattr(output, "shape") else None
            return {
                "invoked_methods": list(dict.fromkeys(invoked)),
                "nfe": nfe["count"],
                "output_shape": shape,
                "raised": None,
            }
        except Exception as exc:  # noqa: BLE001 — the raise IS the observation
            return {
                "invoked_methods": list(dict.fromkeys(invoked)),
                "nfe": nfe["count"],
                "output_shape": None,
                "raised": f"{type(exc).__name__}: {str(exc)[:160]}",
            }
        finally:
            deinstrument(originals)

    results = {}
    probe_algorithms = ["auto", "mdlm", "block_decode", "bd3lm", "block_ar"]
    for algorithm in probe_algorithms:
        # First the DEFAULT-config path — its failure is itself a contract
        # observation (Dream's _prepare_generation_config routes through
        # DreamGenerationConfig.from_model_config, which crashes on
        # transformers 5.15). Then, if needed, an explicit config to map the
        # actual execution.
        default_run = attempt(algorithm, None)
        row = {
            "status": "observed",
            "reason": None,
            "requested_steps": requested_steps,
            "default_config_run": default_run,
        }
        if default_run["raised"] is not None and "does not" not in str(
            default_run["raised"]
        ):
            row["explicit_config_run"] = attempt(
                algorithm, explicit_generation_config()
            )
        results[algorithm] = row

    generate_owner = None
    for base in type(model).__mro__:
        if "generate" in vars(base):
            generate_owner = f"{base.__module__}.{base.__qualname__}"
            break

    return {
        "family": family,
        "status": "observed",
        "registered_algorithms": algorithms,
        "candidate_methods_instrumented": sorted(candidate_names),
        "generate_owner": generate_owner,
        "per_algorithm": results,
    }


# ---------------------------------------------------------------------------
# case: persistence
# ---------------------------------------------------------------------------


def _forward_logits(model, input_ids):
    import torch

    with torch.no_grad():
        return model(input_ids=input_ids).logits.detach().float()


def _state_dict_summary(model) -> dict:
    return {
        name: (str(tuple(tensor.shape)), str(tensor.dtype))
        for name, tensor in model.state_dict().items()
    }


def _compare_outputs(a, b) -> dict:
    delta = (a - b).abs()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-12))
    # The exact deltas are NOT stable across processes (measured on the
    # native_fp cell: 0.0051 / 0.0081 / 0.0062 for identical code — CPU
    # kernel-path variation), so raw floats are volatile evidence; the
    # semantic contract is the boolean shape of the divergence.
    return {
        "bit_identical": bool((a == b).all()),
        "within_rel_norm_0p05": rel <= 0.05,
        "volatile": {
            "max_abs_delta": float(delta.max()),
            "relative_norm": rel,
        },
    }


def probe_persistence(args: dict) -> dict:
    case = args["case"]
    import tempfile

    import torch

    _verify_import_root()
    from unturtle import FastDiffusionModel
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    torch.manual_seed(0)
    input_ids = torch.randint(2, 400, (2, 12))

    if case == "autoconfig_roundtrip":
        from transformers import AutoConfig

        rows = {}
        for family in ("dream", "llada", "tiny_a2d_llama", "modernbert_diffusion"):
            spec = FAMILIES[family]
            try:
                config_cls = _resolve(spec["config"])
                config = config_cls()
                out_dir = tempfile.mkdtemp()
                config.save_pretrained(out_dir)
                resolved = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
                rows[family] = {
                    "status": "observed",
                    "roundtrip": "ok",
                    "resolved_class": f"{type(resolved).__module__}.{type(resolved).__qualname__}",
                }
            except Exception as exc:  # noqa: BLE001 — the failure IS the observation
                rows[family] = {
                    "status": "observed",
                    "roundtrip": "failed",
                    "raised": f"{type(exc).__name__}: {str(exc)[:160]}",
                }
        return {"case": case, "status": "observed", "families": rows}

    model = _tiny_dream_model(dtype=torch.float32)
    before_logits = _forward_logits(model, input_ids)
    before_summary = _state_dict_summary(model)

    if case == "native_fp":
        out_dir = tempfile.mkdtemp()
        model.save_pretrained(out_dir)
        reloaded, _ = FastDiffusionModel.from_pretrained(
            out_dir,
            max_seq_length=64,
            dtype=torch.float32,
            load_in_4bit=False,
            model_class=DreamModel,
        )
        reloaded = reloaded.eval()
        after_summary = _state_dict_summary(reloaded)
        after_logits = _forward_logits(reloaded, input_ids)
        missing = sorted(set(before_summary) - set(after_summary))
        unexpected = sorted(set(after_summary) - set(before_summary))
        dtype_diffs = sorted(
            key
            for key in set(before_summary) & set(after_summary)
            if before_summary[key][1] != after_summary[key][1]
        )
        first_mismatch = None
        for key in sorted(set(before_summary) & set(after_summary)):
            if not torch.equal(model.state_dict()[key], reloaded.state_dict()[key]):
                first_mismatch = key
                break
        return {
            "case": case,
            "status": "observed",
            "reloaded_class": f"{type(reloaded).__module__}.{type(reloaded).__qualname__}",
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "dtype_diffs": dtype_diffs,
            "first_mismatching_key": first_mismatch,
            "output": _compare_outputs(after_logits, before_logits),
            # identical weights with non-identical outputs points at a
            # non-weight contract difference between the two construction
            # paths — capture the usual suspects.
            "buffer_diffs": sorted(
                name
                for (name, a), (name_b, b) in zip(
                    sorted(model.named_buffers()),
                    sorted(reloaded.named_buffers()),
                    strict=False,
                )
                if name == name_b
                and (a.shape != b.shape or not torch.equal(a.float(), b.float()))
            ),
            "buffer_names_equal": sorted(n for n, _ in model.named_buffers())
            == sorted(n for n, _ in reloaded.named_buffers()),
            "attn_implementation": {
                "direct_instantiation": getattr(
                    model.config, "_attn_implementation", None
                ),
                "from_pretrained": getattr(
                    reloaded.config, "_attn_implementation", None
                ),
            },
            "generation_config_class": {
                "direct_instantiation": type(model.generation_config).__name__,
                "from_pretrained": type(reloaded.generation_config).__name__,
            },
        }

    if case in ("native_peft", "custom_adapter"):
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        extra = {}
        if case == "custom_adapter":
            extra["modules_to_save"] = ["lm_head"]
        torch.manual_seed(3)
        peft_model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=4,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                **extra,
            ),
        )
        torch.manual_seed(7)
        for name, param in peft_model.named_parameters():
            if ".lora_B." in name:
                with torch.no_grad():
                    param.normal_(std=0.5)
        peft_model.eval()
        wrapped_logits = _forward_logits(peft_model, input_ids)
        adapter_dir = tempfile.mkdtemp()
        peft_model.save_pretrained(adapter_dir)

        fresh_base = _tiny_dream_model(dtype=torch.float32)
        reloaded = PeftModel.from_pretrained(fresh_base, adapter_dir).eval()
        after_logits = _forward_logits(reloaded, input_ids)
        adapter_keys_before = sorted(
            name for name in peft_model.state_dict() if "lora_" in name
        )
        adapter_keys_after = sorted(
            name for name in reloaded.state_dict() if "lora_" in name
        )
        return {
            "case": case,
            "status": "observed",
            "reloaded_class": f"{type(reloaded).__module__}.{type(reloaded).__qualname__}",
            "adapter_key_count": len(adapter_keys_before),
            "adapter_keys_equal": adapter_keys_before == adapter_keys_after,
            "output": _compare_outputs(after_logits, wrapped_logits),
        }

    if case == "generation_reload":
        out_dir = tempfile.mkdtemp()
        model.save_pretrained(out_dir)
        reloaded, _ = FastDiffusionModel.from_pretrained(
            out_dir,
            max_seq_length=64,
            dtype=torch.float32,
            load_in_4bit=False,
            model_class=DreamModel,
        )
        reloaded = reloaded.eval()
        prompt = torch.randint(2, 400, (1, 8))
        from unturtle.models.backbones.dream.generation_utils import (
            DreamGenerationConfig,
        )

        def sample(m, explicit_config: bool):
            torch.manual_seed(11)
            kwargs = {}
            if explicit_config:
                kwargs["generation_config"] = DreamGenerationConfig(
                    mask_token_id=m.config.mask_token_id,
                    pad_token_id=m.config.pad_token_id,
                )
            with torch.no_grad():
                return m.generate(
                    prompt,
                    algorithm="mdlm",
                    max_new_tokens=8,
                    steps=4,
                    temperature=0.0,
                    **kwargs,
                )

        # default-config behavior on the reloaded model is itself a contract
        # observation (save_pretrained round-trips the generation config
        # through plain GenerationConfig).
        default_reload_raised = None
        try:
            sample(reloaded, explicit_config=False)
        except Exception as exc:  # noqa: BLE001
            default_reload_raised = f"{type(exc).__name__}: {str(exc)[:160]}"

        tokens_before = sample(model, explicit_config=True)
        tokens_after = sample(reloaded, explicit_config=True)
        return {
            "case": case,
            "status": "observed",
            "tokens_equal": bool(torch.equal(tokens_before, tokens_after)),
            "generation_config_class_after_reload": type(
                reloaded.generation_config
            ).__name__,
            "default_config_reload_raised": default_reload_raised,
            "volatile": {
                "tokens_before_digest": _digest(str(tokens_before.tolist())),
                "tokens_after_digest": _digest(str(tokens_after.tolist())),
            },
        }

    return {"case": case, "status": "blocked", "reason": f"unknown case {case!r}"}


# ---------------------------------------------------------------------------
# case: process-global
# ---------------------------------------------------------------------------


def probe_process_global(args: dict) -> dict:
    case = args["case"]
    import torch

    _verify_import_root()

    if case == "sdpa":
        backends = {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math": bool(torch.backends.cuda.math_sdp_enabled()),
            "cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        }
        return {
            "case": case,
            "status": "observed",
            "available_backends": backends,
            "tf32": {
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "allow_bf16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                ),
            },
            "deterministic": {
                "deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            },
            "environment": {
                key: os.environ.get(key)
                for key in ("CUBLAS_WORKSPACE_CONFIG", "TORCH_CUDNN_SDPA_ENABLED")
            },
            "policy": (
                "numeric parity cells that route through "
                "scaled_dot_product_attention must either pin the backend "
                "(torch.nn.attention.sdpa_kernel) or compare below attention "
                "at the unit level; the backend a process happens to select "
                "is not a contract value (#187: layer-0 attention output sum "
                "flipped 33.83/22.68 on identical inputs)."
            ),
        }

    if case == "rng_contract":
        from unturtle import FastDiffusionModel
        from unturtle.diagnostics.architecture import tensor_digest

        def wrap_and_digest(pre_consume: int) -> dict:
            model = _tiny_dream_model(dtype=torch.float32)
            torch.manual_seed(100)
            if pre_consume:
                torch.randn(pre_consume)
            rng_before = _digest(str(torch.get_rng_state().numpy().tobytes()))
            peft_model = FastDiffusionModel.get_peft_model(
                model,
                r=4,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj"],
                use_gradient_checkpointing=False,
                random_state=3407,
            )
            rng_after = _digest(str(torch.get_rng_state().numpy().tobytes()))
            lora_a = None
            for name, param in peft_model.named_parameters():
                if ".lora_A." in name:
                    lora_a = tensor_digest(param)
                    break
            return {
                "rng_before": rng_before,
                "rng_after": rng_after,
                "rng_consumed": rng_before != rng_after,
                "lora_A_digest": lora_a,
            }

        run_a = wrap_and_digest(pre_consume=0)
        run_b = wrap_and_digest(pre_consume=7)
        return {
            "case": case,
            "status": "observed",
            "random_state_argument": 3407,
            "run_without_preconsumption": run_a,
            "run_with_preconsumption": run_b,
            "same_random_state_same_adapters": run_a["lora_A_digest"]
            == run_b["lora_A_digest"],
            "caller_rng_untouched_by_wrap": (
                not run_a["rng_consumed"] and not run_b["rng_consumed"]
            ),
            # measured, not asserted: the row says "known_defect" only while the
            # adapters actually differ (#188 fix: seeded, forked RNG)
            "classification": (
                "deterministic_by_random_state"
                if run_a["lora_A_digest"] == run_b["lora_A_digest"]
                else "known_defect"
            ),
            "linked_issue": 188,
        }

    return {"case": case, "status": "blocked", "reason": f"unknown case {case!r}"}


# ---------------------------------------------------------------------------
# case: registry-hub — the standalone RegistryHub contract (#184 blocker):
# an explicit hub must be constructible empty and side-effect-free, builtin
# bootstrap must be deterministic in content AND order, re-bootstrap behavior
# is frozen as observed, and two hubs must not share backing storage. This is
# the foundation for #185/#186 and external plugins using supplied hubs
# instead of the process default.
# ---------------------------------------------------------------------------

_HUB_AXES = (
    "generation_algorithms",
    "backbone_integrations",
    "processes",
    "training_recipes",
    "conversions",
    "post_training_recipes",
    "methods",
)


def probe_registry_hub(args: dict) -> dict:
    import types as types_mod

    import torch

    import unturtle.registry as registry_mod

    _verify_import_root()
    from unturtle.registry import RegistryHub, bootstrap_builtin_hub

    def ordered_axis_names(hub) -> dict:
        return {
            axis: [value.name for value in getattr(hub, axis).values()]
            for axis in _HUB_AXES
        }

    def default_hub_snapshot():
        hub = registry_mod._default_hub
        return None if hub is None else ordered_axis_names(hub)

    def surroundings():
        random.seed(2025)
        torch.manual_seed(2025)
        return {
            "default_hub": default_hub_snapshot(),
            "autoclass": _autoclass_extra_registrations(),
            "environ": dict(os.environ),
            "py_rng": _digest(str(random.getstate())),
            "torch_rng": _digest(str(torch.get_rng_state().numpy().tobytes())),
        }

    def surroundings_delta(before) -> dict:
        after_environ = dict(os.environ)
        return {
            "default_hub_changed": default_hub_snapshot() != before["default_hub"],
            "autoclass_changed": _autoclass_extra_registrations()
            != before["autoclass"],
            "environ_changed_keys": sorted(
                key
                for key in set(after_environ) | set(before["environ"])
                if after_environ.get(key) != before["environ"].get(key)
            ),
            "python_rng_consumed": _digest(str(random.getstate())) != before["py_rng"],
            "torch_rng_consumed": _digest(str(torch.get_rng_state().numpy().tobytes()))
            != before["torch_rng"],
        }

    # -- cell 1: fresh_empty_hub -------------------------------------------
    before = surroundings()
    empty_hub = RegistryHub()
    cell_fresh = {
        "status": "observed",
        "axis_names": ordered_axis_names(empty_hub),
        "all_axes_empty": all(
            not getattr(empty_hub, axis)._items for axis in _HUB_AXES
        ),
        "bootstrapped_flag": bool(empty_hub._bootstrapped),
        "surroundings": surroundings_delta(before),
    }

    # -- cell 2: explicit_builtin_bootstrap ----------------------------------
    before = surroundings()
    hub_b = bootstrap_builtin_hub(RegistryHub())
    order_b = ordered_axis_names(hub_b)
    hub_c = bootstrap_builtin_hub(RegistryHub())
    order_c = ordered_axis_names(hub_c)
    cell_bootstrap = {
        "status": "observed",
        "ordered_axis_names": order_b,
        "deterministic_across_two_bootstraps": order_b == order_c,
        "bootstrapped_flag": bool(hub_b._bootstrapped),
        "surroundings": surroundings_delta(before),
    }

    # -- cell 3: repeat_bootstrap (frozen as observed) -----------------------
    before_counts = {axis: len(getattr(hub_b, axis)._items) for axis in _HUB_AXES}
    try:
        bootstrap_builtin_hub(hub_b)
        raised = None
    except Exception as exc:  # noqa: BLE001 — the raise IS the observation
        raised = f"{type(exc).__name__}: {str(exc)[:160]}"
    after_counts = {axis: len(getattr(hub_b, axis)._items) for axis in _HUB_AXES}
    cell_repeat = {
        "status": "observed",
        "raised": raised,
        "behavior": "duplicate_rejection" if raised else "idempotent_or_duplicated",
        "axis_counts_unchanged": before_counts == after_counts,
    }

    # -- cell 4: hub_isolation ------------------------------------------------
    before = surroundings()
    hub_a = RegistryHub()
    sentinel = types_mod.SimpleNamespace(name="probe-sentinel-a")
    hub_a.processes.register(sentinel)
    try:
        hub_a.processes.register(types_mod.SimpleNamespace(name="probe-sentinel-a"))
        duplicate_raised = None
    except Exception as exc:  # noqa: BLE001
        duplicate_raised = f"{type(exc).__name__}: {str(exc)[:120]}"
    default_hub = registry_mod._default_hub
    cell_isolation = {
        "status": "observed",
        "sentinel_visible_in_registering_hub": hub_a.processes.find("probe-sentinel-a")
        is sentinel,
        "sentinel_leaked_to_other_hub": hub_b.processes.find("probe-sentinel-a")
        is not None,
        "sentinel_leaked_to_default_hub": (
            default_hub is not None
            and default_hub.processes.find("probe-sentinel-a") is not None
        ),
        "registry_objects_shared": any(
            getattr(hub_a, axis) is getattr(hub_b, axis) for axis in _HUB_AXES
        ),
        "backing_storage_shared": any(
            getattr(hub_a, axis)._items is getattr(hub_b, axis)._items
            for axis in _HUB_AXES
        ),
        "duplicate_registration_raised": duplicate_raised,
        "surroundings": surroundings_delta(before),
    }

    return {
        "status": "observed",
        "fresh_empty_hub": cell_fresh,
        "explicit_builtin_bootstrap": cell_bootstrap,
        "repeat_bootstrap": cell_repeat,
        "hub_isolation": cell_isolation,
    }


# ---------------------------------------------------------------------------
# case: fourbit-contract (CUDA)
# ---------------------------------------------------------------------------


def probe_fourbit_contract(args: dict) -> dict:
    import tempfile

    import torch

    _verify_import_root()
    if not torch.cuda.is_available():
        return {"status": "blocked", "reason": "CUDA unavailable in this process"}

    from unturtle import FastDiffusionModel
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    checkpoint = tempfile.mkdtemp()
    _tiny_dream_model(dtype=torch.bfloat16).save_pretrained(checkpoint)

    def load():
        model, _ = FastDiffusionModel.from_pretrained(
            checkpoint,
            max_seq_length=64,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            device_map={"": "cuda:0"},
            model_class=DreamModel,
        )
        return model

    env_before = os.environ.get("UNSLOTH_MIXED_PRECISION")
    checkpoint_fn_before = (
        f"{torch.utils.checkpoint.checkpoint.__module__}."
        f"{torch.utils.checkpoint.checkpoint.__qualname__}"
    )

    # preparation owner: which implementation unturtle.save delegates to
    try:
        from unsloth.models._utils import (  # noqa: F401
            prepare_model_for_kbit_training as _unsloth_prepare,
        )

        preparation_owner = "unsloth.models._utils.prepare_model_for_kbit_training"
    except Exception:  # noqa: BLE001
        preparation_owner = "peft.prepare_model_for_kbit_training (fallback)"

    model = load()
    from unturtle.fast_diffusion_model import _original_apply_qkv
    from unturtle.kernels.fast_lora import (
        apply_lora_mlp_swiglu,
        apply_lora_o,
        apply_lora_qkv_with_bias,
    )

    attn0 = model.model.layers[0].self_attn
    before_identity = {
        "apply_qkv": getattr(getattr(attn0, "apply_qkv", None), "__name__", None),
        "instance_forward": "forward" in attn0.__dict__,
    }

    peft_model = FastDiffusionModel.get_peft_model(
        model,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing=False,
    )
    layers = peft_model.base_model.model.model.layers
    attn0p = layers[0].self_attn
    after_identity = {
        "apply_qkv_is_bias_kernel": attn0p.apply_qkv is apply_lora_qkv_with_bias,
        "apply_qkv_is_original": attn0p.apply_qkv is _original_apply_qkv,
        "apply_o_is_fast": attn0p.apply_o is apply_lora_o,
        "mlp_forward_is_fast": getattr(layers[0].mlp.forward, "__func__", None)
        is apply_lora_mlp_swiglu,
        "instance_forward_installed": "forward" in attn0p.__dict__,
    }

    from collections import Counter

    dtype_histogram = dict(Counter(str(p.dtype) for p in peft_model.parameters()))
    adapter_dtypes = sorted(
        {str(p.dtype) for n, p in peft_model.named_parameters() if ".lora_" in n}
    )
    embed_dtype = str(peft_model.get_input_embeddings().weight.dtype)

    # forward/backward liveness — the #177 contract: hooks are not evidence.
    peft_model.train()
    ids = torch.randint(2, 400, (2, 12), device="cuda:0")
    out = peft_model(input_ids=ids)
    loss = out.logits.float().square().mean()
    loss.backward()
    grads = sum(
        1
        for n, p in peft_model.named_parameters()
        if ".lora_" in n and p.grad is not None
    )
    peft_model.zero_grad(set_to_none=True)

    env_after = os.environ.get("UNSLOTH_MIXED_PRECISION")
    checkpoint_fn_after = (
        f"{torch.utils.checkpoint.checkpoint.__module__}."
        f"{torch.utils.checkpoint.checkpoint.__qualname__}"
    )

    # fallback behavior: fp32-upcast a fresh load, expect uniform skip
    damaged = load()
    for param in damaged.parameters():
        if (
            param.dtype in (torch.bfloat16, torch.float16)
            and type(param).__name__ != "Params4bit"
        ):
            param.data = param.data.to(torch.float32)
    damaged_peft = FastDiffusionModel.get_peft_model(
        damaged,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing=False,
    )
    damaged_layers = damaged_peft.base_model.model.model.layers
    damaged_hooks = {
        "any_qkv_fast": any(
            getattr(layer.self_attn, "apply_qkv", None) is apply_lora_qkv_with_bias
            for layer in damaged_layers
        ),
        "any_o_fast": any(
            getattr(layer.self_attn, "apply_o", None) is apply_lora_o
            for layer in damaged_layers
        ),
        "any_mlp_fast": any(
            getattr(layer.mlp.forward, "__func__", None) is apply_lora_mlp_swiglu
            for layer in damaged_layers
        ),
        "any_instance_forward": any(
            "forward" in layer.self_attn.__dict__ for layer in damaged_layers
        ),
    }
    damaged_peft.train()
    damaged_out = damaged_peft(input_ids=ids)
    damaged_out.logits.float().square().mean().backward()
    damaged_grads = sum(
        1
        for n, p in damaged_peft.named_parameters()
        if ".lora_" in n and p.grad is not None
    )

    return {
        "status": "observed",
        "preparation_owner": preparation_owner,
        "frozen_parameter_dtypes": dtype_histogram,
        "trainable_adapter_dtypes": adapter_dtypes,
        "embedding_dtype": embed_dtype,
        "mutation_identity": {
            "before": before_identity,
            "after": after_identity,
        },
        "fast_path_verdict": "compatible"
        if after_identity["apply_qkv_is_bias_kernel"]
        else "not_fast",
        "forward_backward_liveness": {
            "loss_finite": bool(torch.isfinite(loss)),
            "lora_grads": grads,
        },
        "fallback_behavior": {
            "fp32_upcast_hooks": damaged_hooks,
            "uniform_skip": not any(damaged_hooks.values()),
            "standard_path_lora_grads": damaged_grads,
            "verdict": "incompatible_compute_dtype",
        },
        "dtype_gate_tristate_note": (
            "the production gate is effectively three-valued: compatible, "
            "incompatible_compute_dtype, and fail-open when embedding "
            "structure is unresolvable; the fail-open branch is recorded as "
            "status=unverified reason=input_embedding_unresolvable in the "
            "verdicts section, never as compatible (carry-forward from #177, "
            "requirement for #185's SupportResult)"
        ),
        "unsloth_environment_mutation": {
            "UNSLOTH_MIXED_PRECISION_before": env_before,
            "UNSLOTH_MIXED_PRECISION_after": env_after,
            "checkpoint_fn_before": checkpoint_fn_before,
            "checkpoint_fn_after": checkpoint_fn_after,
            "scope": "process_global",
            "note": (
                "prepare_model_for_kbit_training (per-model API) mutates "
                "process state; classified separately from object-local "
                "mutations"
            ),
        },
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

CASES = {
    "import": probe_import,
    "model": probe_model,
    "public-api": probe_public_api,
    "integrations": probe_integrations,
    "generation": probe_generation,
    "persistence": probe_persistence,
    "process-global": probe_process_global,
    "registry-hub": probe_registry_hub,
    "fourbit-contract": probe_fourbit_contract,
}


def main() -> None:
    global _OUT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--out", required=True)
    parser.add_argument("--json", default="{}")
    args = parser.parse_args()
    _OUT_PATH = pathlib.Path(args.out)
    payload = json.loads(args.json)
    try:
        result = CASES[args.case](payload)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — typed blocked, not a stacktrace dump
        _emit(
            {
                "probe": args.case,
                "args": payload,
                "status": "blocked",
                "reason": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
        )
        raise SystemExit(2) from exc
    result.setdefault("probe", args.case)
    result.setdefault("args", payload)
    _emit(result)


if __name__ == "__main__":
    main()
