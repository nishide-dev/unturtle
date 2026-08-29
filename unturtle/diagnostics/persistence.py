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

"""Observation-only persistence diagnostics (#174).

Never imported by production code. Provides:

- :func:`compare_tensors` — a state-attributed comparison report (first
  mismatching key, shape/dtype/device, max abs/rel delta, non-equal count,
  bit-identity after CPU serialization) so a parity assertion never fails
  as a bare ``assert False``;
- :func:`buffer_census` — every non-persistent buffer with digest/finiteness,
  which is where transformers' ``from_pretrained`` leaves uninitialized
  memory when a family's ``_init_weights`` does not re-initialize them;
- :func:`classify_rope_attribution` — the #174 decision gate as a PURE
  function over measured digests (never over reported flags), so the causal
  mutants (claimed-but-not-performed restoration, arm-1-vs-arm-1, unpinned
  SDPA, persistent-weight drift) cannot produce a CAUSAL verdict;
- :func:`process_state_snapshot` — the process-global state the #184 ledger
  identified as leak candidates.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

VERDICT_CAUSAL = "ROPE LOAD-PATH CAUSAL"
VERDICT_CONTRIBUTORY = "ROPE CONTRIBUTORY, NOT SUFFICIENT"
VERDICT_NOT_CAUSAL = "ROPE NOT CAUSAL"
VERDICT_NO_DIVERGENCE = "NO DIVERGENCE UNDER THIS CONDITION"
VERDICT_INADMISSIBLE = "INADMISSIBLE COMPARISON"
VERDICT_PERSISTENT_DRIFT = "PERSISTENT WEIGHTS DIFFER"


def tensor_digest(tensor: Any) -> str:
    import torch

    with torch.no_grad():
        data = tensor.detach().to("cpu").contiguous()
        if data.dtype in (torch.bfloat16, torch.float16):
            data = data.to(torch.float32)
    payload = f"{tuple(data.shape)}|{data.dtype}".encode() + data.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def state_dict_digest(model: Any) -> tuple[str, dict[str, str]]:
    """Digest of the PERSISTENT state (what ``save_pretrained`` writes)."""
    per_key = {k: tensor_digest(v) for k, v in sorted(model.state_dict().items())}
    whole = hashlib.sha256("".join(f"{k}={v};" for k, v in per_key.items()).encode())
    return whole.hexdigest()[:16], per_key


def compare_tensors(a: Any, b: Any, *, label: str = "tensor") -> dict[str, Any]:
    """State-attributed comparison of two tensors (never a bare boolean)."""
    import torch

    a_cpu = a.detach().to("cpu")
    b_cpu = b.detach().to("cpu")
    report: dict[str, Any] = {
        "label": label,
        "shape": [list(a_cpu.shape), list(b_cpu.shape)],
        "dtype": [str(a.dtype), str(b.dtype)],
        "device": [str(a.device), str(b.device)],
    }
    if a_cpu.shape != b_cpu.shape:
        report.update(equal=False, reason="shape mismatch")
        return report
    af, bf = a_cpu.to(torch.float64), b_cpu.to(torch.float64)
    delta = (af - bf).abs()
    non_equal = ~torch.isclose(af, bf, rtol=0.0, atol=0.0, equal_nan=True)
    first_index = None
    if bool(non_equal.any()):
        flat = int(non_equal.flatten().nonzero()[0].item())
        first_index = list(
            int(i) for i in torch.unravel_index(torch.tensor(flat), a_cpu.shape)
        )
    finite = bool(torch.isfinite(af).all() and torch.isfinite(bf).all())
    scale = bf.abs().clamp_min(1e-12)
    # bit identity after a CPU serialization round-trip (what save/load sees)
    buf_a, buf_b = io.BytesIO(), io.BytesIO()
    torch.save(a_cpu, buf_a)
    torch.save(b_cpu, buf_b)
    report.update(
        equal=bool(torch.equal(a_cpu, b_cpu)),
        max_abs_delta=float(delta.max()) if finite else None,
        max_rel_delta=float((delta / scale).max()) if finite else None,
        non_equal_count=int(non_equal.sum()),
        first_differing_index=first_index,
        any_nan=bool(torch.isnan(af).any() or torch.isnan(bf).any()),
        any_inf=bool(torch.isinf(af).any() or torch.isinf(bf).any()),
        bit_identical_after_cpu_serialization=buf_a.getvalue() == buf_b.getvalue(),
    )
    return report


def first_state_dict_mismatch(a: Any, b: Any) -> dict[str, Any] | None:
    import torch

    sa, sb = a.state_dict(), b.state_dict()
    for key in sorted(set(sa) | set(sb)):
        if key not in sa or key not in sb:
            return {"key": key, "reason": "missing on one side"}
        if not torch.equal(sa[key].to("cpu"), sb[key].to("cpu")):
            return {"key": key, **compare_tensors(sa[key], sb[key], label=key)}
    return None


def _rotary_formula(module: Any, buffer: Any) -> Any | None:
    """Analytic inv_freq for the two Unturtle rotary implementations, when
    the module exposes enough to recompute it; None otherwise."""
    import torch

    dim = buffer.numel() * 2
    base = getattr(module, "base", None)
    if base is None:
        config = getattr(module, "config", None)
        base = getattr(config, "rope_theta", None) if config is not None else None
    if base is None:
        base = 10_000
    try:
        return 1.0 / (
            float(base) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
    except Exception:  # noqa: BLE001 — best-effort reference
        return None


def buffer_census(model: Any, *, reference: Any | None = None) -> dict[str, dict]:
    """Every non-persistent buffer: shape/dtype/device/digest/finiteness, plus
    equality against the same-named buffer of ``reference`` and against the
    analytic rotary formula where recognisable."""
    import torch

    ref_buffers = dict(reference.named_buffers()) if reference is not None else {}
    modules = dict(model.named_modules())
    census: dict[str, dict] = {}
    for name, buffer in model.named_buffers():
        owner_name, _, attr = name.rpartition(".")
        owner = modules.get(owner_name, model)
        if attr not in getattr(owner, "_non_persistent_buffers_set", set()):
            continue
        entry: dict[str, Any] = {
            "shape": list(buffer.shape),
            "dtype": str(buffer.dtype),
            "device": str(buffer.device),
            "digest": tensor_digest(buffer),
            "finite": bool(torch.isfinite(buffer).all()),
        }
        if name in ref_buffers:
            entry["equals_reference"] = bool(
                torch.equal(buffer.to("cpu"), ref_buffers[name].to("cpu"))
            )
        if attr == "inv_freq":
            formula = _rotary_formula(owner, buffer)
            entry["equals_formula"] = (
                bool(torch.allclose(buffer.to("cpu", torch.float32), formula))
                if formula is not None
                else None
            )
        census[name] = entry
    return census


def classify_rope_attribution(
    *,
    arms: dict[str, dict[str, Any]],
    sdpa_backend: str | None,
) -> dict[str, Any]:
    """The #174 decision gate, computed ONLY from measured fields.

    ``arms`` maps ``original`` / ``direct_state_dict`` / ``reload`` /
    ``reload_restored`` to records carrying:

    - ``load_path`` (label), ``object_id``
    - ``persistent_digest``
    - ``buffers``: {name: {"digest": ..., "equals_reference": ...}}
    - ``output_vs_original``: compare_tensors report (``equal``, ``max_abs_delta``)

    Reported flags such as "restoration performed" are deliberately NOT
    inputs: restoration is verified from buffer digests, divergence from
    output reports, and the reload arm must be a distinct object produced by
    the load path — an arm-1-vs-arm-1 comparison is inadmissible.
    """
    reasons: list[str] = []
    original = arms.get("original")
    reload = arms.get("reload")
    restored = arms.get("reload_restored")
    if not (original and reload and restored):
        return {"verdict": VERDICT_INADMISSIBLE, "reasons": ["missing arms"]}
    if sdpa_backend is None:
        reasons.append("SDPA backend not pinned; process-bistable backend can mask NaN")
        return {"verdict": VERDICT_INADMISSIBLE, "reasons": reasons}
    if reload.get("load_path") != "from_pretrained":
        reasons.append("reload arm did not come from the load path")
    if reload.get("object_id") == original.get("object_id"):
        reasons.append("reload arm is the original object (arm 1 vs arm 1)")
    if reasons:
        return {"verdict": VERDICT_INADMISSIBLE, "reasons": reasons}

    if reload.get("persistent_digest") != original.get("persistent_digest"):
        return {
            "verdict": VERDICT_PERSISTENT_DRIFT,
            "reasons": ["persistent state dict differs after reload"],
        }

    changed = sorted(
        name
        for name, entry in reload.get("buffers", {}).items()
        if entry.get("equals_reference") is False
    )
    reload_equal = bool(reload["output_vs_original"].get("equal"))
    if reload_equal:
        return {
            "verdict": VERDICT_NO_DIVERGENCE,
            "reasons": [
                "reload output equals original under this backend/poison "
                "condition; nothing to attribute"
            ],
            "buffers_changed": changed,
        }
    if not changed:
        return {
            "verdict": VERDICT_NOT_CAUSAL,
            "reasons": ["reload diverges but no non-persistent buffer differs"],
            "buffers_changed": [],
        }
    # restoration must be VERIFIED on every changed buffer, from digests
    restored_buffers = restored.get("buffers", {})
    fully_restored = all(
        restored_buffers.get(name, {}).get("equals_reference") is True
        for name in changed
    )
    restored_equal = bool(restored["output_vs_original"].get("equal"))
    reload_delta = reload["output_vs_original"].get("max_abs_delta")
    restored_delta = restored["output_vs_original"].get("max_abs_delta")
    if fully_restored and restored_equal:
        return {
            "verdict": VERDICT_CAUSAL,
            "reasons": [
                "persistent weights identical; reload diverges; restoring every "
                "changed non-persistent buffer removes the divergence"
            ],
            "buffers_changed": changed,
        }
    if not fully_restored:
        return {
            "verdict": VERDICT_CONTRIBUTORY,
            "reasons": [
                "not every changed buffer was verifiably restored; cannot claim "
                "full attribution"
            ],
            "buffers_changed": changed,
        }
    reduced = (
        reload_delta is not None
        and restored_delta is not None
        and restored_delta < reload_delta
    ) or (
        reload["output_vs_original"].get("any_nan")
        and not restored["output_vs_original"].get("any_nan")
    )
    if reduced:
        return {
            "verdict": VERDICT_CONTRIBUTORY,
            "reasons": ["restoration reduces but does not remove the divergence"],
            "buffers_changed": changed,
        }
    return {
        "verdict": VERDICT_NOT_CAUSAL,
        "reasons": ["restoring the changed buffers has no effect on the divergence"],
        "buffers_changed": changed,
    }


def process_state_snapshot(*, config_type: str | None = None) -> dict[str, Any]:
    """The #184 leak candidates, in one dict (digests, never raw states)."""
    import os
    import random

    import torch

    snapshot: dict[str, Any] = {
        "default_dtype": str(torch.get_default_dtype()),
        "cpu_rng_digest": hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest()[:16],
        "python_rng_digest": hashlib.sha256(
            str(random.getstate()).encode()
        ).hexdigest()[:16],
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_cpu_enabled": bool(torch.is_autocast_enabled("cpu")),
        "sdpa_flags": {
            "flash": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math": bool(torch.backends.cuda.math_sdp_enabled()),
            "cudnn": bool(torch.backends.cuda.cudnn_sdp_enabled()),
        },
        "unsloth_env": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith("UNSLOTH_")
        },
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        snapshot["cuda_rng_digest"] = hashlib.sha256(
            torch.cuda.get_rng_state().numpy().tobytes()
        ).hexdigest()[:16]
    if config_type is not None and "transformers" in __import__("sys").modules:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        extra = getattr(CONFIG_MAPPING, "_extra_content", {})
        snapshot["autoclass_owner_for_config_type"] = (
            getattr(extra.get(config_type), "__name__", None)
            if config_type in extra
            else None
        )
    registry_mod = __import__("sys").modules.get("unturtle.registry")
    hub = getattr(registry_mod, "_default_hub", None) if registry_mod else None
    if hub is not None:
        snapshot["default_hub"] = {
            axis: [v.name for v in getattr(hub, axis).values()]
            for axis in (
                "generation_algorithms",
                "backbone_integrations",
                "processes",
                "training_recipes",
                "conversions",
                "post_training_recipes",
                "methods",
            )
            if hasattr(hub, axis)
        }
    else:
        snapshot["default_hub"] = None
    return snapshot


def instance_patches(model: Any) -> dict[str, list[str]]:
    """Instance-level ``forward`` / ``generate`` / ``apply_*`` overrides —
    the runtime mutations the #184 ledger tracks, as seen on THIS object."""
    patched: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        hits = [
            attr
            for attr in ("forward", "generate", "apply_qkv", "apply_o", "apply_mlp")
            if attr in module.__dict__
        ]
        if hits:
            patched[name or "<root>"] = hits
    return patched


def _config_diff_keys(a: Any, b: Any) -> list[str] | None:
    if not (hasattr(a, "to_dict") and hasattr(b, "to_dict")):
        return None
    da, db = a.to_dict(), b.to_dict()
    return sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))


def persistence_parity_report(original: Any, reloaded: Any, ref: Any, got: Any) -> str:
    """One-string, state-attributed report for a save/reload parity assertion.

    Evaluated only when the assertion fails (it is the assert message), so the
    passing path pays nothing.
    """
    import json

    report = {
        "output": compare_tensors(ref, got, label="logits"),
        "first_persistent_mismatch": first_state_dict_mismatch(original, reloaded),
        "non_persistent_buffers_after_reload": buffer_census(
            reloaded, reference=original
        ),
        "classes": {
            "original": f"{type(original).__module__}.{type(original).__qualname__}",
            "reloaded": f"{type(reloaded).__module__}.{type(reloaded).__qualname__}",
        },
        "mro_equal": type(original).__mro__ == type(reloaded).__mro__,
        "config_diff_keys": _config_diff_keys(
            getattr(original, "config", None), getattr(reloaded, "config", None)
        ),
        "instance_patches": {
            "original": instance_patches(original),
            "reloaded": instance_patches(reloaded),
        },
        "process_state": process_state_snapshot(
            config_type=getattr(getattr(original, "config", None), "model_type", None)
        ),
    }
    return "save/reload parity failed (#174):\n" + json.dumps(
        report, indent=2, sort_keys=True, default=str
    )
