"""Differential contract for the Tiny-A2D fast-path provider (#185).

``oracle_patch_a2d_peft`` is the *verbatim* ``_patch_a2d_peft`` that lived in
the façade on ``main`` (d72b15b) before extraction — renamed only. Everything
else in the PEFT path is held constant; the two runs differ solely in which
patcher runs, so any divergence is the extraction's. Bit-identical outputs are
required, not tolerances.
"""

from __future__ import annotations

import types
from typing import Any, Literal

import pytest
import torch

from unturtle.models.conversion.a2d.tiny_a2d._fast_forward import (
    TinyA2DAttention_fast_forward,
)
from unturtle.models.integrations import fast_path_support as fps
from unturtle.models.integrations.fast_path_support import (
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv,
)

_require_fast_lora = fps.require_fast_lora
_warn_once = fps.warn_once
_no_bias = fps.no_bias
_no_lora_mag = fps.no_lora_magnitude


# --- ORACLE: verbatim pre-extraction implementation (main @ d72b15b) ----------
def oracle_patch_a2d_peft(
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


# -----------------------------------------------------------------------------

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ALL_PROJ = tuple(TARGETS)


def _tiny_a2d(seed: int = 0, dtype=torch.bfloat16):
    from unturtle.models.conversion.a2d.tiny_a2d import (
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
        max_position_embeddings=64,
    )
    return TinyA2DLlamaLMHeadModel(config).to(dtype)


def _wrap(model, monkeypatch, patcher=None, device="cuda"):
    """PEFT-wrap through the façade; optionally swap the patcher (oracle)."""
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.conversion.a2d.tiny_a2d import fast_paths

    if patcher is not None:
        monkeypatch.setattr(fast_paths, "patch_peft", patcher)
    else:
        monkeypatch.undo()
    return FastDiffusionModel.get_peft_model_with_report(
        model.to(device),
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGETS,
        use_gradient_checkpointing=False,
        random_state=1234,
    )


def _fast_identity(model) -> dict[str, tuple[bool, bool, bool, bool]]:
    out = {}
    for idx, layer in enumerate(model.base_model.model.model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        out[str(idx)] = (
            getattr(attn.__dict__.get("forward"), "__func__", None)
            is TinyA2DAttention_fast_forward,
            attn.__dict__.get("apply_qkv") is apply_lora_qkv,
            attn.__dict__.get("apply_o") is apply_lora_o,
            getattr(mlp.__dict__.get("forward"), "__func__", None)
            is apply_lora_mlp_swiglu,
        )
    return out


def _state(model):
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()}


def _hub_snapshot():
    """Names registered on every axis of the process default hub (None = unbuilt)."""
    import unturtle.registry as registry_mod

    hub = registry_mod._default_hub
    if hub is None:
        return None
    return {
        axis: tuple(v.name for v in getattr(hub, axis).values())
        for axis in (
            "generation_algorithms",
            "backbone_integrations",
            "processes",
            "methods",
        )
        if hasattr(hub, axis)
    }


def _trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


@pytest.fixture()
def pair(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Tiny-A2D fast paths need CUDA")
    hub_before = _hub_snapshot()
    rng_before = torch.get_rng_state()
    new_model, new_report = _wrap(_tiny_a2d(), monkeypatch)
    old_model, old_report = _wrap(
        _tiny_a2d(), monkeypatch, patcher=oracle_patch_a2d_peft
    )
    monkeypatch.undo()
    return {
        "new": (new_model, new_report),
        "old": (old_model, old_report),
        "hub_before": hub_before,
        "rng_before": rng_before,
    }


def test_same_concrete_types_and_trainable_set(pair):
    new, old = pair["new"][0], pair["old"][0]
    assert type(new) is type(old)
    assert type(new.base_model.model) is type(old.base_model.model)
    assert _trainable(new) == _trainable(old) and _trainable(new)


def test_same_state_dict_keys_and_dtypes(pair):
    assert _state(pair["new"][0]) == _state(pair["old"][0])


def test_same_report_and_callable_identity(pair):
    new_model, new_report = pair["new"]
    old_model, old_report = pair["old"]
    for field in (
        "requested",
        "applied",
        "skipped",
        "fallback",
        "family",
        "model_type",
    ):
        assert getattr(new_report, field) == getattr(old_report, field), field
    assert new_report.support.to_dict() == old_report.support.to_dict()
    assert new_report.warnings == old_report.warnings  # same report line, same text
    assert _fast_identity(new_model) == _fast_identity(old_model)
    assert all(all(flags) for flags in _fast_identity(new_model).values())


def test_same_forward_and_backward_bit_for_bit(pair):
    new_model, _ = pair["new"]
    old_model, _ = pair["old"]
    gen = torch.Generator("cuda").manual_seed(7)
    ids = torch.randint(2, 500, (2, 16), device="cuda", generator=gen)
    new_model.train()
    old_model.train()
    out_new = new_model(input_ids=ids).logits
    out_old = old_model(input_ids=ids).logits
    assert torch.equal(out_new, out_old)
    out_new.float().square().mean().backward()
    out_old.float().square().mean().backward()
    g_new = {n: p.grad for n, p in new_model.named_parameters() if p.grad is not None}
    g_old = {n: p.grad for n, p in old_model.named_parameters() if p.grad is not None}
    assert g_new.keys() == g_old.keys() and g_new
    for name in g_new:
        assert torch.equal(g_new[name], g_old[name]), name


def test_random_state_contract_identical(pair):
    """Same random_state ⇒ identical LoRA init on both paths; caller RNG untouched."""
    new_sd, old_sd = pair["new"][0].state_dict(), pair["old"][0].state_dict()
    lora_keys = [k for k in new_sd if "lora_A" in k]
    assert lora_keys
    for k in lora_keys:
        assert torch.equal(new_sd[k], old_sd[k]), k
    assert torch.equal(pair["rng_before"], torch.get_rng_state())


def test_save_reload_identical(pair, tmp_path):
    from safetensors.torch import load_file

    pair["new"][0].save_pretrained(tmp_path / "new")
    pair["old"][0].save_pretrained(tmp_path / "old")
    new = load_file(tmp_path / "new" / "adapter_model.safetensors")
    old = load_file(tmp_path / "old" / "adapter_model.safetensors")
    assert new.keys() == old.keys()
    for k in new:
        assert new[k].dtype == old[k].dtype and torch.equal(new[k], old[k]), k


def test_default_hub_unchanged(pair):
    assert _hub_snapshot() == pair["hub_before"]


def test_cpu_paths_report_identically(monkeypatch):
    """On CPU neither path installs anything; the reports must still agree."""
    new_model, new_report = _wrap(
        _tiny_a2d(dtype=torch.float32), monkeypatch, device="cpu"
    )
    old_model, old_report = _wrap(
        _tiny_a2d(dtype=torch.float32),
        monkeypatch,
        patcher=oracle_patch_a2d_peft,
        device="cpu",
    )
    assert new_report.to_dict() == old_report.to_dict()
    assert new_report.applied == {} and new_report.fallback is None
    assert _fast_identity(new_model) == _fast_identity(old_model)


# --- provider-owned knowledge -------------------------------------------------


def test_structure_mismatch_is_typed_not_attribute_error():
    from unturtle.models.conversion.a2d.tiny_a2d import fast_paths

    odd = types.SimpleNamespace(
        base_model=types.SimpleNamespace(model=types.SimpleNamespace(model=object()))
    )
    result = fast_paths.check_structure(odd)
    assert result.status == "unsupported" and result.reason == "structure_mismatch"
    assert result.details["missing"] == "layers"
    assert result.details["reached"] == "base_model.model.model"
    assert fast_paths.decoder_layers(odd) is None


def test_facade_withholds_all_fast_paths_on_structure_mismatch(monkeypatch):
    """The façade consults the provider's structure knowledge and reports a typed,
    whole-set fallback rather than dying on an AttributeError."""
    from unturtle.fast_diffusion_model import FastDiffusionModel

    class _Odd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(model_type="tiny-a2d-llama")
            self.lin = torch.nn.Linear(2, 2)
            self.base_model = types.SimpleNamespace(model=types.SimpleNamespace())

    seen: list[str] = []
    monkeypatch.setattr("unturtle.fast_diffusion_model._warn_once", seen.append)
    report = FastDiffusionModel.patch_peft_model_with_report(_Odd())
    assert report.fallback == "structure_mismatch"
    assert report.support.status == "unsupported"
    assert report.support.reason == "structure_mismatch"
    assert report.applied == {} and report.skipped == {}
    assert any("structure_mismatch" in m for m in seen)


def test_requested_kinds_are_family_owned():
    from unturtle.models.conversion.a2d.tiny_a2d import fast_paths

    assert fast_paths.requested_kinds(["q_proj"], on_cuda=False) == ("qkv",)
    assert fast_paths.requested_kinds(["o_proj", "up_proj"], on_cuda=True) == (
        "o",
        "mlp",
        "attention_forward",
    )
    # LLaDA / ModernBERT names are NOT this family's business
    assert fast_paths.requested_kinds(["attn_out", "Wqkv", "ff_out"], False) == ()


def test_applicability_predicates_gate_lora_bias_dora():
    from unturtle.models.conversion.a2d.tiny_a2d import fast_paths

    def proj(lora=True, bias=False, dora=False):
        m = torch.nn.Linear(4, 4, bias=bias)
        if lora:
            m.lora_A = torch.nn.ModuleDict()
        if dora:
            m.lora_magnitude_vector = torch.nn.ParameterDict(
                {"d": torch.nn.Parameter(torch.ones(1))}
            )
        return m

    ok = {k: proj() for k in ALL_PROJ}
    assert fast_paths.qkv_applicable(ok)
    assert fast_paths.o_applicable(ok)
    assert fast_paths.mlp_applicable(ok)
    assert not fast_paths.qkv_applicable({**ok, "k_proj": proj(lora=False)})
    assert not fast_paths.o_applicable({**ok, "o_proj": proj(bias=True)})
    assert not fast_paths.mlp_applicable({**ok, "up_proj": proj(dora=True)})
    assert not fast_paths.qkv_applicable({**ok, "v_proj": None})


def test_central_no_longer_owns_the_family():
    """Extraction gate: the façade holds no Tiny-A2D patcher; the registry
    resolves the provider; the provider does not import the façade."""
    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.conversion.a2d.tiny_a2d import fast_paths
    from unturtle.models.integrations import find_peft_integration

    assert not hasattr(fdm, "_patch_a2d_peft")
    for model_type in ("tiny-a2d-llama", "tiny-a2d-qwen2", "tiny-a2d-qwen3", "llama"):
        integration = find_peft_integration(model_type)
        assert integration.fast_paths is fast_paths, model_type
        assert integration.peft_patcher is fast_paths.patch_peft
    # `import unturtle` eagerly loads the façade, so a process-level check is
    # meaningless; instead: nothing the provider *bound* came from the façade,
    # and its import statements never name it.
    import ast
    import inspect

    for name, value in vars(fast_paths).items():
        assert getattr(value, "__module__", "") != fdm.__name__, name
    tree = ast.parse(inspect.getsource(fast_paths))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(m.startswith("unturtle.fast_diffusion_model") for m in imported), (
        imported
    )
