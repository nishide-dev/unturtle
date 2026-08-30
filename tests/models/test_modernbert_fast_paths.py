"""Differential contract for the ModernBERT fast-path provider (#185).

``oracle_patch_modernbert_peft`` is the *verbatim* ``_patch_modernbert_peft``
that lived in the façade on ``main`` (b1e2f19) before extraction — renamed
only. Everything else in the PEFT path is held constant; the two runs differ
solely in which patcher runs, so any divergence is the extraction's.
Bit-identical outputs are required, not tolerances.
"""

from __future__ import annotations

import types
from typing import Any, Literal

import pytest
import torch

from unturtle.models.backbones.modernbert._fast_forward import (
    ModernBertAttention_fast_forward,
    _install_modernbert_stubs,
    _original_apply_wo,
)
from unturtle.models.integrations import fast_path_support as fps
from unturtle.models.integrations.fast_path_support import apply_lora_o

_require_fast_lora = fps.require_fast_lora
_warn_once = fps.warn_once
_no_bias = fps.no_bias
_no_lora_mag = fps.no_lora_magnitude


# --- ORACLE: verbatim pre-extraction implementation (main @ b1e2f19) ----------
def oracle_patch_modernbert_peft(
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


# -----------------------------------------------------------------------------

TARGETS = ["Wqkv", "Wo"]


def _tiny_modernbert(seed: int = 0, dtype=torch.float32):
    from unturtle.models.backbones.modernbert import (
        A2DModernBertConfig,
        A2DModernBertForMaskedLM,
    )

    torch.manual_seed(seed)
    config = A2DModernBertConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    return A2DModernBertForMaskedLM(config).to(dtype)


def _wrap(model, monkeypatch, patcher=None, device="cuda"):
    """PEFT-wrap through the façade; optionally swap the patcher (oracle)."""
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.modernbert import fast_paths

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


def _fast_identity(model) -> dict[str, tuple]:
    out = {}
    for idx, layer in enumerate(model.base_model.model.model.layers):
        attn = layer.attn
        out[str(idx)] = (
            getattr(attn.__dict__.get("forward"), "__func__", None)
            is ModernBertAttention_fast_forward,
            attn.__dict__.get("apply_wo") is apply_lora_o,
            attn.__dict__.get("apply_wo") is _original_apply_wo,
            getattr(attn, "o_proj", None) is attn.Wo,  # alias lands in _modules
        )
    return out


def _state(model):
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()}


def _hub_snapshot():
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


@pytest.fixture()
def pair(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ModernBERT fast paths need CUDA")
    hub_before = _hub_snapshot()
    rng_before = torch.get_rng_state()
    new_model, new_report = _wrap(_tiny_modernbert(dtype=torch.bfloat16), monkeypatch)
    old_model, old_report = _wrap(
        _tiny_modernbert(dtype=torch.bfloat16),
        monkeypatch,
        patcher=oracle_patch_modernbert_peft,
    )
    monkeypatch.undo()
    return {
        "new": (new_model, new_report),
        "old": (old_model, old_report),
        "hub_before": hub_before,
        "rng_before": rng_before,
    }


def test_same_concrete_types_trainable_set_and_state_dict(pair):
    new, old = pair["new"][0], pair["old"][0]
    assert type(new) is type(old)
    assert type(new.base_model.model) is type(old.base_model.model)
    trainable_new = {n for n, p in new.named_parameters() if p.requires_grad}
    trainable_old = {n for n, p in old.named_parameters() if p.requires_grad}
    assert trainable_new == trainable_old and trainable_new
    assert _state(new) == _state(old)


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
    assert new_report.warnings == old_report.warnings
    assert _fast_identity(new_model) == _fast_identity(old_model)
    # fully eligible model: fast forward + apply_lora_o + o_proj alias on every layer
    for flags in _fast_identity(new_model).values():
        assert flags == (True, True, False, True)


def test_same_forward_and_backward_bit_for_bit(pair):
    new_model, _ = pair["new"]
    old_model, _ = pair["old"]
    gen = torch.Generator("cuda").manual_seed(7)
    ids = torch.randint(3, 1000, (2, 16), device="cuda", generator=gen)
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


def test_random_state_contract_and_save_identical(pair, tmp_path):
    from safetensors.torch import load_file

    new_model, _ = pair["new"]
    old_model, _ = pair["old"]
    new_sd, old_sd = new_model.state_dict(), old_model.state_dict()
    lora_keys = [k for k in new_sd if "lora_A" in k]
    assert lora_keys
    for k in lora_keys:
        assert torch.equal(new_sd[k], old_sd[k]), k
    assert torch.equal(pair["rng_before"], torch.get_rng_state())
    new_model.save_pretrained(tmp_path / "new")
    old_model.save_pretrained(tmp_path / "old")
    new = load_file(tmp_path / "new" / "adapter_model.safetensors")
    old = load_file(tmp_path / "old" / "adapter_model.safetensors")
    assert new.keys() == old.keys()
    for k in new:
        assert new[k].dtype == old[k].dtype and torch.equal(new[k], old[k]), k
    assert _hub_snapshot() == pair["hub_before"]


def test_cpu_paths_report_identically_and_install_only_stubs(monkeypatch):
    """On CPU both paths install ONLY the apply_wo stubs (before the device
    gate — the family's historical behavior) and nothing else."""
    new_model, new_report = _wrap(_tiny_modernbert(), monkeypatch, device="cpu")
    old_model, old_report = _wrap(
        _tiny_modernbert(),
        monkeypatch,
        patcher=oracle_patch_modernbert_peft,
        device="cpu",
    )
    assert new_report.to_dict() == old_report.to_dict()
    assert new_report.applied == {} and new_report.fallback is None
    assert _fast_identity(new_model) == _fast_identity(old_model)
    for model in (new_model, old_model):
        for layer in model.base_model.model.model.layers:
            assert layer.attn.__dict__.get("apply_wo") is _original_apply_wo
            assert "forward" not in layer.attn.__dict__


# --- CPU-runnable partial-eligibility differential -----------------------------


class _Proj(torch.nn.Linear):
    def __init__(self, lora=True, bias=False, dora=False):
        super().__init__(4, 4, bias=bias)
        if lora:
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(4, 2, bias=False)}
            )
        if dora:
            self.lora_magnitude_vector = torch.nn.ParameterDict(
                {"default": torch.nn.Parameter(torch.ones(4))}
            )


def _synthetic_peft_like(spec: dict[str, dict], device: str) -> torch.nn.Module:
    attn = torch.nn.Module()
    attn.Wqkv = _Proj(**spec.get("Wqkv", {}))
    attn.Wo = _Proj(**spec.get("Wo", {}))
    layer = torch.nn.Module()
    layer.attn = attn
    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([layer])
    lm = torch.nn.Module()
    lm.model = inner
    base = torch.nn.Module()
    base.model = lm
    root = torch.nn.Module()
    root.base_model = base
    return root.to(device)


PARTIAL_CASES = {
    "all_eligible": {},
    "wo_missing_lora": {"Wo": {"lora": False}},
    "wo_with_bias": {"Wo": {"bias": True}},
    "wo_with_dora": {"Wo": {"dora": True}},
}


@pytest.mark.gpu
@pytest.mark.parametrize("case", sorted(PARTIAL_CASES))
@pytest.mark.parametrize(
    "lora_dropout,bias", [(0.0, "none"), (0.1, "none"), (0.0, "all")]
)
def test_partial_eligibility_matches_oracle(monkeypatch, case, lora_dropout, bias):
    """Counts, warnings (text AND order) and installed identities must match the
    verbatim oracle across the eligibility matrix.

    CUDA-gated: on CPU both implementations return before the eligibility loop,
    so every case degenerates to the same trivial pass — green without testing
    the gate (the CPU contract is covered by the façade CPU test above)."""
    if not torch.cuda.is_available():
        pytest.skip("the eligibility matrix only discriminates on CUDA")
    device = "cuda"
    from unturtle.models.backbones.modernbert import fast_paths

    def run(patcher, warn_target):
        seen: list[str] = []
        monkeypatch.setattr(warn_target, "warn_once", seen.append)
        model = _synthetic_peft_like(PARTIAL_CASES[case], device)
        counts = patcher(model, lora_dropout, bias)
        monkeypatch.undo()
        return counts, tuple(seen), _fast_identity(model)

    new = run(fast_paths.patch_peft, fast_paths)
    seen_old: list[str] = []
    monkeypatch.setitem(globals(), "_warn_once", seen_old.append)
    model = _synthetic_peft_like(PARTIAL_CASES[case], device)
    counts_old = oracle_patch_modernbert_peft(model, lora_dropout, bias)
    old = (counts_old, tuple(seen_old), _fast_identity(model))
    assert new == old, (case, lora_dropout, bias, new, old)


def test_missing_structure_matches_oracle_failopen(monkeypatch):
    """Direct provider call on an untraversable tree: warn + install nothing —
    the family's historical behavior, unlike Tiny-A2D's raise."""
    from unturtle.models.backbones.modernbert import fast_paths

    odd = torch.nn.Module()
    odd.lin = torch.nn.Linear(2, 2)
    odd.base_model = types.SimpleNamespace(model=types.SimpleNamespace())

    seen_new: list[str] = []
    monkeypatch.setattr(fast_paths, "warn_once", seen_new.append)
    counts_new = fast_paths.patch_peft(odd, 0.0, "none")
    monkeypatch.undo()

    seen_old: list[str] = []
    monkeypatch.setitem(globals(), "_warn_once", seen_old.append)
    counts_old = oracle_patch_modernbert_peft(odd, 0.0, "none")
    assert counts_new == counts_old == (0, 0, 0)
    assert tuple(seen_new) == tuple(seen_old) and len(seen_new) == 1
    assert "could not locate model.layers" in seen_new[0]


# --- provider-owned knowledge -------------------------------------------------


def test_structure_mismatch_is_typed(monkeypatch):
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.modernbert import fast_paths

    odd = types.SimpleNamespace(
        base_model=types.SimpleNamespace(model=types.SimpleNamespace(model=object()))
    )
    result = fast_paths.check_structure(odd)
    assert result.status == "unsupported" and result.reason == "structure_mismatch"
    assert result.details["missing"] == "layers"
    assert fast_paths.decoder_layers(odd) is None

    class _Odd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(model_type="modernbert-diffusion")
            self.lin = torch.nn.Linear(2, 2)
            self.base_model = types.SimpleNamespace(model=types.SimpleNamespace())

    seen: list[str] = []
    monkeypatch.setattr("unturtle.fast_diffusion_model._warn_once", seen.append)
    report = FastDiffusionModel.patch_peft_model_with_report(_Odd())
    assert report.fallback == "structure_mismatch"
    assert report.support.reason == "structure_mismatch"
    assert report.applied == {} and report.skipped == {}


def test_requested_kinds_are_family_owned():
    from unturtle.models.backbones.modernbert import fast_paths

    assert fast_paths.requested_kinds(["Wqkv"], on_cuda=False) == ("qkv",)
    assert fast_paths.requested_kinds(["Wo", "Wi"], on_cuda=True) == (
        "o",
        "mlp",
        "attention_forward",
    )
    # other families' names are NOT this family's business
    assert fast_paths.requested_kinds(["q_proj", "attn_out", "up_proj"], False) == ()


def test_central_no_longer_owns_the_family():
    """Extraction gate: the façade holds no ModernBERT patcher (family branches
    4→2 across the two extractions); the registry resolves the provider; the
    provider does not import the façade."""
    import ast
    import inspect

    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.backbones.modernbert import fast_paths
    from unturtle.models.integrations import find_peft_integration

    assert not hasattr(fdm, "_patch_modernbert_peft")
    assert not hasattr(fdm, "_install_modernbert_stubs")
    remaining = [n for n in dir(fdm) if n.startswith("_patch_") and n.endswith("_peft")]
    assert sorted(remaining) == ["_patch_llada_peft"]
    integration = find_peft_integration("modernbert-diffusion")
    assert integration.fast_paths is fast_paths
    assert integration.peft_patcher is fast_paths.patch_peft
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


# --- report line: verbatim oracle (main registry _modernbert_report) ----------
def oracle_modernbert_report(model: Any, counts: tuple[int, int, int]) -> str:
    _n_qkv, n_o, _n_mlp = counts
    return (
        f"FastDiffusionModel (ModernBERT) patched {len(model.base_model.model.model.layers)} layers with "
        f"{n_o} Wo (output proj) layers. "
        "Wqkv/MLP Triton kernels not yet supported for ModernBERT — "
        "see issue #59 Phase 2."
    )


def test_report_line_matches_oracle_verbatim():
    from unturtle.models.backbones.modernbert import fast_paths

    model = _synthetic_peft_like({}, "cpu")
    for counts in ((0, 0, 0), (0, 1, 0), (0, 2, 0)):
        assert fast_paths.report(model, counts) == oracle_modernbert_report(
            model, counts
        ), counts


def test_report_on_untraversable_model_is_failopen_zero_layers():
    """The one deliberate divergence from the oracle: on an untraversable model
    the oracle's report crashed (AttributeError via the deep path); the provider
    reports 0 layers instead. Pin the exact text."""
    from unturtle.models.backbones.modernbert import fast_paths

    odd = types.SimpleNamespace(
        base_model=types.SimpleNamespace(model=types.SimpleNamespace())
    )
    line = fast_paths.report(odd, (0, 0, 0))
    assert line.startswith(
        "FastDiffusionModel (ModernBERT) patched 0 layers with 0 Wo"
    ), line
    with pytest.raises(AttributeError):
        oracle_modernbert_report(odd, (0, 0, 0))
