"""Differential contract for the Dream fast-path provider (#185).

``oracle_patch_dream_peft`` / ``oracle_dream_report`` are the *verbatim*
pre-extraction implementations from ``main`` (7fe71a6) — renamed only.
Everything else in the PEFT path is held constant; the runs differ solely in
which patcher runs, so any divergence is the extraction's. Bit-identical
outputs are required, not tolerances. #177's complete fused-path contract for
Dream (bias-aware QKV kernel) and the #174 RoPE reload fix are exercised, and
generation defaults (#189) are untouched.
"""

from __future__ import annotations

import types
from typing import Any, Literal

import pytest
import torch

from unturtle.models.backbones.dream.modeling_dream import (
    DreamAttention_fast_forward,
)
from unturtle.models.integrations import fast_path_support as fps
from unturtle.models.integrations.fast_path_support import (
    apply_lora_mlp_swiglu,
    apply_lora_o,
    apply_lora_qkv_with_bias,
)

_require_fast_lora = fps.require_fast_lora
_warn_once = fps.warn_once
_no_bias = fps.no_bias
_no_lora_mag = fps.no_lora_magnitude


# --- ORACLES: verbatim pre-extraction implementations (main @ 7fe71a6) --------
def oracle_patch_dream_peft(
    model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]
) -> tuple[int, int, int]:
    """Patch Dream model with Triton LoRA kernels.

    Dream's q/k/v_proj have ``bias=True``, so the standard ``apply_lora_qkv``
    is replaced with ``apply_lora_qkv_with_bias`` (``LoRA_QKV_Bias`` kernel).
    o_proj (bias=False) uses the standard ``apply_lora_o``.
    MLP (gate/up/down, all bias=False) uses ``apply_lora_mlp_swiglu``.

    Layer layout: ``model.base_model.model.model.layers``
    (Dream wraps DreamBaseModel as ``self.model``, same depth as LLaMA).

    The injected ``DreamAttention_fast_forward`` covers the non-cache path only;
    cache-enabled block decode (tuple KV caches, ``dual_cache`` /
    ``replace_position``) delegates internally to the standard class forward, so
    ``model.generate(..., use_cache=True)`` keeps working on a patched model.
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


def oracle_dream_report(model: Any, counts: tuple[int, int, int]) -> str:
    n_qkv, n_o, n_mlp = counts
    return (
        f"FastDiffusionModel (Dream) patched {len(model.base_model.model.model.layers)} layers with "
        f"{n_qkv} QKV layers (bias kernel), {n_o} O layers and {n_mlp} MLP layers."
    )


# -----------------------------------------------------------------------------

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ALL_PROJ = tuple(TARGETS)


def _tiny_dream(seed: int = 0, dtype=torch.bfloat16):
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
    return model.to(dtype) if dtype is not None else model


def _wrap(model, monkeypatch, patcher=None, device="cuda"):
    """PEFT-wrap through the façade; optionally swap the patcher (oracle)."""
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.dream import fast_paths

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
        attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        a = attn.__dict__ if attn is not None else {}
        m = mlp.__dict__ if mlp is not None else {}
        out[str(idx)] = (
            getattr(a.get("forward"), "__func__", None) is DreamAttention_fast_forward,
            a.get("apply_qkv") is apply_lora_qkv_with_bias,
            a.get("apply_o") is apply_lora_o,
            getattr(m.get("forward"), "__func__", None) is apply_lora_mlp_swiglu,
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
        pytest.skip("Dream fast paths need CUDA")
    hub_before = _hub_snapshot()
    rng_before = torch.get_rng_state()
    new_model, new_report = _wrap(_tiny_dream(), monkeypatch)
    old_model, old_report = _wrap(
        _tiny_dream(), monkeypatch, patcher=oracle_patch_dream_peft
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
    """#177 complete fused-path contract: every layer gets the bias QKV kernel,
    apply_lora_o, the swiglu MLP forward and the fast attention forward — and
    identically on both paths."""
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
    for flags in _fast_identity(new_model).values():
        assert flags == (True, True, True, True)
    assert new_report.fallback is None  # no partial-fast, no silent withhold


def test_same_forward_and_backward_bit_for_bit(pair):
    """Forward logits must be bit-identical. Dream's Triton backward uses
    non-deterministic accumulation, so run-to-run grads differ even on ONE
    model; the differential requirement is that the cross-model deviation is
    bounded by that self-envelope (the extraction adds nothing)."""
    new_model, _ = pair["new"]
    old_model, _ = pair["old"]
    gen = torch.Generator("cuda").manual_seed(7)
    ids = torch.randint(2, 500, (2, 16), device="cuda", generator=gen)
    new_model.train()
    old_model.train()
    out_new = new_model(input_ids=ids).logits
    out_old = old_model(input_ids=ids).logits
    assert torch.equal(out_new, out_old)

    def grads(model, logits=None):
        model.zero_grad(set_to_none=True)
        out = logits if logits is not None else model(input_ids=ids).logits
        out.float().square().mean().backward()
        return {
            n: p.grad.detach().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }

    g_new = grads(new_model, out_new)
    g_old_1 = grads(old_model, out_old)
    g_old_2 = grads(old_model)  # self-envelope: same model, second run
    assert g_new.keys() == g_old_1.keys() and g_new
    self_env = max(
        (g_old_1[k].float() - g_old_2[k].float()).abs().max().item() for k in g_old_1
    )
    cross = max(
        (g_new[k].float() - g_old_1[k].float()).abs().max().item() for k in g_new
    )
    if self_env == 0.0:
        for k in g_new:
            assert torch.equal(g_new[k], g_old_1[k]), k
    else:
        assert cross <= 4 * self_env, (cross, self_env)


def test_random_state_contract_save_and_reload_identical(pair, tmp_path):
    """#188 RNG contract on both paths; save; and the #174 reload fix intact:
    a from_pretrained reload of the base rebuilds the same RoPE buffers."""
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
    # #174 contract: direct construction and from_pretrained reload rebuild the
    # SAME canonical RoPE buffers. (The bf16-cast training copy is not the
    # reference — the canonical initializer is.)
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    canonical = _tiny_dream(dtype=None)  # fp32 direct construction
    canonical.save_pretrained(tmp_path / "base")
    reloaded = DreamModel.from_pretrained(tmp_path / "base")
    fresh = {n: b for n, b in canonical.named_buffers() if n.endswith("inv_freq")}
    again = {n: b for n, b in reloaded.named_buffers() if n.endswith("inv_freq")}
    assert fresh.keys() == again.keys() and fresh
    for k in fresh:
        assert torch.equal(fresh[k].cpu(), again[k].cpu()), k


def test_cpu_paths_install_nothing_without_touching_structure(monkeypatch):
    """Dream's CUDA gate runs FIRST: on CPU neither path touches the model's
    structure — even a structurally broken tree returns zeros silently."""
    from unturtle.models.backbones.dream import fast_paths

    new_model, new_report = _wrap(
        _tiny_dream(dtype=torch.float32), monkeypatch, device="cpu"
    )
    old_model, old_report = _wrap(
        _tiny_dream(dtype=torch.float32),
        monkeypatch,
        patcher=oracle_patch_dream_peft,
        device="cpu",
    )
    assert new_report.to_dict() == old_report.to_dict()
    assert new_report.applied == {} and new_report.fallback is None
    assert _fast_identity(new_model) == _fast_identity(old_model)

    # the previous _wrap left patch_peft monkeypatched to the oracle — undo so
    # the Booby probes exercise the PROVIDER, not the oracle
    monkeypatch.undo()
    assert fast_paths.patch_peft is not oracle_patch_dream_peft

    class _Booby(torch.nn.Module):
        """Raises if the deep path is dereferenced."""

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        @property
        def base_model(self):
            raise AssertionError("structure must not be touched on CPU")

    booby = _Booby().to("cpu")  # pin explicitly: the gate must see CPU params
    assert next(iter(booby.parameters())).device.type == "cpu"
    assert fast_paths.patch_peft(booby, 0.0, "none") == (0, 0, 0)
    assert oracle_patch_dream_peft(_Booby().to("cpu"), 0.0, "none") == (0, 0, 0)


# --- CPU-runnable? No: the eligibility matrix only discriminates on CUDA ------


class _Proj(torch.nn.Linear):
    def __init__(self, lora=True, bias=True, dora=False):
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
    """PeftModel-shaped tree with one Dream-shaped layer built from ``spec``.

    Dream defaults: q/k/v biased, o/gate/up/down bias-free.
    ``spec={"self_attn": None}`` / ``{"mlp": None}`` omit the submodule.
    """
    layer = torch.nn.Module()
    if spec.get("self_attn", {}) is not None:
        attn = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(attn, name, _Proj(**spec.get(name, {})))
        attn.o_proj = _Proj(**{"bias": False, **spec.get("o_proj", {})})
        layer.self_attn = attn
    if spec.get("mlp", {}) is not None:
        mlp = torch.nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(mlp, name, _Proj(**{"bias": False, **spec.get(name, {})}))
        layer.mlp = mlp
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
    "qkv_missing_lora": {"k_proj": {"lora": False}},
    "qkv_with_dora": {"q_proj": {"dora": True}},
    "o_with_bias": {"o_proj": {"bias": True}},  # silently standard (no warn)
    "mlp_missing_lora": {"up_proj": {"lora": False}},  # silently standard
    "no_self_attn": {"self_attn": None},
    "no_mlp": {"mlp": None},
}


@pytest.mark.gpu
@pytest.mark.parametrize("case", sorted(PARTIAL_CASES))
@pytest.mark.parametrize(
    "lora_dropout,bias", [(0.0, "none"), (0.1, "none"), (0.0, "all")]
)
def test_partial_eligibility_matches_oracle(monkeypatch, case, lora_dropout, bias):
    """Counts, warnings (text AND order — only ineligible QKV warns; O and MLP
    stay silent) and installed identities must match the verbatim oracle.

    CUDA-gated: on CPU both implementations return before touching anything,
    so every case degenerates to the same trivial pass."""
    if not torch.cuda.is_available():
        pytest.skip("the eligibility matrix only discriminates on CUDA")
    from unturtle.models.backbones.dream import fast_paths

    def run(patcher, warn_target):
        seen: list[str] = []
        monkeypatch.setattr(warn_target, "warn_once", seen.append)
        model = _synthetic_peft_like(PARTIAL_CASES[case], "cuda")
        counts = patcher(model, lora_dropout, bias)
        monkeypatch.undo()
        return counts, tuple(seen), _fast_identity(model)

    new = run(fast_paths.patch_peft, fast_paths)
    seen_old: list[str] = []
    monkeypatch.setitem(globals(), "_warn_once", seen_old.append)
    model = _synthetic_peft_like(PARTIAL_CASES[case], "cuda")
    counts_old = oracle_patch_dream_peft(model, lora_dropout, bias)
    old = (counts_old, tuple(seen_old), _fast_identity(model))
    assert new == old, (case, lora_dropout, bias, new, old)


@pytest.mark.gpu
def test_missing_structure_raises_like_oracle_on_cuda(monkeypatch):
    """On CUDA an untraversable tree raised AttributeError on main; the provider
    keeps that for direct calls (with a typed message), and the façade converts
    it into the typed structure_mismatch fallback."""
    if not torch.cuda.is_available():
        pytest.skip("the structure path is only reached on CUDA")
    from unturtle.models.backbones.dream import fast_paths

    def odd():
        m = torch.nn.Module()
        m.lin = torch.nn.Linear(2, 2)
        m.base_model = types.SimpleNamespace(model=types.SimpleNamespace())
        return m.cuda()

    with pytest.raises(AttributeError):
        oracle_patch_dream_peft(odd(), 0.0, "none")
    with pytest.raises(AttributeError, match="structure_mismatch"):
        fast_paths.patch_peft(odd(), 0.0, "none")


def test_structure_mismatch_is_typed_via_facade(monkeypatch):
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.dream import fast_paths

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
            self.config = types.SimpleNamespace(model_type="dream")
            self.lin = torch.nn.Linear(2, 2)
            self.base_model = types.SimpleNamespace(model=types.SimpleNamespace())

    seen: list[str] = []
    monkeypatch.setattr("unturtle.fast_diffusion_model._warn_once", seen.append)
    report = FastDiffusionModel.patch_peft_model_with_report(_Odd())
    assert report.fallback == "structure_mismatch"
    assert report.support.reason == "structure_mismatch"
    assert report.applied == {} and report.skipped == {}


def test_requested_kinds_and_bias_aware_predicates():
    from unturtle.models.backbones.dream import fast_paths

    assert fast_paths.requested_kinds(["q_proj"], on_cuda=False) == ("qkv",)
    assert fast_paths.requested_kinds(["o_proj", "up_proj"], on_cuda=True) == (
        "o",
        "mlp",
        "attention_forward",
    )
    assert fast_paths.requested_kinds(["Wqkv", "attn_out", "Wi"], False) == ()

    ok = fast_paths.layer_targets(
        _synthetic_peft_like({}, "cpu").base_model.model.model.layers[0]
    )
    # bias-aware QKV: biased q/k/v ARE eligible (bias kernel), DoRA is not
    assert fast_paths.qkv_applicable(ok)
    biased = dict(ok)
    biased["q_proj"] = _Proj(bias=True)
    assert fast_paths.qkv_applicable(biased)
    dora = dict(ok)
    dora["v_proj"] = _Proj(dora=True)
    assert not fast_paths.qkv_applicable(dora)
    # o requires bias-free
    o_biased = dict(ok)
    o_biased["o_proj"] = _Proj(bias=True)
    assert fast_paths.o_applicable(ok) and not fast_paths.o_applicable(o_biased)
    assert fast_paths.mlp_applicable(ok)
    assert not fast_paths.mlp_applicable({**ok, "mlp": None})


# --- report line: verbatim oracle ---------------------------------------------


def test_report_line_matches_oracle_verbatim():
    from unturtle.models.backbones.dream import fast_paths

    model = _synthetic_peft_like({}, "cpu")
    for counts in ((0, 0, 0), (2, 2, 2), (1, 0, 2)):
        assert fast_paths.report(model, counts) == oracle_dream_report(model, counts), (
            counts
        )


def test_report_on_untraversable_model_is_failopen_zero_layers():
    """The one deliberate report divergence: the oracle crashed (AttributeError
    via the deep path); the provider reports 0 layers. Pin the exact text."""
    from unturtle.models.backbones.dream import fast_paths

    odd = types.SimpleNamespace(
        base_model=types.SimpleNamespace(model=types.SimpleNamespace())
    )
    line = fast_paths.report(odd, (0, 0, 0))
    assert line.startswith(
        "FastDiffusionModel (Dream) patched 0 layers with 0 QKV layers (bias kernel)"
    ), line
    with pytest.raises(AttributeError):
        oracle_dream_report(odd, (0, 0, 0))


def test_central_no_longer_owns_the_family():
    """Extraction gate: façade family patchers 4→1 across the three extractions;
    the registry resolves the provider; the provider does not import the façade."""
    import ast
    import inspect

    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.backbones.dream import fast_paths
    from unturtle.models.integrations import find_peft_integration

    assert not hasattr(fdm, "_patch_dream_peft")
    remaining = [n for n in dir(fdm) if n.startswith("_patch_") and n.endswith("_peft")]
    assert sorted(remaining) == ["_patch_llada_peft"]
    for model_type in ("dream", "Dream"):
        integration = find_peft_integration(model_type)
        assert integration.fast_paths is fast_paths, model_type
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
