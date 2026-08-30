"""Differential + delta contract for the LLaDA fast-path provider (#185, 4/4).

This PR is extraction PLUS an intended behavior change, verified as split diffs:

- **Unchanged vs oracle** (``oracle_patch_llada_peft`` / ``oracle_llada_report``
  are the *verbatim* pre-extraction implementations from ``main`` @ 81ba9b6,
  renamed only): structure traversal, requested/applied/skipped/fallback,
  rope/MLP installation, warning/report text and order, RNG, save, hub,
  state dict.
- **Intended delta**, proven separately: ``LLaDALlamaBlock.forward`` and
  ``LLaDABlock.attention`` now dispatch through the provider-installed
  ``apply_qkv`` / ``apply_o`` (before: installed-not-live, #184 ledger), and
  the O install adds the ``o_proj -> attn_out`` alias ``apply_lora_o`` needs.
  Liveness is proven by ``probe_liveness`` counters (forward AND LoRA backward
  gradients); the default stubs are proven bit-identical to the direct
  projections, so the standard / non-PEFT path keeps its historical outputs.

Note: an oracle-patched CUDA model is NOT forward-runnable under the new
wiring (the oracle never installed the ``o_proj`` alias — its ``apply_o`` hook
was dead code). CUDA forward parity therefore compares the provider against
``oracle + the one documented alias line``, which is exactly the intended delta.
"""

from __future__ import annotations

import types
from typing import Any, Literal

import pytest
import torch

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


# --- ORACLES: verbatim pre-extraction implementations (main @ 81ba9b6) --------
def oracle_patch_llada_peft(
    model: Any, lora_dropout: float, bias: Literal["none", "all", "lora_only"]
) -> tuple[int, int, int]:
    """Patch LLaDA model with Triton LoRA kernels.

    LLaDA uses a non-standard layer hierarchy:
      ``model.base_model.model.transformer.blocks`` (list of ``LLaDABlock``).

    ``LLaDALlamaBlock`` has ``q_proj/k_proj/v_proj/attn_out/ff_proj/up_proj``.
    Other block types (``LLaDASequentialBlock``) use ``att_proj`` (fused QKV)
    and are not supported by the split QKV kernel — they are skipped with a
    warning.
    """
    from unturtle.models.backbones.llada.modeling_llada import (
        LLaDALlamaBlock,
        _make_llada_fast_rope_forward,
    )

    n_qkv = n_o = n_mlp = 0

    # Triton kernels require the model to be on CUDA.
    first_param = next(iter(model.parameters()), None)
    if first_param is None or first_param.device.type != "cuda":
        return n_qkv, n_o, n_mlp

    # LLaDAModelLM wraps LLaDAModel in self.model, so the path differs:
    # PeftModel → base_model → model (LLaDAModelLM) → model (LLaDAModel) → transformer
    inner = model.base_model.model
    if hasattr(inner, "model") and hasattr(inner.model, "transformer"):
        transformer = inner.model.transformer
    elif hasattr(inner, "transformer"):
        transformer = inner.transformer
    else:
        _warn_once(
            "FastDiffusionModel (LLaDA): could not locate transformer — "
            "cannot patch LoRA kernels. Is this a supported LLaDA checkpoint?"
        )
        return n_qkv, n_o, n_mlp

    if not hasattr(transformer, "blocks"):
        _warn_once(
            "FastDiffusionModel (LLaDA): transformer.blocks not found — "
            "cannot patch LoRA kernels. Is this a supported LLaDA checkpoint?"
        )
        return n_qkv, n_o, n_mlp

    blocks = transformer.blocks

    if lora_dropout == 0 and bias == "none":
        _require_fast_lora()

    for block in blocks:
        if not isinstance(block, LLaDALlamaBlock):
            _warn_once(
                f"FastDiffusionModel (LLaDA): skipping block type {type(block).__name__} "
                "(only LLaDALlamaBlock is supported for Triton LoRA patching)."
            )
            continue

        # Inject Triton RoPE fast forward unconditionally (CUDA already checked above).
        rotary_emb = getattr(block, "rotary_emb", None)
        if rotary_emb is not None and not getattr(
            rotary_emb, "_fast_rope_patched", False
        ):
            import types

            rotary_emb.forward = types.MethodType(
                _make_llada_fast_rope_forward(type(rotary_emb).forward), rotary_emb
            )
            rotary_emb._fast_rope_patched = True

        if lora_dropout != 0 or bias != "none":
            continue

        # LLaDALlamaBlock: q_proj / k_proj / v_proj (bias depends on config)
        q_proj = getattr(block, "q_proj", None)
        k_proj = getattr(block, "k_proj", None)
        v_proj = getattr(block, "v_proj", None)
        if (
            q_proj is not None
            and k_proj is not None
            and v_proj is not None
            and hasattr(q_proj, "lora_A")
            and hasattr(k_proj, "lora_A")
            and hasattr(v_proj, "lora_A")
            and _no_bias(q_proj)
            and _no_bias(k_proj)
            and _no_bias(v_proj)
            and _no_lora_mag(q_proj)
            and _no_lora_mag(k_proj)
            and _no_lora_mag(v_proj)
        ):
            block.apply_qkv = apply_lora_qkv
            n_qkv += 1
        else:
            _warn_once(
                "FastDiffusionModel (LLaDA): cannot patch QKV with Triton kernel "
                "(LoRA not enabled or bias present — config.include_qkv_bias=True)."
            )

        # attn_out (o_proj equivalent)
        attn_out = getattr(block, "attn_out", None)
        if (
            attn_out is not None
            and hasattr(attn_out, "lora_A")
            and _no_bias(attn_out)
            and _no_lora_mag(attn_out)
        ):
            block.apply_o = apply_lora_o
            n_o += 1
        else:
            _warn_once(
                "FastDiffusionModel (LLaDA): cannot patch attn_out with Triton kernel."
            )

        # ff_proj / up_proj / ff_out — gated MLP (gate/up/down).
        # apply_lora_mlp_swiglu reads self.gate_proj / self.up_proj / self.down_proj
        # and uses the SiLU-gated SwiGLU Triton kernel.
        # Only patch when activation_type is SiLU (output_multiplier==1); with SwiGLU
        # (output_multiplier==0.5) ff_proj output is halved by chunk(2) while up_proj
        # stays full-width, producing a shape mismatch in the Triton kernel.
        block_act = getattr(block, "act", None)
        act_is_silu = block_act is not None and isinstance(block_act, torch.nn.SiLU)
        ff_proj = getattr(block, "ff_proj", None)
        up_proj = getattr(block, "up_proj", None)
        ff_out = getattr(block, "ff_out", None)
        if not act_is_silu:
            _warn_once(
                f"FastDiffusionModel (LLaDA): skipping Triton MLP patch for "
                f"{type(block_act).__name__} activation — only SiLU is supported. "
                "MLP LoRA will use PEFT default path."
            )
        elif (
            ff_proj is not None
            and up_proj is not None
            and ff_out is not None
            and hasattr(ff_proj, "lora_A")
            and hasattr(up_proj, "lora_A")
            and hasattr(ff_out, "lora_A")
            and _no_bias(ff_proj)
            and _no_bias(up_proj)
            and _no_bias(ff_out)
            and _no_lora_mag(ff_proj)
            and _no_lora_mag(up_proj)
            and _no_lora_mag(ff_out)
        ):
            # Set gate_proj/down_proj aliases for apply_lora_mlp_swiglu compatibility.
            block.gate_proj = ff_proj
            block.down_proj = ff_out
            block.apply_mlp = apply_lora_mlp_swiglu
            n_mlp += 1
        else:
            _warn_once(
                "FastDiffusionModel (LLaDA): cannot patch MLP with Triton kernel "
                "(LoRA not enabled, bias present, or magnitude scaling active)."
            )

    return n_qkv, n_o, n_mlp


def oracle_llada_report(model: Any, counts: tuple[int, int, int]) -> str:
    n_qkv, n_o, _n_mlp = counts
    # LLaDA nests differently from the Llama-shaped families: blocks live under
    # transformer, at a depth that varies with how the model was wrapped.
    inner = model.base_model.model
    transformer = (
        inner.model.transformer
        if hasattr(inner, "model") and hasattr(inner.model, "transformer")
        else getattr(inner, "transformer", None)
    )
    n_blocks = len(transformer.blocks) if transformer is not None else 0
    return (
        f"FastDiffusionModel (LLaDA) patched {n_blocks} blocks with "
        f"{n_qkv} QKV blocks and {n_o} O (attn_out) blocks."
    )


# -----------------------------------------------------------------------------

TARGETS = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]


def _tiny_llada(seed: int = 0, include_qkv_bias: bool = False):
    from unturtle.models.backbones.llada.configuration_llada import LLaDAConfig
    from unturtle.models.backbones.llada.modeling_llada import LLaDAModelLM

    torch.manual_seed(seed)
    config = LLaDAConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_hidden_size=128,
        vocab_size=512,
        embedding_size=512,
        max_sequence_length=64,
        block_type="llama",
        activation_type="silu",
        rope=True,
        include_bias=False,
        include_qkv_bias=include_qkv_bias,
        weight_tying=False,
    )
    return LLaDAModelLM(config).eval()


def _wrap(model, monkeypatch, patcher=None, device="cuda", targets=TARGETS):
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.llada import fast_paths

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
        target_modules=list(targets),
        use_gradient_checkpointing=False,
        random_state=1234,
    )


def _blocks(model):
    from unturtle.models.backbones.llada.fast_paths import decoder_blocks

    return list(decoder_blocks(model))


def _fast_identity(model) -> dict[str, tuple]:
    out = {}
    for idx, block in enumerate(_blocks(model)):
        own = block.__dict__
        out[str(idx)] = (
            getattr(block.rotary_emb, "_fast_rope_patched", False),
            own.get("apply_qkv") is apply_lora_qkv,
            own.get("apply_o") is apply_lora_o,
            own.get("apply_mlp") is apply_lora_mlp_swiglu,
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
        pytest.skip("LLaDA fast paths need CUDA")
    hub_before = _hub_snapshot()
    rng_before = torch.get_rng_state()
    new_model, new_report = _wrap(_tiny_llada(), monkeypatch)
    old_model, old_report = _wrap(
        _tiny_llada(), monkeypatch, patcher=oracle_patch_llada_peft
    )
    monkeypatch.undo()
    return {
        "new": (new_model, new_report),
        "old": (old_model, old_report),
        "hub_before": hub_before,
        "rng_before": rng_before,
    }


# ============================ UNCHANGED vs ORACLE ============================


def test_same_types_trainable_set_and_state_dict(pair):
    new, old = pair["new"][0], pair["old"][0]
    assert type(new) is type(old)
    trainable_new = {n for n, p in new.named_parameters() if p.requires_grad}
    trainable_old = {n for n, p in old.named_parameters() if p.requires_grad}
    assert trainable_new == trainable_old and trainable_new
    assert _state(new) == _state(old)


def test_same_report_warnings_and_installation(pair):
    """requested/applied/skipped/fallback, warning+report text and order, and the
    rope/QKV/MLP installations are oracle-identical. The one intended install
    delta — the ``o_proj`` alias — is asserted explicitly on the provider side
    and its ABSENCE on the oracle side."""
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
    for block in _blocks(new_model):
        assert block.o_proj is block.attn_out  # intended delta: alias installed
        # __dict__ alias, never module registration: state_dict must not grow
        assert "o_proj" in block.__dict__ and "o_proj" not in block._modules
    for block in _blocks(old_model):
        assert not hasattr(block, "o_proj")  # oracle never aliased (dead hook)


def test_cuda_forward_backward_matches_oracle_plus_alias(pair):
    """Bit-for-bit forward/backward: provider vs oracle + the ONE documented
    alias line (the oracle-patched model is otherwise not runnable — its dead
    apply_o hook becomes live under the new wiring and needs self.o_proj)."""
    new_model, _ = pair["new"]
    old_model, _ = pair["old"]
    for block in _blocks(old_model):
        block.__dict__["o_proj"] = block.attn_out  # the intended delta, manually
    gen = torch.Generator("cuda").manual_seed(7)
    ids = torch.randint(2, 500, (2, 16), device="cuda", generator=gen)
    # eval: LLaDA carries residual dropout, and two sequential stochastic
    # forwards would consume RNG differently — parity must be deterministic.
    new_model.eval()
    old_model.eval()
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

    # The Triton LoRA backward accumulates non-deterministically even on ONE
    # model; the differential bound is the same-model run-to-run envelope.
    g_new = grads(new_model, out_new)
    g_old_1 = grads(old_model, out_old)
    g_old_2 = grads(old_model)
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
        # 8x: the self-envelope is itself a stochastic estimate from one
        # repeat; 4x flaked (~1 in 5 full-suite runs) while a real divergence
        # is orders of magnitude larger than the envelope.
        assert cross <= 8 * self_env, (cross, self_env)


def test_random_state_save_and_hub_identical(pair, tmp_path):
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


def test_cpu_paths_install_nothing_without_touching_structure(monkeypatch):
    """The CUDA gate runs first on both paths; a structurally booby-trapped
    model passes untouched on CPU."""
    from unturtle.models.backbones.llada import fast_paths

    new_model, new_report = _wrap(_tiny_llada(), monkeypatch, device="cpu")
    old_model, old_report = _wrap(
        _tiny_llada(), monkeypatch, patcher=oracle_patch_llada_peft, device="cpu"
    )
    assert new_report.to_dict() == old_report.to_dict()
    assert new_report.applied == {} and new_report.fallback is None
    assert _fast_identity(new_model) == _fast_identity(old_model)

    monkeypatch.undo()

    class _Booby(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        @property
        def base_model(self):
            raise AssertionError("structure must not be touched on CPU")

    assert fast_paths.patch_peft(_Booby().to("cpu"), 0.0, "none") == (0, 0, 0)
    assert oracle_patch_llada_peft(_Booby().to("cpu"), 0.0, "none") == (0, 0, 0)


@pytest.mark.gpu
def test_missing_structure_failopen_matches_oracle(monkeypatch):
    """Unresolvable transformer on CUDA: warn + zeros on both paths (this
    family's historical fail-open); via the façade it is the typed fallback."""
    if not torch.cuda.is_available():
        pytest.skip("the structure path is only reached on CUDA")
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.llada import fast_paths

    def odd():
        m = torch.nn.Module()
        m.lin = torch.nn.Linear(2, 2)
        m.base_model = types.SimpleNamespace(model=types.SimpleNamespace())
        return m.cuda()

    seen_new: list[str] = []
    monkeypatch.setattr(fast_paths, "warn_once", seen_new.append)
    counts_new = fast_paths.patch_peft(odd(), 0.0, "none")
    monkeypatch.undo()
    seen_old: list[str] = []
    monkeypatch.setitem(globals(), "_warn_once", seen_old.append)
    counts_old = oracle_patch_llada_peft(odd(), 0.0, "none")
    assert counts_new == counts_old == (0, 0, 0)
    assert tuple(seen_new) == tuple(seen_old) and len(seen_new) == 1
    assert "could not locate transformer" in seen_new[0]

    class _Odd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(model_type="llada")
            self.lin = torch.nn.Linear(2, 2)
            self.base_model = types.SimpleNamespace(model=types.SimpleNamespace())

    seen: list[str] = []
    monkeypatch.setattr("unturtle.fast_diffusion_model._warn_once", seen.append)
    report = FastDiffusionModel.patch_peft_model_with_report(_Odd())
    assert report.fallback == "structure_mismatch"
    assert report.support.reason == "structure_mismatch"
    assert report.applied == {} and report.skipped == {}


def test_report_line_matches_oracle_verbatim():
    from unturtle.models.backbones.llada import fast_paths

    model = _tiny_llada()

    class _Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.base_model = types.SimpleNamespace(model=inner)

    wrapped = _Wrapped(model)
    for counts in ((0, 0, 0), (2, 2, 2), (1, 0, 2)):
        assert fast_paths.report(wrapped, counts) == oracle_llada_report(
            wrapped, counts
        ), counts


# ============================== INTENDED DELTA ==============================


@pytest.mark.gpu
def test_hooks_are_live_forward_and_backward(pair):
    """The wired forward actually calls the provider-installed hooks: QKV and O
    probe counters positive on every block, and liveness extends to the LoRA
    backward gradients — not just installation, not just forward."""
    from unturtle.fast_diffusion_model import probe_liveness

    new_model, report = pair["new"]
    ids = torch.randint(2, 500, (2, 16), device="cuda")
    liveness = probe_liveness(
        new_model, {"input_ids": ids}, backward=True, applied=report.applied
    )
    by_kind: dict[str, list[int]] = {}
    for key, count in liveness.forward.items():
        by_kind.setdefault(key.rsplit(":", 1)[1], []).append(count)
    assert len(by_kind["qkv"]) == 2 and all(v >= 1 for v in by_kind["qkv"]), by_kind
    assert len(by_kind["o"]) == 2 and all(v >= 1 for v in by_kind["o"]), by_kind
    assert all(v >= 1 for v in by_kind["rope"]), by_kind
    assert all(v >= 1 for v in by_kind["mlp"]), by_kind
    assert liveness.forward_live is True
    assert liveness.backward_live is True and liveness.live is True


def test_default_stubs_are_bit_identical_to_direct_projections():
    """Standard / non-PEFT path keeps its historical outputs: the new default
    stubs compute exactly the direct projections, and an unpatched forward
    equals a forward with the stubs swapped for explicit direct-call lambdas."""
    from unturtle.models.backbones.llada.modeling_llada import (
        LLaDABlock,
        LLaDALlamaBlock,
    )

    model = _tiny_llada()
    inner = model.model.transformer
    block = inner.blocks[0]
    x = torch.randn(2, 8, 64)
    q, k, v = LLaDALlamaBlock._default_apply_qkv(block, x)
    assert torch.equal(q, block.q_proj(x))
    assert torch.equal(k, block.k_proj(x))
    assert torch.equal(v, block.v_proj(x))
    att = torch.randn(2, 8, 64)
    assert torch.equal(LLaDABlock._default_apply_o(block, att), block.attn_out(att))

    ids = torch.randint(2, 500, (2, 16))
    with torch.no_grad():
        baseline = model(input_ids=ids).logits
    for blk in inner.blocks:
        blk.apply_qkv = lambda self, x: (
            self.q_proj(x),
            self.k_proj(x),
            self.v_proj(x),
        )
        blk.apply_o = lambda self, a: self.attn_out(a)
    with torch.no_grad():
        direct = model(input_ids=ids).logits
    assert torch.equal(baseline, direct)


@pytest.mark.gpu
def test_ineligible_qkv_o_fall_back_without_false_liveness(monkeypatch):
    """A biased-QKV LLaDA config: QKV/O still work through the default stubs,
    the report does not claim them applied, and no fake live is reported."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from unturtle.fast_diffusion_model import FastDiffusionModel, probe_liveness

    torch.manual_seed(3)
    model, report = FastDiffusionModel.get_peft_model_with_report(
        _tiny_llada(include_qkv_bias=True).cuda(),
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "ff_proj", "up_proj", "ff_out"],
        use_gradient_checkpointing=False,
        random_state=7,
    )
    assert "qkv" not in report.applied and "o" not in report.applied, report.applied
    assert report.applied.get("mlp") and report.applied.get("rope")
    ids = torch.randint(2, 500, (2, 16), device="cuda")
    liveness = probe_liveness(model, {"input_ids": ids}, applied=report.applied)
    assert liveness.live is True  # what IS applied executes...
    model.eval()
    with torch.no_grad():
        out = model(input_ids=ids).logits  # ...and the stub QKV/O path still runs
    assert out.shape[:2] == (2, 16)
    # no false fast-claim anywhere: the report lists the blocks as skipped, the
    # installed callables are the default stubs (not the fused kernels), and the
    # observation layer agrees.
    from unturtle.fast_diffusion_model import _observe_fast_paths
    from unturtle.models.backbones.llada.modeling_llada import LLaDALlamaBlock

    assert len(report.skipped.get("qkv", ())) == 2, report.skipped
    for block in _blocks(model):
        assert block.__dict__["apply_qkv"] is LLaDALlamaBlock._default_apply_qkv
        assert not hasattr(block, "o_proj")  # no alias without the fast hook
    observed = _observe_fast_paths(model)["applied"]
    assert "qkv" not in observed and "o" not in observed, observed


def test_series_end_facade_owns_no_family():
    """4/4: zero family patchers in the façade; the registry resolves every
    family's provider; the provider does not import the façade."""
    import ast
    import inspect

    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.backbones.llada import fast_paths
    from unturtle.models.integrations import find_peft_integration

    remaining = [n for n in dir(fdm) if n.startswith("_patch_") and n.endswith("_peft")]
    assert remaining == [], remaining
    integration = find_peft_integration("llada")
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


def test_requested_kinds_and_predicates_are_family_owned():
    from unturtle.models.backbones.llada import fast_paths

    assert fast_paths.requested_kinds(["q_proj"], on_cuda=False) == ("qkv",)
    assert fast_paths.requested_kinds(["attn_out", "up_proj"], on_cuda=True) == (
        "o",
        "mlp",
        "rope",
    )
    assert fast_paths.requested_kinds(["ff_out"], on_cuda=False) == ("mlp",)
    # other families' names are NOT this family's business
    assert fast_paths.requested_kinds(["o_proj", "Wqkv", "gate_proj"], False) == ()

    model = _tiny_llada()
    from peft import LoraConfig, TaskType, get_peft_model

    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            target_modules=TARGETS,
            task_type=TaskType.FEATURE_EXTRACTION,
            lora_dropout=0,
            bias="none",
        ),
    )
    block = _blocks(peft_model)[0]
    targets = fast_paths.block_targets(block)
    assert fast_paths.qkv_applicable(targets)
    assert fast_paths.o_applicable(targets)
    assert fast_paths.mlp_applicable(targets)  # SiLU config
    # SiLU is part of MLP applicability (SwiGLU would shape-mismatch chunk(2))
    no_silu = dict(targets)
    no_silu["act"] = torch.nn.GELU()
    assert not fast_paths.mlp_applicable(no_silu)
    # bias-free is part of the LLaDA QKV gate (unlike Dream's bias kernel)
    biased = dict(targets)
    biased["k_proj"] = torch.nn.Linear(4, 4, bias=True)
    biased["k_proj"].lora_A = torch.nn.ModuleDict()
    assert not fast_paths.qkv_applicable(biased)
    assert not fast_paths.o_applicable({**targets, "attn_out": None})


def test_check_structure_states_are_typed():
    from unturtle.models.backbones.llada import fast_paths

    no_transformer = types.SimpleNamespace(
        base_model=types.SimpleNamespace(model=types.SimpleNamespace())
    )
    result = fast_paths.check_structure(no_transformer)
    assert result.status == "unsupported" and result.reason == "structure_mismatch"
    assert result.details["missing"] == "transformer"

    no_blocks = types.SimpleNamespace(
        base_model=types.SimpleNamespace(
            model=types.SimpleNamespace(transformer=object())
        )
    )
    result = fast_paths.check_structure(no_blocks)
    assert result.status == "unsupported" and result.reason == "structure_mismatch"
    assert result.details["missing"] == "blocks"

    ok = fast_paths.check_structure(
        types.SimpleNamespace(base_model=types.SimpleNamespace(model=_tiny_llada()))
    )
    assert ok.status == "supported" and ok.details["blocks"] == 2
