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

"""#185 PR 0 — typed result/report contracts, behavior unchanged.

Mutant coverage: installed marked live · unverified collapsed to supported ·
fallback omitted · counter on the wrong module · report generation mutating
the default hub · random_state applied outside fork_rng · report path
returning a different object than the compatibility path.
"""

from __future__ import annotations

import copy
import importlib.util
import json

import pytest
import torch

pytestmark = [pytest.mark.gpu]  # unsloth import chain

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _tiny_dream(dtype=None):
    from unturtle.models.backbones.dream.configuration_dream import DreamConfig
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    torch.manual_seed(0)
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


def _hub_snapshot():
    import unturtle.registry as registry_mod

    hub = registry_mod._default_hub
    return (
        None
        if hub is None
        else {
            axis: [v.name for v in getattr(hub, axis).values()]
            for axis in (
                "generation_algorithms",
                "backbone_integrations",
                "processes",
                "methods",
            )
        }
    )


# ---------------------------------------------------------------------------
# types
# ---------------------------------------------------------------------------


def test_support_result_is_three_valued_and_typed():
    from unturtle.models.integrations.reports import SupportResult

    assert SupportResult("supported").status == "supported"
    unverified = SupportResult("unverified", reason="input_embedding_unresolvable")
    assert unverified.status == "unverified"
    with pytest.raises(ValueError):
        SupportResult("compatible")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SupportResult("unverified")  # a reason is mandatory


def test_unverified_survives_serialization():
    from unturtle.models.integrations.reports import PatchReport, SupportResult

    report = PatchReport(
        family="dream",
        model_type="Dream",
        support=SupportResult("unverified", reason="input_embedding_unresolvable"),
        requested=("qkv",),
        applied={"qkv": ("model.layers.0.self_attn",)},
        skipped={},
        fallback=None,
        applicability={},
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["support"]["status"] == "unverified"
    assert payload["support"]["reason"] == "input_embedding_unresolvable"
    assert payload["is_fast"] is True and payload["live"] is False


# ---------------------------------------------------------------------------
# dtype gate → SupportResult (unverified is first-class)
# ---------------------------------------------------------------------------


def test_gate_reports_unverified_when_embedding_unresolvable():
    from unturtle.fast_diffusion_model import (
        _fast_path_dtype_incompatibility,
        _fast_path_support,
    )

    class _FakeQuantState:
        dtype = torch.bfloat16

    class _Quantized(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            w = torch.nn.Parameter(
                torch.zeros(8, 1, dtype=torch.uint8), requires_grad=False
            )
            w.quant_state = _FakeQuantState()
            self.proj.weight = w

        def get_input_embeddings(self):
            raise NotImplementedError("not auto-handled")

    model = _Quantized()
    support = _fast_path_support(model)
    assert support.status == "unverified"
    assert support.reason == "input_embedding_unresolvable"
    # production stays fail-open (behavior unchanged): no blocking reason
    assert _fast_path_dtype_incompatibility(model) is None


def test_gate_reports_unsupported_with_typed_reason():
    from unturtle.fast_diffusion_model import _fast_path_support

    class _FakeQuantState:
        dtype = torch.bfloat16

    class _Quantized(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)  # fp32 vs bf16 quant
            self.proj = torch.nn.Linear(4, 4)
            w = torch.nn.Parameter(
                torch.zeros(8, 1, dtype=torch.uint8), requires_grad=False
            )
            w.quant_state = _FakeQuantState()
            self.proj.weight = w

        def get_input_embeddings(self):
            return self.embed

    support = _fast_path_support(_Quantized())
    assert (support.status, support.reason) == (
        "unsupported",
        "incompatible_compute_dtype",
    )


# ---------------------------------------------------------------------------
# report vs facade — observed installation, liveness, identity
# ---------------------------------------------------------------------------


@pytest.fixture()
def cuda_dream_report():
    if not torch.cuda.is_available():
        pytest.skip("fast paths need CUDA")
    from unturtle.fast_diffusion_model import FastDiffusionModel

    model, report = FastDiffusionModel.get_peft_model_with_report(
        _tiny_dream(torch.bfloat16).cuda(),
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGETS,
        use_gradient_checkpointing=False,
    )
    return model, report


def test_report_describes_observed_installation_not_liveness(cuda_dream_report):
    _, report = cuda_dream_report
    assert report.support.status == "supported"
    assert report.fallback is None
    assert set(report.requested) == {"qkv", "o", "mlp", "attention_forward"}
    for kind in ("qkv", "o", "mlp", "attention_forward"):
        assert len(report.applied[kind]) == 2, (kind, report.applied)
    assert report.is_fast is True
    assert report.liveness is None and report.live is False  # installed ≠ live


def test_liveness_only_after_actual_forward(cuda_dream_report):
    from unturtle.fast_diffusion_model import probe_liveness

    model, report = cuda_dream_report
    ids = torch.randint(2, 400, (1, 8), device="cuda")
    liveness = probe_liveness(model, {"input_ids": ids}, applied=report.applied)
    assert liveness.forward_live is True and liveness.live is True
    assert set(liveness.forward) == set(report.applied_targets)
    assert all(v >= 1 for v in liveness.forward.values())
    assert liveness.backward is None and liveness.backward_live is None
    # the probe restored the originals: hooks still installed by identity
    from unturtle.fast_diffusion_model import _observe_fast_paths

    assert _observe_fast_paths(model)["applied"] == report.applied


def test_backward_liveness_is_separate(cuda_dream_report):
    from unturtle.fast_diffusion_model import probe_liveness

    model, report = cuda_dream_report
    ids = torch.randint(2, 400, (2, 8), device="cuda")
    liveness = probe_liveness(
        model, {"input_ids": ids}, backward=True, applied=report.applied
    )
    assert liveness.forward_live is True
    assert liveness.backward_live is True
    assert liveness.live is True
    assert all(v >= 1 for v in liveness.backward.values())


def test_counter_on_wrong_module_cannot_vouch(cuda_dream_report):
    """Liveness is per applied target: an `applied` map naming a module the
    forward never reaches yields live=False even though other counters fire."""
    from unturtle.fast_diffusion_model import probe_liveness

    model, report = cuda_dream_report
    wrong = dict(report.applied)
    wrong["qkv"] = ("model.layers.0.self_attn", "model.layers.99.self_attn")
    ids = torch.randint(2, 400, (1, 8), device="cuda")
    liveness = probe_liveness(model, {"input_ids": ids}, applied=wrong)
    assert liveness.forward["model.layers.99.self_attn:qkv"] == 0
    assert liveness.live is False


def test_fallback_is_recorded_separately_from_skipped(tmp_path):
    """fp32-upcasted 4-bit model: whole fast set withheld → fallback typed,
    nothing applied, skipped empty (not conflated), standard path executes."""
    if (
        not torch.cuda.is_available()
        or importlib.util.find_spec("bitsandbytes") is None
    ):
        pytest.skip("4-bit fixture needs CUDA + bitsandbytes")
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    _tiny_dream(torch.bfloat16).save_pretrained(tmp_path)
    loaded = FastDiffusionModel.from_pretrained_with_report(
        str(tmp_path),
        max_seq_length=64,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map={"": "cuda:0"},
        model_class=DreamModel,
    )
    assert loaded.load_path == "explicit_class" and loaded.details["quantized"] is True
    for param in loaded.model.parameters():
        if param.dtype == torch.bfloat16 and type(param).__name__ != "Params4bit":
            param.data = param.data.float()
    model, report = FastDiffusionModel.get_peft_model_with_report(
        loaded.model,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGETS,
        use_gradient_checkpointing=False,
    )
    assert report.support.status == "unsupported"
    assert report.fallback == "incompatible_compute_dtype"
    assert report.applied == {} and report.skipped == {}
    assert report.is_fast is False
    out = model(input_ids=torch.randint(2, 400, (1, 8), device="cuda"))
    assert torch.isfinite(out.logits).all()


def test_report_generation_does_not_mutate_model_or_default_hub():
    """Snapshot the default hub BEFORE any report is generated (not after a
    fixture already ran one), then wrap + report + probe repeatedly."""
    if not torch.cuda.is_available():
        pytest.skip("fast paths need CUDA")
    from unturtle.fast_diffusion_model import FastDiffusionModel, probe_liveness

    hub_before = _hub_snapshot()
    model, report = FastDiffusionModel.get_peft_model_with_report(
        _tiny_dream(torch.bfloat16).cuda(),
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGETS,
        use_gradient_checkpointing=False,
    )
    assert _hub_snapshot() == hub_before, (
        "get_peft_model_with_report touched the default hub"
    )
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    ids = torch.randint(2, 400, (1, 8), device="cuda")
    for _ in range(2):
        again = FastDiffusionModel.patch_peft_model_with_report(
            model, lora_dropout=0.0, bias="none"
        )
        probe_liveness(model, {"input_ids": ids}, applied=again.applied)
        assert again.applied == report.applied
        assert _hub_snapshot() == hub_before, (
            "report generation touched the default hub"
        )
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key]), key


def test_compat_entry_points_return_the_report_paths_objects(monkeypatch):
    """The compatibility facade IS the report path: same objects, not a parallel build."""
    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.integrations.reports import (
        LoadedModel,
        PatchReport,
        SupportResult,
    )

    sentinel_model, sentinel_tok = object(), object()
    monkeypatch.setattr(
        fdm.FastDiffusionModel,
        "from_pretrained_with_report",
        staticmethod(
            lambda *a, **k: LoadedModel(sentinel_model, sentinel_tok, None, "native")
        ),
    )
    assert fdm.FastDiffusionModel.from_pretrained("x") == (sentinel_model, sentinel_tok)
    report = PatchReport("f", "t", SupportResult("supported"), (), {}, {}, None, {})
    monkeypatch.setattr(
        fdm.FastDiffusionModel,
        "get_peft_model_with_report",
        staticmethod(lambda model, **k: (sentinel_model, report)),
    )
    assert fdm.FastDiffusionModel.get_peft_model(object()) is sentinel_model


def test_random_state_stays_inside_fork_rng_on_report_path():
    from unturtle.fast_diffusion_model import FastDiffusionModel

    def wrap(pre):
        model = _tiny_dream()
        torch.manual_seed(100)
        if pre:
            torch.randn(pre)
        before = torch.get_rng_state().clone()
        peft, _ = FastDiffusionModel.get_peft_model_with_report(
            model,
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj"],
            use_gradient_checkpointing=False,
            random_state=3407,
        )
        assert torch.equal(before, torch.get_rng_state())
        return (
            next(p for n, p in peft.named_parameters() if ".lora_A." in n)
            .detach()
            .clone()
        )

    assert torch.equal(wrap(0), wrap(7))


def test_loaded_model_provenance_native_path(tmp_path):
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    _tiny_dream().save_pretrained(tmp_path)
    loaded = FastDiffusionModel.from_pretrained_with_report(
        str(tmp_path),
        max_seq_length=64,
        dtype=torch.float32,
        load_in_4bit=False,
        model_class=DreamModel,
    )
    assert loaded.load_path == "explicit_class"
    assert loaded.integration == "dream"
    assert loaded.details["class_swapped"] is False
    model, tokenizer = loaded.as_tuple()
    assert model is loaded.model and tokenizer is loaded.tokenizer
    assert copy.copy(loaded.details)["quantized"] is False


def _tiny_llada():
    from unturtle.models.backbones.llada.configuration_llada import LLaDAConfig
    from unturtle.models.backbones.llada.modeling_llada import LLaDAModelLM

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
        include_qkv_bias=False,
        weight_tying=False,
    )
    return LLaDAModelLM(config).eval()


def test_rope_targets_are_counted_and_llada_qkv_o_are_now_live():
    """LLaDA installs a bound fast rope forward: the probe counts it. Since the
    #185 LLaDA wiring, ``apply_qkv`` / ``apply_o`` are dispatched by the real
    forward as well — the report's applied hooks must ALL prove live (this test
    froze the installed-not-live defect until the wiring PR fixed it)."""
    if not torch.cuda.is_available():
        pytest.skip("fast paths need CUDA")
    from unturtle.fast_diffusion_model import FastDiffusionModel, probe_liveness

    model, report = FastDiffusionModel.get_peft_model_with_report(
        _tiny_llada().cuda(),
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "attn_out"],
        use_gradient_checkpointing=False,
    )
    assert len(report.applied.get("rope", ())) == 2, report.applied
    assert len(report.applied["qkv"]) == 2 and len(report.applied["o"]) == 2
    ids = torch.randint(2, 400, (1, 8), device="cuda")
    liveness = probe_liveness(model, {"input_ids": ids}, applied=report.applied)
    by_kind = {}
    for key, count in liveness.forward.items():
        by_kind.setdefault(key.rsplit(":", 1)[1], []).append(count)
    assert all(v >= 1 for v in by_kind["rope"]), by_kind
    assert all(v >= 1 for v in by_kind["qkv"]), by_kind  # wired live (#185)
    assert all(v >= 1 for v in by_kind["o"]), by_kind
    assert report.is_fast is True and liveness.live is True


def test_probe_restores_originals_when_one_attr_is_installed_twice(cuda_dream_report):
    """A caller-supplied `applied` map may list one module under two kinds that both
    resolve to `forward`; restore must be LIFO so the pre-probe original survives."""
    from unturtle.fast_diffusion_model import probe_liveness

    model, report = cuda_dream_report
    path = report.applied["attention_forward"][0]
    module = dict(model.named_modules())[path]
    original = module.__dict__["forward"]
    applied = {"attention_forward": (path,), "rope": (path,)}
    ids = torch.randint(2, 400, (1, 8), device="cuda")
    liveness = probe_liveness(model, {"input_ids": ids}, applied=applied)
    assert module.__dict__["forward"] is original
    assert liveness.forward[f"{path}:attention_forward"] >= 1
    assert liveness.forward[f"{path}:rope"] >= 1
