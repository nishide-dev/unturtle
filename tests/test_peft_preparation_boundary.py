"""#185 PR 2 — the PEFT-preparation / optimization boundary.

Preparation (stubs, k-bit, gradient checkpointing, forked-RNG adapter
creation) is one owner; fast-path optimization is a separate, optional step on
the prepared model. These tests pin the boundary and its #188 contract.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu]  # unsloth import chain

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _tiny_dream(dtype=None, seed: int = 0):
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


def _prepare(model, random_state=1234, use_gradient_checkpointing=False):
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.integrations.peft_preparation import build_lora_config

    return FastDiffusionModel.prepare_peft_model(
        model,
        build_lora_config(r=4, target_modules=TARGETS, lora_alpha=4),
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
    )


def test_prepared_model_is_wrapped_but_not_optimized():
    """The boundary: preparation yields a LoRA-wrapped, stub-carrying model with
    ZERO fast hooks — optimization is a separate step that then installs them."""
    from peft import PeftModel

    from unturtle.fast_diffusion_model import FastDiffusionModel, _observe_fast_paths
    from unturtle.models.integrations.reports import PreparedPeftModel

    prepared = _prepare(_tiny_dream())
    assert isinstance(prepared, PreparedPeftModel)
    assert isinstance(prepared.model, PeftModel)
    assert prepared.quantized is False and prepared.kbit_prepared is False
    assert prepared.random_state == 1234
    from peft import TaskType

    # FEATURE_EXTRACTION avoids PeftModelForCausalLM guards in unsloth
    assert prepared.lora_config.task_type == TaskType.FEATURE_EXTRACTION
    assert _observe_fast_paths(prepared.model)["applied"] == {}
    # stubs are preparation's job (dispatch protocol), installed pre-wrap
    for _, module in prepared.model.named_modules():
        if hasattr(module, "q_proj") and hasattr(module, "o_proj"):
            assert hasattr(module, "apply_qkv") and hasattr(module, "apply_o")

    if torch.cuda.is_available():
        prepared.model.cuda()
        report = FastDiffusionModel.patch_peft_model_with_report(
            prepared.model, lora_dropout=0.0, bias="none"
        )
        assert report.applied  # optimization installs ONLY after the boundary


def test_split_flow_equals_the_one_shot_facade(tmp_path):
    """prepare -> optimize -> saving-patch reproduces get_peft_model_with_report
    exactly: same trainable set, same adapters (same random_state), same state
    dict, same report fields."""
    if not torch.cuda.is_available():
        pytest.skip("optimization step needs CUDA")
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.save import patch_saving_functions

    one_shot, one_report = FastDiffusionModel.get_peft_model_with_report(
        _tiny_dream(torch.bfloat16).cuda(),
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGETS,
        use_gradient_checkpointing=False,
        random_state=1234,
    )
    prepared = _prepare(_tiny_dream(torch.bfloat16).cuda())
    split_report = FastDiffusionModel.patch_peft_model_with_report(
        prepared.model, lora_dropout=0.0, bias="none"
    )
    patch_saving_functions(prepared.model)
    split = prepared.model

    sd_one, sd_split = one_shot.state_dict(), split.state_dict()
    assert sd_one.keys() == sd_split.keys()
    for k in sd_one:
        assert torch.equal(sd_one[k], sd_split[k]), k
    assert {n for n, p in one_shot.named_parameters() if p.requires_grad} == {
        n for n, p in split.named_parameters() if p.requires_grad
    }
    for field in ("requested", "applied", "skipped", "fallback", "family"):
        assert getattr(one_report, field) == getattr(split_report, field), field


def test_rng_fork_sits_exactly_around_adapter_creation():
    """#188 at the boundary: preparation's other steps (stubs, GC wiring) run
    OUTSIDE the fork — the caller RNG is untouched by the whole call, and the
    same random_state gives identical adapters regardless of prior consumption."""
    model_a = _tiny_dream(seed=0)
    model_b = _tiny_dream(seed=0)
    torch.manual_seed(99)
    torch.randn(1000)  # arbitrary prior consumption
    before = torch.get_rng_state()
    prepared_a = _prepare(model_a, use_gradient_checkpointing=True)
    assert torch.equal(before, torch.get_rng_state())

    torch.randn(37)  # different subsequent consumption
    prepared_b = _prepare(model_b, use_gradient_checkpointing=True)
    sd_a, sd_b = prepared_a.model.state_dict(), prepared_b.model.state_dict()
    lora_keys = [k for k in sd_a if "lora_A" in k]
    assert lora_keys
    for k in lora_keys:
        assert torch.equal(sd_a[k], sd_b[k]), k


def test_gradient_checkpointing_mode_is_applied_and_round_trips():
    from unturtle.models.integrations.peft_preparation import (
        get_gradient_checkpointing_mode,
    )

    prepared = _prepare(_tiny_dream(), use_gradient_checkpointing=True)
    assert prepared.gradient_checkpointing is True
    assert get_gradient_checkpointing_mode(prepared.model) is True
    flags = [
        m.gradient_checkpointing
        for m in prepared.model.modules()
        if hasattr(m, "gradient_checkpointing")
    ]
    assert flags and all(flags)

    prepared_off = _prepare(_tiny_dream(), use_gradient_checkpointing=False)
    assert get_gradient_checkpointing_mode(prepared_off.model) is False

    # "unsloth" must round-trip as the string, not collapse to True — the
    # tracked mode attr (set on the PEFT wrapper) is what preserves it
    prepared_unsloth = _prepare(_tiny_dream(), use_gradient_checkpointing="unsloth")
    assert get_gradient_checkpointing_mode(prepared_unsloth.model) == "unsloth"


def test_quantized_models_take_the_kbit_path(monkeypatch):
    """Quantization markers route preparation through prepare_model_for_kbit_training
    (the #177 boundary), recorded in the typed result."""
    from unturtle.models.integrations import peft_preparation as prep

    seen = {}

    def fake_kbit(model, *, use_gradient_checkpointing, use_reentrant):
        seen["called"] = (use_gradient_checkpointing, use_reentrant)
        return model

    monkeypatch.setattr(prep, "prepare_model_for_kbit_training", fake_kbit)
    model = _tiny_dream()
    model.is_loaded_in_4bit = True
    prepared = prep.prepare_peft_model(
        model,
        prep.build_lora_config(r=4, target_modules=TARGETS, lora_alpha=4),
        use_gradient_checkpointing="unsloth",
        random_state=7,
    )
    assert prepared.quantized is True and prepared.kbit_prepared is True
    assert seen["called"] == ("unsloth", True)
    # On the k-bit path the GC helper never runs, so the tracked mode must be
    # persisted on the wrapper itself — "unsloth" must not collapse to a bool.
    from unturtle.models.integrations.peft_preparation import (
        get_gradient_checkpointing_mode,
    )

    assert get_gradient_checkpointing_mode(prepared.model) == "unsloth"

    assert prep.is_quantized_model(model)
    plain = _tiny_dream()
    assert not prep.is_quantized_model(plain)


def test_facade_flows_the_prepared_model_through(monkeypatch):
    """get_peft_model(_with_report) returns the very object preparation built
    (boundary identity), and the saving functions are patched on it."""
    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.integrations import peft_preparation as prep

    real = prep.prepare_peft_model
    captured = {}

    def spy(model, lora_config, **kwargs):
        prepared = real(model, lora_config, **kwargs)
        captured["model"] = prepared.model
        return prepared

    monkeypatch.setattr(fdm, "prepare_peft_model", spy)
    model = fdm.FastDiffusionModel.get_peft_model(
        _tiny_dream(),
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGETS,
        use_gradient_checkpointing=False,
        random_state=7,
    )
    assert model is captured["model"]
    patched_name = model.push_to_hub.__name__
    assert patched_name != "push_to_hub" and "push_to_hub" in patched_name
