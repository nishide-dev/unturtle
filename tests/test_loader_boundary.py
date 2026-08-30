"""#185 PR 3 — the loader boundary and façade delegation.

``unturtle.models.loading.load_model`` is the loader; the façade's
``from_pretrained`` is pure delegation returning the very ``LoadedModel`` the
loader built. Mode transitions delegate to ``peft_preparation`` (the GC-mode
owner); save/export helpers delegate to ``unturtle.save``. The post-load
``__class__`` swap stays façade-owned (#186) and reaches the loader only as
the injected ``post_load`` hook.
"""

from __future__ import annotations

import types

import pytest
import torch

pytestmark = [pytest.mark.gpu]  # unsloth import chain

from unturtle import fast_diffusion_model as fdm  # noqa: E402
from unturtle.models import loading  # noqa: E402


def _tiny_dream_checkpoint(tmp_path):
    from transformers import AutoTokenizer

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
    model.save_pretrained(tmp_path / "ckpt")
    return tmp_path / "ckpt", type(model)


def test_facade_returns_the_loader_boundary_object(tmp_path, monkeypatch):
    """from_pretrained(_with_report) is pure delegation: the SAME LoadedModel
    the loader built, with real provenance, correct concrete types, and the
    façade-owned class-swap hook injected."""
    ckpt, model_cls = _tiny_dream_checkpoint(tmp_path)

    captured = {}
    real = loading.load_model

    def spy(*args, **kwargs):
        loaded = real(*args, **kwargs)
        captured["loaded"] = loaded
        return loaded

    monkeypatch.setattr(fdm, "load_model", spy)
    loaded = fdm.FastDiffusionModel.from_pretrained_with_report(
        str(ckpt), load_in_4bit=False, max_seq_length=64, model_class=model_cls
    )
    assert loaded is captured["loaded"]
    assert loaded.load_path == "explicit_class"
    assert loaded.integration == "dream"
    assert type(loaded.model) is model_cls
    assert loaded.tokenizer is None or hasattr(loaded.tokenizer, "encode")

    model, tokenizer = fdm.FastDiffusionModel.from_pretrained(
        str(ckpt), load_in_4bit=False, max_seq_length=64, model_class=model_cls
    )
    assert type(model) is model_cls  # same concrete type on the compat entry
    assert model.max_seq_length == 64  # diffusion patch ran on the loader path


def test_load_path_provenance_matches_the_route_taken(monkeypatch):
    sentinel_model = types.SimpleNamespace(config=types.SimpleNamespace(model_type="x"))

    monkeypatch.setattr(loading, "_patch_for_diffusion", lambda m, s: m)
    monkeypatch.setattr(loading, "_load_tokenizer", lambda *a: "TOK")
    monkeypatch.setattr(loading, "_integration_name_for", lambda m: None)
    monkeypatch.setattr(
        loading, "find_quantized_linear_modules", lambda m: [], raising=True
    )

    routes = {
        "native": (lambda *a, **k: sentinel_model, lambda *a, **k: None, None),
        "upstream": (
            lambda *a, **k: None,
            lambda *a, **k: (sentinel_model, "FM_TOK"),
            None,
        ),
        "auto": (lambda *a, **k: None, lambda *a, **k: None, sentinel_model),
    }
    for expected, (native, fastmodel, auto_model) in routes.items():
        monkeypatch.setattr(loading, "_load_native", native)
        monkeypatch.setattr(loading, "_load_via_fastmodel", fastmodel)
        monkeypatch.setattr(
            loading, "_load_via_automodel", lambda *a, _m=auto_model, **k: _m
        )
        loaded = loading.load_model("stub/model", load_in_4bit=False)
        assert loaded.load_path == expected, (expected, loaded.load_path)
        if expected == "upstream":
            assert loaded.tokenizer == "FM_TOK"
            assert loaded.details["tokenizer_from_fastmodel"] is True

    class _Explicit:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return sentinel_model

    loaded = loading.load_model("stub/model", model_class=_Explicit, load_in_4bit=False)
    assert loaded.load_path == "explicit_class"


def test_quantization_and_dtype_kwargs_are_assembled_exactly():
    """The historical kwargs contract, frozen: torch_dtype, trust_remote_code,
    token, and on CUDA+bnb the nf4 double-quant BitsAndBytesConfig with the
    compute dtype and a device_map default that never overrides the caller's."""
    kwargs, dtype, eff = loading.build_load_kwargs(
        torch.float32, False, True, None, revision="r1"
    )
    assert kwargs["torch_dtype"] is torch.float32
    assert kwargs["trust_remote_code"] is True
    assert kwargs["revision"] == "r1"
    assert "token" not in kwargs and "quantization_config" not in kwargs
    assert dtype is torch.float32 and eff is False

    kwargs, _, _ = loading.build_load_kwargs(torch.float32, False, False, "tok-123")
    assert kwargs["token"] == "tok-123" and kwargs["trust_remote_code"] is False

    if torch.cuda.is_available():
        import importlib as _il

        has_bnb = _il.util.find_spec("bitsandbytes") is not None
        kwargs, dtype, eff = loading.build_load_kwargs(None, True, True, None)
        expected_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        assert dtype is expected_dtype and eff is True
        if has_bnb:
            cfg = kwargs["quantization_config"]
            assert cfg.load_in_4bit is True
            assert cfg.bnb_4bit_quant_type == "nf4"
            assert cfg.bnb_4bit_use_double_quant is True
            assert cfg.bnb_4bit_compute_dtype is expected_dtype
            assert kwargs["device_map"] == "auto"
            # the caller's device_map wins
            kwargs2, _, _ = loading.build_load_kwargs(
                None, True, True, None, device_map={"": 0}
            )
            assert kwargs2["device_map"] == {"": 0}
    else:  # CPU: 4-bit intent is gated off
        kwargs, dtype, eff = loading.build_load_kwargs(None, True, True, None)
        assert dtype is torch.float32 and eff is False
        assert "quantization_config" not in kwargs


def test_loader_does_not_touch_the_default_hub(tmp_path):
    import unturtle.registry as registry_mod

    ckpt, _ = _tiny_dream_checkpoint(tmp_path)
    hub = registry_mod._default_hub
    before = (
        None
        if hub is None
        else {
            axis: tuple(v.name for v in getattr(hub, axis).values())
            for axis in (
                "generation_algorithms",
                "backbone_integrations",
                "processes",
                "methods",
            )
            if hasattr(hub, axis)
        }
    )
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    fdm.FastDiffusionModel.from_pretrained(
        str(ckpt), load_in_4bit=False, model_class=DreamModel
    )
    hub = registry_mod._default_hub
    after = (
        None
        if hub is None
        else {
            axis: tuple(v.name for v in getattr(hub, axis).values())
            for axis in (
                "generation_algorithms",
                "backbone_integrations",
                "processes",
                "methods",
            )
            if hasattr(hub, axis)
        }
    )
    if before is not None:
        assert after == before


def test_mode_transitions_delegate_to_the_gc_mode_owner(monkeypatch):
    from unturtle.models.integrations import peft_preparation as prep

    calls = []
    monkeypatch.setattr(
        fdm, "_set_inference_mode", lambda m: calls.append(("inf", m)) or m
    )
    monkeypatch.setattr(
        fdm,
        "_set_training_mode",
        lambda m, gc: calls.append(("train", m, gc)) or m,
    )
    model = torch.nn.Linear(2, 2)
    assert fdm.FastDiffusionModel.for_inference(model) is model
    assert fdm.FastDiffusionModel.for_training(model, "unsloth") is model
    assert calls == [("inf", model), ("train", model, "unsloth")]

    # the owner's context manager round-trips the tracked mode
    model.train()
    model._unturtle_gradient_checkpointing_mode = "unsloth"
    with fdm.FastDiffusionModel.inference_context(model) as m:
        assert m is model and not model.training
    assert model.training
    assert prep.get_gradient_checkpointing_mode(model) == "unsloth"


def test_save_helpers_delegate_to_unturtle_save(monkeypatch):
    import unturtle.save as usave

    calls = {}
    monkeypatch.setattr(
        usave,
        "save_pretrained_merged",
        lambda model, d, tokenizer=None, safe_serialization=True, **kw: calls.update(
            merged=(model, d, tokenizer, safe_serialization)
        ),
    )
    monkeypatch.setattr(
        usave,
        "save_lora_adapter",
        lambda model, d, tokenizer=None: calls.update(adapter=(model, d, tokenizer)),
    )
    m = object()
    fdm.FastDiffusionModel.save_pretrained_merged(m, "outdir", tokenizer="T")
    fdm.FastDiffusionModel.save_lora_adapter(m, "outdir2")
    assert calls["merged"] == (m, "outdir", "T", True)
    assert calls["adapter"] == (m, "outdir2", None)


def test_loader_owns_no_class_swap_and_never_imports_the_facade():
    """#186 done: no runtime class swap exists anywhere; the wrapper-ordering
    map lives ONLY in the loader; the loader never imports the façade."""
    import ast
    import inspect

    assert not hasattr(fdm, "_apply_post_load_class_swap")
    assert not hasattr(loading, "_apply_post_load_class_swap")
    assert not hasattr(fdm, "_POST_LOAD_CLASS_SWAPS")
    assert isinstance(loading._POST_LOAD_CLASS_SWAPS, dict)
    tree = ast.parse(inspect.getsource(loading))
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
    for name, value in vars(loading).items():
        assert getattr(value, "__module__", "") != fdm.__name__, name
