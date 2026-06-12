from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _FakeInstance:
    def __init__(self, context: str, gen_kwargs: dict[str, Any]) -> None:
        self.args = (context, gen_kwargs)


def _install_fake_lm_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal fake lm_eval.api.model.LM so the adapter can subclass it."""
    api_mod = types.ModuleType("lm_eval.api")
    model_mod = types.ModuleType("lm_eval.api.model")

    class _LM:
        def __init__(self) -> None:
            pass

    model_mod.LM = _LM
    root = types.ModuleType("lm_eval")
    root.api = api_mod
    api_mod.model = model_mod
    monkeypatch.setitem(sys.modules, "lm_eval", root)
    monkeypatch.setitem(sys.modules, "lm_eval.api", api_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.api.model", model_mod)


class _StubModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, input_ids, *, algorithm="auto", **kwargs):
        import torch

        self.calls.append({"algorithm": algorithm, **kwargs})
        new = torch.tensor([[101, 102, 103]])
        return torch.cat([input_ids, new], dim=1)

    def parameters(self):
        import torch

        yield torch.zeros(1)


class _StubTokenizer:
    model_max_length = 4096

    def encode(self, text, return_tensors=None, add_special_tokens=True):
        import torch

        return torch.tensor([[1, 2, 3, 4]])

    def decode(self, ids, skip_special_tokens=True):
        return "the answer is 42 STOP trailing"


def test_build_harness_lm_requires_lm_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    from unturtle.eval.harness.model_adapter import build_harness_lm

    with pytest.raises(ImportError):
        build_harness_lm(
            model=_StubModel(),
            tokenizer=_StubTokenizer(),
            num_steps=4,
            max_new_tokens=8,
            temperature=0.0,
            use_chat_template=False,
        )


def test_generate_until_routes_through_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lm_eval(monkeypatch)
    from unturtle.eval.harness.model_adapter import build_harness_lm

    model = _StubModel()
    lm = build_harness_lm(
        model=model,
        tokenizer=_StubTokenizer(),
        num_steps=4,
        max_new_tokens=8,
        temperature=0.0,
        use_chat_template=False,
    )
    out = lm.generate_until([_FakeInstance("Q: 2+2?", {"until": ["STOP"]})])
    assert isinstance(out, list) and len(out) == 1
    assert model.calls, "generate was not called"
    assert "STOP" not in out[0]
    assert out[0].startswith("the answer is 42")
    # Verify steps, temperature, mask_token_id, and algorithm pin are forwarded correctly
    assert model.calls[-1]["steps"] == 4
    assert model.calls[-1]["temperature"] == 0.0
    assert model.calls[-1]["mask_token_id"] is None
    assert model.calls[-1]["algorithm"] == "mdlm"


def test_generate_until_handles_string_until(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # lm-eval passes ``until`` as a bare string for many tasks (e.g. until: "STOP").
    # A naive list(until) would split it into characters and truncate at the first
    # matching char; the adapter must wrap a string into a single-element list.
    _install_fake_lm_eval(monkeypatch)
    from unturtle.eval.harness.model_adapter import build_harness_lm

    lm = build_harness_lm(
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
        num_steps=4,
        max_new_tokens=8,
        temperature=0.0,
        use_chat_template=False,
    )
    out = lm.generate_until([_FakeInstance("Q", {"until": "trailing"})])
    # decoded stub = "the answer is 42 STOP trailing". With the bug, list("trailing")
    # truncates at the first 't' (index 0 of "the") → "". Fixed: truncates cleanly
    # before the whole word "trailing".
    assert out[0] == "the answer is 42 STOP "


def test_normalize_until_variants() -> None:
    from unturtle.eval.harness.model_adapter import _normalize_until

    assert _normalize_until(None) == []
    assert _normalize_until("Question:") == ["Question:"]
    assert _normalize_until(["a", "b"]) == ["a", "b"]


def test_generate_until_respects_max_gen_toks_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lm_eval(monkeypatch)
    from unturtle.eval.harness.model_adapter import build_harness_lm

    model = _StubModel()
    lm = build_harness_lm(
        model=model,
        tokenizer=_StubTokenizer(),
        num_steps=4,
        max_new_tokens=8,
        temperature=0.0,
        use_chat_template=False,
    )
    lm.generate_until([_FakeInstance("Q", {"until": [], "max_gen_toks": 5})])
    assert model.calls[-1]["max_length"] == 9


def test_loglikelihood_raises_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_lm_eval(monkeypatch)
    from unturtle.eval.harness.model_adapter import build_harness_lm

    lm = build_harness_lm(
        model=_StubModel(),
        tokenizer=_StubTokenizer(),
        num_steps=4,
        max_new_tokens=8,
        temperature=0.0,
        use_chat_template=False,
    )
    with pytest.raises(NotImplementedError):
        lm.loglikelihood([_FakeInstance("ctx", {})])
    with pytest.raises(NotImplementedError):
        lm.loglikelihood_rolling([_FakeInstance("ctx", {})])


# ---------------------------------------------------------------------------
# block_ar algorithm path tests
# ---------------------------------------------------------------------------


def test_block_ar_algorithm_uses_correct_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """block_ar path forwards algorithm, max_new_tokens, max_denoising_steps only."""
    _install_fake_lm_eval(monkeypatch)
    from unturtle.eval.harness.model_adapter import build_harness_lm

    model = _StubModel()
    lm = build_harness_lm(
        model=model,
        tokenizer=_StubTokenizer(),
        num_steps=48,
        max_new_tokens=256,
        temperature=0.0,
        use_chat_template=False,
        algorithm="block_ar",
    )
    lm.generate_until([_FakeInstance("Q: 2+2?", {"until": []})])
    assert model.calls, "generate was not called"
    call = model.calls[-1]
    assert call["algorithm"] == "block_ar"
    assert call["max_denoising_steps"] == 48
    assert "max_new_tokens" in call
    # block_ar must NOT forward masked-diffusion-specific kwargs
    for forbidden in ("steps", "mask_token_id", "temperature", "max_length"):
        assert forbidden not in call, (
            f"unexpected kwarg in block_ar call: {forbidden!r}"
        )


def test_mdlm_default_algorithm_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default path (no algorithm arg) still uses mdlm + masked kwargs."""
    _install_fake_lm_eval(monkeypatch)
    from unturtle.eval.harness.model_adapter import build_harness_lm

    model = _StubModel()
    lm = build_harness_lm(
        model=model,
        tokenizer=_StubTokenizer(),
        num_steps=4,
        max_new_tokens=8,
        temperature=0.5,
        use_chat_template=False,
    )
    lm.generate_until([_FakeInstance("Q", {"until": []})])
    call = model.calls[-1]
    assert call["algorithm"] == "mdlm"
    assert call["steps"] == 4
    assert call["temperature"] == 0.5
    assert "mask_token_id" in call
    assert "max_length" in call


def test_decoding_config_algorithm_field() -> None:
    """DecodingConfig has algorithm field; existing entries default to 'mdlm'."""
    from unturtle.eval.harness.configs import DecodingConfig, get_decoding_config

    # Existing entries default to mdlm
    cfg_a2d = get_decoding_config("a2d_qwen3", "gsm8k")
    assert cfg_a2d.algorithm == "mdlm"

    # diffusion_gemma entry uses block_ar
    cfg_dg = get_decoding_config("diffusion_gemma", "gsm8k")
    assert cfg_dg.algorithm == "block_ar"
    assert cfg_dg.num_steps == 48
    assert cfg_dg.max_new_tokens == 256
