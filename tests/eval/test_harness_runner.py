from __future__ import annotations

import sys
import types
from typing import Any

import pytest


def _install_fake_lm_eval(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> None:
    root = types.ModuleType("lm_eval")

    def simple_evaluate(*, model, tasks, **kwargs):  # noqa: ANN001, ANN003
        captured["model"] = model
        captured["tasks"] = tasks
        captured["kwargs"] = kwargs
        return {"results": {tasks[0]: {"exact_match,strict-match": 0.5}}}

    root.simple_evaluate = simple_evaluate
    api_mod = types.ModuleType("lm_eval.api")
    model_mod = types.ModuleType("lm_eval.api.model")

    class _LM:
        def __init__(self) -> None: ...

    model_mod.LM = _LM
    root.api = api_mod
    api_mod.model = model_mod
    monkeypatch.setitem(sys.modules, "lm_eval", root)
    monkeypatch.setitem(sys.modules, "lm_eval.api", api_mod)
    monkeypatch.setitem(sys.modules, "lm_eval.api.model", model_mod)


class _StubModel:
    def parameters(self):
        import torch

        yield torch.zeros(1)


def _patch_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    import unturtle.eval.harness.runner as runner_mod

    class _FDM:
        @staticmethod
        def from_pretrained(name, **kwargs):  # noqa: ANN001, ANN003
            return _StubModel(), object()

        @staticmethod
        def for_inference(model):  # noqa: ANN001
            return model

    monkeypatch.setattr(runner_mod, "FastDiffusionModel", _FDM)


def test_runner_records_decoding_config_in_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _install_fake_lm_eval(monkeypatch, captured)
    from unturtle.eval.harness.runner import run_harness_evaluation

    _patch_loader(monkeypatch)
    summary = run_harness_evaluation(
        model_name="dummy/model",
        model_family="a2d_qwen3",
        task="gsm8k",
    )
    assert summary["task"] == "gsm8k"
    assert summary["model"] == "dummy/model"
    assert summary["decoding_config"]["max_new_tokens"] == 256
    assert summary["decoding_config"]["num_steps"] == 256
    assert summary["decoding_config"]["task"] == "gsm8k"
    assert "results" in summary
    assert captured["tasks"] == ["gsm8k"]


def test_runner_rejects_unknown_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    _install_fake_lm_eval(monkeypatch, captured)
    from unturtle.eval.harness.runner import run_harness_evaluation

    _patch_loader(monkeypatch)
    with pytest.raises(KeyError):
        run_harness_evaluation(
            model_name="dummy/model",
            model_family="a2d_qwen3",
            task="not_a_task",
        )


def test_runner_requires_lm_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    from unturtle.eval.harness.runner import run_harness_evaluation

    _patch_loader(monkeypatch)
    with pytest.raises(ImportError):
        run_harness_evaluation(
            model_name="dummy/model",
            model_family="a2d_qwen3",
            task="gsm8k",
        )


def test_runner_threads_algorithm_to_build_harness_lm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DecodingConfig.algorithm is passed through to build_harness_lm."""
    captured: dict[str, Any] = {}
    _install_fake_lm_eval(monkeypatch, captured)
    _patch_loader(monkeypatch)

    import unturtle.eval.harness.runner as runner_mod

    class _StubHarnessLM:
        pass

    def capturing_build_harness_lm(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return _StubHarnessLM()

    monkeypatch.setattr(runner_mod, "build_harness_lm", capturing_build_harness_lm)

    from unturtle.eval.harness.runner import run_harness_evaluation

    run_harness_evaluation(
        model_name="dummy/model",
        model_family="diffusion_gemma",
        task="gsm8k",
    )
    assert captured["algorithm"] == "block_ar"
