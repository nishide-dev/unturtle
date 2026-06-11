import sys
from pathlib import Path
from types import SimpleNamespace

from examples.validate_a2d_real_inference import (
    append_result_record,
    build_dllm_sampler_config,
    build_unturtle_generation_kwargs,
    expand_runs,
    normalize_result_record,
    parse_args,
    summarize_results,
)


def test_expand_runs_smoke_for_all_backends_and_models():
    runs = expand_runs(
        backends=["unturtle", "dllm"],
        model_kinds=["mdlm", "bd3lm"],
        scenario_name="smoke",
    )

    pairs = {(run.backend, run.model_kind) for run in runs}
    assert pairs == {
        ("unturtle", "mdlm"),
        ("unturtle", "bd3lm"),
        ("dllm", "mdlm"),
        ("dllm", "bd3lm"),
    }
    assert all(run.prompt_name for run in runs)
    assert all(run.settings for run in runs)


def test_normalize_result_record_keeps_shared_fields():
    record = normalize_result_record(
        backend="unturtle",
        env_path="./.venv",
        checkpoint="dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        model_kind="mdlm",
        prompt_name="math",
        prompt_text="What is 2+2?",
        settings={"steps": 64, "max_new_tokens": 32},
        success=True,
        generated_text="4",
        runtime_seconds=1.25,
        output_tokens=1,
        exception=None,
        backend_metadata={"mask_token_id": 151643},
        git_head="deadbeef",
    )

    assert record["backend"] == "unturtle"
    assert record["model_kind"] == "mdlm"
    assert record["success"] is True
    assert record["generated_text"] == "4"
    assert record["inference_settings"] == {"steps": 64, "max_new_tokens": 32}
    assert record["backend_metadata"] == {"mask_token_id": 151643}


def test_build_unturtle_generation_kwargs_uses_shared_keys():
    kwargs = build_unturtle_generation_kwargs(
        {"steps": 64, "max_new_tokens": 32, "temperature": 0.0}
    )

    assert kwargs == {
        "steps": 64,
        "max_new_tokens": 32,
        "temperature": 0.0,
        "use_cache": False,
    }


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate_a2d_real_inference.py"])

    args = parse_args()

    assert args.backend == "all"
    assert args.model == "all"
    assert args.scenario == "smoke"
    assert args.prompt == "all"
    assert args.steps is None
    assert args.max_new_tokens is None
    assert args.output_dir == "outputs/real_inference_validation"


def test_summarize_results_mentions_failures():
    text = summarize_results(
        [
            {
                "backend": "unturtle",
                "model_kind": "mdlm",
                "success": True,
                "generated_text": "hello",
                "exception_type": None,
                "exception_message": None,
            },
            {
                "backend": "dllm",
                "model_kind": "mdlm",
                "success": False,
                "generated_text": "",
                "exception_type": "RuntimeError",
                "exception_message": "decode failed",
            },
        ]
    )

    assert "unturtle / mdlm" in text
    assert "dllm / mdlm" in text
    assert "decode failed" in text


def test_append_result_record_creates_jsonl(tmp_path: Path):
    path = append_result_record(tmp_path, {"backend": "unturtle", "success": True})

    assert path == tmp_path / "results.jsonl"
    assert path.exists()
    assert (
        path.read_text(encoding="utf-8").strip()
        == '{"backend": "unturtle", "success": true}'
    )


def test_build_dllm_sampler_config_selects_bd3lm_and_keeps_block_size(monkeypatch):
    class _MdlmConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Bd3lmConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_dllm = SimpleNamespace(
        core=SimpleNamespace(
            samplers=SimpleNamespace(
                MDLMSamplerConfig=_MdlmConfig,
                BD3LMSamplerConfig=_Bd3lmConfig,
            )
        )
    )
    monkeypatch.setitem(sys.modules, "dllm", fake_dllm)

    default_config = build_dllm_sampler_config(
        "bd3lm",
        {"steps": 128, "max_new_tokens": 64, "temperature": 0.0},
    )
    explicit_config = build_dllm_sampler_config(
        "bd3lm",
        {"steps": 128, "max_new_tokens": 64, "temperature": 0.0, "block_size": 16},
    )
    mdlm_config = build_dllm_sampler_config(
        "mdlm",
        {"steps": 64, "max_new_tokens": 32, "temperature": 0.0},
    )

    assert isinstance(default_config, _Bd3lmConfig)
    assert default_config.kwargs["block_size"] == 32
    assert default_config.kwargs["remasking"] == "low_confidence"
    assert default_config.kwargs["right_shift_logits"] is False
    assert isinstance(explicit_config, _Bd3lmConfig)
    assert explicit_config.kwargs["block_size"] == 16
    assert explicit_config.kwargs["remasking"] == "low_confidence"
    assert explicit_config.kwargs["right_shift_logits"] is False
    assert isinstance(mdlm_config, _MdlmConfig)
    assert "block_size" not in mdlm_config.kwargs
    assert mdlm_config.kwargs["remasking"] == "low_confidence"
    assert mdlm_config.kwargs["right_shift_logits"] is False
