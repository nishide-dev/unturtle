from __future__ import annotations

import sys
from pathlib import Path

import benchmarks.a2d.benchmark_a2d_aligned as benchmark_a2d_aligned
from benchmarks.a2d.benchmark_a2d_aligned import (
    BACKENDS,
    BENCHMARK_MODES,
    CHECKPOINT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_STEPS,
    DLLM_MODEL_CACHE,
    ENV_PATHS,
    PROMPTS,
    UNTURTLE_MODEL_CACHE,
    WORKER_MODE,
    RunSpec,
    build_aligned_generation_kwargs,
    build_dllm_sampler_config,
    describe_backend_path,
    execute_cold_start_run,
    expand_runs,
    normalize_benchmark_record,
    parse_args,
    select_runner,
    summarize_records,
    write_summary,
)


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_a2d_aligned.py"])

    args = parse_args()

    assert args.mode == "aligned-warm"
    assert args.backend == "all"
    assert args.prompt == "all"
    assert args.steps is None
    assert args.max_new_tokens is None
    assert args.output_dir == "outputs/a2d_aligned_benchmark"
    assert args.warmup_iters == 2
    assert args.measure_iters == 5


def test_expand_runs_for_aligned_warm_uses_bd3lm_matrix_only():
    runs = expand_runs(
        backends=["unturtle", "dllm"],
        mode="aligned-warm",
        prompt_names=["math", "code"],
        steps_values=[64, 128],
        max_new_token_values=[64, 128],
    )

    assert all(isinstance(run, RunSpec) for run in runs)
    assert {run.backend for run in runs} == {"unturtle", "dllm"}
    assert {run.prompt_name for run in runs} == {"math", "code"}
    assert {run.mode for run in runs} == {"aligned-warm"}
    assert {run.prompt_text for run in runs} == {PROMPTS["math"], PROMPTS["code"]}
    assert {run.settings["steps"] for run in runs} == {64, 128}
    assert {run.settings["max_new_tokens"] for run in runs} == {64, 128}
    assert all(run.settings["block_size"] == 32 for run in runs)
    assert all(run.settings["temperature"] == 0.0 for run in runs)
    assert all(run.settings["right_shift_logits"] is False for run in runs)


def test_expand_runs_for_cold_start_keeps_same_parameter_matrix():
    runs = expand_runs(
        backends=["unturtle", "dllm"],
        mode="cold-start",
        prompt_names=["math", "code"],
        steps_values=[64, 128],
        max_new_token_values=[64, 128],
    )

    assert len(runs) == 16
    assert all(isinstance(run, RunSpec) for run in runs)
    assert {run.backend for run in runs} == {"unturtle", "dllm"}
    assert {run.prompt_name for run in runs} == {"math", "code"}
    assert {run.mode for run in runs} == {"cold-start"}
    assert {run.prompt_text for run in runs} == {PROMPTS["math"], PROMPTS["code"]}
    assert {run.settings["steps"] for run in runs} == {64, 128}
    assert {run.settings["max_new_tokens"] for run in runs} == {64, 128}
    assert all(run.settings["block_size"] == 32 for run in runs)
    assert all(run.settings["temperature"] == 0.0 for run in runs)
    assert all(run.settings["right_shift_logits"] is False for run in runs)


def test_summarize_records_groups_by_mode_backend_prompt_and_settings():
    records = [
        {
            "mode": "aligned-warm",
            "backend": "unturtle",
            "prompt_name": "math",
            "benchmark_settings": {
                "steps": 64,
                "max_new_tokens": 64,
                "block_size": 32,
                "temperature": 0.0,
                "right_shift_logits": False,
            },
            "runtime_seconds": 4.0,
            "success": True,
        },
        {
            "mode": "aligned-warm",
            "backend": "unturtle",
            "prompt_name": "math",
            "benchmark_settings": {
                "steps": 64,
                "max_new_tokens": 64,
                "block_size": 32,
                "temperature": 0.0,
                "right_shift_logits": False,
            },
            "runtime_seconds": 6.0,
            "success": True,
        },
        {
            "mode": "aligned-warm",
            "backend": "unturtle",
            "prompt_name": "math",
            "benchmark_settings": {
                "steps": 64,
                "max_new_tokens": 64,
                "block_size": 32,
                "temperature": 0.0,
                "right_shift_logits": False,
            },
            "runtime_seconds": 0.0,
            "success": False,
        },
        {
            "mode": "aligned-warm",
            "backend": "unturtle",
            "prompt_name": "math",
            "benchmark_settings": {
                "steps": 64,
                "max_new_tokens": 128,
                "block_size": 32,
                "temperature": 0.0,
                "right_shift_logits": False,
            },
            "runtime_seconds": 0.0,
            "success": False,
        },
    ]

    summary = summarize_records(records)

    assert isinstance(summary, str)
    assert summary == (
        "# A2D aligned benchmark summary\n"
        "aligned-warm / unturtle / math / steps=64 / max_new_tokens=64 / block_size=32 / temperature=0.0 / right_shift_logits=False | mean=5.00s | median=5.00s | p95=5.90s | success=2/3\n"
        "aligned-warm / unturtle / math / steps=64 / max_new_tokens=128 / block_size=32 / temperature=0.0 / right_shift_logits=False | mean=n/a | median=n/a | p95=n/a | success=0/1\n"
    )


def test_write_summary_creates_summary_md(tmp_path: Path):
    summary = (
        "# A2D aligned benchmark summary\n"
        "aligned-warm / unturtle / math / steps=64 / max_new_tokens=64 / block_size=32 / temperature=0.0 / right_shift_logits=False | mean=5.00s | median=5.00s | p95=6.00s | success=2/2\n"
    )

    path = write_summary(tmp_path, summary)

    assert path == tmp_path / "summary.md"
    assert path.read_text(encoding="utf-8") == summary


def test_append_result_record_and_write_summary_create_output_files(tmp_path: Path):
    result_path = benchmark_a2d_aligned.append_result_record(
        tmp_path,
        {"backend": "unturtle", "success": True},
    )
    summary_path = write_summary(
        tmp_path, "- aligned-warm / unturtle / math: mean=1.00s"
    )

    assert result_path == tmp_path / "results.jsonl"
    assert summary_path == tmp_path / "summary.md"
    assert (
        result_path.read_text(encoding="utf-8").strip()
        == '{"backend": "unturtle", "success": true}'
    )
    assert "aligned-warm / unturtle / math" in summary_path.read_text(encoding="utf-8")


def test_normalize_benchmark_record_keeps_mode_and_settings():
    settings = {
        "steps": 64,
        "max_new_tokens": 64,
        "block_size": 32,
        "temperature": 0.0,
        "right_shift_logits": False,
    }
    record = normalize_benchmark_record(
        mode="aligned-warm",
        backend="unturtle",
        env_path=ENV_PATHS["unturtle"],
        checkpoint=CHECKPOINT,
        prompt_name="math",
        prompt_text=PROMPTS["math"],
        settings=settings,
        success=True,
        generated_text="96 km",
        runtime_seconds=1.5,
        output_tokens=2,
        exception=None,
        backend_metadata={"path": "block_diffusion_generator"},
        git_head="deadbeef",
    )

    assert record["git_head"] == "deadbeef"
    assert record["mode"] == "aligned-warm"
    assert record["backend"] == "unturtle"
    assert record["environment_path"] == ENV_PATHS["unturtle"]
    assert record["checkpoint"] == CHECKPOINT
    assert record["prompt_name"] == "math"
    assert record["prompt_text"] == PROMPTS["math"]
    assert record["benchmark_settings"] == settings
    assert record["benchmark_settings"]["block_size"] == 32
    assert record["benchmark_settings"]["right_shift_logits"] is False
    assert record["success"] is True
    assert record["generated_text"] == "96 km"
    assert record["runtime_seconds"] == 1.5
    assert record["output_tokens"] == 2
    assert record["backend_metadata"] == {"path": "block_diffusion_generator"}
    assert record["exception_type"] is None
    assert record["exception_message"] is None


def test_build_aligned_generation_kwargs_returns_only_task3_contract_keys():
    settings = {
        "steps": 128,
        "max_new_tokens": 64,
        "block_size": 32,
        "temperature": 0.0,
        "right_shift_logits": False,
    }

    kwargs = build_aligned_generation_kwargs(settings)

    assert kwargs == {
        "max_new_tokens": 64,
        "steps": 128,
        "block_size": 32,
        "temperature": 0.0,
        "right_shift_logits": False,
    }


def test_build_dllm_sampler_config_matches_task4_bd3lm_contract():
    settings = {
        "steps": 128,
        "max_new_tokens": 64,
        "block_size": 32,
        "temperature": 0.0,
        "right_shift_logits": False,
    }

    config = build_dllm_sampler_config(settings)

    assert config == {
        "steps": 128,
        "max_new_tokens": 64,
        "temperature": 0.0,
        "block_size": 32,
        "remasking": "low_confidence",
        "right_shift_logits": False,
    }


def test_select_runner_routes_unturtle_cold_start_to_aligned_runner():
    run = RunSpec(
        mode="cold-start",
        backend="unturtle",
        prompt_name="math",
        prompt_text=PROMPTS["math"],
        settings={
            "steps": 128,
            "max_new_tokens": 64,
            "block_size": 32,
            "temperature": 0.0,
            "right_shift_logits": False,
        },
    )

    assert select_runner(run) is benchmark_a2d_aligned.run_unturtle_aligned_once


def test_execute_cold_start_run_records_separate_load_and_first_generation_timings(
    monkeypatch,
):
    run = RunSpec(
        mode="cold-start",
        backend="unturtle",
        prompt_name="math",
        prompt_text=PROMPTS["math"],
        settings={
            "steps": 128,
            "max_new_tokens": 64,
            "block_size": 32,
            "temperature": 0.0,
            "right_shift_logits": False,
        },
    )

    expected = benchmark_a2d_aligned.ExecutionResult(
        generated_text="generated",
        backend_metadata={
            "path": "block_diffusion_generator",
            "cold_start": True,
            "load_seconds": 1.5,
            "first_generation_seconds": 4.25,
        },
        output_tokens=7,
        runtime_seconds=5.75,
    )

    monkeypatch.setattr(
        benchmark_a2d_aligned, "_run_in_backend_env", lambda inner_run: expected
    )

    result = execute_cold_start_run(run)

    assert result == expected


def test_execute_cold_start_run_in_process_records_separate_load_and_first_generation_timings(
    monkeypatch,
):
    run = RunSpec(
        mode="cold-start",
        backend="unturtle",
        prompt_name="math",
        prompt_text=PROMPTS["math"],
        settings={
            "steps": 128,
            "max_new_tokens": 64,
            "block_size": 32,
            "temperature": 0.0,
            "right_shift_logits": False,
        },
    )
    clear_calls: list[str] = []
    perf_values = iter([10.0, 11.5, 20.0, 24.25])

    def fake_clear_backend_state(backend: str) -> None:
        clear_calls.append(backend)

    def fake_load_unturtle_model():
        return object(), object()

    def fake_run_unturtle_aligned_once(inner_run: RunSpec):
        assert inner_run is run
        return "generated", {"path": "block_diffusion_generator"}, 7

    monkeypatch.setattr(
        benchmark_a2d_aligned, "_clear_backend_state", fake_clear_backend_state
    )
    monkeypatch.setattr(
        benchmark_a2d_aligned, "load_unturtle_model", fake_load_unturtle_model
    )
    monkeypatch.setattr(
        benchmark_a2d_aligned,
        "run_unturtle_aligned_once",
        fake_run_unturtle_aligned_once,
    )
    monkeypatch.setattr(
        benchmark_a2d_aligned.time, "perf_counter", lambda: next(perf_values)
    )

    result = benchmark_a2d_aligned._execute_cold_start_run_in_process(run)

    assert clear_calls == ["unturtle"]
    assert result.generated_text == "generated"
    assert result.output_tokens == 7
    assert result.runtime_seconds == 5.75
    assert result.backend_metadata == {
        "path": "block_diffusion_generator",
        "cold_start": True,
        "load_seconds": 1.5,
        "first_generation_seconds": 4.25,
    }


def test_main_preserves_successful_warm_measurements_when_later_iteration_fails(
    monkeypatch, tmp_path: Path
):
    run = RunSpec(
        mode="aligned-warm",
        backend="unturtle",
        prompt_name="math",
        prompt_text=PROMPTS["math"],
        settings={
            "steps": 64,
            "max_new_tokens": 64,
            "block_size": 32,
            "temperature": 0.0,
            "right_shift_logits": False,
        },
    )
    first = benchmark_a2d_aligned.ExecutionResult(
        generated_text="first",
        backend_metadata={"path": "block_diffusion_generator"},
        output_tokens=2,
        runtime_seconds=1.25,
    )
    second_error = RuntimeError("second measurement failed")
    outcomes = iter([first, second_error])
    recorded: list[dict[str, object]] = []
    summaries: list[str] = []

    monkeypatch.setattr(
        benchmark_a2d_aligned,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "mode": "aligned-warm",
                "backend": "unturtle",
                "prompt": "math",
                "steps": 64,
                "max_new_tokens": 64,
                "output_dir": str(tmp_path),
                "warmup_iters": 0,
                "measure_iters": 2,
            },
        )(),
    )
    monkeypatch.setattr(benchmark_a2d_aligned, "expand_runs", lambda **kwargs: [run])
    monkeypatch.setattr(benchmark_a2d_aligned, "get_git_head", lambda: "deadbeef")

    def fake_run_warm_batch_in_backend_env(
        inner_run: RunSpec, *, warmup_iters: int, measure_iters: int
    ):
        assert inner_run is run
        assert warmup_iters == 0
        assert measure_iters == 2
        return [next(outcomes), next(outcomes)]

    def fake_append_result_record(output_dir: Path, record: dict[str, object]) -> Path:
        path = output_dir / "results.jsonl"
        recorded.append(record)
        return path

    def fake_write_summary(output_dir: Path, summary: str) -> Path:
        summaries.append(summary)
        return output_dir / "summary.md"

    monkeypatch.setattr(
        benchmark_a2d_aligned,
        "_run_warm_batch_in_backend_env",
        fake_run_warm_batch_in_backend_env,
    )
    monkeypatch.setattr(
        benchmark_a2d_aligned, "append_result_record", fake_append_result_record
    )
    monkeypatch.setattr(benchmark_a2d_aligned, "write_summary", fake_write_summary)

    benchmark_a2d_aligned.main()

    assert len(recorded) == 2
    assert recorded[0]["success"] is True
    assert recorded[0]["generated_text"] == "first"
    assert recorded[0]["runtime_seconds"] == 1.25
    assert recorded[1]["success"] is False
    assert recorded[1]["exception_type"] == "RuntimeError"
    assert recorded[1]["exception_message"] == "second measurement failed"
    assert "success=1/2" in summaries[0]


def test_execute_warm_runs_uses_single_backend_env_batch(monkeypatch):
    run = RunSpec(
        mode="aligned-warm",
        backend="unturtle",
        prompt_name="math",
        prompt_text=PROMPTS["math"],
        settings={
            "steps": 64,
            "max_new_tokens": 64,
            "block_size": 32,
            "temperature": 0.0,
            "right_shift_logits": False,
        },
    )
    calls: list[tuple[RunSpec, int, int]] = []
    outcomes = [
        benchmark_a2d_aligned.ExecutionResult(
            generated_text="one",
            backend_metadata={"path": "block_diffusion_generator"},
            output_tokens=3,
            runtime_seconds=1.0,
        ),
        benchmark_a2d_aligned.ExecutionResult(
            generated_text="two",
            backend_metadata={"path": "block_diffusion_generator"},
            output_tokens=4,
            runtime_seconds=1.1,
        ),
        RuntimeError("third measurement failed"),
    ]

    monkeypatch.setattr(
        benchmark_a2d_aligned,
        "_run_warm_batch_in_backend_env",
        lambda inner_run, *, warmup_iters, measure_iters: (
            calls.append((inner_run, warmup_iters, measure_iters)) or outcomes
        ),
    )

    records = benchmark_a2d_aligned.execute_warm_runs(
        run,
        warmup_iters=2,
        measure_iters=3,
        env_path=ENV_PATHS["unturtle"],
        checkpoint=CHECKPOINT,
        git_head="deadbeef",
    )

    assert calls == [(run, 2, 3)]
    assert len(records) == 3
    assert records[0]["success"] is True
    assert records[1]["success"] is True
    assert records[2]["success"] is False
    assert records[2]["exception_message"] == "third measurement failed"


def test_describe_backend_path_returns_approved_task3_labels():
    assert (
        describe_backend_path("aligned-warm", "unturtle") == "block_diffusion_generator"
    )
    assert describe_backend_path("validator-warm", "unturtle") == "generate"
    assert describe_backend_path("aligned-warm", "dllm") == "bd3lm_sampler"
    assert describe_backend_path("validator-warm", "dllm") == "bd3lm_sampler"


def test_run_in_backend_env_uses_configured_worker_python(monkeypatch):
    run = RunSpec(
        mode="aligned-warm",
        backend="dllm",
        prompt_name="math",
        prompt_text=PROMPTS["math"],
        settings={
            "steps": 64,
            "max_new_tokens": 64,
            "block_size": 32,
            "temperature": 0.0,
            "right_shift_logits": False,
        },
    )
    captured: dict[str, object] = {}

    class Completed:
        stdout = '{"generated_text": "ok", "backend_metadata": {"path": "bd3lm_sampler"}, "output_tokens": 2, "runtime_seconds": 1.5}'

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Completed()

    monkeypatch.setattr(benchmark_a2d_aligned.subprocess, "run", fake_run)

    result = benchmark_a2d_aligned._run_in_backend_env(run)

    assert captured["command"] == [
        str(
            benchmark_a2d_aligned.get_env_root() / ENV_PATHS["dllm"] / "bin" / "python"
        ),
        str(Path(benchmark_a2d_aligned.__file__).resolve()),
    ]
    assert WORKER_MODE in captured["env"]
    assert result.generated_text == "ok"
    assert result.backend_metadata == {"path": "bd3lm_sampler"}
    assert result.output_tokens == 2
    assert result.runtime_seconds == 1.5


def test_missing_grpo_dependency_placeholder_raises_module_not_found():
    module_path = (
        Path(benchmark_a2d_aligned.__file__).resolve().parents[2]
        / "unturtle"
        / "diffusion"
        / "__init__.py"
    )
    code = module_path.read_text(encoding="utf-8")
    assert "from missing_exc" in code


def test_model_caches_start_empty():
    assert UNTURTLE_MODEL_CACHE == {}
    assert DLLM_MODEL_CACHE == {}
