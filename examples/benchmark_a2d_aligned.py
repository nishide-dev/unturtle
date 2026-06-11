from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

CHECKPOINT = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
BACKENDS = ("unturtle", "dllm")
BENCHMARK_MODES = ("aligned-warm", "validator-warm", "cold-start")
ENV_PATHS = {
    "unturtle": ".venv",
    "dllm": ".venvDllm",
}
UNTURTLE_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
DLLM_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
PROMPTS = {
    "math": "Lily runs 12 km/h for 4 hours. How far in 8 hours?",
    "code": "Please write an educational python function.",
}
DEFAULT_STEPS = (64, 128)
DEFAULT_MAX_NEW_TOKENS = (64, 128)


@dataclass(frozen=True)
class RunSpec:
    mode: str
    backend: str
    prompt_name: str
    prompt_text: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class ExecutionResult:
    generated_text: str
    backend_metadata: dict[str, Any]
    output_tokens: int | None
    runtime_seconds: float


Runner = Callable[[RunSpec], tuple[str, dict[str, Any], int | None]]
WORKER_MODE = "UNTURTLE_BENCHMARK_WORKER"
WARM_WORKER_MODE = "UNTURTLE_BENCHMARK_WARM_WORKER"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=BENCHMARK_MODES, default="aligned-warm")
    parser.add_argument("--backend", choices=[*BACKENDS, "all"], default="all")
    parser.add_argument("--prompt", choices=[*PROMPTS, "all"], default="all")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output-dir", default="outputs/a2d_aligned_benchmark")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    return parser.parse_args()


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_repo_root() -> Path:
    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], text=True
        ).strip()
        return Path(common_dir).resolve().parent
    except Exception:
        return Path.cwd()


def get_env_root() -> Path:
    executable = Path(sys.executable)
    active_env = executable.parent.parent
    if executable.parent.name == "bin" and active_env.name in set(ENV_PATHS.values()):
        return active_env.parent
    return get_repo_root()


def build_aligned_generation_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_new_tokens": settings["max_new_tokens"],
        "steps": settings["steps"],
        "block_size": settings["block_size"],
        "temperature": settings["temperature"],
        "right_shift_logits": settings["right_shift_logits"],
    }


def build_validator_generation_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_new_tokens": settings["max_new_tokens"],
        "steps": settings["steps"],
        "temperature": settings["temperature"],
        "use_cache": False,
    }


def build_dllm_sampler_config(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": settings["steps"],
        "max_new_tokens": settings["max_new_tokens"],
        "temperature": settings["temperature"],
        "block_size": settings["block_size"],
        "remasking": "low_confidence",
        "right_shift_logits": settings["right_shift_logits"],
    }


def describe_backend_path(mode: str, backend: str) -> str:
    if backend == "unturtle":
        if mode == "aligned-warm":
            return "block_diffusion_generator"
        if mode == "validator-warm":
            return "diffusion_generate"
    if backend == "dllm" and mode in {"aligned-warm", "validator-warm"}:
        return "bd3lm_sampler"
    return f"{backend}:{mode}"


def load_unturtle_model():
    if CHECKPOINT in UNTURTLE_MODEL_CACHE:
        return UNTURTLE_MODEL_CACHE[CHECKPOINT]

    import torch

    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3 import (
        TinyA2DQwen3LMHeadModel,
    )

    model, tokenizer = FastDiffusionModel.from_pretrained(
        CHECKPOINT,
        model_class=TinyA2DQwen3LMHeadModel,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    model.eval()
    UNTURTLE_MODEL_CACHE[CHECKPOINT] = (model, tokenizer)
    return model, tokenizer


def load_dllm_model():
    if CHECKPOINT in DLLM_MODEL_CACHE:
        return DLLM_MODEL_CACHE[CHECKPOINT]

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        padding_side="right",
        trust_remote_code=True,
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.eos_token:
        tokenizer.eos_token = tokenizer.pad_token
    if not tokenizer.bos_token:
        tokenizer.bos_token = tokenizer.pad_token
    tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    tokenizer.eot_token = "<|im_end|>"
    tokenizer.eot_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eot_token)
    original_apply_chat_template = tokenizer.apply_chat_template

    def _apply_chat_template(*args, **kwargs):
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = False
        try:
            return original_apply_chat_template(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return original_apply_chat_template(*args, **kwargs)

    tokenizer.apply_chat_template = _apply_chat_template
    model = AutoModelForMaskedLM.from_pretrained(
        CHECKPOINT,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    DLLM_MODEL_CACHE[CHECKPOINT] = (model, tokenizer)
    return model, tokenizer


def run_unturtle_aligned_once(run: RunSpec) -> tuple[str, dict[str, Any], int | None]:
    model, tokenizer = load_unturtle_model()
    from unturtle.models.block_diffusion_generator import BlockDiffusionGenerator

    messages = [[{"role": "user", "content": run.prompt_text}]]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device)
    prompt_token_ids = input_ids[0].tolist()

    generator = BlockDiffusionGenerator(model=model, tokenizer=tokenizer)
    generation_kwargs = build_aligned_generation_kwargs(run.settings)
    outputs = generator.generate([prompt_token_ids], **generation_kwargs)
    generated_tokens = outputs[:, len(prompt_token_ids) :]
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    token_count = int(outputs.shape[1] - len(prompt_token_ids))
    metadata = {"path": describe_backend_path(run.mode, run.backend)}
    return text, metadata, token_count


def run_unturtle_validator_once(run: RunSpec) -> tuple[str, dict[str, Any], int | None]:
    from unturtle.models.generation.diffusion_generation_utils import (
        MaskedDiffusionGenerationConfig,
    )

    model, tokenizer = load_unturtle_model()
    messages = [[{"role": "user", "content": run.prompt_text}]]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is None:
        raise ValueError("mask_token_id is not available on tokenizer or model.config")

    generation_config = MaskedDiffusionGenerationConfig(
        **build_validator_generation_kwargs(run.settings),
        mask_token_id=mask_token_id,
    )
    outputs = model.diffusion_generate(
        inputs=input_ids, generation_config=generation_config
    )
    generated_tokens = outputs[:, input_ids.shape[1] :]
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    token_count = int(outputs.shape[1] - input_ids.shape[1])
    metadata = {
        "path": describe_backend_path(run.mode, run.backend),
        "mask_token_id": mask_token_id,
    }
    return text, metadata, token_count


def run_dllm_once(run: RunSpec) -> tuple[str, dict[str, Any], int | None]:
    try:
        from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
    except ModuleNotFoundError:
        import dllm

        BD3LMSampler = dllm.core.samplers.BD3LMSampler
        BD3LMSamplerConfig = dllm.core.samplers.BD3LMSamplerConfig
    from dllm.utils.sampling import sample_trim

    model, tokenizer = load_dllm_model()
    messages = [[{"role": "user", "content": run.prompt_text}]]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    sampler_config = BD3LMSamplerConfig(**build_dllm_sampler_config(run.settings))
    sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
    outputs = sampler.sample(inputs, sampler_config, return_dict=True)
    text = sample_trim(tokenizer, outputs.sequences.tolist(), inputs)[0]
    token_count = len(outputs.sequences[0]) - len(inputs[0])
    metadata = {
        "path": describe_backend_path(run.mode, run.backend),
        "sampler": type(sampler).__name__,
        "sample_trim": True,
    }
    return text, metadata, token_count


def expand_runs(
    *,
    backends: list[str],
    mode: str,
    prompt_names: list[str],
    steps_values: list[int],
    max_new_token_values: list[int],
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for backend in backends:
        for prompt_name in prompt_names:
            for steps in steps_values:
                for max_new_tokens in max_new_token_values:
                    runs.append(
                        RunSpec(
                            mode=mode,
                            backend=backend,
                            prompt_name=prompt_name,
                            prompt_text=PROMPTS[prompt_name],
                            settings={
                                "steps": steps,
                                "max_new_tokens": max_new_tokens,
                                "block_size": 32,
                                "temperature": 0.0,
                                "right_shift_logits": False,
                            },
                        )
                    )
    return runs


def normalize_benchmark_record(
    *,
    mode: str,
    backend: str,
    env_path: str,
    checkpoint: str,
    prompt_name: str,
    prompt_text: str,
    settings: dict[str, Any],
    success: bool,
    generated_text: str,
    runtime_seconds: float,
    output_tokens: int | None,
    exception: Exception | None,
    backend_metadata: dict[str, Any],
    git_head: str,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_head": git_head,
        "mode": mode,
        "backend": backend,
        "environment_path": env_path,
        "checkpoint": checkpoint,
        "prompt_name": prompt_name,
        "prompt_text": prompt_text,
        "benchmark_settings": dict(settings),
        "success": success,
        "exception_type": None if exception is None else type(exception).__name__,
        "exception_message": None if exception is None else str(exception),
        "generated_text": generated_text,
        "output_tokens": output_tokens,
        "runtime_seconds": runtime_seconds,
        "backend_metadata": dict(backend_metadata),
    }


def append_result_record(output_dir: Path, record: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    settings = record["benchmark_settings"]
    return (
        record["mode"],
        record["backend"],
        record.get("prompt_name", record.get("prompt")),
        tuple(sorted(settings.items())),
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def summarize_records(records: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_group_key(record), []).append(record)

    def format_settings(settings: dict[str, Any]) -> str:
        return " / ".join(f"{key}={value}" for key, value in settings.items())

    lines = ["# A2D aligned benchmark summary"]
    for key in sorted(grouped):
        group = grouped[key]
        successful_records = [record for record in group if record.get("success")]
        runtimes = [float(record["runtime_seconds"]) for record in successful_records]
        first = group[0]
        success_count = len(successful_records)
        if runtimes:
            timing_summary = (
                f"mean={sum(runtimes) / len(runtimes):.2f}s | "
                + f"median={_percentile(runtimes, 0.5):.2f}s | "
                + f"p95={_percentile(runtimes, 0.95):.2f}s"
            )
        else:
            timing_summary = "mean=n/a | median=n/a | p95=n/a"
        lines.append(
            " / ".join(
                [
                    first["mode"],
                    first["backend"],
                    first.get("prompt_name", first.get("prompt")),
                    format_settings(dict(first["benchmark_settings"])),
                ]
            )
            + " | "
            + timing_summary
            + " | "
            + f"success={success_count}/{len(group)}"
        )
    return "\n".join(lines) + "\n"


def write_summary(output_dir: Path, summary: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.md"
    path.write_text(summary, encoding="utf-8")
    return path


def _clear_backend_state(backend: str) -> None:
    if backend == "unturtle":
        UNTURTLE_MODEL_CACHE.clear()
    elif backend == "dllm":
        DLLM_MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _time_runner(runner: Runner, run: RunSpec) -> ExecutionResult:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    generated_text, backend_metadata, output_tokens = runner(run)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - started
    return ExecutionResult(
        generated_text=generated_text,
        backend_metadata=backend_metadata,
        output_tokens=output_tokens,
        runtime_seconds=runtime_seconds,
    )


def _worker_python(backend: str) -> Path:
    return get_env_root() / ENV_PATHS[backend] / "bin" / "python"


def _run_in_backend_env(run: RunSpec) -> ExecutionResult:
    worker_python = _worker_python(run.backend)
    script_path = Path(__file__).resolve()
    source_root = script_path.parent.parent
    payload = json.dumps(
        {
            "mode": run.mode,
            "backend": run.backend,
            "prompt_name": run.prompt_name,
            "prompt_text": run.prompt_text,
            "settings": run.settings,
        }
    )
    env = {**os.environ, WORKER_MODE: payload, "PYTHONPATH": str(source_root)}
    completed = subprocess.run(
        [str(worker_python), str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return ExecutionResult(**json.loads(stdout_lines[-1]))


def _run_warm_batch_in_backend_env(
    run: RunSpec,
    *,
    warmup_iters: int,
    measure_iters: int,
) -> list[ExecutionResult | Exception]:
    worker_python = _worker_python(run.backend)
    script_path = Path(__file__).resolve()
    source_root = script_path.parent.parent
    payload = json.dumps(
        {
            "run": {
                "mode": run.mode,
                "backend": run.backend,
                "prompt_name": run.prompt_name,
                "prompt_text": run.prompt_text,
                "settings": run.settings,
            },
            "warmup_iters": warmup_iters,
            "measure_iters": measure_iters,
        }
    )
    env = {**os.environ, WARM_WORKER_MODE: payload, "PYTHONPATH": str(source_root)}
    completed = subprocess.run(
        [str(worker_python), str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    records = json.loads(stdout_lines[-1])
    results: list[ExecutionResult | Exception] = []
    for record in records:
        if record["success"]:
            results.append(
                ExecutionResult(
                    generated_text=record["generated_text"],
                    backend_metadata=record["backend_metadata"],
                    output_tokens=record["output_tokens"],
                    runtime_seconds=record["runtime_seconds"],
                )
            )
        else:
            results.append(RuntimeError(record["exception_message"]))
    return results


def execute_warm_runs(
    run: RunSpec,
    warmup_iters: int,
    measure_iters: int,
    *,
    env_path: str,
    checkpoint: str,
    git_head: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for outcome in _run_warm_batch_in_backend_env(
        run,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    ):
        if isinstance(outcome, Exception):
            records.append(
                normalize_benchmark_record(
                    mode=run.mode,
                    backend=run.backend,
                    env_path=env_path,
                    checkpoint=checkpoint,
                    prompt_name=run.prompt_name,
                    prompt_text=run.prompt_text,
                    settings=run.settings,
                    success=False,
                    generated_text="",
                    runtime_seconds=0.0,
                    output_tokens=None,
                    exception=outcome,
                    backend_metadata={
                        "path": describe_backend_path(run.mode, run.backend)
                    },
                    git_head=git_head,
                )
            )
            break
        records.append(
            normalize_benchmark_record(
                mode=run.mode,
                backend=run.backend,
                env_path=env_path,
                checkpoint=checkpoint,
                prompt_name=run.prompt_name,
                prompt_text=run.prompt_text,
                settings=run.settings,
                success=True,
                generated_text=outcome.generated_text,
                runtime_seconds=outcome.runtime_seconds,
                output_tokens=outcome.output_tokens,
                exception=None,
                backend_metadata=outcome.backend_metadata,
                git_head=git_head,
            )
        )
    return records


def execute_cold_start_run(run: RunSpec) -> ExecutionResult:
    return _run_in_backend_env(run)


def _execute_cold_start_run_in_process(run: RunSpec) -> ExecutionResult:
    _clear_backend_state(run.backend)

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    load_started = time.perf_counter()
    if run.backend == "unturtle":
        load_unturtle_model()
    elif run.backend == "dllm":
        load_dllm_model()
    else:
        raise ValueError(f"unsupported backend: {run.backend}")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started

    runner = select_runner(run)

    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    first_generation_started = time.perf_counter()
    generated_text, backend_metadata, output_tokens = runner(run)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    first_generation_seconds = time.perf_counter() - first_generation_started

    return ExecutionResult(
        generated_text=generated_text,
        backend_metadata={
            **backend_metadata,
            "cold_start": True,
            "load_seconds": load_seconds,
            "first_generation_seconds": first_generation_seconds,
        },
        output_tokens=output_tokens,
        runtime_seconds=load_seconds + first_generation_seconds,
    )


def select_runner(run: RunSpec) -> Runner:
    if run.backend == "unturtle":
        if run.mode in {"aligned-warm", "cold-start"}:
            return run_unturtle_aligned_once
        if run.mode == "validator-warm":
            return run_unturtle_validator_once
    if run.backend == "dllm" and run.mode in {
        "aligned-warm",
        "validator-warm",
        "cold-start",
    }:
        return run_dllm_once
    raise ValueError(f"unsupported run: backend={run.backend}, mode={run.mode}")


def _run_worker_from_env() -> int:
    payload = os.environ.get(WORKER_MODE)
    if not payload:
        return 1
    run_data = json.loads(payload)
    run = RunSpec(**run_data)
    if run.mode == "cold-start":
        result = _execute_cold_start_run_in_process(run)
    else:
        runner = select_runner(run)
        result = _time_runner(runner, run)
    print(
        json.dumps(
            {
                "generated_text": result.generated_text,
                "backend_metadata": result.backend_metadata,
                "output_tokens": result.output_tokens,
                "runtime_seconds": result.runtime_seconds,
            }
        )
    )
    return 0


def _run_warm_worker_from_env() -> int:
    payload = os.environ.get(WARM_WORKER_MODE)
    if not payload:
        return 1
    batch = json.loads(payload)
    run = RunSpec(**batch["run"])
    runner = select_runner(run)
    for _ in range(batch["warmup_iters"]):
        _ = _time_runner(runner, run)
    results: list[dict[str, Any]] = []
    for _ in range(batch["measure_iters"]):
        try:
            result = _time_runner(runner, run)
            results.append(
                {
                    "success": True,
                    "generated_text": result.generated_text,
                    "backend_metadata": result.backend_metadata,
                    "output_tokens": result.output_tokens,
                    "runtime_seconds": result.runtime_seconds,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "success": False,
                    "exception_message": str(exc),
                }
            )
            break
    print(json.dumps(results))
    return 0


def main() -> None:
    if WORKER_MODE in os.environ:
        raise SystemExit(_run_worker_from_env())
    if WARM_WORKER_MODE in os.environ:
        raise SystemExit(_run_warm_worker_from_env())

    args = parse_args()
    backends = list(BACKENDS) if args.backend == "all" else [args.backend]
    prompt_names = list(PROMPTS) if args.prompt == "all" else [args.prompt]
    steps_values = [args.steps] if args.steps is not None else list(DEFAULT_STEPS)
    max_new_token_values = (
        [args.max_new_tokens]
        if args.max_new_tokens is not None
        else list(DEFAULT_MAX_NEW_TOKENS)
    )
    runs = expand_runs(
        backends=backends,
        mode=args.mode,
        prompt_names=prompt_names,
        steps_values=steps_values,
        max_new_token_values=max_new_token_values,
    )
    output_dir = Path(args.output_dir)
    git_head = get_git_head()
    records: list[dict[str, Any]] = []

    for run in runs:
        try:
            if run.mode == "cold-start":
                run_records = [
                    normalize_benchmark_record(
                        mode=run.mode,
                        backend=run.backend,
                        env_path=ENV_PATHS[run.backend],
                        checkpoint=CHECKPOINT,
                        prompt_name=run.prompt_name,
                        prompt_text=run.prompt_text,
                        settings=run.settings,
                        success=True,
                        generated_text=(
                            result := execute_cold_start_run(run)
                        ).generated_text,
                        runtime_seconds=result.runtime_seconds,
                        output_tokens=result.output_tokens,
                        exception=None,
                        backend_metadata=result.backend_metadata,
                        git_head=git_head,
                    )
                ]
            else:
                run_records = execute_warm_runs(
                    run,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.measure_iters,
                    env_path=ENV_PATHS[run.backend],
                    checkpoint=CHECKPOINT,
                    git_head=git_head,
                )

            for record in run_records:
                append_result_record(output_dir, record)
                records.append(record)
        except Exception as exc:
            record = normalize_benchmark_record(
                mode=run.mode,
                backend=run.backend,
                env_path=ENV_PATHS[run.backend],
                checkpoint=CHECKPOINT,
                prompt_name=run.prompt_name,
                prompt_text=run.prompt_text,
                settings=run.settings,
                success=False,
                generated_text="",
                runtime_seconds=0.0,
                output_tokens=None,
                exception=exc,
                backend_metadata={"path": describe_backend_path(run.mode, run.backend)},
                git_head=git_head,
            )
            append_result_record(output_dir, record)
            records.append(record)

    summary = summarize_records(records)
    write_summary(output_dir, summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
