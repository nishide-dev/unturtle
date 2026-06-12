from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHECKPOINTS = {
    "mdlm": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    "bd3lm": "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1",
}

BACKENDS = ("unturtle", "dllm")
ENV_PATHS = {
    "unturtle": ".venv",
    "dllm": ".venvDllm",
}
UNTURTLE_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
DLLM_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str


@dataclass(frozen=True)
class RunSpec:
    backend: str
    model_kind: str
    prompt_name: str
    prompt_text: str
    settings: dict[str, Any]


def build_unturtle_generation_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_new_tokens": settings["max_new_tokens"],
        "steps": settings["steps"],
        "temperature": settings["temperature"],
        "use_cache": False,
    }


PROMPTS = {
    "math": PromptSpec(
        name="math",
        text="Lily runs 12 km/h for 4 hours. How far in 8 hours?",
    ),
    "code": PromptSpec(
        name="code",
        text="Please write an educational python function.",
    ),
}

SCENARIOS = {
    "smoke": {
        "mdlm": [{"steps": 64, "max_new_tokens": 64, "temperature": 0.0}],
        "bd3lm": [
            {"steps": 64, "max_new_tokens": 64, "temperature": 0.0, "block_size": 32}
        ],
    },
    "stability": {
        "mdlm": [
            {"steps": 64, "max_new_tokens": 64, "temperature": 0.0},
            {"steps": 128, "max_new_tokens": 128, "temperature": 0.0},
        ],
        "bd3lm": [
            {"steps": 64, "max_new_tokens": 64, "temperature": 0.0, "block_size": 32},
            {"steps": 128, "max_new_tokens": 128, "temperature": 0.0, "block_size": 32},
        ],
    },
}


def expand_runs(
    backends: list[str], model_kinds: list[str], scenario_name: str
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    prompts = (
        [PROMPTS["math"]]
        if scenario_name == "smoke"
        else [PROMPTS["math"], PROMPTS["code"]]
    )
    for backend in backends:
        for model_kind in model_kinds:
            for prompt in prompts:
                for settings in SCENARIOS[scenario_name][model_kind]:
                    runs.append(
                        RunSpec(
                            backend=backend,
                            model_kind=model_kind,
                            prompt_name=prompt.name,
                            prompt_text=prompt.text,
                            settings=dict(settings),
                        )
                    )
    return runs


def normalize_result_record(
    *,
    backend: str,
    env_path: str,
    checkpoint: str,
    model_kind: str,
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
        "backend": backend,
        "environment_path": env_path,
        "checkpoint": checkpoint,
        "model_kind": model_kind,
        "prompt_name": prompt_name,
        "prompt_text": prompt_text,
        "inference_settings": dict(settings),
        "success": success,
        "exception_type": None if exception is None else type(exception).__name__,
        "exception_message": None if exception is None else str(exception),
        "generated_text": generated_text,
        "output_tokens": output_tokens,
        "runtime_seconds": runtime_seconds,
        "backend_metadata": dict(backend_metadata),
    }


def summarize_results(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        header = (
            f"{record['backend']} / {record['model_kind']}"
            f" / {record.get('prompt_name', 'unknown')}"
            f" / {record.get('inference_settings', {})}"
        )
        if record.get("success"):
            generated_text = str(record.get("generated_text", "")).strip()
            detail = generated_text[:80] if generated_text else "<empty>"
            status = "ok"
        else:
            detail = record.get("exception_message") or "unknown failure"
            status = "failed"
        lines.append(f"- {header}: {status} - {detail}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["unturtle", "dllm", "all"], default="all")
    parser.add_argument("--model", choices=["mdlm", "bd3lm", "all"], default="all")
    parser.add_argument("--scenario", choices=["smoke", "stability"], default="smoke")
    parser.add_argument("--prompt", choices=["math", "code", "all"], default="all")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--output-dir", default="outputs/real_inference_validation")
    return parser.parse_args()


def get_git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def append_result_record(output_dir: Path, record: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_unturtle_model(model_kind: str):
    if model_kind in UNTURTLE_MODEL_CACHE:
        return UNTURTLE_MODEL_CACHE[model_kind]

    import torch

    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3 import (
        TinyA2DQwen3LMHeadModel,
    )

    checkpoint = CHECKPOINTS[model_kind]
    model, tokenizer = FastDiffusionModel.from_pretrained(
        checkpoint,
        model_class=TinyA2DQwen3LMHeadModel,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    model.eval()
    UNTURTLE_MODEL_CACHE[model_kind] = (model, tokenizer)
    return model, tokenizer


def run_unturtle_once(run: RunSpec) -> tuple[str, dict[str, Any], int | None]:
    from unturtle.models.generation.diffusion_generation_utils import (
        MaskedDiffusionGenerationConfig,
    )

    model, tokenizer = load_unturtle_model(run.model_kind)
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

    generation_kwargs = build_unturtle_generation_kwargs(run.settings)
    generation_config = MaskedDiffusionGenerationConfig(
        **generation_kwargs,
        mask_token_id=mask_token_id,
    )
    outputs = model.generate(
        inputs=input_ids, generation_config=generation_config, algorithm="mdlm"
    )
    text = tokenizer.batch_decode(
        outputs[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0]
    token_count = int(outputs.shape[1] - input_ids.shape[1])
    return text, {"mask_token_id": mask_token_id}, token_count


@dataclass(frozen=True)
class _DllmModelArgs:
    model_name_or_path: str


def load_dllm_model(model_kind: str):
    if model_kind in DLLM_MODEL_CACHE:
        return DLLM_MODEL_CACHE[model_kind]

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    checkpoint = CHECKPOINTS[model_kind]
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
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
    _orig_apply_chat_template = tokenizer.apply_chat_template

    def _apply_chat_template(*args, **kwargs):
        if "enable_thinking" not in kwargs:
            kwargs["enable_thinking"] = False
        try:
            return _orig_apply_chat_template(*args, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return _orig_apply_chat_template(*args, **kwargs)

    tokenizer.apply_chat_template = _apply_chat_template
    model = AutoModelForMaskedLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    DLLM_MODEL_CACHE[model_kind] = (model, tokenizer)
    return model, tokenizer


def build_dllm_sampler_config(model_kind: str, settings: dict[str, Any]):
    try:
        from dllm.core.samplers import BD3LMSamplerConfig, MDLMSamplerConfig
    except ModuleNotFoundError:
        import dllm

        BD3LMSamplerConfig = dllm.core.samplers.BD3LMSamplerConfig
        MDLMSamplerConfig = dllm.core.samplers.MDLMSamplerConfig

    sampler_kwargs = {
        "steps": settings["steps"],
        "max_new_tokens": settings["max_new_tokens"],
        "temperature": settings["temperature"],
        "remasking": "low_confidence",
        "right_shift_logits": False,
    }
    if model_kind == "mdlm":
        sampler_class = MDLMSamplerConfig
    elif model_kind == "bd3lm":
        sampler_class = BD3LMSamplerConfig
        sampler_kwargs["block_size"] = settings.get("block_size", 32)
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")
    return sampler_class(**sampler_kwargs)


def run_dllm_once(run: RunSpec) -> tuple[str, dict[str, Any], int | None]:
    from dllm.core.samplers import BD3LMSampler, MDLMSampler
    from dllm.utils.sampling import sample_trim

    model, tokenizer = load_dllm_model(run.model_kind)
    messages = [[{"role": "user", "content": run.prompt_text}]]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    sampler_config = build_dllm_sampler_config(run.model_kind, run.settings)
    if run.model_kind == "mdlm":
        sampler = MDLMSampler(model=model, tokenizer=tokenizer)
    elif run.model_kind == "bd3lm":
        sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
    else:
        raise ValueError(f"unsupported model kind: {run.model_kind}")
    outputs = sampler.sample(inputs, sampler_config, return_dict=True)
    text = sample_trim(tokenizer, outputs.sequences.tolist(), inputs)[0]
    token_count = len(outputs.sequences[0]) - len(inputs[0])
    return text, {"sampler": type(sampler).__name__}, token_count


def main() -> None:
    args = parse_args()
    backends = ["unturtle", "dllm"] if args.backend == "all" else [args.backend]
    model_kinds = ["mdlm", "bd3lm"] if args.model == "all" else [args.model]
    runs = expand_runs(
        backends=backends, model_kinds=model_kinds, scenario_name=args.scenario
    )
    if args.prompt != "all":
        runs = [run for run in runs if run.prompt_name == args.prompt]
    if args.steps is not None:
        runs = [run for run in runs if run.settings.get("steps") == args.steps]
    if args.max_new_tokens is not None:
        runs = [
            run
            for run in runs
            if run.settings.get("max_new_tokens") == args.max_new_tokens
        ]
    output_dir = Path(args.output_dir)
    git_head = get_git_head()
    records: list[dict[str, Any]] = []

    for run in runs:
        started = time.perf_counter()
        error = None
        generated_text = ""
        output_tokens = None
        metadata: dict[str, Any] = {}
        try:
            if run.backend == "unturtle":
                generated_text, metadata, output_tokens = run_unturtle_once(run)
            else:
                generated_text, metadata, output_tokens = run_dllm_once(run)
        except Exception as exc:
            error = exc
        runtime_seconds = time.perf_counter() - started
        record = normalize_result_record(
            backend=run.backend,
            env_path=ENV_PATHS[run.backend],
            checkpoint=CHECKPOINTS[run.model_kind],
            model_kind=run.model_kind,
            prompt_name=run.prompt_name,
            prompt_text=run.prompt_text,
            settings=run.settings,
            success=error is None,
            generated_text=generated_text,
            runtime_seconds=runtime_seconds,
            output_tokens=output_tokens,
            exception=error,
            backend_metadata=metadata,
            git_head=git_head,
        )
        append_result_record(output_dir, record)
        records.append(record)

    print(summarize_results(records))


if __name__ == "__main__":
    main()
