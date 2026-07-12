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

from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset
from transformers import BertTokenizerFast
from typer import Exit

from unturtle.cli.commands.eval import _prepare_eval_dataset
from unturtle.cli.commands.export import list_checkpoints
from unturtle.cli.commands.train import _resolve_model_class, train
from unturtle.cli.config import build_masked_diffusion_collator, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "configs"
EXAMPLE_CONFIGS = [
    "llada_sft.yaml",
    "a2d_llama_sft.yaml",
    "dream_sft.yaml",
    "llada_grpo.yaml",
]


@pytest.fixture
def tokenizer(tmp_path):
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text(
        "\n".join(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "hello",
                "world",
                "Question",
                ":",
                "Answer",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return BertTokenizerFast(vocab_file=str(vocab_path))


def test_prepare_eval_dataset_tokenizes_raw_text_for_diffusion(tokenizer):
    dataset = Dataset.from_list([{"text": "hello world"}])

    prepared = _prepare_eval_dataset(
        dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        eval_type="diffusion",
    )

    row = prepared[0]
    assert set(prepared.column_names) == {"input_ids", "attention_mask", "labels"}
    assert row["input_ids"] == row["labels"]
    assert len(row["attention_mask"]) == len(row["input_ids"])


def test_prepare_eval_dataset_builds_generation_labels_from_prompt_and_completion(
    tokenizer,
):
    dataset = Dataset.from_list(
        [
            {"prompt": "Question:", "completion": " Answer"},
        ]
    )

    prepared = _prepare_eval_dataset(
        dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        eval_type="generation",
    )

    row = prepared[0]
    prompt_ids = tokenizer("Question:", add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(" Answer", add_special_tokens=False)["input_ids"]

    assert row["input_ids"] == prompt_ids + completion_ids
    assert row["labels"] == ([-100] * len(prompt_ids)) + completion_ids
    assert row["attention_mask"] == [1] * len(row["input_ids"])


def test_prepare_eval_dataset_preserves_generation_labels_for_both_mode(tokenizer):
    dataset = Dataset.from_list(
        [
            {"prompt": "Question:", "completion": " Answer"},
        ]
    )

    prepared = _prepare_eval_dataset(
        dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        eval_type="both",
    )

    row = prepared[0]
    prompt_ids = tokenizer("Question:", add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(" Answer", add_special_tokens=False)["input_ids"]

    assert row["input_ids"] == prompt_ids + completion_ids
    assert row["labels"] == ([-100] * len(prompt_ids)) + completion_ids


def test_resolve_model_class_returns_none_for_auto():
    assert _resolve_model_class("auto") is None


@pytest.mark.parametrize(
    "model_type,expected_name",
    [
        ("a2d", "TinyA2DLlamaLMHeadModel"),
        ("llada", "LLaDAModelLM"),
        ("dream", "DreamModel"),
    ],
)
def test_resolve_model_class_returns_correct_class(model_type, expected_name):
    from unittest.mock import MagicMock, patch

    fake_a2d = MagicMock(__name__="TinyA2DLlamaLMHeadModel")
    fake_llada = MagicMock(__name__="LLaDAModelLM")
    fake_dream = MagicMock(__name__="DreamModel")
    fake_module = MagicMock(
        TinyA2DLlamaLMHeadModel=fake_a2d,
        LLaDAModelLM=fake_llada,
        DreamModel=fake_dream,
    )
    with patch.dict("sys.modules", {"unturtle": fake_module}):
        cls = _resolve_model_class(model_type)
    assert cls.__name__ == expected_name


def test_resolve_model_class_raises_on_unknown_type():
    from unittest.mock import MagicMock, patch

    fake_module = MagicMock(
        TinyA2DLlamaLMHeadModel=MagicMock(),
        LLaDAModelLM=MagicMock(),
        DreamModel=MagicMock(),
    )
    with patch.dict("sys.modules", {"unturtle": fake_module}), pytest.raises(KeyError):
        _resolve_model_class("unknown")


@pytest.mark.parametrize("config_name", EXAMPLE_CONFIGS)
def test_example_yaml_configs_load_with_current_schema(config_name):
    cfg = load_config(CONFIG_DIR / config_name)
    assert cfg.model.model is not None
    assert cfg.data.dataset is not None
    assert cfg.training.output_dir


def test_llada_grpo_yaml_sets_task_and_builds_diffu_config():
    cfg = load_config(CONFIG_DIR / "llada_grpo.yaml")
    assert cfg.training.task == "grpo"
    args = cfg.build_diffu_grpo_config(mask_token_id=126336, report_to="none")
    assert args.num_generations == 8
    assert args.diffu_policy_objective == "grpo"
    assert args.mask_id == 126336
    assert args.generation_batch_size == 8


def test_build_diffu_grpo_config_scales_generation_batch_with_world_size(monkeypatch):
    # Do not set WORLD_SIZE: DiffuGRPOConfig.__post_init__ loads Accelerate and
    # would require a full distributed env.  Scale is computed in config only.
    monkeypatch.setattr(
        "unturtle.cli.config._grpo_effective_world_size",
        lambda: 2,
    )
    cfg = load_config(CONFIG_DIR / "llada_grpo.yaml")
    args = cfg.build_diffu_grpo_config(mask_token_id=126336, report_to="none")
    assert args.generation_batch_size == 16


@pytest.mark.parametrize("config_name", EXAMPLE_CONFIGS)
def test_example_yaml_configs_work_with_train_dry_run(config_name, capsys):
    config_path = CONFIG_DIR / config_name

    with pytest.raises(Exit) as exc_info:
        train(config=config_path, dry_run=True, config_overrides={})

    assert exc_info.value.exit_code == 0
    captured = capsys.readouterr()
    assert "model:" in captured.out
    assert "training:" in captured.out


def test_build_masked_diffusion_collator_wires_scheduler_and_epsilon(tokenizer):
    # Regression: `unturtle train/eval --alpha-scheduler cosine` used to build
    # the collator with the default linear scheduler (alpha_scheduler never
    # reached the noising collator).
    from unturtle.diffusion.schedulers import CosineAlphaScheduler

    collator = build_masked_diffusion_collator(
        tokenizer,
        alpha_scheduler="cosine",
        time_epsilon=5e-3,
        completion_only=False,
    )

    assert isinstance(collator.scheduler, CosineAlphaScheduler)
    assert collator.time_epsilon == 5e-3
    assert collator.completion_only is False
    assert collator.mask_token_id == tokenizer.mask_token_id


def test_build_masked_diffusion_collator_explicit_mask_id_wins(tokenizer):
    collator = build_masked_diffusion_collator(tokenizer, mask_token_id=9)
    assert collator.mask_token_id == 9


def test_build_masked_diffusion_collator_falls_back_to_model_config(tokenizer):
    # Regression: `unturtle eval --eval-type diffusion` on a tokenizer without
    # mask_token_id must fall back to model.config.mask_token_id (real
    # checkpoints may carry the mask id only on the model config).
    tokenizer.mask_token = None
    tokenizer.mask_token_id = None
    model = type("Model", (), {"config": type("Config", (), {"mask_token_id": 42})()})()

    collator = build_masked_diffusion_collator(tokenizer, model=model)
    assert collator.mask_token_id == 42


def test_list_checkpoints_reports_unreadable_trainer_state(tmp_path, capsys):
    ckpt_dir = tmp_path / "run-1" / "checkpoint-10"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "trainer_state.json").write_text("{not-json", encoding="utf-8")

    list_checkpoints(outputs_dir=tmp_path)

    captured = capsys.readouterr()
    assert "unreadable trainer_state.json" in captured.out
    assert "Warning: failed to read loss" in captured.err


# ---------------------------------------------------------------------------
# CLI override validation — Config.apply_overrides / _SectionConfig
# ---------------------------------------------------------------------------


def test_apply_overrides_invalid_literal_raises_immediately():
    # setattr used to bypass pydantic validation: an invalid --task/--model-type
    # only failed late (or silently). validate_assignment must reject it here.
    from unturtle.cli.config import Config

    cfg = Config()
    with pytest.raises(ValueError, match=r"--task"):
        cfg.apply_overrides(task="not-a-task")

    with pytest.raises(ValueError, match=r"--model-type"):
        cfg.apply_overrides(model_type="not-a-model-type")


def test_apply_overrides_valid_values_still_apply():
    from unturtle.cli.config import Config

    cfg = Config()
    cfg.apply_overrides(task="grpo", model_type="llada", learning_rate=1e-4)
    assert cfg.training.task == "grpo"
    assert cfg.model.model_type == "llada"
    assert cfg.training.learning_rate == 1e-4


def test_train_invalid_override_exits_with_clear_error(capsys):
    config_path = CONFIG_DIR / "llada_sft.yaml"
    # The add_options_from_config wrapper rebuilds config_overrides from field
    # kwargs, so the override is passed as `task=` (as the CLI would).
    with pytest.raises(Exit) as exc_info:
        train(config=config_path, dry_run=True, task="bogus")
    assert exc_info.value.exit_code == 2
    captured = capsys.readouterr()
    assert "Invalid value for --task" in captured.err


# ---------------------------------------------------------------------------
# export --load-in-4bit default is format-aware
# ---------------------------------------------------------------------------


def _run_export_capture_load_kwargs(monkeypatch, tmp_path, *, format, load_in_4bit):
    from unturtle import fast_diffusion_model as fdm
    from unturtle.cli.commands.export import export

    recorded = {}

    def _fake_from_pretrained(**kwargs):
        recorded.update(kwargs)
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(
        fdm.FastDiffusionModel, "from_pretrained", staticmethod(_fake_from_pretrained)
    )
    with pytest.raises(Exit) as exc_info:
        export(
            checkpoint=tmp_path,
            output_dir=tmp_path / "out",
            format=format,
            quantization="q4_k_m",
            push_to_hub=False,
            repo_id=None,
            hf_token=None,
            private=False,
            max_seq_length=2048,
            load_in_4bit=load_in_4bit,
        )
    assert exc_info.value.exit_code == 1  # load intentionally aborted
    return recorded


def test_export_merged_16bit_defaults_to_16bit_load(monkeypatch, tmp_path):
    recorded = _run_export_capture_load_kwargs(
        monkeypatch, tmp_path, format="merged-16bit", load_in_4bit=None
    )
    assert recorded["load_in_4bit"] is False


def test_export_lora_defaults_to_4bit_load(monkeypatch, tmp_path):
    recorded = _run_export_capture_load_kwargs(
        monkeypatch, tmp_path, format="lora", load_in_4bit=None
    )
    assert recorded["load_in_4bit"] is True


def test_export_merged_16bit_explicit_4bit_warns(monkeypatch, tmp_path, capsys):
    recorded = _run_export_capture_load_kwargs(
        monkeypatch, tmp_path, format="merged-16bit", load_in_4bit=True
    )
    assert recorded["load_in_4bit"] is True  # explicit choice honored
    captured = capsys.readouterr()
    assert "dequantized" in captured.err
