# tests/eval/test_harness_configs.py
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from unturtle.eval.harness.configs import (
    DecodingConfig,
    get_decoding_config,
    list_decoding_configs,
)


def test_decoding_config_is_frozen_and_records_hyperparams() -> None:
    cfg = DecodingConfig(
        model_family="a2d_qwen3",
        task="gsm8k",
        max_new_tokens=256,
        num_steps=256,
        temperature=0.0,
        use_chat_template=True,
        fewshot=0,
    )
    assert cfg.max_new_tokens == 256
    with pytest.raises(FrozenInstanceError):
        cfg.max_new_tokens = 1  # frozen dataclass


def test_get_known_config_returns_recorded_hyperparams() -> None:
    cfg = get_decoding_config("a2d_qwen3", "gsm8k_cot")
    assert cfg.task == "gsm8k_cot"
    assert cfg.model_family == "a2d_qwen3"
    assert cfg.max_new_tokens > 0
    assert cfg.num_steps > 0


def test_get_unknown_config_raises_keyerror_with_context() -> None:
    with pytest.raises(KeyError) as exc:
        get_decoding_config("a2d_qwen3", "does_not_exist")
    assert "does_not_exist" in str(exc.value)


def test_list_configs_includes_gsm8k_and_gsm8k_cot() -> None:
    keys = list_decoding_configs()
    assert ("a2d_qwen3", "gsm8k") in keys
    assert ("a2d_qwen3", "gsm8k_cot") in keys


def test_as_dict_roundtrips_for_recording() -> None:
    cfg = get_decoding_config("a2d_qwen3", "gsm8k")
    d = cfg.as_dict()
    assert d["task"] == "gsm8k"
    assert d["max_new_tokens"] == cfg.max_new_tokens
    assert set(d) >= {
        "model_family",
        "task",
        "max_new_tokens",
        "num_steps",
        "temperature",
        "use_chat_template",
        "fewshot",
    }
