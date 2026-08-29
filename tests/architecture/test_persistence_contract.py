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

"""Persistence contract vs a live recomputation of the native_fp cell.

The state-dict comparison and buffer diff are computed here with this test's
own logic; a producer that always reports zero deltas (or hides the RoPE
buffer rewrite) disagrees with this recomputation.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "184-architecture-contract-v1.json"

pytestmark = [pytest.mark.gpu]  # unsloth import chain


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def native_fp_recomputation(tmp_path_factory):
    from unturtle import FastDiffusionModel
    from unturtle.models.backbones.dream.configuration_dream import DreamConfig
    from unturtle.models.backbones.dream.generation_utils import (
        DreamGenerationConfig,
    )
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
    with torch.no_grad():
        for layer in model.model.layers:
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ):
                proj.bias.normal_(std=0.2)
    model.generation_config = DreamGenerationConfig(mask_token_id=1, pad_token_id=0)
    model = model.to(torch.float32).eval()

    out_dir = tmp_path_factory.mktemp("184_native_fp") / "checkpoint"
    model.save_pretrained(out_dir)
    reloaded, _ = FastDiffusionModel.from_pretrained(
        str(out_dir),
        max_seq_length=64,
        dtype=torch.float32,
        load_in_4bit=False,
        model_class=DreamModel,
    )
    reloaded = reloaded.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(2, 400, (2, 12))
    with torch.no_grad():
        before = model(input_ids=input_ids).logits.float()
        after = reloaded(input_ids=input_ids).logits.float()

    before_sd = model.state_dict()
    after_sd = reloaded.state_dict()
    state_dict_equal = set(before_sd) == set(after_sd) and all(
        torch.equal(before_sd[key], after_sd[key]) for key in before_sd
    )
    buffer_diffs = sorted(
        name
        for (name, a), (name_b, b) in zip(
            sorted(model.named_buffers()),
            sorted(reloaded.named_buffers()),
            strict=False,
        )
        if name == name_b
        and (a.shape != b.shape or not torch.equal(a.float(), b.float()))
    )
    return {
        "state_dict_equal": state_dict_equal,
        "buffer_diffs": buffer_diffs,
        "outputs_bit_identical": bool(torch.equal(before, after)),
        "rel_norm": float((after - before).norm() / before.norm()),
    }


def test_native_fp_state_dict_roundtrips(artifact, native_fp_recomputation):
    recorded = artifact["persistence"]["native_fp"]
    assert recorded["missing_keys"] == [] and recorded["unexpected_keys"] == []
    assert recorded["dtype_diffs"] == []
    assert recorded["first_mismatching_key"] is None
    assert native_fp_recomputation["state_dict_equal"] is True


def test_native_fp_rope_buffer_rewrite_is_real(artifact, native_fp_recomputation):
    """The #174-linked finding: identical persistent weights, rewritten RoPE
    inv_freq buffers on the load path, non-identical outputs."""
    recorded = artifact["persistence"]["native_fp"]
    assert native_fp_recomputation["buffer_diffs"] == recorded["buffer_diffs"]
    assert any("inv_freq" in name for name in recorded["buffer_diffs"])
    assert (
        native_fp_recomputation["outputs_bit_identical"]
        == recorded["output"]["bit_identical"]
        is False
    )
    # magnitude class only — the exact float is volatile across processes
    assert (native_fp_recomputation["rel_norm"] <= 0.05) == recorded["output"][
        "within_rel_norm_0p05"
    ]


def test_adapter_roundtrips_recorded(artifact):
    for case in ("native_peft", "custom_adapter"):
        row = artifact["persistence"][case]
        assert row["status"] == "observed"
        assert row["adapter_keys_equal"] is True
        assert row["output"]["bit_identical"] is True, (
            f"{case}: adapter roundtrip is expected to be exact on CPU"
        )


def test_autoconfig_roundtrip_asymmetry_recorded(artifact):
    families = artifact["persistence"]["autoconfig_roundtrip"]["families"]
    assert families["dream"]["roundtrip"] == "failed"
    assert families["llada"]["roundtrip"] == "failed"
    assert families["tiny_a2d_llama"]["roundtrip"] == "ok"
    assert families["modernbert_diffusion"]["roundtrip"] == "ok"


def test_generation_reload_row(artifact):
    row = artifact["persistence"]["generation_reload"]
    assert row["status"] == "observed"
    assert row["tokens_equal"] is True
    # the reloaded model carries DreamGenerationConfig, yet the default-config
    # path crashes — the #189 evidence
    assert row["generation_config_class_after_reload"] == "DreamGenerationConfig"
    assert row["default_config_reload_raised"] is not None
