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

"""Process-global-state rows vs live recomputation.

- #188: the RNG contract defect is re-derived here with this test's own
  digesting (a producer that hides the environment difference disagrees);
- the SDPA flags row must match the runtime flags of a fresh process default.
"""

from __future__ import annotations

import hashlib
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


def _tensor_digest(tensor) -> str:
    data = tensor.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(
        str(tuple(data.shape)).encode() + data.numpy().tobytes()
    ).hexdigest()[:16]


def _wrap_lora_a_digest(pre_consume: int) -> str:
    from unturtle import FastDiffusionModel
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
    model = DreamModel(config).to(torch.float32)
    torch.manual_seed(100)
    if pre_consume:
        torch.randn(pre_consume)
    peft_model = FastDiffusionModel.get_peft_model(
        model,
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj"],
        use_gradient_checkpointing=False,
        random_state=3407,
    )
    for name, param in peft_model.named_parameters():
        if ".lora_A." in name:
            return _tensor_digest(param)
    raise AssertionError("no lora_A parameter found")


def test_rng_contract_defect_is_real_and_recorded(artifact):
    row = artifact["process_global_state"]["rng_contract"]
    assert row["linked_issue"] == 188
    assert row["classification"] == "known_defect"
    assert row["same_random_state_same_adapters"] is False

    digest_clean = _wrap_lora_a_digest(pre_consume=0)
    digest_shifted = _wrap_lora_a_digest(pre_consume=7)
    # same random_state argument, different prior RNG consumption — the
    # adapters MUST currently differ (that IS the defect). When #188 is
    # fixed, this test and the artifact row change together.
    assert digest_clean != digest_shifted


def test_sdpa_row_matches_runtime_defaults(artifact):
    """Compared against a FRESH process: the artifact records process-start
    defaults, and this long-lived pytest process's flags are legitimately
    mutated by other tests (e.g. sdpa_kernel contexts, unsloth patches)."""
    import json as json_mod
    import os
    import subprocess
    import sys
    import tempfile

    row = artifact["process_global_state"]["sdpa"]
    probe = REPO_ROOT / "benchmarks" / "architecture" / "subprocess_probe.py"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = handle.name
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["UNTURTLE_EXPECTED_ROOT"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(probe),
            "process-global",
            "--out",
            out,
            "--json",
            json_mod.dumps({"case": "sdpa"}),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    fresh = json_mod.loads(pathlib.Path(out).read_text())
    assert fresh["available_backends"] == row["available_backends"]
    assert fresh["tf32"] == row["tf32"]
    assert fresh["deterministic"] == row["deterministic"]
    assert "pin the backend" in row["policy"] or "unit level" in row["policy"]


def test_unsloth_environment_mutation_recorded(artifact):
    row = artifact["process_global_state"]["unsloth_environment_mutation"]
    assert row["scope"] == "process_global"
    assert row["UNSLOTH_MIXED_PRECISION_before"] is None
    assert row["UNSLOTH_MIXED_PRECISION_after"] == "float32"
