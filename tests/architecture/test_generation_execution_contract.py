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

"""Generation execution map vs a live run — the artifact's claim about WHICH
method executes (not which was selected) is re-derived here with this test's
own instrumentation.

Mutant coverage this pins:
- block_decode selected but MDLM's ``_sample`` running (invoked-method match);
- requested steps != executed NFE (mdlm NFE == steps);
- explicit unsupported falling back to auto (block_ar must raise);
- the default-config crash (#189) silently disappearing or being mistaken
  for a working path.
"""

from __future__ import annotations

import functools
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


@pytest.fixture()
def tiny_dream():
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
    model = DreamModel(config).eval()
    generation_config = DreamGenerationConfig(mask_token_id=1, pad_token_id=0)
    return model, generation_config


def _generate(model, algorithm, generation_config, *, steps=4):
    torch.manual_seed(0)
    prompt = torch.randint(2, 400, (1, 8))
    kwargs = {"generation_config": generation_config} if generation_config else {}
    with torch.no_grad():
        return model.generate(
            prompt,
            algorithm=algorithm,
            max_new_tokens=8,
            steps=steps,
            temperature=0.0,
            mask_token_id=1,
            block_length=4,
            **kwargs,
        )


def test_mdlm_executes_sample_with_requested_nfe(artifact, tiny_dream):
    model, generation_config = tiny_dream
    invoked: list[str] = []
    nfe = {"count": 0}
    original = type(model)._sample

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        invoked.append("_sample")
        return original(self, *args, **kwargs)

    type(model)._sample = wrapped
    model.register_forward_pre_hook(
        lambda module, inputs: nfe.__setitem__("count", nfe["count"] + 1)
    )
    try:
        _generate(model, "mdlm", generation_config, steps=4)
    finally:
        type(model)._sample = original

    recorded = artifact["generation"]["dream"]["per_algorithm"]["mdlm"]
    executed = recorded.get("explicit_config_run") or recorded["default_config_run"]
    assert "_sample" in invoked
    assert "_sample" in executed["invoked_methods"]
    assert nfe["count"] == 4 == executed["nfe"] == recorded["requested_steps"]


def test_explicit_unsupported_raises_not_falls_back(artifact, tiny_dream):
    model, generation_config = tiny_dream
    with pytest.raises(ValueError, match="block_ar"):
        _generate(model, "block_ar", generation_config)
    recorded = artifact["generation"]["dream"]["per_algorithm"]["block_ar"]
    assert str(recorded["default_config_run"]["raised"]).startswith("ValueError")
    assert recorded["default_config_run"]["invoked_methods"] == []


def test_default_config_defect_is_recorded_and_still_present(artifact, tiny_dream):
    """#189: the default-config path crashes; the artifact must say so, and
    the runtime must agree (when the defect is fixed, BOTH change together)."""
    model, _ = tiny_dream
    recorded = artifact["generation"]["dream"]["per_algorithm"]["mdlm"][
        "default_config_run"
    ]["raised"]
    try:
        _generate(model, "mdlm", None, steps=4)
        runtime_raised = None
    except Exception as exc:  # noqa: BLE001
        runtime_raised = f"{type(exc).__name__}"
    if recorded is None:
        assert runtime_raised is None
    else:
        assert runtime_raised is not None
        assert str(recorded).startswith(runtime_raised)


def test_auto_resolution_recorded(artifact):
    row = artifact["generation"]["dream"]["per_algorithm"]["auto"]
    executed = row.get("explicit_config_run") or row["default_config_run"]
    # auto on Dream resolves to the block-decode path, not MDLM
    assert "_block_decode_loop" in executed["invoked_methods"]
    assert "_sample_with_cache" in executed["invoked_methods"]
