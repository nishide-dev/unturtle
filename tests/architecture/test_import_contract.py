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

"""Fresh-process import side effects vs the artifact's imports section.

Spawns its own subprocess (the probe harness, but the COMPARISON values come
from this test's parsing, not the producer's assembly) and checks the
load-bearing semantic claims.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "184-architecture-contract-v1.json"
PROBE = REPO_ROOT / "benchmarks" / "architecture" / "subprocess_probe.py"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def fresh_import_observation() -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out = pathlib.Path(handle.name)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["UNTURTLE_EXPECTED_ROOT"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "import",
            "--out",
            str(out),
            "--json",
            json.dumps({"module": "unturtle"}),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-500:]
    return json.loads(out.read_text())


def test_import_side_effects_are_stable(artifact, fresh_import_observation):
    recorded = artifact["imports"]["unturtle"]
    fresh = fresh_import_observation
    assert fresh["torch"]["cuda_initialized"] == recorded["torch"]["cuda_initialized"]
    assert (
        fresh["autoclass"]["config_mapping_extra"]
        == (recorded["autoclass"]["config_mapping_extra"])
    )
    assert (
        fresh["default_registry_hub"]["axes"]
        == recorded["default_registry_hub"]["axes"]
    )
    assert sorted(fresh["unsloth_env"]) == sorted(recorded["unsloth_env"])
    assert fresh["modules_after"]["top_level"] == recorded["modules_after"]["top_level"]


def test_recorded_contract_facts(artifact):
    """The load-bearing import facts the rest of the roadmap relies on."""
    recorded = artifact["imports"]["unturtle"]
    # importing unturtle initializes CUDA and bootstraps the default hub —
    # both are process-global side effects #184 exists to make explicit.
    assert recorded["torch"]["cuda_initialized"] is True
    hub = recorded["default_registry_hub"]
    assert hub["default_hub_created"] is True and hub["bootstrapped"] is True
    assert "mdlm" in hub["axes"]["generation_algorithms"]
    # Dream/LLaDA are NOT AutoConfig-registered; the converted families are.
    registered = {
        entry.split("->")[0] for entry in recorded["autoclass"]["config_mapping_extra"]
    }
    assert "tiny-a2d-llama" in registered
    assert "Dream" not in registered and "dream" not in registered
    assert "llada" not in registered
