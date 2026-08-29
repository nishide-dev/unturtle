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

"""Artifact model contracts vs the live runtime — MROs and method owners are
recomputed here with INLINE logic (not the diagnostics helpers), per the #184
independence requirement."""

from __future__ import annotations

import importlib
import inspect
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "184-architecture-contract-v1.json"

pytestmark = [pytest.mark.gpu]  # importing unturtle requires the unsloth chain


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text())


def _resolve_class(fqn: str) -> type:
    module_name, _, qualname = fqn.rpartition(".")
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _own_mro(cls: type) -> list[str]:
    return [f"{base.__module__}.{base.__qualname__}" for base in cls.__mro__]


def _own_owner(cls: type, name: str) -> dict:
    try:
        static = inspect.getattr_static(cls, name)
    except AttributeError:
        return {"status": "absent"}
    defined_in = None
    for base in cls.__mro__:
        if name in vars(base):
            defined_in = f"{base.__module__}.{base.__qualname__}"
            break
    func = getattr(static, "__func__", static)
    return {
        "defined_in": defined_in,
        "qualname": getattr(func, "__qualname__", None),
    }


def _observed_families(artifact) -> list[tuple[str, dict]]:
    import unturtle.models  # noqa: F401 — fire registrations first

    return [
        (family, row)
        for family, row in artifact["models"].items()
        if row.get("status") == "observed"
    ]


def test_mros_match_runtime(artifact):
    for family, row in _observed_families(artifact):
        cls = _resolve_class(row["model_class"])
        assert _own_mro(cls) == row["mro"], f"{family}: MRO drifted"


def test_method_owners_match_runtime(artifact):
    for family, row in _observed_families(artifact):
        cls = _resolve_class(row["model_class"])
        for name, recorded in row["method_owners"].items():
            own = _own_owner(cls, name)
            if recorded.get("status") == "absent":
                assert own.get("status") == "absent", (family, name)
                continue
            assert own["defined_in"] == recorded["defined_in"], (family, name)
            assert own["qualname"] == recorded["qualname"], (family, name)


def test_model_type_and_config_match_runtime(artifact):
    for family, row in _observed_families(artifact):
        cls = _resolve_class(row["model_class"])
        config_cls = getattr(cls, "config_class", None)
        if row["declared_config_class"] is None:
            assert config_cls is None, family
        else:
            assert (
                f"{config_cls.__module__}.{config_cls.__qualname__}"
                == row["declared_config_class"]
            ), family
            assert getattr(config_cls, "model_type", None) == row["model_type"], family


def test_public_all_matches_artifact(artifact):
    """Independent read of __all__ — not the producer's describe() helper."""
    import unturtle
    import unturtle.models

    recorded = artifact["public_api"]
    assert sorted(unturtle.__all__) == recorded["unturtle"]["all"]
    assert sorted(unturtle.models.__all__) == recorded["unturtle.models"]["all"]
    # declared-but-None exports (the DEPRECATE evidence) recomputed inline
    unresolved = sorted(
        name for name in unturtle.__all__ if getattr(unturtle, name, None) is None
    )
    recorded_unresolved = sorted(
        name
        for name, sym in recorded["unturtle"]["symbols"].items()
        if not sym["resolved"]
    )
    assert unresolved == recorded_unresolved


def test_autoclass_registration_matches_runtime(artifact):
    """Independent read of the transformers extra-content mappings."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    import unturtle.models  # noqa: F401

    extra_types = {
        key if isinstance(key, str) else getattr(key, "__name__", str(key))
        for key in getattr(CONFIG_MAPPING, "_extra_content", {})
    }
    for family, row in _observed_families(artifact):
        expected = row["autoclass_config_registered"]
        actual = row["model_type"] in extra_types
        assert actual == expected, (family, row["model_type"], extra_types)
