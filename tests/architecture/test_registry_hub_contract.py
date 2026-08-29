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

"""The standalone RegistryHub contract vs live recomputation (#184 blocker).

Every claim is re-derived with this test's own code — a producer that
pre-populates the "empty" hub, shares ``_items`` between hubs, hides
default-hub side effects, or reorders the bootstrap disagrees here.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "docs" / "artifacts" / "184-architecture-contract-v1.json"

pytestmark = [pytest.mark.gpu]  # importing unturtle requires the unsloth chain

# Literal copy of the axis list — not imported from the producer.
HUB_AXES = (
    "generation_algorithms",
    "backbone_integrations",
    "processes",
    "training_recipes",
    "conversions",
    "post_training_recipes",
    "methods",
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text())


def _own_ordered_names(hub) -> dict:
    return {
        axis: [value.name for value in getattr(hub, axis).values()] for axis in HUB_AXES
    }


def _default_hub_snapshot():
    import unturtle.registry as registry_mod

    hub = registry_mod._default_hub
    return None if hub is None else _own_ordered_names(hub)


def test_fresh_hub_is_empty_and_side_effect_free(artifact):
    """Kills the pre-populated-empty-hub mutant and the hidden-side-effect
    mutant: the artifact must claim emptiness/no-side-effects AND the live
    runtime must agree."""
    from unturtle.registry import RegistryHub

    cell = artifact["registry_hub"]["fresh_empty_hub"]
    assert cell["all_axes_empty"] is True
    assert all(names == [] for names in cell["axis_names"].values())
    surroundings = cell["surroundings"]
    assert surroundings["default_hub_changed"] is False
    assert surroundings["autoclass_changed"] is False
    assert surroundings["environ_changed_keys"] == []
    assert surroundings["python_rng_consumed"] is False
    assert surroundings["torch_rng_consumed"] is False

    default_before = _default_hub_snapshot()
    live = RegistryHub()
    assert all(getattr(live, axis).values() == () for axis in HUB_AXES)
    assert live._bootstrapped is False
    assert _default_hub_snapshot() == default_before


def test_bootstrap_content_and_order_match_runtime(artifact):
    """Kills the bootstrap-reorder mutant: the artifact's ordered axis names
    must equal a LIVE bootstrap's insertion order, exactly."""
    from unturtle.registry import RegistryHub, bootstrap_builtin_hub

    cell = artifact["registry_hub"]["explicit_builtin_bootstrap"]
    assert cell["deterministic_across_two_bootstraps"] is True

    default_before = _default_hub_snapshot()
    live_hub = bootstrap_builtin_hub(RegistryHub())
    assert _own_ordered_names(live_hub) == cell["ordered_axis_names"]
    assert live_hub._bootstrapped is True
    # bootstrapping a SUPPLIED hub must not touch the default hub
    assert _default_hub_snapshot() == default_before


def test_repeat_bootstrap_is_duplicate_rejection(artifact):
    """Frozen as observed: strict rejection, not idempotence — and the hub
    must be left un-mutated by the failed second bootstrap."""
    from unturtle.registry import RegistryHub, bootstrap_builtin_hub

    cell = artifact["registry_hub"]["repeat_bootstrap"]
    assert cell["behavior"] == "duplicate_rejection"
    assert str(cell["raised"]).startswith("ValueError")
    assert cell["axis_counts_unchanged"] is True

    live_hub = bootstrap_builtin_hub(RegistryHub())
    counts_before = {axis: len(getattr(live_hub, axis).values()) for axis in HUB_AXES}
    with pytest.raises(ValueError, match="already bootstrapped"):
        bootstrap_builtin_hub(live_hub)
    assert {
        axis: len(getattr(live_hub, axis).values()) for axis in HUB_AXES
    } == counts_before


def test_hubs_do_not_share_backing_storage(artifact):
    """Kills the shared-_items mutant: a registration into one hub must be
    invisible to another hub AND to the default hub, live and in the
    artifact."""
    from unturtle.registry import (
        DuplicateRegistrationError,
        RegistryHub,
        bootstrap_builtin_hub,
    )

    cell = artifact["registry_hub"]["hub_isolation"]
    assert cell["sentinel_visible_in_registering_hub"] is True
    assert cell["sentinel_leaked_to_other_hub"] is False
    assert cell["sentinel_leaked_to_default_hub"] is False
    assert cell["registry_objects_shared"] is False
    assert cell["backing_storage_shared"] is False
    assert str(cell["duplicate_registration_raised"]).startswith(
        "DuplicateRegistrationError"
    )

    hub_a = RegistryHub()
    hub_b = bootstrap_builtin_hub(RegistryHub())
    default_before = _default_hub_snapshot()
    sentinel = types.SimpleNamespace(name="test-sentinel-isolation")
    hub_a.processes.register(sentinel)
    assert hub_a.processes.find("test-sentinel-isolation") is sentinel
    assert hub_b.processes.find("test-sentinel-isolation") is None
    assert _default_hub_snapshot() == default_before
    assert all(
        getattr(hub_a, axis)._items is not getattr(hub_b, axis)._items
        for axis in HUB_AXES
    )
    with pytest.raises(DuplicateRegistrationError):
        hub_a.processes.register(types.SimpleNamespace(name="test-sentinel-isolation"))
