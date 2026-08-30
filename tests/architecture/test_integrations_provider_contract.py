"""#185 family-extraction gate read off the committed #184 artifact.

A family that ships a fast-path provider must still be recorded as PEFT-patchable
(through the provider), and the provider must be the one the registry resolves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "docs/artifacts/184-architecture-contract-v1.json"
)
A2D_PROVIDER = "unturtle.models.conversion.a2d.tiny_a2d.fast_paths"
MODERNBERT_PROVIDER = "unturtle.models.backbones.modernbert.fast_paths"
A2D_FAMILIES = ("tiny-a2d-llama", "tiny-a2d-qwen2", "tiny-a2d-qwen3")


@pytest.fixture(scope="module")
def integrations():
    data = json.loads(ARTIFACT.read_text())
    section = data["integrations"]
    assert section["status"] == "observed", section
    return section["integrations"]


@pytest.mark.parametrize("family", A2D_FAMILIES)
def test_tiny_a2d_declares_its_patcher_through_the_provider(integrations, family):
    row = integrations[family]
    assert row["peft_patcher"]["declared"] is True, row
    assert row["peft_patcher"]["resolved"] is True, row
    assert row["peft_patcher"]["target"] == f"{A2D_PROVIDER}.patch_peft", row
    assert row["peft_patcher"]["via"] == "fast_paths", row
    assert row["fast_paths"]["resolved"] is True, row
    assert row["fast_paths"]["target"] == A2D_PROVIDER, row


def test_modernbert_declares_its_patcher_through_the_provider(integrations):
    row = integrations["modernbert-diffusion"]
    assert row["peft_patcher"]["declared"] is True, row
    assert row["peft_patcher"]["resolved"] is True, row
    assert row["peft_patcher"]["target"] == f"{MODERNBERT_PROVIDER}.patch_peft", row
    assert row["peft_patcher"]["via"] == "fast_paths", row
    assert row["fast_paths"]["resolved"] is True, row
    assert row["fast_paths"]["target"] == MODERNBERT_PROVIDER, row


@pytest.mark.parametrize("family", ("dream", "llada"))
def test_unextracted_families_still_patch_through_the_facade(integrations, family):
    row = integrations[family]
    assert row["peft_patcher"]["declared"] is True, row
    assert row["peft_patcher"]["via"] == "_peft_patcher", row
    assert row["peft_patcher"]["target"].startswith("unturtle.fast_diffusion_model."), (
        row
    )
    assert row["fast_paths"] == {"declared": False}, row


def test_artifact_matches_the_live_registry():
    """The frozen rows agree with what the registry resolves in this process."""
    from unturtle.models.integrations import find_peft_integration

    for family, provider in [(f, A2D_PROVIDER) for f in A2D_FAMILIES] + [
        ("modernbert-diffusion", MODERNBERT_PROVIDER)
    ]:
        integration = find_peft_integration(family)
        patcher = integration.peft_patcher
        assert (
            f"{patcher.__module__}.{patcher.__qualname__}" == f"{provider}.patch_peft"
        ), family
