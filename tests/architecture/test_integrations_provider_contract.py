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
DREAM_PROVIDER = "unturtle.models.backbones.dream.fast_paths"
LLADA_PROVIDER = "unturtle.models.backbones.llada.fast_paths"
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


ALL_FAMILY_PROVIDERS = [(f, A2D_PROVIDER) for f in A2D_FAMILIES] + [
    ("modernbert-diffusion", MODERNBERT_PROVIDER),
    ("dream", DREAM_PROVIDER),
    ("llada", LLADA_PROVIDER),
]


@pytest.mark.parametrize("family,provider", ALL_FAMILY_PROVIDERS)
def test_every_fast_path_family_resolves_via_its_provider(
    integrations, family, provider
):
    """Series-end contract (#185): every fast-path family declares its patcher
    through a provider — none patches through the façade any more."""
    row = integrations[family]
    assert row["peft_patcher"]["declared"] is True, row
    assert row["peft_patcher"]["resolved"] is True, row
    assert row["peft_patcher"]["target"] == f"{provider}.patch_peft", row
    assert row["peft_patcher"]["via"] == "fast_paths", row
    assert row["fast_paths"]["resolved"] is True, row
    assert row["fast_paths"]["target"] == provider, row


def test_no_family_row_patches_through_the_facade(integrations):
    """No integration row may report a raw-field (`_peft_patcher`) patcher."""
    for family, row in integrations.items():
        patcher = row["peft_patcher"]
        if patcher.get("declared"):
            assert patcher.get("via") == "fast_paths", (family, patcher)


def test_facade_and_registry_hold_no_family_helpers():
    """Series-end shrink gate: zero family patchers in the façade, zero
    family patch/report helpers in the registry, zero provider→façade imports."""
    import ast
    import importlib
    import inspect

    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.integrations import registry as registry_mod

    remaining = [n for n in dir(fdm) if n.startswith("_patch_") and n.endswith("_peft")]
    assert remaining == [], remaining
    # Family-specific patch/report helpers must be gone; the generic public
    # resolver (resolve_peft_patcher) is loader API, not a family helper.
    families = ("a2d", "dream", "llada", "modernbert")
    helpers = [
        n
        for n in dir(registry_mod)
        if n.startswith("_")
        and any(f in n for f in families)
        and (n.endswith("_patcher") or n.endswith("_report"))
    ]
    assert helpers == [], helpers
    assert not hasattr(registry_mod, "_loader_attr")
    assert not hasattr(registry_mod, "_n_layers")
    for _, provider in ALL_FAMILY_PROVIDERS:
        module = importlib.import_module(provider)
        for name, value in vars(module).items():
            assert getattr(value, "__module__", "") != fdm.__name__, (provider, name)
        tree = ast.parse(inspect.getsource(module))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not any(
            m.startswith("unturtle.fast_diffusion_model") for m in imported
        ), (provider, imported)


def test_artifact_matches_the_live_registry():
    """The frozen rows agree with what the registry resolves in this process."""
    from unturtle.models.integrations import find_peft_integration

    for family, provider in ALL_FAMILY_PROVIDERS:
        integration = find_peft_integration(family)
        patcher = integration.peft_patcher
        assert (
            f"{patcher.__module__}.{patcher.__qualname__}" == f"{provider}.patch_peft"
        ), family
