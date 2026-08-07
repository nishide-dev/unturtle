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

"""
The BackboneIntegration registry.

Registrations are declared **here**, centrally, rather than by the backbone
modules themselves.  That is deliberate and worth explaining, because the
inverse ("backbones self-register on import") is the more obvious design:

``unturtle.models.backbones.__init__`` is eager, so a self-registering
backbone would have to import this module, and anything this module imports
from the loader would close the cycle
``fast_diffusion_model -> models.backbones -> registry -> fast_diffusion_model``
against a partially-initialized module.  Central declaration keeps the
dependency one-directional.

What #68 actually asks for is preserved either way: adding a model family is
one declarative entry rather than a hand-written ``try/except`` block plus an
``elif`` arm in the loader.
"""

from __future__ import annotations

from typing import Any

from .base import BackboneIntegration


def _llada() -> Any:
    from unturtle.models.backbones.llada import LLaDAModelLM

    return LLaDAModelLM


def _mdlm_dit() -> Any:
    from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

    return MDLMDiTForMaskedDiffusionLM


def _dream() -> Any:
    from unturtle.models.backbones.dream import DreamModel

    return DreamModel


def _tiny_a2d_llama() -> Any:
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaLMHeadModel,
    )

    return TinyA2DLlamaLMHeadModel


def _tiny_a2d_qwen2() -> Any:
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen2 import (
        TinyA2DQwen2LMHeadModel,
    )

    return TinyA2DQwen2LMHeadModel


def _tiny_a2d_qwen3() -> Any:
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3 import (
        TinyA2DQwen3LMHeadModel,
    )

    return TinyA2DQwen3LMHeadModel


def _diffusion_gemma_wrapper() -> Any:
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    return UnturtleDiffusionGemmaForBlockDiffusion


_INTEGRATIONS: list[BackboneIntegration] = [
    BackboneIntegration(
        name="llada",
        model_types=("llada",),
        _native_resolver=_llada,
        capabilities=frozenset({"masked_generation", "block_decode"}),
    ),
    BackboneIntegration(
        name="mdlm-dit",
        model_types=("mdlm-dit",),
        _native_resolver=_mdlm_dit,
        capabilities=frozenset({"masked_generation"}),
    ),
    BackboneIntegration(
        name="dream",
        # DreamConfig.model_type is "Dream" (capital D); Hub configs use both.
        model_types=("dream", "Dream"),
        _native_resolver=_dream,
        capabilities=frozenset({"masked_generation", "block_decode"}),
    ),
    BackboneIntegration(
        name="tiny-a2d-llama",
        model_types=("tiny-a2d-llama",),
        _native_resolver=_tiny_a2d_llama,
        capabilities=frozenset({"masked_generation", "block_decode"}),
    ),
    BackboneIntegration(
        name="tiny-a2d-qwen2",
        model_types=("tiny-a2d-qwen2",),
        _native_resolver=_tiny_a2d_qwen2,
        capabilities=frozenset({"masked_generation", "block_decode"}),
    ),
    BackboneIntegration(
        name="tiny-a2d-qwen3",
        model_types=("tiny-a2d-qwen3",),
        _native_resolver=_tiny_a2d_qwen3,
        capabilities=frozenset({"masked_generation", "block_decode"}),
    ),
    BackboneIntegration(
        name="diffusion-gemma",
        # Loads through upstream/FastModel; Unturtle adds only a generate shim,
        # so the wrapper is installed by __class__ swap rather than by loading.
        model_types=("diffusion_gemma",),
        _wrapper_resolver=_diffusion_gemma_wrapper,
        capabilities=frozenset({"canvas_block_generation"}),
    ),
]


def iter_integrations() -> tuple[BackboneIntegration, ...]:
    """Every registered integration, in registration order."""
    return tuple(_INTEGRATIONS)


def register_integration(integration: BackboneIntegration) -> None:
    """Add an integration to the registry.

    Raises:
        ValueError: if any of its ``model_types`` is already claimed.  Two
            integrations answering to one ``model_type`` means the winner
            depends on registration order, which is exactly the kind of
            silent behavior change this registry exists to prevent.
    """
    for existing in _INTEGRATIONS:
        clashes = set(existing.model_types) & set(integration.model_types)
        if clashes:
            raise ValueError(
                f"model_type(s) {sorted(clashes)} already registered by "
                f"{existing.name!r}; cannot also register {integration.name!r}"
            )
    _INTEGRATIONS.append(integration)


def _unregister_integration(integration: BackboneIntegration) -> None:
    """Remove an integration.  Test/teardown helper, not a public API.

    Removes by *identity*: ``BackboneIntegration`` is a frozen dataclass, so
    ``list.remove`` would match by value and two tests registering equal-looking
    entries would silently unregister each other's, leaving the registry
    permanently short for everything that runs after.
    """
    for index, existing in enumerate(_INTEGRATIONS):
        if existing is integration:
            del _INTEGRATIONS[index]
            return


def find_integration(model_type: str | None) -> BackboneIntegration | None:
    """The integration claiming ``model_type``, or ``None``."""
    if model_type is None:
        return None
    for integration in _INTEGRATIONS:
        if model_type in integration.model_types:
            return integration
    return None


def resolve_native_class(model_type: str | None) -> Any | None:
    """The Unturtle class that loads ``model_type`` natively, or ``None``."""
    integration = find_integration(model_type)
    return integration.native_model_cls if integration is not None else None


def resolve_post_load_wrapper(model_type: str | None) -> Any | None:
    """The wrapper class to ``__class__``-swap after an upstream load."""
    integration = find_integration(model_type)
    return integration.post_load_wrapper_cls if integration is not None else None


def native_model_classes() -> dict[str, Any]:
    """``model_type`` → native class, skipping families that fail to import.

    Families whose optional dependencies are missing drop out individually,
    matching the loader's long-standing per-entry ``except ImportError``.
    """
    classes: dict[str, Any] = {}
    for integration in _INTEGRATIONS:
        cls = integration.native_model_cls
        if cls is None:
            continue
        for model_type in integration.model_types:
            classes[model_type] = cls
    return classes


def post_load_class_swaps() -> dict[str, Any]:
    """``model_type`` → zero-arg resolver returning the wrapper class.

    Returns resolvers rather than classes so that building this map stays
    free of backbone imports.
    """
    swaps: dict[str, Any] = {}
    for integration in _INTEGRATIONS:
        if integration._wrapper_resolver is None:
            continue
        for model_type in integration.model_types:
            swaps[model_type] = integration._wrapper_resolver
    return swaps
