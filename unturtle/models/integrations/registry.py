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

from collections.abc import Sequence
from typing import Any

from .base import BackboneIntegration
from .sparse_output import (
    SPARSE_OUTPUT_CAPABILITY,
    SparseOutputAccess,
    standard_sparse_output,
)


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


# --- Fast-path providers (#185) ---------------------------------------------
#
# Every family's PEFT patching lives in its own provider module. The resolvers
# are zero-arg and import lazily, keeping this module free of backbone and
# kernel imports; the registry holds NO per-family patch or report helpers.


def _a2d_fast_paths() -> Any:
    """The Tiny-A2D provider module (#185) — imported only when a lookup needs it."""
    from unturtle.models.conversion.a2d.tiny_a2d import fast_paths

    return fast_paths


def _dream_fast_paths() -> Any:
    """The Dream provider module (#185) — imported only when a lookup needs it."""
    from unturtle.models.backbones.dream import fast_paths

    return fast_paths


def _llada_fast_paths() -> Any:
    """The LLaDA provider module (#185) — imported only when a lookup needs it."""
    from unturtle.models.backbones.llada import fast_paths

    return fast_paths


def _modernbert_fast_paths() -> Any:
    """The ModernBERT provider module (#185) — imported only when a lookup needs it."""
    from unturtle.models.backbones.modernbert import fast_paths

    return fast_paths


def _builtin_integrations() -> list[BackboneIntegration]:
    """The builtin integration set, in its frozen registration order (#142)."""
    return [
        BackboneIntegration(
            name="llada",
            model_types=("llada",),
            _native_resolver=_llada,
            peft_model_types=("llada",),
            _fast_paths_resolver=_llada_fast_paths,
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
            peft_model_types=("dream", "Dream"),
            _fast_paths_resolver=_dream_fast_paths,
            capabilities=frozenset({"masked_generation", "block_decode"}),
        ),
        BackboneIntegration(
            name="tiny-a2d-llama",
            model_types=("tiny-a2d-llama",),
            _native_resolver=_tiny_a2d_llama,
            # A PEFT-wrapped converted model reports its base architecture, so the
            # plain names must dispatch here too.
            peft_model_types=("tiny-a2d-llama", "llama"),
            _fast_paths_resolver=_a2d_fast_paths,
            _sparse_output_resolver=standard_sparse_output,
            capabilities=frozenset(
                {"masked_generation", "block_decode", SPARSE_OUTPUT_CAPABILITY}
            ),
        ),
        BackboneIntegration(
            name="tiny-a2d-qwen2",
            model_types=("tiny-a2d-qwen2",),
            _native_resolver=_tiny_a2d_qwen2,
            peft_model_types=("tiny-a2d-qwen2", "qwen2"),
            _fast_paths_resolver=_a2d_fast_paths,
            _sparse_output_resolver=standard_sparse_output,
            capabilities=frozenset(
                {"masked_generation", "block_decode", SPARSE_OUTPUT_CAPABILITY}
            ),
        ),
        BackboneIntegration(
            name="tiny-a2d-qwen3",
            model_types=("tiny-a2d-qwen3",),
            _native_resolver=_tiny_a2d_qwen3,
            peft_model_types=("tiny-a2d-qwen3", "qwen3"),
            _fast_paths_resolver=_a2d_fast_paths,
            _sparse_output_resolver=standard_sparse_output,
            capabilities=frozenset(
                {"masked_generation", "block_decode", SPARSE_OUTPUT_CAPABILITY}
            ),
        ),
        BackboneIntegration(
            name="modernbert-diffusion",
            # No native class: loads through FastModel, but is PEFT-patchable.
            model_types=(),
            peft_model_types=("modernbert-diffusion",),
            _fast_paths_resolver=_modernbert_fast_paths,
            capabilities=frozenset({"masked_generation"}),
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


def register_integration_into(hub, integration: BackboneIntegration) -> None:
    """Register onto an explicit hub, with this registry's conflict rules.

    The model_type / peft_model_type namespaces are integration-domain
    invariants, so they live here rather than in the generic substrate.
    """
    for existing in hub.backbone_integrations.values():
        clashes = set(existing.model_types) & set(integration.model_types)
        if clashes:
            raise ValueError(
                f"model_type(s) {sorted(clashes)} already registered by "
                f"{existing.name!r}; cannot also register {integration.name!r}"
            )
        peft_clashes = set(existing.peft_model_types) & set(
            integration.peft_model_types
        )
        if peft_clashes:
            raise ValueError(
                f"peft model_type(s) {sorted(peft_clashes)} already registered by "
                f"{existing.name!r}; cannot also register {integration.name!r}"
            )
    hub.backbone_integrations.register(integration)


def populate_integration_registry(hub) -> None:
    """Explicit builtin bootstrap for a RegistryHub (#142).

    Importing this module registers nothing; the zero-arg resolvers keep
    backbone imports lazy exactly as before.
    """
    for integration in _builtin_integrations():
        register_integration_into(hub, integration)


def _default_integrations() -> list[BackboneIntegration]:
    """The default hub's LIVE backing list (also served as `_INTEGRATIONS`
    for long-standing white-box tests that mutate it directly)."""
    from unturtle.registry import ensure_default_hub

    return ensure_default_hub().backbone_integrations._items


def iter_integrations() -> tuple[BackboneIntegration, ...]:
    """Every registered integration, in registration order."""
    return tuple(_default_integrations())


def register_integration(integration: BackboneIntegration) -> None:
    """Add an integration to the registry.

    Raises:
        ValueError: if any of its ``model_types`` or ``peft_model_types`` is
            already claimed, or (since #142, a recorded tightening over the
            old module-global list) if its ``name`` is already registered.
            Two integrations answering to one key means the winner depends
            on registration order, which is exactly the kind of silent
            behavior change this registry exists to prevent.
    """
    from unturtle.registry import ensure_default_hub

    register_integration_into(ensure_default_hub(), integration)


def _unregister_integration(integration: BackboneIntegration) -> None:
    """Remove an integration.  Test/teardown helper, not a public API.

    Removes by *identity*: ``BackboneIntegration`` is a frozen dataclass, so
    ``list.remove`` would match by value and two tests registering equal-looking
    entries would silently unregister each other's, leaving the registry
    permanently short for everything that runs after.
    """
    from unturtle.registry import ensure_default_hub

    ensure_default_hub().backbone_integrations.unregister(integration)


def find_integration_in(
    integrations: Sequence[BackboneIntegration], model_type: str | None
) -> BackboneIntegration | None:
    """Explicit-source lookup (#143 seam): the load-vocabulary namespace."""
    if model_type is None:
        return None
    for integration in integrations:
        if model_type in integration.model_types:
            return integration
    return None


def find_peft_integration_in(
    integrations: Sequence[BackboneIntegration], model_type: str | None
) -> BackboneIntegration | None:
    """Explicit-source lookup (#143 seam): the PEFT-vocabulary namespace."""
    if model_type is None:
        return None
    for integration in integrations:
        if model_type in integration.peft_model_types:
            return integration
    return None


def find_integration(model_type: str | None) -> BackboneIntegration | None:
    """The integration claiming ``model_type``, or ``None``."""
    return find_integration_in(_default_integrations(), model_type)


def resolve_native_class(model_type: str | None) -> Any | None:
    """The Unturtle class that loads ``model_type`` natively, or ``None``."""
    integration = find_integration(model_type)
    return integration.native_model_cls if integration is not None else None


def resolve_post_load_wrapper(model_type: str | None) -> Any | None:
    """The wrapper class to ``__class__``-swap after an upstream load."""
    integration = find_integration(model_type)
    return integration.post_load_wrapper_cls if integration is not None else None


def find_peft_integration(model_type: str | None) -> BackboneIntegration | None:
    """The integration that PEFT-patches ``model_type``, or ``None``.

    Separate from :func:`find_integration` because the PEFT vocabulary is not
    the load vocabulary: a PEFT-wrapped Tiny-A2D model reports plain ``llama``,
    and ModernBERT is patchable without being natively loadable.
    """
    return find_peft_integration_in(_default_integrations(), model_type)


def resolve_peft_patcher(model_type: str | None) -> Any | None:
    """The PEFT patch function for ``model_type``, or ``None``."""
    integration = find_peft_integration(model_type)
    return integration.peft_patcher if integration is not None else None


def supported_peft_model_types() -> list[str]:
    """Every PEFT-patchable ``model_type``, sorted — for error messages."""
    return sorted({mt for i in _default_integrations() for mt in i.peft_model_types})


def native_model_classes() -> dict[str, Any]:
    """``model_type`` → native class, skipping families that fail to import.

    Families whose optional dependencies are missing drop out individually,
    matching the loader's long-standing per-entry ``except ImportError``.
    """
    classes: dict[str, Any] = {}
    for integration in _default_integrations():
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
    for integration in _default_integrations():
        if integration._wrapper_resolver is None:
            continue
        for model_type in integration.model_types:
            swaps[model_type] = integration._wrapper_resolver
    return swaps


def resolve_sparse_output(model: Any) -> SparseOutputAccess | None:
    """Sparse-output access for ``model``, or ``None`` if unsupported.

    Takes the model rather than a ``model_type`` so callers (#61) never touch
    ``config``, an ``isinstance`` ladder, or a model hierarchy.  Returns
    ``None`` both for families that have not opted in and for models whose
    shape cannot support the path, so the dense fallback stays automatic.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    integration = find_integration(model_type)
    if integration is None or not integration.has_capability(SPARSE_OUTPUT_CAPABILITY):
        return None
    resolver = integration._sparse_output_resolver
    if resolver is None:
        return None
    return resolver(model)


def __getattr__(name: str):
    if name == "_INTEGRATIONS":
        # The LIVE default-hub backing list — the long-standing white-box test
        # seam (direct appends/removals) keeps working unchanged.
        return _default_integrations()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
