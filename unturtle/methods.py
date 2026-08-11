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

"""Method composition layer (#143): describe methods, do not execute them.

Core rule (#141): **generalize composition, not implementations.**  A
:class:`MethodSpec` is a manifest referencing components by name; every
component is an opaque, LAZY :class:`ComponentRecipe` whose factory imports
the existing implementation only when called.  There is no universal
Objective, Trainer, Scheduler, or GenerationConfig here, and
:class:`ResolvedMethod` is not an execution plan — existing direct training
and generation APIs remain authoritative.

Capability hygiene: a research-only component referenced by a recipe stays
research-only.  The DFM recipe validates against the model's explicit
``supports_dfm_generation`` opt-in through the generation registry's own
probe — registering the recipe promotes nothing (#65 boundary, tested).

Field ledger for ``MethodSpec`` (the #141 stop condition — every field must
be justified by at least two concrete consumers):

- ``process`` / ``training`` / ``generation``: mdlm, dfm, flowlm, hybrid;
- ``conversion``: tiny_a2d, prediff_hybrid;
- ``required_capabilities``: dfm (opt-in), prediff_hybrid + tiny_a2d
  (masked_generation via their integrations);
- ``post_training``: the field exists per the issue sketch and the OPD
  component is registered, but no builtin spec references it yet — a full
  OPD composition needs a real consumer (a base method whose post-training
  phase is OPD) and forcing one now would invent semantics.  Deliberately
  deferred, recorded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from unturtle.registry import RegistryHub, ensure_default_hub

__all__ = [
    "ComponentRecipe",
    "MethodSpec",
    "ResolvedMethod",
    "describe_method",
    "list_methods",
    "populate_method_registry",
    "resolve_method",
    "validate_method",
]


@dataclass(frozen=True)
class ComponentRecipe:
    """A lazy, opaque reference to an existing implementation.

    ``factory`` imports and returns the implementation entry (class,
    function, or module) — introspection (:func:`describe_method`) never
    calls it, so describing recipes stays import-light."""

    name: str
    kind: str  # "process" | "training" | "conversion" | "post_training"
    factory: Callable[[], Any]
    summary: str = ""


@dataclass(frozen=True)
class MethodSpec:
    """A declarative composition manifest.  References, not behavior."""

    name: str
    process: str | None = None
    training: str | None = None
    conversion: str | None = None
    post_training: str | None = None
    generation: tuple[str, ...] = ()
    required_capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ResolvedMethod:
    """A spec with every reference resolved against ONE hub.

    Not an execution plan: it owns no loops and exposes the same lazy
    recipes the registries hold.  ``unverified_capabilities`` records
    honestly which required capabilities could not be checked (no
    integration resolvable for the model) rather than pretending they hold.
    """

    spec: MethodSpec
    process: ComponentRecipe | None
    training: ComponentRecipe | None
    conversion: ComponentRecipe | None
    post_training: ComponentRecipe | None
    generation: tuple[Any, ...]
    unverified_capabilities: frozenset[str] = field(default_factory=frozenset)


def _resolve_component(
    hub: RegistryHub, axis: str, name: str | None, method: str
) -> ComponentRecipe | None:
    if name is None:
        return None
    registry = getattr(hub, _AXIS_REGISTRIES[axis])
    recipe = registry.find(name)
    if recipe is None:
        raise ValueError(
            f"method {method!r} references {axis} {name!r}, which is not "
            f"registered in this hub (known: "
            f"{sorted(r.name for r in registry.values())})"
        )
    return recipe


_AXIS_REGISTRIES = {
    "process": "processes",
    "training": "training_recipes",
    "conversion": "conversions",
    "post_training": "post_training_recipes",
}


def resolve_method(name: str, *, hub: RegistryHub | None = None) -> ResolvedMethod:
    """Resolve every component reference of method ``name`` against ``hub``
    (default hub when omitted).  Raises loudly on any dangling reference."""
    hub = hub or ensure_default_hub()
    spec: MethodSpec = hub.methods.get(name)

    generation = []
    for algorithm_name in spec.generation:
        algorithm = hub.generation_algorithms.find(algorithm_name)
        if algorithm is None:
            raise ValueError(
                f"method {name!r} references generation algorithm "
                f"{algorithm_name!r}, which is not registered in this hub"
            )
        generation.append(algorithm)

    return ResolvedMethod(
        spec=spec,
        process=_resolve_component(hub, "process", spec.process, name),
        training=_resolve_component(hub, "training", spec.training, name),
        conversion=_resolve_component(hub, "conversion", spec.conversion, name),
        post_training=_resolve_component(
            hub, "post_training", spec.post_training, name
        ),
        generation=tuple(generation),
    )


def validate_method(
    name: str, *, model: Any = None, hub: RegistryHub | None = None
) -> ResolvedMethod:
    """Resolve ``name`` and, given a model, check the combination is
    supported BEFORE anything executes.

    - every referenced generation algorithm must support the model (the
      registry's own capability probes — the same messages the direct
      generation path raises, so e.g. the DFM opt-in and the hybrid
      block-decode exclusion hold here by construction);
    - ``required_capabilities`` are checked against the integration the
      SUPPLIED hub resolves for the model's ``config.model_type``; if no
      integration is resolvable they are recorded as unverified, never
      silently assumed to hold."""
    hub = hub or ensure_default_hub()
    resolved = resolve_method(name, hub=hub)
    if model is None:
        return resolved

    for algorithm in resolved.generation:
        if not algorithm.supports(model):
            raise ValueError(
                f"method {name!r}: {algorithm.describe_unsupported(model)}"
            )

    unverified: set[str] = set()
    if resolved.spec.required_capabilities:
        model_type = getattr(getattr(model, "config", None), "model_type", None)
        integration = hub.find_integration(model_type)
        if integration is None:
            unverified = set(resolved.spec.required_capabilities)
        else:
            missing = resolved.spec.required_capabilities - integration.capabilities
            if missing:
                raise ValueError(
                    f"method {name!r} requires capabilities {sorted(missing)} "
                    f"that integration {integration.name!r} does not declare"
                )
    if unverified:
        return ResolvedMethod(
            spec=resolved.spec,
            process=resolved.process,
            training=resolved.training,
            conversion=resolved.conversion,
            post_training=resolved.post_training,
            generation=resolved.generation,
            unverified_capabilities=frozenset(unverified),
        )
    return resolved


def describe_method(name: str, *, hub: RegistryHub | None = None) -> dict:
    """A plain-data, JSON-serializable description.  Never calls factories,
    so it imports no heavy model/trainer modules (tested in a subprocess)."""
    hub = hub or ensure_default_hub()
    resolved = resolve_method(name, hub=hub)

    def component(recipe: ComponentRecipe | None) -> dict | None:
        if recipe is None:
            return None
        return {"name": recipe.name, "kind": recipe.kind, "summary": recipe.summary}

    return {
        "name": resolved.spec.name,
        "process": component(resolved.process),
        "training": component(resolved.training),
        "conversion": component(resolved.conversion),
        "post_training": component(resolved.post_training),
        "generation": [a.name for a in resolved.generation],
        "required_capabilities": sorted(resolved.spec.required_capabilities),
    }


def list_methods(*, hub: RegistryHub | None = None) -> tuple[str, ...]:
    hub = hub or ensure_default_hub()
    return tuple(spec.name for spec in hub.methods.values())


# ---------------------------------------------------------------------------
# Builtin proof set (#143): descriptions of EXISTING paths, zero behavior
# change.  Factories lazily import the implementation current callers use.
# ---------------------------------------------------------------------------


def _masked_process() -> Any:
    from unturtle.processes.masked import MaskedDiffusionProcess

    return MaskedDiffusionProcess


def _discrete_flow_process() -> Any:
    from unturtle.processes.discrete_flow import DiscreteFlowProcess

    return DiscreteFlowProcess


def _continuous_flow_process() -> Any:
    from unturtle.processes.continuous_flow import ContinuousFlowProcess

    return ContinuousFlowProcess


def _mdlm_training() -> Any:
    from unturtle.diffusion.trainer import DiffusionTrainer

    return DiffusionTrainer


def _dfm_training() -> Any:
    from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss

    return discrete_flow_matching_loss


def _flowlm_training() -> Any:
    from unturtle.models.latent.objective import flowlm_loss

    return flowlm_loss


def _tiny_a2d_conversion() -> Any:
    from unturtle.models.conversion.a2d import tiny_a2d

    return tiny_a2d


def _prediff_hybrid_conversion() -> Any:
    from unturtle.models.conversion.a2d.tiny_a2d._hybrid import (
        maybe_build_hybrid_mask,
    )

    return maybe_build_hybrid_mask


def _opd_post_training() -> Any:
    import unturtle.post_training as opd

    return opd


def populate_method_registry(hub: RegistryHub) -> None:
    """Builtin components + the MethodSpec proof set (#143).

    Explicit, like every bootstrap since #142: importing this module
    registers nothing."""
    hub.process(
        "masked",
        factory=_masked_process,
        summary="MDLM-style masked corruption (unturtle.processes.masked)",
    )
    hub.process(
        "discrete_flow",
        factory=_discrete_flow_process,
        summary="DFM jump process with kappa schedules (#65)",
    )
    hub.process(
        "continuous_flow",
        factory=_continuous_flow_process,
        summary="continuous flow interpolation (FlowLM, #66)",
    )

    hub.training(
        "mdlm",
        factory=_mdlm_training,
        summary="DiffusionTrainer — masked diffusion objective (TRL-tier)",
    )
    hub.training(
        "dfm",
        factory=_dfm_training,
        summary="discrete flow matching loss (research-only entry, #65)",
    )
    hub.training(
        "flowlm",
        factory=_flowlm_training,
        summary="FlowLM continuous objective (prototype, #66)",
    )

    hub.conversion(
        "tiny_a2d",
        factory=_tiny_a2d_conversion,
        summary="Tiny-A2D AR->Diffusion recipe (models/conversion)",
    )
    hub.conversion(
        "prediff_hybrid",
        factory=_prediff_hybrid_conversion,
        summary="PreDiff eq.(3) hybrid attention on Tiny-A2D (#63/#127)",
    )

    hub.post_training_recipe(
        "opd",
        factory=_opd_post_training,
        summary="on-policy distillation orchestration (#64); no builtin MethodSpec references it yet — see the field ledger in unturtle/methods.py",
    )

    hub.method(
        MethodSpec(
            name="mdlm",
            process="masked",
            training="mdlm",
            generation=("mdlm",),
            required_capabilities=frozenset({"masked_generation"}),
        )
    )
    hub.method(
        MethodSpec(
            name="dfm",
            process="discrete_flow",
            training="dfm",
            generation=("dfm",),
            # The opt-in is enforced by the dfm algorithm's own supports probe
            # at validate_method(model=...) time; no integration declares a dfm
            # capability BY DESIGN (#65: research-only, unpromoted).
        )
    )
    hub.method(
        MethodSpec(
            name="flowlm",
            process="continuous_flow",
            training="flowlm",
            generation=("flowlm",),
        )
    )
    hub.method(
        MethodSpec(
            name="tiny_a2d",
            process="masked",
            training="mdlm",
            conversion="tiny_a2d",
            generation=("mdlm",),
            required_capabilities=frozenset({"masked_generation"}),
        )
    )
    hub.method(
        MethodSpec(
            name="prediff_hybrid",
            process="masked",
            training="mdlm",
            conversion="prediff_hybrid",
            # mdlm only: hybrid models must never take a cache path (#127/#128);
            # the block_decode probe enforces it at validation time too.
            generation=("mdlm",),
            required_capabilities=frozenset({"masked_generation"}),
        )
    )
