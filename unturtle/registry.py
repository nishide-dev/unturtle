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

"""Instance-backed registry substrate (#142).

Deliberately boring: deterministic insertion order, duplicate rejection
(canonical names and aliases), immutable iteration snapshots, and an
explicit builtin bootstrap.  No dependency injection, no lifecycle
callbacks, no config machinery — "generalize composition, not
implementations" (#141) starts by NOT generalizing here.

What instance ownership unlocks (the #142 review question, answered in
code): an isolated hub can be built, populated, extended, and thrown away
per test or per plugin context without touching the process-global default
— the old module lists made every experiment in registration a mutation of
shared state that had to be hand-unwound.

Lookup is a linear scan over the insertion-ordered item list, NOT a dict
index: the module-level compatibility seams (``sampler._ALGORITHMS``,
``integrations._INTEGRATIONS``) expose the backing list to long-standing
white-box tests that reorder or insert into it directly, and an index would
silently desync from those mutations.  Registries hold <= a few dozen
entries; O(n) find is what the module globals did anyway.

Import discipline: this module imports nothing from ``unturtle`` at module
level.  Builtins enter a hub only through :func:`bootstrap_builtin_hub`,
which lazily imports the two population functions.  Importing any backbone
or process module never mutates a hub (tested in a subprocess).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")

__all__ = [
    "DuplicateRegistrationError",
    "Registry",
    "RegistryHub",
    "bootstrap_builtin_hub",
    "ensure_default_hub",
]


class DuplicateRegistrationError(ValueError):
    """A canonical name (or alias) is already registered.

    Subclasses ValueError so every existing ``pytest.raises(ValueError)``
    contract holds; carries the structured fields (#145) that let the plugin
    loader attribute BOTH sides of a conflict instead of parsing messages.
    """

    def __init__(self, kind: str, name: str) -> None:
        super().__init__(f"{kind} {name!r} is already registered")
        self.kind = kind
        self.name = name


class Registry(Generic[T]):
    """A named, insertion-ordered collection with duplicate rejection.

    Values must carry a ``name`` attribute (their canonical name).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        # The backing list is intentionally reachable (module seams expose it
        # to white-box tests); aliases are registry-owned bookkeeping.
        self._items: list[T] = []
        self._aliases: dict[str, str] = {}

    def _known_names(self) -> list[str]:
        return [item.name for item in self._items] + list(self._aliases)

    def register(self, value: T, *, aliases: tuple[str, ...] = ()) -> T:
        name = value.name
        taken = set(self._known_names())
        for candidate in (name, *aliases):
            if candidate in taken:
                raise DuplicateRegistrationError(self.kind, candidate)
        if len({name, *aliases}) != 1 + len(aliases):
            raise ValueError(
                f"{self.kind} {name!r}: aliases must be distinct from the "
                f"canonical name and each other"
            )
        self._items.append(value)
        for alias in aliases:
            self._aliases[alias] = name
        return value

    def find(self, name: str) -> T | None:
        canonical = self._aliases.get(name, name)
        for item in self._items:
            if item.name == canonical:
                return item
        return None

    def get(self, name: str) -> T:
        found = self.find(name)
        if found is None:
            raise KeyError(
                f"unknown {self.kind} {name!r}; known: {sorted(self._known_names())}"
            )
        return found

    def values(self) -> tuple[T, ...]:
        """Immutable snapshot in insertion order."""
        return tuple(self._items)

    def unregister(self, value: T) -> None:
        """Identity-based removal (test/teardown helper).

        Identity, not equality: frozen dataclasses compare by value, and
        removing by value could drop a different, equal-looking entry."""
        for index, item in enumerate(self._items):
            if item is value:
                del self._items[index]
                name = value.name
                # Sweep aliases only when the LAST holder of the name is
                # gone: a same-named twin removed by identity must not strip
                # the survivor's aliases (#147 review).
                if all(item.name != name for item in self._items):
                    self._aliases = {
                        alias: target
                        for alias, target in self._aliases.items()
                        if target != name
                    }
                return


class RegistryHub:
    """Owns the registry instances one registration context sees.

    This slice hosts the two registries that already exist (generation
    algorithms, backbone integrations); later #141 slices add more kinds.
    """

    def __init__(self) -> None:
        self.generation_algorithms: Registry[Any] = Registry("generation algorithm")
        self.backbone_integrations: Registry[Any] = Registry("backbone integration")
        # #143 component axes (lazy ComponentRecipe values) + method manifests.
        self.processes: Registry[Any] = Registry("process")
        self.training_recipes: Registry[Any] = Registry("training recipe")
        self.conversions: Registry[Any] = Registry("conversion recipe")
        self.post_training_recipes: Registry[Any] = Registry("post-training recipe")
        self.methods: Registry[Any] = Registry("method")
        # (registry kind, canonical name) -> provenance record for names that
        # arrived via unturtle.plugins.load_plugins.  Filled at the plugin
        # load boundary only; builtins and direct registrations are absent —
        # that absence IS their attribution ("builtin or direct
        # registration"), so builtin declarations need no rewriting (#145).
        self.plugin_provenance: dict[tuple[str, str], Any] = {}
        self._bootstrapped = False

    # -- decorator sugar (registry-bound; importing a module registers
    # nothing — only calling this, on this hub, does) ---------------------

    def generation(
        self,
        name: str,
        *,
        family: str,
        supports: Callable[[Any], bool],
        flags: dict[str, bool] | None = None,
        auto_priority: int,
        auto_eligible: bool = True,
        unsupported_message: Callable[[Any], str],
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register the decorated function as an algorithm runner on THIS hub.

        Returns the original function unchanged and applies the same
        duplicate checks as explicit registration."""

        def decorate(runner: Callable[..., Any]) -> Callable[..., Any]:
            from unturtle.models.generation.sampler import GenerationAlgorithm

            self.generation_algorithms.register(
                GenerationAlgorithm(
                    name=name,
                    family=family,
                    supports=supports,
                    flags=dict(flags or {}),
                    auto_priority=auto_priority,
                    auto_eligible=auto_eligible,
                    unsupported_message=unsupported_message,
                    runner=runner,
                )
            )
            return runner

        return decorate

    # -- component/method registration sugar (#143) ------------------------

    def _register_component(self, registry_name, kind, name, factory, summary):
        from unturtle.methods import ComponentRecipe

        recipe = ComponentRecipe(name=name, kind=kind, factory=factory, summary=summary)
        getattr(self, registry_name).register(recipe)
        return recipe

    def process(self, name: str, *, factory: Callable[[], Any], summary: str = ""):
        return self._register_component("processes", "process", name, factory, summary)

    def training(self, name: str, *, factory: Callable[[], Any], summary: str = ""):
        return self._register_component(
            "training_recipes", "training", name, factory, summary
        )

    def conversion(self, name: str, *, factory: Callable[[], Any], summary: str = ""):
        return self._register_component(
            "conversions", "conversion", name, factory, summary
        )

    def post_training_recipe(
        self, name: str, *, factory: Callable[[], Any], summary: str = ""
    ):
        return self._register_component(
            "post_training_recipes", "post_training", name, factory, summary
        )

    def method(self, spec: Any) -> Any:
        """Register a MethodSpec manifest on THIS hub."""
        return self.methods.register(spec)

    # -- hub-scoped resolution / dispatch / lookup (#143 seams) ------------

    def dispatch_generation(
        self,
        model: Any,
        request: Any,
        algorithm: str = "auto",
        *,
        bd3lm_requested: bool = False,
    ) -> Any:
        """Dispatch against THIS hub's algorithms — same code path as the
        module-level ``dispatch_generation``, so a hub-registered algorithm
        can never silently fall back to the default hub."""
        from unturtle.models.generation.sampler import dispatch_generation_from

        return dispatch_generation_from(
            self.generation_algorithms.values(),
            model,
            request,
            algorithm,
            bd3lm_requested=bd3lm_requested,
        )

    def find_integration(self, model_type: str | None) -> Any | None:
        """Load-vocabulary integration lookup against THIS hub."""
        from unturtle.models.integrations.registry import find_integration_in

        return find_integration_in(self.backbone_integrations.values(), model_type)

    def find_peft_integration(self, model_type: str | None) -> Any | None:
        """PEFT-vocabulary integration lookup against THIS hub."""
        from unturtle.models.integrations.registry import find_peft_integration_in

        return find_peft_integration_in(self.backbone_integrations.values(), model_type)

    def resolve_generation(
        self, algorithm: str, model: Any, *, bd3lm_requested: bool
    ) -> str:
        """Resolve against THIS hub's algorithms with the exact semantics of
        the module-level ``resolve_algorithm`` (same code path)."""
        from unturtle.models.generation.sampler import resolve_algorithm_from

        return resolve_algorithm_from(
            self.generation_algorithms.values(),
            algorithm,
            model,
            bd3lm_requested=bd3lm_requested,
        )


def bootstrap_builtin_hub(hub: RegistryHub) -> RegistryHub:
    """Populate ``hub`` with the builtin algorithms and integrations.

    Explicit and strict: bootstrapping the same hub twice raises rather than
    double-registering.  Lazy imports keep empty-hub construction free of
    heavy modules."""
    if hub._bootstrapped:
        raise ValueError(
            "hub is already bootstrapped; builtin bootstrap is not idempotent "
            "by design — create a fresh RegistryHub instead"
        )
    from unturtle.methods import populate_method_registry
    from unturtle.models.generation.sampler import populate_generation_registry
    from unturtle.models.integrations.registry import populate_integration_registry

    populate_generation_registry(hub)
    populate_integration_registry(hub)
    populate_method_registry(hub)
    hub._bootstrapped = True
    return hub


_default_hub: RegistryHub | None = None
_default_hub_lock = threading.Lock()


def ensure_default_hub() -> RegistryHub:
    """The process-default hub, bootstrapped exactly once (memoized).

    The module-level compatibility APIs delegate here; everything else
    should take a hub explicitly."""
    global _default_hub
    if _default_hub is None:
        with _default_hub_lock:
            if _default_hub is None:
                _default_hub = bootstrap_builtin_hub(RegistryHub())
    return _default_hub
