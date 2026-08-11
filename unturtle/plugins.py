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

"""EXPERIMENTAL entry-point discovery for third-party method packs (#145).

A method pack is an installed Python distribution exposing entry points in
the ``unturtle.plugins`` group, each resolving to a registration callable of
the #144-proven shape::

    def register_unturtle(hub: RegistryHub) -> None: ...

Frozen decisions (each pinned in tests/test_plugin_discovery.py):

- **Discovery is explicit.**  ``import unturtle`` neither imports this
  module nor enumerates the group; only :func:`discover_plugins` /
  :func:`load_plugins` do, and only when called.
- **Enumeration is lazy.**  :func:`discover_plugins` reads packaging
  metadata only; plugin modules are imported by :func:`load_plugins` via
  ``EntryPoint.load()``, never during enumeration.
- **Loading is fail-fast** for the requested plugin set: the first broken
  plugin raises :class:`PluginError` naming the entry point, distribution,
  and cause; plugins ordered after it are not loaded.  There is no partial
  surface presented as complete and no silent skip.
- **Registration is non-transactional**, exactly as #142/#144 documented
  for direct registration: components a failing plugin registered before
  its failure — and plugins loaded before it — remain in the hub.  The
  caller owns hub disposal on failure (hubs are cheap instances).  No
  rollback machinery is promised.
- **Order is deterministic**: (distribution name, entry-point name),
  never filesystem order.
- **Provenance, not promotion.**  Each name a plugin registers is recorded
  in ``hub.plugin_provenance`` as (distribution, version, entry point) so
  conflicts and debugging can attribute providers.  It answers only "where
  did this registration come from?" — it is not capability or support
  status, which flow through the unchanged registry/validation machinery.
- **No bypass.**  Plugins register through the same hub API, duplicate
  checks, and capability probes as builtins; there is no plugin-only path.
- **No network, no installation.**  Only already-installed (or explicitly
  ``search_path``-supplied) distributions are considered.

This surface is experimental: no stable third-party ABI is promised until a
real method pack living outside core has used it.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

from unturtle.registry import DuplicateRegistrationError, Registry, RegistryHub

PLUGIN_GROUP = "unturtle.plugins"

__all__ = [
    "PLUGIN_GROUP",
    "LoadedPlugin",
    "PluginError",
    "PluginProvider",
    "PluginRef",
    "discover_plugins",
    "load_plugins",
]


class PluginError(RuntimeError):
    """A plugin could not be discovered, resolved, or registered.

    Always names the plugin (entry point + distribution + version) so the
    failure is attributable without reading tracebacks.

    Because loading is fail-fast AND non-transactional, the error also
    carries what survived (#150 review): ``loaded`` — the plugins that
    loaded successfully before the failure — and ``partial_registered`` —
    the (kind, name) entries the FAILING plugin registered before it raised
    (already recorded in ``hub.plugin_provenance``).  This makes the
    documented residue diagnosable; it does not promise rollback."""

    def __init__(
        self,
        message: str,
        *,
        loaded: tuple[LoadedPlugin, ...] = (),
        partial_registered: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__(message)
        self.loaded = loaded
        self.partial_registered = partial_registered


@dataclass(frozen=True)
class PluginRef:
    """One discovered entry point — metadata only, nothing imported."""

    name: str
    value: str
    group: str
    distribution: str
    version: str


@dataclass(frozen=True)
class PluginProvider:
    """Provenance of one registered name: which plugin put it in the hub."""

    distribution: str
    version: str
    entry_point: str


@dataclass(frozen=True)
class LoadedPlugin:
    """Load report for one plugin: its ref and every (registry kind, name)
    its registration call added to the hub — canonical names AND aliases,
    across the hub's registries including any the plugin attached itself.

    Known limitation (#150 review, pinned in tests): the diff tracks NAMES,
    not object identity.  A plugin that swaps the object behind an existing
    name via ``unregister`` + ``register`` (``unregister`` is a
    test/teardown helper, not plugin API) is invisible to this report and
    to provenance."""

    ref: PluginRef
    registered: tuple[tuple[str, str], ...]


def _provider_label(ref: PluginRef) -> str:
    return f"plugin {ref.name!r} from {ref.distribution} {ref.version}"


def _iter_entry_points(group: str, search_path: list[str] | None):
    if search_path is None:
        yield from importlib.metadata.entry_points(group=group)
        return
    for dist in importlib.metadata.distributions(path=list(search_path)):
        for entry_point in dist.entry_points:
            if entry_point.group == group:
                yield entry_point


def _dist_name(entry_point) -> tuple[str, str]:
    dist = entry_point.dist
    if dist is None:
        return "unknown-distribution", "unknown"
    # metadata["Name"] is None (not KeyError) for a malformed/incomplete
    # dist-info; without the fallback one such directory anywhere on the
    # path crashes discovery for every well-formed plugin (#150 review).
    return dist.metadata["Name"] or "unknown-distribution", dist.version or "unknown"


def _discover(group: str, search_path: list[str] | None) -> list[tuple[PluginRef, Any]]:
    pairs = []
    for entry_point in _iter_entry_points(group, search_path):
        distribution, version = _dist_name(entry_point)
        ref = PluginRef(
            name=entry_point.name,
            value=entry_point.value,
            group=group,
            distribution=distribution,
            version=version,
        )
        pairs.append((ref, entry_point))
    # Deterministic load/report order — never filesystem order.
    pairs.sort(key=lambda pair: (pair[0].distribution, pair[0].name))
    return pairs


def discover_plugins(
    *, group: str = PLUGIN_GROUP, search_path: list[str] | None = None
) -> tuple[PluginRef, ...]:
    """Enumerate installed method packs WITHOUT importing any of them.

    ``search_path`` restricts discovery to explicit directories (the test
    path); ``None`` means the interpreter's installed distributions."""
    return tuple(ref for ref, _ in _discover(group, search_path))


def _hub_registries(hub: RegistryHub) -> list[Registry[Any]]:
    return [value for value in vars(hub).values() if isinstance(value, Registry)]


def _known_name_snapshot(registries: list[Registry[Any]]) -> dict[int, set[str]]:
    # _known_names() covers canonical names AND aliases — an alias a plugin
    # claims is a name it owns for conflict attribution (#150 review).
    return {id(reg): set(reg._known_names()) for reg in registries}


def _diff_and_record(
    hub: RegistryHub, before: dict[int, set[str]], ref: PluginRef
) -> tuple[tuple[str, str], ...]:
    """Attribute every name the plugin added — re-scanning the hub so
    registries the plugin attached itself are included (#150 review)."""
    registered = []
    for reg in _hub_registries(hub):
        added = set(reg._known_names()) - before.get(id(reg), set())
        for name in added:
            registered.append((reg.kind, name))
            hub.plugin_provenance[(reg.kind, name)] = PluginProvider(
                distribution=ref.distribution,
                version=ref.version,
                entry_point=ref.name,
            )
    return tuple(sorted(registered))


def load_plugins(
    hub: RegistryHub,
    *,
    names: list[str] | None = None,
    group: str = PLUGIN_GROUP,
    search_path: list[str] | None = None,
) -> tuple[LoadedPlugin, ...]:
    """Explicitly load method packs into ``hub`` — and only into ``hub``.

    Each entry point is imported, resolved to its ``register_unturtle``-shaped
    callable, and called with the supplied hub.  See the module docstring for
    the frozen fail-fast / non-transactional / provenance semantics.

    Raises:
        PluginError: unknown requested name; import/resolution failure;
            non-callable entry point; registration failure — duplicate names
            are attributed to BOTH providers (the incoming plugin and, via
            ``hub.plugin_provenance``, the existing plugin, or "builtin or
            direct registration" when the name predates plugin loading).
    """
    pairs = _discover(group, search_path)
    if names is not None:
        known = {ref.name for ref, _ in pairs}
        missing = sorted(set(names) - known)
        if missing:
            raise PluginError(
                f"requested plugin(s) {missing} not found in group {group!r}; "
                f"discovered: {sorted(known)}"
            )
        pairs = [(ref, ep) for ref, ep in pairs if ref.name in names]

    loaded: list[LoadedPlugin] = []
    for ref, entry_point in pairs:
        try:
            register = entry_point.load()
        except Exception as exc:
            raise PluginError(
                f"{_provider_label(ref)} failed to import/resolve ({ref.value}): {exc}",
                loaded=tuple(loaded),
            ) from exc
        if not callable(register):
            raise PluginError(
                f"{_provider_label(ref)} resolved to a non-callable "
                f"({ref.value}); expected register_unturtle(hub)",
                loaded=tuple(loaded),
            )

        # Hold strong references across the call: prevents id() reuse if the
        # plugin replaces a registry attribute on the hub.
        registries = _hub_registries(hub)
        before = _known_name_snapshot(registries)
        try:
            register(hub)
        except DuplicateRegistrationError as exc:
            partial = _diff_and_record(hub, before, ref)
            existing = hub.plugin_provenance.get((exc.kind, exc.name))
            existing_label = (
                f"plugin {existing.entry_point!r} from "
                f"{existing.distribution} {existing.version}"
                if existing is not None
                else "builtin or direct registration"
            )
            raise PluginError(
                f"{_provider_label(ref)}: {exc.kind} {exc.name!r} is already "
                f"registered by {existing_label}",
                loaded=tuple(loaded),
                partial_registered=partial,
            ) from exc
        except Exception as exc:
            partial = _diff_and_record(hub, before, ref)
            raise PluginError(
                f"{_provider_label(ref)} failed during registration: {exc}",
                loaded=tuple(loaded),
                partial_registered=partial,
            ) from exc

        loaded.append(
            LoadedPlugin(ref=ref, registered=_diff_and_record(hub, before, ref))
        )
    return tuple(loaded)
