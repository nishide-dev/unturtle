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

"""#145 entry-point method packs: discovery + provenance.

Frozen decisions pinned here (see unturtle/plugins.py for the docstring
versions):

- entry-point group is ``unturtle.plugins``;
- discovery/loading is EXPLICIT — ``import unturtle`` enumerates nothing;
- enumeration never imports plugin modules (``EntryPoint.load`` does);
- loading is FAIL-FAST for the requested plugin set: the first broken
  plugin raises, later plugins in the set are not loaded, and — per the
  #144-documented non-transactional semantics — everything registered
  before the failure remains in the hub, whose disposal the caller owns;
- load order is deterministic: sorted by (distribution name, entry-point
  name), never filesystem order;
- provenance (distribution name, version, entry-point name) is recorded on
  the hub at the load boundary and answers only "where did this
  registration come from?" — it is not capability promotion.

All fixtures are local dist-info directories driven through the REAL
``importlib.metadata`` machinery (no mocks, no network, nothing installed).
"""

import pathlib
import subprocess
import sys
import textwrap

import pytest

from tests.fixtures import toy_extension_plugin
from unturtle.models.generation.sampler import GenerationRequest
from unturtle.registry import RegistryHub, bootstrap_builtin_hub

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def fresh_hub() -> RegistryHub:
    return bootstrap_builtin_hub(RegistryHub())


def make_dist(
    root: pathlib.Path,
    dist_name: str,
    version: str,
    entry_points: dict[str, dict[str, str]],
    modules: dict[str, str] | None = None,
) -> pathlib.Path:
    """A synthetic installed distribution: dist-info metadata plus (optional)
    plugin module sources, exactly the layout pip leaves on disk."""
    di = root / f"{dist_name.replace('-', '_')}-{version}.dist-info"
    di.mkdir(parents=True)
    (di / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n"
    )
    sections = []
    for group, eps in entry_points.items():
        lines = "\n".join(f"{n} = {target}" for n, target in eps.items())
        sections.append(f"[{group}]\n{lines}\n")
    (di / "entry_points.txt").write_text("\n".join(sections))
    for module_name, source in (modules or {}).items():
        (root / f"{module_name}.py").write_text(textwrap.dedent(source))
    return root


@pytest.fixture
def toy_pack_path(tmp_path, monkeypatch):
    """A dist whose entry point IS the #144 method pack — the runtime-proof
    fixture reused verbatim, now arriving via packaging metadata."""
    path = make_dist(
        tmp_path / "site",
        "toy-pack",
        "1.2.3",
        {
            "unturtle.plugins": {
                "toy_echo": "tests.fixtures.toy_extension_plugin:register_unturtle"
            },
            "console_scripts": {"unrelated": "nonexistent_module:main"},
        },
    )
    monkeypatch.syspath_prepend(str(path))
    return path


class CompatibleModel:
    supports_toy_echo = True

    class _Cfg:
        model_type = "mdlm-dit"
        hybrid_attention = False

    config = _Cfg()

    def _sample(self):
        pass


class TestNoImplicitDiscovery:
    def test_import_unturtle_performs_no_discovery(self):
        """`import unturtle` must neither import unturtle.plugins nor
        enumerate the `unturtle.plugins` entry-point group — checked in a
        subprocess with a spy wrapped around importlib.metadata BEFORE the
        import.  Dependencies (unsloth/transformers) legitimately call
        entry_points() for their OWN groups during import; only our group
        is pinned."""
        code = (
            "import importlib.metadata as md\n"
            "calls = []\n"
            "orig_eps = md.entry_points\n"
            "def spy_eps(*a, **k):\n"
            "    if k.get('group') == 'unturtle.plugins':\n"
            "        calls.append(('eps', k))\n"
            "    return orig_eps(*a, **k)\n"
            "md.entry_points = spy_eps\n"
            "import unturtle\n"
            "import sys\n"
            "assert 'unturtle.plugins' not in sys.modules, 'plugins module imported eagerly'\n"
            "assert not calls, f'unturtle.plugins group enumerated at import: {calls}'\n"
            "print('NO-DISCOVERY')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert "NO-DISCOVERY" in result.stdout, result.stderr[-800:]


class TestDiscovery:
    def test_explicit_discovery_finds_the_synthetic_entry_point(self, toy_pack_path):
        from unturtle.plugins import discover_plugins

        refs = discover_plugins(search_path=[str(toy_pack_path)])
        assert len(refs) == 1
        ref = refs[0]
        assert ref.name == "toy_echo"
        assert ref.distribution == "toy-pack"
        assert ref.version == "1.2.3"
        assert ref.value == "tests.fixtures.toy_extension_plugin:register_unturtle"

    def test_unrelated_entry_point_groups_are_ignored(self, toy_pack_path):
        from unturtle.plugins import discover_plugins

        refs = discover_plugins(search_path=[str(toy_pack_path)])
        assert all(r.group == "unturtle.plugins" for r in refs)
        assert "unrelated" not in [r.name for r in refs]

    def test_enumeration_does_not_import_plugin_modules(self, tmp_path, monkeypatch):
        from unturtle.plugins import discover_plugins

        sentinel = tmp_path / "site" / "lazy_probe_mod.imported"
        path = make_dist(
            tmp_path / "site",
            "lazy-probe",
            "0.1",
            {"unturtle.plugins": {"probe": "lazy_probe_mod:register_unturtle"}},
            modules={
                "lazy_probe_mod": """\
                import pathlib

                pathlib.Path(__file__).with_suffix(".imported").touch()


                def register_unturtle(hub):
                    pass
                """
            },
        )
        monkeypatch.syspath_prepend(str(path))
        refs = discover_plugins(search_path=[str(path)])
        assert [r.name for r in refs] == ["probe"]
        assert not sentinel.exists(), "enumeration imported the plugin module"


class TestExplicitLoading:
    def test_load_targets_the_supplied_hub_only(self, toy_pack_path):
        from unturtle.models.generation import sampler
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        loaded = load_plugins(hub, search_path=[str(toy_pack_path)])
        assert [p.ref.name for p in loaded] == ["toy_echo"]
        assert hub.generation_algorithms.find("toy_echo") is not None
        assert hub.methods.find("toy_echo") is not None
        assert sampler.find_algorithm("toy_echo") is None

        class Masked:
            def _sample(self):
                pass

        assert (
            sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False) == "mdlm"
        )

    def test_same_plugin_loads_into_two_independent_hubs(self, toy_pack_path):
        from unturtle.plugins import load_plugins

        hub_a, hub_b = fresh_hub(), fresh_hub()
        load_plugins(hub_a, search_path=[str(toy_pack_path)])
        load_plugins(hub_b, search_path=[str(toy_pack_path)])
        request = GenerationRequest(inputs=[5, 6], generation_config=None, kwargs={})
        for hub in (hub_a, hub_b):
            assert hub.dispatch_generation(
                CompatibleModel(), request, algorithm="toy_echo"
            ) == {"method": "toy_echo", "tokens": [6, 5]}

    def test_entry_point_pack_passes_the_144_runtime_proof(self, toy_pack_path):
        """The #144-style method pack, arriving via packaging metadata, is
        describable / resolvable / validatable / dispatchable through the
        SAME surfaces as internal extensions."""
        from unturtle.methods import describe_method, validate_method
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, search_path=[str(toy_pack_path)])

        description = describe_method("toy_echo", hub=hub)
        assert description["process"]["name"] == "masked"
        assert description["generation"] == ["toy_echo"]

        validated = validate_method("toy_echo", model=CompatibleModel(), hub=hub)
        assert validated.unverified_capabilities == frozenset()

        request = GenerationRequest(inputs=[1, 2, 3], generation_config=None, kwargs={})
        result = hub.dispatch_generation(
            CompatibleModel(), request, algorithm="toy_echo"
        )
        assert result == {"method": "toy_echo", "tokens": [3, 2, 1]}

    def test_capability_validation_is_not_bypassed(self, toy_pack_path):
        """A plugin cannot promote support: the entry-point pack's method
        still fails validation for a model that does not opt in, through the
        same capability machinery, with the pack's own message."""
        from unturtle.methods import validate_method
        from unturtle.plugins import load_plugins

        class NoOptIn:
            class _Cfg:
                model_type = "mdlm-dit"

            config = _Cfg()

            def _sample(self):
                pass

        hub = fresh_hub()
        load_plugins(hub, search_path=[str(toy_pack_path)])
        with pytest.raises(ValueError, match="set supports_toy_echo = True"):
            validate_method("toy_echo", model=NoOptIn(), hub=hub)

    def test_requesting_an_unknown_plugin_name_is_loud(self, toy_pack_path):
        from unturtle.plugins import PluginError, load_plugins

        hub = fresh_hub()
        with pytest.raises(PluginError, match="no_such_plugin"):
            load_plugins(
                hub, names=["no_such_plugin"], search_path=[str(toy_pack_path)]
            )


class TestProvenance:
    def test_provenance_is_recorded_per_registered_name(self, toy_pack_path):
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, search_path=[str(toy_pack_path)])
        prov = hub.plugin_provenance[("generation algorithm", "toy_echo")]
        assert prov.distribution == "toy-pack"
        assert prov.version == "1.2.3"
        assert prov.entry_point == "toy_echo"
        assert ("method", "toy_echo") in hub.plugin_provenance

    def test_builtins_carry_no_plugin_provenance(self):
        hub = fresh_hub()
        assert ("generation algorithm", "mdlm") not in hub.plugin_provenance

    def test_loaded_plugin_report_lists_what_was_registered(self, toy_pack_path):
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        (loaded,) = load_plugins(hub, search_path=[str(toy_pack_path)])
        assert ("generation algorithm", "toy_echo") in loaded.registered
        assert ("method", "toy_echo") in loaded.registered


class TestConflicts:
    def _builtin_clash_dist(self, tmp_path, monkeypatch):
        path = make_dist(
            tmp_path / "clash",
            "mdlm-imposter",
            "9.9",
            {"unturtle.plugins": {"imposter": "mdlm_imposter_mod:register_unturtle"}},
            modules={
                "mdlm_imposter_mod": """\
                def register_unturtle(hub):
                    @hub.generation(
                        name="mdlm",
                        family="masked_discrete",
                        supports=lambda model: True,
                        auto_priority=99,
                        unsupported_message=lambda model: "never",
                    )
                    def run(model, request):
                        pass
                """
            },
        )
        monkeypatch.syspath_prepend(str(path))
        return path

    def test_builtin_conflict_names_kind_key_and_both_providers(
        self, tmp_path, monkeypatch
    ):
        from unturtle.plugins import PluginError, load_plugins

        path = self._builtin_clash_dist(tmp_path, monkeypatch)
        hub = fresh_hub()
        with pytest.raises(PluginError) as excinfo:
            load_plugins(hub, search_path=[str(path)])
        message = str(excinfo.value)
        assert "generation algorithm" in message  # registry kind
        assert "'mdlm'" in message  # conflicting key
        assert "mdlm-imposter 9.9" in message  # incoming provider + version
        assert "imposter" in message  # incoming entry point
        assert "builtin or direct registration" in message  # existing provider

    def test_plugin_plugin_conflict_attributes_the_earlier_plugin(
        self, tmp_path, monkeypatch
    ):
        """Two distributions claim the same canonical name: the second load
        must fail naming the FIRST plugin's distribution as the existing
        provider — no last-writer-wins."""
        from unturtle.plugins import PluginError, load_plugins

        def echo_dist(dirname, dist, module):
            return make_dist(
                tmp_path / dirname,
                dist,
                "1.0",
                {"unturtle.plugins": {"claimer": f"{module}:register_unturtle"}},
                modules={
                    module: """\
                    def register_unturtle(hub):
                        @hub.generation(
                            name="contested",
                            family="masked_discrete",
                            supports=lambda model: True,
                            auto_priority=95,
                            unsupported_message=lambda model: "never",
                        )
                        def run(model, request):
                            pass
                    """
                },
            )

        path_a = echo_dist("a", "pack-alpha", "pack_alpha_mod")
        path_b = echo_dist("b", "pack-beta", "pack_beta_mod")
        monkeypatch.syspath_prepend(str(path_a))
        monkeypatch.syspath_prepend(str(path_b))

        hub = fresh_hub()
        load_plugins(hub, search_path=[str(path_a)])
        with pytest.raises(PluginError) as excinfo:
            load_plugins(hub, search_path=[str(path_b)])
        message = str(excinfo.value)
        assert "pack-beta 1.0" in message  # incoming provider
        assert "pack-alpha 1.0" in message  # existing provider, attributed
        assert "'contested'" in message

    def test_same_call_conflict_order_is_deterministic(self, tmp_path, monkeypatch):
        """Both claimers in ONE search path: load order is sorted by
        (distribution, entry point), so pack-alpha wins deterministically and
        pack-beta is the attributed loser — never filesystem order."""
        from unturtle.plugins import PluginError, load_plugins

        site = tmp_path / "site"
        # Module names unique to THIS test: ep.load() caches modules in
        # sys.modules, so reusing another test's module name would silently
        # import that test's cached module instead of the files below.
        for dist, module in (
            ("det-beta", "det_beta_mod"),  # created FIRST on disk
            ("det-alpha", "det_alpha_mod"),
        ):
            make_dist(
                site,
                dist,
                "1.0",
                {"unturtle.plugins": {"claimer": f"{module}:register_unturtle"}},
                modules={
                    module: f"""\
                    def register_unturtle(hub):
                        @hub.generation(
                            name="contested",
                            family="masked_discrete",
                            supports=lambda model: True,
                            auto_priority=95,
                            unsupported_message=lambda model: "never",
                        )
                        def run_{module}(model, request):
                            return "{dist}"
                    """
                },
            )
        monkeypatch.syspath_prepend(str(site))

        hub = fresh_hub()
        with pytest.raises(PluginError) as excinfo:
            load_plugins(hub, search_path=[str(site)])
        message = str(excinfo.value)
        assert "det-beta 1.0" in message  # sorted-second → the loser
        assert "det-alpha 1.0" in message  # sorted-first → existing provider
        entry = hub.generation_algorithms.get("contested")
        assert entry.runner(None, None) == "det-alpha"


class TestFailureSemantics:
    def test_broken_plugin_is_fail_fast_and_actionable(self, tmp_path, monkeypatch):
        """Frozen policy: fail-fast for the requested set. The error names
        the entry point, distribution, and the underlying cause; plugins
        sorted after the broken one are NOT loaded."""
        from unturtle.plugins import PluginError, load_plugins

        site = tmp_path / "site"
        make_dist(
            site,
            "broken-pack",
            "0.5",
            {"unturtle.plugins": {"broken": "broken_pack_mod:register_unturtle"}},
            modules={
                "broken_pack_mod": """\
                raise ImportError("this plugin is deliberately broken")
                """
            },
        )
        make_dist(
            site,
            "zz-later-pack",
            "0.5",
            {"unturtle.plugins": {"later": "zz_later_pack_mod:register_unturtle"}},
            modules={
                "zz_later_pack_mod": """\
                def register_unturtle(hub):
                    hub.method  # would touch the hub if loaded
                    raise AssertionError("must not be reached under fail-fast")
                """
            },
        )
        monkeypatch.syspath_prepend(str(site))

        hub = fresh_hub()
        with pytest.raises(PluginError) as excinfo:
            load_plugins(hub, search_path=[str(site)])
        message = str(excinfo.value)
        assert "broken-pack 0.5" in message
        assert "broken" in message
        assert "deliberately broken" in str(excinfo.value.__cause__)

    def test_partial_registration_semantics_match_the_144_pin(
        self, tmp_path, monkeypatch, toy_pack_path
    ):
        """Non-transactional, documented: a plugin loaded BEFORE the broken
        one stays registered (caller owns hub disposal). No rollback
        machinery is promised or implemented."""
        from unturtle.plugins import PluginError, load_plugins

        site = tmp_path / "later"
        make_dist(
            site,
            "zz-broken",
            "0.1",
            {"unturtle.plugins": {"boom": "zz_boom_mod:register_unturtle"}},
            modules={
                "zz_boom_mod": """\
                def register_unturtle(hub):
                    raise RuntimeError("boom after toy pack loaded")
                """
            },
        )
        monkeypatch.syspath_prepend(str(site))

        hub = fresh_hub()
        with pytest.raises(PluginError):
            load_plugins(
                hub, search_path=[str(toy_pack_path), str(site)]
            )  # toy-pack sorts before zz-broken
        assert hub.generation_algorithms.find("toy_echo") is not None


class TestAutoSelectionDiscipline:
    def test_plugin_does_not_change_auto_unless_it_wins_normally(self, toy_pack_path):
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, search_path=[str(toy_pack_path)])

        class MaskedOptIn:
            supports_toy_echo = True

            def _sample(self):
                pass

        assert (
            hub.resolve_generation("auto", MaskedOptIn(), bd3lm_requested=False)
            == "mdlm"
        )

    def test_plugin_can_win_auto_only_under_the_normal_rules(
        self, tmp_path, monkeypatch
    ):
        """No side channel in either direction: a plugin that registers an
        eligible, supported algorithm with a winning priority wins auto
        through the ordinary resolution loop."""
        from unturtle.plugins import load_plugins

        path = make_dist(
            tmp_path / "site",
            "fast-pack",
            "2.0",
            {"unturtle.plugins": {"fast": "fast_pack_mod:register_unturtle"}},
            modules={
                "fast_pack_mod": """\
                def register_unturtle(hub):
                    @hub.generation(
                        name="fast_toy",
                        family="masked_discrete",
                        supports=lambda model: hasattr(model, "_sample"),
                        auto_priority=5,
                        unsupported_message=lambda model: "no _sample",
                    )
                    def run_fast(model, request):
                        return "fast"
                """
            },
        )
        monkeypatch.syspath_prepend(str(path))

        hub = fresh_hub()
        load_plugins(hub, search_path=[str(path)])

        class Masked:
            def _sample(self):
                pass

        assert (
            hub.resolve_generation("auto", Masked(), bd3lm_requested=False)
            == "fast_toy"
        )

        from unturtle.models.generation import sampler

        assert sampler.find_algorithm("fast_toy") is None


class TestReviewPins150:
    """Pins for the #150 review findings, RED-first."""

    def test_nameless_dist_info_does_not_poison_discovery(self, tmp_path, monkeypatch):
        """Review F1 (CRITICAL): a dist-info whose METADATA lacks `Name`
        (interrupted pip install, malformed wheel) must not crash discovery
        for every well-formed plugin on the path with an unattributable
        TypeError.  The malformed dist is surfaced under a stable fallback
        name instead."""
        from unturtle.plugins import discover_plugins, load_plugins

        site = tmp_path / "site"
        make_dist(
            site,
            "real-pack",
            "2.0",
            {"unturtle.plugins": {"real": "real_pack_mod:register_unturtle"}},
            modules={
                "real_pack_mod": """\
                def register_unturtle(hub):
                    pass
                """
            },
        )
        nameless = site / "nameless-0.0.dist-info"
        nameless.mkdir()
        (nameless / "METADATA").write_text("Metadata-Version: 2.1\n")  # no Name
        (nameless / "entry_points.txt").write_text(
            "[unturtle.plugins]\nghost = ghost_mod:register_unturtle\n"
        )
        monkeypatch.syspath_prepend(str(site))

        refs = discover_plugins(search_path=[str(site)])
        by_name = {r.name: r for r in refs}
        assert by_name["real"].distribution == "real-pack"
        assert by_name["ghost"].distribution == "unknown-distribution"

        hub = fresh_hub()
        loaded = load_plugins(hub, names=["real"], search_path=[str(site)])
        assert [p.ref.name for p in loaded] == ["real"]

    def test_alias_conflict_is_attributed_to_the_owning_plugin(
        self, tmp_path, monkeypatch
    ):
        """Review F2 (CRITICAL): plugin A registers a component with an
        ALIAS; plugin B later claims that alias as a canonical name.  The
        conflict must attribute plugin A — not exonerate it as 'builtin or
        direct registration' — and the alias must be present in provenance
        and in A's load report."""
        from unturtle.plugins import PluginError, load_plugins

        site_a = make_dist(
            tmp_path / "a",
            "alias-pack",
            "1.0",
            {"unturtle.plugins": {"aliased": "alias_pack_mod:register_unturtle"}},
            modules={
                "alias_pack_mod": """\
                def register_unturtle(hub):
                    from unturtle.methods import ComponentRecipe

                    hub.processes.register(
                        ComponentRecipe(
                            name="real_name",
                            kind="process",
                            factory=lambda: None,
                            summary="",
                        ),
                        aliases=("nick",),
                    )
                """
            },
        )
        site_b = make_dist(
            tmp_path / "b",
            "clash-pack",
            "1.0",
            {"unturtle.plugins": {"clasher": "clash_pack_mod:register_unturtle"}},
            modules={
                "clash_pack_mod": """\
                def register_unturtle(hub):
                    hub.process(name="nick", factory=lambda: None)
                """
            },
        )
        monkeypatch.syspath_prepend(str(site_a))
        monkeypatch.syspath_prepend(str(site_b))

        hub = fresh_hub()
        (loaded,) = load_plugins(hub, search_path=[str(site_a)])
        assert ("process", "nick") in loaded.registered
        prov = hub.plugin_provenance[("process", "nick")]
        assert prov.distribution == "alias-pack"

        with pytest.raises(PluginError) as excinfo:
            load_plugins(hub, search_path=[str(site_b)])
        message = str(excinfo.value)
        assert "'nick'" in message
        assert "clash-pack 1.0" in message  # incoming
        assert "alias-pack 1.0" in message  # existing provider, not exonerated
        assert "builtin or direct registration" not in message

    def test_plugin_added_registries_are_reported_and_attributed(
        self, tmp_path, monkeypatch
    ):
        """Review F4: a plugin that attaches a NEW registry axis to the hub
        and registers into it must appear in the load report and in
        provenance — silent under-reporting is the wrong failure direction."""
        from unturtle.plugins import load_plugins

        path = make_dist(
            tmp_path / "site",
            "axis-pack",
            "3.0",
            {"unturtle.plugins": {"axis": "axis_pack_mod:register_unturtle"}},
            modules={
                "axis_pack_mod": """\
                def register_unturtle(hub):
                    from unturtle.registry import Registry

                    class OdeSolver:
                        name = "odesolve"

                    hub.solvers = Registry("solver")
                    hub.solvers.register(OdeSolver())
                """
            },
        )
        monkeypatch.syspath_prepend(str(path))

        hub = fresh_hub()
        (loaded,) = load_plugins(hub, search_path=[str(path)])
        assert ("solver", "odesolve") in loaded.registered
        assert hub.plugin_provenance[("solver", "odesolve")].distribution == "axis-pack"

    def test_failure_carries_the_partial_load_report(
        self, tmp_path, monkeypatch, toy_pack_path
    ):
        """Review F5: fail-fast discards nothing the caller needs — the
        PluginError carries the plugins loaded before the failure AND what
        the failing plugin registered before it raised (with provenance),
        so the documented non-transactional residue is diagnosable without
        promising rollback."""
        from unturtle.plugins import PluginError, load_plugins

        site = tmp_path / "later"
        make_dist(
            site,
            "zz-half-pack",
            "0.2",
            {"unturtle.plugins": {"half": "zz_half_pack_mod:register_unturtle"}},
            modules={
                "zz_half_pack_mod": """\
                def register_unturtle(hub):
                    @hub.generation(
                        name="half_done",
                        family="masked_discrete",
                        supports=lambda model: True,
                        auto_priority=93,
                        unsupported_message=lambda model: "never",
                    )
                    def run_half(model, request):
                        pass

                    raise RuntimeError("late boom")
                """
            },
        )
        monkeypatch.syspath_prepend(str(site))

        hub = fresh_hub()
        with pytest.raises(PluginError) as excinfo:
            load_plugins(hub, search_path=[str(toy_pack_path), str(site)])
        error = excinfo.value
        assert [p.ref.name for p in error.loaded] == ["toy_echo"]
        assert ("generation algorithm", "half_done") in error.partial_registered
        # The residue is attributable: a later conflict names zz-half-pack.
        prov = hub.plugin_provenance[("generation algorithm", "half_done")]
        assert prov.distribution == "zz-half-pack"

    def test_unregister_then_register_hijack_is_a_documented_limitation(
        self, tmp_path, monkeypatch
    ):
        """Review F3, pinned as a DOCUMENTED LIMITATION (not silently
        assumed away): `Registry.unregister` is a test/teardown helper, not
        plugin API.  A plugin that swaps an object behind an existing name
        leaves no provenance and an empty report — the loader tracks names,
        not object identity.  If this pin breaks because identity tracking
        was added, update the module docstring's limitation note too."""
        from unturtle.plugins import load_plugins

        path = make_dist(
            tmp_path / "site",
            "hijack-pack",
            "6.6",
            {"unturtle.plugins": {"hijack": "hijack_pack_mod:register_unturtle"}},
            modules={
                "hijack_pack_mod": """\
                def register_unturtle(hub):
                    from unturtle.methods import ComponentRecipe

                    victim = hub.processes.get("masked")
                    hub.processes.unregister(victim)
                    hub.processes.register(
                        ComponentRecipe(
                            name="masked",
                            kind="process",
                            factory=lambda: None,
                            summary="imposter",
                        )
                    )
                """
            },
        )
        monkeypatch.syspath_prepend(str(path))

        hub = fresh_hub()
        (loaded,) = load_plugins(hub, search_path=[str(path)])
        assert hub.processes.get("masked").summary == "imposter"  # swap happened
        assert loaded.registered == ()  # and the loader cannot see it
        assert ("process", "masked") not in hub.plugin_provenance


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
