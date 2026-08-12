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

"""#153 Stage-1a: the ELF pack's import/discovery/default-hub isolation.

The pack (`packs/unturtle-elf`, editable-installed into the test env) is
the first REAL consumer of the #145 `unturtle.plugins` contract.  Pinned
before any model adaptation lands:

- importing `unturtle_elf` registers nothing anywhere;
- the REAL installed entry-point metadata (no search_path fixture) declares
  `elf` under `unturtle.plugins` with provenance `unturtle-elf 0.0.1`;
- loading targets the supplied hub only; two hubs are independent;
- default `auto` resolution for existing models is unchanged;
- the supports probe refuses every non-ELF model with an actionable
  message — code existence promotes nothing.
"""

import subprocess
import sys

import pytest

pytest.importorskip(
    "unturtle_elf",
    reason="ELF pack not installed (uv pip install -e packs/unturtle-elf)",
)

from unturtle.registry import RegistryHub, bootstrap_builtin_hub  # noqa: E402


def fresh_hub() -> RegistryHub:
    return bootstrap_builtin_hub(RegistryHub())


class TestImportAndDiscoveryIsolation:
    def test_import_alone_registers_nothing(self):
        code = (
            "import unturtle_elf  # import only\n"
            "from unturtle.models.generation import sampler\n"
            "assert sampler.find_algorithm('elf') is None\n"
            "from unturtle.registry import RegistryHub, bootstrap_builtin_hub\n"
            "hub = bootstrap_builtin_hub(RegistryHub())\n"
            "assert hub.generation_algorithms.find('elf') is None\n"
            "assert hub.methods.find('elf') is None\n"
            "print('ELF-SILENT')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert "ELF-SILENT" in result.stdout, result.stderr[-800:]

    def test_real_entry_point_metadata_declares_the_pack(self):
        """No search_path fixture: the REAL installed distribution is
        discovered — the first genuine consumer of the #145 surface."""
        from unturtle.plugins import discover_plugins

        refs = {ref.name: ref for ref in discover_plugins()}
        assert "elf" in refs
        assert refs["elf"].distribution == "unturtle-elf"
        assert refs["elf"].value == "unturtle_elf:register_unturtle"


class TestExplicitLoading:
    def test_load_targets_the_supplied_hub_with_provenance(self):
        from unturtle.models.generation import sampler
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        (loaded,) = load_plugins(hub, names=["elf"])
        assert hub.generation_algorithms.find("elf") is not None
        assert hub.methods.find("elf") is not None
        assert sampler.find_algorithm("elf") is None
        assert ("generation algorithm", "elf") in loaded.registered
        prov = hub.plugin_provenance[("generation algorithm", "elf")]
        assert prov.distribution == "unturtle-elf"

    def test_two_hubs_load_independently(self):
        from unturtle.plugins import load_plugins

        hub_a, hub_b = fresh_hub(), fresh_hub()
        load_plugins(hub_a, names=["elf"])
        load_plugins(hub_b, names=["elf"])
        assert hub_a.generation_algorithms.find("elf") is not None
        assert hub_b.generation_algorithms.find("elf") is not None

    def test_default_auto_resolution_is_unchanged(self):
        from unturtle.models.generation import sampler
        from unturtle.plugins import load_plugins

        class Masked:
            def _sample(self):
                pass

        before = sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False)
        load_plugins(fresh_hub(), names=["elf"])
        after = sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False)
        assert before == after == "mdlm"


class TestNoPromotion:
    def test_existing_models_fail_the_probe_actionably(self):
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, names=["elf"])

        class ExistingMaskedModel:
            def _sample(self):
                pass

        with pytest.raises(ValueError, match="load_elf_model"):
            hub.resolve_generation(
                "elf", ExistingMaskedModel(), bd3lm_requested=False
            )

    def test_elf_never_wins_auto_even_in_its_own_hub(self):
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, names=["elf"])

        class Masked:
            def _sample(self):
                pass

        assert (
            hub.resolve_generation("auto", Masked(), bd3lm_requested=False)
            == "mdlm"
        )

    def test_method_manifest_resolves_generation_only(self):
        """#153 is parity+generation only: the manifest must not claim a
        training/conversion component (that is #154's scope)."""
        from unturtle.methods import resolve_method
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, names=["elf"])
        resolved = resolve_method("elf", hub=hub)
        assert resolved.training is None
        assert resolved.conversion is None
        assert [algorithm.name for algorithm in resolved.generation] == ["elf"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
