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

"""#155 Stage-1a: FLM/FMLM pack import/discovery/default-hub isolation.

Mirrors the #153 ELF template.  Additional #155-specific pins:

- TWO methods register (`flm`, `fmlm`) with STRUCTURALLY distinct probes —
  an FLM denoiser does not satisfy the FMLM probe (an early tripwire for
  the "FMLM is just FLM steps=1" failure mode);
- the historical builtin `flowlm` is untouched by loading this pack.
"""

import subprocess
import sys

import pytest

pytest.importorskip(
    "unturtle_flm",
    reason="FLM pack not installed (uv pip install -e packs/unturtle-flm)",
)

from unturtle.registry import RegistryHub, bootstrap_builtin_hub  # noqa: E402


def fresh_hub() -> RegistryHub:
    return bootstrap_builtin_hub(RegistryHub())


class TestImportAndDiscoveryIsolation:
    def test_import_alone_registers_nothing(self):
        code = (
            "import unturtle_flm  # import only\n"
            "from unturtle.models.generation import sampler\n"
            "assert sampler.find_algorithm('flm') is None\n"
            "assert sampler.find_algorithm('fmlm') is None\n"
            "from unturtle.registry import RegistryHub, bootstrap_builtin_hub\n"
            "hub = bootstrap_builtin_hub(RegistryHub())\n"
            "assert hub.generation_algorithms.find('flm') is None\n"
            "assert hub.methods.find('fmlm') is None\n"
            "print('FLM-SILENT')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert "FLM-SILENT" in result.stdout, result.stderr[-800:]

    def test_real_entry_point_metadata_declares_the_pack(self):
        from unturtle.plugins import discover_plugins

        refs = {ref.name: ref for ref in discover_plugins()}
        assert "flm" in refs
        assert refs["flm"].distribution == "unturtle-flm"
        assert refs["flm"].value == "unturtle_flm:register_unturtle"


class TestExplicitLoading:
    def test_both_methods_load_into_the_supplied_hub_only(self):
        from unturtle.models.generation import sampler
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        (loaded,) = load_plugins(hub, names=["flm"])
        for name in ("flm", "fmlm"):
            assert hub.generation_algorithms.find(name) is not None
            assert hub.methods.find(name) is not None
            assert sampler.find_algorithm(name) is None
            assert ("generation algorithm", name) in loaded.registered
        prov = hub.plugin_provenance[("generation algorithm", "fmlm")]
        assert prov.distribution == "unturtle-flm"

    def test_the_historical_flowlm_builtin_is_untouched(self):
        """#155 naming rule: the pack must not rename/shadow Unturtle's
        `flowlm` prototype — distinct names, distinct entries."""
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        builtin_flowlm = hub.generation_algorithms.find("flowlm")
        assert builtin_flowlm is not None  # the prototype exists as builtin
        load_plugins(hub, names=["flm"])
        assert hub.generation_algorithms.find("flowlm") is builtin_flowlm
        assert hub.generation_algorithms.find("flm") is not builtin_flowlm

    def test_default_auto_resolution_is_unchanged(self):
        from unturtle.models.generation import sampler
        from unturtle.plugins import load_plugins

        class Masked:
            def _sample(self):
                pass

        before = sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False)
        load_plugins(fresh_hub(), names=["flm"])
        after = sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False)
        assert before == after == "mdlm"


class TestNoPromotionAndProbeSeparation:
    def test_existing_models_fail_both_probes_actionably(self):
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, names=["flm"])

        class ExistingMaskedModel:
            def _sample(self):
                pass

        with pytest.raises(ValueError, match="load_flm_model"):
            hub.resolve_generation("flm", ExistingMaskedModel(), bd3lm_requested=False)
        with pytest.raises(ValueError, match="load_fmlm_model"):
            hub.resolve_generation(
                "fmlm", ExistingMaskedModel(), bd3lm_requested=False
            )

    def test_an_flm_denoiser_does_not_satisfy_the_fmlm_probe(self):
        """THE #155 tripwire: the flow map is a different model contract
        (double time conditioning), so an FLM checkpoint must be refused by
        `fmlm` — loudly, naming the contract difference."""
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, names=["flm"])

        class FlmOnly:
            is_flm_denoiser = True

        assert (
            hub.resolve_generation("flm", FlmOnly(), bd3lm_requested=False) == "flm"
        )
        with pytest.raises(ValueError, match="flow map"):
            hub.resolve_generation("fmlm", FlmOnly(), bd3lm_requested=False)

    def test_neither_method_wins_auto_in_its_own_hub(self):
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, names=["flm"])

        class Masked:
            def _sample(self):
                pass

        assert hub.resolve_generation("auto", Masked(), bd3lm_requested=False) == "mdlm"

    def test_method_manifests_are_generation_only(self):
        from unturtle.methods import resolve_method
        from unturtle.plugins import load_plugins

        hub = fresh_hub()
        load_plugins(hub, names=["flm"])
        for name in ("flm", "fmlm"):
            resolved = resolve_method(name, hub=hub)
            assert resolved.training is None
            assert resolved.conversion is None
            assert [algorithm.name for algorithm in resolved.generation] == [name]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
