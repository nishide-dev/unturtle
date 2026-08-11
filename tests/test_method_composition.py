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

"""Method composition layer (#143): component registries + MethodSpec.

Core rule under test: **a MethodSpec describes a composition, it does not
execute one**.  Recipes are descriptions/resolution surfaces; existing
direct execution stays authoritative (behavior-zero-change differentials
below).  Research-only components must not become promoted capabilities by
being referenced from a recipe.
"""

import subprocess
import sys

import pytest

from unturtle.methods import (
    ComponentRecipe,
    MethodSpec,
    describe_method,
    list_methods,
    resolve_method,
    validate_method,
)
from unturtle.registry import RegistryHub, bootstrap_builtin_hub


def fresh_hub() -> RegistryHub:
    return bootstrap_builtin_hub(RegistryHub())


class TestSpecImmutability:
    def test_method_spec_is_frozen(self):
        spec = MethodSpec(name="x")
        with pytest.raises(AttributeError):
            spec.name = "y"

    def test_resolved_method_is_frozen(self):
        resolved = resolve_method("mdlm", hub=fresh_hub())
        with pytest.raises(AttributeError):
            resolved.spec = MethodSpec(name="hijack")


class TestBuiltinProofSet:
    def test_the_four_builtin_methods_are_registered(self):
        assert set(list_methods(hub=fresh_hub())) >= {
            "mdlm",
            "dfm",
            "flowlm",
            "prediff_hybrid",
        }

    def test_mdlm_resolves_masked_process_and_generation(self):
        resolved = resolve_method("mdlm", hub=fresh_hub())
        assert resolved.spec.process == "masked"
        assert resolved.process.name == "masked"
        assert [a.name for a in resolved.generation] == ["mdlm"]

    def test_dfm_resolves_discrete_flow_axes(self):
        resolved = resolve_method("dfm", hub=fresh_hub())
        assert resolved.process.name == "discrete_flow"
        assert resolved.training.name == "dfm"
        assert [a.name for a in resolved.generation] == ["dfm"]

    def test_flowlm_resolves_continuous_axes(self):
        resolved = resolve_method("flowlm", hub=fresh_hub())
        assert resolved.process.name == "continuous_flow"
        assert [a.name for a in resolved.generation] == ["flowlm"]

    def test_prediff_hybrid_exercises_the_conversion_axis(self):
        resolved = resolve_method("prediff_hybrid", hub=fresh_hub())
        assert resolved.conversion is not None
        assert resolved.conversion.name == "prediff_hybrid"
        # conversion code stays under models/conversion — the recipe only
        # references it (the factory is lazy; not called here).

    def test_process_factories_return_the_current_implementations(self):
        """Behavior-zero-change: the factory reaches the SAME class current
        callers import directly."""
        from unturtle.processes.discrete_flow import DiscreteFlowProcess
        from unturtle.processes.masked import MaskedDiffusionProcess

        hub = fresh_hub()
        assert hub.processes.get("masked").factory() is MaskedDiffusionProcess
        assert hub.processes.get("discrete_flow").factory() is DiscreteFlowProcess

    def test_training_factories_reach_the_existing_entry_points(self):
        from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
        from unturtle.diffusion.trainer import DiffusionTrainer

        hub = fresh_hub()
        assert hub.training_recipes.get("mdlm").factory() is DiffusionTrainer
        assert hub.training_recipes.get("dfm").factory() is discrete_flow_matching_loss


class TestValidation:
    def test_unknown_method_is_loud(self):
        with pytest.raises(KeyError, match="nope"):
            resolve_method("nope", hub=fresh_hub())

    def test_missing_component_reference_is_loud_and_names_the_axis(self):
        hub = fresh_hub()
        hub.method(MethodSpec(name="broken", process="no-such-process"))
        with pytest.raises(ValueError, match="process.*no-such-process"):
            resolve_method("broken", hub=hub)

    def test_missing_generation_reference_is_loud(self):
        hub = fresh_hub()
        hub.method(MethodSpec(name="broken-gen", generation=("no-such-algo",)))
        with pytest.raises(ValueError, match="no-such-algo"):
            resolve_method("broken-gen", hub=hub)

    def test_method_name_collision_is_loud(self):
        hub = fresh_hub()
        with pytest.raises(ValueError, match="mdlm"):
            hub.method(MethodSpec(name="mdlm"))

    def test_dfm_on_a_model_without_the_opt_in_fails_actionably(self):
        """The capability boundary (#65) survives recipe existence: the DFM
        recipe validates against the model's explicit opt-in, and a plain
        masked model fails BEFORE anything executes."""

        class PlainMasked:
            def _sample(self):
                pass

        with pytest.raises(ValueError, match="dfm"):
            validate_method("dfm", model=PlainMasked(), hub=fresh_hub())

    def test_dfm_recipe_existence_does_not_promote_the_capability(self):
        """`auto` selection and the opt-in default are untouched by the
        recipe being registered (no promotion by reference)."""
        from unturtle.models.generation import sampler

        fresh_hub()  # recipes registered here must not leak anywhere

        class PlainMasked:
            def _sample(self):
                pass

        assert (
            sampler.resolve_algorithm("auto", PlainMasked(), bd3lm_requested=False)
            == "mdlm"
        )

    def test_hybrid_model_with_block_decode_generation_fails_validation(self):
        """The frozen topology constraint (#127/#128) surfaces through
        validation: a hybrid model + a spec referencing block_decode is an
        unsupported combination, caught before execution."""

        class Cfg:
            hybrid_attention = True

        class HybridModel:
            config = Cfg()
            supports_block_decode = True

            def _sample(self):
                pass

            def _model_forward_with_cache(self):
                pass

        hub = fresh_hub()
        hub.method(MethodSpec(name="hybrid-bad", generation=("block_decode",)))
        with pytest.raises(ValueError, match="block.decode|block_decode"):
            validate_method("hybrid-bad", model=HybridModel(), hub=hub)

    def test_validation_passes_for_a_compatible_model(self):
        class PlainMasked:
            def _sample(self):
                pass

        resolved = validate_method("mdlm", model=PlainMasked(), hub=fresh_hub())
        assert resolved.spec.name == "mdlm"

    def test_required_capabilities_check_against_the_hub_integration(self):
        """Capability validation uses the SUPPLIED hub's integrations, never
        the default hub (the #143 amendment's whole point)."""
        from unturtle.models.integrations.registry import (
            BackboneIntegration,
            register_integration_into,
        )

        hub = fresh_hub()
        register_integration_into(
            hub,
            BackboneIntegration(
                name="toy-bb",
                model_types=("toy-type",),
                _native_resolver=lambda: None,
                capabilities=frozenset({"toy_generation"}),
            ),
        )
        hub.method(
            MethodSpec(
                name="needs-toy",
                required_capabilities=frozenset({"toy_generation"}),
            )
        )

        class Cfg:
            model_type = "toy-type"

        class ToyModel:
            config = Cfg()

        resolved = validate_method("needs-toy", model=ToyModel(), hub=hub)
        assert resolved.spec.name == "needs-toy"

        class OtherCfg:
            model_type = "llada"

        class WrongModel:
            config = OtherCfg()

        with pytest.raises(ValueError, match="toy_generation"):
            validate_method("needs-toy", model=WrongModel(), hub=hub)


class TestHonestUnverifiedRecording:
    def test_unresolvable_integration_records_capabilities_as_unverified(self):
        """The issue's honesty clause: when no integration is resolvable for
        the model, required capabilities are RECORDED as unverified — never
        silently treated as satisfied."""

        class NoTypeConfig:
            pass

        class UnknownModel:
            config = NoTypeConfig()

            def _sample(self):
                pass

        resolved = validate_method("mdlm", model=UnknownModel(), hub=fresh_hub())
        assert resolved.unverified_capabilities == frozenset({"masked_generation"})

    def test_resolvable_integration_leaves_nothing_unverified(self):
        hub = fresh_hub()

        class Cfg:
            model_type = "mdlm-dit"
            hybrid_attention = False

        class MdlmDit:
            config = Cfg()

            def _sample(self):
                pass

        resolved = validate_method("mdlm", model=MdlmDit(), hub=hub)
        assert resolved.unverified_capabilities == frozenset()


class TestIsolationAndDeterminism:
    def test_method_registration_does_not_leak_across_hubs(self):
        hub_a, hub_b = fresh_hub(), fresh_hub()
        hub_a.method(MethodSpec(name="private-method"))
        assert "private-method" in list_methods(hub=hub_a)
        assert "private-method" not in list_methods(hub=hub_b)

    def test_registration_order_does_not_change_resolution(self):
        """Two hubs with extra methods registered in opposite orders must
        resolve every method identically."""
        hub_a, hub_b = fresh_hub(), fresh_hub()
        one = MethodSpec(name="one", process="masked")
        two = MethodSpec(name="two", process="discrete_flow")
        hub_a.method(one), hub_a.method(two)
        hub_b.method(two), hub_b.method(one)
        for name in ("one", "two", "mdlm", "dfm"):
            a = resolve_method(name, hub=hub_a)
            b = resolve_method(name, hub=hub_b)
            assert a.spec == b.spec
            assert (a.process and a.process.name) == (b.process and b.process.name)


class TestIntrospection:
    def test_describe_is_serializable_without_calling_factories(self):
        import json

        description = describe_method("dfm", hub=fresh_hub())
        json.dumps(description)  # plain data, no factories/classes inside
        assert description["name"] == "dfm"
        assert description["process"]["name"] == "discrete_flow"

    def test_describe_does_not_import_heavy_modules(self):
        """Recipe introspection must work without loading backbones/trainers
        (subprocess so prior imports cannot mask it)."""
        # `import unturtle` itself eagerly loads the diffusion package, so
        # the honest property is the DELTA: describing every builtin recipe
        # must add no new modules at all (factories are never called).
        code = (
            "import sys\n"
            "from unturtle.methods import describe_method\n"
            "from unturtle.registry import RegistryHub, bootstrap_builtin_hub\n"
            "hub = bootstrap_builtin_hub(RegistryHub())\n"
            "before = set(sys.modules)\n"
            "for name in ('mdlm', 'dfm', 'flowlm', 'tiny_a2d', 'prediff_hybrid'):\n"
            "    describe_method(name, hub=hub)\n"
            "new = sorted(set(sys.modules) - before)\n"
            "assert not new, new\n"
            "print('LIGHT')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert "LIGHT" in result.stdout, result.stderr[-800:]

    def test_component_recipe_is_frozen(self):
        recipe = ComponentRecipe(
            name="x", kind="process", factory=lambda: None, summary="s"
        )
        with pytest.raises(AttributeError):
            recipe.name = "y"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
