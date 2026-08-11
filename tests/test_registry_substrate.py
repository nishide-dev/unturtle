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

Deliberately boring container semantics: deterministic insertion order,
duplicate rejection (canonical names AND aliases), immutable iteration
snapshots, explicit bootstrap.  Instance ownership is the point — two hubs
must never share state, and importing modules must never mutate a hub.
"""

import subprocess
import sys
from dataclasses import dataclass

import pytest

from unturtle.registry import Registry, RegistryHub, bootstrap_builtin_hub


@dataclass(frozen=True)
class Thing:
    name: str
    payload: int = 0


class TestRegistryContainer:
    def test_register_get_find_roundtrip(self):
        reg = Registry("thing")
        thing = Thing("a")
        assert reg.register(thing) is thing  # returns the original object
        assert reg.get("a") is thing
        assert reg.find("a") is thing
        assert reg.find("missing") is None

    def test_get_unknown_raises_with_kind_and_known_names(self):
        reg = Registry("thing")
        reg.register(Thing("a"))
        with pytest.raises(KeyError, match="thing.*'b'.*a"):
            reg.get("b")

    def test_duplicate_canonical_names_fail(self):
        reg = Registry("thing")
        reg.register(Thing("a"))
        with pytest.raises(ValueError, match="'a'"):
            reg.register(Thing("a", payload=1))

    def test_duplicate_aliases_fail_in_both_directions(self):
        reg = Registry("thing")
        reg.register(Thing("a"), aliases=("alpha",))
        with pytest.raises(ValueError, match="alpha"):
            reg.register(Thing("b"), aliases=("alpha",))
        with pytest.raises(ValueError, match="'a'"):
            reg.register(Thing("c"), aliases=("a",))
        # and a new canonical name colliding with an existing alias
        with pytest.raises(ValueError, match="alpha"):
            reg.register(Thing("alpha"))

    def test_aliases_resolve_to_the_same_object(self):
        reg = Registry("thing")
        thing = reg.register(Thing("a"), aliases=("alpha", "A"))
        assert reg.get("alpha") is thing
        assert reg.get("A") is thing

    def test_values_snapshot_is_immutable_and_ordered(self):
        reg = Registry("thing")
        one, two = Thing("one"), Thing("two")
        reg.register(one)
        reg.register(two)
        snapshot = reg.values()
        assert isinstance(snapshot, tuple)
        assert snapshot == (one, two)
        reg.register(Thing("three"))
        assert snapshot == (one, two), "snapshot must not follow later mutation"

    def test_unregister_is_identity_based(self):
        reg = Registry("thing")
        first = reg.register(Thing("x"))
        equal_but_different = Thing("x")  # value-equal, different identity
        reg.unregister(equal_but_different)
        assert reg.find("x") is first, "value-equal impostor must not remove"
        reg.unregister(first)
        assert reg.find("x") is None

    def test_unregister_clears_aliases(self):
        reg = Registry("thing")
        thing = reg.register(Thing("a"), aliases=("alpha",))
        reg.unregister(thing)
        assert reg.find("alpha") is None
        reg.register(Thing("b"), aliases=("alpha",))  # alias reusable again


class TestHubIsolation:
    def test_two_hubs_do_not_share_registrations(self):
        hub_a, hub_b = RegistryHub(), RegistryHub()
        hub_a.generation_algorithms.register(Thing("only-in-a"))
        assert hub_b.generation_algorithms.find("only-in-a") is None
        assert hub_a.generation_algorithms.find("only-in-a") is not None

    def test_empty_hub_has_empty_registries(self):
        hub = RegistryHub()
        assert hub.generation_algorithms.values() == ()
        assert hub.backbone_integrations.values() == ()


class TestBuiltinBootstrap:
    def test_bootstrap_is_deterministic(self):
        first = bootstrap_builtin_hub(RegistryHub())
        second = bootstrap_builtin_hub(RegistryHub())
        assert [a.name for a in first.generation_algorithms.values()] == [
            a.name for a in second.generation_algorithms.values()
        ]
        assert [i.name for i in first.backbone_integrations.values()] == [
            i.name for i in second.backbone_integrations.values()
        ]

    def test_double_bootstrap_of_the_same_hub_is_loud(self):
        hub = bootstrap_builtin_hub(RegistryHub())
        with pytest.raises(ValueError, match="bootstrap"):
            bootstrap_builtin_hub(hub)

    def test_fresh_hub_matches_the_module_level_registries(self):
        """The issue's differential contract: a fresh isolated hub populated
        by builtin bootstrap equals the current global path."""
        from unturtle.models.generation import sampler
        from unturtle.models.integrations import registry as integrations

        hub = bootstrap_builtin_hub(RegistryHub())
        assert [
            (a.name, a.family, a.auto_priority, a.auto_eligible, dict(a.flags))
            for a in hub.generation_algorithms.values()
        ] == [
            (a.name, a.family, a.auto_priority, a.auto_eligible, dict(a.flags))
            for a in sampler.iter_algorithms()
        ]
        assert [
            (i.name, tuple(i.model_types), tuple(i.peft_model_types))
            for i in hub.backbone_integrations.values()
        ] == [
            (i.name, tuple(i.model_types), tuple(i.peft_model_types))
            for i in integrations.iter_integrations()
        ]

    def test_fresh_hub_resolves_like_the_global_path(self):
        """Representative auto-resolutions through an ISOLATED hub."""
        from unturtle.models.generation.sampler import resolve_algorithm

        hub = bootstrap_builtin_hub(RegistryHub())

        class Masked:
            def _sample(self):
                pass

        class Cached(Masked):
            supports_block_decode = True

            def _model_forward_with_cache(self):
                pass

        for model, expected in ((Masked(), "mdlm"), (Cached(), "block_decode")):
            assert (
                hub.resolve_generation("auto", model, bd3lm_requested=False)
                == resolve_algorithm("auto", model, bd3lm_requested=False)
                == expected
            )

    def test_registrations_into_a_fresh_hub_do_not_leak_to_the_default(self):
        from unturtle.models.generation import sampler

        hub = bootstrap_builtin_hub(RegistryHub())
        hub.generation_algorithms.register(Thing("private-algo"))
        assert sampler.find_algorithm("private-algo") is None

    def test_importing_backbone_modules_does_not_mutate_a_hub(self):
        """Constraint 6: no arbitrary-module self-registration.  Run in a
        subprocess so prior test imports cannot mask the check."""
        code = (
            "from unturtle.registry import RegistryHub\n"
            "hub = RegistryHub()\n"
            "import unturtle.models.backbones  # eager package\n"
            "import unturtle.models.latent\n"
            "assert hub.generation_algorithms.values() == ()\n"
            "assert hub.backbone_integrations.values() == ()\n"
            "print('CLEAN')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert "CLEAN" in result.stdout, result.stderr[-800:]


class TestHubBoundDecorator:
    def test_generation_decorator_registers_and_returns_the_function(self):
        hub = RegistryHub()

        @hub.generation(
            name="toy",
            family="masked_discrete",
            supports=lambda model: True,
            flags={"use_cache": False},
            auto_priority=99,
            unsupported_message=lambda model: "toy unsupported",
        )
        def run_toy(model, request):
            return "ran"

        assert run_toy.__name__ == "run_toy"  # original object, unchanged
        entry = hub.generation_algorithms.get("toy")
        assert entry.runner is run_toy
        assert entry.family == "masked_discrete"

    def test_decorator_applies_the_same_duplicate_checks(self):
        hub = RegistryHub()

        def make():
            @hub.generation(
                name="dup",
                family="masked_discrete",
                supports=lambda model: True,
                auto_priority=98,
                unsupported_message=lambda model: "no",
            )
            def runner(model, request):
                pass

        make()
        with pytest.raises(ValueError, match="'dup'"):
            make()

    def test_decorated_algorithm_participates_in_hub_resolution(self):
        hub = bootstrap_builtin_hub(RegistryHub())

        class Weird:
            speaks_toy = True

        @hub.generation(
            name="toy",
            family="masked_discrete",
            supports=lambda model: getattr(model, "speaks_toy", False),
            auto_priority=5,  # ahead of everything auto-eligible
            unsupported_message=lambda model: "model does not speak toy",
        )
        def run_toy(model, request):
            return "ran"

        assert hub.resolve_generation("auto", Weird(), bd3lm_requested=False) == "toy"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


class TestReviewPins147:
    def test_unregister_of_a_same_named_twin_keeps_the_survivors_aliases(self):
        """#147 review: the alias sweep must honor the same identity
        semantics as item removal — removing a same-named twin must not
        strip the SURVIVOR's aliases."""
        reg = Registry("thing")
        survivor = reg.register(Thing("a"), aliases=("alpha",))
        twin = Thing("a")
        reg._items.append(twin)  # the white-box seam the docstring warns about
        reg.unregister(twin)
        assert reg.find("a") is survivor
        assert reg.find("alpha") is survivor, "twin removal stripped the alias"
        # and once the LAST holder of the name goes, the alias goes with it
        reg.unregister(survivor)
        assert reg.find("alpha") is None

    def test_duplicate_integration_names_are_now_rejected(self):
        """#147 review: a RECORDED tightening vs the old module-global code
        (which checked only model_type namespaces).  Two integrations
        answering to one name would make hub lookups order-dependent."""
        from unturtle.models.integrations.registry import (
            BackboneIntegration,
            register_integration_into,
        )
        from unturtle.registry import RegistryHub

        hub = RegistryHub()
        register_integration_into(
            hub,
            BackboneIntegration(
                name="twin", model_types=("t-one",), _native_resolver=lambda: None
            ),
        )
        with pytest.raises(ValueError, match="twin"):
            register_integration_into(
                hub,
                BackboneIntegration(
                    name="twin", model_types=("t-two",), _native_resolver=lambda: None
                ),
            )
