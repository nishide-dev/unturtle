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

"""Hub-scoped dispatch + integration lookup (#143 scope amendment).

The two seams #142 deliberately deferred, absorbed here BEFORE MethodSpec so
#144's pressure test measures composition, not missing substrate:

- an algorithm registered only in an isolated hub must resolve AND DISPATCH
  through that hub, invisible to the default;
- the same algorithm name in two hubs with different runners must dispatch
  to each hub's own runner (kills any hidden default-hub fallback);
- an integration registered only in an isolated hub must be visible to
  hub-scoped lookup and absent from module-default lookup;
- none of this may change default `auto` behavior or module-level APIs.
"""

import pytest

from unturtle.models.generation.sampler import GenerationRequest
from unturtle.registry import RegistryHub, bootstrap_builtin_hub


def _hub_with_toy(runner_tag: str) -> RegistryHub:
    hub = bootstrap_builtin_hub(RegistryHub())

    @hub.generation(
        name="toy",
        family="masked_discrete",
        supports=lambda model: getattr(model, "speaks_toy", False),
        auto_priority=5,
        unsupported_message=lambda model: "model does not speak toy",
    )
    def run_toy(model, request):
        return f"ran:{runner_tag}"

    return hub


class ToyModel:
    speaks_toy = True


def _request() -> GenerationRequest:
    return GenerationRequest(inputs=None, generation_config=None, kwargs={})


class TestHubScopedDispatch:
    def test_isolated_algorithm_dispatches_through_its_hub(self):
        hub = _hub_with_toy("isolated")
        result = hub.dispatch_generation(ToyModel(), _request(), algorithm="toy")
        assert result == "ran:isolated"

    def test_auto_dispatch_uses_the_hub_priorities(self):
        hub = _hub_with_toy("auto")
        result = hub.dispatch_generation(ToyModel(), _request(), algorithm="auto")
        assert result == "ran:auto"

    def test_the_default_hub_cannot_see_the_isolated_algorithm(self):
        from unturtle.models.generation import sampler

        _hub_with_toy("leak-check")
        assert sampler.find_algorithm("toy") is None
        with pytest.raises(ValueError, match="toy"):
            sampler.dispatch_generation(ToyModel(), _request(), algorithm="toy")

    def test_same_name_in_two_hubs_dispatches_each_hubs_runner(self):
        """Adversarial: a hidden fallback to the default hub (or to the
        first-created hub) would send both calls to one runner."""
        hub_a = _hub_with_toy("A")
        hub_b = _hub_with_toy("B")
        assert (
            hub_a.dispatch_generation(ToyModel(), _request(), algorithm="toy")
            == "ran:A"
        )
        assert (
            hub_b.dispatch_generation(ToyModel(), _request(), algorithm="toy")
            == "ran:B"
        )

    def test_hub_dispatch_preserves_capability_errors(self):
        hub = _hub_with_toy("err")

        class Mute:
            pass

        with pytest.raises(ValueError, match="toy"):
            hub.dispatch_generation(Mute(), _request(), algorithm="toy")

    def test_isolated_registration_leaves_default_auto_unchanged(self):
        from unturtle.models.generation import sampler

        class Masked:
            def _sample(self):
                pass

        before = sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False)
        _hub_with_toy("no-effect")
        after = sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False)
        assert before == after == "mdlm"

    def test_module_dispatch_still_reaches_builtin_runners(self):
        """The compatibility wrapper: module-level dispatch over the default
        hub is behaviorally the pre-amendment path (spied runner)."""
        from unturtle.models.generation import sampler

        calls = []

        class Masked:
            def _sample(self, inputs=None, **kwargs):
                calls.append("mdlm-loop")
                return "out"

        result = sampler.dispatch_generation(Masked(), _request(), algorithm="mdlm")
        assert result == "out" and calls == ["mdlm-loop"]


class TestHubScopedIntegrationLookup:
    def _isolated_integration_hub(self):
        from unturtle.models.integrations.registry import (
            BackboneIntegration,
            register_integration_into,
        )

        hub = bootstrap_builtin_hub(RegistryHub())
        integration = BackboneIntegration(
            name="toy-backbone",
            model_types=("toy-model-type",),
            _native_resolver=lambda: None,
            peft_model_types=("toy-peft-type",),
            _peft_patcher=lambda: None,
            capabilities=frozenset({"toy_generation"}),
        )
        register_integration_into(hub, integration)
        return hub, integration

    def test_hub_lookup_sees_the_isolated_integration(self):
        hub, integration = self._isolated_integration_hub()
        assert hub.find_integration("toy-model-type") is integration
        assert hub.find_peft_integration("toy-peft-type") is integration

    def test_default_lookup_does_not_see_it(self):
        from unturtle.models.integrations import registry as integrations

        self._isolated_integration_hub()
        assert integrations.find_integration("toy-model-type") is None
        assert integrations.find_peft_integration("toy-peft-type") is None

    def test_hub_lookup_still_finds_builtins(self):
        hub, _ = self._isolated_integration_hub()
        assert hub.find_integration("mdlm-dit").name == "mdlm-dit"
        assert hub.find_peft_integration("llama").name == "tiny-a2d-llama"

    def test_hub_lookup_none_and_unknown_semantics_match_the_module(self):
        from unturtle.models.integrations import registry as integrations

        hub, _ = self._isolated_integration_hub()
        assert hub.find_integration(None) is None
        assert hub.find_integration("nonexistent") is None
        assert integrations.find_integration(None) is None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
