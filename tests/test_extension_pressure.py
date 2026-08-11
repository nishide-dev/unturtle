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

"""#144 extension pressure test — the #141 acceptance gate.

The fixture (tests/fixtures/toy_extension_plugin.py) is an out-of-core-style
extension: existing masked process + existing mdlm training recipe + one NEW
generation runner + one NEW MethodSpec + real capability validation.  The
tests below are the issue's nine-step runtime proof plus the adversarial
lifecycle cases, and this file plus the fixture are the ONLY files the
extension needed — the accounting table lives in the PR/issue evidence.

Lifecycle honesty note: registration is NOT transactional (a documented
#142 decision, not an accident) — a partially failing registration leaves
the components registered before the failure.  The test pins exactly that
documented behavior rather than inventing rollback machinery for this test.
"""

import subprocess
import sys

import pytest

from tests.fixtures import toy_extension_plugin
from unturtle.methods import (
    MethodSpec,
    describe_method,
    resolve_method,
    validate_method,
)
from unturtle.models.generation.sampler import GenerationRequest
from unturtle.registry import RegistryHub, bootstrap_builtin_hub


def fresh_hub() -> RegistryHub:
    return bootstrap_builtin_hub(RegistryHub())


class CompatibleModel:
    """Opts into toy_echo AND resolves to an integration declaring
    masked_generation (the mdlm-dit builtin)."""

    supports_toy_echo = True

    class _Cfg:
        model_type = "mdlm-dit"
        hybrid_attention = False

    config = _Cfg()

    def _sample(self):
        pass


class IncompatibleModel:
    """No toy_echo opt-in: must fail through the existing capability
    machinery, before any runner executes."""

    class _Cfg:
        model_type = "mdlm-dit"

    config = _Cfg()

    def _sample(self):
        pass


class TestRuntimeProof:
    """The issue's steps 1-9, in order."""

    def test_step1_fresh_hub_lacks_the_method(self):
        hub = fresh_hub()
        assert hub.methods.find(toy_extension_plugin.METHOD_NAME) is None
        assert hub.generation_algorithms.find("toy_echo") is None

    def test_step2_import_alone_registers_nothing(self):
        """Also checked against the DEFAULT hub in a subprocess where the
        import order is fully controlled."""
        code = (
            "import tests.fixtures.toy_extension_plugin  # import only\n"
            "from unturtle.models.generation import sampler\n"
            "assert sampler.find_algorithm('toy_echo') is None\n"
            "from unturtle.registry import RegistryHub, bootstrap_builtin_hub\n"
            "hub = bootstrap_builtin_hub(RegistryHub())\n"
            "assert hub.generation_algorithms.find('toy_echo') is None\n"
            "print('SILENT')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd="/grouper/nishide.21066-1000003/projects/unturtle-new",
        )
        assert "SILENT" in result.stdout, result.stderr[-800:]

    def test_step3_registration_makes_it_visible_in_that_hub_only(self):
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)
        assert hub.methods.find("toy_echo") is not None
        assert hub.generation_algorithms.find("toy_echo") is not None

        from unturtle.models.generation import sampler

        assert sampler.find_algorithm("toy_echo") is None

    def test_step4_describe_resolve_validate_through_the_supplied_hub(self):
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)

        description = describe_method("toy_echo", hub=hub)
        assert description["process"]["name"] == "masked"
        assert description["training"]["name"] == "mdlm"
        assert description["generation"] == ["toy_echo"]

        resolved = resolve_method("toy_echo", hub=hub)
        assert resolved.process.name == "masked"

        validated = validate_method("toy_echo", model=CompatibleModel(), hub=hub)
        assert validated.unverified_capabilities == frozenset()

    def test_step5_incompatible_model_fails_before_any_runner(self):
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)
        with pytest.raises(ValueError, match="toy_echo"):
            validate_method("toy_echo", model=IncompatibleModel(), hub=hub)

    def test_step6_dispatch_runs_the_plugin_runner_like_a_builtin(self):
        """The heart of the gate: the plugin's runner executes through the
        SAME GenerationAlgorithm dispatch path as builtins, with no core
        conditional naming it — proven by the deterministic transformation
        only THIS runner performs."""
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)
        request = GenerationRequest(
            inputs=[1, 2, 3, 4], generation_config=None, kwargs={}
        )
        result = hub.dispatch_generation(
            CompatibleModel(), request, algorithm="toy_echo"
        )
        assert result == {"method": "toy_echo", "tokens": [4, 3, 2, 1]}

    def test_step7_second_hub_registers_independently(self):
        hub_a, hub_b = fresh_hub(), fresh_hub()
        toy_extension_plugin.register_unturtle(hub_a)
        toy_extension_plugin.register_unturtle(hub_b)
        request = GenerationRequest(inputs=[7, 8], generation_config=None, kwargs={})
        for hub in (hub_a, hub_b):
            assert hub.dispatch_generation(
                CompatibleModel(), request, algorithm="toy_echo"
            ) == {"method": "toy_echo", "tokens": [8, 7]}

    def test_step8_discarding_the_hub_leaves_the_default_untouched(self):
        from unturtle.models.generation import sampler

        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)
        del hub
        assert sampler.find_algorithm("toy_echo") is None
        assert "toy_echo" not in [a.name for a in sampler.iter_algorithms()]

    def test_step9_default_auto_resolution_is_invariant(self):
        from unturtle.models.generation import sampler

        class Masked:
            def _sample(self):
                pass

        before = sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False)
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)
        after = sampler.resolve_algorithm("auto", Masked(), bd3lm_requested=False)
        assert before == after == "mdlm"


class TestAdversarialLifecycle:
    def test_double_registration_into_one_hub_is_loud(self):
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)
        with pytest.raises(ValueError, match="toy_echo"):
            toy_extension_plugin.register_unturtle(hub)

    def test_partial_registration_is_documented_non_transactional(self):
        """No rollback framework exists (deliberate #142/#144 decision).
        The pinned guarantee is only: the failure is loud, and everything
        registered BEFORE the failing call remains — callers own hub
        disposal on failure (cheap, since hubs are instances)."""
        hub = fresh_hub()
        hub.method(MethodSpec(name="toy_echo"))  # occupy the METHOD name
        with pytest.raises(ValueError, match="toy_echo"):
            toy_extension_plugin.register_unturtle(hub)
        # The generation algorithm registered before the method collision
        # stays — non-transactional, exactly as documented.
        assert hub.generation_algorithms.find("toy_echo") is not None

    def test_algorithm_name_collision_with_a_builtin_is_loud(self):
        hub = fresh_hub()

        def clash(hub_):
            @hub_.generation(
                name="mdlm",  # collides with the builtin
                family="masked_discrete",
                supports=lambda m: True,
                auto_priority=91,
                unsupported_message=lambda m: "no",
            )
            def run(model, request):
                pass

        with pytest.raises(ValueError, match="mdlm"):
            clash(hub)

    def test_plugin_exception_does_not_poison_the_default_hub(self):
        from unturtle.models.generation import sampler

        hub = fresh_hub()
        hub.method(MethodSpec(name="toy_echo"))
        with pytest.raises(ValueError):
            toy_extension_plugin.register_unturtle(hub)
        assert sampler.find_algorithm("toy_echo") is None
        assert (
            sampler.resolve_algorithm("auto", CompatibleModel(), bd3lm_requested=False)
            == "mdlm"
        )

    def test_plugin_never_wins_auto_even_in_its_own_hub(self):
        """auto_priority=90 sits behind every builtin: a masked model in the
        plugin's hub still auto-resolves to mdlm — the plugin is explicit
        opt-in there too."""
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)

        class MaskedOptIn:
            supports_toy_echo = True

            def _sample(self):
                pass

        assert (
            hub.resolve_generation("auto", MaskedOptIn(), bd3lm_requested=False)
            == "mdlm"
        )


class TestDecoratorErgonomics:
    def test_the_decorated_runner_keeps_its_identity(self):
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)
        entry = hub.generation_algorithms.get("toy_echo")
        assert entry.runner.__name__ == "run_toy_echo"
        assert entry.runner.__doc__ and "solver" in entry.runner.__doc__

    def test_registration_errors_attach_to_the_registration_call(self):
        """Errors surface at register_unturtle(hub), never at import time —
        pinned by the fact that step 2 (import only) cannot raise and the
        double-registration error carries the extension's own name."""
        hub = fresh_hub()
        toy_extension_plugin.register_unturtle(hub)
        with pytest.raises(ValueError) as excinfo:
            toy_extension_plugin.register_unturtle(hub)
        assert "toy_echo" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
