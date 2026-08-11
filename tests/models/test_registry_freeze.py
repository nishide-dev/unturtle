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

"""Behavior freeze for the two live registries (#142, written BEFORE the
instance-backed substrate).

Everything here is an explicit literal of CURRENT behavior, asserted only
through the public module APIs.  These tests must stay green, unmodified,
across the #142 refactor — they are the differential contract the issue
requires ("a fresh isolated registry populated by builtin bootstrap must
resolve the same representative models/requests as the current global
path").
"""

import pytest
import torch

from unturtle.models.generation import sampler
from unturtle.models.integrations import registry as integrations


class TestGenerationBuiltinsFrozen:
    def test_builtin_set_order_names_families(self):
        entries = [(a.name, a.family) for a in sampler.iter_algorithms()]
        assert entries == [
            ("block_ar", "canvas"),
            ("bd3lm", "masked_discrete"),
            ("block_decode", "masked_discrete"),
            ("mdlm", "masked_discrete"),
            ("flowlm", "continuous_flow"),
            ("ladiff", "latent_guided"),
            ("dfm", "discrete_flow"),
        ]

    def test_auto_priorities_and_eligibility(self):
        frozen = {
            "block_ar": (10, True),
            "bd3lm": (20, False),  # opt-in only, never auto
            "block_decode": (30, True),
            "mdlm": (40, True),
            "flowlm": (50, True),
            "ladiff": (60, True),
            "dfm": (70, True),
        }
        for algorithm in sampler.iter_algorithms():
            assert (
                algorithm.auto_priority,
                algorithm.auto_eligible,
            ) == frozen[algorithm.name], algorithm.name

    def test_flags_frozen(self):
        assert sampler.algorithm_to_flags("bd3lm") == {
            "use_cache": False,
            "use_block_diffusion": True,
        }
        assert sampler.algorithm_to_flags("block_decode") == {
            "use_cache": True,
            "use_block_diffusion": False,
        }
        assert sampler.algorithm_to_flags("mdlm") == {
            "use_cache": False,
            "use_block_diffusion": False,
        }
        for unflagged in ("block_ar", "flowlm", "ladiff", "dfm"):
            assert sampler.algorithm_to_flags(unflagged) == {}

    def test_flags_copies_cannot_corrupt_the_registry(self):
        flags = sampler.algorithm_to_flags("mdlm")
        flags["use_cache"] = True
        assert sampler.algorithm_to_flags("mdlm")["use_cache"] is False


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _mdlm_only():
    class M:
        config = _Cfg(hybrid_attention=False)

        def _sample(self):  # masked loop hook
            pass

    return M()


def _block_decode_capable():
    class M:
        config = _Cfg(hybrid_attention=False)
        supports_block_decode = True

        def _sample(self):
            pass

        def _model_forward_with_cache(self):
            pass

    return M()


def _hybrid_block_decode_capable():
    class M:
        config = _Cfg(hybrid_attention=True)
        supports_block_decode = True

        def _sample(self):
            pass

        def _model_forward_with_cache(self):
            pass

    return M()


def _bd3lm_capable():
    class M:
        config = _Cfg(hybrid_attention=False)
        supports_block_decode = True

        def _sample(self):
            pass

        def _model_forward_with_cache(self):
            pass

        def _sample_block_diffusion(self):
            pass

    return M()


def _canvas():
    class M:
        config = _Cfg()

        def _denoising_step(self):
            pass

        def generate(self, *a, **k):
            pass

    return M()


def _dfm_model():
    class M(torch.nn.Module):
        supports_dfm_generation = True

        def dfm_denoiser(self, x_t, t, h):
            pass

    return M()


class TestAutoResolutionFrozen:
    def test_plain_masked_model_resolves_to_mdlm(self):
        assert (
            sampler.resolve_algorithm("auto", _mdlm_only(), bd3lm_requested=False)
            == "mdlm"
        )

    def test_cache_capable_model_resolves_to_block_decode(self):
        assert (
            sampler.resolve_algorithm(
                "auto", _block_decode_capable(), bd3lm_requested=False
            )
            == "block_decode"
        )

    def test_hybrid_model_never_auto_selects_a_cache_path(self):
        """#128: the block-decode probe excludes hybrid_attention models."""
        assert (
            sampler.resolve_algorithm(
                "auto", _hybrid_block_decode_capable(), bd3lm_requested=False
            )
            == "mdlm"
        )

    def test_explicit_block_decode_on_hybrid_is_a_loud_error(self):
        with pytest.raises(ValueError, match="block.decode|block_decode"):
            sampler.resolve_algorithm(
                "block_decode", _hybrid_block_decode_capable(), bd3lm_requested=False
            )

    def test_bd3lm_is_opt_in_only(self):
        model = _bd3lm_capable()
        assert (
            sampler.resolve_algorithm("auto", model, bd3lm_requested=False)
            == "block_decode"
        ), "bd3lm must never be chosen without the opt-in"
        assert sampler.resolve_algorithm("auto", model, bd3lm_requested=True) == "bd3lm"

    def test_bd3lm_requested_on_incapable_model_never_falls_back(self):
        with pytest.raises(ValueError):
            sampler.resolve_algorithm("auto", _mdlm_only(), bd3lm_requested=True)

    def test_canvas_model_wins_auto_even_with_bd3lm_requested(self):
        assert (
            sampler.resolve_algorithm("auto", _canvas(), bd3lm_requested=True)
            == "block_ar"
        )

    def test_dfm_only_model_resolves_to_dfm_via_the_priority_loop(self):
        """#65's opt-in lives on the MODEL (supports_dfm_generation defaults
        False and only the explicit mixin sets it) — the ALGORITHM itself is
        auto-eligible, so a model that carries the opt-in and nothing else
        resolves to dfm."""
        model = _dfm_model()
        assert sampler.resolve_algorithm("auto", model, bd3lm_requested=False) == "dfm"
        assert sampler.resolve_algorithm("dfm", model, bd3lm_requested=False) == "dfm"

    def test_explicit_dfm_without_the_capability_is_refused(self):
        with pytest.raises(ValueError, match="dfm"):
            sampler.resolve_algorithm("dfm", _mdlm_only(), bd3lm_requested=False)

    def test_unknown_algorithm_names_all_supported(self):
        with pytest.raises(ValueError, match="mdlm"):
            sampler.algorithm_to_flags("nonexistent")


class TestIntegrationBuiltinsFrozen:
    def test_builtin_names_and_model_types(self):
        table = {
            i.name: (tuple(i.model_types), tuple(i.peft_model_types))
            for i in integrations.iter_integrations()
        }
        assert table == {
            "llada": (("llada",), ("llada",)),
            "mdlm-dit": (("mdlm-dit",), ()),
            "dream": (("dream", "Dream"), ("dream", "Dream")),
            "tiny-a2d-llama": (("tiny-a2d-llama",), ("tiny-a2d-llama", "llama")),
            "tiny-a2d-qwen2": (("tiny-a2d-qwen2",), ("tiny-a2d-qwen2", "qwen2")),
            "tiny-a2d-qwen3": (("tiny-a2d-qwen3",), ("tiny-a2d-qwen3", "qwen3")),
            "modernbert-diffusion": ((), ("modernbert-diffusion",)),
            "diffusion-gemma": (("diffusion_gemma",), ()),
        }

    def test_registration_order_is_deterministic(self):
        names = [i.name for i in integrations.iter_integrations()]
        assert names == [
            "llada",
            "mdlm-dit",
            "dream",
            "tiny-a2d-llama",
            "tiny-a2d-qwen2",
            "tiny-a2d-qwen3",
            "modernbert-diffusion",
            "diffusion-gemma",
        ]

    def test_model_type_conflicts_are_rejected(self):
        clash = integrations.BackboneIntegration(
            name="clash",
            model_types=("llada",),
            _native_resolver=lambda: None,
        )
        with pytest.raises(ValueError, match="llada"):
            integrations.register_integration(clash)

    def test_peft_model_type_conflicts_are_rejected(self):
        clash = integrations.BackboneIntegration(
            name="clash-peft",
            model_types=("brand-new-type",),
            _native_resolver=lambda: None,
            peft_model_types=("qwen3",),
            _peft_patcher=lambda: None,
        )
        with pytest.raises(ValueError, match="qwen3"):
            integrations.register_integration(clash)

    def test_find_integration_by_model_type(self):
        assert integrations.find_integration("mdlm-dit").name == "mdlm-dit"
        assert integrations.find_integration("nonexistent") is None
        assert integrations.find_integration(None) is None

    def test_peft_lookup_is_a_separate_namespace(self):
        """A PEFT-wrapped Tiny-A2D reports plain llama/qwen — the load
        model_types must NOT serve PEFT lookups and vice versa."""
        assert integrations.find_peft_integration("llama").name == "tiny-a2d-llama"
        assert integrations.find_integration("llama") is None
        assert integrations.find_peft_integration("mdlm-dit") is None
