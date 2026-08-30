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

"""#174 fix: direct construction and ``from_pretrained`` rebuild the SAME
rotary buffer through ONE canonical initializer.

Mutant coverage: initializer call removed / wrong base / wrong exponent or
index stride / wrong head dim / dtype pinned / device pinned to CPU / only
the first layer initialized / Dream or MDLM-DiT not initialized / constructor
and reload using different initializers / test bypassing from_pretrained.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = [pytest.mark.gpu]  # importing unturtle needs the unsloth chain


def _formula(dim: int, base: float) -> torch.Tensor:
    # independent, index-by-index (not a copy of build_inv_freq's expression)
    return torch.tensor(
        [1.0 / math.pow(base, (2 * i) / dim) for i in range(dim // 2)],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# MDLM-DiT Rotary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim,base", [(16, 10_000), (32, 10_000), (24, 500)])
def test_dit_rotary_matches_independent_formula(dim, base):
    from unturtle.models.backbones.mdlm_dit.modeling_mdlm_dit import Rotary

    rotary = Rotary(dim, base=base)
    assert rotary.inv_freq.dtype == torch.float32
    assert torch.allclose(rotary.inv_freq, _formula(dim, base), rtol=0, atol=1e-7)


def test_dit_reset_rebuilds_exactly_what_the_constructor_built():
    from unturtle.models.backbones.mdlm_dit.modeling_mdlm_dit import Rotary

    rotary = Rotary(16)
    built = rotary.inv_freq.clone()
    with torch.no_grad():
        rotary.inv_freq.fill_(float("nan"))
    rotary.reset_parameters()
    assert torch.equal(rotary.inv_freq, built)


def test_dit_constructor_and_reset_share_one_initializer(monkeypatch):
    """Kills 'constructor and reload use different initializers': both must
    route through build_inv_freq — a sentinel replaces it and both paths see it."""
    from unturtle.models.backbones.mdlm_dit import modeling_mdlm_dit as mod

    calls = []

    def sentinel(dim, base, *, device=None, dtype=torch.float32):
        calls.append((dim, base, str(device), dtype))
        return torch.full((dim // 2,), 0.5, device=device, dtype=dtype)

    monkeypatch.setattr(mod, "build_inv_freq", sentinel)
    rotary = mod.Rotary(16, base=777)
    assert calls and calls[-1][:2] == (16, 777)
    assert torch.all(rotary.inv_freq == 0.5)
    rotary.reset_parameters()
    assert len(calls) == 2 and calls[-1][:2] == (16, 777)


def test_dit_reset_respects_buffer_dtype_and_device():
    from unturtle.models.backbones.mdlm_dit.modeling_mdlm_dit import Rotary

    rotary = Rotary(16).to(torch.bfloat16)
    rotary.reset_parameters()
    assert rotary.inv_freq.dtype == torch.bfloat16
    assert torch.allclose(
        rotary.inv_freq.float(), _formula(16, 10_000).to(torch.bfloat16).float()
    )
    if torch.cuda.is_available():
        rotary = Rotary(16).cuda()
        with torch.no_grad():
            rotary.inv_freq.fill_(float("nan"))
        rotary.reset_parameters()
        assert rotary.inv_freq.device.type == "cuda"
        # cross-device: CUDA pow differs from CPU pow by ULPs; same-device
        # constructor-vs-reset identity is covered by test_dit_reset_rebuilds_*
        assert torch.allclose(
            rotary.inv_freq.cpu(), _formula(16, 10_000), rtol=0, atol=1e-7
        )


def test_dit_init_weights_reinitializes_rotary_only():
    """_init_weights must fill a garbage Rotary buffer and leave the adaLN-Zero
    zero-init contract (and every other weight) untouched."""
    from unturtle.models.backbones.mdlm_dit import (
        MDLMDiTConfig,
        MDLMDiTForMaskedDiffusionLM,
    )

    config = MDLMDiTConfig(
        vocab_size=64,
        hidden_size=32,
        cond_dim=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        dropout=0.0,
        max_position_embeddings=16,
        mask_token_id=63,
    )
    torch.manual_seed(0)
    model = MDLMDiTForMaskedDiffusionLM(config)
    weights_before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.model.rotary.inv_freq.fill_(float("nan"))
    model.apply(model._init_weights)
    assert torch.allclose(
        model.model.rotary.inv_freq, _formula(16, 10_000), rtol=0, atol=1e-7
    )
    for key, value in model.state_dict().items():
        assert torch.equal(value, weights_before[key]), key


# ---------------------------------------------------------------------------
# Dream rotary
# ---------------------------------------------------------------------------


def _dream_config(**overrides):
    from unturtle.models.backbones.dream.configuration_dream import DreamConfig

    base = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        mask_token_id=1,
        pad_token_id=0,
    )
    base.update(overrides)
    return DreamConfig(**base)


def test_dream_reset_rebuilds_formula_on_every_rotary():
    from unturtle.models.backbones.dream.modeling_dream import (
        DreamModel,
        DreamRotaryEmbedding,
    )

    config = _dream_config(rope_theta=12_345.0)
    torch.manual_seed(0)
    model = DreamModel(config)
    rotaries = [m for m in model.modules() if isinstance(m, DreamRotaryEmbedding)]
    assert len(rotaries) >= 2, "expected per-layer and model-level rotaries"
    head_dim = config.hidden_size // config.num_attention_heads
    for rotary in rotaries:
        with torch.no_grad():
            rotary.inv_freq.fill_(float("nan"))
    model.apply(model._init_weights)
    for rotary in rotaries:
        assert torch.isfinite(rotary.inv_freq).all()
        assert torch.allclose(
            rotary.inv_freq.float(), _formula(head_dim, 12_345.0), rtol=0, atol=1e-6
        )


def test_dream_init_weights_keeps_linear_embedding_init():
    """The Rotary branch is additive: Linear/Embedding init statistics are the
    existing ones (std=initializer_range, zero bias), not the base class's."""
    from unturtle.models.backbones.dream.modeling_dream import DreamModel

    config = _dream_config(initializer_range=0.02)
    model = DreamModel(config)
    linear = model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        linear.weight.fill_(5.0)
        linear.bias.fill_(5.0)
    model._init_weights(linear)
    assert abs(linear.weight.std().item() - 0.02) < 0.01
    assert torch.all(linear.bias == 0)


# ---------------------------------------------------------------------------
# from_pretrained itself (the load path, not a stand-in) under NaN-poisoned
# uninitialized memory — both families, every rotary buffer
# ---------------------------------------------------------------------------


class _empty_like_nan:
    def __enter__(self):
        self.original = torch.empty_like
        self.calls = 0

        def wrapped(*args, **kwargs):
            self.calls += 1
            return self.original(*args, **kwargs).fill_(float("nan"))

        torch.empty_like = wrapped
        return self

    def __exit__(self, *exc):
        torch.empty_like = self.original


@pytest.mark.parametrize("family", ["mdlm_dit", "dream"])
def test_from_pretrained_rebuilds_rotary_under_poisoned_memory(family, tmp_path):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if family == "mdlm_dit":
        from unturtle.models.backbones.mdlm_dit import (
            MDLMDiTConfig,
        )
        from unturtle.models.backbones.mdlm_dit import (
            MDLMDiTForMaskedDiffusionLM as cls,
        )

        config = MDLMDiTConfig(
            vocab_size=64,
            hidden_size=32,
            cond_dim=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            dropout=0.0,
            max_position_embeddings=16,
            mask_token_id=63,
        )
        head_dim, base = 16, 10_000.0
    else:
        from unturtle.models.backbones.dream.modeling_dream import DreamModel as cls

        config = _dream_config()
        head_dim, base = 16, float(config.rope_theta)
    torch.manual_seed(0)
    model = cls(config).eval()
    inputs = torch.randint(2, config.vocab_size, (1, 8))
    model.save_pretrained(tmp_path)
    with _empty_like_nan() as poison:
        reloaded = cls.from_pretrained(tmp_path).eval()
    assert poison.calls > 0, "the test bypassed the poisoned load path"
    buffers = [(n, b) for n, b in reloaded.named_buffers() if n.endswith("inv_freq")]
    assert buffers
    for name, buffer in buffers:
        assert torch.isfinite(buffer).all(), name
        assert torch.allclose(
            buffer.float(), _formula(head_dim, base), rtol=0, atol=1e-6
        ), name
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        ref = model(input_ids=inputs).logits
        got = reloaded(input_ids=inputs).logits
    assert torch.equal(ref, got), "reload is not bit-identical under MATH"
