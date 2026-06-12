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

"""Tests for BD3LM generation — prepare_for_sampling and A2D mixin API."""

from __future__ import annotations

import math

import pytest
import torch

from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
    MaskedDiffusionGenerationMixin,
    prepare_for_sampling,
)

PAD_ID = 0
MASK_ID = 100
EOS_ID = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model():
    from unturtle.models.conversion.a2d.tiny_a2d import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    config = TinyA2DLlamaConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=256,
        mask_token_id=MASK_ID,
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
    )
    model = TinyA2DLlamaLMHeadModel(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# prepare_for_sampling — pure function tests (unchanged logic, new import path)
# ---------------------------------------------------------------------------


class TestPrepareForSampling:
    def test_output_shapes(self):
        x = torch.tensor([[5, 6, 7, 0, 0]])  # 3 valid, 2 padding
        attn, pos = prepare_for_sampling(x, block_size=2, pad_token_id=PAD_ID)
        assert attn.shape == (1, 1, 5, 5)
        assert pos.shape == (1, 5)

    def test_position_ids_skip_padding(self):
        x = torch.tensor([[5, 6, 7, 0, 0]])
        _, pos = prepare_for_sampling(x, block_size=2, pad_token_id=PAD_ID)
        assert pos[0, 0].item() == 0
        assert pos[0, 1].item() == 1
        assert pos[0, 2].item() == 2
        # Padding positions → 0
        assert pos[0, 3].item() == 0
        assert pos[0, 4].item() == 0

    def test_attention_mask_excludes_padding(self):
        x = torch.tensor([[5, 6, 7, 0, 0]])
        mask, _ = prepare_for_sampling(x, block_size=2, pad_token_id=PAD_ID)
        assert not mask[0, 0, :, 3:].any(), "Padding columns must not be attended to"
        assert not mask[0, 0, 3:, :].any(), "Padding rows must not attend"


# ---------------------------------------------------------------------------
# BD3LM generation via model.generate(use_block_diffusion=True)
# ---------------------------------------------------------------------------


class TestBD3LMViaModelAPI:
    def test_returns_tensor_with_correct_shape(self, tiny_model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 4
        padded_prompt_len = math.ceil(prompt.shape[1] / block_size) * block_size
        with torch.no_grad():
            out = tiny_model.generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=2,
                mask_token_id=MASK_ID,
                pad_token_id=PAD_ID,
            )
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, padded_prompt_len + max_new_tokens)

    def test_output_longer_than_prompt(self, tiny_model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 8
        padded_prompt_len = math.ceil(prompt.shape[1] / block_size) * block_size
        with torch.no_grad():
            out = tiny_model.generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=4,
                mask_token_id=MASK_ID,
                pad_token_id=PAD_ID,
            )
        assert out.shape == (1, padded_prompt_len + max_new_tokens)

    def test_no_mask_tokens_in_generated_region(self, tiny_model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 8
        padded_prompt_len = math.ceil(prompt.shape[1] / block_size) * block_size
        with torch.no_grad():
            out = tiny_model.generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=4,
                mask_token_id=MASK_ID,
                pad_token_id=PAD_ID,
                temperature=0.0,
            )
        assert not (out[:, padded_prompt_len:] == MASK_ID).any()

    def test_right_shift_logits_no_error(self, tiny_model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 4
        padded_prompt_len = math.ceil(prompt.shape[1] / block_size) * block_size
        with torch.no_grad():
            out = tiny_model.generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=2,
                mask_token_id=MASK_ID,
                pad_token_id=PAD_ID,
                right_shift_logits=True,
                temperature=0.0,
            )
        assert out.shape == (1, padded_prompt_len + max_new_tokens)
        assert not (out[:, padded_prompt_len:] == MASK_ID).any()

    def test_cfg_scale_no_error(self, tiny_model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 4
        padded_prompt_len = math.ceil(prompt.shape[1] / block_size) * block_size
        with torch.no_grad():
            out = tiny_model.generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=2,
                mask_token_id=MASK_ID,
                pad_token_id=PAD_ID,
                cfg_scale=1.0,
                temperature=0.0,
            )
        assert out.shape == (1, padded_prompt_len + max_new_tokens)
        assert not (out[:, padded_prompt_len:] == MASK_ID).any()

    def test_eos_stopping_runs(self, tiny_model):
        """EOS token in output does not crash generation; no masks after EOS."""
        prompt = torch.tensor([[1, 2, 3, 4]])
        block_size = 4
        max_new_tokens = 8
        padded_prompt_len = math.ceil(prompt.shape[1] / block_size) * block_size
        with torch.no_grad():
            out = tiny_model.generate(
                inputs=prompt,
                use_block_diffusion=True,
                bd3lm_block_size=block_size,
                max_new_tokens=max_new_tokens,
                steps=1,
                mask_token_id=MASK_ID,
                pad_token_id=PAD_ID,
            )
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] >= padded_prompt_len
        # If EOS appeared, no mask tokens should remain after the first EOS position
        eos_positions = (out == EOS_ID).nonzero(as_tuple=True)
        if len(eos_positions[0]) > 0:
            first_eos = eos_positions[1].min().item()
            assert not (out[:, first_eos + 1 :] == MASK_ID).any(), (
                "No mask tokens should remain after EOS"
            )


# ---------------------------------------------------------------------------
# Import / export smoke test
# ---------------------------------------------------------------------------


def test_prepare_for_sampling_importable_from_models():
    from unturtle.models import prepare_for_sampling as pfs  # noqa: F401

    assert callable(pfs)


class _TinyDiffusionModel(torch.nn.Module, MaskedDiffusionGenerationMixin):
    def __init__(self, vocab_size: int = 16, mask_token_id: int = 0):
        super().__init__()
        self.config = type("Cfg", (), {"mask_token_id": mask_token_id})
        self.vocab_size = vocab_size

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Deterministic logits: always favor token 1 at every position.
        B, L = input_ids.shape
        logits = torch.zeros(B, L, self.vocab_size, dtype=torch.float32)
        logits[..., 1] = 10.0
        return type("Out", (), {"logits": logits})


def test_diffusion_stream_callback_called_with_x_snapshot():
    model = _TinyDiffusionModel()
    prompt = torch.tensor([[5, 6]], dtype=torch.long)
    called = []

    def stream_cb(step: int, total: int, x: torch.LongTensor):
        called.append((step, total, x))

    cfg = MaskedDiffusionGenerationConfig(
        steps=3,
        max_new_tokens=2,
        mask_token_id=0,
        return_dict=False,
        stream_callback=stream_cb,
    )
    _ = model.generate(prompt, generation_config=cfg)

    assert [c[0] for c in called] == [1, 2, 3]
    assert all(c[1] == 3 for c in called)
    ptrs = [t.data_ptr() for _, _, t in called]
    assert len(set(ptrs)) == len(ptrs)
    assert called[-1][2].dtype == torch.long
    assert called[-1][2].ndim == 2
