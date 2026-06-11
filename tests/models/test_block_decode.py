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

"""Tests for block-decode generation with KV-cache."""

import pytest
import torch

from unturtle.models.conversion.a2d.tiny_a2d import (
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
)

MASK_TOKEN_ID = 100


class TestA2DBlockDecode:
    """Test A2D block-decode generation with cache."""

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny A2D LLaMA model for testing."""
        config = TinyA2DLlamaConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            mask_token_id=MASK_TOKEN_ID,
        )
        model = TinyA2DLlamaLMHeadModel(config)
        model.eval()
        return model

    def test_cache_generation_runs(self, tiny_model):
        """Cache-based generation runs without errors and produces correct shape."""
        prompt = torch.tensor([[1, 2, 3, 4, 5]])

        with torch.no_grad():
            output = tiny_model.generate(
                inputs=prompt,
                max_new_tokens=16,
                steps=4,
                use_cache=True,
                block_length=8,
                mask_token_id=MASK_TOKEN_ID,
            )

        assert output.shape == (1, 21)  # 5 + 16 = 21
        # Generated region must have no remaining mask tokens
        assert not torch.any(output[:, 5:] == MASK_TOKEN_ID)

    def test_cache_vs_no_cache_shape_equivalence(self, tiny_model):
        """Cache and no-cache outputs have the same shape."""
        prompt = torch.tensor([[1, 2, 3, 4]])
        max_new = 12
        # block_length=4 → num_blocks=3; steps must be divisible by num_blocks
        steps = 3

        with torch.no_grad():
            output_no_cache = tiny_model.generate(
                inputs=prompt,
                algorithm="mdlm",  # explicit: auto would resolve block_decode and overwrite use_cache=False
                max_new_tokens=max_new,
                steps=steps,
                use_cache=False,
                mask_token_id=MASK_TOKEN_ID,
            )
            output_with_cache = tiny_model.generate(
                inputs=prompt,
                max_new_tokens=max_new,
                steps=steps,
                use_cache=True,
                block_length=4,  # must divide max_new (12); num_blocks=3
                mask_token_id=MASK_TOKEN_ID,
            )

        assert output_no_cache.shape == output_with_cache.shape
        # Generated region must have no remaining mask tokens
        assert not torch.any(output_with_cache[:, 4:] == MASK_TOKEN_ID)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cache_generation_cuda(self, tiny_model):
        """Cache-based generation on CUDA produces correct shape with no mask tokens."""
        tiny_model = tiny_model.cuda()
        prompt = torch.tensor([[1, 2, 3, 4, 5]]).cuda()

        with torch.no_grad():
            output = tiny_model.generate(
                inputs=prompt,
                max_new_tokens=16,
                steps=4,
                use_cache=True,
                block_length=8,
                mask_token_id=MASK_TOKEN_ID,
            )

        assert output.device.type == "cuda"
        assert output.shape == (1, 21)
        assert not torch.any(output[:, 5:] == MASK_TOKEN_ID)


class TestA2DBlockDecodeEquivalence:
    """Test correctness of block-decode generation."""

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny A2D model with fixed seed for reproducibility."""
        torch.manual_seed(42)
        config = TinyA2DLlamaConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            mask_token_id=MASK_TOKEN_ID,
        )
        model = TinyA2DLlamaLMHeadModel(config)
        model.eval()
        return model

    def test_deterministic_output_with_seed(self, tiny_model):
        """Setting the same seed produces deterministic uncached output."""
        prompt = torch.tensor([[1, 2, 3]])

        torch.manual_seed(123)
        with torch.no_grad():
            output1 = tiny_model.generate(
                inputs=prompt,
                algorithm="mdlm",  # explicit: auto would resolve block_decode and overwrite use_cache=False
                max_new_tokens=8,
                steps=4,
                use_cache=False,
                mask_token_id=MASK_TOKEN_ID,
                temperature=0.0,
            )

        torch.manual_seed(123)
        with torch.no_grad():
            output2 = tiny_model.generate(
                inputs=prompt,
                algorithm="mdlm",  # explicit: auto would resolve block_decode and overwrite use_cache=False
                max_new_tokens=8,
                steps=4,
                use_cache=False,
                mask_token_id=MASK_TOKEN_ID,
                temperature=0.0,
            )

        assert torch.equal(output1, output2), (
            "Seeded generation should be deterministic"
        )

    def test_cache_generates_valid_tokens(self, tiny_model):
        """Block-decode with cache produces valid (non-mask) tokens in generated region.

        Note: Block-decode is an approximation of uncached generation for bidirectional
        models — it uses a trimmed KV-cache so attended context differs from the full
        no-cache forward pass.  We verify output correctness (no mask tokens remain,
        shape is right), not exact value matching against the uncached baseline.
        """
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_len = prompt.shape[1]
        max_new = 12

        with torch.no_grad():
            output = tiny_model.generate(
                inputs=prompt,
                max_new_tokens=max_new,
                steps=6,  # num_blocks=3; steps must be divisible by num_blocks
                use_cache=True,
                block_length=4,  # must divide max_new (12)
                mask_token_id=MASK_TOKEN_ID,
                temperature=0.0,
            )

        assert output.shape == (1, prompt_len + max_new)
        # Prompt tokens must be preserved
        assert torch.equal(output[:, :prompt_len], prompt)
        # Generated region must have no remaining mask tokens
        assert not torch.any(output[:, prompt_len:] == MASK_TOKEN_ID)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cache_generates_valid_tokens_cuda(self, tiny_model):
        """Block-decode on CUDA produces valid tokens in generated region."""
        tiny_model = tiny_model.cuda()
        prompt = torch.tensor([[1, 2, 3, 4]]).cuda()
        prompt_len = prompt.shape[1]
        max_new = 8

        with torch.no_grad():
            output = tiny_model.generate(
                inputs=prompt,
                max_new_tokens=max_new,
                steps=4,
                use_cache=True,
                block_length=4,  # must divide max_new (8)
                mask_token_id=MASK_TOKEN_ID,
                temperature=0.0,
            )

        assert output.device.type == "cuda"
        assert output.shape == (1, prompt_len + max_new)
        assert torch.equal(output[:, :prompt_len], prompt)
        assert not torch.any(output[:, prompt_len:] == MASK_TOKEN_ID)


class TestA2DUsesBlockDecodeMixin:
    """Verify TinyA2DGenerationMixin delegates use_cache=True to BlockDecodeMixin."""

    def test_a2d_inherits_block_decode_mixin(self):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DGenerationMixin
        from unturtle.models.generation.block_decode_mixin import BlockDecodeMixin

        assert issubclass(TinyA2DGenerationMixin, BlockDecodeMixin)

    def test_a2d_has_model_forward_with_cache(self):
        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DGenerationMixin

        assert hasattr(TinyA2DGenerationMixin, "_model_forward_with_cache")
        # Must not be the abstract base — concrete implementation
        import inspect

        src = inspect.getsource(TinyA2DGenerationMixin._model_forward_with_cache)
        assert "NotImplementedError" not in src

    def test_a2d_use_cache_routes_through_block_decode_loop(self):
        """use_cache=True must route through _block_decode_loop, not _sample_with_cache."""
        import inspect

        from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DGenerationMixin

        src = inspect.getsource(TinyA2DGenerationMixin._sample_with_cache)
        assert "_block_decode_loop" in src, (
            "TinyA2DGenerationMixin._sample_with_cache must delegate to _block_decode_loop"
        )
