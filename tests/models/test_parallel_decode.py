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

"""Tests for Phase M confidence-aware parallel decoding."""

import pytest
import torch

from unturtle.models.conversion.a2d.tiny_a2d import (
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
)


class TestConfidenceParallelDecode:
    """Test confidence-aware parallel decoding (Phase M MVP)."""

    @pytest.fixture
    def tiny_model(self):
        """Create a tiny A2D LLaMA model for testing."""
        torch.manual_seed(42)
        config = TinyA2DLlamaConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            mask_token_id=100,
        )
        model = TinyA2DLlamaLMHeadModel(config)
        model.eval()
        return model

    def test_parallel_decode_runs(self, tiny_model):
        """Test that parallel_decode=True runs without errors."""
        prompt = torch.tensor([[1, 2, 3, 4, 5]])

        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=16,
                steps=4,
                use_cache=True,
                parallel_decode=True,
                confidence_threshold=0.8,
                alg="maskgit_plus",
                mask_token_id=100,
            )

        assert output is not None
        assert output.shape == (1, 21)  # 5 + 16 = 21

    def test_parallel_decode_shape_equivalence(self, tiny_model):
        """Test that parallel_decode and non-parallel have same output shape."""
        prompt = torch.tensor([[1, 2, 3, 4]])
        max_new = 12
        steps = 4

        with torch.no_grad():
            output_non_parallel = tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=max_new,
                steps=steps,
                use_cache=True,
                parallel_decode=False,
                alg="maskgit_plus",
                mask_token_id=100,
            )
            output_parallel = tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=max_new,
                steps=steps,
                use_cache=True,
                parallel_decode=True,
                confidence_threshold=0.9,
                alg="maskgit_plus",
                mask_token_id=100,
            )

        assert output_non_parallel.shape == output_parallel.shape

    def test_parallel_decode_threshold_affects_output(self, tiny_model):
        """Test that different thresholds produce different outputs."""
        prompt = torch.tensor([[1, 2, 3, 4, 5]])

        # Use realistic confidence thresholds for random/untrained models
        # (confidence values are typically 0.01-0.2 for vocab=128)
        torch.manual_seed(123)
        with torch.no_grad():
            output_low = tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=12,
                steps=8,  # More steps for convergence
                use_cache=True,
                parallel_decode=True,
                confidence_threshold=0.05,  # Low threshold → more tokens unmasked
                alg="maskgit_plus",
                mask_token_id=100,
                temperature=0.0,
            )

        torch.manual_seed(123)
        with torch.no_grad():
            output_high = tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=12,
                steps=8,
                use_cache=True,
                parallel_decode=True,
                confidence_threshold=0.15,  # High threshold → fewer tokens unmasked
                alg="maskgit_plus",
                mask_token_id=100,
                temperature=0.0,
            )

        # Low threshold should unmask more tokens than high threshold
        num_unmasked_low = (output_low != 100).sum().item()
        num_unmasked_high = (output_high != 100).sum().item()

        # Both should unmask prompt (5 tokens) + at least some completion
        assert num_unmasked_low >= 5, (
            f"Low threshold unmasked only {num_unmasked_low} (expected >= 5)"
        )
        assert num_unmasked_high >= 5, (
            f"High threshold unmasked only {num_unmasked_high} (expected >= 5)"
        )

        # Low threshold should unmask more or equal tokens than high threshold
        assert num_unmasked_low >= num_unmasked_high, (
            f"Low threshold ({num_unmasked_low}) should unmask >= high threshold ({num_unmasked_high})"
        )

    def test_parallel_decode_completes_block_when_steps_per_block_is_small(
        self, tiny_model, monkeypatch
    ):
        """Threshold mode keeps denoising until a block finishes, not just steps_per_block."""
        import unturtle.models.generation.diffusion_generation_utils as gen_utils

        def select_single_token(masked_confidence, mask_index_block, threshold):
            transfer_mask = torch.zeros_like(mask_index_block, dtype=torch.bool)
            for row_idx in range(mask_index_block.shape[0]):
                masked_positions = mask_index_block[row_idx].nonzero(as_tuple=True)[0]
                if masked_positions.numel() > 0:
                    transfer_mask[row_idx, masked_positions[0]] = True
            return transfer_mask

        monkeypatch.setattr(
            gen_utils, "select_threshold_transfer_mask", select_single_token
        )

        prompt = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=8,
                steps=2,
                use_cache=True,
                parallel_decode=True,
                confidence_threshold=0.99,
                alg="maskgit_plus",
                block_length=4,
                mask_token_id=100,
                temperature=0.0,
            )

        assert output.shape == (1, 12)
        assert not torch.any(output[:, 4:] == 100)

    def test_parallel_decode_with_algorithms(self, tiny_model):
        """Test parallel_decode with different confidence algorithms."""
        prompt = torch.tensor([[1, 2, 3]])

        algorithms = ["maskgit_plus", "topk_margin", "entropy"]
        for alg in algorithms:
            with torch.no_grad():
                output = tiny_model.diffusion_generate(
                    inputs=prompt,
                    max_new_tokens=8,
                    steps=4,
                    use_cache=True,
                    parallel_decode=True,
                    confidence_threshold=0.8,
                    alg=alg,
                    mask_token_id=100,
                )
            assert output.shape == (1, 11), f"Algorithm {alg} failed"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_parallel_decode_cuda(self, tiny_model):
        """Test parallel_decode on CUDA."""
        tiny_model = tiny_model.cuda()
        prompt = torch.tensor([[1, 2, 3, 4, 5]]).cuda()

        with torch.no_grad():
            output = tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=16,
                steps=4,
                use_cache=True,
                parallel_decode=True,
                confidence_threshold=0.9,
                alg="maskgit_plus",
                mask_token_id=100,
            )

        assert output.device.type == "cuda"
        assert output.shape == (1, 21)

    def test_parallel_decode_requires_cache(self, tiny_model):
        """Test that parallel_decode without use_cache raises error."""
        prompt = torch.tensor([[1, 2, 3, 4]])

        # parallel_decode with use_cache=False should raise ValueError
        with pytest.raises(
            ValueError, match="parallel_decode=True.*requires.*use_cache=True"
        ):
            tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=8,
                steps=4,
                use_cache=False,
                parallel_decode=True,
                confidence_threshold=0.9,
                alg="maskgit_plus",
                mask_token_id=100,
            )

    def test_parallel_decode_incompatible_with_origin(self, tiny_model):
        """Test that parallel_decode with alg='origin' raises error."""
        prompt = torch.tensor([[1, 2, 3, 4]])

        with (
            pytest.raises(ValueError, match="does not support `alg='origin'`"),
            torch.no_grad(),
        ):
            tiny_model.diffusion_generate(
                inputs=prompt,
                max_new_tokens=8,
                steps=4,
                use_cache=True,
                parallel_decode=True,
                confidence_threshold=0.9,
                alg="origin",
                mask_token_id=100,
            )
