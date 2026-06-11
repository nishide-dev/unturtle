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

import pytest
import torch

from unturtle.models.generation.cache import BlockKVCache


class TestBlockKVCache:
    """Test BlockKVCache abstraction."""

    def test_init(self):
        """Test cache initialization."""
        cache = BlockKVCache(block_size=32, num_layers=12)
        assert cache.block_size == 32
        assert cache.num_layers == 12
        assert len(cache.key_cache) == 12
        assert len(cache.value_cache) == 12
        assert cache.seen_tokens == 0
        assert cache.current_block == 0

    def test_update_first_layer(self):
        """Test updating cache for first layer."""
        cache = BlockKVCache(block_size=32, num_layers=2)
        batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64

        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        updated_key, updated_value = cache.update(key, value, layer_idx=0)

        assert updated_key.shape == (batch_size, num_heads, seq_len, head_dim)
        assert updated_value.shape == (batch_size, num_heads, seq_len, head_dim)
        assert cache.seen_tokens == seq_len
        assert cache.get_seq_length(0) == seq_len

    def test_update_concatenates_past(self):
        """Test that update concatenates with past cache."""
        cache = BlockKVCache(block_size=32, num_layers=2)
        batch_size, num_heads, head_dim = 2, 8, 64

        # First update
        key1 = torch.randn(batch_size, num_heads, 16, head_dim)
        value1 = torch.randn(batch_size, num_heads, 16, head_dim)
        cache.update(key1, value1, layer_idx=0)

        # Second update
        key2 = torch.randn(batch_size, num_heads, 16, head_dim)
        value2 = torch.randn(batch_size, num_heads, 16, head_dim)
        updated_key, updated_value = cache.update(key2, value2, layer_idx=0)

        assert updated_key.shape == (batch_size, num_heads, 32, head_dim)
        assert updated_value.shape == (batch_size, num_heads, 32, head_dim)
        assert cache.seen_tokens == 32
        assert cache.get_seq_length(0) == 32

    def test_update_multiple_layers(self):
        """Test updating multiple layers independently."""
        cache = BlockKVCache(block_size=32, num_layers=3)
        batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64

        for layer_idx in range(3):
            key = torch.randn(batch_size, num_heads, seq_len, head_dim)
            value = torch.randn(batch_size, num_heads, seq_len, head_dim)
            cache.update(key, value, layer_idx=layer_idx)

        # Only first layer should update seen_tokens
        assert cache.seen_tokens == seq_len

        # All layers should have cached states
        for layer_idx in range(3):
            assert cache.get_seq_length(layer_idx) == seq_len

    def test_reset(self):
        """Test cache reset."""
        cache = BlockKVCache(block_size=32, num_layers=2)
        batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64

        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(key, value, layer_idx=0)

        assert cache.seen_tokens > 0
        cache.reset()

        assert cache.seen_tokens == 0
        assert cache.current_block == 0
        assert cache.get_seq_length(0) == 0

    def test_reset_block(self):
        """Test resetting cache to a specific block."""
        cache = BlockKVCache(block_size=16, num_layers=2)
        batch_size, num_heads, head_dim = 2, 8, 64

        # Fill cache with 48 tokens (3 blocks)
        for _ in range(3):
            key = torch.randn(batch_size, num_heads, 16, head_dim)
            value = torch.randn(batch_size, num_heads, 16, head_dim)
            cache.update(key, value, layer_idx=0)

        assert cache.get_seq_length(0) == 48

        # Reset to block 1 (keep first 16 tokens)
        cache.reset_block(1)
        assert cache.get_seq_length(0) == 16
        assert cache.seen_tokens == 16
        assert cache.current_block == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_to_device(self):
        """Test moving cache to CUDA."""
        cache = BlockKVCache(block_size=32, num_layers=2)
        batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64

        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        cache.update(key, value, layer_idx=0)

        cache.to(torch.device("cuda"))
        assert cache.device.type == "cuda"
        assert cache.key_cache[0].device.type == "cuda"
        assert cache.value_cache[0].device.type == "cuda"

    def test_lazy_layer_extension(self):
        """Test that cache extends lazily when accessing new layers."""
        cache = BlockKVCache(block_size=32)  # num_layers=None
        assert len(cache.key_cache) == 0

        batch_size, num_heads, seq_len, head_dim = 2, 8, 16, 64
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Update layer 5 (should extend to 6 layers)
        cache.update(key, value, layer_idx=5)
        assert len(cache.key_cache) == 6
        assert cache.get_seq_length(5) == seq_len

    def test_get_max_length(self):
        """Test that max_length returns None (unlimited)."""
        cache = BlockKVCache(block_size=32, num_layers=2)
        assert cache.get_max_length() is None
