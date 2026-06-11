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
#
# Block-wise KV-cache abstraction for dLLM inference.
# Duck-type compatible with transformers Cache API (update/get_seq_length/get_max_length)
# but does NOT inherit from transformers.Cache (intentional — incompatible lifecycle).

"""Block-wise KV-cache for diffusion language model inference.

This module provides :class:`BlockKVCache` which stores key-value activations
in block-wise chunks to enable efficient reuse during iterative denoising.

Usage::

    from unturtle.models.generation.cache import BlockKVCache

    cache = BlockKVCache(block_size=32, num_layers=24)

    # During generation
    for block_idx in range(num_blocks):
        outputs = model(
            input_ids=block_tokens,
            past_key_values=cache,
            use_cache=True,
        )
        cache = outputs.past_key_values
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch


class BlockKVCache:
    """Block-wise KV-cache for diffusion LM inference.

    Stores key-value activations in blocks to support future cache reuse optimization.

    .. warning::
       Phase L (baseline): Cache is reset after each denoising step. No performance
       improvement expected until Phase M implements confidence-aware parallel decoding
       with true block-wise reuse.

    Attributes:
        block_size (int): Number of tokens per cache block.
        num_layers (int): Number of transformer layers.
        key_cache (List[torch.Tensor]): Cached key states per layer.
        value_cache (List[torch.Tensor]): Cached value states per layer.
        seen_tokens (int): Total number of tokens processed.
        current_block (int): Index of the currently active block.

    Args:
        block_size (int): Tokens per block (e.g., 32).
        num_layers (int, optional): Number of layers. If None, layers are added lazily.
        dtype (torch.dtype, optional): Data type for cache tensors. Defaults to float32.
        device (torch.device, optional): Device for cache tensors. Defaults to CPU.

    Example::

        cache = BlockKVCache(block_size=32, num_layers=24)

        # In model forward
        outputs = model(
            input_ids[:, block_start:block_end],
            past_key_values=cache,
            use_cache=True,
        )
        cache = outputs.past_key_values  # Updated cache
    """

    def __init__(
        self,
        block_size: int = 32,
        num_layers: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.block_size = block_size
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        # Initialize cache storage
        self.key_cache: List[Optional[torch.Tensor]] = [None] * (num_layers or 0)
        self.value_cache: List[Optional[torch.Tensor]] = [None] * (num_layers or 0)

        self.seen_tokens = 0
        self.current_block = 0

    def __repr__(self):
        return (
            f"BlockKVCache(block_size={self.block_size}, "
            f"num_layers={len(self.key_cache)}, "
            f"seen_tokens={self.seen_tokens}, "
            f"current_block={self.current_block})"
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key/value states for a layer.

        Args:
            key_states: New key tensor of shape [batch, num_heads, seq_len, head_dim].
            value_states: New value tensor of shape [batch, num_heads, seq_len, head_dim].
            layer_idx: Index of the transformer layer (0-indexed).
            cache_kwargs: Optional cache-specific arguments (e.g., 'block_idx').

        Returns:
            Tuple of (updated_keys, updated_values) including past cached states.
        """
        # Lazily extend cache if needed
        if layer_idx >= len(self.key_cache):
            self._extend_to_layer(layer_idx)

        # For first layer, update token count
        if layer_idx == 0:
            self.seen_tokens += key_states.shape[2]

        # Concatenate with past cache if exists
        cached_key = self.key_cache[layer_idx]
        cached_value = self.value_cache[layer_idx]
        if cached_key is not None and cached_value is not None:
            key_states = torch.cat([cached_key, key_states], dim=2)
            value_states = torch.cat([cached_value, value_states], dim=2)

        # Store updated cache
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states

        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return the sequence length of cached states.

        Args:
            layer_idx: Layer index to query. Defaults to 0.

        Returns:
            Sequence length of the cached key/value states.
        """
        if layer_idx is None:
            layer_idx = 0
        if layer_idx >= len(self.key_cache):
            return 0
        cached = self.key_cache[layer_idx]
        if cached is None:
            return 0
        return cached.shape[2]

    def get_max_length(self) -> Optional[int]:
        """Return the maximum cache capacity (None = unlimited)."""
        return None

    def reset(self):
        """Clear all cached states."""
        self.key_cache = [None] * len(self.key_cache)
        self.value_cache = [None] * len(self.value_cache)
        self.seen_tokens = 0
        self.current_block = 0

    def reset_block(self, block_idx: int):
        """Reset cache to a specific block (for block-decode restart).

        Args:
            block_idx: Target block index. Cache will retain states up to
                       block_idx * block_size tokens.
        """
        target_len = block_idx * self.block_size
        for layer_idx in range(len(self.key_cache)):
            cached_key = self.key_cache[layer_idx]
            cached_value = self.value_cache[layer_idx]
            if cached_key is not None and cached_value is not None:
                self.key_cache[layer_idx] = cached_key[:, :, :target_len, :]
                self.value_cache[layer_idx] = cached_value[:, :, :target_len, :]
        self.seen_tokens = target_len
        self.current_block = block_idx

    def _extend_to_layer(self, layer_idx: int):
        """Extend cache lists to accommodate layer_idx."""
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

    def to(self, device: torch.device) -> "BlockKVCache":
        """Move all cached tensors to a device.

        Args:
            device: Target device (e.g., torch.device("cuda")).

        Returns:
            Self (for method chaining).
        """
        self.device = device
        for layer_idx in range(len(self.key_cache)):
            cached_key = self.key_cache[layer_idx]
            cached_value = self.value_cache[layer_idx]
            if cached_key is not None and cached_value is not None:
                self.key_cache[layer_idx] = cached_key.to(device)
                self.value_cache[layer_idx] = cached_value.to(device)
        return self
