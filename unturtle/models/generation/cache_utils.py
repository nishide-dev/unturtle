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
# Block-decode KV cache utilities for dLLM models.
# Ported from Fast-dLLM (NVIDIA): dev/repos/fast-dllm/llada/model/modeling_llada.py

"""Block-decode KV cache utilities for dLLM models.

This module provides utilities for managing KV cache in block-decode generation:
- Cache trimming: Keep only first N tokens (prefix cache mode)
- Cache replacement: Selectively update positions (dual cache mode, Fast-dLLM style)
- Format conversion: Tuple ↔ DynamicCache (transformers compatibility)
"""

from typing import Any, Optional, Tuple

import torch
from transformers.cache_utils import Cache, DynamicCache


def trim_kv_cache(
    past_key_values: Any,
    target_length: int,
) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    """Trim KV cache to retain only the first target_length tokens.

    This follows Fast-dLLM's prefix cache approach: after full-sequence forward,
    trim cache to previous blocks only, then forward current block with trimmed cache.

    Args:
        past_key_values: Either DynamicCache or tuple of (key, value) tuples per layer.
            Shape per tensor: [batch, num_heads, seq_len, head_dim]
        target_length: Number of tokens to retain in cache.

    Returns:
        Trimmed past_key_values as tuple format with seq_len = target_length.

    Raises:
        ValueError: If target_length is invalid.
        TypeError: If cache format is unexpected.

    Example:
        >>> cache = model(x, use_cache=True).past_key_values  # Full sequence
        >>> trimmed = trim_kv_cache(cache, prompt_len + block_idx * block_len)
        >>> # Now forward from current block with trimmed cache
    """
    if target_length < 0:
        raise ValueError(f"target_length must be non-negative, got {target_length}")

    if target_length == 0:
        raise ValueError("target_length=0 would result in empty cache")

    # Convert DynamicCache to tuple format if needed
    if not isinstance(past_key_values, tuple):
        if not hasattr(past_key_values, "layers"):
            raise TypeError(
                f"Expected cache object to have '.layers' attribute, but got {type(past_key_values).__name__}. "
                f"Supported cache types: DynamicCache (transformers) or raw tuple format."
            )

        new_past_key_values = []
        try:
            for layer in past_key_values.layers:
                key_trimmed = layer.keys[:, :, :target_length, :]
                value_trimmed = layer.values[:, :, :target_length, :]
                new_past_key_values.append((key_trimmed, value_trimmed))
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to access cache layer attributes. Cache type: {type(past_key_values).__name__}. "
                f"Expected 'keys' and 'values' attributes on each layer. Original error: {e}"
            ) from e

        return tuple(new_past_key_values)
    else:
        # Already tuple format
        if len(past_key_values) == 0:
            raise ValueError("Received empty tuple cache")

        new_past_key_values = []
        for layer_idx in range(len(past_key_values)):
            layer_cache = past_key_values[layer_idx]

            if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
                raise TypeError(
                    f"Layer {layer_idx}: expected tuple of (key, value), "
                    f"got {type(layer_cache).__name__} with length {len(layer_cache) if hasattr(layer_cache, '__len__') else 'N/A'}"
                )

            key, value = layer_cache
            if key.ndim == 4 and value.ndim == 4:
                key_trimmed = key[:, :, :target_length, :]
                value_trimmed = value[:, :, :target_length, :]
            elif key.ndim == 3 and value.ndim == 3:
                key_trimmed = key[:, :target_length, :]
                value_trimmed = value[:, :target_length, :]
            else:
                raise TypeError(
                    f"Layer {layer_idx}: expected 3D or 4D key/value tensors, got "
                    f"key.ndim={key.ndim}, value.ndim={value.ndim}"
                )
            new_past_key_values.append((key_trimmed, value_trimmed))

        return tuple(new_past_key_values)


def replace_kv_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    replace_position: torch.Tensor,
    layer_idx: int,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """Replace KV cache at specified positions (Fast-dLLM style).

    This implements Fast-dLLM's dual-cache mode: instead of trimming and concatenating,
    selectively replace cache positions in-place. This is more efficient for block-decode
    where we refine tokens iteratively within a block.

    Algorithm (Fast-dLLM llada/model/modeling_llada.py lines 732-748):
    1. For each batch element, find indices where replace_position[b] == 1
    2. Replace past_key[b, :, indices] with new_key[b, :, :len(indices)]
    3. Return modified cache (in-place update)

    Args:
        past_key_values: Tuple of (key, value) pairs per layer.
            Shape per tensor: [batch, num_heads, seq_len, head_dim]
        new_key: New key tensor [batch, num_heads, selected_len, head_dim]
        new_value: New value tensor [batch, num_heads, selected_len, head_dim]
        replace_position: Bool tensor [batch, seq_len] where 1 = replace, 0 = keep
        layer_idx: Layer index to update (0-indexed)

    Returns:
        Updated cache in same tuple format.

    Raises:
        IndexError: If layer_idx is out of range.
        ValueError: If shapes don't match.

    Example:
        >>> # Initial forward: build full cache
        >>> cache = model(x, use_cache=True).past_key_values
        >>> # Mark current block for replacement
        >>> replace_pos = torch.zeros(B, L, dtype=torch.bool)
        >>> replace_pos[:, block_start:block_end] = True
        >>> # Forward current block, get new K/V
        >>> new_k, new_v = ...
        >>> # Replace cache at marked positions
        >>> for layer_idx in range(num_layers):
        >>>     cache = replace_kv_cache(cache, new_k, new_v, replace_pos, layer_idx)
    """
    if layer_idx < 0 or layer_idx >= len(past_key_values):
        raise IndexError(
            f"layer_idx={layer_idx} out of range for cache with {len(past_key_values)} layers"
        )

    # Validate replace_position
    if replace_position.ndim != 2:
        raise ValueError(
            f"replace_position must be 2D [batch, seq_len], got shape {replace_position.shape}"
        )

    batch_size = replace_position.shape[0]

    # Extract current layer cache
    past_key, past_value = past_key_values[layer_idx]

    # Validate shapes
    if past_key.shape[0] != batch_size:
        raise ValueError(
            f"Batch size mismatch: replace_position has {batch_size}, "
            f"but past_key has {past_key.shape[0]}"
        )

    # Clone cache to avoid modifying input (side-effect free)
    past_key = past_key.clone()
    past_value = past_value.clone()

    # Perform replacement per batch (Fast-dLLM algorithm)
    for b in range(batch_size):
        # Get indices where replace_position[b] == 1
        batch_replace_indices = replace_position[b].nonzero(as_tuple=True)[0]

        if len(batch_replace_indices) > 0:
            # Validate new tensor has enough elements
            num_replace = len(batch_replace_indices)
            if new_key.shape[2] < num_replace:
                raise ValueError(
                    f"Batch {b}: replace_position marks {num_replace} positions, "
                    f"but new_key only has {new_key.shape[2]} tokens"
                )

            # Replace positions in past_key and past_value for this batch
            # past_key[b, :, indices] = new_key[b, :, :len(indices)]
            past_key[b, :, batch_replace_indices] = new_key[b, :, :num_replace]
            past_value[b, :, batch_replace_indices] = new_value[b, :, :num_replace]

    # Rebuild cache tuple with updated layer
    new_past_key_values = list(past_key_values)
    new_past_key_values[layer_idx] = (past_key, past_value)

    return tuple(new_past_key_values)


def tuple_to_cache(
    past_key_values: Tuple[Tuple[torch.Tensor, ...], ...],
    device: torch.device,
) -> DynamicCache:
    """Convert raw tuple cache to DynamicCache for transformers compatibility.

    Args:
        past_key_values: Tuple of (key, value) tuples per layer
        device: Target device for cache

    Returns:
        DynamicCache object

    Raises:
        TypeError: If input is not tuple format
        ValueError: If cache is empty
    """
    if not isinstance(past_key_values, tuple):
        raise TypeError(
            f"Expected tuple format, got {type(past_key_values).__name__}. "
            f"Use this function to convert tuple → DynamicCache for transformers models."
        )

    if len(past_key_values) == 0:
        raise ValueError("Cannot convert empty tuple to DynamicCache")

    cache = DynamicCache()

    for layer_idx, layer_cache in enumerate(past_key_values):
        if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
            raise TypeError(
                f"Layer {layer_idx}: expected (key, value) tuple, "
                f"got {type(layer_cache).__name__}"
            )

        key, value = layer_cache

        # Ensure tensors are on target device
        if key.device != device:
            key = key.to(device)
        if value.device != device:
            value = value.to(device)

        cache.update(key, value, layer_idx)

    return cache


__all__ = [
    "trim_kv_cache",
    "replace_kv_cache",
    "tuple_to_cache",
]
