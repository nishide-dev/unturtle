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

"""ModernBERT bidirectional fast attention forward.

ModernBERT is a *native* bidirectional diffusion backbone (not an AR→diffusion
conversion), so its Triton fast-path helpers live with the backbone, not in the
``conversion`` method packages.
"""

from __future__ import annotations

from typing import Optional

import torch


def ModernBertAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[tuple] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple:
    """Drop-in replacement for ``ModernBertAttention.forward``.

    Delegates entirely to the upstream implementation for QKV projection, RoPE,
    Flash/SDPA attention (including sliding_window, dropout, deterministic),
    and ``out_drop`` — then replaces only the final ``self.Wo(attn_output)``
    call with ``self.apply_wo(self, attn_output)`` so that the Triton-fused
    ``apply_lora_o`` kernel is used when patched in by
    ``modernbert.fast_paths.patch_peft`` (#185 provider).

    Upstream ``ModernBertAttention`` already sets ``self.is_causal = False``, so
    bidirectionality is preserved without any attention path changes.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.modernbert.modeling_modernbert import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]

    qkv = self.Wqkv(hidden_states)
    qkv = qkv.view(*input_shape, 3, -1, self.head_dim)
    query_states, key_states, value_states = qkv.unbind(dim=-3)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.head_dim**-0.5,
        sliding_window=self.sliding_window,
        deterministic=self.deterministic_flash_attn,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # Dispatch through apply_wo instead of self.Wo directly —
    # enables Triton apply_lora_o when patched by modernbert.fast_paths.patch_peft.
    attn_output = self.out_drop(self.apply_wo(self, attn_output))
    return attn_output, attn_weights


def _original_apply_wo(self, X: torch.Tensor) -> torch.Tensor:
    """Default stub: pass through self.Wo (the output projection)."""
    return self.Wo(X)


def _install_modernbert_stubs(model) -> None:
    """Set apply_wo stub on all ModernBertAttention layers that lack it.

    Required before modernbert.fast_paths.patch_peft or before using
    ModernBertAttention_fast_forward without PEFT.
    """
    for module in model.modules():
        if (
            hasattr(module, "Wqkv")
            and hasattr(module, "Wo")
            and not hasattr(module, "apply_wo")
        ):
            module.apply_wo = _original_apply_wo
