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

"""MDLM-DiT configuration"""

from __future__ import annotations

from transformers import PretrainedConfig


class MDLMDiTConfig(PretrainedConfig):
    """Config for the MDLM-DiT native diffusion backbone.

    Time-agnostic adaLN-Zero Diffusion Transformer (kuleshov-group/mdlm DiT). Unturtle
    drops the sigma path entirely and conditions on a single learnable constant vector;
    this is functionally (not structurally) equivalent to kuleshov's
    ``time_conditioning=False``, which zeroes sigma but still runs ``TimestepEmbedder``.
    Field names are HF-standard so no ``@property`` mapping is needed.
    """

    model_type = "mdlm-dit"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        cond_dim: int = 128,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        max_position_embeddings: int = 1024,
        mask_token_id: int | None = None,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        tie_word_embeddings: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # mask_token_id is not a standard PretrainedConfig arg; set after super().
        self.mask_token_id = mask_token_id
        self.architectures = self.architectures or ["MDLMDiTForMaskedDiffusionLM"]
