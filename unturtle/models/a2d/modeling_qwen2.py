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
# Ported from zhziszz/dllm (dllm/pipelines/a2d/models/qwen2/modeling_qwen2.py).
# Removed __main__ block (dllm.utils dependency). Model code is unchanged.

"""A2D (AutoRegressive→Diffusion) adapter for Qwen2 models.

Converts a causal Qwen2 model to a bidirectional masked diffusion LM by
replacing the causal attention mask (including sliding-window layers) with
a padding-only attention mask. All pretrained weights are reused as-is.

Usage::

    from unturtle.models.a2d import A2DQwen2Config, A2DQwen2LMHeadModel

    config = A2DQwen2Config.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = A2DQwen2LMHeadModel(config)
"""

from typing import Optional

import torch
from torch import nn

import transformers
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.processing_utils import Unpack

from .generation_utils import A2DGenerationMixin
from transformers.utils import TransformersKwargs

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor


class A2DQwen2Config(transformers.Qwen2Config):
    model_type = "a2d-qwen2"


class A2DQwen2Model(transformers.Qwen2Model):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Bidirectional (padding-only) mask — replaces the upstream causal mask.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )

            if not (
                isinstance(attention_mask, BlockMask)
                or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
            ):
                attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)

            causal_mask_mapping = {"full_attention": attention_mask}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class A2DQwen2LMHeadModel(A2DGenerationMixin, transformers.Qwen2ForCausalLM):
    config: A2DQwen2Config

    def __init__(self, config):
        transformers.Qwen2PreTrainedModel.__init__(self, config)
        self.model = A2DQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


transformers.AutoConfig.register("a2d-qwen2", A2DQwen2Config)
transformers.AutoModel.register(A2DQwen2Config, A2DQwen2LMHeadModel)
transformers.AutoModelForMaskedLM.register(A2DQwen2Config, A2DQwen2LMHeadModel)
