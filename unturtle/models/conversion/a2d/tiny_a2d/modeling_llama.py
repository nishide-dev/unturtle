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
# Ported from zhziszz/dllm (dllm/pipelines/a2d/models/llama/modeling_llama.py).
# Removed __main__ block (dllm.utils dependency). Model code is unchanged.

"""Tiny-A2D (AutoRegressive→Diffusion) adapter for LLaMA models.

Converts a causal LLaMA model to a bidirectional masked diffusion LM by
replacing the causal attention mask with a padding-only attention mask.
All pretrained weights are reused without modification.

Usage::

    from unturtle.models.conversion.a2d.tiny_a2d import TinyA2DLlamaConfig, TinyA2DLlamaLMHeadModel

    config = TinyA2DLlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = TinyA2DLlamaLMHeadModel(config)
    # fine-tune with DiffusionTrainer
"""

from typing import Optional

import torch
import transformers
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ._hybrid import maybe_build_hybrid_mask
from .generation_utils import TinyA2DGenerationMixin

if transformers.utils.is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor


class TinyA2DLlamaConfig(transformers.LlamaConfig):
    model_type = "tiny-a2d-llama"

    def __init__(self, hybrid_attention: bool = False, **kwargs):
        """`hybrid_attention` is declared rather than left to **kwargs (#63).

        `PretrainedConfig` would store it either way, but an undeclared field
        is invisible to anyone reading the class and depends on upstream
        kwarg-stashing behaviour.
        """
        super().__init__(**kwargs)
        self.hybrid_attention = hybrid_attention


class TinyA2DLlamaModel(transformers.LlamaModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        prompt_lengths: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Bidirectional (padding-only) mask — replaces the upstream causal mask.
        # With cache, the key/value length includes both cached prefix and current tokens.
        key_value_length = past_seen_tokens + inputs_embeds.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], key_value_length),
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        elif (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.shape[1] == inputs_embeds.shape[1]
            and past_seen_tokens > 0
        ):
            prefix_mask = torch.ones(
                (attention_mask.shape[0], past_seen_tokens),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # PreDiff-LM hybrid topology (#63).  Returns None unless the model was
        # converted *and* a prompt boundary was supplied, so an unconverted
        # model ignores `prompt_lengths` and a converted one without it keeps
        # its previous behaviour byte-for-byte.
        hybrid = maybe_build_hybrid_mask(
            self.config,
            prompt_lengths,
            attention_mask,
            batch_size=inputs_embeds.shape[0],
            seq_len=inputs_embeds.shape[1],
            key_value_length=key_value_length,
            dtype=self.dtype,
            device=inputs_embeds.device,
        )
        if hybrid is not None:
            attention_mask = hybrid

        # 2) Convert 2-D padding mask to 4-D additive attention bias.
        if not (
            isinstance(attention_mask, BlockMask)
            or (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4)
        ):
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                self.dtype,
                tgt_len=inputs_embeds.shape[1],
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class TinyA2DLlamaLMHeadModel(TinyA2DGenerationMixin, transformers.LlamaForCausalLM):
    config: TinyA2DLlamaConfig

    def __init__(self, config):
        transformers.LlamaPreTrainedModel.__init__(self, config)
        self.model = TinyA2DLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


transformers.AutoConfig.register("tiny-a2d-llama", TinyA2DLlamaConfig)
transformers.AutoModel.register(TinyA2DLlamaConfig, TinyA2DLlamaLMHeadModel)
transformers.AutoModelForMaskedLM.register(TinyA2DLlamaConfig, TinyA2DLlamaLMHeadModel)
