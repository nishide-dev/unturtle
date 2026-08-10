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

"""Latent-conditioned MDLM-DiT (#130 codec slice, PR-codec-1).

The DiLaDiff decoder-side conditioning, eq. (32):

    h <- h + ZeroConv(CrossAttention(ZeroConv(h); z))

Cross-attention layers are inserted BETWEEN the pretrained MDLM decoder's
self-attention blocks (paper main config: the first and the last inter-block
gap; "pointwise convolution" on a token sequence is a per-position Linear).
The cross-attention extracts information from the latent channel ONLY — the
paper found the symmetric eq. (31) design (attending over ``[h, z]``) pays
excessive attention to the latent and degrades the pretrained decoder's use
of clean context.  Both wrapping convolutions are zero-initialized, so at
init the model is bitwise the pretrained decoder; gradients open the outer
conv first (zero queries still yield a uniform attention over V(z)), then
the interior.

``latents=None`` SKIPS the adapters — that is the plain-MDLM path, bitwise
stable across training.  The unconditional decoding MODE, by contrast, is
``latents = Gaussian noise`` (the paper's p_zdropout mechanism), which runs
the adapters.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from unturtle.models.backbones.mdlm_dit.configuration_mdlm_dit import MDLMDiTConfig
from unturtle.models.backbones.mdlm_dit.modeling_mdlm_dit import (
    MDLMDiTModel,
    MDLMDiTPreTrainedModel,
    _normalize_attention_mask,
)
from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationMixin,
)


class LaDiffDiTConfig(MDLMDiTConfig):
    model_type = "ladiff-dit"

    def __init__(
        self,
        num_latents: int = 512,
        latent_dim: int | None = None,
        latent_adapter_gaps: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_latents = num_latents
        # Default: the decoder hidden size (768 for mdlm-owt) — the paper's
        # per-position latent dimensionality is not recoverable from the
        # text, recorded as a choice on #130.
        self.latent_dim = latent_dim or self.hidden_size
        # Two adapters in the first and last inter-block gaps. This is an
        # Unturtle default (recorded on #130): the paper's MAIN config uses
        # ONE cross-attention layer ("from one to three" in the ablation);
        # its analysis section separately describes a two-layer first/last
        # variant, which is what this default interpolates.
        if latent_adapter_gaps is None:
            latent_adapter_gaps = (0, self.num_hidden_layers - 2)
        latent_adapter_gaps = tuple(latent_adapter_gaps)
        # Valid gaps are the INTER-block gaps 0..L-2 (a "gap" at L-1 would
        # sit after the final block, and any farther index would build an
        # adapter that never fires: trained-but-inert parameters saved to
        # every checkpoint). Duplicates would silently collapse in the
        # ModuleDict. At L<=2 the default itself degenerates — state gaps
        # explicitly there.
        valid = range(self.num_hidden_layers - 1)
        if len(set(latent_adapter_gaps)) != len(latent_adapter_gaps) or any(
            gap not in valid for gap in latent_adapter_gaps
        ):
            raise ValueError(
                f"latent_adapter_gaps must be unique inter-block gap indices "
                f"in [0, {self.num_hidden_layers - 2}], got {latent_adapter_gaps}"
            )
        self.latent_adapter_gaps = latent_adapter_gaps


class LatentCrossAttentionAdapter(nn.Module):
    """eq. (32) without the residual add (the caller owns ``h + ...``)."""

    def __init__(self, config: LaDiffDiTConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.conv_in = nn.Linear(hidden, hidden)
        self.cross_attn = nn.MultiheadAttention(
            hidden,
            config.num_attention_heads,
            kdim=config.latent_dim,
            vdim=config.latent_dim,
            batch_first=True,
        )
        self.conv_out = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, hidden: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        queries = self.conv_in(hidden)
        attended, _ = self.cross_attn(queries, latents, latents, need_weights=False)
        return self.conv_out(attended)


class LatentConditionedMDLMDiT(MDLMDiTPreTrainedModel, MaskedDiffusionGenerationMixin):
    """The pretrained MDLM-DiT trunk with eq.-(32) latent adapters.

    The trunk loop is re-run here (embed -> constant cond -> rotary ->
    blocks+adapters -> output head) with the SAME modules and op order as
    ``MDLMDiTModel.forward``; the init-bitwise-identity test pins this
    against drift.
    """

    config_class = LaDiffDiTConfig
    supports_block_decode = False  # DiT has no KV cache

    def __init__(self, config: LaDiffDiTConfig) -> None:
        super().__init__(config)
        self.model = MDLMDiTModel(config)
        self.latent_adapters = nn.ModuleDict(
            {
                str(gap): LatentCrossAttentionAdapter(config)
                for gap in config.latent_adapter_gaps
            }
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.vocab_embed

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.vocab_embed = value

    def get_output_embeddings(self) -> nn.Module:
        return self.model.output_layer.linear

    def freeze_for_autoencoder_training(self) -> None:
        """Paper C.1: the pretrained embedding table is frozen during AE
        training (stability); everything else stays trainable.  The paper's
        encoder/decoder warmup asymmetry is the training loop's job."""
        self.model.vocab_embed.embedding.requires_grad_(False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,  # absorbs timesteps / use_cache (time-agnostic)
    ) -> CausalLMOutputWithPast:
        attn_bias = _normalize_attention_mask(attention_mask, self.model.cond.dtype)
        batch, length = input_ids.shape
        x = self.model.vocab_embed(input_ids)
        c = F.silu(self.model.cond).unsqueeze(0).expand(batch, -1)
        cos, sin = self.model.rotary(length, input_ids.device)
        for i, block in enumerate(self.model.blocks):
            if self.model.gradient_checkpointing and self.training:
                x = self.model._gradient_checkpointing_func(
                    block.__call__, x, cos, sin, c, attn_bias
                )
            else:
                x = block(x, cos, sin, c, attn_bias)
            if latents is not None and str(i) in self.latent_adapters:
                x = x + self.latent_adapters[str(i)](x, latents)
        logits = self.model.output_layer(x, c)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)
