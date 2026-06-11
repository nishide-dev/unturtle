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

"""Generation utilities for LLaDA models.

LLaDA-specific overrides of the shared MDLM generation kernel.

Key difference from the shared mixin:
- ``attention_mask`` must stay **2-D** ``[B, L]`` (HF-style padding mask).
  The shared mixin expands it to 4-D for SDPA; LLaDA's own ``LLaDAModel``
  expects a flat 2-D mask and performs the expansion itself at line 1329 of
  ``modeling_llada.py`` via ``.view(batch_size, -1)``.  Pre-expanding to 4-D
  would break that path.

Phase M.1: Block-decode support via BlockDecodeMixin.
"""

from typing import Any, Optional, Union

import torch
import torch.nn.functional as F

from unturtle.models.generation.block_decode_mixin import BlockDecodeMixin
from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
    MaskedDiffusionGenerationMixin,
    MaskedDiffusionModelOutput,
    sample_tokens,
)


class LLaDAGenerationConfig(MaskedDiffusionGenerationConfig):
    """Generation config for LLaDA models (currently identical to the shared config)."""

    pass


class LLaDAGenerationMixin(BlockDecodeMixin, MaskedDiffusionGenerationMixin):
    """Generation mixin for LLaDA models with block-decode support.

    Phase M.1: Inherits BlockDecodeMixin for Fast-dLLM style block-decode KV cache.

    Overrides :meth:`_sample` to keep ``attention_mask`` as a 2-D HF-style
    padding mask — ``LLaDAModel`` expects this shape and performs its own
    internal expansion.  Everything else delegates to the shared mixin.
    """

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: MaskedDiffusionGenerationConfig,
    ) -> Union[MaskedDiffusionModelOutput, torch.LongTensor]:
        """MDLM denoising loop for LLaDA.

        Phase M.1: Delegates to block-decode path when use_cache=True.

        Identical to :meth:`MaskedDiffusionGenerationMixin._sample` except that
        ``attention_mask`` is **not** expanded to 4-D.  LLaDA's internal
        ``LLaDAModel.forward`` calls ``.view(batch_size, -1)`` on the mask at
        line 1329 of ``modeling_llada.py`` and then adds it as an additive bias,
        so it must receive a 2-D ``[B, L]`` tensor.
        """
        # Phase M.1: Delegate to block-decode when use_cache=True
        if generation_config.use_cache:
            return self._sample_with_cache(input_ids, attention_mask, generation_config)
        output_history = generation_config.output_history
        return_dict_out = generation_config.return_dict
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        if mask_token_id is None:
            mask_token_id = getattr(self.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` must be set in `generation_config` or `model.config` before calling "
                "`diffusion_generate()`.  Pass it explicitly: "
                "`model.diffusion_generate(inputs, mask_token_id=<id>, ...)`"
            )

        histories = [] if (return_dict_out and output_history) else None

        # Pad completion region with mask tokens.
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        # Keep attention_mask as 2-D [B, L] — LLaDAModel handles its own expansion.
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(
                attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
            )
            # 2-D only — do NOT broadcast to 4-D here.
        else:
            attention_mask = None

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        for i in range(steps):
            mask_index = x == mask_token_id

            # Forward pass — no logit shift (contrast with DreamGenerationMixin).
            logits = self(
                input_ids=x, attention_mask=attention_mask
            ).logits  # [B, L, V]

            mask_logits = logits[mask_index]  # [N_masked, V]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1.0
                x0 = torch.full_like(x[mask_index], mask_token_id, dtype=torch.long)
                transfer = torch.rand(*x0.shape, device=x.device) < p_transfer
                _, sampled = sample_tokens(
                    mask_logits[transfer],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                x0[transfer] = sampled
                x[mask_index] = x0
            else:
                if alg == "maskgit_plus":
                    confidence, x0 = sample_tokens(
                        mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                    )
                elif alg == "topk_margin":
                    confidence, x0 = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        margin_confidence=True,
                    )
                elif alg == "entropy":
                    confidence, x0 = sample_tokens(
                        mask_logits,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        neg_entropy=True,
                    )
                else:
                    raise RuntimeError(
                        f"Unknown alg: {alg!r}. Choose from 'origin', 'maskgit_plus', 'topk_margin', 'entropy'."
                    )

                num_mask_token = mask_index.sum() / mask_index.shape[0]
                n_transfer = (
                    int(num_mask_token * (1 - s / t))
                    if i < steps - 1
                    else int(num_mask_token)
                )
                full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype)
                full_confidence[mask_index] = confidence

                if n_transfer > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_confidence, n_transfer)
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(
                            full_confidence, num_samples=n_transfer
                        )

                    x_ = torch.full_like(x, mask_token_id, dtype=torch.long)
                    x_[mask_index] = x0
                    row_idx = (
                        torch.arange(x.size(0), device=x.device)
                        .unsqueeze(1)
                        .expand_as(transfer_index)
                    )
                    x[row_idx, transfer_index] = x_[row_idx, transfer_index]

            if histories is not None:
                histories.append(x.clone())

        if return_dict_out:
            return MaskedDiffusionModelOutput(
                sequences=x,
                history=tuple(histories) if histories is not None else None,
            )
        return x

    def _sample_with_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: MaskedDiffusionGenerationConfig,
    ) -> Union[MaskedDiffusionModelOutput, torch.LongTensor]:
        """Block-decode generation with KV cache for LLaDA.

        Note: attention_mask stays 2-D for LLaDA (see _sample docstring).
        """
        block_decode_output = self._block_decode_loop(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        timing = None
        if isinstance(block_decode_output, tuple):
            x, timing = block_decode_output
        else:
            x = block_decode_output

        # Wrap output according to generation_config.return_dict
        if generation_config.return_dict:
            return MaskedDiffusionModelOutput(sequences=x, history=None, timing=timing)
        return x

    def _model_forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Any],
        use_cache: bool,
        replace_position: Optional[torch.Tensor] = None,
    ):
        """LLaDA-specific forward with cache.

        Phase M.1: Implements BlockDecodeMixin's abstract method.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: 2-D attention mask [batch, seq_len] (LLaDA requires 2-D)
            past_key_values: KV cache (tuple format or DynamicCache)
            use_cache: Whether to return cache
            replace_position: Bool tensor [batch, seq_len] for dual-cache mode

        Returns:
            Object with .past_key_values and .logits attributes
        """
        # LLaDAModelLM expects self.model to be LLaDAModel
        # Forward signature: forward(input_ids, attention_mask, past_key_values, use_cache, replace_position)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            replace_position=replace_position,
        )

        # BlockDecodeMixin expects .past_key_values attribute, but LLaDAOutput has .attn_key_values
        # Create a simple object with the expected interface
        class OutputAdapter:
            def __init__(self, logits, past_key_values):
                self.logits = logits
                self.past_key_values = past_key_values

        # Convert list to tuple for cache_utils compatibility
        cache = tuple(out.attn_key_values) if out.attn_key_values is not None else None
        return OutputAdapter(logits=out.logits, past_key_values=cache)


__all__ = [
    "LLaDAGenerationConfig",
    "LLaDAGenerationMixin",
    "MaskedDiffusionModelOutput",
]
