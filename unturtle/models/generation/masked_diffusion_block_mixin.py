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

"""Shared masked-diffusion block generation mixin (block-decode + MDLM + BD3LM).

MaskedDiffusionBlockGenerationMixin inherits both BlockDecodeMixin (Fast-dLLM KV-cache block-decode
for use_cache=True) and MaskedDiffusionGenerationMixin (standard MDLM generation).
BD3LM block-diffusion generation is added via _sample_block_diffusion()
(use_block_diffusion=True).
"""

import math
from types import SimpleNamespace

import torch
from transformers.utils import logging

from unturtle.models.generation.block_decode_mixin import BlockDecodeMixin
from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
    MaskedDiffusionGenerationMixin,
    MaskedDiffusionModelOutput,
)

logger = logging.get_logger(__name__)


class MaskedDiffusionBlockGenerationConfig(MaskedDiffusionGenerationConfig):
    """Generation config for Tiny-A2D models (currently identical to the shared config)."""

    pass


def _snapshot_prefix_cache(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, tuple):
        return past_key_values
    if hasattr(past_key_values, "layers"):
        return tuple((layer.keys, layer.values) for layer in past_key_values.layers)
    raise TypeError(
        f"Unsupported cache type for BD3LM prefix snapshot: {type(past_key_values).__name__}"
    )


def _rewrap_prefix_cache(past_key_values, device):
    if past_key_values is None:
        return None

    from unturtle.models.generation.cache_utils import tuple_to_cache

    if isinstance(past_key_values, tuple):
        return tuple_to_cache(past_key_values, device)
    return past_key_values


class MaskedDiffusionBlockGenerationMixin(
    BlockDecodeMixin, MaskedDiffusionGenerationMixin
):
    """Generation mixin for Tiny-A2D models.

    Inherits:
    - BlockDecodeMixin: Fast-dLLM style block-decode KV-cache generation
      (activated via ``use_cache=True``). Resolves #182 — Tiny-A2D now uses the
      same ``_block_decode_loop`` as LLaDA and Dream.
    - MaskedDiffusionGenerationMixin: standard MDLM no-cache generation loop.

    BD3LM block-diffusion generation (``use_block_diffusion=True``) is
    provided by :meth:`_sample_block_diffusion`.
    """

    def _model_forward_with_cache(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        use_cache: bool,
        replace_position=None,
    ):
        """Standard HF forward with cache — required by BlockDecodeMixin.

        ``replace_position`` is accepted for API compatibility but ignored:
        Tiny-A2D models (LLaMA/Qwen) do not implement dual-cache replace_position.

        Tuple caches (returned by ``trim_kv_cache``) are converted to
        ``DynamicCache`` before the forward call because HF LLaMA/Qwen
        models expect a cache object with ``.get_seq_length()``.
        """
        # Convert raw tuple cache to DynamicCache (HF LLaMA/Qwen requirement).
        # trim_kv_cache returns 2-tuples per layer; HF models need a DynamicCache
        # with .get_seq_length(), so we convert before the forward call.
        if isinstance(past_key_values, tuple) and len(past_key_values) > 0:
            from unturtle.models.generation.cache_utils import tuple_to_cache

            past_key_values = tuple_to_cache(past_key_values, input_ids.device)

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        # Return DynamicCache as-is so trim_kv_cache can use the .layers interface.
        # BlockDecodeMixin only stores and trims the cache — it never iterates it
        # as a tuple, so we can keep the native DynamicCache format here.
        cache = outputs.past_key_values
        return SimpleNamespace(logits=outputs.logits, past_key_values=cache)

    def _sample_with_cache(
        self,
        input_ids,
        attention_mask,
        generation_config: "MaskedDiffusionGenerationConfig",
    ):
        """Block-decode generation with KV cache for Tiny-A2D (delegates to BlockDecodeMixin).

        Resolves #182 — Tiny-A2D now uses the same ``_block_decode_loop`` as LLaDA
        and Dream instead of the previously duplicated ``_sample_with_cache``
        implementation in ``MaskedDiffusionGenerationMixin``.
        """
        if getattr(generation_config, "use_replace_cache", False):
            generation_config.use_replace_cache = False

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

        if generation_config.return_dict:
            return MaskedDiffusionModelOutput(sequences=x, history=None, timing=timing)
        return x

    @torch.no_grad()
    def _sample_block_diffusion(
        self,
        input_ids: torch.Tensor,
        generation_config: "MaskedDiffusionGenerationConfig",
    ) -> torch.Tensor:
        """BD3LM block-diffusion generation loop.

        Generates tokens block-by-block using a bidirectional block-causal
        attention mask (``prepare_for_sampling``).  This is the correct
        generation algorithm for Tiny-A2D models — causal AR models converted to
        bidirectional masked diffusion.

        Called by :meth:`MaskedDiffusionGenerationMixin._sample` when
        ``generation_config.use_block_diffusion=True``.
        """
        from unturtle.models.generation.diffusion_generation_utils import (
            _add_gumbel_noise,
            _diffusion_step_block,
            _get_num_transfer_tokens,
            prepare_for_sampling,
        )

        block_size = generation_config.bd3lm_block_size
        steps = generation_config.steps
        max_new_tokens = generation_config.max_new_tokens
        temperature = generation_config.temperature
        cfg_scale = generation_config.cfg_scale
        right_shift_logits = generation_config.right_shift_logits
        step_callback = generation_config.step_callback
        stream_callback = generation_config.stream_callback

        # Resolve special token IDs (config → model config fallback)
        mask_id = generation_config.mask_token_id
        if mask_id is None:
            mask_id = getattr(self.config, "mask_token_id", None)
        if mask_id is None:
            raise ValueError(
                "`mask_token_id` must be set in `generation_config` or `model.config` "
                "before calling `generate(algorithm='bd3lm')`."
            )

        pad_id = generation_config.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None:
            raise ValueError(
                "`pad_token_id` must be set in `generation_config` or `model.config` "
                "before calling `generate(algorithm='bd3lm')`."
            )

        eos_id = generation_config.eos_token_id
        if eos_id is None:
            eos_id = getattr(self.config, "eos_token_id", None)

        device = input_ids.device
        B, prompt_len = input_ids.shape

        if max_new_tokens is None:
            if generation_config.max_length is None:
                raise ValueError(
                    "`use_block_diffusion=True` requires `max_new_tokens` or `max_length`."
                )
            max_new_tokens = generation_config.max_length - prompt_len

        if max_new_tokens <= 0:
            raise ValueError(
                f"BD3LM generation requires positive generation length, got max_new_tokens={max_new_tokens}."
            )

        # Left-pad prompt to a multiple of block_size
        padded_prompt_len = math.ceil(prompt_len / block_size) * block_size
        prompt_left_pad = padded_prompt_len - prompt_len
        x = torch.full((B, padded_prompt_len), pad_id, dtype=torch.long, device=device)
        for b in range(B):
            offset = padded_prompt_len - prompt_len
            x[b, offset : offset + prompt_len] = input_ids[b]

        # Track "given" tokens for CFG unconditional branch
        unmasked_index = (x != mask_id) & (x != pad_id)

        done = torch.zeros(B, dtype=torch.bool, device=device)

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = max(1, math.ceil(steps / num_blocks))
        total_steps = num_blocks * steps_per_block

        generated = 0
        global_step = 0

        for _ in range(num_blocks):
            if done.all():
                break

            cur_block_len = min(block_size, max_new_tokens - generated)
            if cur_block_len <= 0:
                break

            T_prefix = x.shape[1]

            # Build block-causal attention mask + logical position IDs for prefix
            prefix_attn, prefix_pos = prepare_for_sampling(x, block_size, pad_id)

            # Try to obtain KV cache for prefix; fall back to full-seq silently
            cond_past = None
            cond_prefix_last_logits = None
            uncond_past = None
            uncond_prefix_last_logits = None

            try:
                out_prefix = self(
                    x,
                    attention_mask=prefix_attn,
                    position_ids=prefix_pos,
                    use_cache=True,
                )
                if (
                    hasattr(out_prefix, "past_key_values")
                    and out_prefix.past_key_values is not None
                ):
                    cond_past = _snapshot_prefix_cache(out_prefix.past_key_values)
                    cond_prefix_last_logits = out_prefix.logits[:, -1:, :]

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    out_un = self(
                        un_x,
                        attention_mask=prefix_attn,
                        position_ids=prefix_pos,
                        use_cache=True,
                    )
                    if (
                        hasattr(out_un, "past_key_values")
                        and out_un.past_key_values is not None
                    ):
                        uncond_past = _snapshot_prefix_cache(out_un.past_key_values)
                        uncond_prefix_last_logits = out_un.logits[:, -1:, :]
            except TypeError:
                cond_past = None

            # Append masked block
            new_block = torch.full(
                (B, cur_block_len), mask_id, dtype=torch.long, device=device
            )
            new_block[done] = pad_id
            x = torch.cat([x, new_block], dim=1)
            unmasked_index = torch.cat(
                [
                    unmasked_index,
                    torch.zeros(B, cur_block_len, dtype=torch.bool, device=device),
                ],
                dim=1,
            )

            T_total = x.shape[1]
            block_mask_index = x[:, T_prefix:T_total] == mask_id

            num_transfer_tokens = _get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )
            effective_steps = num_transfer_tokens.shape[1]

            full_attn, full_pos = prepare_for_sampling(x, block_size, pad_id)
            attn_block = full_attn[:, :, T_prefix:T_total, :]
            pos_block = full_pos[:, T_prefix:T_total]

            use_kv_cache = cond_past is not None

            for i_step in range(effective_steps):
                x_block = x[:, T_prefix:T_total]
                mask_block = x_block == mask_id

                if not mask_block.any():
                    break

                if use_kv_cache:
                    try:
                        cond_out = self(
                            x_block,
                            attention_mask=attn_block,
                            position_ids=pos_block,
                            past_key_values=_rewrap_prefix_cache(
                                cond_past, x_block.device
                            ),
                            use_cache=False,
                        )
                        cond_logits = cond_out.logits

                        if cfg_scale > 0.0 and uncond_past is not None:
                            uncond_out = self(
                                x_block,
                                attention_mask=attn_block,
                                position_ids=pos_block,
                                past_key_values=_rewrap_prefix_cache(
                                    uncond_past, x_block.device
                                ),
                                use_cache=False,
                            )
                            logits_block = uncond_out.logits + (cfg_scale + 1.0) * (
                                cond_logits - uncond_out.logits
                            )
                        else:
                            logits_block = cond_logits

                    except (TypeError, RuntimeError):
                        use_kv_cache = False
                        cond_past = None

                if not use_kv_cache:
                    full_attn_step, full_pos_step = prepare_for_sampling(
                        x, block_size, pad_id
                    )
                    try:
                        out = self(
                            x, attention_mask=full_attn_step, position_ids=full_pos_step
                        )
                    except TypeError:
                        out = self(x, attention_mask=full_attn_step)

                    if right_shift_logits and T_prefix > 0:
                        cond_prefix_last_logits = out.logits[
                            :, T_prefix - 1 : T_prefix, :
                        ]

                    logits_block = out.logits[:, T_prefix:T_total, :]

                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[unmasked_index] = mask_id
                        try:
                            un_out = self(
                                un_x,
                                attention_mask=full_attn_step,
                                position_ids=full_pos_step,
                            )
                        except TypeError:
                            un_out = self(un_x, attention_mask=full_attn_step)
                        un_logits = un_out.logits[:, T_prefix:T_total, :]
                        if right_shift_logits and T_prefix > 0:
                            uncond_prefix_last_logits = un_out.logits[
                                :, T_prefix - 1 : T_prefix, :
                            ]
                        logits_block = un_logits + (cfg_scale + 1.0) * (
                            logits_block - un_logits
                        )

                # Dream-compat right-shift
                if right_shift_logits and cond_prefix_last_logits is not None:
                    if cfg_scale > 0.0 and uncond_prefix_last_logits is not None:
                        prefix_last = uncond_prefix_last_logits + (cfg_scale + 1.0) * (
                            cond_prefix_last_logits - uncond_prefix_last_logits
                        )
                    else:
                        prefix_last = cond_prefix_last_logits
                    shifted = torch.empty_like(logits_block)
                    shifted[:, 0:1, :] = prefix_last
                    shifted[:, 1:, :] = logits_block[:, :-1, :]
                    logits_block = shifted

                x_block_updated = _diffusion_step_block(
                    logits=logits_block,
                    x_block=x_block,
                    mask_block=mask_block,
                    num_transfer_step=num_transfer_tokens[:, i_step],
                    temperature=temperature,
                )
                x[:, T_prefix:T_total] = x_block_updated
                global_step += 1

                if stream_callback is not None:
                    try:
                        stream_callback(
                            global_step,
                            total_steps,
                            x[:, prompt_left_pad:].detach().clone(),
                        )
                    except Exception as _cb_exc:
                        logger.warning(
                            "stream_callback raised at step %d: %s",
                            global_step,
                            _cb_exc,
                        )

                if step_callback is not None:
                    try:
                        step_callback(global_step, total_steps)
                    except Exception as _cb_exc:
                        logger.warning(
                            "step_callback raised at step %d: %s", global_step, _cb_exc
                        )

            if eos_id is not None:
                eos_in_block = (x[:, T_prefix:T_total] == eos_id).any(dim=1)
                done = done | eos_in_block

            generated += cur_block_len

        if prompt_left_pad > 0:
            x = x[:, prompt_left_pad:]

        return x


__all__ = [
    "MaskedDiffusionBlockGenerationConfig",
    "MaskedDiffusionBlockGenerationMixin",
    "MaskedDiffusionModelOutput",
]
