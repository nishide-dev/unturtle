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
# Block-decode generation mixin for dLLM models.
# Algorithm structure ported from Fast-dLLM: dev/repos/fast-dllm/dream/model/generation_utils_block.py

"""Block-decode generation mixin for dLLM models.

Provides shared block-decode algorithm that can be used by LLaDA, Dream, and other dLLM models.
Subclasses implement model-specific forward wrappers while inheriting common block-decode logic.
"""

import logging
import math
import time
import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from torch.nn import functional as F

from .diffusion_generation_utils import sample_tokens, select_threshold_transfer_mask

logger = logging.getLogger(__name__)


class BlockDecodeMixin:
    """Mixin providing block-decode generation for dLLM models.

    This mixin implements Fast-dLLM's block-decode algorithm:
    1. Divide generation into blocks of block_length tokens
    2. For each block:
       a. Initial forward: full sequence → build KV cache
       b. Cache handling: trim to previous blocks OR mark current block for replacement
       c. Denoising loop: iteratively refine current block with cached context
       d. Token sampling: unmask tokens based on timestep schedule

    Subclasses must implement:
    - _model_forward_with_cache(): Model-specific forward call with cache handling
    """

    def _block_decode_loop(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Common block-decode loop logic (Fast-dLLM style).

        Args:
            input_ids: Input token IDs [batch, prompt_len]
            attention_mask: Attention mask [batch, prompt_len] or None
            generation_config: MaskedDiffusionGenerationConfig with:
                - max_length: Total sequence length (prompt + generation)
                - block_length: Tokens per block (default: gen_length)
                - steps: Total denoising steps
                - mask_token_id: Token ID for [MASK]
                - use_replace_cache: If True, use replace_position mode (default: False)
                - alg: Algorithm ('origin', 'maskgit_plus', 'topk_margin', 'entropy')
                - temperature, top_p, top_k: Sampling parameters

        Returns:
            Generated sequence [batch, max_length]

        Raises:
            ValueError: If gen_length % block_length != 0
        """
        # Extract config (Fast-dLLM Dream style: use max_length, compute gen_length)
        max_length = generation_config.max_length
        block_length = getattr(generation_config, "block_length", None)
        steps = generation_config.steps
        mask_token_id = generation_config.mask_token_id
        if mask_token_id is None:
            mask_token_id = getattr(self.config, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "`mask_token_id` must be set in `generation_config` or `model.config` before calling "
                "`generate()` with `use_cache=True`."
            )
        use_replace_cache = getattr(generation_config, "use_replace_cache", False)
        alg = generation_config.alg
        alg_temp = getattr(generation_config, "alg_temp", None)
        temperature = getattr(generation_config, "temperature", 1.0)
        top_p = getattr(generation_config, "top_p", None)
        top_k = getattr(generation_config, "top_k", None)
        eps = getattr(generation_config, "eps", 0.001)

        if generation_config.parallel_decode and alg == "entropy":
            # Fast-dLLM's threshold mode uses max-probability confidence only
            # (dev/repos/fast-dllm/v1/dream/model/generation_utils_block.py L495-524).
            # Negative-entropy confidences are <= 0 and can never reach a
            # confidence_threshold in [0, 1], so threshold selection always takes
            # the single max-confidence fallback token.
            warnings.warn(
                "alg='entropy' uses negative-entropy confidences (<= 0), which never "
                "reach a confidence_threshold in [0, 1]; threshold-based parallel "
                "decode degenerates to one token per step (the max-confidence "
                "fallback). Use alg='maskgit_plus' or 'topk_margin' with "
                "parallel_decode, or disable parallel_decode for entropy ordering.",
                UserWarning,
            )

        output_timing = getattr(generation_config, "output_timing", False)
        timing: Optional[Dict[str, Any]] = None

        def _time_start() -> Optional[float]:
            if not output_timing:
                return None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return time.perf_counter()

        def _time_end(start: Optional[float], key: str) -> None:
            if start is None or timing is None:
                return
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timing[key] += time.perf_counter() - start

        # Setup (Fast-dLLM line 413: gen_length = max_length - input_ids.shape[1])
        device = input_ids.device
        _, prompt_len = input_ids.shape
        gen_length = max_length - prompt_len

        # Pad input_ids with [MASK] tokens (Fast-dLLM line 412)
        x = F.pad(input_ids, (0, gen_length), value=mask_token_id)

        # Extend attention_mask if provided
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = F.pad(attention_mask, (0, gen_length), value=1)

        # Default block_length: single block (original behavior)
        if block_length is None:
            block_length = gen_length

        # Validate divisibility
        if gen_length % block_length != 0:
            raise ValueError(
                f"gen_length ({gen_length}) must be divisible by block_length ({block_length}). "
                f"Use block_length that evenly divides max_new_tokens."
            )

        num_blocks = gen_length // block_length
        if steps < num_blocks:
            raise ValueError(f"steps ({steps}) must be >= num_blocks ({num_blocks}).")
        if steps % num_blocks != 0:
            raise ValueError(
                f"steps ({steps}) must be divisible by num_blocks ({num_blocks}) for block decode."
            )
        steps_per_block = steps // num_blocks

        if output_timing:
            timing = {
                "total_block_decode_s": 0.0,
                "initial_cache_build_s": 0.0,
                "cache_prep_s": 0.0,
                "denoise_forward_s": 0.0,
                "logits_slice_s": 0.0,
                "sampling_s": 0.0,
                "threshold_transfer_s": 0.0,
                "num_blocks": num_blocks,
                "steps_per_block": steps_per_block,
                "num_block_iterations": 0,
                "total_masked_tokens": 0,
                "parallel_decode": bool(generation_config.parallel_decode),
                "use_replace_cache": bool(use_replace_cache),
                "alg": alg,
            }

        # Timestep schedule (uniform per block)
        timesteps = torch.linspace(1.0, eps, steps_per_block + 1, device=device)

        total_start = _time_start()

        # Block-decode loop
        # #157 (b'): the block loop's own `step_idx` RESETS at every block, so
        # it cannot be the number a commit trajectory reports — [1,2,1,2] would
        # make two different iterations look like the same step. This counter is
        # global and monotonic across blocks. It counts only iterations that
        # committed token state: cache-construction forwards, trims and
        # refreshes commit nothing and are excluded.
        # Tolerant read: this shared loop is also reached by
        # DreamGenerationConfig, which defines its own fields and has no
        # `stream_callback`. A bare attribute access broke generation for a
        # backbone that never asked for tracing (10 existing tests).
        stream_callback = getattr(generation_config, "stream_callback", None)
        global_commit_step = 0

        for block_idx in range(num_blocks):
            current_block_start = prompt_len + block_idx * block_length
            current_block_end = current_block_start + block_length

            # Model-specific query start (constant per block). Dream's
            # right-shifted logits need position current_block_start - 1 in the
            # query window to predict the block's first token (its hook returns
            # current_block_start - 1); LLaDA / TinyA2D return
            # current_block_start, keeping their paths unchanged.
            query_start = self._get_block_decode_query_start(
                current_block_start=current_block_start,
                current_block_end=current_block_end,
                use_replace_cache=use_replace_cache,
            )

            # Step 1: Initial forward (full sequence) to build cache
            initial_cache_start = _time_start()
            outputs = self._model_forward_with_cache(
                input_ids=x,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                replace_position=None,
            )
            _time_end(initial_cache_start, "initial_cache_build_s")
            past_key_values = outputs.past_key_values

            # Step 2: Prepare cache for denoising
            cache_prep_start = _time_start()
            if use_replace_cache:
                # Dual cache mode: keep full cache, will use replace_position later
                cache_for_denoise = past_key_values
            else:
                # Trim mode: keep only the positions before the query window.
                # For models with query_start == current_block_start this keeps
                # exactly the previous blocks (Fast-dLLM non-dual trimming,
                # dev/repos/fast-dllm/v1/dream/model/generation_utils_block.py
                # L459-465); for Dream (query_start == current_block_start - 1)
                # the overlapping position is recomputed by the forward instead.
                from .cache_utils import trim_kv_cache

                if query_start > 0:
                    cache_for_denoise = trim_kv_cache(past_key_values, query_start)
                else:
                    cache_for_denoise = None
            _time_end(cache_prep_start, "cache_prep_s")

            # Step 3: Denoising loop for current block
            # Fast-dLLM threshold mode runs until the block is fully denoised.
            # We keep a hard safety cap of block_length iterations because the
            # threshold selector guarantees at least one newly unmasked token
            # per row per iteration.
            step_idx = 0
            max_block_iterations = (
                block_length if generation_config.parallel_decode else steps_per_block
            )
            while step_idx < max_block_iterations:
                # Check if block is fully denoised
                mask_index_block = (
                    x[:, current_block_start:current_block_end] == mask_token_id
                )
                n_masked = mask_index_block.sum().item()

                if n_masked == 0:
                    break  # Block complete, move to next

                if timing is not None:
                    timing["num_block_iterations"] += 1
                    timing["total_masked_tokens"] += n_masked

                # Prepare input for forward pass
                if use_replace_cache:
                    # Dual mode: forward the current block while replacing the
                    # corresponding absolute positions in the full cache.
                    x_forward = x[:, query_start:current_block_end]
                    if attention_mask is not None and attention_mask.ndim >= 3:
                        attn_forward = attention_mask[
                            :, :, query_start:current_block_end, :
                        ]
                    else:
                        attn_forward = attention_mask

                    # Mark current block positions for replacement
                    replace_position = torch.zeros_like(x, dtype=torch.bool)
                    replace_position[:, current_block_start:current_block_end] = True
                else:
                    # Trim mode: forward from the model's query start onwards
                    # (query_start == current_block_start for LLaDA/TinyA2D;
                    # current_block_start - 1 for Dream's right-shifted logits).
                    x_forward = x[:, query_start:]
                    if attention_mask is not None and attention_mask.ndim >= 3:
                        attn_forward = attention_mask[:, :, query_start:, :]
                    else:
                        # 2-D padding masks describe the KEYS, which span the full
                        # sequence here (trimmed cache prefix + suffix). Slicing off
                        # the prefix would drop exactly the prompt-padding info and
                        # desync the mask from the KV length — pass it through whole.
                        attn_forward = attention_mask
                    replace_position = None

                # Forward pass with cache
                denoise_forward_start = _time_start()
                outputs = self._model_forward_with_cache(
                    input_ids=x_forward,
                    attention_mask=attn_forward,
                    past_key_values=cache_for_denoise,
                    use_cache=True,
                    replace_position=replace_position,
                )
                _time_end(denoise_forward_start, "denoise_forward_s")

                logits_slice_start = _time_start()
                logits = cast(Any, self)._postprocess_block_decode_logits(
                    outputs.logits
                )

                # Extract logits for current block
                if use_replace_cache:
                    # Dual-cache forwards may return only the current block logits,
                    # or a block-local window with one-token left context.
                    if logits.shape[1] <= current_block_end:
                        block_logits = logits[:, -block_length:, :]
                    else:
                        block_logits = logits[
                            :, current_block_start:current_block_end, :
                        ]
                else:
                    # Incremental forward: the window starts at query_start, so
                    # the current block begins at offset
                    # current_block_start - query_start (0 for LLaDA/TinyA2D;
                    # 1 for Dream, whose right-shift postprocess moves the
                    # position current_block_start - 1 prediction onto the
                    # block's first token).
                    offset = current_block_start - query_start
                    block_logits = logits[:, offset : offset + block_length, :]

                # Get masked positions' logits
                mask_logits = block_logits[mask_index_block]  # [N_masked, vocab_size]
                if 0 <= mask_token_id < mask_logits.shape[-1]:
                    # Masked-diffusion denoising places zero mass on the mask token
                    # (MDLM SUBS "zero masking probabilities"). Without this, a
                    # committed token can be the mask sentinel itself, so the block
                    # never completes and mask tokens leak into the returned output.
                    mask_logits = mask_logits.clone()
                    mask_logits[:, mask_token_id] = torch.finfo(mask_logits.dtype).min
                _time_end(logits_slice_start, "logits_slice_s")

                # Sample tokens with confidence (alg selects the confidence
                # measure, matching the no-cache `_sample` dispatch).
                sampling_start = _time_start()
                confidence_type_by_alg = {
                    "origin": "max_prob",
                    "maskgit_plus": "max_prob",
                    "topk_margin": "margin",
                    "entropy": "neg_entropy",
                }
                if alg not in confidence_type_by_alg:
                    raise RuntimeError(
                        f"Unknown alg: {alg!r}. Choose from 'origin', 'maskgit_plus', 'topk_margin', 'entropy'."
                    )
                sampled_confidence, sampled = sample_tokens(
                    mask_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    confidence_type=confidence_type_by_alg[alg],
                )
                _time_end(sampling_start, "sampling_s")

                transfer_start = _time_start()
                if generation_config.parallel_decode:
                    current_block = x[:, current_block_start:current_block_end]
                    masked_confidence = torch.zeros(
                        current_block.shape,
                        dtype=logits.dtype,
                        device=current_block.device,
                    )
                    masked_confidence[mask_index_block] = sampled_confidence

                    transfer_index = select_threshold_transfer_mask(
                        masked_confidence=masked_confidence,
                        mask_index_block=mask_index_block,
                        threshold=generation_config.confidence_threshold,
                    )

                    selected_mask = mask_index_block & transfer_index
                    if selected_mask.any():
                        current_block[selected_mask] = sampled[
                            selected_mask[mask_index_block]
                        ]
                elif alg == "origin":
                    # Sequential decoding (origin algorithm): random transfer.
                    # Timestep transition
                    t = timesteps[step_idx]
                    s = timesteps[step_idx + 1]

                    # p_transfer: probability of unmasking at this step
                    p_transfer = 1.0 - s / t if step_idx < steps_per_block - 1 else 1.0

                    # Apply transfer probability (unmask some tokens)
                    transfer_mask = (
                        torch.rand(sampled.shape, device=device) < p_transfer
                    )

                    # Build updated block
                    x0_block = x[:, current_block_start:current_block_end].clone()
                    x0_flat = x0_block[mask_index_block]  # Masked positions only
                    x0_flat[transfer_mask] = sampled[
                        transfer_mask
                    ]  # Unmask transferred tokens
                    x0_block[mask_index_block] = x0_flat

                    # Update sequence
                    x[:, current_block_start:current_block_end] = x0_block
                else:
                    # Confidence-ordered transfer: keep the schedule-driven
                    # per-step count but pick positions by top confidence,
                    # matching the no-cache `_sample` semantics
                    # (diffusion_generation_utils.py, non-origin branch) and
                    # Fast-dLLM's non-threshold ordering
                    # (dev/repos/fast-dllm/v1/dream/model/generation_utils_block.py
                    # L526-559: topk / alg_temp-multinomial over confidence).
                    t = timesteps[step_idx]
                    s = timesteps[step_idx + 1]
                    num_mask_token = mask_index_block.sum() / mask_index_block.shape[0]
                    n_transfer = (
                        int(num_mask_token * (1 - s / t))
                        if step_idx < steps_per_block - 1
                        else int(num_mask_token)
                    )

                    if n_transfer > 0:
                        full_confidence = torch.full_like(
                            x[:, current_block_start:current_block_end],
                            -torch.inf,
                            dtype=logits.dtype,
                        )
                        full_confidence[mask_index_block] = sampled_confidence

                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(full_confidence, n_transfer)
                        else:
                            full_confidence = full_confidence / alg_temp
                            full_confidence = F.softmax(full_confidence, dim=-1)
                            transfer_index = torch.multinomial(
                                full_confidence, num_samples=n_transfer
                            )

                        sampled_block = torch.full_like(
                            x[:, current_block_start:current_block_end],
                            mask_token_id,
                            dtype=torch.long,
                        )
                        sampled_block[mask_index_block] = sampled
                        row_idx = (
                            torch.arange(x.size(0), device=device)
                            .unsqueeze(1)
                            .expand_as(transfer_index)
                        )
                        x[:, current_block_start:current_block_end][
                            row_idx, transfer_index
                        ] = sampled_block[row_idx, transfer_index]

                _time_end(transfer_start, "threshold_transfer_s")
                step_idx += 1

                # One invocation per denoising iteration that updated token
                # state, numbered globally (#157 (b')). No new API: this reads
                # only `generation_config.stream_callback`.
                global_commit_step += 1
                if stream_callback is not None:
                    try:
                        stream_callback(global_commit_step, steps, x.detach().clone())
                    except Exception as _cb_exc:  # noqa: BLE001
                        logger.warning(
                            "stream_callback raised at global step %d: %s",
                            global_commit_step,
                            _cb_exc,
                        )

                # Note: Cache is NOT updated during denoising loop
                # (Fast-dLLM: cache stays constant within block)

        _time_end(total_start, "total_block_decode_s")
        if timing is not None:
            return x, timing
        return x

    def _get_block_decode_query_start(
        self,
        current_block_start: int,
        current_block_end: int,
        use_replace_cache: bool,
    ) -> int:
        """Model-specific query start for block-decode forwards."""
        return current_block_start

    def _postprocess_block_decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Model-specific logit postprocessing before block slicing."""
        return logits

    def _model_forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Any],
        use_cache: bool,
        replace_position: Optional[torch.Tensor] = None,
    ):
        """Model-specific forward with cache.

        Subclasses must implement this to handle:
        - Model-specific forward signature
        - Cache format conversion (if needed)
        - replace_position handling (if supported)

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] or None
            past_key_values: KV cache (model-specific format)
            use_cache: Whether to return cache
            replace_position: Bool tensor [batch, seq_len] for dual-cache mode (optional)

        Returns:
            Model output with .past_key_values and .logits attributes

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _model_forward_with_cache(). "
            f"This method should call self.model() or self() with appropriate cache handling."
        )


__all__ = ["BlockDecodeMixin"]
