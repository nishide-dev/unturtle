# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Vendored from unsloth/utils/packing.py for issue #67.

"""Utilities for enabling packed (padding-free) batches across Unturtle."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable, Optional, Sequence, Tuple

import torch

try:
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalMask as _XFormersBlockMask,
    )
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalMask as _XFormersBidirectionalBlockMask,
    )
except Exception:
    try:
        from xformers.attn_bias import (
            BlockDiagonalCausalMask as _XFormersBlockMask,
        )
        from xformers.attn_bias import (
            BlockDiagonalMask as _XFormersBidirectionalBlockMask,
        )
    except Exception:
        _XFormersBlockMask = None
        _XFormersBidirectionalBlockMask = None

_XFORMERS_MASK_CACHE_MAXSIZE = 32
_XFORMERS_MASK_CACHE: OrderedDict[Tuple[Tuple[int, ...], int], Any] = OrderedDict()
_XFORMERS_BIDIRECTIONAL_MASK_CACHE: OrderedDict[Tuple[int, ...], Any] = OrderedDict()

# Cache per device for get_packed_info_from_kwargs to avoid repeated D2H sync across layers
_PACKED_INFO_CACHE: dict = {}

# Cache per device for build_sdpa_packed_attention_mask to avoid repeated D2H sync across layers
_SDPA_MASK_CACHE: dict = {}

# Cache per device for build_sdpa_packed_bidirectional_attention_mask
_SDPA_BIDIRECTIONAL_MASK_CACHE: dict = {}

logger = logging.getLogger(__name__)

# Cache per device for build_xformers_block_causal_mask to avoid repeated D2H sync across layers
_XFORMERS_BLOCK_MASK_CACHE: dict = {}

# Cache per device for build_xformers_block_bidirectional_mask
_XFORMERS_BIDIRECTIONAL_BLOCK_MASK_CACHE: dict = {}


def _window_cache_key(sliding_window: Optional[int]) -> int:
    if sliding_window is None or sliding_window <= 0:
        return 0
    return int(sliding_window)


def _get_cached_block_mask(
    lengths: Tuple[int, ...],
    sliding_window: Optional[int],
):
    if _XFormersBlockMask is None:
        return None

    window_key = _window_cache_key(sliding_window)
    cache_key = (lengths, window_key)
    cached = _XFORMERS_MASK_CACHE.get(cache_key)
    if cached is not None:
        _XFORMERS_MASK_CACHE.move_to_end(cache_key)
        return cached

    mask = _XFormersBlockMask.from_seqlens(list(lengths))
    if window_key and mask is not None and hasattr(mask, "make_local_attention"):
        mask = mask.make_local_attention(window_size=window_key)

    _XFORMERS_MASK_CACHE[cache_key] = mask
    if len(_XFORMERS_MASK_CACHE) > _XFORMERS_MASK_CACHE_MAXSIZE:
        _XFORMERS_MASK_CACHE.popitem(last=False)
    return mask


def _get_cached_bidirectional_block_mask(lengths: Tuple[int, ...]):
    if _XFormersBidirectionalBlockMask is None:
        return None

    cached = _XFORMERS_BIDIRECTIONAL_MASK_CACHE.get(lengths)
    if cached is not None:
        _XFORMERS_BIDIRECTIONAL_MASK_CACHE.move_to_end(lengths)
        return cached

    mask = _XFormersBidirectionalBlockMask.from_seqlens(list(lengths))

    _XFORMERS_BIDIRECTIONAL_MASK_CACHE[lengths] = mask
    if len(_XFORMERS_BIDIRECTIONAL_MASK_CACHE) > _XFORMERS_MASK_CACHE_MAXSIZE:
        _XFORMERS_BIDIRECTIONAL_MASK_CACHE.popitem(last=False)
    return mask


class _TrlPackingWarningFilter(logging.Filter):
    to_filter = (
        "attention implementation is not",
        "kernels-community",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(substring in message for substring in self.to_filter)


_TRL_FILTER_INSTALLED = False


def _ensure_trl_warning_filter():
    global _TRL_FILTER_INSTALLED
    if _TRL_FILTER_INSTALLED:
        return
    logging.getLogger("trl.trainer.sft_trainer").addFilter(_TrlPackingWarningFilter())
    _TRL_FILTER_INSTALLED = True


def mark_allow_overlength(module):
    """Mark a module hierarchy so padding-free batches can exceed max_seq_length."""
    if module is None:
        return
    if hasattr(module, "max_seq_length"):
        module._unsloth_allow_packed_overlength = True
    children = getattr(module, "children", None)
    if children is None:
        return
    for child in children():
        mark_allow_overlength(child)


def configure_sample_packing(config):
    """Mutate an ``SFTConfig`` so TRL prepares packed batches."""
    _ensure_trl_warning_filter()
    config.packing = True
    config.padding_free = True
    config.remove_unused_columns = False


def configure_padding_free(config):
    """Mutate an ``SFTConfig`` so TRL enables padding-free batching without packing."""
    _ensure_trl_warning_filter()
    config.padding_free = True
    config.remove_unused_columns = False


def enable_sample_packing(
    model,
    trainer,
    *,
    sequence_lengths_key: str = "seq_lengths",
) -> None:
    """Enable runtime support for packed batches on an existing trainer."""
    if model is None or trainer is None:
        raise ValueError("model and trainer must not be None")

    mark_allow_overlength(model)

    if hasattr(trainer, "args") and hasattr(trainer.args, "remove_unused_columns"):
        trainer.args.remove_unused_columns = False

    collator = getattr(trainer, "data_collator", None)
    if collator is None or not hasattr(collator, "torch_call"):
        return
    if getattr(collator, "_unsloth_packing_wrapped", False):
        return

    if hasattr(collator, "padding_free"):
        collator.padding_free = True
    if hasattr(collator, "return_position_ids"):
        collator.return_position_ids = True

    original_torch_call = collator.torch_call

    def torch_call_with_lengths(examples: Sequence[dict]):
        batch = original_torch_call(examples)
        if examples and isinstance(examples[0], dict):
            seq_lengths: list[int] = []
            for example in examples:
                lengths = example.get(sequence_lengths_key)
                if isinstance(lengths, Iterable):
                    seq_lengths.extend(int(length) for length in lengths)
            if not seq_lengths:
                for example in examples:
                    ids = example.get("input_ids")
                    if isinstance(ids, Iterable):
                        seq_lengths.append(len(ids))
            if seq_lengths:
                batch["packed_seq_lengths"] = torch.tensor(
                    seq_lengths, dtype=torch.int32
                )
                if "attention_mask" in batch:
                    batch.pop("attention_mask")
        return batch

    collator.torch_call = torch_call_with_lengths
    collator._unsloth_packing_wrapped = True


def enable_padding_free_metadata(model, trainer):
    """Inject seq-length metadata when padding-free batching is enabled without packing."""
    collator = getattr(trainer, "data_collator", None)
    if (
        collator is None
        or getattr(collator, "_unsloth_padding_free_lengths_wrapped", False)
        or not getattr(collator, "padding_free", False)
    ):
        return

    mark_allow_overlength(model)
    if hasattr(collator, "return_position_ids"):
        collator.return_position_ids = True
    if hasattr(trainer, "args") and hasattr(trainer.args, "remove_unused_columns"):
        trainer.args.remove_unused_columns = False

    original_torch_call = collator.torch_call

    def torch_call_with_padding_free_metadata(examples: Sequence[dict]):
        seq_lengths: list[int] = []
        if examples and isinstance(examples[0], dict):
            for example in examples:
                lengths = example.get("seq_lengths")
                if lengths is None:
                    ids = example.get("input_ids")
                    if ids is None:
                        continue
                    lengths = [len(ids)]
                    example["seq_lengths"] = lengths
                seq_lengths.extend(lengths)

        batch = original_torch_call(examples)
        if seq_lengths:
            batch["packed_seq_lengths"] = torch.tensor(
                seq_lengths,
                dtype=torch.int32,
            )
        return batch

    collator.torch_call = torch_call_with_padding_free_metadata
    collator._unsloth_padding_free_lengths_wrapped = True


def get_packed_info_from_kwargs(
    kwargs: dict,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Return packed sequence metadata expected by the attention kernels."""

    seq_lengths = kwargs.get("packed_seq_lengths")
    if seq_lengths is None:
        return None

    entry = _PACKED_INFO_CACHE.get(device)
    if entry is not None and entry["seq_lengths"] is seq_lengths:
        return entry["result"]

    lengths = seq_lengths.to(device=device, dtype=torch.int32, non_blocking=True)
    cu_seqlens = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=device)
    torch.cumsum(lengths, dim=0, dtype=torch.int32, out=cu_seqlens[1:])

    max_seqlen = int(lengths.max().item())
    result = (lengths, cu_seqlens, max_seqlen)
    _PACKED_INFO_CACHE[device] = {"seq_lengths": seq_lengths, "result": result}
    return result


def build_xformers_block_causal_mask(
    seq_info: Optional[Tuple[torch.Tensor, torch.Tensor, int]],
    *,
    sliding_window: Optional[int] = None,
    base_mask: Optional[Any] = None,
):
    if _XFormersBlockMask is None:
        return None
    if seq_info is not None:
        seq_lengths, _, _ = seq_info
        device = seq_lengths.device
        params = (sliding_window,)
        entry = _XFORMERS_BLOCK_MASK_CACHE.get(device)
        if (
            entry is not None
            and entry["seq_lengths"] is seq_lengths
            and entry["params"] == params
        ):
            return entry["mask"]

        lengths_tensor = seq_lengths.to("cpu", torch.int32)
        if lengths_tensor.numel() == 0:
            return None
        lengths = tuple(int(x) for x in lengths_tensor.tolist())
        mask = _get_cached_block_mask(lengths, sliding_window)

        _XFORMERS_BLOCK_MASK_CACHE[device] = {
            "seq_lengths": seq_lengths,
            "params": params,
            "mask": mask,
        }
    else:
        mask = base_mask

        if (
            sliding_window is not None
            and sliding_window > 0
            and mask is not None
            and hasattr(mask, "make_local_attention")
        ):
            mask = mask.make_local_attention(window_size=sliding_window)
    return mask


def build_xformers_block_bidirectional_mask(
    seq_info: Optional[Tuple[torch.Tensor, torch.Tensor, int]],
    *,
    sliding_window: Optional[int] = None,
):
    """Non-causal ``BlockDiagonalMask`` for packed *bidirectional* attention.

    Mirrors :func:`build_xformers_block_causal_mask` but every token in a
    packed sample attends to every other token of the same sample — the
    xformers-native equivalent of
    :func:`build_sdpa_packed_bidirectional_attention_mask` (O(T) bias instead
    of a dense O(T^2) additive mask).

    Returns ``None`` — callers must fall back to the SDPA dense mask — when:

      * xformers (or ``BlockDiagonalMask``) is unavailable,
      * ``seq_info`` is ``None`` or empty, or
      * ``sliding_window`` is set: ``BlockDiagonalMask.make_local_attention``
        returns a *causal* local bias (``BlockDiagonalCausalLocalAttentionMask``
        as of xformers 0.0.35), so the symmetric bidirectional band
        (``|q - k| < sliding_window``) cannot be represented as an xformers
        attn_bias.
    """
    if _XFormersBidirectionalBlockMask is None or seq_info is None:
        return None
    if sliding_window is not None and sliding_window > 0:
        return None

    seq_lengths, _, _ = seq_info
    device = seq_lengths.device
    entry = _XFORMERS_BIDIRECTIONAL_BLOCK_MASK_CACHE.get(device)
    if entry is not None and entry["seq_lengths"] is seq_lengths:
        return entry["mask"]

    lengths_tensor = seq_lengths.to("cpu", torch.int32)
    if lengths_tensor.numel() == 0:
        return None
    lengths = tuple(int(x) for x in lengths_tensor.tolist())
    mask = _get_cached_bidirectional_block_mask(lengths)

    _XFORMERS_BIDIRECTIONAL_BLOCK_MASK_CACHE[device] = {
        "seq_lengths": seq_lengths,
        "mask": mask,
    }
    return mask


def build_sdpa_packed_attention_mask(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    seq_lengths, _, _ = seq_info

    params = (dtype, sliding_window)
    entry = _SDPA_MASK_CACHE.get(device)
    if (
        entry is not None
        and entry["seq_lengths"] is seq_lengths
        and entry["params"] == params
    ):
        return entry["mask"]

    total_tokens = int(seq_lengths.sum().item())
    mask = torch.full(
        (total_tokens, total_tokens),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    offset = 0
    for length in seq_lengths.tolist():
        length = int(length)
        if length <= 0:
            continue
        block = torch.zeros((length, length), dtype=dtype, device=device)
        upper = torch.triu(
            torch.ones((length, length), device=device), diagonal=1
        ).bool()
        block = block.masked_fill(upper, float("-inf"))
        if (
            sliding_window is not None
            and sliding_window > 0
            and length > sliding_window
        ):
            idx = torch.arange(length, device=device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            window_mask = dist >= sliding_window
            block = block.masked_fill(window_mask, float("-inf"))
        mask[offset : offset + length, offset : offset + length] = block
        offset += length

    result = mask.unsqueeze(0).unsqueeze(0)
    _SDPA_MASK_CACHE[device] = {
        "seq_lengths": seq_lengths,
        "params": params,
        "mask": result,
    }
    return result


def build_sdpa_packed_bidirectional_attention_mask(
    seq_info: Tuple[torch.Tensor, torch.Tensor, int],
    *,
    dtype: torch.dtype,
    device: torch.device,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Block-diagonal *bidirectional* SDPA mask for packed dLLM attention.

    Same block-diagonal structure as :func:`build_sdpa_packed_attention_mask`
    but WITHOUT the causal (upper-triangular) fill inside each block — every
    token in a packed sample attends to every other token of the same sample.
    An optional ``sliding_window`` applies a symmetric band
    (``|q - k| < sliding_window``) inside each block.
    """
    seq_lengths, _, _ = seq_info

    params = (dtype, sliding_window)
    entry = _SDPA_BIDIRECTIONAL_MASK_CACHE.get(device)
    if (
        entry is not None
        and entry["seq_lengths"] is seq_lengths
        and entry["params"] == params
    ):
        return entry["mask"]

    total_tokens = int(seq_lengths.sum().item())
    mask = torch.full(
        (total_tokens, total_tokens),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    offset = 0
    for length in seq_lengths.tolist():
        length = int(length)
        if length <= 0:
            continue
        block = torch.zeros((length, length), dtype=dtype, device=device)
        if (
            sliding_window is not None
            and sliding_window > 0
            and length > sliding_window
        ):
            idx = torch.arange(length, device=device)
            dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
            block = block.masked_fill(dist >= sliding_window, float("-inf"))
        mask[offset : offset + length, offset : offset + length] = block
        offset += length

    result = mask.unsqueeze(0).unsqueeze(0)
    _SDPA_BIDIRECTIONAL_MASK_CACHE[device] = {
        "seq_lengths": seq_lengths,
        "params": params,
        "mask": result,
    }
    return result


def build_hybrid_prefix_attention_mask(
    prompt_lengths: torch.Tensor,
    seq_len: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hybrid causal-bidirectional SDPA mask for PreDiff-LM adaptation (#63).

    Implements equation (3) of arXiv:2607.25157 §3.2 over a prompt prefix of
    length ``Lp`` followed by a target region::

                | 1[j <= i]   i, j in prompt
        M_ij =  | 1           i in target, j in prompt
                | 1           i, j in target
                | 0           i in prompt, j in target

    The prompt keeps the causal pattern the AR weights were pretrained under;
    the target denoises bidirectionally with full access to the prompt; and
    the prompt is shielded from the corrupted target, which is the asymmetry
    that distinguishes this from uniform bidirectional adaptation.  Note the
    target -> prompt quadrant is unconditional, *not* ``1[j <= i]``: a target
    token sees the whole prompt.

    The prompt is a contiguous prefix, as in the paper.  Interleaved
    prompt/target segments have no defined semantics there and are not
    supported here.

    Distinct from the packed builders above: those are block-diagonal over
    concatenated samples and return ``[1, 1, T, T]``; this is per-row over a
    padded batch and returns ``[B, 1, L, L]``.  It is also the only mask here
    that is *partly* causal, so do not substitute it for either
    :func:`build_sdpa_packed_attention_mask` (fully causal) or
    :func:`build_sdpa_packed_bidirectional_attention_mask` (fully
    bidirectional).

    Args:
        prompt_lengths: ``[B]`` integer prompt prefix length per row.
        seq_len:        Padded sequence length ``L``.
        attention_mask: Optional ``[B, L]`` real-token mask.  Padding is
                        excluded on both axes — a padded position neither
                        attends nor is attended to.

    Returns:
        Additive ``[B, 1, L, L]`` mask: ``0`` where attention is allowed,
        ``-inf`` where it is blocked.
    """
    lengths = prompt_lengths.to(device=device, dtype=torch.int64).reshape(-1)
    if bool(((lengths < 0) | (lengths > seq_len)).any()):
        raise ValueError(
            f"prompt_lengths must satisfy 0 <= Lp <= seq_len={seq_len}, got "
            f"{lengths.tolist()}"
        )

    batch = lengths.shape[0]
    idx = torch.arange(seq_len, device=device)
    query = idx.view(1, seq_len, 1)  # i
    key = idx.view(1, 1, seq_len)  # j
    boundary = lengths.view(batch, 1, 1)

    query_is_prompt = query < boundary
    key_is_prompt = key < boundary

    # Equation (3), assembled as one boolean expression rather than four
    # quadrant writes: every (i, j) is covered by exactly one arm, so a
    # missing case would be a shape error rather than a silently unwritten
    # region.
    allowed = torch.where(
        query_is_prompt,
        key_is_prompt & (key <= query),  # causal inside the prompt; target blocked
        torch.ones_like(key_is_prompt),  # target sees all prompt and all target
    )

    if attention_mask is not None:
        real = attention_mask.to(device=device, dtype=torch.bool).reshape(
            batch, seq_len
        )
        allowed = allowed & real.view(batch, 1, seq_len) & real.view(batch, seq_len, 1)

        # A padded query is blocked on both axes, which would leave its whole
        # row masked -- and softmax over an all-blocked row is NaN, which then
        # spreads across the batch.  SDPA's MATH backend happens to guard
        # against this internally, so a forward pass looks clean and the
        # hazard stays hidden until some consumer computes attention itself
        # (verified: a manual `softmax(scores + mask)` does produce NaN).
        # Let padded queries attend to themselves instead: the row stays
        # finite, real keys stay closed to them, and their output is
        # discarded downstream anyway.
        self_attention = torch.eye(seq_len, dtype=torch.bool, device=device)
        allowed = allowed | (~real.view(batch, seq_len, 1) & self_attention)

    # `finfo.min`, not `-inf`, per the convention documented in
    # `modeling_mdlm_dit.py` / `modeling_llada.py`: SDPA can emit NaNs on
    # fully-masked query rows with `-inf`, and additive composition of two
    # `-inf` biases is undefined.  Padded rows are already kept non-empty
    # above; this is defense in depth and consistency with every other
    # masking site in the repo.
    mask = torch.zeros((batch, seq_len, seq_len), dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(1)


def _normalize_packed_lengths(
    seq_lengths: Any,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if seq_lengths is None:
        return None
    if isinstance(seq_lengths, torch.Tensor):
        lengths = seq_lengths.to(device=device, dtype=torch.int64)
    else:
        lengths = torch.tensor(seq_lengths, device=device, dtype=torch.int64)
    if lengths.ndim != 1:
        lengths = lengths.reshape(-1)
    if lengths.numel() == 0:
        return None
    return lengths


def mask_packed_sequence_boundaries(
    shift_labels: torch.Tensor,
    seq_lengths: Any,
    *,
    ignore_index: int = -100,
) -> bool:
    """Mark final token of every packed sample so CE ignores boundary predictions."""
    lengths = _normalize_packed_lengths(seq_lengths, device=shift_labels.device)
    if lengths is None:
        return False

    flat = shift_labels.reshape(-1)
    total_tokens = flat.shape[0]
    boundary_positions = torch.cumsum(lengths, dim=0) - 1
    valid = boundary_positions < total_tokens
    if not torch.all(valid):
        # Debug guard (#49): packed lengths silently summing past the flat
        # buffer means the collator metadata and the batch disagree.
        logger.warning(
            "mask_packed_sequence_boundaries: packed seq_lengths sum to %d but "
            "only %d tokens are available; boundaries past the buffer are ignored.",
            int(boundary_positions[-1].item()) + 1,
            total_tokens,
        )
        boundary_positions = boundary_positions[valid]
    if boundary_positions.numel() == 0:
        return False
    flat[boundary_positions] = ignore_index
    return True


def clear_packed_caches():
    """Release cached masks/metadata to free device memory."""
    _PACKED_INFO_CACHE.clear()
    _SDPA_MASK_CACHE.clear()
    _SDPA_BIDIRECTIONAL_MASK_CACHE.clear()
    _XFORMERS_BLOCK_MASK_CACHE.clear()
    _XFORMERS_BIDIRECTIONAL_BLOCK_MASK_CACHE.clear()
    _XFORMERS_MASK_CACHE.clear()
    _XFORMERS_BIDIRECTIONAL_MASK_CACHE.clear()


__all__ = [
    "configure_sample_packing",
    "configure_padding_free",
    "enable_sample_packing",
    "enable_padding_free_metadata",
    "mark_allow_overlength",
    "get_packed_info_from_kwargs",
    "build_xformers_block_causal_mask",
    "build_xformers_block_bidirectional_mask",
    "build_sdpa_packed_attention_mask",
    "build_sdpa_packed_bidirectional_attention_mask",
    "mask_packed_sequence_boundaries",
    "clear_packed_caches",
]
