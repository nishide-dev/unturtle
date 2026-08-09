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

"""
Hybrid causal-bidirectional attention for the Tiny-A2D families (#63).

PreDiff-LM (arXiv:2607.25157 §3.2, eq. 3) keeps the prompt's causal pattern
while denoising the target bidirectionally, and blocks the corrupted target
from reaching prompt representations.  The reference mask lives in
``unturtle.utils.packing.build_hybrid_prefix_attention_mask``; this module is
only the wiring that lets a Tiny-A2D forward use it.

Why a config flag rather than a ``PreDiff*Model`` subclass: the difference is
one mask.  ``TinyA2D*Model.forward`` already replaces the causal mask with a
bidirectional one and already passes a caller-supplied 4-D mask through
untouched, and ``run_attention`` consumes such a mask verbatim (#89).  A
subclass would fragment fast-forward patching and the integration registry for
no semantic gain, and #63 explicitly asks for no new model family.

The flag is inert until ``prompt_lengths`` is supplied, so enabling it cannot
perturb an existing run.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from unturtle.utils.packing import build_hybrid_prefix_attention_mask


def maybe_build_hybrid_mask(
    config: Any,
    prompt_lengths: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    seq_len: int,
    key_value_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Return the eq-(3) mask, or ``None`` to leave the caller's path alone.

    Returns ``None`` — meaning "carry on as before" — when the model was not
    converted or the caller supplied no prompt boundary.  Both halves matter:
    a converted model without ``prompt_lengths`` has no prompt to preserve,
    and an unconverted model must ignore ``prompt_lengths`` entirely rather
    than silently changing its training semantics.

    Args:
        attention_mask: The caller's padding mask, 2-D or already 4-D.  A 2-D
                        mask is intersected with the hybrid topology so
                        padding stays excluded; anything else is left to the
                        caller, since a prebuilt 4-D mask is already a
                        complete specification.
    """
    if not getattr(config, "hybrid_attention", False) or prompt_lengths is None:
        return None

    if key_value_length != seq_len:
        # A KV cache makes attention rectangular ([q_len, kv_len]) while the
        # reference mask is square: equation (3) is defined over one sequence,
        # not over a query window against a longer key history.  Building the
        # square mask anyway would silently mis-align every row.
        #
        # Rejected rather than approximated because #63's scope is training,
        # and the masked-diffusion generation path never supplies
        # `prompt_lengths` — so this is only reachable by deliberately pairing
        # hybrid attention with incremental decoding, which needs a rectangular
        # formulation this slice does not define.
        raise ValueError(
            "hybrid_attention does not support a KV cache: attention is "
            f"rectangular (q_len={seq_len}, kv_len={key_value_length}) while "
            "the eq.-(3) mask is square. Pass use_cache=False for hybrid "
            "training, or omit prompt_lengths to fall back to bidirectional "
            "attention."
        )

    lengths = prompt_lengths.reshape(-1)
    if lengths.shape[0] != batch_size:
        raise ValueError(
            f"prompt_lengths has {lengths.shape[0]} entries but the batch has "
            f"{batch_size} rows; broadcasting one boundary across the batch "
            "would apply the wrong prompt split to every other row"
        )
    if bool(((lengths < 0) | (lengths > seq_len)).any()):
        raise ValueError(
            f"prompt_lengths must satisfy 0 <= p <= seq_len={seq_len}, got "
            f"{lengths.tolist()}"
        )

    padding = (
        attention_mask
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2
        else None
    )
    hybrid = build_hybrid_prefix_attention_mask(
        prompt_lengths=lengths,
        seq_len=seq_len,
        dtype=dtype,
        device=device,
        attention_mask=padding,
    )

    # A prebuilt 4-D mask carries topology the hybrid mask knows nothing
    # about — most importantly packed block-diagonal isolation.  Returning the
    # hybrid mask alone would *replace* it, letting packed samples attend
    # across their boundaries: attention still runs, the loss still decreases,
    # and cross-sample contamination is invisible.  So intersect instead.
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
        blocked = torch.finfo(dtype).min
        existing = attention_mask.to(device=device, dtype=dtype)
        hybrid = torch.where(
            (existing == 0) & (hybrid == 0),
            torch.zeros_like(hybrid),
            torch.full_like(hybrid, blocked),
        )

    return hybrid


_PACKED_KWARG_KEYS = ("packed_seq_lengths", "seq_lengths", "block_attention_mask")


def hybrid_fast_path_lengths(
    config: Any,
    seq_len: int,
    prompt_lengths: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    kwargs: dict,
) -> Optional[torch.Tensor]:
    """Return CPU ``[B]`` lengths when the mask-free split is exactly
    equivalent to the dense eq.-(3) mask, else ``None``.

    Consumed by the model forwards to decide whether to add
    ``hybrid_prompt_lengths`` to the kwargs travelling down the layer stack.
    The signal is advisory: the dense mask is still built and handed down, so
    an unpatched attention (or a future transformers version that filters
    unknown kwargs) simply keeps the dense path — the failure direction is
    speed, never semantics.

    Equivalence requires all of:

    - **No padding.**  An excluded key is excluded for every query, which
      breaks the pure row split.  ``attention_mask`` here is the caller's mask
      *before* the hybrid replacement — 2-D all-ones (or the all-ones tensor
      the forward substitutes for ``None``) qualifies; a 2-D mask with any
      zero, a prebuilt 4-D mask, or a ``BlockMask`` does not.
    - **No packed metadata.**  Packed isolation lives in a block mask the
      split cannot express; any of ``packed_seq_lengths`` / ``seq_lengths`` /
      ``block_attention_mask`` vetoes the signal.
    - **Uniform boundaries.**  The split point is a slice index.  Ragged
      boundaries are suppressed *here* rather than left to the kernel's own
      fallback, because that fallback rebuilds the dense mask per layer —
      once per forward is the whole budget.

    The returned tensor is moved to CPU once, so the kernel's per-layer
    uniformity check (`bool(...)` on the lengths) does not force a device
    sync on every layer.

    Range validation is deliberately absent: ``maybe_build_hybrid_mask``
    already rejected out-of-range boundaries before this runs, and the kernel
    validates again for direct callers.
    """
    # Below the measured crossover the two-call split is a net LOSS: the
    # extra kernel launch, `cat` and output transpose outweigh the attention
    # win (full forward 0.90x at L=1024 vs 1.50x at L=2048 on an 8-layer bf16
    # model).  The gate is a declared config field, not a buried constant, so
    # a caller on different hardware can move it -- in either direction the
    # dense mask is still built, so this only ever trades speed.
    if seq_len < getattr(config, "hybrid_fast_min_seq_len", 2048):
        return None

    if any(kwargs.get(key) is not None for key in _PACKED_KWARG_KEYS):
        return None

    # The ndim check states the contract; on realistic inputs it is
    # belt-and-braces over the padding check below (mutation-verified: an
    # additive 4-D mask uses 0 for "allowed", which is falsy, so any mask
    # admitting at least one position already fails `.all()`.  The only 4-D
    # input this line uniquely rejects is a degenerate all-blocked mask, and a
    # BlockMask would raise loudly on `.all()`).  Kept because "2-D all-ones
    # only" is the readable eligibility rule, not because a test can isolate
    # it.
    if not (isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2):
        return None
    if not bool(attention_mask.all()):
        return None

    lengths = prompt_lengths.reshape(-1).detach().to("cpu")
    if lengths.numel() and not bool((lengths == lengths[0]).all()):
        return None
    return lengths


def prompt_lengths_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """Derive the eq.-(3) prompt boundary from SFT-convention labels.

    The boundary is the **first supervised position** (``labels != -100``) of
    each row; a fully unsupervised row maps to the row length (all prompt — it
    contributes no loss, and ``p = L`` keeps it validly causal rather than
    silently flipping it to fully bidirectional).  Later ``-100`` holes inside
    the target are a labels concern, not a topology concern, so they do not
    move the boundary.

    Under right padding the leading ``-100`` run is exactly the prompt.  Under
    left padding the pad region lands on the prompt-causal side of the split,
    which is harmless: padding is excluded by the ``attention_mask``
    intersection in :func:`maybe_build_hybrid_mask` either way.
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D [batch, seq], got {labels.ndim}-D")
    supervised = labels != -100
    boundary = torch.argmax(supervised.long(), dim=1)
    return torch.where(
        supervised.any(dim=1),
        boundary,
        torch.full_like(boundary, labels.shape[1]),
    )


class HybridPromptCollator:
    """Ride the prompt boundary on batches from an existing collator.

    ``DiffusionTrainer`` ships every batch key to ``model(**inputs)``, so the
    only missing link for hybrid training is *who puts* ``prompt_lengths`` in
    the batch.  This wrapper is that link: purely additive (every base key
    passes through untouched), computed on the **padded** labels so the
    boundary is correct for whatever padding the base collator applied, and
    harmless on non-hybrid models, which ignore the key by contract.
    """

    def __init__(self, base_collator: Any) -> None:
        # Deferred import: `unturtle.diffusion` is a heavier package than this
        # wiring module and must not become an import-time dependency of the
        # model family.
        from unturtle.diffusion.packed_collator import (
            PackedMaskedDiffusionDataCollator,
        )

        if isinstance(base_collator, PackedMaskedDiffusionDataCollator):
            raise ValueError(
                "HybridPromptCollator does not support packed collators: a "
                "packed row holds several samples, so one per-row boundary "
                "applies sample A's prompt split to every sample in the row, "
                "and the packed block mask arrives as a kwarg the dense "
                "eq.-(3) intersection never sees (cross-sample attention). "
                "Wrapping would also hide the packed collator's type from "
                "DiffusionTrainer's own packed-collator guards. Use the "
                "unpacked MaskedDiffusionDataCollator for hybrid training."
            )
        self.base_collator = base_collator

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.base_collator(features)
        packed_keys = [key for key in _PACKED_KWARG_KEYS if key in batch]
        if packed_keys:
            raise ValueError(
                "HybridPromptCollator received a packed batch "
                f"(carries {packed_keys}); one per-row prompt boundary cannot "
                "express per-sample splits inside a packed row, so hybrid "
                "training requires unpacked batches"
            )
        labels = batch.get("labels")
        if labels is None:
            raise ValueError(
                "HybridPromptCollator needs `labels` in the collated batch to "
                "derive the prompt boundary; the base collator produced none"
            )
        batch["prompt_lengths"] = prompt_lengths_from_labels(labels)
        return batch


__all__ = [
    "HybridPromptCollator",
    "hybrid_fast_path_lengths",
    "maybe_build_hybrid_mask",
    "prompt_lengths_from_labels",
]
