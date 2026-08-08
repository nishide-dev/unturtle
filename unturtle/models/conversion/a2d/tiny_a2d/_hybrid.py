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
    return build_hybrid_prefix_attention_mask(
        prompt_lengths=lengths,
        seq_len=seq_len,
        dtype=dtype,
        device=device,
        attention_mask=padding,
    )


__all__ = ["maybe_build_hybrid_mask"]
