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
Model-specific access needed to skip the dense LM head (#68 PR C, for #61).

Masked-diffusion training computes ``[B, L, V]`` logits and then throws away
every unmasked position.  With a large vocabulary that wasted GEMM and its
activation memory are among the biggest Unturtle-specific training costs, so
#61 wants::

    hidden [B, L, H] -> gather [M, H] -> project [M, V] -> CE on M targets

Two pieces of that are model-specific: running the backbone *without* its
output head, and applying the output projection on its own.  This module
supplies both so the trainer never has to inspect a model hierarchy.

The implementation deliberately leans on standard ``transformers`` hooks
(``get_output_embeddings()``, the ``.model`` backbone) rather than
per-architecture attribute paths, so a family opting in usually needs only
the capability flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

SPARSE_OUTPUT_CAPABILITY = "sparse_output_projection"


@dataclass(frozen=True)
class SparseOutputAccess:
    """How to obtain hidden states and apply the output projection.

    Attributes:
        hidden_states: ``(model, **forward_kwargs) -> [B, L, H]``.  Runs the
                       backbone only; the LM head must not execute.
        project:       ``(model, hidden) -> logits``.  Applies the model's own
                       output embedding, so tied weights stay tied.
    """

    hidden_states: Callable[..., Any]
    project: Callable[[Any, Any], Any]


def _standard_hidden_states(model: Any, **forward_kwargs: Any) -> Any:
    """Run the backbone and return its last hidden state.

    ``model.model`` is the transformers convention for "everything below the
    output head", so this covers any family that follows it.
    """
    outputs = model.model(**forward_kwargs)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        # Some backbones return a plain tuple.
        hidden = outputs[0]
    return hidden


def _standard_project(model: Any, hidden: Any) -> Any:
    """Apply the model's own output embedding to already-gathered hidden states."""
    return model.get_output_embeddings()(hidden)


def standard_sparse_output(model: Any) -> SparseOutputAccess | None:
    """Access for a model following the standard transformers layout.

    Returns ``None`` when the model cannot support the sparse path — no output
    embedding to project with, or no ``.model`` backbone to run without the
    head — so callers fall back to the dense path rather than crashing.
    """
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if get_output_embeddings is None or get_output_embeddings() is None:
        return None
    if getattr(model, "model", None) is None:
        return None
    return SparseOutputAccess(
        hidden_states=_standard_hidden_states,
        project=_standard_project,
    )
