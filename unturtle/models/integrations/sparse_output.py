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


def _resolve_backbone(model: Any) -> Any | None:
    """The module that produces hidden states, below the output head.

    ``get_decoder()`` rather than ``model.model``: on a ``PeftModel``,
    ``.model`` is the *LM-head model*, not the backbone
    (``PeftModel.model -> TinyA2DLlamaLMHeadModel``), so running it executes
    the head and returns logits — the exact cost this capability exists to
    avoid, and shaped just like hidden states.  ``get_decoder()`` unwraps
    correctly through PEFT and is the transformers-standard accessor.
    """
    get_decoder = getattr(model, "get_decoder", None)
    if get_decoder is not None:
        try:
            backbone = get_decoder()
        except (AttributeError, NotImplementedError):
            backbone = None
        if backbone is not None and backbone is not model:
            return backbone
    return None


def _standard_hidden_states(model: Any, **forward_kwargs: Any) -> Any:
    """Run the backbone and return its last hidden state.

    Raises:
        TypeError: if the backbone does not return a ``last_hidden_state``.
            Deliberately loud: a module returning logits here would look like
            a valid hidden-state tensor and silently double-apply the output
            head, corrupting both loss and gradients.
    """
    backbone = _resolve_backbone(model)
    if backbone is None:
        raise TypeError(
            f"{type(model).__name__} exposes no decoder backbone; "
            "sparse output should not have been resolved for it."
        )
    outputs = backbone(**forward_kwargs)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise TypeError(
            f"{type(backbone).__name__} returned "
            f"{type(outputs).__name__} without `last_hidden_state`; the "
            "sparse-output path needs hidden states, and treating another "
            "tensor as hidden states would double-apply the output head."
        )
    return hidden


def _standard_project(model: Any, hidden: Any) -> Any:
    """Apply the model's own output embedding to already-gathered hidden states."""
    return model.get_output_embeddings()(hidden)


def standard_sparse_output(model: Any) -> SparseOutputAccess | None:
    """Access for a model whose backbone is reachable via ``get_decoder()``.

    Returns ``None`` when the model cannot support the sparse path — no output
    embedding to project with, or no separate backbone to run without the
    head — so callers fall back to the dense path rather than crashing.

    Note this checks that the pieces *exist*, not that the backbone returns
    hidden states; that is verified at call time (see
    :func:`_standard_hidden_states`).  A family whose ``get_decoder()``
    returns logits must therefore not be given this resolver just because the
    attributes are present.
    """
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if get_output_embeddings is None or get_output_embeddings() is None:
        return None
    if _resolve_backbone(model) is None:
        return None
    return SparseOutputAccess(
        hidden_states=_standard_hidden_states,
        project=_standard_project,
    )
