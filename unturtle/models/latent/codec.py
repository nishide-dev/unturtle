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
Codec boundary for continuous/latent methods (#66).

Named ``Codec``, not ``LatentCodec`` — closing the RFC's open question: the
FlowLM instance below is just the embedding matrix plus a rounding head, and
"latent" would overstate it.  The surface is the RFC's, and the load-bearing
parts are the two the original ``encode()/decode()`` sketch lacked:

- ``trainable`` — LDLM trains its codec jointly, TextLDM/AURORA-LM freeze
  theirs; the trainer must be able to ask;
- ``auxiliary_losses() -> dict`` — **named** terms the codec owns.  Even the
  trivial instance here has one (FlowLM's rounding CE, Algorithm 1), which is
  the strongest argument that a bare protocol is insufficient: TextLDM's REPA
  term and LDLM's decoder loss would otherwise land in the trainer as method
  flags.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn


@runtime_checkable
class Codec(Protocol):
    """Token <-> continuous representation semantics, method-owned."""

    trainable: bool

    def encode(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...

    def decode(self, latents: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...

    def auxiliary_losses(
        self, latents: torch.Tensor, input_ids: torch.Tensor, **kwargs: Any
    ) -> dict[str, torch.Tensor]: ...


class EmbeddingRoundingCodec(nn.Module):
    """FlowLM's codec: embedding lookup in, rounding head out.

    ``decode`` returns logits (rounding is the caller's argmax) through a head
    weight-tied to the embedding, matching the DiffuSeq lineage.  The one
    auxiliary term is Algorithm 1's anchor loss ``CE(decoder_head(z_0), w)``
    over the CLEAN latents — it trains the embedding/head pair to stay
    round-trippable, which nothing in the x0 MSE enforces.
    """

    trainable = True

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def encode(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.embedding(input_ids)

    def decode(self, latents: torch.Tensor, **_: Any) -> torch.Tensor:
        return F.linear(latents, self.embedding.weight)

    def auxiliary_losses(
        self, latents: torch.Tensor, input_ids: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        logits = self.decode(latents)
        return {
            "rounding_ce": F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1)
            )
        }


__all__ = ["Codec", "EmbeddingRoundingCodec"]
