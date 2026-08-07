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
Base contracts for the forward-process layer.

``unturtle.processes`` owns **training-state construction**: turning a clean,
already-collated batch into the noised/corrupted state a diffusion objective
trains on.  It deliberately owns nothing else — Trainer orchestration, loss
computation, and generation all live elsewhere.

``ForwardProcess`` is a structural contract, not a universal tensor schema.
Masked-discrete diffusion, discrete flow matching, and continuous/latent
methods each produce different ``model_inputs`` / ``objective_inputs`` keys;
the protocol only fixes *how* a process is called and *what shape of container*
it returns.

This module must not import ``unturtle.diffusion``.  Schedulers are consumed
structurally (see ``AlphaSchedule``) so the dependency direction stays::

    processes   <-   diffusion trainer/config
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass
class ProcessOutput:
    """Result of applying a forward process to a clean batch.

    Attributes:
        model_inputs:     Tensors/metadata forwarded to the model.  Contains the
                          transformed inputs plus any pass-through fields the
                          model needs (``attention_mask``, ``position_ids``,
                          packed metadata, model-specific kwargs, …).
        objective_inputs: Process-specific supervision the objective/loss needs
                          but the model forward does not (e.g. clean ``labels``,
                          the corruption mask, sampled timesteps).
    """

    model_inputs: dict[str, Any] = field(default_factory=dict)
    objective_inputs: dict[str, Any] = field(default_factory=dict)


class AlphaSchedule(Protocol):
    """Structural view of the scheduler behavior a masked process needs.

    Satisfied by ``unturtle.diffusion.schedulers.BaseAlphaScheduler`` without
    the process layer importing it.  Implementations must be vectorized:
    ``alpha`` receives the full ``[B]`` timestep tensor in one call and must
    return a scalar or a matching ``[B]`` result — a shorter tensor would
    broadcast one row's masking rate across the batch, so it is rejected.
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor | float: ...


class ForwardProcess(Protocol):
    """Callable that maps a clean batch to a noised training state.

    Implementations do not need to subclass this; duck typing is sufficient.

    Mutability contract: ``__call__`` must not mutate the input batch or any of
    its tensors.  Post-training methods may need the clean batch after process
    application, and Trainer-side buffering makes hidden mutation very hard to
    debug.
    """

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> ProcessOutput: ...
