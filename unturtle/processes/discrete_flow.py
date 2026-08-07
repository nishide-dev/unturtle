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
Discrete flow-matching forward process (#65).

Implements the conditional probability path of FS-DFM (arXiv:2509.20624 §3)::

    p_t(x^i | x_0, x_1) = (1 - kappa_t) * delta_{x_0}(x^i) + kappa_t * delta_{x_1}(x^i)

Each position independently holds either its **source** token ``x_0`` or its
**target** token ``x_1``, mixed by ``kappa_t``.  The paper uses the linear
scheduler ``kappa(t) = t`` and says so explicitly, to keep the study on
step-aware training rather than scheduler design; other monotone schedulers are
valid DFM and can be injected.

Two source distributions, both from the paper: an all-``[MASK]`` source, and a
**uniform** source over the vocabulary (which the released checkpoints use).

**This is not a masked-diffusion variant, and the difference is load-bearing.**
Masked corruption is absorbing — a corrupted position holds the mask token, so
``diffusion_mask`` fully describes the state and the clean targets can be
recovered from ``labels`` alone.  A uniform-source flow is not absorbing: a
corrupted position holds an ordinary token, indistinguishable by inspection
from an uncorrupted one.  The process therefore emits the source state
``source_ids``, which masked diffusion has no analogue for.  Expressing this as
a mode flag on ``MaskedDiffusionProcess`` would mean threading a tensor that is
meaningless for half its configurations.

Reimplemented from the paper.  ``apple/ml-fs-dfm`` is under the Apple Sample
Code License and was deliberately not read or ported;
``facebookresearch/flow_matching`` is CC BY-NC and likewise only a conceptual
reference.

Reference:
    FS-DFM  https://arxiv.org/abs/2509.20624
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from .base import ProcessOutput

# Owned by the process and therefore rebuilt rather than passed through.
_SUPERVISION_KEYS = ("labels", "source_ids", "timesteps")

# Packing topology the process consumes but the model forward does not.
_TOPOLOGY_KEYS = ("segment_ids",)

_SOURCES = ("mask", "uniform")


class KappaSchedule(Protocol):
    """Structural view of a DFM path scheduler.

    ``kappa(t)`` must be monotone with ``kappa(0) = 0`` and ``kappa(1) = 1``.
    Vectorized: it receives the whole timestep tensor and must return a
    matching shape (or a scalar).
    """

    def kappa(self, t: torch.Tensor) -> torch.Tensor | float: ...


@dataclass
class LinearKappa:
    """``kappa(t) = t`` — the paper's choice (§5)."""

    def kappa(self, t: torch.Tensor) -> torch.Tensor:
        return t


@dataclass
class DiscreteFlowProcess:
    """Discrete flow-matching corruption of a clean, already-collated batch.

    Args:
        vocab_size:    Size of the token vocabulary, for the uniform source.
        mask_token_id: Id of ``[MASK]``.  Required for ``source="mask"``;
                       optional otherwise.
        source:        ``"mask"`` (all-mask source) or ``"uniform"`` (uniform
                       over the vocabulary).
        scheduler:     Path scheduler.  Defaults to ``kappa(t) = t``.
        time_epsilon:  Lower bound on the sampled timestep.
    """

    vocab_size: int
    mask_token_id: int | None = None
    source: str = "mask"
    scheduler: KappaSchedule = field(default_factory=LinearKappa)
    time_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError(
                f"Unknown source {self.source!r}. Choose from: {_SOURCES}."
            )
        if self.source == "mask" and self.mask_token_id is None:
            raise ValueError("source='mask' requires mask_token_id")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if not 0.0 <= self.time_epsilon < 1.0:
            raise ValueError(
                f"time_epsilon must satisfy 0 <= eps < 1, got {self.time_epsilon}"
            )

    def kappa(self, t: torch.Tensor) -> torch.Tensor:
        """Path weight at ``t``, normalized to a tensor matching ``t``."""
        value = torch.as_tensor(self.scheduler.kappa(t), device=t.device, dtype=t.dtype)
        if value.dim() == 0:
            return value.expand(t.shape)
        if value.shape != t.shape:
            raise ValueError(
                f"scheduler.kappa(t) must return a scalar or a {tuple(t.shape)} "
                f"tensor, got {tuple(value.shape)}"
            )
        return value

    def _sample_source(
        self,
        input_ids: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if self.source == "mask":
            return torch.full_like(input_ids, int(self.mask_token_id))  # type: ignore[arg-type]
        return torch.randint(
            0,
            self.vocab_size,
            input_ids.shape,
            device=input_ids.device,
            dtype=input_ids.dtype,
            generator=generator,
        )

    def _sample_timesteps(
        self,
        batch: dict[str, Any],
        shape: tuple[int, int],
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """One ``t`` per row, or per packed segment when ``segment_ids`` is given."""
        B, L = shape
        segment_ids = batch.get("segment_ids")

        def draw(size: tuple[int, ...]) -> torch.Tensor:
            raw = torch.rand(size, device=device, generator=generator)
            return self.time_epsilon + (1.0 - self.time_epsilon) * raw

        if segment_ids is None:
            return draw((B,))

        if segment_ids.shape != (B, L):
            raise ValueError(
                f"segment_ids must have shape {(B, L)} to match input_ids, "
                f"got {tuple(segment_ids.shape)}"
            )
        # Padding carries a negative id that no sample owns.  `gather` cannot
        # take it — a raw -1 raises on CPU and fires an async device-side
        # assert on CUDA that poisons the context (see #62) — so those
        # positions borrow segment 0's draw and are zeroed afterwards.
        # `clamp_min(0)` rather than another -1 remap: the borrowed value is
        # overwritten by the `torch.where` below, so `abs()` is byte-identical
        # here (mutation-verified). Clamping is chosen because it keeps padding
        # on an id that certainly exists — `abs()` maps -1 to segment 1, which
        # need not.
        owned = segment_ids >= 0
        n_segments = int(segment_ids.max()) + 1 if bool(owned.any()) else 1
        per_segment = draw((B, max(n_segments, 1)))
        t = torch.gather(per_segment, 1, segment_ids.clamp_min(0).to(torch.long))
        return torch.where(owned, t, torch.zeros_like(t))

    def __call__(
        self,
        batch: dict[str, Any],
        *,
        timesteps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ProcessOutput:
        """Interpolate a clean batch toward its source distribution.

        Args:
            batch:     Must contain ``input_ids`` ``[B, L]``.  ``attention_mask``,
                       ``labels``, ``segment_ids`` and model metadata are
                       optional.
            timesteps: Explicit ``t``, ``[B]`` or ``[B, L]``.  Sampled when
                       omitted.
            generator: Optional RNG; equal seeds reproduce the state exactly.

        Returns:
            A :class:`~.base.ProcessOutput` whose ``model_inputs`` carry the
            interpolated ``input_ids`` plus every pass-through field, and whose
            ``objective_inputs`` carry ``labels`` (clean targets, ``-100``
            where unsupervised), ``source_ids`` (the sampled ``x_0``) and
            ``timesteps``.

        The input batch and its tensors are never mutated.  Pass-through values
        are the same tensor objects, not copies.
        """
        input_ids: torch.Tensor = batch["input_ids"]
        B, L = input_ids.shape
        device = input_ids.device

        attention_mask = batch.get("attention_mask")
        real = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(input_ids, dtype=torch.bool)
        )

        if timesteps is None:
            t = self._sample_timesteps(batch, (B, L), device, generator)
        else:
            t = timesteps.to(device=device, dtype=torch.float32)
            if t.shape not in ((B,), (B, L)):
                raise ValueError(
                    f"timesteps must have shape {(B,)} or {(B, L)}, "
                    f"got {tuple(t.shape)}"
                )

        source_ids = self._sample_source(input_ids, generator)

        # p_t = (1 - kappa) * delta_{x_0} + kappa * delta_{x_1}, drawn
        # independently per position: a per-sequence draw would give the right
        # batch average while training on a different path.
        weight = self.kappa(t)
        threshold = weight if weight.dim() == 2 else weight[:, None]
        take_target = torch.rand((B, L), device=device, generator=generator) < threshold

        # Padding carries no supervision, so corrupting it would only feed the
        # model noise it is never scored on.
        take_target = take_target | ~real

        x_t = torch.where(take_target, input_ids, source_ids)

        labels_in = batch.get("labels")
        labels = labels_in.clone() if labels_in is not None else input_ids.clone()
        if labels_in is None:
            labels[~real] = -100

        model_inputs = {
            k: v
            for k, v in batch.items()
            if k not in _SUPERVISION_KEYS and k not in _TOPOLOGY_KEYS
        }
        model_inputs["input_ids"] = x_t

        return ProcessOutput(
            model_inputs=model_inputs,
            objective_inputs={
                "labels": labels,
                "source_ids": source_ids,
                "timesteps": t,
            },
        )


__all__ = ["DiscreteFlowProcess", "LinearKappa", "KappaSchedule"]
