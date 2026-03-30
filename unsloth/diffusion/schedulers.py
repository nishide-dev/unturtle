# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
Alpha schedulers for masked diffusion language models.

The scheduler defines the masking rate α(t) as a function of diffusion
time t ∈ [0, 1]:
  - α(t) ≈ 1 at t=0  →  almost no tokens are masked (clean)
  - α(t) ≈ 0 at t=1  →  almost all tokens are masked (noisy)

The masking probability per token is:
  p_mask(t) = 1 - α(t)

The loss weight used by MDLM's "scheduler" weighting mode is:
  w(t) = -α'(t) / (1 - α(t))

Reference implementations:
  zhziszz/dllm  dllm/core/schedulers/alpha.py
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Union

import torch

Number = Union[float, torch.Tensor]


class BaseAlphaScheduler(ABC):
    """Abstract base for alpha (masking-rate) schedulers.

    Subclasses must implement ``_alpha`` and ``_alpha_derivative``.
    All public methods accept either a plain Python ``float`` or a
    ``torch.Tensor`` and return the same type.
    """

    # Registry of concrete scheduler classes, populated by __init_subclass__
    _registry: dict[str, type[BaseAlphaScheduler]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        BaseAlphaScheduler._registry[cls.__name__] = cls
        BaseAlphaScheduler._registry[cls.__name__.lower()] = cls

    # Make instances callable: scheduler(t) → alpha(t)
    def __call__(self, t: Number) -> Number:
        return self.alpha(t)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def alpha(self, t: Number) -> Number:
        """Masking-rate α(t) ∈ [0, 1] for timestep t ∈ [0, 1]."""
        t_tensor = self._to_tensor(t)
        out = self._alpha(t_tensor)
        return out.item() if isinstance(t, float) else out

    def alpha_derivative(self, t: Number) -> Number:
        """Derivative dα/dt."""
        t_tensor = self._to_tensor(t)
        out = self._alpha_derivative(t_tensor)
        return out.item() if isinstance(t, float) else out

    def weight(self, t: Number) -> Number:
        """MDLM loss weight w(t) = -α'(t) / (1 - α(t)).

        Used when ``loss_weight_type="scheduler"`` in MDLMConfig.
        """
        alpha_t = self.alpha(t)
        d_alpha_t = self.alpha_derivative(t)
        denom: Number
        if isinstance(alpha_t, torch.Tensor):
            denom = (1.0 - alpha_t).clamp_min(1e-6)
        else:
            denom = max(1.0 - alpha_t, 1e-6)
        return -d_alpha_t / denom

    def masking_prob(self, t: Number) -> Number:
        """Per-token masking probability p_mask(t) = 1 - α(t)."""
        alpha_t = self.alpha(t)
        if isinstance(alpha_t, torch.Tensor):
            return (1.0 - alpha_t).clamp(0.0, 1.0)
        return max(0.0, min(1.0, 1.0 - alpha_t))

    # ------------------------------------------------------------------ #
    #  Subclass hooks                                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        ...

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_tensor(t: Number) -> torch.Tensor:
        if isinstance(t, torch.Tensor):
            return t.float()
        return torch.tensor(t, dtype=torch.float32)


# ------------------------------------------------------------------ #
#  Concrete schedulers                                                #
# ------------------------------------------------------------------ #


class LinearAlphaScheduler(BaseAlphaScheduler):
    """α(t) = 1 - t  (used by LLaDA and d1)."""

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)


class CosineAlphaScheduler(BaseAlphaScheduler):
    """α(t) = 1 - cos(π/2 · (1 - t))  (smoother alternative)."""

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.cos((math.pi / 2.0) * (1.0 - t))

    def _alpha_derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -(math.pi / 2.0) * torch.sin((math.pi / 2.0) * (1.0 - t))


# Register short aliases so make_alpha_scheduler("linear") works
BaseAlphaScheduler._registry["linear"] = LinearAlphaScheduler
BaseAlphaScheduler._registry["cosine"] = CosineAlphaScheduler


# ------------------------------------------------------------------ #
#  Factory helper                                                     #
# ------------------------------------------------------------------ #


def make_alpha_scheduler(name: str) -> BaseAlphaScheduler:
    """Instantiate an alpha scheduler by name (case-insensitive).

    Args:
        name: ``"linear"`` or ``"cosine"`` (or full class names).

    Returns:
        A concrete :class:`BaseAlphaScheduler` instance.

    Raises:
        ValueError: If ``name`` is not recognised.
    """
    cls = BaseAlphaScheduler._registry.get(name) or BaseAlphaScheduler._registry.get(
        name.lower()
    )
    if cls is None:
        available = sorted(
            k for k in BaseAlphaScheduler._registry if k[0].isupper()
        )
        raise ValueError(
            f"Unknown alpha scheduler '{name}'. Available: {available}"
        )
    return cls()
