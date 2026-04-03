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

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader


class BaseEvaluator(ABC):
    """Abstract base class for unturtle evaluation helpers."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            first_param = next(iter(model.parameters()), None)
            device = first_param.device if first_param is not None else torch.device("cpu")
        self.device = torch.device(device)

    def _make_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        collate_fn: Any | None = None,
    ) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    def _move_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._move_to_device(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            converted = [self._move_to_device(v) for v in value]
            return type(value)(converted)
        return value

    def _metric_key(self, prefix: str, name: str) -> str:
        return f"{prefix}_{name}" if prefix else name

    @contextmanager
    def evaluation_mode(self) -> Iterator[None]:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            if was_training:
                self.model.train()

    @abstractmethod
    def evaluate(self, *_args: Any, **_kwargs: Any) -> dict[str, float]:
        ...
