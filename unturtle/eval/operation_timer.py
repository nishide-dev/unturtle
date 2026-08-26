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

"""#166 Stage-1 operation timer — per-operation attribution for one cell.

Accumulates inclusive time and call counts per named operation, keyed by the
caller that invoked it, so a single entry point reached from several call sites
can still be attributed. That is exactly the ELF training case: four distinct
`model(...)` calls live in `elf_training_loss` — the shared unconditional
forward, the conditional self-conditioning forward, and the trained forward —
and Stage 0's hypothesis is about those being SEPARATE costs. Collapsing them
into one `model_forward` event would make the hypothesis untestable.

Timing is sync-bracketed on CUDA. Nothing here decides a verdict: the frozen
protocol's verdict is the instrumentation-off wall clock, and these numbers only
attribute it (`docs/acceleration-profile-protocol.md`).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OperationTimer", "caller_scope"]


@dataclass
class OperationTimer:
    """Running per-operation totals for one cell.

    Only accumulates; the caller decides which events are
    ``coverage_eligible`` when it builds the record, because exclusivity is a
    property of the taxonomy rather than of the measurement.
    """

    device: str = "cpu"
    inclusive: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _stack: list[str] = field(default_factory=list)

    def _sync(self) -> None:
        if not self.device.startswith("cuda"):
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, name: str):
        """Time one operation, synchronizing at both boundaries on CUDA.

        Async kernel launches would otherwise credit the time to whichever
        operation happens to synchronize next.
        """
        self._sync()
        self._stack.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self._stack.pop()
            self.inclusive[name] = self.inclusive.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1

    def reset(self) -> None:
        self.inclusive.clear()
        self.counts.clear()
        self._stack.clear()

    def result(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"inclusive_seconds": seconds, "call_count": self.counts[name]}
            for name, seconds in self.inclusive.items()
        }


def caller_scope(depth: int = 2) -> str:
    """Name of the function `depth` frames up, for call-site attribution.

    Used to tell apart several calls to the SAME callable from different places
    — the ELF auxiliary forwards versus the trained forward — without editing
    the pack, whose reference semantics must stay untouched while profiling.
    """
    import inspect

    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is None:
                return "unknown"
            frame = frame.f_back
        return frame.f_code.co_name if frame is not None else "unknown"
    finally:
        del frame
