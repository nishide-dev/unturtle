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

"""#166 Stage-1 — nested per-operation timing via CUDA events.

`OperationTimer` synchronizes on entry AND exit of every scope, so an 8-layer
attention taxonomy adds 16 device synchronizations per step. This timer records
CUDA event pairs instead and synchronizes once per step.

**Measured honestly**: at the #166 masked fixture's scale the two are within
run-to-run noise, and the ordering flips between repeats — 2 layers gave
event/scope deltas of +1.8/+2.8, +2.3/+1.8, +3.0/+2.5 percent, and 8 layers gave
-0.1/+0.7, +0.9/+1.4, +0.1/+0.6. Each attention call is already ~1 ms of real
GPU work, so a fixed per-scope sync cost does not stand out. The event timer is
adopted because it CANNOT be worse and scales to deeper models, not because a
measurement at this scale demanded it; claiming a demonstrated win here would
overstate the evidence.

This timer records CUDA event pairs instead. Events are enqueued on the stream
with no synchronization, and elapsed times are read once per step, after a
single outer synchronize. The verdict remains the instrumentation-off wall
clock (`docs/acceleration-profile-protocol.md`); these numbers only attribute
it, and the instrumentation delta is always reported.

On CPU there are no events, so wall-clock deltas are used and the class stays
usable in tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = ["CudaEventTimer"]


@dataclass
class CudaEventTimer:
    """Per-operation accumulation with at most one sync per collection.

    Usage is two-phase by design: `measure()` only enqueues events during the
    step, and `collect()` performs the single synchronize and folds the
    elapsed times in. Reading an event before its work completes would
    otherwise force a sync per scope — the cost this class exists to avoid.
    """

    device: str = "cpu"
    inclusive: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    _pending: list[tuple[str, Any, Any]] = field(default_factory=list)
    _wall_pending: list[tuple[str, float]] = field(default_factory=list)

    @property
    def _cuda(self) -> bool:
        if not self.device.startswith("cuda"):
            return False
        import torch

        return torch.cuda.is_available()

    @contextmanager
    def measure(self, name: str):
        """Bracket one operation. No synchronization happens here."""
        if not self._cuda:
            import time

            start = time.perf_counter()
            try:
                yield
            finally:
                self._wall_pending.append((name, time.perf_counter() - start))
            return

        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._pending.append((name, start, end))

    def collect(self, *, synchronize: bool = True) -> None:
        """Fold every pending pair into the totals.

        `synchronize=False` is for a caller that has ALREADY synchronized at the
        step boundary — passing True there would sync twice per step, and the
        protocol's rule is one boundary synchronize.
        """
        for name, seconds in self._wall_pending:
            self.inclusive[name] = self.inclusive.get(name, 0.0) + seconds
            self.counts[name] = self.counts.get(name, 0) + 1
        self._wall_pending.clear()

        if not self._pending:
            return
        import torch

        if synchronize:
            torch.cuda.synchronize()
        for name, start, end in self._pending:
            seconds = start.elapsed_time(end) / 1000.0
            self.inclusive[name] = self.inclusive.get(name, 0.0) + seconds
            self.counts[name] = self.counts.get(name, 0) + 1
        self._pending.clear()

    def reset(self) -> None:
        """Drop everything, including pending events.

        Called after warmup so the timed window and the reported wall time
        cover the same steps.
        """
        self.inclusive.clear()
        self.counts.clear()
        self._pending.clear()
        self._wall_pending.clear()

    def result(self) -> dict[str, dict[str, Any]]:
        if self._pending or self._wall_pending:
            raise RuntimeError(
                f"{len(self._pending) + len(self._wall_pending)} timing scopes "
                "were never collected; call collect() after each step or the "
                "totals silently omit them"
            )
        return {
            name: {"inclusive_seconds": seconds, "call_count": self.counts[name]}
            for name, seconds in self.inclusive.items()
        }
