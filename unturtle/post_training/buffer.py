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
On-policy supervision buffer for OPDLM distillation (#64).

Sits between the rollout and the optimizer.  A rollout yields
:class:`~unturtle.post_training.trajectory.SupervisionState` objects in
whatever grouping the sampler produces; gradient accumulation consumes them in
fixed-size microbatches.  Those groupings do not line up, so states accumulate
here until a full batch exists.

**Identity is carried by ``sample_id``, never by list position.**  That is the
whole reason this is a class rather than a ``list`` and a slice: a state whose
identity depends on where it sits in a list is mispaired by any reorder, drop
or uneven regroup, and a mispaired student/teacher pair produces a finite,
plausible loss that trains the model to imitate the wrong sample.  Nothing in
a loss curve shows it.

Deliberately dependency-free beyond the trajectory contract: no model, no
device, no optimizer.  Bookkeeping this load-bearing should be verifiable
without any of them.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from .trajectory import SupervisionBatch, SupervisionState


class SupervisionBuffer:
    """Accumulate supervision states and emit fixed-size batches.

    Args:
        batch_size: Microbatch width for gradient accumulation.  Batches are
                    emitted only at exactly this size; the short tail is
                    reachable through :meth:`drain`.

    Example::

        buffer = SupervisionBuffer(batch_size=4)
        for rollout_states in sampler:
            for batch in buffer.extend(rollout_states):
                train_step(batch)
        for batch in buffer.drain():        # epoch boundary
            train_step(batch)
    """

    def __init__(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self._batch_size = batch_size
        self._pending: deque[SupervisionState] = deque()
        # Every id ever accepted, not just the ones still pending.  A duplicate
        # arriving after its twin was already emitted is exactly as ambiguous
        # as one arriving alongside it, and `SupervisionBatch.from_states`
        # cannot see across batch boundaries.
        self._seen: set[str] = set()
        self._block_size: int | None = None
        self._prompt_length: int | None = None

    def __len__(self) -> int:
        """States held but not yet emitted."""
        return len(self._pending)

    def extend(self, states: Iterable[SupervisionState]) -> list[SupervisionBatch]:
        """Add states, returning every full batch this completes.

        Returns an empty list until ``batch_size`` states have accumulated, and
        may return several batches at once when a single push carries more than
        one batch's worth.  A short remainder stays buffered.

        Raises:
            ValueError: on a repeated ``sample_id``, or a ``block_size`` that
                disagrees with what the buffer already holds.  Both are checked
                at push time, where the offending state is still identifiable
                by id — at loss time all that survives is a wrong number.
        """
        incoming = list(states)

        # Validate the WHOLE push before mutating anything.  Committing state
        # by state and raising partway through silently loses every state
        # after the offending one -- they never reach `_seen` or `_pending`, so
        # no diagnostic can even name them, and the caller's effective batch
        # shrinks invisibly.  It also makes the push unretryable: the states
        # already committed come back as duplicates, and that second error
        # blames the wrong state entirely.
        #
        # Staging `block_size` in a local likewise stops a rejected push from
        # pinning it: otherwise one malformed first state poisons the buffer
        # for its whole lifetime, rejecting well-formed states against a width
        # that was never accepted.
        block_size = self._block_size
        prompt_length = self._prompt_length
        arriving: set[str] = set()

        for state in incoming:
            if state.sample_id in self._seen or state.sample_id in arriving:
                raise ValueError(
                    f"duplicate sample_id {state.sample_id!r}: supervision is "
                    "paired by id, so two states claiming one id cannot be "
                    "matched to their teacher scores unambiguously"
                )
            arriving.add(state.sample_id)

            if block_size is None:
                block_size = state.block_size
            elif state.block_size != block_size:
                raise ValueError(
                    f"state {state.sample_id!r} has block_size "
                    f"{state.block_size} but the buffer holds {block_size}; "
                    "one batch is scored under a single denoising block "
                    "width, and mixing them applies the wrong block structure "
                    "to some rows"
                )

            # `block_size` alone does not make a batch coherent.  A state's
            # block count is `(length - prompt_length) / block_size`, so two
            # rows sharing a length and a width can still span different
            # numbers of blocks; a consumer indexing block boundaries
            # uniformly then reads the wrong positions for some rows.
            if prompt_length is None:
                prompt_length = state.prompt_length
            elif state.prompt_length != prompt_length:
                raise ValueError(
                    f"state {state.sample_id!r} has prompt_length "
                    f"{state.prompt_length} but the buffer holds "
                    f"{prompt_length}; rows would span different numbers of "
                    "denoising blocks under one batch-wide block index"
                )

        self._block_size = block_size
        self._prompt_length = prompt_length
        self._seen |= arriving
        self._pending.extend(incoming)

        batches = []
        while len(self._pending) >= self._batch_size:
            taken = [self._pending.popleft() for _ in range(self._batch_size)]
            batches.append(SupervisionBatch.from_states(taken))
        return batches

    def drain(self) -> list[SupervisionBatch]:
        """Emit whatever remains, including a short final batch.

        Separate from :meth:`extend` on purpose.  A short batch appearing
        mid-epoch would silently change the effective batch size for that step;
        making the flush explicit means it happens only where the caller
        intends it, at an epoch or rollout boundary.

        Does not clear the seen-id set: an id is spent for the buffer's
        lifetime, so a re-pushed sample is still caught after a drain.
        """
        if not self._pending:
            return []
        taken = list(self._pending)
        self._pending.clear()
        return [SupervisionBatch.from_states(taken)]


__all__ = ["SupervisionBuffer"]
