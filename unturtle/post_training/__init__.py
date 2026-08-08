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


"""Post-training methods: on-policy distillation and related objectives (#64)."""

from .buffer import SupervisionBuffer
from .divergence import teacher_student_divergence
from .rollout import (
    TrajectoryRecorder,
    combine_rounds_one_state_per_block,
    commit_steps_from_trajectory,
    random_mask_state,
    replay_rounds,
)
from .teacher import FrozenTeacher, resolve_top_k_logits
from .trajectory import SupervisionBatch, SupervisionState

__all__ = [
    "FrozenTeacher",
    "SupervisionBatch",
    "SupervisionBuffer",
    "SupervisionState",
    "TrajectoryRecorder",
    "combine_rounds_one_state_per_block",
    "commit_steps_from_trajectory",
    "random_mask_state",
    "replay_rounds",
    "resolve_top_k_logits",
    "teacher_student_divergence",
]
