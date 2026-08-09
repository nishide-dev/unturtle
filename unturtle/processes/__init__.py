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

"""unturtle.processes — forward (noising) processes for diffusion training.

Experimental/internal until ``DiffusionTrainer`` integration lands (#62 PR2);
these names are intentionally not re-exported from top-level ``unturtle``.
"""

from .base import AlphaSchedule, ForwardProcess, ProcessOutput
from .continuous_flow import ContinuousFlowProcess
from .discrete_flow import DiscreteFlowProcess, KappaSchedule, LinearKappa
from .masked import MaskedDiffusionProcess

__all__ = [
    "ContinuousFlowProcess",
    "DiscreteFlowProcess",
    "KappaSchedule",
    "LinearKappa",
    "AlphaSchedule",
    "ForwardProcess",
    "MaskedDiffusionProcess",
    "ProcessOutput",
]
