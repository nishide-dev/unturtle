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

"""MDLM-DiT native diffusion backbone (kuleshov-group/mdlm DiT, time-agnostic).

Reference: https://arxiv.org/abs/2406.07524 (Sahoo et al., NeurIPS 2024).
Native re-implementation baseline — not weight-compatible with the published
kuleshov checkpoints.
"""

from .configuration_mdlm_dit import MDLMDiTConfig
from .modeling_mdlm_dit import (
    MDLMDiTForMaskedDiffusionLM,
    MDLMDiTModel,
    MDLMDiTPreTrainedModel,
)

__all__ = [
    "MDLMDiTConfig",
    "MDLMDiTForMaskedDiffusionLM",
    "MDLMDiTModel",
    "MDLMDiTPreTrainedModel",
]
