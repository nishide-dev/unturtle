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

"""Dream diffusion language models.

Native bidirectional diffusion language model with custom generation utilities.
Original implementation by The Dream team, HKUNLP Group and the HuggingFace Inc. team.

Usage::

    from unturtle.models.dream import DreamConfig, DreamModel

    config = DreamConfig(
        vocab_size=10000, hidden_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=4,
    )
    model = DreamModel(config)
"""

from .configuration_dream import DreamConfig
from .modeling_dream import DreamPreTrainedModel, DreamBaseModel, DreamModel
from .generation_utils import DreamGenerationMixin, DreamGenerationConfig

__all__ = [
    "DreamConfig",
    "DreamPreTrainedModel",
    "DreamBaseModel",
    "DreamModel",
    "DreamGenerationMixin",
    "DreamGenerationConfig",
]
