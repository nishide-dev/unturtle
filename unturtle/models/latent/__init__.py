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

"""Continuous/latent method components (#66): codec boundary, FlowLM prototype.

See ``docs/rfcs/continuous-latent.md`` for the design source.  The FlowLM
generation algorithm registers in ``unturtle.models.generation.sampler``
(centrally, like every other family — this package never self-registers).
"""

from .codec import Codec, EmbeddingRoundingCodec
from .modeling_flowlm import (
    FlowLMConfig,
    FlowLMDenoiser,
    FlowLMDenoiserOutput,
    FlowLMModel,
)
from .modeling_ladiff import (
    LaDiffConfig,
    LaDiffModel,
    LatentAutoencoderCodec,
    LatentConditionedMDLM,
    PerceiverLiteEncoder,
    latent_autoencoder_loss,
)
from .objective import flowlm_loss

__all__ = [
    "Codec",
    "EmbeddingRoundingCodec",
    "FlowLMConfig",
    "FlowLMDenoiser",
    "FlowLMDenoiserOutput",
    "FlowLMModel",
    "flowlm_loss",
    "LaDiffConfig",
    "LaDiffModel",
    "LatentAutoencoderCodec",
    "LatentConditionedMDLM",
    "PerceiverLiteEncoder",
    "latent_autoencoder_loss",
]
