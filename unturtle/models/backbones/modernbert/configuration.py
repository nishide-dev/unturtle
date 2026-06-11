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

"""Configuration for ModernBERT diffusion models."""

from transformers import ModernBertConfig


class DiffusionModernBertConfig(ModernBertConfig):
    """ModernBertConfig for diffusive fine-tuning at unturtle.

    Uses a distinct ``model_type`` so that config round-trips via
    ``AutoConfig.from_pretrained`` resolve to the unturtle subclass
    instead of colliding with upstream ``"modernbert"``.
    """

    model_type = "modernbert-diffusion"
