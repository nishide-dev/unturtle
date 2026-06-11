# unturtle/eval/harness/__init__.py
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

"""unturtle.eval.harness — canonical lm-evaluation-harness integration.

This is the authoritative benchmark path for Unturtle dLLMs. ``lm_eval`` is an optional
dependency; importing this package does NOT require it (the adapter and runner import
``lm_eval`` lazily at call time).
"""

from .configs import DecodingConfig, get_decoding_config, list_decoding_configs
from .runner import run_harness_evaluation

__all__ = [
    "DecodingConfig",
    "get_decoding_config",
    "list_decoding_configs",
    "run_harness_evaluation",
]
