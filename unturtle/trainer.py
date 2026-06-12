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

"""Unturtle trainer layer.

This module is a thin dLLM-aware layer on top of unsloth's trainer machinery:

* :class:`UnturtleTrainer` / :class:`UnturtleTrainingArguments` subclass unsloth's
  :class:`~unsloth.trainer.UnslothTrainer` / :class:`~unsloth.trainer.UnslothTrainingArguments`
  so that the embedding-LR / Q-GaLore optimizer logic is inherited rather than
  vendored. ``DiffusionTrainer`` builds on top of ``UnturtleTrainer``.

* dLLM packed-sequence training is wired via the explicit opt-in API
  :func:`unturtle.utils.packing.enable_sample_packing` (exercised by
  ``tests/utils/test_packing.py``). An unturtle-specific auto-wiring TRL patcher
  (``_patch_trl_trainer``) was removed (issue #12): it was never invoked, and
  unsloth's own ``__UNSLOTH_BACKWARDS_COMPATIBLE__`` guard makes auto-wiring
  structurally unreachable when unsloth is imported first (which unturtle
  always does).
"""

import logging

from unsloth.trainer import (
    QGaloreConfig,
    UnslothTrainer,
    UnslothTrainingArguments,
    UnslothVisionDataCollator,
)
from unsloth.trainer import (
    unsloth_train as unturtle_train,
)

__all__ = [
    "UnturtleTrainingArguments",
    "UnturtleTrainer",
    "unturtle_train",
    "UnslothVisionDataCollator",
    "QGaloreConfig",
]

logger = logging.getLogger(__name__)


class UnturtleTrainingArguments(UnslothTrainingArguments):
    """Training arguments for unturtle.

    Inherits the embedding-LR / Q-GaLore fields from
    :class:`~unsloth.trainer.UnslothTrainingArguments`. dLLM-specific fields
    (alpha scheduler, loss weighting, packed-sequence flags, ...) are added by
    :class:`~unturtle.diffusion.trainer.DiffusionTrainingArguments`.
    """


class UnturtleTrainer(UnslothTrainer):
    """Base trainer for unturtle.

    Inherits the embedding-LR + Q-GaLore optimizer construction from
    :class:`~unsloth.trainer.UnslothTrainer` (which itself extends TRL's patched
    ``SFTTrainer``). dLLM behaviour (masked-diffusion loss, collator wiring) is
    added by :class:`~unturtle.diffusion.trainer.DiffusionTrainer`.
    """
