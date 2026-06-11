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

* ``_patch_trl_trainer`` is an unturtle-specific variant of unsloth's TRL patcher
  that would wire the *dLLM-aware* packing helpers from :mod:`unturtle.utils.packing`
  (bidirectional / varlen attention, packed-sequence masking) into TRL's
  ``SFTTrainer``. NOTE: it is currently **not invoked automatically** anywhere —
  at import time only unsloth's own patcher fires. The supported path for dLLM
  packed training today is the explicit opt-in API
  :func:`unturtle.utils.packing.enable_sample_packing` (exercised by
  ``tests/utils/test_packing.py``). Whether to auto-wire ``_patch_trl_trainer``
  (and how to order it against unsloth's ``__UNSLOTH_BACKWARDS_COMPATIBLE__`` /
  ``_unsloth_auto_packing_wrapped`` guards) is tracked as a follow-up issue.
"""

import dataclasses
import inspect
import logging
import os
from functools import wraps

import trl
from unsloth.trainer import (
    QGaloreConfig,
    UnslothTrainer,
    UnslothTrainingArguments,
    UnslothVisionDataCollator,
)
from unsloth.trainer import (
    unsloth_train as unturtle_train,
)
from unsloth_zoo.hf_utils import get_transformers_model_type
from unsloth_zoo.utils import Version

from unturtle.utils.packing import (
    configure_padding_free,
    configure_sample_packing,
    enable_padding_free_metadata,
    enable_sample_packing,
)

__all__ = [
    "UnturtleTrainingArguments",
    "UnturtleTrainer",
    "unturtle_train",
    "_patch_trl_trainer",
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


# ---------------------------------------------------------------------------
# dLLM-aware TRL patching
# ---------------------------------------------------------------------------
# The functions below mirror unsloth's ``_patch_trl_trainer`` but deliberately
# wire the *unturtle* packing helpers (bidirectional / varlen, packed-sequence
# boundary masking) from ``unturtle.utils.packing`` instead of unsloth's
# causal-only helpers. They are kept in full here because dLLM packing semantics
# differ from unsloth's causal SFT packing.

_AUTO_PADDING_FREE_ENV_DISABLED = os.environ.get(
    "UNSLOTH_DISABLE_AUTO_PADDING_FREE", ""
).strip().lower() in {"1", "true", "yes", "on"}

PADDING_FREE_BLOCKLIST = {
    "gemma2",  # - gemma2:  Uses slow_attention_softcapping which has torch.compile issues
    "gpt_oss",  # - gpt_oss: Uses Flex Attention which doesn't handle padding_free correctly
}


def _should_pack(config) -> bool:
    if config is None or not getattr(config, "packing", False):
        return False
    return not getattr(config, "_unsloth_disable_auto_packing", False)


def _should_auto_padding_free(config) -> bool:
    if (
        config is None
        or _AUTO_PADDING_FREE_ENV_DISABLED
        or getattr(config, "packing", False)
    ):
        return False
    return getattr(config, "padding_free", None) is None


def _disable_sample_packing(config):
    if config is None:
        return
    for attr, value in (("packing", False), ("padding_free", False)):
        if hasattr(config, attr):
            setattr(config, attr, value)
    if hasattr(config, "remove_unused_columns"):
        config.remove_unused_columns = True
    config._unsloth_disable_auto_packing = True


_AUTO_PACK_SKIP_MESSAGES = (
    "packing is not supported",
    "padding-free training",
    "passing a custom data collator",
)


def _should_skip_auto_packing_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(msg in message for msg in _AUTO_PACK_SKIP_MESSAGES)


from transformers import ProcessorMixin  # noqa: E402


# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth
def _resolve_trainer_params(trainer_class, init_fn):
    """Resolve the real named parameters for a trainer __init__.

    Some TRL trainers (e.g., ORPOTrainer in TRL 0.27.1) are thin wrappers
    with only ``def __init__(self, *args, **kwargs)``.  For those, walk the
    MRO and return the first parent class that has real named parameters.
    """
    params = inspect.signature(init_fn).parameters
    named = {
        k
        for k, v in params.items()
        if v.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and k != "self"
    }
    if named:
        return set(params.keys())

    # Thin wrapper detected - walk MRO for real signature
    for cls in trainer_class.__mro__[1:]:
        if cls is object:
            continue
        parent_init = cls.__dict__.get("__init__")
        if parent_init is None:
            continue
        try:
            parent_params = inspect.signature(parent_init).parameters
            parent_named = {
                k
                for k, v in parent_params.items()
                if v.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and k != "self"
            }
            if parent_named:
                return set(parent_params.keys())
        except (ValueError, TypeError):
            continue
    return set(params.keys())


def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # All Trainer tokenizer are now called processing_class
        trainer_params = _resolve_trainer_params(trainer_class, original_init)

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if ("args" in kwargs) and (Version(trl) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove("self")
            trainer_params.remove("args")

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field
                for field in dataclasses.fields(config_class)
                if field.init
            }

            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
            from transformers import TrainingArguments

            moved_params = set(inspect.signature(config_class).parameters.keys()) - set(
                inspect.signature(TrainingArguments).parameters.keys()
            )

            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params:
                    trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            # Reinitialising config class with parameters (that were none initially but populated on first init)
            # causes the 2nd init to fail as there are mutual exclusive checks on pairs of parameters.
            # Refer: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L499-L502 for example
            # So we only create config class if the previous init was not TrainingArguments
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                config = training_args

            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        original_init(self, *args, **kwargs)

    return new_init


def _patch_sft_trainer_auto_packing(trl_module):
    sft_trainer = getattr(trl_module, "SFTTrainer", None)
    if sft_trainer is None:
        return
    if getattr(sft_trainer, "_unsloth_auto_packing_wrapped", False):
        return

    original_init = sft_trainer.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        config_arg = args[1] if len(args) >= 2 else kwargs.get("args")

        # Check if model type is unsupported for padding_free
        model = kwargs.get("model")
        is_unsupported_model = False
        is_vlm = False
        if model is not None:
            model_config = getattr(model, "config", None)
            if model_config is not None:
                model_types = get_transformers_model_type(model_config)
                # Blocklist: models that don't work correctly with padding_free
                is_unsupported_model = any(
                    x in PADDING_FREE_BLOCKLIST for x in model_types
                )

                # Check if VLM
                architectures = getattr(model_config, "architectures", None)
                if architectures is None:
                    architectures = []
                is_vlm = any(
                    x.endswith("ForConditionalGeneration") for x in architectures
                )
                is_vlm = is_vlm or hasattr(model_config, "vision_config")

        processing_class = kwargs.get("processing_class") or kwargs.get("tokenizer")
        data_collator = kwargs.get("data_collator")

        # We also disable vision language models for padding free collators
        blocked = (
            (data_collator is not None)
            or isinstance(processing_class, ProcessorMixin)
            or is_vlm
            or is_unsupported_model
            or (
                os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "1"
            )  # Disable padding free on forced logits
        )
        requested_pack = bool(getattr(config_arg, "packing", False))
        if blocked:
            if hasattr(config_arg, "packing"):
                config_arg.packing = False
            if hasattr(config_arg, "padding_free"):
                config_arg.padding_free = False

        if blocked and requested_pack:
            reason = "custom data collator"
            if data_collator is None and isinstance(processing_class, ProcessorMixin):
                reason = "processor-based model"
            elif is_vlm:
                reason = "vision-language model"
            elif is_unsupported_model:
                reason = f"unsupported model type(s): {', '.join(model_types)}"
            message = f"Unturtle: Sample packing skipped ({reason} detected)."
            print(message)

        packing_active = False
        if _should_pack(config_arg) and not blocked:
            configure_sample_packing(config_arg)
            packing_active = True
            logger.info("Unturtle: Sample packing enabled for SFTTrainer instance.")

        # Resolve padding_free: None (default) = auto-enable unless env-disabled or packing
        auto_padding_free_active = False
        padding_free_requested = getattr(config_arg, "padding_free", None) is True
        if not blocked:
            if padding_free_requested:
                configure_padding_free(config_arg)
            elif _should_auto_padding_free(config_arg):
                configure_padding_free(config_arg)
                auto_padding_free_active = True
                logger.info(
                    "Unturtle: Padding-free batching auto-enabled for SFTTrainer instance."
                )

        try:
            original_init(self, *args, **kwargs)
        except ValueError as exc:
            if packing_active and _should_skip_auto_packing_error(exc):
                logger.info(
                    "Unturtle: Auto sample packing failed because trainer reported an incompatible setup (%s).",
                    exc,
                )
                _disable_sample_packing(config_arg)
                packing_active = False
                original_init(self, *args, **kwargs)
            else:
                raise

        trainer_args = getattr(self, "args", None)
        trainer_packing = bool(trainer_args and getattr(trainer_args, "packing", False))
        trainer_padding_free = bool(
            trainer_args and getattr(trainer_args, "padding_free", False)
        )

        if blocked and trainer_args is not None:
            # Mirror the block on the trainer args to avoid re-enabling later
            trainer_args.packing = False
            trainer_args.padding_free = False

        if (
            not blocked
            and trainer_packing
            and (packing_active or _should_pack(trainer_args))
        ):
            enable_sample_packing(self.model, self)
            print(
                "🦥 Unturtle: Packing enabled - training is >2x faster and uses less VRAM!"
            )
        elif not blocked and trainer_padding_free:
            enable_padding_free_metadata(self.model, self)
            message = (
                "🦥 Unturtle: Padding-free auto-enabled, enabling faster training."
                if auto_padding_free_active
                else "🦥 Unturtle: Padding-free enabled, enabling faster training."
            )
            print(message)

    sft_trainer.__init__ = new_init
    sft_trainer._unsloth_auto_packing_wrapped = True


def _patch_trl_trainer():
    import trl

    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"):
        return
    if Version(trl) <= Version("0.11.0"):
        return

    import trl.trainer

    trl_classes = dir(trl.trainer)
    trl_trainers = set(
        x[: -len("Trainer")] for x in trl_classes if x.endswith("Trainer")
    )
    trl_configs = set(x[: -len("Config")] for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:
            exec(
                f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)",
                globals(),
            )
        except (AttributeError, TypeError):
            # AttributeError: class does not exist on this TRL version
            # TypeError: config_class is not a dataclass (unexpected TRL internals change)
            continue

    _patch_sft_trainer_auto_packing(trl)

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
