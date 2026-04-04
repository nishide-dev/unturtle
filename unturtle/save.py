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

"""Saving utilities for unturtle models.

Adapted from unsloth/save.py — the push_to_hub wrapper is changed to tag
models with "unturtle" instead of "unsloth".
"""

from __future__ import annotations

import inspect
import types


def patch_saving_functions(model, vision: bool = False):
    """Patch ``push_to_hub`` and related methods on a PEFT model.

    Wraps ``push_to_hub`` so that "unturtle" is appended to the model's tags
    and to the commit message / description when pushing to HuggingFace Hub.

    Also delegates the heavier merged/GGUF/GGML/TorchAO save methods back to
    the upstream ``unsloth.save`` implementation so that those workflows still
    work without requiring a full re-port here.

    Args:
        model:  PEFT-wrapped model returned by ``FastDiffusionModel.get_peft_model``.
        vision: Unused; kept for API compatibility with the unsloth version.
    """
    # Determine the original (un-patched) push_to_hub
    if (
        hasattr(model, "push_to_hub")
        and model.push_to_hub.__name__ == "_unturtle_push_to_hub"
        and hasattr(model, "original_push_to_hub")
    ):
        # Already patched; no-op
        return model

    # Walk the model chain and patch each push_to_hub
    original_model = model
    while True:
        if (
            hasattr(original_model, "push_to_hub")
            and original_model.push_to_hub.__name__ != "_unturtle_push_to_hub"
        ):
            original_model.original_push_to_hub = original_model.push_to_hub
            original_model.push_to_hub = types.MethodType(
                _unturtle_push_to_hub, original_model
            )
            if hasattr(original_model, "add_model_tags"):
                original_model.add_model_tags(["unturtle"])

        if hasattr(original_model, "model"):
            original_model = original_model.model
        else:
            break

    # Delegate heavier save methods to unsloth.save so they remain functional
    try:
        from unsloth.save import patch_saving_functions as _unsloth_patch

        _unsloth_patch(model, vision=vision)
    except (ImportError, Exception):
        # If unsloth.save is not available, skip the extra methods silently.
        pass

    return model


def _unturtle_push_to_hub(self, *args, **kwargs):
    """push_to_hub wrapper that injects the 'unturtle' tag."""
    # Collect all arguments via the original signature
    sig = inspect.signature(self.original_push_to_hub)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)

    # Inject tag
    if "tags" in arguments and arguments["tags"] is not None:
        if isinstance(arguments["tags"], (list, tuple)):
            if "unturtle" not in arguments["tags"]:
                arguments["tags"] = list(arguments["tags"]) + ["unturtle"]
    elif "tags" in arguments:
        arguments["tags"] = ["unturtle"]
    elif hasattr(self, "add_model_tags"):
        self.add_model_tags(["unturtle"])

    # Inject commit_message
    if "commit_message" in arguments:
        msg = arguments["commit_message"]
        if msg is not None:
            if not msg.endswith(" "):
                msg += " "
            if "Unturtle" not in msg:
                msg += "(Trained with Unturtle)"
        else:
            msg = "Upload model trained with Unturtle"
        arguments["commit_message"] = msg

    # Inject commit_description
    if "commit_description" in arguments:
        desc = arguments["commit_description"]
        if desc is not None:
            if not desc.endswith(" "):
                desc += " "
            if "Unturtle" not in desc:
                desc += "(Trained with Unturtle)"
        else:
            desc = "Upload model trained with Unturtle"
        arguments["commit_description"] = desc

    try:
        return self.original_push_to_hub(**arguments)
    except TypeError:
        # Fallback: drop tags if the original method doesn't accept them
        arguments.pop("tags", None)
        return self.original_push_to_hub(**arguments)


def prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    use_reentrant: bool = True,
):
    """Thin wrapper around ``peft.prepare_model_for_kbit_training``.

    Accepts ``use_reentrant`` (used by the unsloth version) and maps it to
    PEFT's ``gradient_checkpointing_kwargs``.
    """
    from peft import prepare_model_for_kbit_training as _peft_prepare

    return _peft_prepare(
        model,
        use_gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
    )


__all__ = ["patch_saving_functions", "prepare_model_for_kbit_training"]
