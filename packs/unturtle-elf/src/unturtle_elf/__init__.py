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

"""ELF (Embedded Language Flows) reference method pack — EXPERIMENTAL (#153).

Registration is EXPLICIT: importing this package mutates nothing.  A host
opts in via :func:`register_unturtle` (the #145 `unturtle.plugins` contract,
declared as the ``elf`` entry point).

The pack registers, onto the SUPPLIED hub only:

- generation algorithm ``elf`` (family ``embedding_flow``) — the official
  PyTorch ELF ODE/SDE sampling loop with in-context self-conditioning CFG
  and endpoint-only discretization;
- method manifest ``elf`` (generation-only; training is #154's scope).

The supports probe requires an actual pack-loaded ELF model
(:class:`~unturtle_elf.model.ELF` marker) — registration never promotes any
existing Unturtle model to "supports ELF".
"""

from __future__ import annotations

from typing import Any

METHOD_NAME = "elf"


def _supports_elf(model: Any) -> bool:
    """Only a pack-constructed ELF denoiser (or an object deliberately
    exposing the same marker) is supported — code existence promotes
    nothing."""
    return getattr(model, "is_elf_denoiser", False) is True


def _elf_unsupported(model: Any) -> str:
    return (
        f"{type(model).__name__} is not an ELF denoiser; load one with "
        "unturtle_elf.load_elf_model(checkpoint=...) — the ELF pack does "
        "not promote existing models (#153)."
    )


def run_elf(model: Any, request: Any) -> Any:
    """Runner: official-reference ELF sampling (lazy import keeps
    registration free of torch)."""
    from unturtle_elf.sampler import run_generation_request

    return run_generation_request(model, request)


def register_unturtle(hub: Any) -> None:
    """Register the ELF method onto ``hub`` — and only onto ``hub``."""
    from unturtle.methods import MethodSpec

    hub.generation(
        name=METHOD_NAME,
        family="embedding_flow",
        supports=_supports_elf,
        # Behind every builtin: loading the pack must never change what
        # `auto` picks for existing models (they fail the probe anyway).
        auto_priority=80,
        unsupported_message=_elf_unsupported,
    )(run_elf)

    hub.method(
        MethodSpec(
            name=METHOD_NAME,
            generation=(METHOD_NAME,),
            # Training/conversion deliberately absent: #153 is reference
            # parity + generation only; #154 owns training.
        )
    )


__all__ = ["METHOD_NAME", "register_unturtle", "run_elf"]
