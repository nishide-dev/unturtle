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

"""The #144 pressure-test fixture: an out-of-core-style extension.

This module plays the role of a hypothetical external package implementing
"a future paper that is mostly a new combination of existing primitives plus
one new solver":

- REUSES the existing ``masked`` process component;
- REUSES the existing ``mdlm`` training recipe;
- adds ONE new generation algorithm (``toy_echo`` — a deterministic
  reverse-the-prompt "solver", enough to prove the runner really executed);
- ties them together with ONE new ``MethodSpec`` requiring the existing
  ``masked_generation`` backbone capability.

Registration is EXPLICIT: importing this module mutates nothing.  A host
opts in by calling :func:`register_unturtle` with the hub that should carry
the extension.  This is exactly the contributor-facing shape #141 sketched.

It is a synthetic architecture fixture — not a research method, and it makes
no scientific claims.
"""

from __future__ import annotations

from unturtle.methods import MethodSpec

METHOD_NAME = "toy_echo"


def _supports_toy_echo(model) -> bool:
    """Capability probe: the model must explicitly opt in (mirrors how DFM's
    research-only boundary works — the plugin does not get to promote
    itself)."""
    return getattr(model, "supports_toy_echo", False) is True


def _toy_echo_unsupported(model) -> str:
    return (
        f"{type(model).__name__} does not opt into toy_echo generation "
        "(set supports_toy_echo = True); this is a #144 architecture fixture."
    )


def register_unturtle(hub) -> None:
    """Register the extension onto ``hub`` — and only onto ``hub``.

    Uses the registry-bound decorator for the generation runner (the piece
    with function identity worth preserving) and explicit calls for the
    manifest, per the #144 decorator-ergonomics brief.
    """

    @hub.generation(
        name=METHOD_NAME,
        family="masked_discrete",
        supports=_supports_toy_echo,
        auto_priority=90,  # behind every builtin: never wins auto by accident
        unsupported_message=_toy_echo_unsupported,
    )
    def run_toy_echo(model, request):
        """The 'solver': deterministically reverse the prompt token list.

        Real enough to prove dispatch executed THIS runner (the sentinel is
        the transformation, not a mock), small enough to need no model."""
        tokens = list(request.inputs)
        return {"method": METHOD_NAME, "tokens": tokens[::-1]}

    hub.method(
        MethodSpec(
            name=METHOD_NAME,
            process="masked",  # existing component, reused
            training="mdlm",  # existing recipe, reused
            generation=(METHOD_NAME,),
            required_capabilities=frozenset({"masked_generation"}),
        )
    )
