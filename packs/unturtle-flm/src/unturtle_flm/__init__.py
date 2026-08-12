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

"""FLM/FMLM reference method pack — EXPERIMENTAL (#155).

Registration is EXPLICIT: importing this package mutates nothing.  A host
opts in via :func:`register_unturtle` (the #145 `unturtle.plugins` contract,
declared as the ``flm`` entry point).

TWO methods are registered, because their semantics differ structurally
(Stage-0 freeze): ``flm`` runs the one-time-conditioned Euler flow;
``fmlm`` composes the TWO-time-conditioned distilled flow map — FMLM is
never dispatched through the FLM solver (the issue's headline mutation
target).  The supports probes require pack-loaded models carrying the
matching marker; registration promotes nothing, and the historical Unturtle
``flowlm`` prototype is untouched.
"""

from __future__ import annotations

from typing import Any

FLM_NAME = "flm"
FMLM_NAME = "fmlm"


def _supports_flm(model: Any) -> bool:
    return getattr(model, "is_flm_denoiser", False) is True


def _supports_fmlm(model: Any) -> bool:
    # The flow map needs the double time-embedding backbone; a plain FLM
    # checkpoint must NOT satisfy this probe.
    return getattr(model, "is_fmlm_flow_map", False) is True


def _flm_unsupported(model: Any) -> str:
    return (
        f"{type(model).__name__} is not an FLM denoiser; load one with "
        "unturtle_flm.load_flm_model(...) — the FLM pack does not promote "
        "existing models (#155)."
    )


def _fmlm_unsupported(model: Any) -> str:
    return (
        f"{type(model).__name__} is not an FMLM flow map (double-time "
        "conditioning required); load one with "
        "unturtle_flm.load_fmlm_model(...) — an FLM denoiser is NOT a "
        "flow map (#155)."
    )


def run_flm(model: Any, request: Any) -> Any:
    from unturtle_flm.sampler import run_flm_request

    return run_flm_request(model, request)


def run_fmlm(model: Any, request: Any) -> Any:
    from unturtle_flm.sampler import run_fmlm_request

    return run_fmlm_request(model, request)


def register_unturtle(hub: Any) -> None:
    """Register both methods onto ``hub`` — and only onto ``hub``."""
    from unturtle.methods import MethodSpec

    hub.generation(
        name=FLM_NAME,
        family="onehot_flow",
        supports=_supports_flm,
        auto_priority=81,  # behind every builtin
        unsupported_message=_flm_unsupported,
    )(run_flm)

    hub.generation(
        name=FMLM_NAME,
        family="flow_map",
        supports=_supports_fmlm,
        auto_priority=82,
        unsupported_message=_fmlm_unsupported,
    )(run_fmlm)

    hub.method(MethodSpec(name=FLM_NAME, generation=(FLM_NAME,)))
    hub.method(MethodSpec(name=FMLM_NAME, generation=(FMLM_NAME,)))


__all__ = [
    "FLM_NAME",
    "FMLM_NAME",
    "register_unturtle",
    "run_flm",
    "run_fmlm",
]
