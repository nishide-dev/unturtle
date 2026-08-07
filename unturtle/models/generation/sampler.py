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

"""Decoding-algorithm selection for Unturtle dLLM generation.

Makes the decoding algorithm an explicit, first-class choice instead of an implicit
combination of ``MaskedDiffusionGenerationConfig`` flags.  Each algorithm is a
registered :class:`GenerationAlgorithm` describing what it needs from a model and
which flags (if any) it selects — so this is pure *selection*, with no generation
logic of its own.

Algorithms (discrete masked diffusion):
  - ``mdlm``         : plain MDLM denoising loop
  - ``block_decode`` : Fast-dLLM KV-cache block decode (parallel decode is an option within);
                       requires the model to implement ``_model_forward_with_cache`` and opt in
                       via ``supports_block_decode`` (defaults to ``True`` when absent).
  - ``bd3lm``        : Unturtle's masked block diffusion (BD3LM); requires
                       ``_sample_block_diffusion`` (TinyA2D family today).

Algorithm (self-conditioned canvas block diffusion):
  - ``block_ar``     : upstream native canvas block diffusion for the DiffusionGemma family;
                       requires ``_denoising_step`` (the DiffusionGemmaGenerationMixin probe).
                       No mask token is used — the upstream ``GenerationConfig`` governs the
                       generation loop entirely, so ``algorithm_to_flags("block_ar")`` returns
                       ``{}`` (no ``use_cache``/``use_block_diffusion`` injection).

Key distinction — ``bd3lm`` vs ``block_ar``:
  - ``bd3lm``        : Unturtle's *masked* block diffusion; requires a mask token; uses
                       ``_sample_block_diffusion``; flag ``use_block_diffusion=True`` injected.
  - ``block_ar``     : upstream *self-conditioned* canvas block diffusion (DiffusionGemma);
                       no mask token; governed by the upstream generation config; no flags
                       injected.

Explicit algorithm choices are capability-checked: passing an algorithm the model cannot
execute raises ``ValueError`` immediately rather than silently falling back or crashing
mid-generation.

The registry is open to future families (discrete flow matching, continuous/latent).
``auto_priority`` is an explicit number rather than registration order, so a newly
registered algorithm does **not** outrank the masked ones merely by existing — #69 calls
that out specifically.  ``flags`` defaults to empty, so a non-masked family is never
handed ``use_cache`` / ``use_block_diffusion``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

#: Lower numbers win when resolving ``auto``.  Masked/canvas algorithms occupy
#: 10-40; anything registered later defaults to 1000 and therefore loses.
_DEFAULT_AUTO_PRIORITY = 1000


@dataclass
class GenerationRequest:
    """One generation call, in the shape every runner receives.

    Deliberately not a config schema: #69 rules out merging every
    family-specific generation config into one.  This carries the call, and
    each family keeps interpreting ``generation_config``/``kwargs`` its own way.
    """

    inputs: Any = None
    generation_config: Any = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationAlgorithm:
    """One decoding algorithm's selection metadata.

    Args:
        name:                Algorithm name as passed to ``generate(algorithm=...)``.
        family:              Descriptive grouping (``"masked_discrete"``,
                             ``"canvas"``, …).  Not an ontology — it exists so a
                             future family is not forced to describe itself in
                             masked-diffusion terms.
        supports:            ``(model) -> bool`` capability probe.
        flags:               Flags injected into ``generate`` kwargs.  Empty for
                             families whose loop is governed elsewhere.
        auto_priority:       Tie-break for ``auto``; lower wins.  Explicit rather
                             than registration-order so registering an algorithm
                             cannot silently change existing selection.
        auto_eligible:       Whether ``auto`` may pick this at all.  ``bd3lm`` is
                             opt-in via ``bd3lm_requested``, never automatic.
        unsupported_message: ``(model) -> str`` for the explicit-selection error.
                             Per-algorithm because each names its missing hook and
                             a concrete alternative.
        runner:              ``(model, request) -> output``.  Executes the
                             algorithm.  Each masked runner calls its own sampling
                             loop directly rather than round-tripping the choice
                             through ``use_cache``/``use_block_diffusion``
                             booleans, which is what lets a family with no
                             corresponding boolean exist at all.
    """

    name: str
    family: str
    supports: Callable[[Any], bool]
    flags: Mapping[str, bool] = field(default_factory=dict)
    auto_priority: int = _DEFAULT_AUTO_PRIORITY
    auto_eligible: bool = True
    unsupported_message: Callable[[Any], str] | None = None
    runner: Callable[[Any, GenerationRequest], Any] | None = None

    def __post_init__(self) -> None:
        # `frozen=True` blocks rebinding the field, not mutating the dict it
        # points at — so a caller could otherwise edit the registry's own flags
        # in place.  Store a read-only view instead.
        if not isinstance(self.flags, MappingProxyType):
            object.__setattr__(self, "flags", MappingProxyType(dict(self.flags)))

    def describe_unsupported(self, model: Any) -> str:
        if self.unsupported_message is not None:
            return self.unsupported_message(model)
        return f"{type(model).__name__} does not support algorithm {self.name!r}."


def _supports_block_ar(model: Any) -> bool:
    """True if the model is a DiffusionGemma-family canvas block-diffusion model.

    The presence of ``_denoising_step`` is the canonical probe for
    ``DiffusionGemmaGenerationMixin``.  These models use self-conditioned canvas
    block diffusion (no mask token) and their generation loop is governed by the
    upstream ``GenerationConfig`` rather than Unturtle flags.
    """
    return callable(getattr(model, "_denoising_step", None))


def _supports_mdlm(model: Any) -> bool:
    """True if the model implements the masked-diffusion sampling loop.

    The presence of ``_sample`` is the canonical probe for the masked-diffusion
    generation mixin (LLaDA / Dream / TinyA2D / ModernBERT all define it).
    Models without ``_sample`` have no mask-token semantics and cannot run
    mdlm / block_decode / bd3lm algorithms.
    """
    return callable(getattr(model, "_sample", None))


def _supports_block_decode(model: Any) -> bool:
    """True if the model implements the block-decode cache hook AND opts in.

    ``supports_block_decode = False`` lets a backbone that inherits the mixin
    generically (e.g. encoder backbones without KV cache) opt out of the
    block-decode fast path.
    """
    if not getattr(model, "supports_block_decode", True):
        return False
    return callable(getattr(model, "_model_forward_with_cache", None))


def _supports_bd3lm(model: Any) -> bool:
    """True if the model implements BD3LM block-diffusion sampling."""
    return callable(getattr(model, "_sample_block_diffusion", None))


def _block_ar_unsupported(model: Any) -> str:
    return (
        f"{type(model).__name__} does not support block_ar "
        f"(native canvas block diffusion, DiffusionGemma family); "
        f"use algorithm='mdlm' or 'block_decode' for masked models."
    )


def _mdlm_unsupported(model: Any) -> str:
    return (
        f"{type(model).__name__} has no masked-diffusion sampling loop "
        f"(no mask-token semantics); use algorithm='block_ar'."
    )


def _block_decode_unsupported(model: Any) -> str:
    return (
        f"{type(model).__name__} does not support block-decode "
        f"(no usable KV-cache forward); use algorithm='mdlm'."
    )


def _bd3lm_unsupported(model: Any) -> str:
    return (
        f"{type(model).__name__} does not implement BD3LM "
        f"(_sample_block_diffusion); supported on the TinyA2D family. "
        f"Use algorithm='mdlm' or 'block_decode'."
    )


def _run_mdlm(model: Any, request: GenerationRequest) -> Any:
    return model._sample(
        request.inputs,
        generation_config=request.generation_config,
        **request.kwargs,
    )


def _run_block_decode(model: Any, request: GenerationRequest) -> Any:
    return model._sample_with_cache(
        request.inputs,
        generation_config=request.generation_config,
        **request.kwargs,
    )


def _run_bd3lm(model: Any, request: GenerationRequest) -> Any:
    return model._sample_block_diffusion(
        request.inputs,
        generation_config=request.generation_config,
        **request.kwargs,
    )


def _run_block_ar(model: Any, request: GenerationRequest) -> Any:
    """Delegate to the model's own (upstream) generate.

    The canvas family's loop belongs to upstream `transformers`; Unturtle only
    selects it.  Calling `generate` here is safe because the DiffusionGemma
    shim resolves the algorithm and then calls `super().generate` — it does not
    re-enter dispatch.
    """
    return model.generate(
        request.inputs,
        generation_config=request.generation_config,
        **request.kwargs,
    )


_ALGORITHMS: list[GenerationAlgorithm] = [
    GenerationAlgorithm(
        name="block_ar",
        family="canvas",
        supports=_supports_block_ar,
        # No flags: the upstream GenerationConfig governs the loop entirely.
        flags={},
        # Inert: `resolve_algorithm` checks block_ar *before* the bd3lm opt-in,
        # so it is never reached by the priority loop.  That sequencing is not
        # expressible as a priority — it is what makes a canvas model win even
        # when bd3lm was requested — so do not "simplify" the special case away
        # on the assumption this number covers it.
        auto_priority=10,
        unsupported_message=_block_ar_unsupported,
        runner=_run_block_ar,
    ),
    GenerationAlgorithm(
        name="bd3lm",
        family="masked_discrete",
        supports=_supports_bd3lm,
        flags={"use_cache": False, "use_block_diffusion": True},
        # Also inert, for the same reason: `auto_eligible=False` keeps bd3lm out
        # of the loop, and the bd3lm_requested branch selects it positionally.
        auto_priority=20,
        # Never chosen automatically — only via the bd3lm_requested opt-in.
        auto_eligible=False,
        unsupported_message=_bd3lm_unsupported,
        runner=_run_bd3lm,
    ),
    GenerationAlgorithm(
        name="block_decode",
        family="masked_discrete",
        supports=_supports_block_decode,
        flags={"use_cache": True, "use_block_diffusion": False},
        auto_priority=30,
        unsupported_message=_block_decode_unsupported,
        runner=_run_block_decode,
    ),
    GenerationAlgorithm(
        name="mdlm",
        family="masked_discrete",
        supports=_supports_mdlm,
        flags={"use_cache": False, "use_block_diffusion": False},
        auto_priority=40,
        unsupported_message=_mdlm_unsupported,
        runner=_run_mdlm,
    ),
]


def iter_algorithms() -> tuple[GenerationAlgorithm, ...]:
    """Every registered algorithm, in registration order."""
    return tuple(_ALGORITHMS)


def find_algorithm(name: str) -> GenerationAlgorithm | None:
    """The registered algorithm called ``name``, or ``None``."""
    for algorithm in _ALGORITHMS:
        if algorithm.name == name:
            return algorithm
    return None


def register_algorithm(algorithm: GenerationAlgorithm) -> None:
    """Add an algorithm to the registry.

    Raises:
        ValueError: if the name is already taken.  Two algorithms sharing a name
            would make selection depend on registration order.
    """
    if find_algorithm(algorithm.name) is not None:
        raise ValueError(f"decoding algorithm {algorithm.name!r} is already registered")
    _ALGORITHMS.append(algorithm)


def _unregister_algorithm(algorithm: GenerationAlgorithm) -> None:
    """Remove an algorithm.  Test/teardown helper, not a public API.

    Removes by identity: ``GenerationAlgorithm`` is a frozen dataclass with value
    equality, so removing by value could drop a different, equal-looking entry.
    """
    for index, existing in enumerate(_ALGORITHMS):
        if existing is algorithm:
            del _ALGORITHMS[index]
            return


def _algorithm_names() -> list[str]:
    return sorted(a.name for a in _ALGORITHMS)


def algorithm_to_flags(algorithm: str) -> dict[str, bool]:
    """Return the generate() flag set for a named algorithm.

    For discrete masked algorithms (mdlm / block_decode / bd3lm) this returns the
    ``use_cache`` / ``use_block_diffusion`` flags that the model's generate dispatch
    understands.

    For ``block_ar`` (DiffusionGemma canvas block diffusion) this returns ``{}`` —
    no flags are injected because the upstream ``GenerationConfig`` governs the loop
    entirely; Unturtle only selects the algorithm, it does not override the config.
    A copy is returned so a caller merging it into kwargs cannot corrupt the registry.
    """
    entry = find_algorithm(algorithm)
    if entry is None:
        raise ValueError(
            f"Unknown decoding algorithm {algorithm!r}. "
            f"Supported: {_algorithm_names()}."
        )
    return dict(entry.flags)


def resolve_algorithm(algorithm: str, model: Any, *, bd3lm_requested: bool) -> str:
    """Resolve ``algorithm`` to a concrete algorithm name.

    ``auto`` picks the fastest path the model supports, by ``auto_priority``:
      - ``block_ar`` first, when the model is a DiffusionGemma-family canvas model
        (implements ``_denoising_step``); this takes priority over masked algorithms.
      - Else BD3LM if requested (and the model implements ``_sample_block_diffusion``),
      - else block-decode (Fast-dLLM) when the model supports the cache hook,
      - else plain MDLM.

    Explicit algorithm names are capability-checked, each with a message naming the
    missing hook and a workable alternative.  An explicitly requested algorithm never
    silently falls back — including ``bd3lm_requested`` under ``auto``.
    """
    if algorithm == "auto":
        canvas = find_algorithm("block_ar")
        if canvas is not None and canvas.supports(model):
            return canvas.name

        if bd3lm_requested:
            # Requested explicitly, so an incapable model is an error rather
            # than a reason to quietly pick something else.
            bd3lm = find_algorithm("bd3lm")
            if bd3lm is None or not bd3lm.supports(model):
                raise ValueError(_bd3lm_unsupported(model))
            return bd3lm.name

        for entry in sorted(_ALGORITHMS, key=lambda a: a.auto_priority):
            if entry.auto_eligible and entry.supports(model):
                return entry.name

        raise ValueError(
            f"{type(model).__name__} does not implement any known decoding algorithm "
            "(no _denoising_step, _model_forward_with_cache, or _sample). "
            "Ensure the model is a supported dLLM backbone."
        )

    entry = find_algorithm(algorithm)
    if entry is None:
        raise ValueError(
            f"Unknown decoding algorithm {algorithm!r}. "
            f"Supported: {_algorithm_names()} (or 'auto')."
        )
    if not entry.supports(model):
        raise ValueError(entry.describe_unsupported(model))
    return entry.name


def dispatch_generation(
    model: Any,
    request: GenerationRequest,
    algorithm: str = "auto",
    *,
    bd3lm_requested: bool = False,
) -> Any:
    """Resolve ``algorithm`` for ``model`` and run it.

    Selection is capability-checked first, so an unsupported explicit choice
    raises before any sampling loop starts — never mid-generation, and never
    by silently falling back.

    Each algorithm's runner calls its own loop directly.  The choice does not
    round-trip through ``use_cache`` / ``use_block_diffusion``, which is what
    lets a family with no corresponding boolean (discrete flow, continuous)
    register a runner at all.

    Raises:
        ValueError: unknown algorithm, unsupported algorithm, or a registered
            algorithm with no runner.
    """
    resolved = resolve_algorithm(algorithm, model, bd3lm_requested=bd3lm_requested)
    entry = find_algorithm(resolved)
    if entry is None or entry.runner is None:
        raise ValueError(
            f"decoding algorithm {resolved!r} has no runner; register one via "
            "GenerationAlgorithm(runner=...)."
        )
    return entry.runner(model, request)


def __getattr__(name: str) -> dict[str, dict[str, bool]]:
    """Derive the historical algorithm tables from the registry.

    ``DISCRETE_ALGORITHMS`` / ``CANVAS_ALGORITHMS`` / ``ALL_ALGORITHMS`` were
    module-level dicts.  Nothing outside this module reads them, but computing
    them on demand keeps them from becoming a stale second source of truth.
    """
    if name == "ALL_ALGORITHMS":
        return {a.name: dict(a.flags) for a in _ALGORITHMS}
    if name == "DISCRETE_ALGORITHMS":
        return {
            a.name: dict(a.flags) for a in _ALGORITHMS if a.family == "masked_discrete"
        }
    if name == "CANVAS_ALGORITHMS":
        return {a.name: dict(a.flags) for a in _ALGORITHMS if a.family == "canvas"}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
