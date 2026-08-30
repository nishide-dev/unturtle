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
A family needs a name, a capability probe, and a runner — nothing else.  In particular
the core never reads ``use_cache`` or ``use_block_diffusion``: those live on the masked
algorithms' own registrations, so a continuous or flow family is never handed a concept
borrowed from masked diffusion.  ``auto_priority`` is an explicit number rather than
registration order, so a newly registered algorithm does **not** outrank the masked ones
merely by existing — #69 calls that out specifically.

Note the flags remain *user-visible* ``MaskedDiffusionGenerationConfig`` fields with
their own cross-validation (``use_block_diffusion`` and ``use_cache`` are mutually
exclusive), and ``generate`` still injects the resolved algorithm's flags so a caller
setting them directly keeps working.  De-masking the registry core is not the same as
deleting the flags.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    A hybrid-attention model (#127) is excluded outright: the eq.-(3) mask
    is square by contract (hybrid + KV cache is deliberately undefined), and
    the cache loop threads no prompt boundary — so picking block-decode
    would silently decode under the train/inference topology mismatch the
    boundary threading exists to close.  ``auto`` falls through to mdlm.
    """
    if getattr(getattr(model, "config", None), "hybrid_attention", False):
        return False
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
    if getattr(getattr(model, "config", None), "hybrid_attention", False):
        return (
            f"{type(model).__name__} has hybrid_attention=True: block-decode "
            "would decode with a KV cache the square eq.-(3) mask cannot "
            "express, silently dropping the prompt topology the model was "
            "trained with (#127). Use algorithm='mdlm'."
        )
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


def _call_sampling_loop(method: Any, request: GenerationRequest) -> Any:
    """Invoke a sampling loop through the DECLARED loop contract (#186).

    Every registered masked sampling loop conforms to one explicit shape::

        loop(input_ids, attention_mask=..., generation_config=..., **options)

    (`_sample`, `_sample_with_cache` and `_sample_block_diffusion` across all
    families take exactly these as positional-or-keyword parameters; Dream's
    extra hook parameters default to ``None`` and are normalized inside its
    loop.)  The previous implementation *guessed* the call shape with
    ``inspect.signature`` and positionally filled unknown required parameters
    with ``None`` — a signature-hiding wrapper (plain ``*args, **kwargs``
    decorator) silently changed the binding (#184 evidence, #186).  Explicit
    keywords are immune to wrapping and fail loudly on a non-conforming loop.
    """
    kwargs = dict(request.kwargs)
    attention_mask = kwargs.pop("attention_mask", None)
    return method(
        request.inputs,
        attention_mask=attention_mask,
        generation_config=request.generation_config,
        **kwargs,
    )


def _run_mdlm(model: Any, request: GenerationRequest) -> Any:
    return _call_sampling_loop(model._sample, request)


def _run_block_decode(model: Any, request: GenerationRequest) -> Any:
    return _call_sampling_loop(model._sample_with_cache, request)


def _run_bd3lm(model: Any, request: GenerationRequest) -> Any:
    return _call_sampling_loop(model._sample_block_diffusion, request)


def _supports_flowlm(model: Any) -> bool:
    """True for the FlowLM continuous prototype (#66).

    Attribute-based so this module never imports ``unturtle.models.latent``
    (registrations are declared centrally; families do not self-register).
    """
    return getattr(model, "supports_flowlm_generation", False) is True


def _flowlm_unsupported(model: Any) -> str:
    return (
        f"model {type(model).__name__} does not support 'flowlm' "
        "(continuous average-velocity sampling; requires a FlowLM prototype "
        "model with a codec and denoiser). Masked-diffusion models should "
        "use 'mdlm' / 'block_decode' / 'bd3lm' instead."
    )


def _run_flowlm(model: Any, request: GenerationRequest) -> Any:
    # inputs is forwarded so the (unconditional) prototype can REJECT a
    # prompt rather than silently ignoring it.
    return model._generate_flowlm(request.inputs, **request.kwargs)


def _supports_dfm(model: Any) -> bool:
    """True for models that opted into discrete flow-matching sampling (#65).

    Opt-in on purpose: any masked model could run the jump process
    mechanically, but DFM quality is only tiny-control-validated, and the
    registry must not present a research path as a supported capability.
    """
    return getattr(model, "supports_dfm_generation", False) is True


def _dfm_unsupported(model: Any) -> str:
    return (
        f"model {type(model).__name__} does not support 'dfm' (discrete "
        "flow-matching jump-process sampling; requires "
        "DiscreteFlowGenerationMixin). Masked-diffusion models should use "
        "'mdlm' / 'block_decode' / 'bd3lm' instead."
    )


def _run_dfm(model: Any, request: GenerationRequest) -> Any:
    return model._generate_dfm(request.inputs, **request.kwargs)


def _supports_ladiff(model: Any) -> bool:
    """True for the LaDiff latent-guided prototype (#117); attribute-based
    (this module never imports ``unturtle.models.latent``)."""
    return getattr(model, "supports_ladiff_generation", False) is True


def _ladiff_unsupported(model: Any) -> str:
    return (
        f"model {type(model).__name__} does not support 'ladiff' "
        "(latent-guided decoding; requires a LaDiff prototype bundling a "
        "codec, latent prior and latent-conditioned masked decoder)."
    )


def _run_ladiff(model: Any, request: GenerationRequest) -> Any:
    # inputs forwarded so the prototype can reject a prompt loudly.
    return model._generate_ladiff(request.inputs, **request.kwargs)


def _run_block_ar(model: Any, request: GenerationRequest) -> Any:
    """Run the canvas family's upstream generation loop, explicitly (#186).

    The loop belongs to upstream ``transformers``; Unturtle only selects it.
    The wrapper class exposes it as ``_generate_canvas`` (a direct upstream
    ``generate`` delegation with NO algorithm resolution), so dispatch cannot
    re-enter itself; a model still carrying the plain upstream class runs its
    class-level ``generate`` directly — class-level on purpose, so an
    instance-level shim (e.g. unsloth's fast-generate wrapper) cannot hijack
    the canvas loop.
    """
    canvas = getattr(type(model), "_generate_canvas", None)
    if canvas is not None:
        return canvas(
            model,
            request.inputs,
            generation_config=request.generation_config,
            **request.kwargs,
        )
    return type(model).generate(
        model,
        input_ids=request.inputs,
        generation_config=request.generation_config,
        **request.kwargs,
    )


def _builtin_algorithms() -> list[GenerationAlgorithm]:
    """The builtin algorithm set, in its frozen registration order (#142)."""
    return [
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
        GenerationAlgorithm(
            name="flowlm",
            family="continuous_flow",
            supports=_supports_flowlm,
            # No masked flags: a continuous family is never described in
            # use_cache/use_block_diffusion terms (#66 acceptance criterion).
            flags={},
            auto_priority=50,
            unsupported_message=_flowlm_unsupported,
            runner=_run_flowlm,
        ),
        GenerationAlgorithm(
            name="ladiff",
            family="latent_guided",
            supports=_supports_ladiff,
            # Hybrid family (continuous latent prior guiding a discrete masked
            # loop) — still no masked booleans; the discrete loop is the
            # prototype's own, not a flagged variant of the masked family.
            flags={},
            auto_priority=60,
            unsupported_message=_ladiff_unsupported,
            runner=_run_ladiff,
        ),
        GenerationAlgorithm(
            name="dfm",
            family="discrete_flow",
            supports=_supports_dfm,
            # The family the registry docstring reserved from day one; no masked
            # booleans — the jump process is its own loop (#65).
            flags={},
            auto_priority=70,
            unsupported_message=_dfm_unsupported,
            runner=_run_dfm,
        ),
    ]


def populate_generation_registry(hub) -> None:
    """Explicit builtin bootstrap for a RegistryHub (#142).

    Importing this module registers nothing; only this call (or the
    module-level compatibility APIs, which bootstrap the default hub) does.
    """
    for algorithm in _builtin_algorithms():
        hub.generation_algorithms.register(algorithm)


def _default_algorithms() -> list[GenerationAlgorithm]:
    """The default hub's LIVE backing list (also served as `_ALGORITHMS` for
    long-standing white-box tests that reorder/insert into it directly)."""
    from unturtle.registry import ensure_default_hub

    return ensure_default_hub().generation_algorithms._items


def iter_algorithms() -> tuple[GenerationAlgorithm, ...]:
    """Every registered algorithm, in registration order."""
    return tuple(_default_algorithms())


def _find_in(
    algorithms: Sequence[GenerationAlgorithm], name: str
) -> GenerationAlgorithm | None:
    for algorithm in algorithms:
        if algorithm.name == name:
            return algorithm
    return None


def find_algorithm(name: str) -> GenerationAlgorithm | None:
    """The registered algorithm called ``name``, or ``None``."""
    return _find_in(_default_algorithms(), name)


def register_algorithm(algorithm: GenerationAlgorithm) -> None:
    """Add an algorithm to the registry.

    Raises:
        ValueError: if the name is already taken.  Two algorithms sharing a name
            would make selection depend on registration order.
    """
    if find_algorithm(algorithm.name) is not None:
        raise ValueError(f"decoding algorithm {algorithm.name!r} is already registered")
    from unturtle.registry import ensure_default_hub

    ensure_default_hub().generation_algorithms.register(algorithm)


def _unregister_algorithm(algorithm: GenerationAlgorithm) -> None:
    """Remove an algorithm.  Test/teardown helper, not a public API.

    Removes by identity: ``GenerationAlgorithm`` is a frozen dataclass with value
    equality, so removing by value could drop a different, equal-looking entry.
    """
    from unturtle.registry import ensure_default_hub

    ensure_default_hub().generation_algorithms.unregister(algorithm)


def _algorithm_names() -> list[str]:
    return sorted(a.name for a in _default_algorithms())


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


def resolve_algorithm_from(
    algorithms: Sequence[GenerationAlgorithm],
    algorithm: str,
    model: Any,
    *,
    bd3lm_requested: bool,
) -> str:
    """Resolve ``algorithm`` against an explicit algorithm collection.

    The single implementation behind both the module-level
    ``resolve_algorithm`` (default hub) and ``RegistryHub.resolve_generation``
    (isolated hubs) — #142's differential contract holds by construction.

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
        canvas = _find_in(algorithms, "block_ar")
        if canvas is not None and canvas.supports(model):
            return canvas.name

        if bd3lm_requested:
            # Requested explicitly, so an incapable model is an error rather
            # than a reason to quietly pick something else.
            bd3lm = _find_in(algorithms, "bd3lm")
            if bd3lm is None or not bd3lm.supports(model):
                raise ValueError(_bd3lm_unsupported(model))
            return bd3lm.name

        for entry in sorted(algorithms, key=lambda a: a.auto_priority):
            if entry.auto_eligible and entry.supports(model):
                return entry.name

        # Derived from the registry rather than naming the masked hooks: a
        # model failing only a newly registered family's probe should not be
        # told to implement `_sample`.  Opt-in algorithms are included — a
        # bd3lm-capable model reaching here needs to be told bd3lm exists and
        # how to ask for it, not that nothing works.
        lines = []
        available: list[str] = []
        for entry in sorted(algorithms, key=lambda a: a.auto_priority):
            if entry.supports(model):
                # Reachable, but only on request: `auto` skipped it because it
                # is not auto-eligible.  Saying "does not support" here would
                # be flatly wrong, and would hide the one usable option.
                available.append(entry.name)
                lines.append(
                    f"  - {entry.name} ({entry.family}): supported, but never "
                    f"selected automatically — pass algorithm={entry.name!r}."
                )
            else:
                lines.append(
                    f"  - {entry.name} ({entry.family}): "
                    f"{entry.describe_unsupported(model)}"
                )

        if available:
            headline = (
                f"{type(model).__name__} supports "
                f"{', '.join(available)}, but `auto` selects none of them."
            )
        else:
            headline = (
                f"{type(model).__name__} supports none of the registered "
                "decoding algorithms."
            )
        raise ValueError(headline + "\n" + "\n".join(lines))

    entry = _find_in(algorithms, algorithm)
    if entry is None:
        raise ValueError(
            f"Unknown decoding algorithm {algorithm!r}. "
            f"Supported: {sorted(a.name for a in algorithms)} (or 'auto')."
        )
    if not entry.supports(model):
        raise ValueError(entry.describe_unsupported(model))
    return entry.name


def resolve_algorithm(algorithm: str, model: Any, *, bd3lm_requested: bool) -> str:
    """Resolve against the default hub (see ``resolve_algorithm_from``)."""
    return resolve_algorithm_from(
        _default_algorithms(), algorithm, model, bd3lm_requested=bd3lm_requested
    )


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
    return dispatch_generation_from(
        _default_algorithms(),
        model,
        request,
        algorithm,
        bd3lm_requested=bd3lm_requested,
    )


def dispatch_generation_from(
    algorithms: Sequence[GenerationAlgorithm],
    model: Any,
    request: GenerationRequest,
    algorithm: str = "auto",
    *,
    bd3lm_requested: bool = False,
) -> Any:
    """Dispatch against an explicit algorithm collection (#143 seam).

    The single implementation behind both the module-level
    ``dispatch_generation`` (default hub) and
    ``RegistryHub.dispatch_generation`` (isolated hubs) — resolution, runner
    lookup, and every capability error come from the SAME collection, so a
    hub-registered algorithm can never fall back to the default hub."""
    resolved = resolve_algorithm_from(
        algorithms, algorithm, model, bd3lm_requested=bd3lm_requested
    )
    entry = _find_in(algorithms, resolved)
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
    if name == "_ALGORITHMS":
        # The LIVE default-hub backing list — the long-standing white-box test
        # seam (order shuffles, direct inserts) keeps working unchanged.
        return _default_algorithms()
    if name == "ALL_ALGORITHMS":
        return {a.name: dict(a.flags) for a in _default_algorithms()}
    if name == "DISCRETE_ALGORITHMS":
        return {
            a.name: dict(a.flags)
            for a in _default_algorithms()
            if a.family == "masked_discrete"
        }
    if name == "CANVAS_ALGORITHMS":
        return {
            a.name: dict(a.flags) for a in _default_algorithms() if a.family == "canvas"
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
