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

"""Typed result/report contracts for loading and PEFT patching (#185 PR 0).

Descriptive only: these types are populated from OBSERVED execution by
``unturtle.fast_diffusion_model`` and never change behavior. Three rules,
each frozen by #184/#177:

- ``SupportResult.status`` is three-valued. ``unverified`` (e.g. the dtype
  gate's ``input_embedding_unresolvable``) is a first-class state that must
  never be collapsed into ``supported`` or a generic ``unsupported``.
- ``PatchReport`` separates what was *requested*, what is *applied* (observed
  by callable identity after patching), what was *skipped* by per-target
  gates, and a *fallback* (the whole fast set withheld with a typed reason).
- An installed callable is never ``live``. ``LivenessReport`` is filled only
  by an actual probe forward (and backward, for training claims) whose
  counters are attached to the very targets the report lists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SupportStatus = Literal["supported", "unsupported", "unverified"]
SUPPORT_STATUSES: tuple[str, ...] = ("supported", "unsupported", "unverified")

#: Fast-path kinds a family patcher can install. ``requested`` uses this
#: vocabulary; ``applied``/``skipped`` map each kind to module paths.
PATCH_KINDS: tuple[str, ...] = ("qkv", "o", "mlp", "attention_forward", "rope")


@dataclass(frozen=True)
class SupportResult:
    """Whether a fast path may execute on this model — with a typed reason.

    ``unverified`` means the structure could not be inspected; production may
    still proceed fail-open (per-target gates apply), but the report must say
    that nothing was proven.
    """

    status: SupportStatus
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in SUPPORT_STATUSES:
            raise ValueError(
                f"invalid support status {self.status!r}; expected {SUPPORT_STATUSES}"
            )
        if self.status != "supported" and not self.reason:
            raise ValueError(f"status {self.status!r} requires a typed reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LivenessReport:
    """Invocation evidence from an actual probe run.

    ``forward`` / ``backward`` map ``"<module path>:<kind>"`` targets to the
    number of times the *installed fast callable on that module* ran. ``live``
    is true only when every applied target was invoked at least once in the
    forward probe (and, when a backward was requested, every LoRA parameter
    under an applied target received a gradient).
    """

    forward: dict[str, int]
    backward: dict[str, int] | None
    live: bool
    forward_live: bool
    backward_live: bool | None
    probe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatchReport:
    """What the PEFT/fast-path entry point did to ONE model, observed."""

    family: str
    model_type: str
    support: SupportResult
    requested: tuple[str, ...]
    applied: dict[str, tuple[str, ...]]
    skipped: dict[str, tuple[str, ...]]
    fallback: str | None
    applicability: dict[str, Any]
    liveness: LivenessReport | None = None
    warnings: tuple[str, ...] = ()

    @property
    def applied_targets(self) -> tuple[str, ...]:
        return tuple(
            f"{path}:{kind}"
            for kind, paths in sorted(self.applied.items())
            for path in paths
        )

    @property
    def is_fast(self) -> bool:
        """Some fast callable is installed. NOT liveness."""
        return any(self.applied.values())

    @property
    def live(self) -> bool:
        """Only a probe can make this true."""
        return bool(self.liveness and self.liveness.live)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model_type": self.model_type,
            "support": self.support.to_dict(),
            "requested": list(self.requested),
            "applied": {k: list(v) for k, v in self.applied.items()},
            "skipped": {k: list(v) for k, v in self.skipped.items()},
            "fallback": self.fallback,
            "applicability": dict(self.applicability),
            "liveness": self.liveness.to_dict() if self.liveness else None,
            "warnings": list(self.warnings),
            "is_fast": self.is_fast,
            "live": self.live,
        }


@dataclass(frozen=True)
class PreparedPeftModel:
    """The PEFT-preparation boundary (#185 PR 2).

    ``model`` is LoRA-wrapped and training-ready (stubs installed, k-bit
    preparation or gradient checkpointing applied, adapters created under the
    #188 forked-RNG contract) but carries NO fast-path optimization yet — the
    façade hands it to the family optimization provider, which may decline
    with a typed fallback without affecting preparation.
    """

    model: Any
    lora_config: Any
    quantized: bool
    kbit_prepared: bool
    gradient_checkpointing: Any
    random_state: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantized": self.quantized,
            "kbit_prepared": self.kbit_prepared,
            "gradient_checkpointing": self.gradient_checkpointing,
            "random_state": self.random_state,
        }


@dataclass
class LoadedModel:
    """``from_pretrained`` result with its provenance."""

    model: Any
    tokenizer: Any
    integration: str | None
    load_path: Literal[
        "native", "upstream", "auto", "explicit_class", "adapter", "unknown"
    ]
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[Any, Any]:
        """The compatibility return shape of ``FastDiffusionModel.from_pretrained``."""
        return self.model, self.tokenizer


__all__ = [
    "PATCH_KINDS",
    "SUPPORT_STATUSES",
    "LivenessReport",
    "LoadedModel",
    "PatchReport",
    "SupportResult",
]
