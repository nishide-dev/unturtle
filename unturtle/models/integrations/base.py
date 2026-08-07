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

"""
The ``BackboneIntegration`` record.

One integration describes everything the loader needs to know about a model
family that is *specific to that family*: which ``model_type`` strings it
answers to, which Unturtle class (if any) should load it natively, and which
wrapper class (if any) should be swapped in after an upstream load.

Deliberately a small immutable record rather than an abstract base class —
#68 is explicit that no generic ``Backbone`` ABC should be imposed on
``transformers`` model classes.  Integrations describe models; they are not
models.

Every class reference is held as a **zero-argument resolver**, never as the
class itself.  That is what keeps ``import unturtle.models.integrations``
free of heavy backbone imports, and it is what lets a backbone whose
dependencies are missing drop out of the map instead of breaking it.

Named "integration" rather than "adapter" on purpose: ``adapter`` already
means PEFT/LoRA throughout this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class BackboneIntegration:
    """Model-family-specific loading knowledge.

    Args:
        name:              Human-readable family name, used in diagnostics.
        model_types:       ``config.model_type`` strings this family answers
                           to.  Several spellings are normal — Dream's config
                           declares ``"Dream"`` while Hub configs also use
                           ``"dream"``.
        _native_resolver:  Returns the Unturtle class that loads this family
                           natively, bypassing ``trust_remote_code`` Hub code.
                           ``None`` for families that load through upstream
                           (see ``_wrapper_resolver``).
        _wrapper_resolver: Returns a thin wrapper class to install via
                           ``__class__`` assignment *after* an upstream load.
                           Used when upstream owns the implementation and
                           Unturtle only adds a shim.
        capabilities:      Descriptive internal facts (e.g.
                           ``"masked_generation"``).  Not a public ontology;
                           runtime-dependent conditions belong in a predicate,
                           not here.
    """

    name: str
    model_types: tuple[str, ...]
    _native_resolver: Callable[[], Any] | None = None
    _wrapper_resolver: Callable[[], Any] | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)

    @property
    def native_model_cls(self) -> Any | None:
        """Resolve the native class, or ``None`` if unavailable.

        An ``ImportError`` means an optional backbone's dependencies are
        missing; that drops this family from the map rather than failing the
        whole lookup, matching the loader's long-standing behavior.  Other
        exceptions propagate — they indicate a real bug in the backbone.
        """
        if self._native_resolver is None:
            return None
        try:
            return self._native_resolver()
        except ImportError:
            return None

    @property
    def post_load_wrapper_cls(self) -> Any | None:
        """Resolve the post-load wrapper class, or ``None`` if unavailable."""
        if self._wrapper_resolver is None:
            return None
        try:
            return self._wrapper_resolver()
        except ImportError:
            return None

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities
