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
Discrete flow-matching generation as a mixin (#65 public-API slice).

Puts ``solve_discrete_flow`` (#95/#110) behind the unified
``model.generate(algorithm="dfm")`` surface.  The capability is **opt-in**
(``supports_dfm_generation = True``): DFM sampling quality on real backbones
is only tiny-control-validated, so no masked family is claimed silently —
adopting the mixin IS the declaration.

``dfm_denoiser(x_t, t, h)`` is the overridable seam.  The default is the
time-agnostic forward (logits from ids; ``t``/``h`` unused — an ordinary DFM
model in FS-DFM's terms); step-aware models (FS-DFM's ``theta(x_t, t; h)``,
e.g. through ``StepAwareWrapper``) override it.  No ``step_aware`` flag —
the seam carries the whole signature either way.

Generation is unconditional in this slice: DFM *conditioning* is outside
#65's validated scope, so a prompt is rejected loudly rather than ignored.
"""

from __future__ import annotations

from typing import Any

import torch

from .dfm_solver import solve_discrete_flow


class DiscreteFlowGenerationMixin:
    """Adds registry-dispatched DFM sampling to a model with a logits forward."""

    supports_dfm_generation = True

    def dfm_denoiser(
        self, x_t: torch.Tensor, t: torch.Tensor, h: float
    ) -> torch.Tensor:
        """The ``(x_t, t, h) -> logits`` seam ``solve_discrete_flow`` calls.

        Default: the time-agnostic forward.  ``t`` and ``h`` are part of the
        signature so FS-DFM step-aware models can override without changing
        the solver contract.
        """
        with torch.no_grad():
            return self(input_ids=x_t).logits

    def _dfm_source(
        self,
        *,
        batch_size: int,
        seq_len: int,
        source: str,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        if source == "mask":
            mask_token_id = getattr(self.config, "mask_token_id", None)
            if mask_token_id is None:
                raise ValueError(
                    "source='mask' requires config.mask_token_id; silently "
                    "falling back to a uniform source would sample from a "
                    "different process than the one the model was trained "
                    "on. Pass source='uniform' explicitly if that is the "
                    "intent."
                )
            return torch.full(
                (batch_size, seq_len), mask_token_id, dtype=torch.long, device=device
            )
        if source == "uniform":
            # Drawn on the generator's device then transferred: a CUDA model
            # with a CPU generator (or vice versa) must not make the source
            # draw and the solver's draws demand contradictory devices.
            return torch.randint(
                0,
                self.config.vocab_size,
                (batch_size, seq_len),
                generator=generator,
                device=device if generator is None else generator.device,
            ).to(device)
        raise ValueError(f"unknown source {source!r}; choose 'mask' or 'uniform'")

    def _generate_dfm(
        self,
        inputs: Any = None,
        *,
        batch_size: int = 1,
        steps: int = 8,
        seq_len: int | None = None,
        source: str = "mask",
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if inputs is not None:
            raise ValueError(
                "DFM generation is unconditional in this slice; a prompt "
                "would be silently ignored, so it is rejected instead"
            )
        length = seq_len or self.config.max_position_embeddings
        device = next(self.parameters()).device
        x_0 = self._dfm_source(
            batch_size=batch_size,
            seq_len=length,
            source=source,
            generator=generator,
            device=device,
        )
        # no_grad at the CALL SITE, not only inside the default seam: an
        # FS-DFM override that forgot it would otherwise pay ~2.7x activation
        # memory per step (measured) for gradients nothing consumes.
        with torch.no_grad():
            return solve_discrete_flow(
                self.dfm_denoiser,
                x_0,
                steps=steps,
                temperature=temperature,
                generator=generator,
            )

    def generate(  # type: ignore[override]
        self,
        inputs: Any = None,
        algorithm: str = "auto",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Route ``dfm`` here; delegate everything else down the MRO.

        This mixin is designed to be ADOPTED by masked models, whose own
        ``generate`` carries a config/flag preamble this signature does not
        (dropping it resolved ``mdlm`` correctly and then crashed in the
        call convention — the #120 review's Critical).  So: resolve first,
        run only ``dfm`` through the registry request, and hand any other
        algorithm to ``super().generate`` untouched.  On a standalone model
        the non-dfm resolve raises before delegation, so plain
        ``PreTrainedModel`` supers are never reached with masked kwargs.
        """
        from .sampler import GenerationRequest, dispatch_generation, resolve_algorithm

        resolved = resolve_algorithm(
            algorithm,
            self,
            bd3lm_requested=bool(kwargs.get("use_block_diffusion", False)),
        )
        if resolved != "dfm":
            return super().generate(inputs, algorithm=algorithm, **kwargs)
        return dispatch_generation(
            self,
            GenerationRequest(inputs=inputs, kwargs=kwargs),
            resolved,
        )


__all__ = ["DiscreteFlowGenerationMixin"]
