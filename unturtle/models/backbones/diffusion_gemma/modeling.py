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

"""DiffusionGemma backbone wrapper.

DiffusionGemma is NOT a masked diffusion LM: it denoises a per-block "canvas"
with self-conditioning under entropy/confidence acceptance (no mask token).
This wrapper adds ONLY the unified ``generate(algorithm=...)`` shim — no
masked-diffusion mixins, no config subclass, and the upstream
``model_type = "diffusion_gemma"`` is unchanged (real checkpoints carry it).
The class is deliberately field-free so the loader can ``__class__``-swap a
FastModel-loaded upstream instance (see ``_POST_LOAD_CLASS_SWAPS``).
"""

from __future__ import annotations

from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion


class UnturtleDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion):
    """Unturtle wrapper for ``DiffusionGemmaForBlockDiffusion``.

    Adds a unified ``generate(algorithm=...)`` entry-point while delegating
    entirely to the upstream canvas block-diffusion loop. Masked-diffusion
    algorithms (mdlm / block_decode / bd3lm) raise ``ValueError`` via
    ``resolve_algorithm`` — this family has no mask-token semantics.

    This class is field-free by design: the loader ``__class__``-swaps a
    FastModel-loaded upstream instance onto this class so that real checkpoints
    acquire the shim without re-instantiation.
    """

    def generate(
        self,
        inputs=None,
        *,
        algorithm: str = "auto",
        generation_config=None,
        **kwargs,
    ):
        """Generate via the upstream block-AR canvas diffusion.

        Parameters
        ----------
        inputs:
            Token IDs passed as ``input_ids`` to the upstream ``generate``.
        algorithm:
            ``"auto"`` or ``"block_ar"`` — both delegate to the upstream loop
            verbatim; no vocabulary translation.  Masked-diffusion algorithms
            (``"mdlm"`` / ``"block_decode"`` / ``"bd3lm"``) raise
            ``ValueError`` immediately via ``resolve_algorithm``.
        generation_config:
            A ``DiffusionGemmaGenerationConfig`` (or compatible mapping).
            Forwarded unchanged to the upstream ``generate``.
        **kwargs:
            Forwarded unchanged to the upstream ``generate``.

        See Also
        --------
        transformers.models.diffusion_gemma.DiffusionGemmaGenerationConfig :
            Controls max_denoising_steps, sampler_config, t_min/t_max,
            stability_threshold, and confidence_threshold.
        """
        from unturtle.models.generation.sampler import (
            GenerationRequest,
            dispatch_generation,
        )

        if inputs is None and "input_ids" in kwargs:
            # HF canonical call style: model.generate(input_ids=...).
            inputs = kwargs.pop("input_ids")

        # Unified entry: resolution AND execution go through the registry
        # runner (#186) — the shim never calls the upstream loop itself, so
        # a registered/replaced runner is always the one that executes.
        return dispatch_generation(
            self,
            GenerationRequest(
                inputs=inputs,
                generation_config=generation_config,
                kwargs=dict(kwargs),
            ),
            algorithm=algorithm,
        )

    def _generate_canvas(self, inputs=None, *, generation_config=None, **kwargs):
        """The upstream canvas loop, invoked verbatim — the #186 runner target.

        No algorithm resolution here (dispatch already did it); this is the
        single place the wrapper touches upstream ``generate``.
        """
        return super().generate(
            input_ids=inputs,
            generation_config=generation_config,
            **kwargs,
        )
