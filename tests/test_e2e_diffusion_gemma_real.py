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

"""E2E test: DiffusionGemma real checkpoint via FastDiffusionModel.

This test downloads the official DiffusionGemma checkpoint from HuggingFace and
exercises the unturtle-owned pipeline: FastModel delegation -> post-load class
swap -> unified ``generate(algorithm=...)`` shim dispatching to the upstream
canvas loop.

**Skipped by default.** To run::

    pytest tests/test_e2e_diffusion_gemma_real.py -m "slow and gpu" -v

Requirements:
- CUDA GPU (~18 GB free for the 4-bit load)
- Sufficient disk space (~32 GB checkpoint)

Known upstream limitations (verified 2026-06-12, torch 2.10 / transformers 5.11
/ unsloth 2026.6.2):

- **bnb-4bit MoE generation is broken upstream**: transformers'
  ``integrations/moe.py`` ``_can_use_grouped_mm`` does not check dtype on CUDA,
  so 4-bit-packed expert weights (uint8) reach ``torch.grouped_mm`` and raise
  ``RuntimeError: Expected mat_a to be Float32, BFloat16 or Float16, got Byte``.
  The generation step is therefore ``xfail`` until upstream supports quantized
  experts (unsloth recommends the GGUF/llama.cpp route for DG inference today).
- **bf16 multi-GPU via FastModel leaves meta tensors**: passing
  ``load_in_4bit=False, device_map="auto"`` through unsloth FastModel returns a
  non-materialized (meta) model, and 26B bf16 does not fit a single 48 GB GPU —
  so full-precision generation cannot be exercised here either.
"""

from __future__ import annotations

import pytest
import torch

CHECKPOINT = "google/diffusiongemma-26B-A4B-it"


@pytest.mark.slow
@pytest.mark.gpu
def test_real_checkpoint_load_swap_and_shim_generate():
    """4-bit load succeeds, the class swap installs the shim, and generate
    dispatches through it into the upstream canvas loop (generation itself is
    xfail on the upstream 4-bit MoE grouped_mm gap)."""
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for slow E2E test")

    model, tokenizer = FastDiffusionModel.from_pretrained(CHECKPOINT, load_in_4bit=True)

    # unturtle-owned contract: swap installed the wrapper and removed unsloth's
    # instance-level fast-generate patch so the unified shim wins.
    assert type(model) is UnturtleDiffusionGemmaForBlockDiffusion
    assert "generate" not in model.__dict__
    assert (
        type(model).generate
        is UnturtleDiffusionGemmaForBlockDiffusion.__dict__["generate"]
    )

    # The shim rejects masked algorithms even on the real checkpoint.
    bad_prompt = torch.tensor([[1, 2, 3, 4]], device=model.device)
    with pytest.raises(ValueError):
        model.generate(bad_prompt, algorithm="mdlm", max_new_tokens=4)

    # FastModel returns the multimodal Gemma4 *processor*; its (unsloth-patched)
    # __call__ takes images first, so text must be passed by keyword.
    enc = tokenizer(text="The capital of France is", return_tensors="pt")
    prompt = enc["input_ids"].to(model.device)
    try:
        with torch.no_grad():
            out = model.generate(prompt, max_new_tokens=16)
    except RuntimeError as exc:
        if "got Byte" in str(exc):
            pytest.xfail(
                "upstream: transformers grouped_mm does not support bnb-4bit "
                f"MoE expert weights ({exc})"
            )
        raise
    seq = out.sequences if hasattr(out, "sequences") else out
    text = tokenizer.decode(seq[0], skip_special_tokens=True)
    assert len(text) > 0
