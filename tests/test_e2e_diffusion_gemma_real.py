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

"""E2E test: DiffusionGemma real checkpoint loads and generates via FastDiffusionModel.

This test downloads the official DiffusionGemma checkpoint from HuggingFace and
exercises the full load + inference pipeline.

**Skipped by default.** To run::

    pytest tests/test_e2e_diffusion_gemma_real.py -m "slow and gpu" -v

Requirements:
- CUDA GPU
- Sufficient disk space (~32 GB for 26B 4-bit checkpoint)
"""

from __future__ import annotations

import pytest
import torch

CHECKPOINT = "google/diffusiongemma-26B-A4B-it"


@pytest.mark.slow
@pytest.mark.gpu
def test_real_checkpoint_loads_via_fastmodel_and_generates():
    """DiffusionGemma checkpoint loads via FastDiffusionModel and generates successfully."""
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for slow E2E test")

    model, tokenizer = FastDiffusionModel.from_pretrained(CHECKPOINT, load_in_4bit=True)
    assert type(model) is UnturtleDiffusionGemmaForBlockDiffusion
    prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(
        model.device
    )
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=16)
    seq = out.sequences if hasattr(out, "sequences") else out
    text = tokenizer.decode(seq[0], skip_special_tokens=True)
    assert len(text) > 0
