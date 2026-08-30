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

"""Tests for the UnturtleDiffusionGemmaForBlockDiffusion wrapper.

Step-0 findings (CPU generate confirmed working):
  - DiffusionGemmaForBlockDiffusion(config) takes the top-level DiffusionGemmaConfig.
  - vision_config=None triggers AutoModel.from_config(None) → ValueError; a real
    (tiny) Gemma4VisionConfig is required.
  - DiffusionGemmaTextConfig requires num_experts / top_k_experts / moe_intermediate_size
    (all default to None, but the encoder layer always instantiates the router+experts).
  - num_global_key_value_heads must be set; full-attention layers use it for GQA.
  - Minimal working config: hidden_size=32, num_hidden_layers=2, num_experts=2,
    top_k_experts=1, moe_intermediate_size=32, canvas_length=16.
  - Tiny generate call: DiffusionGemmaGenerationConfig(max_new_tokens=8,
    max_denoising_steps=2); output is DiffusionGemmaGenerationOutput with .sequences.
"""

from __future__ import annotations

import pytest
import torch


def _tiny_model():
    """Build a minimal CPU DiffusionGemma model for fast unit tests.

    All field sizes are kept as small as possible (hidden=32, 2 layers, vocab=256,
    2 experts) so that model construction and a short generate pass run in seconds
    on CPU.
    """
    from transformers import Gemma4VisionConfig
    from transformers.models.diffusion_gemma import (
        DiffusionGemmaConfig,
        DiffusionGemmaTextConfig,
    )

    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    vis_cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        pooling_kernel_size=2,
        patch_size=4,
        position_embedding_size=16,
    )

    text_cfg = DiffusionGemmaTextConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        max_position_embeddings=64,
        sliding_window=64,
        num_experts=2,
        top_k_experts=1,
        moe_intermediate_size=32,
    )

    cfg = DiffusionGemmaConfig(
        text_config=text_cfg,
        vision_config=vis_cfg,
        canvas_length=16,
    )

    model = UnturtleDiffusionGemmaForBlockDiffusion(cfg)
    model.eval()
    return model


def _tiny_upstream_model():
    """Build a minimal upstream DiffusionGemmaForBlockDiffusion (not wrapped).

    Uses the same tiny config as _tiny_model() but returns the upstream class
    directly, for testing class swap behavior.
    """
    from transformers import Gemma4VisionConfig
    from transformers.models.diffusion_gemma import (
        DiffusionGemmaConfig,
        DiffusionGemmaForBlockDiffusion,
        DiffusionGemmaTextConfig,
    )

    vis_cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        pooling_kernel_size=2,
        patch_size=4,
        position_embedding_size=16,
    )

    text_cfg = DiffusionGemmaTextConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        max_position_embeddings=64,
        sliding_window=64,
        num_experts=2,
        top_k_experts=1,
        moe_intermediate_size=32,
    )

    cfg = DiffusionGemmaConfig(
        text_config=text_cfg,
        vision_config=vis_cfg,
        canvas_length=16,
    )

    model = DiffusionGemmaForBlockDiffusion(cfg)
    model.eval()
    return model


def _tiny_gen_cfg(max_denoising_steps: int = 2, max_new_tokens: int = 8):
    from transformers.models.diffusion_gemma import DiffusionGemmaGenerationConfig

    cfg = DiffusionGemmaGenerationConfig(
        max_new_tokens=max_new_tokens,
        max_denoising_steps=max_denoising_steps,
    )
    cfg.pad_token_id = 0
    cfg.eos_token_id = 1
    return cfg


def test_generate_auto_runs_block_ar():
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model.generate(prompt, generation_config=_tiny_gen_cfg())
    seq = out.sequences if hasattr(out, "sequences") else out
    assert seq.shape[0] == 1
    assert seq.shape[-1] >= prompt.shape[-1]


@pytest.mark.parametrize("algorithm", ["mdlm", "block_decode", "bd3lm"])
def test_generate_masked_algorithms_raise(algorithm):
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(ValueError):
        model.generate(prompt, algorithm=algorithm, generation_config=_tiny_gen_cfg())


def test_generate_explicit_block_ar_matches_auto():
    """Explicit ``algorithm='block_ar'`` must produce the same result as ``'auto'``."""
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    gen_cfg_auto = _tiny_gen_cfg()
    gen_cfg_explicit = _tiny_gen_cfg()

    with torch.no_grad():
        # Use same seed so the stochastic denoising loop produces identical outputs.
        torch.manual_seed(42)
        out_auto = model.generate(
            prompt, algorithm="auto", generation_config=gen_cfg_auto
        )
        torch.manual_seed(42)
        out_explicit = model.generate(
            prompt, algorithm="block_ar", generation_config=gen_cfg_explicit
        )

    seq_auto = out_auto.sequences if hasattr(out_auto, "sequences") else out_auto
    seq_explicit = (
        out_explicit.sequences if hasattr(out_explicit, "sequences") else out_explicit
    )
    assert torch.equal(seq_auto, seq_explicit)


def test_generate_is_shim_not_upstream():
    from transformers.models.diffusion_gemma import DiffusionGemmaGenerationMixin

    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    assert (
        UnturtleDiffusionGemmaForBlockDiffusion.generate
        is not DiffusionGemmaGenerationMixin.generate
    )


def test_model_type_unchanged():
    """Wrapping must NOT change the upstream model_type (real checkpoints carry it)."""
    model = _tiny_model()
    assert model.config.model_type == "diffusion_gemma"


def test_no_masked_diffusion_mixin():
    """The wrapper must NOT inherit any masked-diffusion mixin."""
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    # These mixins carry mask-token semantics and must not be in the MRO.
    from unturtle.models.generation.diffusion_generation_utils import (
        MaskedDiffusionGenerationMixin,
    )

    assert not issubclass(
        UnturtleDiffusionGemmaForBlockDiffusion, MaskedDiffusionGenerationMixin
    ), "Wrapper must not inherit MaskedDiffusionGenerationMixin"


def test_generate_keyword_input_ids():
    """model.generate(input_ids=...) must work (HF canonical call style)."""
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model.generate(input_ids=prompt, generation_config=_tiny_gen_cfg())
    seq = out.sequences if hasattr(out, "sequences") else out
    assert seq.shape[0] == 1
    assert seq.shape[-1] >= prompt.shape[-1]


def test_wrapper_resolver_registered_for_diffusion_gemma():
    from unturtle.models import loading
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    resolver = loading._POST_LOAD_CLASS_SWAPS.get("diffusion_gemma")
    assert resolver is not None
    assert resolver() is UnturtleDiffusionGemmaForBlockDiffusion


def test_runtime_class_swap_is_gone():
    """#186: the load owns the class — no swap API exists any more, and an
    upstream-class instance keeps its class forever."""
    from unturtle import fast_diffusion_model as fdm
    from unturtle.models import loading

    assert not hasattr(fdm, "_apply_post_load_class_swap")
    assert not hasattr(fdm, "_POST_LOAD_CLASS_SWAPS")
    assert not hasattr(loading, "_apply_post_load_class_swap")


def test_block_ar_runner_ignores_instance_generate_on_upstream_class():
    """unsloth-style instance-level generate patches can no longer hijack the
    canvas loop: the #186 runner invokes the CLASS-level upstream generate on
    an un-wrapped model, without any class mutation."""
    from unturtle.models.generation.sampler import (
        GenerationRequest,
        dispatch_generation,
    )

    model = _tiny_upstream_model()
    upstream_cls = type(model)
    sentinel_called = []

    def _fake_unsloth_generate(*a, **k):
        sentinel_called.append(True)
        return "hijacked"

    model.__dict__["generate"] = _fake_unsloth_generate  # instance-level patch
    out = dispatch_generation(
        model,
        GenerationRequest(
            inputs=torch.tensor([[1, 2, 3, 4]]),
            generation_config=_tiny_gen_cfg(),
        ),
        algorithm="block_ar",
    )
    assert not sentinel_called
    assert type(model) is upstream_cls  # never restamped
    assert out is not None and not isinstance(out, str)
