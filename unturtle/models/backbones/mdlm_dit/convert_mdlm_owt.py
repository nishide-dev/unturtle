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

"""kuleshov-group MDLM checkpoint -> native MDLM-DiT conversion (#130 PR0).

The published mdlm-owt checkpoint (OpenWebText, 1M steps, ``time_conditioning=False``)
is structurally a DITBackbone whose only divergence from Unturtle's native
``MDLMDiTModel`` is the sigma path: upstream runs a zeroed sigma through a
``TimestepEmbedder`` MLP every forward, ours conditions on a learnable constant
vector.  For a time-agnostic checkpoint the two are exactly equivalent under

    cond := sigma_map(timestep_embedding(0))

since upstream's ``c = silu(sigma_map(0))`` then equals our ``c = silu(cond)``.
Everything else maps name-for-name (``backbone.*`` -> ``model.*``).

The conversion refuses rather than guesses: unrecognized source keys raise, a
rotary ``inv_freq`` that differs from the recomputed one raises (a different
RoPE base would otherwise be dropped silently), and ``time_conditioning=True``
checkpoints are rejected outright — collapsing a *live* sigma path at sigma=0
would silently change the model's function.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from .configuration_mdlm_dit import MDLMDiTConfig
from .modeling_mdlm_dit import MDLMDiTForMaskedDiffusionLM

# Structural constant of upstream's TimestepEmbedder (frequency_embedding_size).
FREQUENCY_EMBEDDING_SIZE = 256

_SIGMA_MAP_KEYS = (
    "backbone.sigma_map.mlp.0.weight",
    "backbone.sigma_map.mlp.0.bias",
    "backbone.sigma_map.mlp.2.weight",
    "backbone.sigma_map.mlp.2.bias",
)


def config_from_mdlm_owt(source_config: dict) -> MDLMDiTConfig:
    """Map upstream ``MDLMConfig`` fields onto a native :class:`MDLMDiTConfig`.

    ``mask_token_id`` is ``vocab_size - 1``: mdlm appends the mask token to the
    mask-less gpt2 tokenizer (``diffusion.py``: ``mask_index = vocab_size;
    vocab_size += 1``), so the id lives in the config, not the tokenizer.
    """
    if source_config.get("time_conditioning", False):
        raise ValueError(
            "This checkpoint has time_conditioning=True; collapsing its sigma path "
            "into a constant conditioning vector at sigma=0 would silently change "
            "the model. Only time-agnostic (time_conditioning=False) MDLM "
            "checkpoints convert to the native MDLM-DiT backbone."
        )
    vocab_size = source_config["vocab_size"]
    return MDLMDiTConfig(
        vocab_size=vocab_size,
        hidden_size=source_config["hidden_dim"],
        cond_dim=source_config["cond_dim"],
        num_hidden_layers=source_config["n_blocks"],
        num_attention_heads=source_config["n_heads"],
        dropout=source_config["dropout"],
        max_position_embeddings=source_config["model_length"],
        mask_token_id=vocab_size - 1,
    )


def _collapse_sigma_map(source: dict[str, torch.Tensor]) -> torch.Tensor:
    """``cond := sigma_map(timestep_embedding(0))``.

    ``timestep_embedding(t=0, 256)`` is ``[cos(0)]*128 ++ [sin(0)]*128``
    = ``[1]*128 ++ [0]*128`` — exact, so the collapse introduces no
    approximation beyond running the same two Linears upstream runs.
    """
    emb0 = torch.cat(
        [
            torch.ones(
                FREQUENCY_EMBEDDING_SIZE // 2,
                dtype=source["backbone.sigma_map.mlp.0.weight"].dtype,
            ),
            torch.zeros(
                FREQUENCY_EMBEDDING_SIZE // 2,
                dtype=source["backbone.sigma_map.mlp.0.weight"].dtype,
            ),
        ]
    )
    h = F.linear(
        emb0,
        source["backbone.sigma_map.mlp.0.weight"],
        source["backbone.sigma_map.mlp.0.bias"],
    )
    return F.linear(
        F.silu(h),
        source["backbone.sigma_map.mlp.2.weight"],
        source["backbone.sigma_map.mlp.2.bias"],
    )


def convert_mdlm_state_dict(
    source: dict[str, torch.Tensor], config: MDLMDiTConfig
) -> dict[str, torch.Tensor]:
    """Convert an upstream ``backbone.*`` state dict to native ``model.*`` keys.

    Raises ``ValueError`` on any source key it does not understand and on a
    rotary ``inv_freq`` that differs from the one the native model recomputes.
    """
    remaining = dict(source)

    converted: dict[str, torch.Tensor] = {"model.cond": _collapse_sigma_map(remaining)}
    for key in _SIGMA_MAP_KEYS:
        remaining.pop(key)

    head_dim = config.hidden_size // config.num_attention_heads
    inv_freq = remaining.pop("backbone.rotary_emb.inv_freq")
    expected_inv_freq = 1.0 / (
        10_000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    if not torch.allclose(inv_freq.float(), expected_inv_freq):
        raise ValueError(
            "backbone.rotary_emb.inv_freq differs from the RoPE table the native "
            "model recomputes (base 10000); refusing to drop it silently."
        )

    for key, tensor in remaining.items():
        if not key.startswith("backbone."):
            raise ValueError(f"Unrecognized source key: {key}")
        converted["model." + key.removeprefix("backbone.")] = tensor

    # Fail here, not at load_state_dict, so the error names the offender.
    native_keys = {
        name for name, _ in MDLMDiTForMaskedDiffusionLM(config).named_parameters()
    }
    unknown = sorted(set(converted) - native_keys)
    if unknown:
        raise ValueError(
            "Source keys with no native MDLM-DiT destination: "
            + ", ".join(k.replace("model.", "backbone.", 1) for k in unknown)
        )
    return converted


def build_native_model(
    config: MDLMDiTConfig,
    source: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> MDLMDiTForMaskedDiffusionLM:
    """Build a genuinely native model carrying the converted weights.

    Default dtype is fp32 — the checkpoint's own dtype, loaded bitwise. bf16
    is an explicit conversion requested via ``dtype`` (never a silent cast,
    #112); ``Module.to(dtype)`` converts every floating-point parameter
    unconditionally, and the tests pin that.
    """
    converted = convert_mdlm_state_dict(source, config)
    model = MDLMDiTForMaskedDiffusionLM(config)
    model.load_state_dict(converted, strict=True)
    if dtype is not torch.float32:
        model = model.to(dtype)
    return model


def load_mdlm_owt(
    repo_id: str = "kuleshov-group/mdlm-owt",
    dtype: torch.dtype = torch.float32,
) -> MDLMDiTForMaskedDiffusionLM:
    """Load the published mdlm-owt checkpoint as a native MDLM-DiT model.

    Reads ``config.json`` + ``model.safetensors`` directly — the upstream
    remote code (flash-attn hard dependency) never executes.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config_path = hf_hub_download(repo_id, "config.json")
    weights_path = hf_hub_download(repo_id, "model.safetensors")
    with open(config_path) as f:
        config = config_from_mdlm_owt(json.load(f))
    return build_native_model(config, load_file(weights_path), dtype=dtype)
