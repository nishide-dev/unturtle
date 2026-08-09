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
AR-checkpoint loading for the Tiny-A2D conversion (#63).

A Tiny-A2D model is a thin adapter over the corresponding upstream backbone:
same modules, same tensor names, different *behaviour* (bidirectional
attention, masked-diffusion objective).  Conversion from a pretrained AR
checkpoint is therefore a load-and-rehome, not a surgery — and the loader's
job is to make the two #107 failure modes impossible by construction:

- **`model_type` proves nothing about the head.**  Resolution checks the
  checkpoint's ``architectures`` field against the concrete upstream class
  the recipe expects, *before* any weights load.  A spoofed or mismatched
  checkpoint is rejected, never class-stamped into a wrapper.
- **No generic Auto\\* fallback.**  An unmapped ``model_type`` has no
  Tiny-A2D recipe; the answer is a loud error, not whatever head
  ``AutoModel`` resolves.  The mapping lives in :func:`ar_head_classes`, an
  Unturtle-owned seam tests patch directly (never the transformers module —
  unsloth replaces ``sys.modules["transformers"]``, making such patches
  silently inert).

**Mask token establishment** is the one place the recipe must invent
something: no AR checkpoint carries a mask token.  Unturtle's choice,
recorded as such: ``mask_token_id=None`` extends the vocabulary by one row
(the new embedding initialized to the mean of the existing rows, a standard
new-token init) and records the new id on the config; an explicit id reuses
an existing vocabulary slot unchanged.  Either way ``config.mask_token_id``
is authoritative afterwards — per the repo gotcha, real-checkpoint mask ids
come from ``model.config``, not the tokenizer.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def ar_head_classes() -> dict[str, tuple[Any, Any, Any]]:
    """``model_type`` → ``(upstream head, tiny config, tiny model)``.

    Resolved at call time, Unturtle-owned: the #107 seam pattern.  Adding a
    family here is the whole registration.
    """
    from transformers import (
        LlamaForCausalLM,
        Qwen2ForCausalLM,
        Qwen3ForCausalLM,
    )

    from .modeling_llama import TinyA2DLlamaConfig, TinyA2DLlamaLMHeadModel
    from .modeling_qwen2 import TinyA2DQwen2Config, TinyA2DQwen2LMHeadModel
    from .modeling_qwen3 import TinyA2DQwen3Config, TinyA2DQwen3LMHeadModel

    return {
        "llama": (LlamaForCausalLM, TinyA2DLlamaConfig, TinyA2DLlamaLMHeadModel),
        "qwen2": (Qwen2ForCausalLM, TinyA2DQwen2Config, TinyA2DQwen2LMHeadModel),
        "qwen3": (Qwen3ForCausalLM, TinyA2DQwen3Config, TinyA2DQwen3LMHeadModel),
    }


def convert_ar_model(
    ar_model: Any,
    *,
    mask_token_id: Optional[int] = None,
    hybrid_attention: bool = False,
    **config_overrides: Any,
) -> Any:
    """Rehome a loaded AR model as its Tiny-A2D counterpart.

    The dependency-injection entry: callers who already hold the AR model
    (tests, in-memory pipelines) skip checkpoint resolution entirely.
    Every checkpoint tensor is preserved bit-for-bit — the conversion's
    adaptation is behavioural (the Tiny-A2D forward replaces the causal mask),
    and a silent re-init anywhere would make the planned hybrid-vs-baseline
    benchmark compare noise.

    Args:
        ar_model:        A loaded upstream causal-LM head from a mapped
                         family.
        mask_token_id:   ``None`` mints a new token (vocabulary extended by
                         one, new row = mean of existing embeddings);
                         an int reuses that vocabulary slot.
        hybrid_attention: Forwarded to the Tiny-A2D config (#63 slice B).

    Returns:
        The Tiny-A2D LM-head model carrying the checkpoint weights.
    """
    mapping = ar_head_classes()
    model_type = getattr(ar_model.config, "model_type", None)
    entry = mapping.get(model_type)
    if entry is None:
        raise ValueError(
            f"no Tiny-A2D recipe for model_type {model_type!r}; supported: "
            f"{sorted(mapping)}"
        )
    upstream_cls, config_cls, model_cls = entry
    if not isinstance(ar_model, upstream_cls):
        raise ValueError(
            f"expected a {upstream_cls.__name__} for model_type "
            f"{model_type!r}, got {type(ar_model).__name__}; converting a "
            "structurally different head produces a chimera (#107)"
        )

    if mask_token_id is None:
        # Mint a mask token: one new row, mean-initialized, appended so every
        # original row keeps its index and its values.
        original_vocab = ar_model.config.vocab_size
        ar_model.resize_token_embeddings(original_vocab + 1)
        with torch.no_grad():
            embeddings = ar_model.get_input_embeddings().weight
            embeddings[original_vocab] = embeddings[:original_vocab].mean(dim=0)
            output = ar_model.get_output_embeddings()
            if output is not None and output.weight.data_ptr() != (
                embeddings.data_ptr()
            ):
                output.weight[original_vocab] = output.weight[:original_vocab].mean(
                    dim=0
                )
        mask_token_id = original_vocab
    elif not 0 <= mask_token_id < ar_model.config.vocab_size:
        raise ValueError(
            f"mask_token_id must satisfy 0 <= id < vocab_size="
            f"{ar_model.config.vocab_size}, got {mask_token_id}"
        )

    config_payload = ar_model.config.to_dict()
    for key in ("model_type", "architectures", "torch_dtype", "dtype"):
        config_payload.pop(key, None)
    config = config_cls(
        hybrid_attention=hybrid_attention, **{**config_payload, **config_overrides}
    )
    config.mask_token_id = mask_token_id

    # model_cls(config) constructs in torch's default dtype, and
    # load_state_dict casts *source to destination* — without the explicit
    # cast a bf16 checkpoint silently widens to fp32, doubling its memory.
    converted = model_cls(config).to(next(ar_model.parameters()).dtype)
    # strict=True: the Tiny-A2D module tree mirrors upstream exactly, so any
    # missing or unexpected key is a structural drift worth failing on.
    converted.load_state_dict(ar_model.state_dict(), strict=True)
    return converted


def load_tiny_a2d_from_ar(
    model_name_or_path: str,
    *,
    mask_token_id: Optional[int] = None,
    hybrid_attention: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    **config_overrides: Any,
) -> Any:
    """Load an AR checkpoint and convert it — the checkpoint-resolution entry.

    Resolution is deliberately narrow: the checkpoint's ``model_type`` must
    be mapped in :func:`ar_head_classes`, its ``architectures`` field must
    name the mapped upstream head, and the load goes through that concrete
    class — never a generic Auto\\* loader.  All three #107 regression
    properties, enforced before any weights move.
    """
    from transformers import AutoConfig

    checkpoint_config = AutoConfig.from_pretrained(model_name_or_path)
    mapping = ar_head_classes()
    entry = mapping.get(checkpoint_config.model_type)
    if entry is None:
        raise ValueError(
            f"no Tiny-A2D recipe for model_type "
            f"{checkpoint_config.model_type!r}; supported: {sorted(mapping)}"
        )
    upstream_cls = entry[0]

    declared = getattr(checkpoint_config, "architectures", None) or []
    if not declared:
        raise ValueError(
            "checkpoint config declares no architectures, so the concrete "
            "head cannot be proven (#107) and the checkpoint is rejected; "
            f"add {upstream_cls.__name__!r} to the config's architectures "
            "if this head is what the weights really are"
        )
    if upstream_cls.__name__ not in declared:
        raise ValueError(
            f"checkpoint architectures {declared!r} do not include "
            f"{upstream_cls.__name__}; model_type alone proves nothing about "
            "the head (#107), so this checkpoint is rejected rather than "
            "loaded into the wrong class"
        )

    load_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    ar_model = upstream_cls.from_pretrained(model_name_or_path, **load_kwargs)
    return convert_ar_model(
        ar_model,
        mask_token_id=mask_token_id,
        hybrid_attention=hybrid_attention,
        **config_overrides,
    )


__all__ = ["ar_head_classes", "convert_ar_model", "load_tiny_a2d_from_ar"]
