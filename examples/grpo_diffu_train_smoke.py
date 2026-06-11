#!/usr/bin/env python3
# Copyright 2025-present nishide-dev & the Unturtle team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Minimal GPU smoke: run :meth:`~unturtle.diffusion.DiffuGRPOTrainer.train` for a few steps.

This verifies the full TRL + diffu-GRPO training loop (rollout → rewards → advantages →
backward) on CUDA without downloading checkpoints.

**Dependencies** (editable install with Hugging Face + GRPO extras)::

    uv pip install --python .venv/bin/python -e ".[huggingface,grpo]"

(or ``-e ".[huggingface]"`` then ``uv pip install trl mergekit tokenizers``).

**Run** (single GPU)::

    python examples/grpo_diffu_train_smoke.py

Optional **wd1** objective (ratio-free weighted log-likelihood, ICLR 2026 wd1)::

    python examples/grpo_diffu_train_smoke.py --wd1

Optional **wd1++** (MC log-prob at denoise snapshots; see ``docs/wd1plusplus-design.md``)::

    python examples/grpo_diffu_train_smoke.py --wd1++

``import unsloth`` is executed before ``torch`` / ``transformers`` / TRL so Unsloth patches apply.
Expects ``torch.cuda.is_available()``. For d1-aligned advantages (no per-group std scaling),
we pass ``scale_rewards=\"none\"`` to :class:`~unturtle.diffusion.DiffuGRPOConfig` (TRL maps
``False`` the same way). See ``docs/diffu-grpo-d1-notes.md``.

The smoke uses ``num_iterations=2`` so the TRL rollout buffer and mask seeds are exercised
across inner GRPO steps.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import torch
import unsloth  # noqa: F401 — must run before torch/transformers/trl (Unsloth + project convention).
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForMaskedLM, PreTrainedTokenizerFast


def _stub_mod(name: str, **attrs: object) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__package__ = name.rpartition(".")[0] or name
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _patch_trl_if_broken_imports() -> None:
    """Pre-load stubs so TRL can import without optional deps (see ``conftest.py``)."""
    import importlib

    trl_import_utils = importlib.import_module("trl.import_utils")
    # ``_is_package_available("weave")`` can be true while ``import weave`` still fails in some envs.
    try:
        import weave  # noqa: F401
    except ModuleNotFoundError:
        trl_import_utils._weave_available = False  # type: ignore[attr-defined]

    merge_missing = importlib.util.find_spec("mergekit") is None
    blender_missing = importlib.util.find_spec("llm_blender") is None
    if merge_missing and "trl.mergekit_utils" not in sys.modules:
        sys.modules["trl.mergekit_utils"] = _stub_mod(
            "trl.mergekit_utils",
            MergeConfiguration=object,
            MergeOptions=object,
            MergeConfig=object,
            run_merge=lambda *a, **kw: None,
            merge_models=lambda *a, **kw: None,
            upload_model_to_hub=lambda *a, **kw: None,
        )
    if blender_missing and "trl.trainer.judges" not in sys.modules:
        sys.modules["trl.trainer.judges"] = _stub_mod(
            "trl.trainer.judges",
            BasePairRMJudge=object,
            BaseJudge=object,
            BasePairwiseJudge=object,
            HfPairwiseJudge=object,
            OpenAIPairwiseJudge=object,
            AllTrueJudge=object,
        )


_patch_trl_if_broken_imports()

from unturtle.diffusion import DiffuGRPOConfig, DiffuGRPOTrainer


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        print(
            "This smoke test requires CUDA (DiffuGRPO generation uses CUDA autocast paths).",
            file=sys.stderr,
        )
        sys.exit(1)


def _make_tokenizer_and_model() -> tuple[PreTrainedTokenizerFast, BertForMaskedLM]:
    pad, unk, bos, eos = "<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"
    special = [pad, unk, "[MASK]", bos, eos]
    vocab = special + [f"w{i}" for i in range(95)]
    tok = Tokenizer(WordLevel(vocab={w: i for i, w in enumerate(vocab)}, unk_token=unk))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok)
    tokenizer.add_special_tokens(
        {
            "pad_token": pad,
            "unk_token": unk,
            "mask_token": "[MASK]",
            "bos_token": bos,
            "eos_token": eos,
        }
    )
    vocab_size = len(vocab)
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(cfg)
    model.to(torch.device("cuda"))
    model.train()
    return tokenizer, model


def _toy_reward(prompts, completions, **_kwargs):
    # Vary rewards so group-relative advantages are non-degenerate.
    out = []
    for c in completions:
        t = c if isinstance(c, str) else str(c)
        out.append(1.0 if len(t) % 2 == 0 else 0.25)
    return out


def main() -> None:
    _require_cuda()
    use_wd1pp = "--wd1++" in sys.argv
    use_wd1 = "--wd1" in sys.argv and not use_wd1pp

    try:
        import trl.trainer.grpo_trainer  # noqa: F401
    except ModuleNotFoundError as e:
        print(
            "Missing optional GRPO dependencies. Install with:\n"
            '  uv pip install --python .venv/bin/python -e ".[huggingface,grpo]"\n'
            "  or: uv pip install trl mergekit tokenizers\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer, model = _make_tokenizer_and_model()
    mask_id = tokenizer.mask_token_id
    assert mask_id is not None

    rows = [{"prompt": f"Repeat the tokens w{i} w{(i + 1) % 20}."} for i in range(8)]
    train_ds = Dataset.from_list(rows)

    out_dir = "/tmp/unturtle_grpo_smoke"
    args = DiffuGRPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        max_steps=8,
        max_completion_length=32,
        max_prompt_length=48,
        block_length=8,
        diffusion_steps=8,
        mask_id=int(mask_id),
        p_mask_prompt=0.3,
        beta=0.0,
        scale_rewards="none",
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        bf16=False,
        fp16=False,
        num_iterations=2,
        diffu_policy_objective=("wd1++" if use_wd1pp else "wd1" if use_wd1 else "grpo"),
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = DiffuGRPOTrainer(
        model=model,
        reward_funcs=_toy_reward,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    train_output = trainer.train()
    mode = "wd1++" if use_wd1pp else "wd1" if use_wd1 else "grpo"
    print(
        "train() finished:", train_output.global_step, "steps", f"({mode})", flush=True
    )
    assert train_output.global_step >= 8, "expected max_steps to run"


if __name__ == "__main__":
    main()
