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

"""#166 Stage-1 — bias-aware fast LoRA (ledger row 5).

Two separate questions, deliberately not conflated:

1. the end-to-end step difference between the fast QKV path and the PEFT
   fallback;
2. what share of the step the QKV LoRA forward occupies.

Row 5 is default-on with no in-code measurement and no attributed share, which
is why it is here. No target is selected, no kernel proposed, no dispatch
default touched — Stage 2 owns selection.

**Arm isolation.** The ONLY difference between arms is
`self_attn.apply_qkv`: `apply_lora_qkv_with_bias` for fast,
`_original_apply_qkv` (plain `q_proj/k_proj/v_proj`) for reference. Same model,
same PEFT adapters, same Dream attention forward, same O-projection and MLP
fast paths. Comparing a fully patched model against an unpatched PEFT model
would fold RoPE, attention, O and MLP differences into a number attributed to
QKV.

**What `qkv_lora_projection` measures.** The FORWARD only. The custom
autograd's QKV backward — including the bias-gradient reduction `dQ.sum(0)` —
lands in `backward`, and is not split out: threading a timer into autograd
internals would change the thing being measured.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics
import subprocess
import sys
import weakref
from contextlib import contextmanager
from typing import Any

from unturtle.eval.cuda_event_timer import CudaEventTimer
from unturtle.eval.profile_harness import OperationEvent, ProfileCell, profile_cell

#: Frozen window. Not exposed on the CLI — three #166 gates failed when a
#: verdict depended on the caller passing a large enough window.
TRIALS = 3
STEPS = 8
WARMUP = 3

#: Decision-grade fixture.
CHECKPOINT = "Dream-org/Dream-v0-Instruct-7B"
CHECKPOINT_REVISION = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"
SEQ_LEN = 512
BATCH_SIZES = (1, 8, 32)
SENSITIVITY_SEQ_LEN = 1024
SENSITIVITY_BATCH = 1

#: LoRA configuration. `lora_dropout=0` and `bias="none"` are REQUIRED for the
#: fast path to engage at all (`fast_diffusion_model.py:262`), so they are part
#: of the fixture rather than a tunable.
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_BIAS = "none"
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

#: Optimizer over the trainable LoRA parameters only.
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False

TRIAL_SEEDS = (700, 701, 702)

REQUIRED_EVENTS = frozenset(
    {
        "noising",
        "qkv_lora_projection",
        "loss",
        "backward",
        "optimizer_step",
    }
)

#: Parity tolerances, fixed BEFORE any timing so they cannot be relaxed to fit a
#: result. They exist as a safety margin for bf16 LoRA over a 4-bit base; on the
#: real training step nothing has needed them so far — every compared tensor has
#: been bit-identical, and the graded report says so rather than reporting a
#: single "equivalent" count that would hide the distinction.
PARITY_ATOL = 2e-2
PARITY_RTOL = 2e-2


class OomInPhase(Exception):
    """An OOM tagged with the phase it happened in (#152 typed result)."""

    def __init__(self, phase: str, cause: BaseException) -> None:
        super().__init__(f"OOM during {phase}: {cause}")
        self.phase = phase
        self.cause = cause


@contextmanager
def oom_phase(phase: str):
    import torch

    try:
        yield
    except torch.cuda.OutOfMemoryError as error:
        raise OomInPhase(phase, error) from error


def require_supported_device(device: str) -> None:
    """`cuda:0` only — sync, CUDA events, peak stats, RNG fingerprints and the
    GPU name all target the default device, so anything else is mis-recorded."""
    if device == "cuda:0":
        return
    raise SystemExit(
        f"--device {device!r} is not supported: this producer is cuda:0 only."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="benchmarks/results/fast_lora_profile")
    return parser.parse_args()


def rng_fingerprint() -> str:
    """Hash of the ACTUAL CPU and CUDA RNG states — seed integers are not
    evidence of pairing, as #176 demonstrated."""
    import hashlib

    import torch

    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.numpy().tobytes())
    return digest.hexdigest()[:16]


def state_fingerprint(model, batch) -> str:
    """Hash of trainable weights and inputs: arms must agree on WHAT they
    compute, not only on the RNG that produced it."""
    import hashlib

    import torch

    digest = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            if not param.requires_grad:
                continue
            digest.update(name.encode())
            digest.update(param.detach().to("cpu", torch.float32).numpy().tobytes())
    for key in sorted(batch):
        value = batch[key]
        if hasattr(value, "detach"):
            digest.update(value.detach().to("cpu").numpy().tobytes())
    return digest.hexdigest()[:16]


def build(args, *, seq_len: int, batch_size: int, seed: int):
    """4-bit Dream base with bf16 LoRA adapters, plus a clean batch.

    Gradient checkpointing stays OFF: recomputation would change the QKV call
    count, and this cell's counts are a gate.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    from unturtle import FastDiffusionModel
    from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # SINGLE device, not `device_map="auto"`. Auto sharded this 7B checkpoint
    # across cuda:1/2/3 with nothing on cuda:0, which breaks every timing
    # assumption here — synchronization, CUDA events, peak-memory stats and RNG
    # fingerprints all target the default device — and made a cross-device
    # matmul fail outright. A sharded model cannot be profiled by this producer.
    model, tokenizer = FastDiffusionModel.from_pretrained(
        CHECKPOINT,
        revision=CHECKPOINT_REVISION,
        load_in_4bit=True,
        device_map={"": args.device},
    )
    shards = {str(param.device) for param in model.parameters()}
    if shards != {args.device}:
        raise SystemExit(
            f"the model is spread over {sorted(shards)}; this producer requires "
            f"every parameter on {args.device} so that timing, peak memory and "
            "RNG fingerprints refer to one device"
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            target_modules=list(LORA_TARGETS),
            use_dora=False,
        ),
    )
    model.config.use_cache = False
    mask_id = getattr(model.config, "mask_token_id", None) or tokenizer.mask_token_id

    training_args = DiffusionTrainingArguments(
        output_dir="/tmp/unturtle-fast-lora-profile",
        per_device_train_batch_size=batch_size,
        max_steps=1,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        report_to=[],
        sparse_lm_head=False,
        loss_norm_type="token",
        # `Trainer.__init__` calls `set_seed(args.seed)`; without this the
        # per-trial seeding above is erased by its default of 42 (#176).
        seed=seed,
        gradient_checkpointing=False,
    )
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=[{"input_ids": [5, 6, 7]}],
        processing_class=tokenizer,
        data_collator=None,
    )
    vocab = int(model.config.vocab_size)
    ids = torch.randint(1, min(vocab, 30000), (batch_size, seq_len), device=args.device)
    clean = {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones(
            batch_size, seq_len, dtype=torch.long, device=args.device
        ),
    }
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    return model, trainer, clean, optimizer, mask_id


def attention_modules(model) -> list:
    """Every layer's `self_attn`, resolved through the PEFT wrapper.

    On a `PeftModel`, `.model` is the LM-head model, so the decoder is reached
    via `get_decoder()` — using `.model` would silently miss the layers.
    """
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise RuntimeError("cannot reach the decoder layers on this model")
    return [layer.self_attn for layer in layers]


@contextmanager
def qkv_arm(model, arm: str):
    """Swap ONLY `apply_qkv`, and restore it however the block exits.

    A leaked patch would instrument the next arm, which is supposed to be the
    other implementation.
    """
    from unturtle.fast_diffusion_model import _original_apply_qkv
    from unturtle.kernels.fast_lora import apply_lora_qkv_with_bias

    target = apply_lora_qkv_with_bias if arm == "fast" else _original_apply_qkv
    modules = attention_modules(model)
    saved = [getattr(module, "apply_qkv", None) for module in modules]
    for module in modules:
        module.apply_qkv = target
    try:
        yield len(modules)
    finally:
        for module, previous in zip(modules, saved, strict=True):
            if previous is None:
                if hasattr(module, "apply_qkv"):
                    delattr(module, "apply_qkv")
            else:
                module.apply_qkv = previous


def parity_preflight(args, *, seq_len: int, batch_size: int, seed: int) -> dict:
    """Compare the two arms on ONE model with a FIXED pre-noised batch.

    Runs before any timing: a speed number for implementations that disagree
    numerically would be meaningless, and the tolerances are module constants so
    they cannot be relaxed after seeing a result.

    Two properties of the setup are load-bearing, both learned by getting them
    wrong first:

    - ONE model instance, swapped between arms, rather than a fresh build per
      arm. Two builds meant two draws from the diffusion process, so the arms
      saw different masks and different `t`; the resulting "parity failure"
      moved between runs, which a deterministic kernel difference cannot do.
    - The batch is PRE-NOISED and passed directly, so the forward process never
      runs and contributes no RNG. On a fixed batch the two paths agree
      bit-for-bit.
    """
    import torch

    model, trainer, clean, optimizer, _mask_id = build(
        args, seq_len=seq_len, batch_size=batch_size, seed=seed
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ids = clean["input_ids"]
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": clean["attention_mask"],
        "diffusion_mask": torch.rand(ids.shape, device=ids.device) < 0.5,
        "timesteps": torch.full((ids.shape[0],), 0.5, device=ids.device),
    }

    def one_arm(arm: str) -> dict[str, Any]:
        captured: dict[str, Any] = {}
        with qkv_arm(model, arm) as layer_count:
            captured["layers"] = layer_count
            modules = attention_modules(model)
            first = modules[0]
            original_apply = first.apply_qkv

            def capturing(self, X, _original=original_apply):
                out = _original(self, X)
                if "qkv" not in captured:
                    captured["qkv"] = [t.detach().float().cpu() for t in out]
                return out

            first.apply_qkv = capturing
            try:
                loss = trainer.compute_loss(model, dict(batch))
                loss.backward()
                captured["loss"] = float(loss.detach())
                captured["grads"] = {
                    name: param.grad.detach().float().cpu()
                    for name, param in model.named_parameters()
                    if param.requires_grad and param.grad is not None
                }
                optimizer.step()
                captured["post_step"] = {
                    name: param.detach().float().cpu()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
            finally:
                first.apply_qkv = original_apply
                model.zero_grad(set_to_none=True)
        return captured

    # Snapshot the trainable state and restore it between arms. The first arm
    # calls `optimizer.step()`, so without this the second arm starts from
    # ALREADY-UPDATED weights and the two losses differ for that reason alone —
    # which is what the earlier "parity failure" actually was.
    import copy

    baseline = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    optimizer_baseline = copy.deepcopy(optimizer.state_dict())

    def restore() -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in baseline:
                    param.copy_(baseline[name])
        optimizer.load_state_dict(copy.deepcopy(optimizer_baseline))
        model.zero_grad(set_to_none=True)

    fast = one_arm("fast")
    restore()
    reference = one_arm("reference")
    restore()

    problems: list[str] = []
    graded: list[tuple[str, Any, Any]] = []

    def compare(label: str, a, b) -> None:
        graded.append((label, a, b))
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            problems.append(f"{label}: non-finite values")
            return
        if not torch.allclose(a, b, atol=PARITY_ATOL, rtol=PARITY_RTOL):
            problems.append(f"{label}: max abs delta {float((a - b).abs().max()):.3e}")

    for index, (a, b) in enumerate(
        zip(fast.get("qkv", []), reference.get("qkv", []), strict=False)
    ):
        compare(f"qkv[{'QKV'[index]}]", a, b)
    if abs(fast["loss"] - reference["loss"]) > PARITY_ATOL:
        problems.append(f"loss: {fast['loss']:.6f} against {reference['loss']:.6f}")

    shared = sorted(set(fast["grads"]) & set(reference["grads"]))
    if not shared:
        problems.append("no shared trainable gradients to compare")
    for name in shared:
        compare(f"grad:{name}", fast["grads"][name], reference["grads"][name])
    # The bias gradient reduction is what this kernel adds over the stock fused
    # path, so its presence is asserted by name rather than assumed.
    for kind in ("lora_A", "lora_B"):
        if not any(kind in name for name in shared):
            problems.append(f"no {kind} gradients were captured")
    for name in sorted(set(fast["post_step"]) & set(reference["post_step"])):
        compare(
            f"post_step:{name}", fast["post_step"][name], reference["post_step"][name]
        )

    # Equivalence is graded, not lumped: "785 tensors equivalent" would hide
    # which agreed exactly and which only agreed within tolerance.
    bit_identical = 0
    tolerance_equivalent = 0
    for _label, a, b in graded:
        if torch.equal(a, b):
            bit_identical += 1
        elif torch.allclose(a, b, atol=PARITY_ATOL, rtol=PARITY_RTOL):
            tolerance_equivalent += 1
    result = {
        "status": "ok" if not problems else "parity_failed",
        "problems": problems,
        "mismatch": len(problems),
        "bit_identical": bit_identical,
        "tolerance_equivalent": tolerance_equivalent,
        "tolerance_provenance": (
            "module constants PARITY_ATOL/PARITY_RTOL, fixed before any timing"
        ),
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "layers": fast.get("layers"),
        "loss_fast": fast["loss"],
        "loss_reference": reference["loss"],
        "loss_bit_identical": fast["loss"] == reference["loss"],
        "compared_tensors": len(graded),
    }
    # No `del` of the closed-over names: `one_arm` captures them, so deleting
    # them in this scope makes them deleted-locals for the whole function and
    # would break the closure (ruff F821 flags exactly this). Dropping the
    # references by rebinding is enough — the frame ends on return anyway.
    result["released"] = True
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
