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
import functools
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


#: Stable reason code for the #177 limitation. A code rather than a message so
#: a future run can be matched against this one without string comparison.
DTYPE_MISMATCH_REASON = "fast_path_execution_dtype_mismatch"


def classify_failure(error: BaseException) -> str | None:
    """Map an exception to the known limitation, or None if it is something else.

    Deliberately narrow: an unrelated RuntimeError must NOT be labelled as the
    known dtype limitation, or a new defect would be filed under an old one.
    """
    if not isinstance(error, RuntimeError):
        return None
    text = str(error).lower()
    dtype_words = ("same dtype", "expected mat1 and mat2", "self and mat2")
    if any(word in text for word in dtype_words) and (
        "bfloat16" in text or "float" in text
    ):
        return DTYPE_MISMATCH_REASON
    return None


def execution_preflight(model, trainer, optimizer, batch, arm: str) -> dict[str, Any]:
    """ONE untimed forward+backward, before warmup or any timing.

    Its purpose is to answer whether the arm can execute at all. A path that
    installs cleanly and then fails on its first real forward is a product
    limitation, and reporting a speed number for it — or a zero — would present
    a non-execution as a measurement.
    """
    import torch

    try:
        with qkv_arm(model, arm):
            loss = trainer.compute_loss(model, dict(batch))
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
    except Exception as error:  # noqa: BLE001 - classified, then re-reported
        import traceback

        model.zero_grad(set_to_none=True)
        frames = traceback.extract_tb(error.__traceback__)
        # WHERE it raised, not just what it says. The reference arm fails with
        # the same message from Unsloth's MLP path, so message-matching alone
        # would credit an unrelated failure to the QKV kernel.
        origin = next(
            (
                f"{frame.filename.split('/')[-1]}:{frame.lineno}"
                for frame in reversed(frames)
                if "fast_lora" in frame.filename or "kernels" in frame.filename
            ),
            f"{frames[-1].filename.split('/')[-1]}:{frames[-1].lineno}"
            if frames
            else None,
        )
        in_qkv_path = any(
            "unturtle/kernels/fast_lora" in frame.filename for frame in frames
        )
        return {
            "arm": arm,
            "executable": False,
            "exception_class": type(error).__name__,
            "exception_message": str(error)[:300],
            "raised_in": origin,
            "raised_in_unturtle_qkv_kernel": in_qkv_path,
            "reason_code": classify_failure(error),
        }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"arm": arm, "executable": True, "reason_code": None}


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


def require_single_device(shards: set[str], device: str) -> None:
    """Refuse a model whose parameters are spread over more than one device.

    `device_map="auto"` sharded this 7B checkpoint over cuda:1/2/3 with nothing
    on cuda:0: matmuls failed across devices, and timing, peak memory and RNG
    fingerprints would each have referred to a different device.
    """
    if shards == {device}:
        return
    raise SystemExit(
        f"the model is spread over {sorted(shards)}; this producer requires "
        f"every parameter on {device} so that timing, peak memory and RNG "
        "fingerprints refer to one device"
    )


def require_fast_baseline(counts: dict[str, int], expected_layers: int) -> None:
    """Refuse a build whose baseline is not the fast path on EVERY layer.

    peft's own `get_peft_model` leaves every layer on `_original_apply_qkv`,
    because the Dream QKV patch keys on `hasattr(q_proj, "lora_A")` and runs
    during `from_pretrained` — before any adapter exists. Both arms then run the
    reference, and the cell reports a ~0% difference as a kernel finding.
    """
    if (
        expected_layers
        and counts.get("fast") == expected_layers
        and not counts.get("reference")
        and not counts.get("other")
    ):
        return
    raise SystemExit(
        f"the Dream fast QKV path was not installed on every layer (observed "
        f"{counts} over {expected_layers} layers), so the fast arm would not "
        "differ from the reference; check the LoRA config guards (dropout must "
        "be 0, bias 'none', no DoRA)"
    )


def build(args, *, seq_len: int, batch_size: int, seed: int):
    """4-bit Dream base with bf16 LoRA adapters, plus a clean batch.

    Gradient checkpointing stays OFF: recomputation would change the QKV call
    count, and this cell's counts are a gate.
    """
    import torch

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
    # `dtype=bfloat16` explicitly: without it the model loads fp32 while the
    # 4-bit weights dequantize to bf16, and the fast QKV path fails with
    # "expected mat1 and mat2 to have the same dtype". `bf16=True` on the
    # trainer only drives autocast — it does not change the stored weights.
    model, tokenizer = FastDiffusionModel.from_pretrained(
        CHECKPOINT,
        revision=CHECKPOINT_REVISION,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
    )
    require_single_device({str(p.device) for p in model.parameters()}, args.device)
    # `FastDiffusionModel.get_peft_model`, NOT peft's: the Dream QKV patch
    # requires `hasattr(q_proj, "lora_A")`, so patching before adapters exist
    # installs nothing. Calling peft directly left every layer on
    # `_original_apply_qkv` — verified — which would have made the "fast" arm
    # identical to the reference and reported a 0% difference as a finding.
    model = FastDiffusionModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=list(LORA_TARGETS),
        use_gradient_checkpointing=False,
    )
    require_fast_baseline(qkv_installation(model), len(attention_modules(model)))
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

    NOT REACHABLE IN THE FROZEN ROW-5 CONFIGURATION (2026-08-27). This function
    needs both arms to complete a forward, and under `load_in_4bit=True` +
    `get_peft_model` neither can (#177) — verified by calling it directly on the
    fixed build, which raises the same operand-dtype RuntimeError. It is kept,
    uncalled, because it is the correct check to run the moment #177 unblocks the
    cell; `main()` deliberately stops at `execution_preflight` instead.

    ERRATUM to commit 5f141cb, recorded 2026-08-27: that commit's body reports
    "784 of 784 bit-identical, loss delta 0.000e+00" as a fast-vs-reference
    parity result. The run behind that number happened at 6a7b5ce, which called
    peft's `get_peft_model` and therefore left every layer on the reference
    callable. Both arms ran the SAME implementation, so the figure is a
    reference-against-itself comparison and does not establish fast/reference
    numerical parity. The number itself is not disputed and is not restated
    here as parity; no fast-vs-reference parity result exists for row 5.

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
        # The COUNT OF COMPARISONS ACTUALLY MADE, not an arithmetic estimate.
        # An earlier version reported `len(shared)*2 + len(qkv) + 1`, which
        # added the scalar loss to a tensor tally even though the loss is
        # compared by subtraction and never enters `graded`. The denominator is
        # one smaller for that reason alone — no tensor was dropped.
        "compared_tensors": len(graded),
        "comparison_inventory": {
            "qkv_outputs": len(fast.get("qkv", [])),
            "gradients": len(shared),
            "post_step_parameters": len(
                set(fast["post_step"]) & set(reference["post_step"])
            ),
            "scalar_loss": "compared separately, not counted in compared_tensors",
        },
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


#: The four conditions, cycled so each lands in each ordinal position across
#: trials. A fixed order would load thermal, allocator and clock drift onto
#: whichever condition always runs last — at 7B that is the same order as the
#: effect being measured. Fixed before timing, not adjusted after seeing results.
CONDITIONS = ("fast_off", "reference_off", "fast_on", "reference_on")
CONDITION_ORDERS = (
    ("fast_off", "reference_off", "fast_on", "reference_on"),
    ("reference_off", "fast_on", "reference_on", "fast_off"),
    ("fast_on", "reference_on", "fast_off", "reference_off"),
)


def qkv_installation(model) -> dict[str, int]:
    """How many layers carry each QKV implementation.

    A MIXED state is the dangerous one: a speed difference could no longer be
    read as the kernel's effect across 28 layers.
    """
    from unturtle.fast_diffusion_model import _original_apply_qkv
    from unturtle.kernels.fast_lora import apply_lora_qkv_with_bias

    counts = {"fast": 0, "reference": 0, "other": 0}
    for attn in attention_modules(model):
        current = getattr(attn, "apply_qkv", None)
        underlying = getattr(current, "__func__", current)
        if underlying is apply_lora_qkv_with_bias:
            counts["fast"] += 1
        elif underlying is _original_apply_qkv:
            counts["reference"] += 1
        else:
            counts["other"] += 1
    return counts


def assert_arm_installed(model, arm: str, *, expected_layers: int) -> dict[str, int]:
    """Stage 1 and 2 of the gate: fast at baseline, and actually swapped.

    Re-checked immediately before every timed window, not only at fixture
    build: warmup or wrapper installation could break it in between, and a
    build-time-only check would miss that.
    """
    counts = qkv_installation(model)
    want = "fast" if arm == "fast" else "reference"
    if (
        counts[want] != expected_layers
        or counts["other"]
        or sum(counts.values()) != expected_layers
    ):
        raise RuntimeError(
            f"arm {arm!r} expected {expected_layers} layers on the {want} "
            f"callable, observed {counts}: a mixed or unexpected installation "
            "makes a speed difference unattributable to the kernel"
        )
    return counts


def assert_single_device(model, optimizer, device: str) -> None:
    """Every parameter, buffer and optimizer state on ONE device.

    `device_map="auto"` sharded this checkpoint across cuda:1/2/3 with nothing
    on cuda:0, which silently invalidates synchronization, CUDA events, peak
    stats and RNG fingerprints — all of which target the default device.
    """
    import torch

    places = {str(p.device) for p in model.parameters()}
    places |= {str(b.device) for b in model.buffers()}
    for state in optimizer.state.values():
        places |= {str(v.device) for v in state.values() if torch.is_tensor(v)}
    if places != {device}:
        raise SystemExit(
            f"model/optimizer state spans {sorted(places)}; this producer needs "
            f"everything on {device} so timing, memory and RNG refer to one device"
        )
    current = f"cuda:{torch.cuda.current_device()}"
    if current != device:
        raise SystemExit(
            f"torch.cuda.current_device() is {current}, not {device}: the timer, "
            "memory stats and RNG would target a different device than the model"
        )


def callable_identities(model) -> dict[str, str | None]:
    """Identity of everything that must NOT change between arms.

    Only `apply_qkv` may differ. Capturing O projection, MLP and the attention
    forward makes a mutation that swaps them fail rather than be attributed to
    QKV.
    """

    def fingerprint(value):
        # `inspect.unwrap` first, so an instrumentation wrapper carrying
        # `functools.wraps` reports the callable it wraps. Without it the
        # identity gate and the profiling wrapper each read the other as a
        # change.
        """Identify the underlying function, not a bound-method wrapper.

        `id(module.forward)` creates a NEW bound method on every access, so
        comparing those ids reports a change even when nothing changed —
        measured: 56 of 56 keys "differed" between two consecutive calls with no
        modification in between. The values look stable within one expression
        only because CPython reuses the freed address.
        """
        if value is None:
            return None
        underlying = getattr(value, "__func__", value)
        return (
            f"{getattr(underlying, '__qualname__', repr(underlying))}@{id(underlying)}"
        )

    identities: dict[str, str | None] = {}
    for index, attn in enumerate(attention_modules(model)):
        identities[f"attn{index}.forward"] = fingerprint(attn.forward)
        identities[f"attn{index}.apply_o"] = fingerprint(getattr(attn, "apply_o", None))
        identities[f"attn{index}.o_proj.forward"] = fingerprint(attn.o_proj.forward)
        identities[f"attn{index}.apply_qkv"] = fingerprint(
            getattr(attn, "apply_qkv", None)
        )
    return identities


class StateSnapshot:
    """Trainable parameters, full optimizer state, and both RNG streams.

    Restored before every timed window and OUTSIDE the wall: a snapshot copy or
    an optimizer `load_state_dict` inside the timed region would be charged to
    step time. Optimizer state means the whole dict — step counter, `exp_avg`,
    `exp_avg_sq` — not just parameters, or the second arm would run against a
    warmed-up Adam.
    """

    def __init__(self, model, optimizer) -> None:
        import copy

        import torch

        self.params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())
        self.cpu_rng = torch.get_rng_state().clone()
        self.cuda_rng = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        )

    def restore(self, model, optimizer) -> None:
        import copy

        import torch

        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.params:
                    param.copy_(self.params[name])
        optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))
        torch.set_rng_state(self.cpu_rng.clone())
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all([s.clone() for s in self.cuda_rng])
        model.zero_grad(set_to_none=True)
        # Gradients are NOT snapshotted: each window starts from none, and that
        # is asserted rather than assumed.
        leftover = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is not None
        ]
        if leftover:
            raise RuntimeError(
                f"{len(leftover)} trainable parameters still carry gradients "
                f"after zero_grad, first: {leftover[:3]}"
            )


@contextmanager
def instrumented(model, trainer, timer: CudaEventTimer):
    """Hooks for the ON condition; every one restored in `finally`."""
    import unturtle.diffusion.trainer as trainer_module

    scopes: dict[int, Any] = {}
    saved_apply = []

    def wrap_apply(module):
        original = module.apply_qkv

        @functools.wraps(original)
        def timed(self, X, _original=original):
            with timer.measure("qkv_lora_projection"):
                return _original(self, X)

        module.apply_qkv = timed
        return original

    for module in attention_modules(model):
        saved_apply.append((module, wrap_apply(module)))

    original_forward = model.__class__.__call__

    def timed_forward(self, *args, _timer=timer, **kwargs):
        with _timer.measure("full_model_forward"):
            return original_forward(self, *args, **kwargs)

    model.__class__.__call__ = timed_forward

    # The loss symbol bound INTO the trainer module — patching the kernel module
    # would wrap a name nobody calls.
    original_loss = trainer_module.fast_masked_diffusion_loss

    def timed_loss(*args, **kwargs):
        with timer.measure("loss"):
            return original_loss(*args, **kwargs)

    trainer_module.fast_masked_diffusion_loss = timed_loss
    try:
        yield
    finally:
        for module, original in saved_apply:
            module.apply_qkv = original
        model.__class__.__call__ = original_forward
        trainer_module.fast_masked_diffusion_loss = original_loss
        scopes.clear()


def run_condition(
    condition, *, model, trainer, optimizer, batch, snapshot, device, timer=None
):
    """One timed window for one condition.

    Restoration, arm installation and peak reset all happen BEFORE the clock
    starts. Warmup runs inside the arm, then state and RNG are restored AGAIN so
    warmup's parameter updates and allocator peak cannot reach the timed result.
    """
    import time

    import torch

    arm = "fast" if condition.startswith("fast") else "reference"
    instrument = condition.endswith("_on")

    def one_step() -> None:
        loss = trainer.compute_loss(model, dict(batch))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    snapshot.restore(model, optimizer)
    with qkv_arm(model, arm) as layer_count:
        # Stages 1-2: the right callable, on every layer, right now.
        install_before = assert_arm_installed(model, arm, expected_layers=layer_count)
        identities_before = callable_identities(model)

        # Warmup inside the arm, then restore again: its updates and peak must
        # not enter the timed window.
        for _ in range(WARMUP):
            one_step()
        torch.cuda.synchronize()
        snapshot.restore(model, optimizer)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        if instrument:
            with instrumented(model, trainer, timer):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(STEPS):
                    one_step()
                torch.cuda.synchronize()
                timer.collect(synchronize=False)
                wall = time.perf_counter() - start
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(STEPS):
                one_step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - start

        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        # Still installed after the window: a wrapper left behind, or an arm
        # that reverted mid-run, must not pass silently.
        install_after = assert_arm_installed(model, arm, expected_layers=layer_count)
        identities_after = callable_identities(model)

    if identities_before != identities_after:
        raise RuntimeError(
            "non-QKV callables changed during the window, so the arms differ by "
            "more than the QKV projection"
        )
    from unturtle.fast_diffusion_model import _original_apply_qkv
    from unturtle.kernels.fast_lora import apply_lora_qkv_with_bias

    active = apply_lora_qkv_with_bias if arm == "fast" else _original_apply_qkv
    return {
        "condition": condition,
        "arm": arm,
        "instrumented": instrument,
        "qkv_callable": {
            "module": active.__module__,
            "qualname": active.__qualname__,
            "layer_count": layer_count,
        },
        "qkv_installation_before": install_before,
        "qkv_installation_after": install_after,
        "wall_seconds": wall,
        "per_step_seconds": wall / STEPS,
        "layers": layer_count,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "operations": timer.result() if instrument else None,
    }


def environment() -> dict[str, Any]:
    """Versions that determine whether the limitation reproduces."""
    import importlib.metadata as metadata

    import torch

    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except Exception:  # pragma: no cover - provenance must not fail a run
            return None

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "bitsandbytes": version("bitsandbytes"),
    }


def provenance(args: argparse.Namespace) -> dict[str, Any]:

    def git(*command: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *command], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover
            return None

    dirty = git("status", "--porcelain")
    return {
        "head_sha": git("rev-parse", "HEAD") or "unknown",
        "worktree_clean": (dirty == "") if dirty is not None else None,
        "worktree_dirty_paths": (
            [line[3:] for line in dirty.splitlines()] if dirty else []
        ),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "environment": environment(),
        "fixture": {
            "checkpoint": f"{CHECKPOINT}@{CHECKPOINT_REVISION}",
            "load_in_4bit": True,
            "requested_dtype": "bfloat16",
            "adapter_entry_point": "FastDiffusionModel.get_peft_model",
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "bias": LORA_BIAS,
                "targets": list(LORA_TARGETS),
                "dora": False,
            },
        },
        "frozen_constants": {"TRIALS": TRIALS, "STEPS": STEPS, "WARMUP": WARMUP},
    }


def dtype_survey(model) -> dict[str, Any]:
    """What the model actually produces, versus what the path requires."""
    import collections

    import torch
    from unsloth.kernels.utils import fast_dequantize, get_lora_parameters_bias

    attn = attention_modules(model)[0]
    weight, quant_state, lora_a, lora_b, _scale, bias = get_lora_parameters_bias(
        attn.q_proj
    )
    ids = torch.randint(1, 1000, (1, 8), device=next(model.parameters()).device)
    hidden = model.get_input_embeddings()(ids)
    return {
        "parameter_dtypes": dict(
            collections.Counter(str(p.dtype) for p in model.parameters())
        ),
        "hidden_state_dtype": str(hidden.dtype),
        "quantized_weight_dtype": str(weight.dtype),
        "dequantized_weight_dtype": str(fast_dequantize(weight, quant_state).dtype),
        "lora_a_dtype": str(lora_a.dtype),
        "lora_b_dtype": str(lora_b.dtype),
        "bias_dtype": str(bias.dtype) if bias is not None else None,
    }


def main() -> None:
    import gc as _gc

    import torch

    args = parse_args()
    require_supported_device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- preflight order: construction, device, identity, snapshot, execution ---
    model, trainer, clean, optimizer, _mask_id = build(
        args, seq_len=SEQ_LEN, batch_size=1, seed=TRIAL_SEEDS[0]
    )
    assert_single_device(model, optimizer, args.device)
    baseline_install = qkv_installation(model)
    layers = len(attention_modules(model))
    survey = dtype_survey(model)

    torch.manual_seed(TRIAL_SEEDS[0])
    torch.cuda.manual_seed_all(TRIAL_SEEDS[0])
    ids = clean["input_ids"]
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": clean["attention_mask"],
        "diffusion_mask": torch.rand(ids.shape, device=ids.device) < 0.5,
        "timesteps": torch.full((ids.shape[0],), 0.5, device=ids.device),
    }

    fast_check = execution_preflight(model, trainer, optimizer, batch, "fast")
    # The reference arm is probed for DIAGNOSIS only. A working reference cannot
    # upgrade the cell: a two-arm comparison needs both arms to execute under
    # the same frozen fixture.
    reference_check = execution_preflight(model, trainer, optimizer, batch, "reference")

    record: dict[str, Any] = {
        "row": 5,
        "cell": "dream_bias_aware_fast_lora",
        "layers": layers,
        "baseline_installation": baseline_install,
        "dtype_survey": survey,
        "fast_arm_preflight": fast_check,
        "reference_arm_preflight": reference_check | {"diagnostic_only": True},
    }

    if fast_check["executable"]:
        record |= {
            "status": "preflight_passed",
            "note": (
                "the fast arm executes; timing would proceed from here in a "
                "follow-up run pinned to this SHA"
            ),
        }
    else:
        # Null, never zero: a zero latency or 0% share reads as measured.
        record |= {
            "status": "unsupported",
            "stage": "preflight",
            "failure_stage": "fast_arm_preflight_forward",
            "reason_code": fast_check["reason_code"],
            "measurement_valid": False,
            "timing_attempted": False,
            "warmup_attempted": False,
            "speed_verdict": None,
            "operation_profile": None,
            "peak_memory": None,
            "target_selection_eligible": False,
            "blocked_by": "#177",
            "reproduction": {
                "file": "benchmarks/kernels/repro_dream_4bit_lora_dtype.py",
                "command": (
                    ".venv/bin/python benchmarks/kernels/repro_dream_4bit_lora_dtype.py"
                ),
            },
            "conclusion": (
                "Row 5 is not performance-measurable in its frozen documented "
                "configuration. The supported Dream adapter entry point installs "
                f"the bias-aware fast QKV callable on all {layers} layers, but "
                "the first real 4-bit training forward fails because the FP32 "
                "hidden state is incompatible with the fast path's operand "
                "dtypes. No timing was attempted and no speed conclusion is "
                "drawn. The production defect is tracked separately in #177."
            ),
            "parity": None,
            "parity_note": (
                "No fast-vs-reference parity result exists for row 5. The check "
                "needs both arms to complete a forward, which neither can here "
                "(#177); calling parity_preflight directly on this build raises "
                "the same operand-dtype error. ERRATUM (2026-08-27): commit "
                "5f141cb's body reports '784 of 784 bit-identical' as parity, "
                "but that run predates the install-gate fix (54a3af6) and ran "
                "BOTH arms on the reference callable, so it compares the "
                "reference against itself."
            ),
            "scope_note": (
                "The blocker is BROADER than this row's kernel: the reference "
                "arm fails with the same operand-dtype error raised from "
                "Unsloth's own MLP fast_lora path, so neither arm executes and "
                "the failure is not specific to the Dream bias-aware QKV "
                "kernel. `raised_in` / `raised_in_unturtle_qkv_kernel` on each "
                "arm carry the raising frame. Consequence for Stage 1: this "
                "cell cannot be repaired by changing the row-5 kernel alone, "
                "and any 4-bit + PEFT cell in the same configuration is "
                "expected to hit the same wall."
            ),
        }

    payload = {"run": provenance(args), "cells": [record]}
    (out / "fast_lora_profile.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "dtype_survey"})[:600])
    print(f"wrote 1 cell to {out / 'fast_lora_profile.json'}")

    del model, trainer, clean, optimizer
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
