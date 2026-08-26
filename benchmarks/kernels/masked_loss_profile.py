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

"""#166 Stage-1 — end-to-end profile of the DENSE masked training step.

Ledger row 1 (dense masked CE / fused-mask wrapper) is the default loss path for
every masked family and has never had an attributed share of step time. This
producer measures it. No target is selected, no kernel proposed, no dispatch
default touched — Stage 2 owns selection.

**Dense only.** The sparse LM-head path is a separate producer and artifact: it
bypasses `model(**inputs)` and the ordinary `lm_head`, projecting masked hidden
states directly, so its hook topology and required event set differ. Its
mask-ratio sweep also needs pre-noised batches, while this cell measures
`noising` itself and therefore needs clean ones — mixing them would produce an
arm-dependent taxonomy where one arm has no `noising` event.

Frozen taxonomy, every event coverage-eligible:

    noising              1 call/step
    attention            `layers` calls/step
    lm_head_projection   1 call/step
    loss                 1 call/step
    backward             1 call/step
    optimizer_step       1 call/step

`model(**inputs)` is deliberately NOT an event. Its inclusive time and the
residual left after subtracting attention and the LM head (embeddings, MLP,
norms) are recorded as diagnostics in `extra`; the residual is NOT added to
coverage, so it stays in `unattributed_seconds`. Publishing a `model_other`
operation would invent a taxonomy entry the protocol never froze.

Nested attribution uses `CudaEventTimer`, which records event pairs in the hooks
and reads them after the outer step synchronize. Its adoption is structural —
host synchronization must not scale with layer count — and at this fixture's
scale it is within noise of the scope-synchronizing timer.
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

#: Frozen measurement window; not exposed on the CLI, because three #166 gates
#: failed when a verdict depended on the caller passing a large enough window.
TRIALS = 3
STEPS = 8
WARMUP = 3

#: #77's controlled shape, with batch expanded to the protocol's cells.
HIDDEN_SIZE = 512
SEQ_LEN = 512
HEADS = 8
CANONICAL_LAYERS = 2
BATCH_SIZES = (1, 8, 32)
VOCAB_SIZES = (32000, 128256)

#: Optimizer pinned explicitly, not left to version-dependent defaults: a
#: torch release changing an AdamW default would silently change the step this
#: profile attributes.
OPTIMIZER = "torch.optim.AdamW"
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01
AMSGRAD = False

#: Per-trial seeds. Paired arms within a trial share a seed so OFF and ON see
#: the same stream; trials differ so the mask diagnostic gets distinct draws.
TRIAL_SEEDS = (700, 701, 702)

#: One bounded depth-sensitivity cell. A 2-layer fixture can overstate the LM
#: head's share, which is the trap the ELF synthetic fixture fell into; this
#: checks it without a broad layer sweep.
SENSITIVITY_LAYERS = 8
SENSITIVITY_BATCH = 8

REQUIRED_EVENTS = frozenset(
    {
        "noising",
        "attention",
        "lm_head_projection",
        "loss",
        "backward",
        "optimizer_step",
    }
)


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


def state_fingerprint(model, clean) -> str:
    """Hash of model parameters and the clean inputs.

    Paired arms must agree on what they are computing, not only on the RNG that
    produced it: identical states can still diverge if construction picked
    different weights, and an input difference would make the two walls
    incomparable while every seed check still passed.
    """
    import hashlib

    import torch

    digest = hashlib.sha256()
    with torch.no_grad():
        for _name, param in sorted(model.named_parameters()):
            digest.update(param.detach().to("cpu", torch.float32).numpy().tobytes())
    for key in sorted(clean):
        value = clean[key]
        if hasattr(value, "detach"):
            digest.update(value.detach().to("cpu").numpy().tobytes())
    return digest.hexdigest()[:16]


def rng_fingerprint() -> str:
    """Hash of the ACTUAL CPU and CUDA RNG states.

    Comparing seed integers is not enough: library initialisation or lazy
    optimizer-state creation can consume the stream, so two arms handed the same
    seed can still start from different states. This fingerprints the state
    itself, taken after construction and immediately before warmup.
    """
    import hashlib

    import torch

    digest = hashlib.sha256()
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.numpy().tobytes())
    return digest.hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="benchmarks/results/masked_loss_profile")
    return parser.parse_args()


def optimizer_fingerprint(optimizer) -> dict[str, Any]:
    """The optimizer's ACTUAL resolved settings.

    Read from `defaults` and the param groups rather than echoed from the
    constructor arguments, so implementation-selection flags appear as the
    `None` / `False` / `True` torch chose.
    """
    import torch

    interesting = (
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "foreach",
        "fused",
        "capturable",
        "maximize",
        "differentiable",
    )
    resolved: dict[str, Any] = {}
    for key in interesting:
        if key in optimizer.defaults:
            value = optimizer.defaults[key]
            resolved[key] = list(value) if isinstance(value, tuple) else value
        else:
            resolved[key] = "absent_from_defaults"
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: (list(group[key]) if isinstance(group[key], tuple) else group[key])
                for key in interesting
                if key in group
            }
        )
    return {
        "class": type(optimizer).__name__,
        "qualified_class": OPTIMIZER,
        "resolved_defaults": resolved,
        "param_groups": groups,
        "torch_version": torch.__version__,
    }


def require_supported_device(device: str) -> None:
    """Refuse a device this producer would mis-record.

    Synchronization, CUDA events, peak stats, RNG state and the GPU name all go
    through the current/default device or every device. On `cuda:1` the record
    would name GPU 0 and the events could land on another device's stream, so
    the honest options are to thread an exact `torch.device` everywhere or to
    refuse. This refuses.
    """
    if device == "cuda:0":
        return
    raise SystemExit(
        f"--device {device!r} is not supported: this producer is cuda:0 only. "
        "Its synchronization, CUDA events, peak-memory stats, RNG fingerprints "
        "and GPU-name provenance all go through the DEFAULT device or every "
        "device, so another index would be mis-recorded — and `cpu` on a "
        "GPU-equipped host would fold CUDA RNG state, CUDA peak memory and "
        "GPU 0's name into a CPU cell. Narrowing the contract is honest; "
        "threading an exact torch.device through every one of those call sites "
        "is the alternative and is not what this Stage-1 cell needs."
    )


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    def git(*command: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *command], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance must not fail a run
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
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "frozen_constants": {
            "TRIALS": TRIALS,
            "STEPS": STEPS,
            "WARMUP": WARMUP,
            "HIDDEN_SIZE": HIDDEN_SIZE,
            "SEQ_LEN": SEQ_LEN,
            "HEADS": HEADS,
            "CANONICAL_LAYERS": CANONICAL_LAYERS,
            "SENSITIVITY_LAYERS": SENSITIVITY_LAYERS,
        },
        "fixture": {
            "family": "tiny-a2d-llama",
            "shape": "#77 controlled fixture: hidden 512, 8 heads, fp32",
            "path": "dense default only; sparse_lm_head=False",
            "dtype": "float32",
            "unattributed_includes": (
                "model-forward residual (embeddings, MLP, norms) and any "
                "default trainer work outside the frozen taxonomy"
            ),
        },
        "verdict_source": "wall_off_trials median (instrumentation-off)",
    }


def _sparse_benchmark():
    """Load the #77 benchmark module by path for its fixture helpers.

    Shared rather than re-derived, so this cell's tokenizer and the sparse
    row's are the same object — a second definition could drift.
    """
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[1] / "sparse_lm_head_training.py"
    spec = importlib.util.spec_from_file_location("_sparse_bench_fixture", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the sparse benchmark at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(
    args: argparse.Namespace,
    *,
    vocab_size: int,
    layers: int,
    batch_size: int,
    seed: int,
):
    """Fresh model, trainer and clean batch at an explicit seed.

    Paired arms pass the SAME seed, so weights, clean inputs and the RNG state
    match; different trials pass different seeds, so the mask diagnostic sees
    distinct draws instead of replaying one.
    """
    import torch

    from unturtle.diffusion import DiffusionTrainer, DiffusionTrainingArguments
    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    # Identical construction stream for both arms of a paired trial. Note the
    # trainer re-seeds from `args.seed` below, which is why the same value is
    # passed there too.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TinyA2DLlamaLMHeadModel(
        TinyA2DLlamaConfig(
            vocab_size=vocab_size,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=HIDDEN_SIZE * 2,
            num_hidden_layers=layers,
            num_attention_heads=HEADS,
            num_key_value_heads=HEADS,
            max_position_embeddings=SEQ_LEN * 2,
        )
    ).to(args.device)

    # Reuse the #77 benchmark's minimal tokenizer rather than gpt2: gpt2's EOS
    # is id 50256, which is OUT OF RANGE for a 32000-vocab model and triggers a
    # device-side assert the moment the process writes a mask token.
    tokenizer = _sparse_benchmark()._tokenizer()

    training_args = DiffusionTrainingArguments(
        output_dir="/tmp/unturtle-masked-profile",
        per_device_train_batch_size=batch_size,
        max_steps=1,
        use_cpu=(args.device == "cpu"),
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to=[],
        sparse_lm_head=False,
        loss_norm_type="token",
        # `transformers.Trainer.__init__` calls `set_seed(args.seed)`, which
        # would OVERWRITE the per-trial seeding above with its default of 42 —
        # verified: all three trials then reported one identical RNG
        # fingerprint. Passing the trial seed makes the trainer's own seeding
        # agree with ours instead of erasing it.
        seed=seed,
    )
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=[{"input_ids": [5, 6, 7]}],
        processing_class=tokenizer,
        data_collator=None,
    )
    # CLEAN batch: the device-side process corrupts it every step, and that
    # corruption is the `noising` event. A pre-noised batch would leave nothing
    # for the process to do.
    clean = {
        "input_ids": torch.randint(
            1, vocab_size, (batch_size, SEQ_LEN), device=args.device
        ),
        "labels": None,
        "attention_mask": torch.ones(
            batch_size, SEQ_LEN, dtype=torch.long, device=args.device
        ),
    }
    clean["labels"] = clean["input_ids"].clone()
    optimizer_object = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    optimizer = optimizer_object
    return model, trainer, clean, optimizer


@contextmanager
def instrumented(model, trainer, timer: CudaEventTimer):
    """Install every hook and symbol patch, and restore all of them.

    Restoration is in `finally` so an exception mid-step cannot leave a wrapped
    symbol or a live hook behind — a leaked patch would silently instrument the
    NEXT arm, which is supposed to be the clean one.

    Hooks only RECORD events; none of them synchronizes. Elapsed times are read
    by `timer.collect()` after the outer step boundary's single synchronize.
    """
    import unturtle.diffusion.trainer as trainer_module

    handles = []
    scopes: dict[int, Any] = {}

    def enter(name):
        def hook(module, args):
            scope = timer.measure(name)
            scope.__enter__()
            scopes[id(module)] = scope

        return hook

    def leave(module, args, output):
        scope = scopes.pop(id(module), None)
        if scope is not None:
            scope.__exit__(None, None, None)
        return output

    for layer in model.model.layers:
        attention = layer.self_attn
        handles.append(attention.register_forward_pre_hook(enter("attention")))
        handles.append(attention.register_forward_hook(leave))
    handles.append(model.lm_head.register_forward_pre_hook(enter("lm_head_projection")))
    handles.append(model.lm_head.register_forward_hook(leave))

    # The loss is called through the symbol bound INTO the trainer module
    # (`trainer.py` imports it directly). Patching the kernel module instead
    # would wrap a name nobody calls — the wrong-symbol failure the #166 hybrid
    # gate already produced once.
    original_loss = trainer_module.fast_masked_diffusion_loss

    def timed_loss(*call_args, **call_kwargs):
        with timer.measure("loss"):
            return original_loss(*call_args, **call_kwargs)

    trainer_module.fast_masked_diffusion_loss = timed_loss

    # The process is reached through a trainer INSTANCE attribute, so the
    # instance is proxied rather than the class.
    original_process = trainer.forward_process

    def timed_process(*call_args, **call_kwargs):
        with timer.measure("noising"):
            return original_process(*call_args, **call_kwargs)

    trainer.forward_process = timed_process
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        trainer_module.fast_masked_diffusion_loss = original_loss
        trainer.forward_process = original_process


def one_step(model, trainer, clean, optimizer, timer=None):
    """One dense masked training step."""
    scope = timer.measure if timer is not None else _null_scope

    loss = trainer.compute_loss(model, dict(clean))
    with scope("backward"):
        loss.backward()
    with scope("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss


@contextmanager
def _null_scope(_name: str):
    yield


def timed_step_loop(
    step, *, device: str, timer=None, on_warmup_done=None
) -> tuple[float, list[float]]:
    """Warmup (recorded separately), then the timed window.

    The timer is reset after warmup so events and wall times describe the same
    steps. `collect()` runs once per step, after that step's synchronize.
    """
    import time

    import torch

    cuda = device.startswith("cuda") and torch.cuda.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    sync()
    warmup_start = time.perf_counter()
    with oom_phase("warmup"):
        for _ in range(WARMUP):
            step()
            if timer is not None:
                timer.collect()  # syncs: no boundary sync inside warmup
    sync()
    warmup_seconds = time.perf_counter() - warmup_start
    if timer is not None:
        timer.reset()
    # Hook for the caller to reset peak-memory stats HERE: before this point the
    # peak would include warmup transients and lazy optimizer-state creation,
    # neither of which is the timed step's working set. The persistent
    # allocation at this moment becomes the baseline.
    if on_warmup_done is not None:
        on_warmup_done()

    seconds: list[float] = []
    for _ in range(STEPS):
        sync()
        start = time.perf_counter()
        step()
        # ONE boundary synchronize, then fold the events in WITHOUT syncing
        # again, and stop the clock after. Two things were wrong before: the
        # clock stopped before `collect()`, so collection overhead sat outside
        # `wall_on_trials`, and `collect()` synchronized a second time.
        sync()
        if timer is not None:
            timer.collect(synchronize=False)
        seconds.append(time.perf_counter() - start)
    return warmup_seconds, seconds


def profile_cell_for(args, *, vocab_size: int, layers: int, batch_size: int):
    """One (vocab, layers, batch) cell with interleaved paired arms."""
    import torch

    off_trials: list[float] = []
    on_trials: list[float] = []
    warmup_trials: list[float] = []
    peak_allocated: list[int] = []
    peak_reserved: list[int] = []
    released: list[bool] = []
    per_trial_ops: list[dict[str, dict[str, Any]]] = []
    per_trial_model_forward: list[float] = []
    rng_states: dict[int, dict[str, str]] = {}
    optimizer_provenance: dict[str, Any] = {}
    state_states: dict[int, dict[str, str]] = {}

    def measure(arm: str, seed: int) -> None:
        with oom_phase("build"):
            model, trainer, clean, optimizer = build(
                args,
                vocab_size=vocab_size,
                layers=layers,
                batch_size=batch_size,
                seed=seed,
            )
        model_probe = weakref.ref(model)
        trainer_probe = weakref.ref(trainer)
        # The optimizer holds references to the model's PARAMETERS, so an
        # optimizer that outlives its trial keeps the model's allocation alive
        # even when the model and trainer probes both report released.
        optimizer_probe = weakref.ref(optimizer)
        # After construction, before warmup: the point at which paired arms must
        # already agree.
        rng_states.setdefault(seed, {})[arm] = rng_fingerprint()
        if not optimizer_provenance:
            optimizer_provenance.update(optimizer_fingerprint(optimizer))
        # Fingerprint the weights and inputs too: the RNG state says the draws
        # matched, this says the computation did.
        state_states.setdefault(seed, {})[arm] = state_fingerprint(model, clean)
        timer = CudaEventTimer(device=args.device) if arm == "on" else None

        def run(
            model=model, trainer=trainer, clean=clean, optimizer=optimizer, timer=timer
        ):
            return one_step(model, trainer, clean, optimizer, timer=timer)

        def after_warmup(arm=arm):
            if arm == "off" and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

        if timer is None:
            warmup_seconds, timed = timed_step_loop(
                run, device=args.device, on_warmup_done=after_warmup
            )
        else:
            model_forward_total = {"seconds": 0.0}
            original_forward = model.__class__.__call__

            def timed_forward(self, *call_args, _timer=timer, **call_kwargs):
                with _timer.measure("_model_forward"):
                    return original_forward(self, *call_args, **call_kwargs)

            model.__class__.__call__ = timed_forward
            try:
                with instrumented(model, trainer, timer):
                    warmup_seconds, timed = timed_step_loop(
                        run,
                        device=args.device,
                        timer=timer,
                        on_warmup_done=after_warmup,
                    )
            finally:
                model.__class__.__call__ = original_forward
            observed = timer.result()
            model_forward_total["seconds"] = observed.get(
                "_model_forward", {"inclusive_seconds": 0.0}
            )["inclusive_seconds"]
            per_trial_model_forward.append(model_forward_total["seconds"])
            per_trial_ops.append(
                {k: v for k, v in observed.items() if k != "_model_forward"}
            )

        if arm == "off":
            off_trials.append(sum(timed) / len(timed))
            warmup_trials.append(warmup_seconds)
            if torch.cuda.is_available():
                peak_allocated.append(torch.cuda.max_memory_allocated())
                peak_reserved.append(torch.cuda.max_memory_reserved())
        else:
            on_trials.append(sum(timed) / len(timed))

        del run, model, trainer, clean, optimizer, timer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        released.append(
            model_probe() is None
            and trainer_probe() is None
            and optimizer_probe() is None
        )

    for trial in range(TRIALS):
        order = ("off", "on") if trial % 2 == 0 else ("on", "off")
        # Paired arms share the trial's seed; trials differ.
        seed = TRIAL_SEEDS[trial % len(TRIAL_SEEDS)]
        for arm in order:
            with oom_phase("timed"):
                measure(arm, seed)

    if not all(released):
        return invalid(
            vocab_size,
            layers,
            batch_size,
            "a trial's model, trainer or optimizer outlived its measurement "
            f"call: {released}",
        )

    # Paired arms must agree on the STATE they started from, not merely on the
    # seed integer: three distinct seeds coexisted with one identical RNG state
    # when the trainer re-seeded from its own default.
    for label, table in (("RNG state", rng_states), ("weights/inputs", state_states)):
        mismatched = {
            seed: arms
            for seed, arms in table.items()
            if len(arms) == 2 and arms["off"] != arms["on"]
        }
        if mismatched:
            return invalid(
                vocab_size,
                layers,
                batch_size,
                f"paired arms disagree on {label}, so their walls are not "
                f"comparable: {mismatched}",
            )

    for seed, arms in rng_states.items():
        if set(arms) != {"off", "on"}:
            return invalid(
                vocab_size,
                layers,
                batch_size,
                f"seed {seed} ran arms {sorted(arms)}, expected exactly one "
                "off and one on",
            )

    # Trials must be INDEPENDENT: identical fingerprints across trials mean one
    # stream replayed, which is the defect the trainer-seed fix removed.
    if len({arms.get("off") for arms in rng_states.values()}) < len(rng_states):
        return invalid(
            vocab_size,
            layers,
            batch_size,
            "two trials started from the same RNG state, so they are not "
            f"independent draws: {rng_states}",
        )

    expected_counts = {name: STEPS for name in REQUIRED_EVENTS}
    expected_counts["attention"] = layers * STEPS
    for index, ops in enumerate(per_trial_ops):
        if set(ops) != REQUIRED_EVENTS:
            return invalid(
                vocab_size,
                layers,
                batch_size,
                f"trial {index} observed {sorted(ops)}, expected exactly "
                f"{sorted(REQUIRED_EVENTS)}",
            )
        for name, body in ops.items():
            if body["call_count"] != expected_counts[name]:
                return invalid(
                    vocab_size,
                    layers,
                    batch_size,
                    f"trial {index} event {name!r} ran {body['call_count']} "
                    f"times, expected {expected_counts[name]}",
                )

    # --- mask diagnostic: a SEPARATE replay, after all timing ---
    # Kept out of the measured path entirely. Collecting it inside the OFF arm
    # would put a GPU reduction and a Python append into the very wall time that
    # serves as the verdict. It feeds no wall, event or memory figure.
    #
    # Each trial replays a FULL step, not `forward_process` alone. Verified:
    # eight bare process calls diverge from the real stream at the SECOND draw
    # (0.5469 against 0.4492) because the model forward and dropout also consume
    # the global RNG, so a process-only replay reports a mask regime the timed
    # steps never saw.
    mask_by_trial: list[list[float]] = []
    replay_fingerprints: dict[int, str] = {}
    replay_state_fingerprints: dict[int, str] = {}
    mask_invalid: str | None = None
    for seed in TRIAL_SEEDS:
        model, trainer, clean, optimizer = build(
            args,
            vocab_size=vocab_size,
            layers=layers,
            batch_size=batch_size,
            seed=seed,
        )
        model_probe = weakref.ref(model)
        trainer_probe = weakref.ref(trainer)
        optimizer_probe = weakref.ref(optimizer)
        replay_fingerprints[seed] = rng_fingerprint()
        replay_state_fingerprints[seed] = state_fingerprint(model, clean)
        if trainer.forward_process is None:
            return invalid(
                vocab_size,
                layers,
                batch_size,
                "the trainer has no forward process, so `noising` cannot be "
                "measured and the mask replay has nothing to capture",
            )
        draws: list[float] = []
        captured: list[tuple[Any, Any]] = []
        original_process = trainer.forward_process

        def capturing(
            *call_args,
            _original=original_process,
            _captured=captured,
            **call_kwargs,
        ):
            # Bound as default arguments, not captured from the loop: ruff's
            # B023 flags the closure form, and a late-bound list would collect
            # into whichever trial's list happened to be last.
            output = _original(*call_args, **call_kwargs)
            _captured.append(
                (
                    output.objective_inputs.get("diffusion_mask"),
                    output.objective_inputs.get("labels"),
                )
            )
            return output

        if original_process is not None:
            trainer.forward_process = capturing
        try:
            # Warmup first, exactly as the timed loop does, so the stream is at
            # the same offset before the counted steps begin.
            for _ in range(WARMUP):
                one_step(model, trainer, clean, optimizer)
            captured.clear()
            for _ in range(STEPS):
                one_step(model, trainer, clean, optimizer)
        finally:
            trainer.forward_process = original_process

        for mask, labels in captured:
            if mask is None:
                continue
            # Denominator is MASKABLE tokens, not B x L: every token is
            # eligible in this fixture, but the definition is pinned so a
            # fixture with padding or completion-only supervision cannot
            # silently change what the ratio means.
            if labels is not None:
                maskable = int((labels != -100).sum())
            else:
                maskable = int(mask.numel())
            if maskable == 0:
                # Typed, never a zero-division or a fabricated value.
                mask_invalid = (
                    "a diagnostic step produced zero maskable tokens, so the "
                    "mask fraction has no denominator"
                )
                break
            draws.append(float(int(mask.sum()) / maskable))
        mask_by_trial.append(draws)
        del model, trainer, clean, optimizer, captured, original_process
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (
            model_probe() is not None
            or trainer_probe() is not None
            or optimizer_probe() is not None
        ):
            mask_invalid = (
                "a diagnostic replay's model, trainer or optimizer outlived its "
                "trial, so the next cell would start with its allocation "
                "resident"
            )
        if mask_invalid is not None:
            break
    if mask_invalid is not None:
        return invalid(vocab_size, layers, batch_size, mask_invalid)

    # Completeness, checked rather than assumed: an absent process, a missing
    # capture or zero observations would otherwise publish a cell as `ok` with
    # a silently short distribution.
    if len(mask_by_trial) != TRIALS:
        return invalid(
            vocab_size,
            layers,
            batch_size,
            f"the mask replay produced {len(mask_by_trial)} trials, expected {TRIALS}",
        )
    for index, draws in enumerate(mask_by_trial):
        if len(draws) != STEPS:
            return invalid(
                vocab_size,
                layers,
                batch_size,
                f"mask replay trial {index} captured {len(draws)} draws, "
                f"expected exactly {STEPS}",
            )
    mask_fractions = [value for trial in mask_by_trial for value in trial]
    if len(mask_fractions) != TRIALS * STEPS:
        return invalid(
            vocab_size,
            layers,
            batch_size,
            f"the mask replay produced {len(mask_fractions)} observations, "
            f"expected exactly {TRIALS * STEPS}",
        )
    # Each replay trial must start from the same state as its measured twin,
    # so the distribution describes the execution that was timed.
    # Both tables must be POPULATED before they are compared: an empty table
    # makes the loop below iterate zero times and pass vacuously, which is what
    # happened when the capture was removed but the gate left in place.
    for label, table in (
        ("replay RNG", replay_fingerprints),
        ("replay weights/inputs", replay_state_fingerprints),
    ):
        if len(table) != len(TRIAL_SEEDS):
            return invalid(
                vocab_size,
                layers,
                batch_size,
                f"{label} fingerprints cover {len(table)} of "
                f"{len(TRIAL_SEEDS)} trials, so the replay-versus-measured "
                "comparison would pass without checking anything",
            )

    # RNG state AND weights/inputs, both: matching the RNG alone would let a
    # replay with different weights or a different clean batch pass as "the
    # execution that was timed".
    for label, replay_table, measured_table in (
        ("RNG state", replay_fingerprints, rng_states),
        ("weights/inputs", replay_state_fingerprints, state_states),
    ):
        for seed, fingerprint in replay_table.items():
            measured = measured_table.get(seed, {}).get("off")
            if measured is not None and fingerprint != measured:
                return invalid(
                    vocab_size,
                    layers,
                    batch_size,
                    f"the mask replay for seed {seed} started from {label} "
                    f"{fingerprint} but the measured trial started from "
                    f"{measured}, so the replayed distribution is not the one "
                    "that was timed",
                )

    def per_step(name: str) -> float:
        return statistics.median(
            ops[name]["inclusive_seconds"] / STEPS for ops in per_trial_ops
        )

    model_forward = statistics.median(
        total / STEPS for total in per_trial_model_forward
    )
    # Residual PER TRIAL, before any median: a difference of medians mixes
    # values from different trials, so one trial's negative residual could be
    # masked by another's surplus.
    per_trial_residual = [
        (
            total
            - ops["attention"]["inclusive_seconds"]
            - ops["lm_head_projection"]["inclusive_seconds"]
        )
        / STEPS
        for total, ops in zip(per_trial_model_forward, per_trial_ops, strict=True)
    ]
    import math

    for index, value in enumerate(per_trial_residual):
        if not math.isfinite(value) or value < 0:
            return invalid(
                vocab_size,
                layers,
                batch_size,
                f"trial {index} model-forward residual is {value:.6f}s: "
                "attention plus the LM head exceed the whole forward, so "
                "attribution is wrong for that trial",
            )
    residual = statistics.median(per_trial_residual)
    if residual < 0:
        # Not clamped: a negative residual means the hook topology or the
        # event pairing is wrong, and hiding it would publish a broken profile.
        return invalid(
            vocab_size,
            layers,
            batch_size,
            f"model-forward residual is negative ({residual:.6f}s): attention "
            "plus the LM head exceed the whole forward, so attribution is wrong",
        )

    events = [
        OperationEvent(
            name=name,
            inclusive_seconds=per_step(name),
            call_count=layers if name == "attention" else 1,
            parent=None,
            coverage_eligible=True,
        )
        for name in sorted(REQUIRED_EVENTS)
    ]
    cell = ProfileCell(
        family="masked",
        cell=f"dense_training_step_v{vocab_size}_l{layers}",
        batch_size=batch_size,
        sequence_length=SEQ_LEN,
        dtype="float32",
        wall_off_trials=off_trials,
        wall_on_trials=on_trials,
        events=events,
        peak_allocated_bytes=max(peak_allocated) if peak_allocated else None,
        peak_reserved_bytes=max(peak_reserved) if peak_reserved else None,
        warmup_seconds=statistics.median(warmup_trials),
        hardware=(
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        extra={
            "vocab_size": vocab_size,
            "layers": layers,
            "steps_per_trial": STEPS,
            "warmup_steps": WARMUP,
            "trials": TRIALS,
            # Diagnostics, NOT coverage: the residual is real work (embeddings,
            # MLP, norms) but not a frozen taxonomy operation, so it stays in
            # `unattributed_seconds` rather than becoming a `model_other` event.
            "model_forward_inclusive_seconds": model_forward,
            "model_forward_residual_seconds": residual,
            "model_forward_residual_per_trial": per_trial_residual,
            # The distribution with its provenance, not a point: this row's
            # regime IS the mask ratio, and a single number would hide both the
            # spread and where it came from.
            # Three things kept apart, because collapsing them invites reading
            # a fixed-seed execution as the process's expected behaviour.
            "sampling_contract": (
                "timesteps are sampled from the frozen MaskedDiffusionProcess "
                "distribution (linear alpha schedule, t ~ U(0,1)); this "
                "producer pins no mask ratio"
            ),
            "realized_mask_fraction": {
                "source": "separate_diagnostic_replay",
                "raw_by_trial": mask_by_trial,
                "observations": len(mask_fractions),
                "median": (
                    statistics.median(mask_fractions) if mask_fractions else None
                ),
                "mean": (statistics.fmean(mask_fractions) if mask_fractions else None),
                "min": min(mask_fractions) if mask_fractions else None,
                "max": max(mask_fractions) if mask_fractions else None,
                "denominator": "maskable_tokens",
                "trial_seeds": list(TRIAL_SEEDS),
                "used_for": "diagnostic only; feeds no wall, event or memory figure",
            },
            "realized_mask_fraction_interpretation": (
                "What THIS artifact's paired execution realized under its fixed "
                "seed schedule, not an estimate of the process's population "
                "mask rate. At B=1 each step's fraction reflects close to a "
                "single timestep draw, so 24 observations retain seed-dependent "
                "bias; larger batches average within the batch and narrow the "
                "spread, which is why every cell stores its own raw "
                "distribution. Report it as 'this execution was median X, range "
                "Y-Z', never as 'the default regime is X% masking'."
            ),
            "peak_representative": "max over trials (fixed before measuring)",
            "rng_state_fingerprints": rng_states,
            "replay_rng_fingerprints": replay_fingerprints,
            "replay_state_fingerprints": replay_state_fingerprints,
            "weight_input_fingerprints": state_states,
            # The RESOLVED optimizer configuration, read off the instantiated
            # object: "default-resolved" was a placeholder, not a value, and
            # which AdamW kernel runs affects the step time being attributed.
            "optimizer": optimizer_provenance,
            # Both series, because they answer different questions: allocated
            # is the working set, reserved is what the caching allocator held
            # from the driver. The protocol records both, and only the raw
            # arrays let a reader check the max aggregates above.
            "peak_allocated_per_trial": peak_allocated,
            "peak_reserved_per_trial": peak_reserved,
            "per_trial_call_counts": [
                {name: body["call_count"] for name, body in sorted(ops.items())}
                for ops in per_trial_ops
            ],
        },
    )
    return profile_cell(cell)


def invalid(vocab_size: int, layers: int, batch_size: int, reason: str):
    return {
        "family": "masked",
        "cell": f"dense_training_step_v{vocab_size}_l{layers}",
        "batch_size": batch_size,
        "status": "measurement_invalid",
        "reason": reason,
    }


def main() -> None:
    import torch

    args = parse_args()
    require_supported_device(args.device)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []

    plan = [
        (vocab, CANONICAL_LAYERS, batch)
        for vocab in VOCAB_SIZES
        for batch in BATCH_SIZES
    ]
    # One bounded depth-sensitivity cell per vocabulary.
    plan += [(vocab, SENSITIVITY_LAYERS, SENSITIVITY_BATCH) for vocab in VOCAB_SIZES]

    for vocab, layers, batch in plan:
        try:
            record = profile_cell_for(
                args, vocab_size=vocab, layers=layers, batch_size=batch
            )
        except OomInPhase as error:
            record = {
                "family": "masked",
                "cell": f"dense_training_step_v{vocab}_l{layers}",
                "batch_size": batch,
                "status": "oom",
                "oom_phase": error.phase,
                "reason": str(error.cause)[:400],
                "sequence_length": SEQ_LEN,
                "vocab_size": vocab,
                "layers": layers,
                "precision": "float32",
                "hardware": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "cpu"
                ),
            }
        cells.append(record)
        print(
            json.dumps({k: v for k, v in record.items() if k != "operations"})[:340],
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {"run": provenance(args), "cells": cells}
    (out / "masked_loss_profile.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(cells)} cells to {out / 'masked_loss_profile.json'}")


if __name__ == "__main__":
    main()
