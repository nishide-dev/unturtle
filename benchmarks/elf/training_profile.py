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

"""#166 Stage-1 — end-to-end profile of the ELF training step.

Profiled FIRST because it is the largest unmeasured default path in the
acceleration ledger, **not** because it is a preselected target. This producer
emits profiles and nothing else: no candidate is declared, no kernel proposed,
no dispatch default touched. Stage 2 owns selection.

Stage 0's hypothesis, from reading the call graph rather than a profile: at the
shipped defaults (`self_cond_prob=0.5`, `num_self_cond_cfg_tokens=4`, both
non-zero) one step performs up to TWO extra `no_grad` model forwards before the
trained forward and backward. Whether that call count converts into wall share
is what this measures — a `no_grad` forward carries no backward and need not
cost what the trained forward costs.

**The two auxiliary forwards are separate events.** They sit at distinct call
sites under different conditions (`compute_shared_uncond` fires when
`self_cond_prob > 0 or num_self_cond_cfg_tokens > 0`; the conditional one only
when `self_cond_prob > 0`), so recording them as one `self_conditioning` event
would destroy the finding this profile exists to test.

Verdict discipline, per `docs/acceleration-profile-protocol.md`: the
instrumentation-OFF wall clock is the verdict, the instrumented pass only
attributes it, coverage sums mutually exclusive intervals, and the unattributed
remainder is reported rather than normalized away.

Usage::

    .venv/bin/python benchmarks/elf/training_profile.py --device cuda:0
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

from unturtle.eval.operation_timer import OperationTimer, caller_scope
from unturtle.eval.profile_harness import OperationEvent, ProfileCell, profile_cell

#: Frozen measurement window. Fixed in the module, not read from the CLI, so a
#: verdict cannot depend on the caller passing a large enough window — the
#: defect the #166 sanity gates surfaced three times.
TRIALS = 3
STEPS = 8
WARMUP = 3

#: Protocol representative batches. Larger sizes are attempted and their OOM is
#: recorded as typed data rather than dropped (#152).
BATCH_SIZES = (1, 8, 32)

#: Call sites inside `elf_training_loss`, mapped to taxonomy event names. The
#: two auxiliary forwards MUST stay distinct; see the module docstring.
CALL_SITE_EVENTS = {
    "compute_shared_uncond": "sc_shared_uncond_forward",
    "get_sc_cond_and_uncond": "sc_conditional_forward",
    "elf_training_loss": "trained_forward",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(b) for b in BATCH_SIZES),
        help="representative batches; OOM is recorded as typed data",
    )
    parser.add_argument("--out", default="benchmarks/results/elf_training_profile")
    return parser.parse_args()


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Run identity, recorded at run time and never hand-edited."""
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
        "frozen_constants": {"TRIALS": TRIALS, "STEPS": STEPS, "WARMUP": WARMUP},
        "verdict_source": "wall_off_trials median (instrumentation-off)",
    }


def build(args: argparse.Namespace, batch_size: int):
    """Model, encoder shim, optimizer and one batch at smoke scale.

    Constructs `ELF` directly rather than loading the #153 checkpoint: this is a
    step-SHAPE profile, and a real checkpoint would add download and load cost
    without changing which operations the step performs. The config is a plain
    namespace carrying the oracle defaults the objective reads, matching how
    `benchmarks/elf/stage3_reduced_gate.py` assembles it — so a missing key
    cannot become a silent default.

    NON-QUALITY-BEARING: reads no generation output and does not reinterpret
    Stage-3 results.
    """
    import torch
    from unturtle_elf._reference.model import ELF

    class Config:
        pass

    config = Config()
    # Exactly the fields the objective reads, enumerated from
    # `training.py` and `_reference/sampling_utils.py` rather than guessed, so a
    # missing key cannot become a silent default. `vocab_size` is kept for the
    # model constructor below, not for the objective.
    for key, value in {
        "pad_token": "pad",
        "t_eps": 5e-2,
        "self_cond_prob": 0.5,
        "self_cond_cfg_min": 0.5,
        "self_cond_cfg_max": 5.0,
        "num_self_cond_cfg_tokens": 4,
        "time_schedule": "logit_normal",
        "denoiser_noise_scale": 1.0,
        "denoiser_p_mean": 0.0,
        "denoiser_p_std": 1.0,
        "decoder_noise_scale": 1.0,
        "decoder_p_mean": 0.0,
        "decoder_p_std": 1.0,
        "decoder_prob": 0.5,
        "latent_mean": 0.0,
        "latent_std": 1.0,
        "vocab_size": 32000,
    }.items():
        setattr(config, key, value)

    encoder_dim = 128
    seq_len = args.seq_len
    torch.manual_seed(7)
    model = ELF(
        text_encoder_dim=encoder_dim,
        max_length=seq_len,
        hidden_size=256,
        depth=4,
        num_heads=4,
        bottleneck_dim=64,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=config.vocab_size,
    ).to(device=args.device, dtype=getattr(torch, args.dtype))

    class EncoderShim(torch.nn.Module):
        """Stands in for the frozen T5 encoder at smoke scale.

        The encoding event is still timed, so its share is visible rather than
        folded into another event; only its magnitude is unrepresentative of the
        real encoder, which the record states.
        """

        def forward(self, input_ids, attention_mask=None, deterministic=True):
            return torch.randn(
                input_ids.shape[0],
                seq_len,
                encoder_dim,
                device=input_ids.device,
            )

    # CPU rows only: mask construction and the H2D copy belong to the timed
    # `data_collation` event, so `build()` must not pre-move anything.
    import numpy as np

    rows = np.random.default_rng(7).integers(
        1, config.vocab_size, size=(batch_size, seq_len), dtype=np.int64
    )
    cpu_batch = {
        "input_ids": rows,
        "true_lengths": np.full((batch_size, 1), int(seq_len * 0.85), dtype=np.int32),
    }
    from unturtle_elf.training import build_muon_optimizer

    optimizer = build_muon_optimizer(model, lr=1e-4)
    return model, EncoderShim().to(args.device), optimizer, cpu_batch, config


def collate(cpu_batch: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Oracle mask construction and host-to-device transfer.

    Runs every step inside the `data_collation` scope so padding, mask
    derivation and the H2D copy are all charged to it. Masks follow
    `stage3_reduced_gate.make_batch`: derived from true lengths, never all-ones,
    which would train on padding.
    """
    import numpy as np
    import torch
    from unturtle_elf._reference.encoder_utils import build_self_attn_cond_masks

    ids = cpu_batch["input_ids"]
    lengths = cpu_batch["true_lengths"]
    batch_size, seq_len = ids.shape
    positions = np.arange(seq_len)[None, :]
    is_cond = positions < np.zeros((batch_size, 1), dtype=np.int32)
    is_valid = positions < lengths
    encoder_attn, attn, cond = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
    return {
        "input_ids": torch.from_numpy(ids).long().to(device),
        "attention_mask": torch.from_numpy(attn).to(device),
        "encoder_attention_mask": torch.from_numpy(encoder_attn).to(device),
        "cond_seq_mask": torch.from_numpy(cond).to(device),
    }


def instrument_encoder(encoder, timer: OperationTimer):
    """Time the frozen-encoder pass.

    `encode_text` calls the encoder from inside `elf_training_loss`, so
    caller-frame keying on `model.__call__` cannot see it — the protocol lists
    `t5_encoding` as its own required event, and an unwrapped encoder would
    land in the unattributed remainder instead.
    """
    original = encoder.__class__.forward

    def timed(self, *call_args, **call_kwargs):
        with timer.measure("t5_encoding"):
            return original(self, *call_args, **call_kwargs)

    encoder.__class__.forward = timed
    return original


def instrument(model, timer: OperationTimer):
    """Attribute each `model(...)` call to its call site.

    The pack is left untouched: reference semantics must not change while
    profiling. Attribution is by CALLER, because `elf_training_loss` reaches the
    same callable from three different places and the whole point is to keep
    them apart.
    """
    original = model.__class__.__call__

    def dispatching(self, *call_args, **call_kwargs):
        site = caller_scope(depth=2)
        event = CALL_SITE_EVENTS.get(site)
        if event is None:
            return original(self, *call_args, **call_kwargs)
        with timer.measure(event):
            return original(self, *call_args, **call_kwargs)

    model.__class__.__call__ = dispatching
    return original


def main() -> None:
    import torch

    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
    records: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        try:
            record = profile_batch(args, batch_size)
        except torch.cuda.OutOfMemoryError as error:
            record = {
                "family": "elf",
                "cell": "training_step",
                "batch_size": batch_size,
                "status": "oom",
                "reason": str(error)[:400],
            }
        records.append(record)
        print(
            json.dumps({k: v for k, v in record.items() if k not in ("operations",)})[
                :600
            ],
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {"run": provenance(args), "cells": records}
    (out / "elf_training_profile.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(records)} cells to {out / 'elf_training_profile.json'}")


def profile_batch(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    """One (batch) cell: instrumentation-off trials, then an attributed pass.

    Both passes measure the SAME window. Warmup is run, its time recorded
    separately, the timer reset, and only the timed steps accumulate — an
    earlier version divided event totals by `WARMUP + STEPS` while the wall
    kept only `STEPS`, so coverage and wall described different intervals.
    """
    import torch

    required = {
        "data_collation",
        "t5_encoding",
        "sc_shared_uncond_forward",
        "sc_conditional_forward",
        "trained_forward",
        "objective_loss",
        "backward",
        "optimizer_step",
    }

    off_trials: list[float] = []
    warmup_trials: list[float] = []
    peak_allocated: list[int] = []
    peak_reserved: list[int] = []
    released: list[bool] = []

    for _ in range(TRIALS):
        model, encoder, optimizer, cpu_batch, config = build(args, batch_size)
        model_probe = weakref.ref(model)
        encoder_probe = weakref.ref(encoder)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def run_off(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            cpu_batch=cpu_batch,
            config=config,
        ):
            return one_step(model, encoder, optimizer, cpu_batch, config, args)

        warmup_seconds, timed = timed_step_loop(run_off, device=args.device)
        off_trials.append(sum(timed) / len(timed))
        warmup_trials.append(warmup_seconds)
        if torch.cuda.is_available():
            peak_allocated.append(torch.cuda.max_memory_allocated())
            peak_reserved.append(torch.cuda.max_memory_reserved())
        # `run_off` binds the model and encoder as DEFAULT ARGUMENTS, so it must
        # be deleted BEFORE them or the bindings keep the previous trial's
        # weights resident while the next `build()` runs (#173's closure leak).
        del run_off, model, encoder, optimizer, cpu_batch, config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        released.append(model_probe() is None and encoder_probe() is None)

    on_trials: list[float] = []
    per_trial_ops: list[dict[str, dict[str, Any]]] = []

    for _ in range(TRIALS):
        model, encoder, optimizer, cpu_batch, config = build(args, batch_size)
        model_probe = weakref.ref(model)
        encoder_probe = weakref.ref(encoder)
        timer = OperationTimer(device=args.device)
        original = instrument(model, timer)
        encoder_original = instrument_encoder(encoder, timer)

        def run_on(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            cpu_batch=cpu_batch,
            config=config,
            timer=timer,
        ):
            return one_step(
                model, encoder, optimizer, cpu_batch, config, args, timer=timer
            )

        try:
            warmup_seconds, timed = timed_step_loop(
                run_on, device=args.device, timer=timer
            )
        finally:
            model.__class__.__call__ = original
            encoder.__class__.forward = encoder_original
        on_trials.append(sum(timed) / len(timed))
        per_trial_ops.append(timer.result())
        del run_on, model, encoder, optimizer, cpu_batch, config, timer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        released.append(model_probe() is None and encoder_probe() is None)

    if not all(released):
        return {
            "family": "elf",
            "cell": "training_step",
            "batch_size": batch_size,
            "status": "measurement_invalid",
            "reason": (
                "a trial's model or encoder outlived its measurement call, so a "
                "later trial may carry an earlier trial's allocation: "
                f"{released}"
            ),
        }

    # Required events and their per-step call counts are ASSERTED, not inferred
    # from whichever timer names happened to appear: a broken caller lookup
    # would otherwise silently drop an event and still produce an artifact.
    for index, ops in enumerate(per_trial_ops):
        missing = sorted(required - set(ops))
        if missing:
            return {
                "family": "elf",
                "cell": "training_step",
                "batch_size": batch_size,
                "status": "measurement_invalid",
                "reason": (
                    f"instrumented trial {index} recorded no {missing}; call-site "
                    "attribution is broken and the taxonomy is incomplete"
                ),
                "observed_events": sorted(ops),
            }
        for name, body in ops.items():
            if body["call_count"] != STEPS:
                return {
                    "family": "elf",
                    "cell": "training_step",
                    "batch_size": batch_size,
                    "status": "measurement_invalid",
                    "reason": (
                        f"trial {index} event {name!r} ran {body['call_count']} "
                        f"times over {STEPS} timed steps; the taxonomy expects "
                        "exactly one call per step"
                    ),
                }

    def per_step(name: str) -> float:
        return statistics.median(
            ops[name]["inclusive_seconds"] / STEPS for ops in per_trial_ops
        )

    # `objective_loss` is measured INCLUSIVE of the encoder and the three
    # forwards, so its own share is the difference. Publishing only the
    # inclusive parent would push the objective's real work — CE/L2, target
    # construction, masking, normalization — into the unattributed remainder.
    contained = (
        "t5_encoding",
        "sc_shared_uncond_forward",
        "sc_conditional_forward",
        "trained_forward",
    )
    objective_exclusive = statistics.median(
        (
            ops["objective_loss"]["inclusive_seconds"]
            - sum(ops[name]["inclusive_seconds"] for name in contained)
        )
        / STEPS
        for ops in per_trial_ops
    )

    # Every published event is a mutually exclusive sibling: the four contained
    # operations, the objective's own exclusive remainder, and the three outside
    # it. No parent is published as eligible, so coverage cannot double count.
    events = [
        OperationEvent(
            name=name,
            inclusive_seconds=per_step(name),
            call_count=STEPS // STEPS,
            parent=None,
            coverage_eligible=True,
        )
        for name in ("data_collation", "backward", "optimizer_step", *contained)
    ]
    events.append(
        OperationEvent(
            name="objective_loss_exclusive",
            inclusive_seconds=objective_exclusive,
            call_count=1,
            parent=None,
            coverage_eligible=True,
        )
    )

    cell = ProfileCell(
        family="elf",
        cell="training_step",
        batch_size=batch_size,
        sequence_length=args.seq_len,
        dtype=args.dtype,
        wall_off_trials=off_trials,
        wall_on_trials=on_trials,
        events=events,
        # max, fixed in advance: a capacity question wants the worst trial, and
        # choosing between max and median after seeing the numbers would be a
        # post-hoc pick.
        peak_allocated_bytes=max(peak_allocated) if peak_allocated else None,
        peak_reserved_bytes=max(peak_reserved) if peak_reserved else None,
        warmup_seconds=statistics.median(warmup_trials),
        hardware=(
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        extra={
            "steps_per_trial": STEPS,
            "warmup_steps": WARMUP,
            "trials": TRIALS,
            "self_cond_prob": 0.5,
            "num_self_cond_cfg_tokens": 4,
            "peak_representative": "max over trials (fixed before measuring)",
            "peak_allocated_per_trial": peak_allocated,
            "peak_reserved_per_trial": peak_reserved,
            "objective_loss_inclusive_seconds": per_step("objective_loss"),
            "per_trial_call_counts": [
                {name: body["call_count"] for name, body in sorted(ops.items())}
                for ops in per_trial_ops
            ],
            "non_quality_bearing": (
                "step-shape profile only; reads no generation output and does "
                "not reinterpret Stage-3 results"
            ),
        },
    )
    return profile_cell(cell)


def one_step(model, encoder, optimizer, cpu_batch, config, args, timer=None):
    """One training step, with collation and transfer inside the timed scope.

    `data_collation` covers the oracle's mask construction and the host-to-device
    transfer, performed every step from CPU rows. An earlier version pre-moved
    the batch in `build()` and timed a dictionary comprehension, which measured
    nothing.
    """
    import torch
    from unturtle_elf.training import elf_training_loss

    scope = timer.measure if timer is not None else _null_scope

    with scope("data_collation"):
        batch = collate(cpu_batch, device=args.device)

    with scope("objective_loss"):
        loss, _metrics, _aux = elf_training_loss(
            model,
            encoder,
            batch,
            config,
            dropout_generator=torch.Generator(device="cpu").manual_seed(0),
        )
    with scope("backward"):
        loss.backward()
    with scope("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return loss


@contextmanager
def _null_scope(_name: str):
    yield


def timed_step_loop(step, *, device: str, timer=None) -> tuple[float, list[float]]:
    """Run warmup, then the timed window. Returns (warmup_seconds, timings).

    The timer is reset after warmup so events cover exactly the same steps the
    returned wall times do.
    """
    import time

    import torch

    cuda = device.startswith("cuda") and torch.cuda.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    sync()
    warmup_start = time.perf_counter()
    for _ in range(WARMUP):
        step()
    sync()
    warmup_seconds = time.perf_counter() - warmup_start
    if timer is not None:
        timer.reset()

    seconds: list[float] = []
    for _ in range(STEPS):
        sync()
        start = time.perf_counter()
        step()
        sync()
        seconds.append(time.perf_counter() - start)
    return warmup_seconds, seconds


if __name__ == "__main__":
    main()
