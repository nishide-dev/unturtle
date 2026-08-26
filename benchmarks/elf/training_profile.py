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
    ).to(args.device)

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

    # Masks are derived with the ORACLE's collation semantics rather than set
    # to all-ones: `stage3_reduced_gate.make_batch` records that all-ones masks
    # would train on padding. Rows get a realistic sub-full true length so the
    # padded region is excluded, and `cond_seq_mask` is the uncond-OWT case.
    import numpy as np
    from unturtle_elf._reference.encoder_utils import build_self_attn_cond_masks

    true_lengths = np.full((batch_size, 1), int(seq_len * 0.85), dtype=np.int32)
    positions = np.arange(seq_len)[None, :]
    is_cond = positions < np.zeros((batch_size, 1), dtype=np.int32)
    is_valid = positions < true_lengths
    encoder_attn, attn, cond = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
    batch = {
        "input_ids": torch.randint(
            1, config.vocab_size, (batch_size, seq_len), device=args.device
        ),
        "attention_mask": torch.from_numpy(attn).to(args.device),
        "encoder_attention_mask": torch.from_numpy(encoder_attn).to(args.device),
        "cond_seq_mask": torch.from_numpy(cond).to(args.device),
    }
    from unturtle_elf.training import build_muon_optimizer

    optimizer = build_muon_optimizer(model, lr=1e-4)
    return model, EncoderShim().to(args.device), optimizer, batch, config


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
    """One (batch) cell: instrumentation-off trials, then an attributed pass."""
    import torch
    from unturtle_elf.training import elf_training_loss

    def one_step(model, encoder, optimizer, batch, config, timer=None):
        generator = torch.Generator(device="cpu").manual_seed(0)
        scope = timer.measure if timer is not None else None
        if scope is not None:
            with scope("data_collation"):
                prepared = {key: value for key, value in batch.items()}
        else:
            prepared = {key: value for key, value in batch.items()}
        if scope is not None:
            # `objective_loss` CONTAINS the encoder and the three model
            # forwards, so it is a parent event and must not be
            # coverage_eligible alongside them — that is the nested
            # double-count the harness refuses.
            with scope("objective_loss"):
                loss, _metrics, _aux = elf_training_loss(
                    model,
                    encoder,
                    prepared,
                    config,
                    dropout_generator=generator,
                )
        else:
            loss, _metrics, _aux = elf_training_loss(
                model,
                encoder,
                prepared,
                config,
                dropout_generator=generator,
            )
        if scope is not None:
            with scope("backward"):
                loss.backward()
            with scope("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return loss

    # --- instrumentation-OFF trials: the verdict ---
    off_trials: list[float] = []
    peak_allocated = None
    peak_reserved = None
    for _ in range(TRIALS):
        model, encoder, optimizer, batch, config = build(args, batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Bound as default arguments, not captured: ruff's B023 flags the
        # closure form, and a late-bound lambda inside a trial loop would time
        # whichever objects the last iteration happened to leave behind.
        def run_off(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            batch=batch,
            config=config,
        ):
            return one_step(model, encoder, optimizer, batch, config)

        seconds = timed_step_loop(run_off, device=args.device)
        off_trials.append(statistics.median(seconds))
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
        del model, encoder, optimizer, batch, config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- instrumentation-ON trials: attribution only ---
    on_trials: list[float] = []
    operations: dict[str, dict[str, Any]] = {}
    for _ in range(TRIALS):
        model, encoder, optimizer, batch, config = build(args, batch_size)
        timer = OperationTimer(device=args.device)
        original = instrument(model, timer)
        encoder_original = instrument_encoder(encoder, timer)
        try:

            def run_on(
                model=model,
                encoder=encoder,
                optimizer=optimizer,
                batch=batch,
                config=config,
                timer=timer,
            ):
                return one_step(model, encoder, optimizer, batch, config, timer=timer)

            seconds = timed_step_loop(run_on, device=args.device)
        finally:
            model.__class__.__call__ = original
            encoder.__class__.forward = encoder_original
        on_trials.append(statistics.median(seconds))
        for name, body in timer.result().items():
            slot = operations.setdefault(
                name, {"inclusive_seconds": [], "call_count": []}
            )
            slot["inclusive_seconds"].append(body["inclusive_seconds"])
            slot["call_count"].append(body["call_count"])
        del model, encoder, optimizer, batch, config, timer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Per-step medians across trials. Every event is a sibling of the others —
    # `trained_forward` does not contain the auxiliary forwards, and backward
    # and optimizer_step are disjoint — so all are coverage_eligible.
    total_steps = WARMUP + STEPS
    # `objective_loss` is the PARENT of the encoder and the three forwards, so
    # exactly one level of that path may be coverage_eligible. The children are
    # eligible because the question Stage 0 asks is about them individually;
    # the parent is retained for diagnosis. `backward`, `optimizer_step` and
    # `data_collation` are siblings and eligible.
    children = {
        "t5_encoding",
        "sc_shared_uncond_forward",
        "sc_conditional_forward",
        "trained_forward",
    }
    events = [
        OperationEvent(
            name=name,
            inclusive_seconds=statistics.median(body["inclusive_seconds"])
            / total_steps,
            call_count=int(statistics.median(body["call_count"]) // total_steps),
            parent="objective_loss" if name in children else None,
            coverage_eligible=name != "objective_loss",
        )
        for name, body in sorted(operations.items())
    ]
    cell = ProfileCell(
        family="elf",
        cell="training_step",
        batch_size=batch_size,
        sequence_length=args.seq_len,
        dtype="float32",
        wall_off_trials=off_trials,
        wall_on_trials=on_trials,
        events=events,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        hardware=(
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        extra={
            "steps_per_trial": STEPS,
            "warmup": WARMUP,
            "self_cond_prob": 0.5,
            "num_self_cond_cfg_tokens": 4,
            "non_quality_bearing": (
                "step-shape profile only; reads no generation output and does "
                "not reinterpret Stage-3 results"
            ),
        },
    )
    return profile_cell(cell)


def timed_step_loop(step, *, device: str) -> list[float]:
    """Sync-bracketed steady-state timings with warmup excluded."""
    import time

    import torch

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    seconds: list[float] = []
    for index in range(WARMUP + STEPS):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        step()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if index >= WARMUP:
            seconds.append(elapsed)
    return seconds


if __name__ == "__main__":
    main()
