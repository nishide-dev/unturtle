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

#: Frozen fixture identity — the #154 smoke path, not a synthetic stand-in.
#: Every component is revision-pinned, including T5: `stage3_reduced_gate`
#: resolves the encoder by NAME only, which is not sufficient provenance for an
#: artifact, so the resolved commit is recorded at run time.
CHECKPOINT = "embedded-language-flows/ELF-B-owt-torch"
CHECKPOINT_REVISION = "146f84133c1389bfd4ef47f14ec7a955da22faa7"
DATASET = "embedded-language-flows/openwebtext-t5"
DATASET_REVISION = "0a8443e847ee6206e4737a6b9a93218347eabc08"
#: Frozen from the pilot's resolved commit. Recording what a run happened to
#: fetch documents the past; pinning it makes the next run reproducible.
T5_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
TOTAL_SHARDS = 75
SEQUENCE_LENGTH = 1024
GRAD_CLIP = 1.0
EMA_DECAY = 0.9999
LEARNING_RATE = 2.5e-4

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


class OomInPhase(Exception):
    """An OOM tagged with the phase it happened in.

    A typed OOM is a RESULT (#152), and `build` / `warmup` / `timed` are
    different findings: failing to allocate the model is not the same as
    failing under the timed step's activations.
    """

    def __init__(self, phase: str, cause: BaseException) -> None:
        super().__init__(f"OOM during {phase}: {cause}")
        self.phase = phase
        self.cause = cause


@contextmanager
def oom_phase(phase: str):
    """Re-raise a CUDA OOM tagged with the phase, leaving others untouched."""
    import torch

    try:
        yield
    except torch.cuda.OutOfMemoryError as error:
        raise OomInPhase(phase, error) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    # No --seq-len or --dtype: the run is pinned to SEQUENCE_LENGTH with fp32
    # master params under bf16 autocast, and a flag that could not change the
    # measurement while changing the RECORD would make the artifact lie about
    # what it measured (it recorded 256 while truncating to 1024).
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
        "frozen_constants": {
            "TRIALS": TRIALS,
            "STEPS": STEPS,
            "WARMUP": WARMUP,
            "SEQUENCE_LENGTH": SEQUENCE_LENGTH,
            "GRAD_CLIP": GRAD_CLIP,
            "EMA_DECAY": EMA_DECAY,
            "LEARNING_RATE": LEARNING_RATE,
            "grad_accum_steps": 1,
        },
        "fixture": {
            "checkpoint": f"{CHECKPOINT}@{CHECKPOINT_REVISION}",
            "dataset": f"{DATASET}@{DATASET_REVISION}",
            "model_init": "fresh-init from config.yml (no checkpoint weights)",
            "precision": "fp32 master params; bf16 autocast over the objective",
            "unattributed_includes": "grad clipping and the EMA update",
        },
        "verdict_source": "wall_off_trials median (instrumentation-off)",
    }


def load_fixture(args: argparse.Namespace) -> dict[str, Any]:
    """Download and materialize the frozen #154 inputs. NOT timed.

    Data materialization is deliberately outside every timed scope; only the
    per-step collation and transfer are charged to `data_collation`.
    """
    import pyarrow as pa
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from transformers import T5EncoderModel

    config_path = hf_hub_download(
        CHECKPOINT, "config.yml", revision=CHECKPOINT_REVISION
    )
    raw_config = yaml.safe_load(pathlib.Path(config_path).read_text())

    encoder_name = str(raw_config.get("encoder_model_name", "t5-small"))
    inner = T5EncoderModel.from_pretrained(encoder_name, revision=T5_REVISION)
    encoder_revision = getattr(getattr(inner, "config", None), "_commit_hash", None)
    if encoder_revision != T5_REVISION:
        raise RuntimeError(
            f"encoder resolved to {encoder_revision!r}, expected the pinned "
            f"{T5_REVISION!r}: an unpinned encoder makes the artifact "
            "irreproducible"
        )
    inner = inner.to(args.device).eval().requires_grad_(False)

    class EncoderShim(torch.nn.Module):
        """Adapts HF `T5EncoderModel` to the oracle's encoder contract.

        Correction #4 from the Stage-3 gate: the oracle hands its 3-D float
        self-attention mask straight to `T5EncoderModel`, which transformers 5.x
        rejects (`bitwise_and` on Float). For the UNCONDITIONAL scope every
        query row of that mask equals the 2-D validity mask, so collapsing is
        exact here — a conditional run would need a different adapter, and this
        profile is unconditional.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask=None, deterministic=True):
            del deterministic
            if attention_mask is not None and attention_mask.dim() == 3:
                attention_mask = attention_mask[:, 0, :]
            if attention_mask is not None:
                attention_mask = attention_mask.long()
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    encoder = EncoderShim(inner).to(args.device).eval()

    # Shard naming and the Arrow stream reader follow
    # `stage3_reduced_gate._shard_path` / `load_table`; one shard is enough for
    # a step-shape profile, and the row schedule is fixed regardless.
    shard = hf_hub_download(
        DATASET,
        f"data-{0:05d}-of-{TOTAL_SHARDS:05d}.arrow",
        revision=DATASET_REVISION,
        repo_type="dataset",
    )
    with pa.memory_map(shard, "rb") as source:
        table = pa.ipc.open_stream(source).read_all()
    return {
        "raw_config": raw_config,
        "encoder": encoder,
        "encoder_name": encoder_name,
        "encoder_revision": encoder_revision,
        "table": table,
        "torch": torch,
    }


def build(args: argparse.Namespace, fixture: dict[str, Any], batch_size: int):
    """Fresh-init ELF-B from the frozen config, plus optimizer and EMA state.

    Fresh-init rather than loading checkpoint weights: this is a TRAINING
    profile, so the evaluation EMA weights are the wrong boundary. Model and
    optimizer master parameters stay fp32 — the objective runs under bf16
    autocast at the call site, which is the #154 precision semantics; casting
    the whole model to bf16 would be a different configuration.
    """
    import numpy as np
    import torch
    from unturtle_elf.loader import build_elf_model
    from unturtle_elf.training import build_muon_optimizer, init_ema

    raw_config = fixture["raw_config"]
    config = build_config(raw_config)

    # Same global RNG state before every trial's construction, so weights and
    # dropout draws are the same stream in the OFF and ON passes.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = build_elf_model(raw_config).to(args.device)  # fp32 master params
    optimizer = build_muon_optimizer(model, lr=LEARNING_RATE)
    ema = init_ema(model)

    # A fixed row-index schedule, replayed identically by every trial. RAGGED
    # rows are handed over deliberately: truncation, padding and stacking are
    # per-step collator work and belong inside the timed `data_collation`
    # scope, not in this untimed setup.
    table = fixture["table"]
    rng = np.random.default_rng(0)
    row_order = rng.permutation(table.num_rows)
    indices = [int(row_order[i % table.num_rows]) for i in range(batch_size)]
    cpu_rows = [
        np.asarray(table["input_ids"][index].as_py(), dtype=np.int64)
        for index in indices
    ]

    # Persistent across the trial's steps, advancing as the oracle's does — a
    # fresh seed-0 generator per step would not be the #154 RNG stream.
    generator = torch.Generator(device="cpu").manual_seed(0)
    return model, fixture["encoder"], optimizer, cpu_rows, config, ema, generator


def build_config(raw_config: dict[str, Any]):
    """The frozen training config: oracle defaults, then checkpoint values.

    Defaults are copied explicitly so a key the checkpoint omits cannot become
    a silent default (the Stage-3 correction `stage3_reduced_gate` records).
    """

    class Config:
        pass

    config = Config()
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
        "ema_decay1": EMA_DECAY,
    }.items():
        setattr(config, key, value)
    for key, value in raw_config.items():  # checkpoint values win
        setattr(config, key, value)
    return config


def collate(cpu_rows: list[Any], *, device: str) -> dict[str, Any]:
    """Truncate, pad, stack, build the oracle masks, and transfer — every step.

    All of it is charged to `data_collation`. An earlier version pre-padded and
    pre-stacked in `build()` and timed only the mask construction and copy,
    which understated the collator's real cost.

    Masks follow `stage3_reduced_gate.make_batch`: derived from the TRUE
    lengths, never all-ones, which would train on padding.
    """
    import numpy as np
    import torch
    from unturtle_elf._reference.encoder_utils import build_self_attn_cond_masks

    padded, lengths = [], []
    for ids in cpu_rows:
        true_len = min(len(ids), SEQUENCE_LENGTH)
        ids = ids[:SEQUENCE_LENGTH]
        if true_len < SEQUENCE_LENGTH:
            ids = np.concatenate(
                [ids, np.zeros(SEQUENCE_LENGTH - true_len, dtype=ids.dtype)]
            )
        padded.append(ids)
        lengths.append(true_len)
    stacked = np.stack(padded)
    batch_size = stacked.shape[0]
    positions = np.arange(SEQUENCE_LENGTH)[None, :]
    is_cond = positions < np.zeros((batch_size, 1), dtype=np.int32)
    is_valid = positions < np.asarray(lengths, dtype=np.int32)[:, None]
    encoder_attn, attn, cond = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
    return {
        "input_ids": torch.from_numpy(stacked).long().to(device),
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
    # Loaded once, outside every timed scope: download and materialization are
    # not part of the step being profiled.
    fixture = load_fixture(args)
    records: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        try:
            record = profile_batch(args, fixture, batch_size)
        except OomInPhase as error:
            record = {
                "family": "elf",
                "cell": "training_step",
                "batch_size": batch_size,
                "status": "oom",
                "reason": str(error.cause)[:400],
                # A typed OOM is a RESULT, so it carries the same context a
                # successful cell would (#152).
                "oom_phase": error.phase,
                "sequence_length": SEQUENCE_LENGTH,
                "precision": ("fp32 master params; bf16 autocast over the objective"),
                "hardware": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "cpu"
                ),
                "checkpoint": f"{CHECKPOINT}@{CHECKPOINT_REVISION}",
                "dataset": f"{DATASET}@{DATASET_REVISION}",
                "encoder": (f"{fixture['encoder_name']}@{fixture['encoder_revision']}"),
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


def profile_batch(
    args: argparse.Namespace, fixture: dict[str, Any], batch_size: int
) -> dict[str, Any]:
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
    on_trials: list[float] = []
    warmup_trials: list[float] = []
    peak_allocated: list[int] = []
    peak_reserved: list[int] = []
    released: list[bool] = []
    per_trial_ops: list[dict[str, dict[str, Any]]] = []

    def measure(arm: str) -> None:
        """One arm of one trial: build fresh, measure, release.

        Peak memory is taken from the OFF arm only, so the instrumentation's own
        allocations never enter the capacity figure.
        """
        with oom_phase("build"):
            model, encoder, optimizer, cpu_rows, config, ema, generator = build(
                args, fixture, batch_size
            )
        model_probe = weakref.ref(model)
        timer = OperationTimer(device=args.device) if arm == "on" else None
        original = encoder_original = None
        if timer is not None:
            original = instrument(model, timer)
            encoder_original = instrument_encoder(encoder, timer)
        if arm == "off" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def run(
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            cpu_rows=cpu_rows,
            config=config,
            ema=ema,
            generator=generator,
            timer=timer,
        ):
            return one_step(
                model,
                encoder,
                optimizer,
                cpu_rows,
                config,
                args,
                ema,
                generator,
                timer=timer,
            )

        try:
            warmup_seconds, timed = timed_step_loop(
                run, device=args.device, timer=timer
            )
        finally:
            if original is not None:
                model.__class__.__call__ = original
            if encoder_original is not None:
                encoder.__class__.forward = encoder_original

        if arm == "off":
            off_trials.append(sum(timed) / len(timed))
            warmup_trials.append(warmup_seconds)
            if torch.cuda.is_available():
                peak_allocated.append(torch.cuda.max_memory_allocated())
                peak_reserved.append(torch.cuda.max_memory_reserved())
        else:
            on_trials.append(sum(timed) / len(timed))
            per_trial_ops.append(timer.result())

        # `run` binds the model as a DEFAULT ARGUMENT, so it must go first or
        # the binding keeps this trial's weights resident through the next
        # build. The encoder is a SHARED fixture object and is not expected to
        # be released here.
        del run, model, optimizer, cpu_rows, config, ema, generator, timer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        released.append(model_probe() is None)

    # Arms are INTERLEAVED with alternating order, so thermal or clock drift
    # does not land entirely on whichever arm always runs second — the frozen
    # protocol's rule, and the instrumentation overhead here is ~1% of the
    # step, the same order as such drift.
    for trial in range(TRIALS):
        order = ("off", "on") if trial % 2 == 0 else ("on", "off")
        for arm in order:
            with oom_phase("timed"):
                measure(arm)

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
        # EXACT equality, not just "nothing missing": an unexpected event would
        # otherwise be ignored at publication time, hiding a taxonomy change or
        # a mis-keyed call site.
        if set(ops) != required:
            return {
                "family": "elf",
                "cell": "training_step",
                "batch_size": batch_size,
                "status": "measurement_invalid",
                "reason": (
                    f"instrumented trial {index} observed events "
                    f"{sorted(ops)}, expected exactly {sorted(required)}: "
                    f"missing {sorted(required - set(ops))}, unexpected "
                    f"{sorted(set(ops) - required)}"
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
            call_count=1,
            parent=None,
            coverage_eligible=True,
        )
        for name in ("data_collation", "backward", "optimizer_step", *contained)
    ]
    # `objective_loss` keeps its frozen taxonomy name. Its INCLUSIVE total is
    # the diagnostic; coverage uses the EXCLUSIVE share, so the objective's own
    # work counts once and the contained forwards are not double counted.
    events.append(
        OperationEvent(
            name="objective_loss",
            inclusive_seconds=per_step("objective_loss"),
            exclusive_seconds=objective_exclusive,
            call_count=1,
            parent=None,
            coverage_eligible=True,
        )
    )

    cell = ProfileCell(
        family="elf",
        cell="training_step",
        batch_size=batch_size,
        sequence_length=SEQUENCE_LENGTH,
        dtype="bf16_autocast",
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
            "master_parameter_dtype": "fp32",
            "autocast_dtype": "bfloat16",
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


def one_step(
    model, encoder, optimizer, cpu_rows, config, args, ema, generator, timer=None
):
    """One optimizer update with the frozen #154 mechanics.

    Sequence matches `stage3_reduced_gate` lines 460-475: bf16 autocast over the
    objective with fp32 master params, backward, grad clip 1.0, Muon step, EMA
    0.9999, zero-grad.

    Grad clipping and the EMA update are NOT given their own taxonomy events, so
    their cost lands in `unattributed_seconds` rather than being folded silently
    into `optimizer_step`. Coverage below 100% is the honest outcome; inflating
    it by mixing distinct work into a Muon event would not be.
    """
    import torch
    from unturtle_elf.training import elf_training_loss, ema_update

    scope = timer.measure if timer is not None else _null_scope

    with scope("data_collation"):
        batch = collate(cpu_rows, device=args.device)

    with (
        scope("objective_loss"),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        loss, _metrics, _aux = elf_training_loss(
            model,
            encoder,
            batch,
            config,
            dropout_generator=generator,
        )
    with scope("backward"):
        loss.backward()
    # grad_accum_steps = 1: one microbatch per optimizer update (#166 cell).
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
    )
    with scope("optimizer_step"):
        optimizer.step()
    ema_update(ema, model, EMA_DECAY)
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
    with oom_phase("warmup"):
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
