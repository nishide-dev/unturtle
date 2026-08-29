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

"""#166 Stage-2 adoption gate: paired outer-wall latency and memory.

THIS IS THE MEASUREMENT THAT DECIDES ADOPTION. Stage-2 selection predicted 8-12%
end-to-end from Amdahl on the measured shares; that prediction is not evidence.
If the gain does not survive here, the specialization is not adopted.

Both arms call the SAME public `run_fmlm_request`. The only difference is
whether the scope guard is allowed to reach the fast path — the OFF arm forces
the reference sequence by monkeypatching `fast_path_applies` to return False,
which is benchmark-local and restored in `finally`.

Discipline inherited from the #166 Stage-1 producers:

- The verdict is the OUTER WALL, measured with the window-closing synchronize
  INSIDE the timed span; generation is async and a clock stopped before the
  queue drains reports a shorter run than happened.
- Arms are PAIRED and INTERLEAVED with the order reversed each trial, so drift
  does not load onto whichever arm always runs second.
- Overhead is the median of per-trial paired deltas, never a difference of
  medians: the two medians can come from different trials.
- At TRIALS=3 no significance test is performed. The speedup is reported with
  its per-trial range and the OFF spread, and `resolution_status` says the
  window cannot estimate a noise floor.
- Peak memory is reset per trial and recorded as an array plus the max.
- The device must be exclusive; a co-resident workload makes latency and peak
  attributable to someone else's job.

Usage:
    .venv/bin/python benchmarks/flm/state_update_outer_wall.py --device cuda:3 \
        --out docs/artifacts
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any

TRIALS = 3
WARMUP = 2

STEPS = 32
GAMMA = 1.0
SEED = 100
MAX_LENGTH = 1024
BATCH_SIZES = (1, 8, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


class Request:
    def __init__(self, batch: int):
        self.kwargs = {
            "steps": STEPS,
            "num_samples": batch,
            "seed": SEED,
            "gamma": GAMMA,
        }


@contextlib.contextmanager
def fast_path_disabled():
    """Force the reference sequence for the OFF arm.

    Benchmark-local: patches the module attribute the sampler reads, and
    restores it unconditionally. The sampler itself is unmodified, so both arms
    execute the same public entry point and differ only in which branch the
    guard selects.
    """
    from unturtle_flm import state_update

    original = state_update.fast_path_applies
    state_update.fast_path_applies = lambda *args, **kwargs: False
    try:
        yield
    finally:
        state_update.fast_path_applies = original


def _one_trial(model, request, torch, arm: str) -> dict[str, Any]:
    from unturtle_flm.sampler import run_fmlm_request

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if arm == "off":
        with fast_path_disabled():
            begin = time.perf_counter()
            run_fmlm_request(model, request)
            # The drain is INSIDE the timed span.
            torch.cuda.synchronize()
            wall = time.perf_counter() - begin
    else:
        begin = time.perf_counter()
        run_fmlm_request(model, request)
        torch.cuda.synchronize()
        wall = time.perf_counter() - begin
    return {
        "wall_seconds": wall,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def paired_trials(model, batch: int) -> list[dict[str, Any]]:
    import torch

    request = Request(batch)
    for _ in range(WARMUP):
        with fast_path_disabled():
            _one_trial(model, request, torch, "off")
        _one_trial(model, request, torch, "on")
    torch.cuda.synchronize()

    trials = []
    for index in range(TRIALS):
        order = ("off", "on") if index % 2 == 0 else ("on", "off")
        measured = {arm: _one_trial(model, request, torch, arm) for arm in order}
        trials.append(
            {
                "trial": index,
                "order": list(order),
                "reference_wall_seconds": measured["off"]["wall_seconds"],
                "specialized_wall_seconds": measured["on"]["wall_seconds"],
                # Both arms ran adjacently under the same conditions.
                "paired_delta_seconds": (
                    measured["off"]["wall_seconds"] - measured["on"]["wall_seconds"]
                ),
                "reference_peak_allocated_bytes": measured["off"][
                    "peak_allocated_bytes"
                ],
                "specialized_peak_allocated_bytes": measured["on"][
                    "peak_allocated_bytes"
                ],
                "reference_peak_reserved_bytes": measured["off"]["peak_reserved_bytes"],
                "specialized_peak_reserved_bytes": measured["on"][
                    "peak_reserved_bytes"
                ],
            }
        )
    return trials


def summarize(trials: list[dict]) -> dict[str, Any]:
    reference = [t["reference_wall_seconds"] for t in trials]
    specialized = [t["specialized_wall_seconds"] for t in trials]
    deltas = [t["paired_delta_seconds"] for t in trials]
    per_trial_speedup = [
        t["reference_wall_seconds"] / t["specialized_wall_seconds"] for t in trials
    ]
    median_delta = statistics.median(deltas)
    reference_spread = max(reference) - min(reference)
    reference_median = statistics.median(reference)
    return {
        "reference_wall_trials": reference,
        "specialized_wall_trials": specialized,
        "reference_wall_median": reference_median,
        "specialized_wall_median": statistics.median(specialized),
        "paired_delta_trials": deltas,
        "median_paired_delta_seconds": median_delta,
        # Median of per-trial ratios, not a ratio of medians.
        "median_paired_speedup": statistics.median(per_trial_speedup),
        "per_trial_speedup": per_trial_speedup,
        "median_relative_improvement": (
            median_delta / reference_median if reference_median else None
        ),
        "reference_trial_spread_seconds": reference_spread,
        "direction_consistent": bool(deltas)
        and (all(d > 0 for d in deltas) or all(d < 0 for d in deltas)),
        # No significance test: three trials cannot estimate a noise floor.
        "resolvable": None,
        "resolution_status": "not_assessed",
        "resolution_reason": (
            "the frozen three-trial window is insufficient to estimate the "
            "noise floor; the delta is reported with its per-trial range and "
            "the reference spread so a reader can judge it directly"
        ),
    }


def memory_summary(trials: list[dict]) -> dict[str, Any]:
    ref_alloc = [t["reference_peak_allocated_bytes"] for t in trials]
    new_alloc = [t["specialized_peak_allocated_bytes"] for t in trials]
    ref_res = [t["reference_peak_reserved_bytes"] for t in trials]
    new_res = [t["specialized_peak_reserved_bytes"] for t in trials]
    return {
        "reference_allocated_trials": ref_alloc,
        "specialized_allocated_trials": new_alloc,
        "reference_max_allocated_bytes": max(ref_alloc),
        "specialized_max_allocated_bytes": max(new_alloc),
        "reference_max_reserved_bytes": max(ref_res),
        "specialized_max_reserved_bytes": max(new_res),
        "allocated_delta_bytes": max(ref_alloc) - max(new_alloc),
        # A regression here fails the gate even if latency improves.
        "regression": max(new_alloc) > max(ref_alloc),
    }


def device_occupancy(device: str) -> dict[str, Any]:
    import torch

    index = int(device.split(":")[1])
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)

    def query(fields: str) -> list[str]:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in result.stdout.strip().splitlines() if line.strip()]

    uuid = None
    for row in query("gpu=index,uuid"):
        parts = [field.strip() for field in row.split(",")]
        if parts and parts[0] == str(index):
            uuid = parts[1]
            break
    processes = []
    for row in query("compute-apps=pid,used_memory,gpu_uuid"):
        pid, used, row_uuid = (field.strip() for field in row.split(","))
        if row_uuid == uuid:
            processes.append({"pid": int(pid), "used_mib": int(used)})
    return {
        "device": device,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "compute_processes": processes,
        "foreign_process_count": len([p for p in processes if p["pid"] != os.getpid()]),
    }


def provenance(command: str, occupancy: dict[str, Any]) -> dict[str, Any]:
    import torch
    from unturtle_flm.loader import FMLM_CHECKPOINT, FMLM_REVISION

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:  # pragma: no cover
            return None

    head, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain")
    if head is None or dirty is None:
        raise SystemExit("cannot establish provenance; refusing to write")
    return {
        "head_sha": head,
        "worktree_clean": dirty == "",
        "command": command,
        "purpose": (
            "adoption gate for the #166 Stage-2 specialization: paired "
            "outer-wall latency and peak memory through the PUBLIC entry point"
        ),
        "records_end_to_end_latency": True,
        "arms": {
            "reference": (
                "public run_fmlm_request with the fast-path guard forced False"
            ),
            "specialized": "public run_fmlm_request, guard unmodified",
            "note": (
                "both arms call the same entry point; only the guard's answer "
                "differs, so the comparison isolates the specialization"
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "device_index": torch.cuda.current_device(),
        },
        "device_occupancy_at_start": occupancy,
        "fixture": {
            "checkpoint": f"{FMLM_CHECKPOINT}@{FMLM_REVISION}",
            "steps": STEPS,
            "gamma": GAMMA,
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "batch_sizes": list(BATCH_SIZES),
        },
        "frozen_constants": {"TRIALS": TRIALS, "WARMUP": WARMUP},
        "predicted_before_measurement": {
            "batch_1": 0.085,
            "batch_8": 0.125,
            "batch_32": 0.122,
            "basis": (
                "Amdahl on the Stage-1 measured shares and the local "
                "microbenchmark; recorded here so the prediction can be "
                "compared against what was actually measured"
            ),
        },
    }


def main() -> None:
    import torch

    args = parse_args()
    index = int(args.device.split(":")[1])
    torch.cuda.set_device(index)
    occupancy = device_occupancy(args.device)
    if occupancy["foreign_process_count"]:
        raise SystemExit(
            f"{args.device} is shared: {occupancy['compute_processes']}. "
            "Latency and peak memory would be attributable to another workload."
        )
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from unturtle_flm.loader import load_fmlm_model

    model = load_fmlm_model(device=args.device).eval()

    cells = []
    for batch in BATCH_SIZES:
        trials = paired_trials(model, batch)
        latency = summarize(trials)
        memory = memory_summary(trials)
        cells.append(
            {
                "batch": batch,
                "steps": STEPS,
                "gamma": GAMMA,
                "device_occupancy_before": device_occupancy(args.device),
                "trials": trials,
                "latency": latency,
                "memory": memory,
            }
        )
        print(
            f"[cell] batch={batch:3d} "
            f"reference={latency['reference_wall_median']:.4f}s "
            f"specialized={latency['specialized_wall_median']:.4f}s "
            f"improvement={latency['median_relative_improvement'] * 100:+.1f}% "
            f"speedup={latency['median_paired_speedup']:.3f}x "
            f"mem_regression={memory['regression']}"
        )
        torch.cuda.empty_cache()

    payload = {
        "run": provenance(" ".join(sys.argv), occupancy),
        "cells": cells,
        "summary": {
            "any_memory_regression": any(c["memory"]["regression"] for c in cells),
            "median_relative_improvement": {
                str(c["batch"]): c["latency"]["median_relative_improvement"]
                for c in cells
            },
        },
    }
    target = out / "166-fmlm-state-update-outer-wall.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(cells)} cells to {target}")


if __name__ == "__main__":
    main()
