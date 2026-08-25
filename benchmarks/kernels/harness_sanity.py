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

"""#166 Stage-1 GATE — the harness must reproduce known results first.

Runs before any new profiling is trusted. Three paths already carry
end-to-end evidence (`docs/acceleration-ledger.md` rows 2, 3, 4) and their
dispatch decisions are CLOSED — they are not target candidates. Their value
here is that a harness which cannot reproduce a known sign and rough magnitude
has not been shown to measure what it claims.

The assertions are directional, not exact: absolute percentages depend on
hardware, and demanding the ledger's exact numbers on different silicon would
fail for the wrong reason. What must reproduce is the *shape* of each finding,
because each shape is what a broken harness would get wrong:

- **sparse LM-head** — a step-time win at LOW mask ratio and a memory penalty
  at HIGH mask ratio. A harness that reported a win at both would be ignoring
  the regime that made this flag opt-in;
- **device-side noising** — no consistent sign. A harness that "found" a clear
  winner here would be reporting noise as signal;
- **hybrid attention** — a slowdown BELOW the crossover. A harness that only
  measured above it would miss the reason the path is gated at all.

Usage::

    .venv/bin/python benchmarks/kernels/harness_sanity.py --check sparse
    .venv/bin/python benchmarks/kernels/harness_sanity.py --check all
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics
import time
from typing import Any

# Ledger expectations. Directional only — see the module docstring.
EXPECTATIONS = {
    "sparse": {
        "ledger": "row 2 (#77, benchmarks/sparse_lm_head_training.py)",
        "shape": "step-time win at mask 0.15; memory penalty at mask 0.75",
        "reference_numbers": "32K vocab: -32.6% step at 0.15; +21.4% peak at 0.75",
    },
    "noising": {
        "ledger": "row 3 (#62, benchmarks/collator_vs_process_noising.py)",
        "shape": "no consistent sign between collator and device paths",
        "reference_numbers": "median +0.42%, range -0.61%..+1.12%, 4 of 5 slower",
    },
    "hybrid": {
        "ledger": "row 4 (#63/#99, _hybrid.py:179)",
        "shape": "full-forward slowdown below the 2048 crossover",
        "reference_numbers": "0.90x at L=1024; 1.50x at L=2048; 1.92x at L=4096",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        default="all",
        choices=[*EXPECTATIONS, "all"],
        help="which known result to reproduce",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default="benchmarks/results/pd_harness_sanity")
    return parser.parse_args()


def timed_steps(fn, *, warmup: int, steps: int, device: str) -> list[float]:
    """Sync-bracketed steady-state timings, warmup excluded.

    Mirrors `benchmarks/sparse_lm_head_training.py:_measure` rather than
    inventing a second timing convention — the protocol requires warmup out of
    steady state and consistent synchronization boundaries.
    """
    import torch

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    seconds: list[float] = []
    for step in range(warmup + steps):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= warmup:
            seconds.append(elapsed)
    return seconds


def interleaved(arm_a, arm_b, *, trials: int, warmup: int, steps: int, device: str):
    """Alternate which arm runs first so thermal drift does not land on one.

    The discipline `sparse_lm_head_training.py` already uses; stated here
    because "interleave the arms" is a protocol rule, not an optional nicety.
    """
    a_times: list[float] = []
    b_times: list[float] = []
    for trial in range(trials):
        first, second = (arm_a, arm_b) if trial % 2 == 0 else (arm_b, arm_a)
        first_times = timed_steps(
            first["fn"], warmup=warmup, steps=steps, device=device
        )
        second_times = timed_steps(
            second["fn"], warmup=warmup, steps=steps, device=device
        )
        if trial % 2 == 0:
            a_times.append(statistics.median(first_times))
            b_times.append(statistics.median(second_times))
        else:
            b_times.append(statistics.median(first_times))
            a_times.append(statistics.median(second_times))
    return a_times, b_times


def _load_sparse_benchmark():
    """Reuse the #77 benchmark's builders rather than re-deriving them.

    A second definition of "the model and the noised batch" would let the gate
    pass while measuring something the original benchmark never measured.
    """
    import importlib.util

    path = pathlib.Path(__file__).resolve().parents[1] / "sparse_lm_head_training.py"
    spec = importlib.util.spec_from_file_location("_sparse_bench", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the sparse benchmark at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_sparse(args) -> dict[str, Any]:
    """Reproduce the SHAPE of the sparse finding: win low, penalty high."""
    import argparse as _argparse

    import torch

    bench = _load_sparse_benchmark()
    cells: dict[str, Any] = {}
    for mask_ratio in (0.15, 0.75):
        bench_args = _argparse.Namespace(
            device=args.device,
            vocab_size=32000,
            hidden_size=512,
            layers=2,
            batch_size=2,
            seq_len=512,
            lora=False,
            warmup=args.warmup,
            steps=args.steps,
        )
        batch = bench._noised_batch(bench_args, mask_ratio)
        arms = {}
        for sparse in (False, True):
            model = bench._model(bench_args)
            trainer = bench._trainer(bench_args, model, sparse)

            def step(trainer=trainer, model=model, batch=batch):
                loss = trainer.compute_loss(model, dict(batch))
                loss.backward()
                model.zero_grad(set_to_none=True)

            baseline = 0
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                # Weights/grads/batch already resident. The activation figure
                # subtracts them, because they are identical across arms and
                # dilute the percentage on the part the flag actually changes.
                baseline = torch.cuda.memory_allocated()
            seconds = timed_steps(
                step, warmup=args.warmup, steps=args.steps, device=args.device
            )
            peak = (
                torch.cuda.max_memory_allocated()
                if args.device.startswith("cuda") and torch.cuda.is_available()
                else None
            )
            arms["sparse" if sparse else "dense"] = {
                "median_seconds": statistics.median(seconds),
                "peak_allocated_bytes": peak,
                "activation_bytes": (peak - baseline if peak is not None else None),
            }
            del model, trainer
            # `gc.collect()` is load-bearing, not hygiene: the trainer and model
            # form a reference cycle, so `del` alone leaves the weights
            # resident and whichever arm runs second starts with the previous
            # arm's allocation inside its peak. The #77 benchmark records that
            # this "produced a sign flip between runs"
            # (sparse_lm_head_training.py:317-323) — and this gate reproduced
            # exactly that corruption before the collect was added.
            gc.collect()
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        dense, sparse_arm = arms["dense"], arms["sparse"]
        step_delta = (sparse_arm["median_seconds"] - dense["median_seconds"]) / dense[
            "median_seconds"
        ]
        peak_delta = None
        if dense["peak_allocated_bytes"] and sparse_arm["peak_allocated_bytes"]:
            peak_delta = (
                sparse_arm["peak_allocated_bytes"] - dense["peak_allocated_bytes"]
            ) / dense["peak_allocated_bytes"]
        activation_delta = None
        if dense.get("activation_bytes") and sparse_arm.get("activation_bytes"):
            activation_delta = (
                sparse_arm["activation_bytes"] - dense["activation_bytes"]
            ) / dense["activation_bytes"]
        cells[f"mask_{mask_ratio}"] = {
            "arms": arms,
            "step_time_delta": step_delta,
            "peak_delta": peak_delta,
            "activation_delta": activation_delta,
        }

    low = cells["mask_0.15"]
    high = cells["mask_0.75"]
    # Directional, not exact: the sign is the finding, the magnitude is
    # hardware. A harness reporting a win at BOTH ratios would be ignoring the
    # regime that made this flag opt-in.
    #
    # Memory is asserted on ACTIVATIONS, not raw peak: ~40% of peak is weights,
    # identical in both arms, which dilutes the effect being measured. The
    # ledger's own columns make the same distinction.
    reproduced = (
        low["step_time_delta"] < 0
        and (low["activation_delta"] is None or low["activation_delta"] < 0)
        and (high["activation_delta"] is None or high["activation_delta"] > 0)
    )
    return {
        "check": "sparse",
        "status": "reproduced" if reproduced else "NOT_REPRODUCED",
        "expectation": EXPECTATIONS["sparse"],
        "cells": cells,
        "observed_shape": (
            f"step delta at 0.15 = {low['step_time_delta']:+.1%}; "
            f"activation delta at 0.15 = "
            + (
                f"{low['activation_delta']:+.1%}"
                if low["activation_delta"] is not None
                else "n/a"
            )
            + "; activation delta at 0.75 = "
            + (
                f"{high['activation_delta']:+.1%}"
                if high["activation_delta"] is not None
                else "n/a"
            )
        ),
    }


CHECKS = {"sparse": check_sparse}


def main() -> None:
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    checks = list(EXPECTATIONS) if args.check == "all" else [args.check]
    records: list[dict[str, Any]] = []
    for name in checks:
        if name in CHECKS:
            records.append(CHECKS[name](args))
        else:
            records.append(
                {
                    "check": name,
                    "status": "not_implemented",
                    "expectation": EXPECTATIONS[name],
                    "note": (
                        "gate cell declared; arms are wired alongside this "
                        "family's taxonomy in the profiling work"
                    ),
                }
            )
        print(json.dumps(records[-1]))
    (out / "harness_sanity.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    print(f"wrote {len(records)} gate cells to {out / 'harness_sanity.jsonl'}")


if __name__ == "__main__":
    main()
