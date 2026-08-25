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
import weakref
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


def interleaved_trials(
    measure_arm, labels, *, trials: int, warmup: int, steps: int, device: str
) -> dict[str, list[dict[str, Any]]]:
    """Replicated, order-alternating trials over a ONE-SHOT measure function.

    ``measure_arm(label, warmup=..., steps=..., device=...)`` must build its
    model and trainer, measure, drop every reference and collect, all before
    returning. It returns only the measurement dict.

    An earlier design passed ``(run, teardown)`` closure pairs, which did NOT
    work: both closures captured ``model`` and ``trainer``, so ``teardown``
    dropped local aliases while the ``run`` closure and the default arguments
    kept the weights resident — verified with a weakref, the model outlived
    teardown AND was still alive while the next arm was being constructed. The
    cross-arm memory contamination this gate exists to avoid would then recur
    through the closure instead of through the missing collect.

    A one-shot function makes the lifetime a property of the call: nothing the
    caller holds can reference the previous arm's model.

    Order alternates per trial so thermal drift does not land entirely on
    whichever arm runs second.
    """
    results: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    for trial in range(trials):
        order = list(labels) if trial % 2 == 0 else list(reversed(labels))
        for label in order:
            measurement = measure_arm(label, warmup=warmup, steps=steps, device=device)
            measurement["trial"] = trial
            measurement["ran_first"] = label == order[0]
            results[label].append(measurement)
    return results


def sign_consistency(deltas: list[float], *, expect_negative: bool) -> dict[str, Any]:
    """How many trials agreed with the expected direction, and by how much.

    Sign alone is not "rough magnitude": a -0.1% delta is noise. The median
    effect must also clear the spread across trials, which keeps the check
    hardware-independent without pinning it to the ledger's exact percentages.
    """
    if not deltas:
        raise ValueError("no per-trial deltas; nothing to check consistency over")
    agree = sum(1 for d in deltas if (d < 0) == expect_negative)
    median = statistics.median(deltas)
    spread = max(deltas) - min(deltas)
    return {
        "per_trial": deltas,
        "median": median,
        "spread": spread,
        "trials_agreeing": agree,
        "trials": len(deltas),
        "majority_agrees": agree * 2 > len(deltas),
        # The effect must be larger than the run-to-run drift it is measured
        # against, otherwise the sign is not evidence of anything.
        "exceeds_spread": abs(median) > spread,
    }


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
    """Reproduce the SHAPE of the sparse finding under replicated trials.

    Three axes are asserted independently — low-mask step time, low-mask
    activations, high-mask activations — because a harness that got only the
    step sign right would still be missing the regime that made the flag
    opt-in. Memory is CUDA-only and its absence is a failure, not a pass.
    """
    import argparse as _argparse

    import torch

    if not (args.device.startswith("cuda") and torch.cuda.is_available()):
        return {
            "check": "sparse",
            "status": "unsupported",
            "reason": (
                "the sparse gate asserts on activation memory, which needs "
                "CUDA; a CPU run cannot exercise the axis that made this flag "
                "opt-in"
            ),
            "expectation": EXPECTATIONS["sparse"],
        }

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

        def measure_arm(
            label, *, warmup, steps, device, bench_args=bench_args, batch=batch
        ):
            """Build, measure, release — all inside one call.

            Everything the measurement touches is local to this frame, so after
            the return there is no closure, default argument or caller-held
            binding that can keep the previous arm's weights resident. That
            lifetime property is what the gate depends on, and it is asserted
            by `tests/eval/test_profile_harness.py` via a weakref.
            """
            model = bench._model(bench_args)
            trainer = bench._trainer(bench_args, model, label == "sparse")
            probe = weakref.ref(model)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            # Weights/grads/batch are resident and identical across arms;
            # subtracting them leaves the transient set the flag changes.
            baseline = torch.cuda.memory_allocated()

            def step(model=model, trainer=trainer):
                # Bound as default arguments rather than captured, so the only
                # references live in this function object and both die with it.
                loss = trainer.compute_loss(model, dict(batch))
                loss.backward()
                model.zero_grad(set_to_none=True)

            seconds = timed_steps(step, warmup=warmup, steps=steps, device=device)
            peak = torch.cuda.max_memory_allocated()

            # Drop every reference created in this frame before returning:
            # `step` holds the model and trainer in its defaults, so it goes
            # first, then the local names.
            del step
            trainer = None
            model = None
            # Load-bearing: the trainer and model form a reference cycle, so
            # dropping names is not enough. sparse_lm_head_training.py:317-323
            # records that skipping this "produced a sign flip between runs",
            # and an earlier version of this gate reproduced that corruption.
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "median_seconds": statistics.median(seconds),
                "peak_allocated_bytes": peak,
                "activation_bytes": peak - baseline,
                "model_released": probe() is None,
            }

        per_arm = interleaved_trials(
            measure_arm,
            ("dense", "sparse"),
            trials=args.trials,
            warmup=args.warmup,
            steps=args.steps,
            device=args.device,
        )
        step_deltas = []
        activation_deltas = []
        for dense, sparse_arm in zip(per_arm["dense"], per_arm["sparse"], strict=True):
            if not dense["median_seconds"] or not dense["activation_bytes"]:
                raise ValueError(
                    "zero dense baseline: a delta cannot be formed against it "
                    f"(step={dense['median_seconds']}, "
                    f"activation={dense['activation_bytes']})"
                )
            step_deltas.append(
                (sparse_arm["median_seconds"] - dense["median_seconds"])
                / dense["median_seconds"]
            )
            activation_deltas.append(
                (sparse_arm["activation_bytes"] - dense["activation_bytes"])
                / dense["activation_bytes"]
            )
        cells[f"mask_{mask_ratio}"] = {
            "per_arm": per_arm,
            "step_time": sign_consistency(step_deltas, expect_negative=True),
            "activation": sign_consistency(
                activation_deltas, expect_negative=(mask_ratio < 0.4)
            ),
        }

    low, high = cells["mask_0.15"], cells["mask_0.75"]
    axes = {
        "low_mask_step_time_win": low["step_time"],
        "low_mask_activation_win": low["activation"],
        "high_mask_activation_penalty": high["activation"],
    }
    for name, axis in axes.items():
        if any(
            value is None or value != value  # NaN
            for value in (axis["median"], axis["spread"])
        ):
            return {
                "check": "sparse",
                "status": "measurement_invalid",
                "reason": f"axis {name} produced a non-finite statistic",
                "expectation": EXPECTATIONS["sparse"],
                "cells": cells,
            }
    reproduced = all(
        axis["majority_agrees"] and axis["exceeds_spread"] for axis in axes.values()
    )
    return {
        "check": "sparse",
        "status": "reproduced" if reproduced else "NOT_REPRODUCED",
        "expectation": EXPECTATIONS["sparse"],
        "trials": args.trials,
        "axes": {
            name: {
                "median": axis["median"],
                "spread": axis["spread"],
                "trials_agreeing": f"{axis['trials_agreeing']}/{axis['trials']}",
                "majority_agrees": axis["majority_agrees"],
                "exceeds_spread": axis["exceeds_spread"],
            }
            for name, axis in axes.items()
        },
        "cells": cells,
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
