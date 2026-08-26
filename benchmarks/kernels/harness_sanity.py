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

#: Equivalence band for the noising gate. The established finding is the
#: ABSENCE of an effect, which needs an equivalence test, not a dispersion
#: test: `+50%, -50%, +50%` has a median inside its own spread with mixed
#: signs, so a dispersion criterion would pass measurement chaos as agreement.
#: The original run spanned -0.61%..+1.12%, so 2% clears known noise while
#: still being tight enough to mean "not material".
NOISING_EQUIVALENCE_MARGIN = 0.02

#: Trials for the noising gate. More than the other cells because an
#: equivalence claim over a near-zero effect needs the replication.
NOISING_TRIALS = 5

# Ledger expectations. Directional only — see the module docstring.
EXPECTATIONS = {
    "sparse": {
        "ledger": "row 2 (#77, benchmarks/sparse_lm_head_training.py)",
        "shape": "step-time win at mask 0.15; memory penalty at mask 0.75",
        "reference_numbers": "32K vocab: -32.6% step at 0.15; +21.4% peak at 0.75",
    },
    "noising": {
        "ledger": "row 3 (#62, benchmarks/collator_vs_process_noising.py)",
        "shape": "NO measurable difference between collator and device paths",
        "reference_numbers": "median +0.42%, range -0.61%..+1.12%, 4 of 5 slower",
        # The established finding is the ABSENCE of an effect, so the gate
        # criterion must be an absence test — not a reproduction of the
        # historical noise pattern. "4 of 5 trials slower" was one sample of a
        # sign that does not hold; requiring it would demand that the harness
        # reproduce noise, and would fail on a correct run that happened to
        # land 3 of 5.
        "criterion": (
            "EQUIVALENCE, not dispersion: |median delta| <= "
            "NOISING_EQUIVALENCE_MARGIN and every per-trial delta within the "
            "same margin, over 5 interleaved paired trials. A large deviation "
            "types the cell `unstable / NOT_REPRODUCED`. Reproducing the "
            "historical 4-of-5 direction is explicitly NOT required."
        ),
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
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help=(
            "trials for the effect-shape checks. The noising gate ignores this "
            f"and always uses NOISING_TRIALS={NOISING_TRIALS}: an equivalence "
            "claim over a near-zero effect needs its own replication, and a "
            "CLI default must not be able to weaken it."
        ),
    )
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


def equivalence(deltas: list[float], *, margin: float) -> dict[str, Any]:
    """Is the paired effect small enough to call "no measurable difference"?

    An absence finding cannot be checked with a dispersion test. `+50%, -50%,
    +50%` has its median inside its own spread and mixed signs, so a
    dispersion criterion would accept wild measurement as evidence of
    equivalence. This requires the median AND every individual trial to sit
    inside the margin, so noise fails instead of passing.
    """
    if not deltas:
        raise ValueError("no per-trial deltas; nothing to check equivalence over")
    median = statistics.median(deltas)
    worst = max(abs(delta) for delta in deltas)
    return {
        "per_trial": deltas,
        "median": median,
        "worst_abs": worst,
        "margin": margin,
        "median_within_margin": abs(median) <= margin,
        "all_trials_within_margin": worst <= margin,
        "equivalent": abs(median) <= margin and worst <= margin,
    }


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

    # The release invariant GATES, it does not merely get recorded. If arm
    # lifetimes regress to 12/12 leaked, a shape that happens to still look
    # right must not pass — the same hole as measuring low-mask memory without
    # asserting on it.
    unreleased = [
        f"{mask}/{label}/trial{m['trial']}"
        for mask, cell in cells.items()
        for label, arm in cell["per_arm"].items()
        for m in arm
        if not m.get("model_released")
    ]
    if unreleased:
        return {
            "check": "sparse",
            "status": "measurement_invalid",
            "reason": (
                "arm model(s) outlived their measurement call, so a later arm "
                f"may carry an earlier arm's allocation: {unreleased}"
            ),
            "expectation": EXPECTATIONS["sparse"],
            "cells": cells,
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


def _load_noising_benchmark():
    """Reuse the #62 benchmark's builders rather than re-deriving them."""
    import importlib.util

    path = (
        pathlib.Path(__file__).resolve().parents[1] / "collator_vs_process_noising.py"
    )
    spec = importlib.util.spec_from_file_location("_noising_bench", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load the noising benchmark at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_noising(args) -> dict[str, Any]:
    """Reproduce the ABSENCE of a noising effect — an equivalence test.

    The established finding is "no measurable difference", so the criterion is
    equivalence within a frozen margin, NOT a reproduction of the historical
    "4 of 5 trials slower" pattern. That was one sample of a sign that does not
    hold; requiring it would demand the harness reproduce noise.

    Always uses `NOISING_TRIALS`, never the CLI `--trials`: an equivalence claim
    over a near-zero effect needs its own replication, and a CLI default must
    not be able to weaken it.
    """
    import argparse as _argparse

    import torch

    bench = _load_noising_benchmark()
    bench_args = _argparse.Namespace(
        device=args.device,
        vocab_size=32000,
        hidden_size=512,
        layers=2,
        batch_size=4,
        seq_len=512,
        warmup=args.warmup,
        steps=args.steps,
        trials=NOISING_TRIALS,
    )
    tokenizer = bench._tokenizer()
    features = bench._features(bench_args)

    def measure_arm(label, *, warmup, steps, device):
        """Build, measure, release — one shot, nothing outlives the call."""
        from unturtle.diffusion import (
            DiffusionTrainer,
            DiffusionTrainingArguments,
            MaskedDiffusionDataCollator,
        )

        model = bench._model(bench_args)
        collator = MaskedDiffusionDataCollator(
            tokenizer=tokenizer,
            mask_token_id=tokenizer.mask_token_id,
            # "collator" noises in the collator; "process" defers to the
            # device-side process, which is the path under test.
            noise=(label == "collator"),
        )
        trainer = DiffusionTrainer(
            model=model,
            args=DiffusionTrainingArguments(
                output_dir=str(bench._output_dir()),
                per_device_train_batch_size=bench_args.batch_size,
                max_steps=1,
                use_cpu=(device == "cpu"),
                bf16=False,
                fp16=False,
                remove_unused_columns=False,
                report_to=[],
            ),
            train_dataset=features,
            processing_class=tokenizer,
            data_collator=collator,
        )
        probe = weakref.ref(model)
        seconds = bench._time_path(trainer, model, collator, features, bench_args)
        del trainer, collator
        model = None
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "median_seconds": statistics.median(seconds),
            "model_released": probe() is None,
        }

    per_arm = interleaved_trials(
        measure_arm,
        ("collator", "process"),
        trials=NOISING_TRIALS,
        warmup=args.warmup,
        steps=args.steps,
        device=args.device,
    )
    unreleased = [
        f"{label}/trial{m['trial']}"
        for label, arm in per_arm.items()
        for m in arm
        if not m.get("model_released")
    ]
    if unreleased:
        return {
            "check": "noising",
            "status": "measurement_invalid",
            "reason": f"arm model(s) outlived their measurement call: {unreleased}",
            "expectation": EXPECTATIONS["noising"],
        }

    deltas = []
    for collator_arm, process_arm in zip(
        per_arm["collator"], per_arm["process"], strict=True
    ):
        if not collator_arm["median_seconds"]:
            raise ValueError("zero collator baseline; a delta cannot be formed")
        deltas.append(
            (process_arm["median_seconds"] - collator_arm["median_seconds"])
            / collator_arm["median_seconds"]
        )
    result = equivalence(deltas, margin=NOISING_EQUIVALENCE_MARGIN)
    if result["median_within_margin"] and not result["all_trials_within_margin"]:
        status = "unstable / NOT_REPRODUCED"
    elif result["equivalent"]:
        status = "reproduced"
    else:
        status = "NOT_REPRODUCED"
    return {
        "check": "noising",
        "status": status,
        "expectation": EXPECTATIONS["noising"],
        "trials": NOISING_TRIALS,
        "equivalence": result,
        "per_arm": per_arm,
    }


def check_hybrid(args) -> dict[str, Any]:
    """Reproduce the sub-crossover SLOWDOWN of the hybrid fast path.

    The gated finding is that below `hybrid_fast_min_seq_len` the two-call
    split is a net loss — full forward 0.90x at L=1024 against 1.50x at 2048.
    The slowdown is the load-bearing half: a harness that only measured above
    the crossover would miss the reason the path is gated at all.

    Measured on a FULL model forward, not the attention kernel alone, because
    the extra kernel launch, `cat` and output transpose the split adds are
    exactly what the isolated kernel does not show.
    """
    import torch

    from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    if not (args.device.startswith("cuda") and torch.cuda.is_available()):
        return {
            "check": "hybrid",
            "status": "unsupported",
            "reason": "the crossover is a CUDA kernel-launch effect",
            "expectation": EXPECTATIONS["hybrid"],
        }

    cells: dict[str, Any] = {}
    for seq_len in (1024, 2048):

        def measure_arm(label, *, warmup, steps, device, seq_len=seq_len):
            """One shot: build, measure, release."""
            torch.manual_seed(7)
            model = (
                TinyA2DLlamaLMHeadModel(
                    TinyA2DLlamaConfig(
                        vocab_size=32000,
                        hidden_size=512,
                        intermediate_size=1024,
                        num_hidden_layers=8,
                        num_attention_heads=8,
                        num_key_value_heads=8,
                        max_position_embeddings=seq_len * 2,
                        hybrid_attention=True,
                        # The gate is a declared config field, so the arms differ
                        # only in where the threshold sits: 0 forces the fast path,
                        # a huge value disables it. Correctness is identical either
                        # way — the dense mask is always built.
                        hybrid_fast_min_seq_len=0 if label == "fast" else 10**9,
                    )
                )
                .to(device=device, dtype=torch.bfloat16)
                .eval()
            )
            probe = weakref.ref(model)
            input_ids = torch.randint(1, 32000, (1, seq_len), device=device)
            attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
            # `prompt_lengths` is REQUIRED for the hybrid path to engage at
            # all: `maybe_build_hybrid_mask` returns None when it is absent
            # (_hybrid.py, "a converted model without prompt_lengths has no
            # prompt to preserve"). Omitting it made both arms run the same
            # bidirectional path, and the ~1.05x that produced was noise
            # between two identical configurations — which the gate correctly
            # refused to accept as a reproduction.
            prompt_lengths = torch.full(
                (1,), seq_len // 2, dtype=torch.long, device=device
            )

            def step(model=model):
                with torch.no_grad():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        prompt_lengths=prompt_lengths,
                    )

            seconds = timed_steps(step, warmup=warmup, steps=steps, device=device)
            del step
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "median_seconds": statistics.median(seconds),
                "model_released": probe() is None,
            }

        per_arm = interleaved_trials(
            measure_arm,
            ("dense", "fast"),
            trials=args.trials,
            warmup=args.warmup,
            steps=args.steps,
            device=args.device,
        )
        unreleased = [
            f"L{seq_len}/{label}/trial{m['trial']}"
            for label, arm in per_arm.items()
            for m in arm
            if not m.get("model_released")
        ]
        if unreleased:
            return {
                "check": "hybrid",
                "status": "measurement_invalid",
                "reason": f"arm model(s) outlived their call: {unreleased}",
                "expectation": EXPECTATIONS["hybrid"],
            }
        # Speedup, matching the ledger's convention: >1 means the fast path
        # wins, <1 means it loses.
        speedups = [
            dense["median_seconds"] / fast["median_seconds"]
            for dense, fast in zip(per_arm["dense"], per_arm["fast"], strict=True)
        ]
        cells[f"L{seq_len}"] = {
            "per_arm": per_arm,
            "speedups": speedups,
            "median_speedup": statistics.median(speedups),
            "trials_below_one": sum(1 for value in speedups if value < 1.0),
        }

    low, high = cells["L1024"], cells["L2048"]
    # The shape: a LOSS below the crossover and a gain above it. Requiring the
    # majority of trials on each side keeps a single noisy trial from deciding.
    #
    # Both halves are asserted deliberately. The sub-crossover slowdown alone
    # would be satisfied by a fast path that simply never wins, which is not
    # the gated finding — the finding is a CROSSOVER, and a harness that only
    # checked the loss side could not tell the two apart.
    reproduced = (
        low["median_speedup"] < 1.0
        and low["trials_below_one"] * 2 > len(low["speedups"])
        and high["median_speedup"] > 1.0
    )
    halves = {
        "sub_crossover_loss": low["median_speedup"] < 1.0
        and low["trials_below_one"] * 2 > len(low["speedups"]),
        "above_crossover_gain": high["median_speedup"] > 1.0,
    }
    return {
        "check": "hybrid",
        "status": "reproduced" if reproduced else "NOT_REPRODUCED",
        "expectation": EXPECTATIONS["hybrid"],
        "trials": args.trials,
        "halves_reproduced": halves,
        "observed_shape": (
            f"L=1024 median speedup {low['median_speedup']:.2f}x "
            f"({low['trials_below_one']}/{len(low['speedups'])} trials below 1.0); "
            f"L=2048 median speedup {high['median_speedup']:.2f}x"
        ),
        "cells": cells,
    }


CHECKS = {
    "sparse": check_sparse,
    "noising": check_noising,
    "hybrid": check_hybrid,
}


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
