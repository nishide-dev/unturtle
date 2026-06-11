"""
Compare benchmark results from unturtle and dllm.

Usage:
    python benchmarks/modernbert/compare.py [--unturtle-dir OUTPUT_DIR [--dllm-dir DIR]]

Reads benchmark.json and loss_history.json from both frameworks and prints
a comparison table.
"""

import json
import os
import sys

DEFAULT_UNTURTLE_DIR = "outputs/benchmark_modernbert_unturtle"
DEFAULT_DLLM_DIR = "outputs/benchmark_modernbert_dllm"


def load_results(dir_path, label):
    bench_path = os.path.join(dir_path, "benchmark.json")
    loss_path = os.path.join(dir_path, "loss_history.json")
    if not os.path.isfile(bench_path):
        print(f"Warning: {bench_path} not found. Skipping {label}.")
        return None
    with open(bench_path) as f:
        bench = json.load(f)
    loss_history = []
    if os.path.isfile(loss_path):
        with open(loss_path) as f:
            loss_history = json.load(f)
    bench["loss_history"] = loss_history
    return bench


def fmt(val, unit=""):
    if val is None:
        return "N/A"
    if unit == "s":
        return f"{val:.1f}s"
    if unit == "min":
        return f"{val / 60:.1f}min"
    if unit == "gb":
        return f"{val:.2f}GB"
    if unit == "/s":
        return f"{val:.2f}"
    if unit == "%":
        return f"{val:.1f}%"
    return f"{val:.4f}"


def main():
    unturtle_dir = DEFAULT_UNTURTLE_DIR
    dllm_dir = DEFAULT_DLLM_DIR

    # Parse simple CLI args
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--unturtle-dir":
            unturtle_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--dllm-dir":
            dllm_dir = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown arg: {sys.argv[i]}")
            i += 1

    unturtle_r = load_results(unturtle_dir, "unturtle")
    dllm_r = load_results(dllm_dir, "dllm")

    print("\n" + "=" * 65)
    print("  Benchmark: ModernBERT-base dLLM SFT")
    print("=" * 65)

    rows = [
        ("Model", "unturtle", "dllm"),
        (
            "Train loss (avg)",
            unturtle_r["train_loss_avg"] if unturtle_r else None,
            dllm_r["train_loss_avg"] if dllm_r else None,
        ),
        (
            "Train loss (first 50)",
            unturtle_r["train_loss_first50"] if unturtle_r else None,
            dllm_r["train_loss_first50"] if dllm_r else None,
        ),
        (
            "Train loss (last 50)",
            unturtle_r["train_loss_last50"] if unturtle_r else None,
            dllm_r["train_loss_last50"] if dllm_r else None,
        ),
        (
            "Elapsed",
            unturtle_r["elapsed_seconds"] if unturtle_r else None,
            dllm_r["elapsed_seconds"] if dllm_r else None,
        ),
        (
            "Peak VRAM",
            unturtle_r["peak_vram_gb"] if unturtle_r else None,
            dllm_r["peak_vram_gb"] if dllm_r else None,
        ),
        (
            "Steps/sec",
            unturtle_r["steps_per_second"] if unturtle_r else None,
            dllm_r["steps_per_second"] if dllm_r else None,
        ),
        (
            "Total steps",
            unturtle_r["steps"] if unturtle_r else None,
            dllm_r["steps"] if dllm_r else None,
        ),
        (
            "LoRA params (M)",
            unturtle_r["n_params_trainable"] / 1e6 if unturtle_r else None,
            dllm_r["n_params_trainable"] / 1e6 if dllm_r else None,
        ),
        (
            "LoRA r",
            unturtle_r["lora_r"] if unturtle_r else None,
            dllm_r["lora_r"] if dllm_r else None,
        ),
    ]

    print(f"\n  {'Metric':<25} {'unturtle':>15} {'dllm':>15} {'Delta':>10}")
    print(f"  {'-' * 25} {'-' * 15} {'-' * 15} {'-' * 10}")

    for name, u, d in rows:
        # Format values for display
        if name in (
            "Elapsed",
            "Peak VRAM",
            "Steps/sec",
            "Train loss (avg)",
            "Train loss (first 50)",
            "Train loss (last 50)",
            "Total steps",
            "LoRA params (M)",
            "LoRA r",
        ):
            if isinstance(u, (int, float)) and isinstance(d, (int, float)):
                if name == "Elapsed":
                    u_s = f"{u / 60:.1f}min"
                    d_s = f"{d / 60:.1f}min"
                elif name == "Peak VRAM":
                    u_s = f"{u:.2f}GB"
                    d_s = f"{d:.2f}GB"
                elif name == "Steps/sec":
                    u_s = f"{u:.2f}"
                    d_s = f"{d:.2f}"
                elif "loss" in name.lower():
                    u_s = f"{u:.4f}"
                    d_s = f"{d:.4f}"
                elif name == "Total steps":
                    u_s = str(int(u))
                    d_s = str(int(d))
                elif name == "LoRA params (M)":
                    u_s = f"{u:.1f}"
                    d_s = f"{d:.1f}"
                elif name == "LoRA r":
                    u_s = str(int(u))
                    d_s = str(int(d))
                else:
                    u_s = "N/A"
                    d_s = "N/A"

                # Compute delta for numeric rows
                if d != 0:
                    delta = ((u - d) / d) * 100
                    if name in ("Elapsed", "Peak VRAM"):
                        arrow_sym = "\u2193" if delta < 0 else "\u2191"
                    elif name in ("Steps/sec",):
                        arrow_sym = "\u2191" if delta > 0 else "\u2193"
                    elif "loss" in name.lower():
                        arrow_sym = "\u2193" if delta < 0 else "\u2191"
                    else:
                        arrow_sym = "\u2191" if delta > 0 else "\u2193"
                    delta_s = f"{arrow_sym} {abs(delta):.1f}%"
                else:
                    delta_s = "\u2014"
            else:
                u_s = "N/A"
                d_s = "N/A"
                delta_s = "\u2014"
        else:
            u_s = "unturtle"
            d_s = "dllm"
            delta_s = "\u2014"

        print(f"  {name:<25} {u_s:>15} {d_s:>15} {delta_s:>10}")

    print("\n" + "=" * 65)
    print()


if __name__ == "__main__":
    main()
