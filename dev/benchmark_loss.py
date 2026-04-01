"""Benchmark: fast_masked_diffusion_loss (Triton) vs reference dLLM implementations.

Three implementations are compared:

1. **Triton** (unturtle): fast_masked_diffusion_loss — chunked Triton CE kernel,
   operates only on masked positions (label=-100 trick).

2. **d1-style** (reference): F.cross_entropy over all tokens with labels=-100 at
   unmasked positions, then divide by t.  This is exactly what d1/LLaDA use in
   practice (dev/repos/d1/SFT/sft_trainer.py).

3. **PyTorch masked** (baseline): same mask logic as Triton but using stock
   F.cross_entropy — the minimal Python-level equivalent without Triton.

Measuring throughput (ms/iter) and peak GPU memory across typical dLLM shapes.

Usage:
    python dev/benchmark_loss.py

Output:
    Markdown table suitable for CLAUDE.md / PR comments.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss
from unturtle.kernels.fused_masked_diffusion_loss import fused_masked_diffusion_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 10
ITERS = 100
MASK_RATIO = 0.5  # typical for dLLM training

SHAPES: list[tuple[int, int, int]] = [
    # (batch, seq_len, vocab)   — representative dLLM shapes
    (4,   128,  32000),   # LLaMA-3 vocab, short seq
    (4,   512,  32000),   # LLaMA-3 vocab, medium seq
    (2,   128, 128256),   # LLaMA-3.1 vocab (128K)
    (2,   512, 128256),   # LLaMA-3.1 vocab, medium seq
    (1,  1024, 128256),   # long seq
    (4,   128, 126464),   # LLaDA vocab (~126K)
    (4,   512, 126464),   # LLaDA vocab, medium seq
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(
    B: int, L: int, V: int, device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    logits = torch.randn(B, L, V, device=device, dtype=torch.float32)
    labels = torch.randint(0, V, (B, L), device=device)
    mask = torch.rand(B, L, device=device) < MASK_RATIO
    mask[:, 0] = True
    return logits, labels, mask


def _pytorch_masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
) -> torch.Tensor:
    """PyTorch baseline: same mask logic as Triton, no Triton kernel."""
    B, L, V = logits.shape
    masked_labels = labels.clone()
    masked_labels[~diffusion_mask] = -100
    per_token = F.cross_entropy(
        logits.reshape(B * L, V),
        masked_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view(B, L)
    return per_token.sum() / diffusion_mask.sum().clamp_min(1)


def _d1_style_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    t: float = 0.5,
) -> torch.Tensor:
    """d1/LLaDA reference: F.cross_entropy over all tokens, then divide by t.

    Source: dev/repos/d1/SFT/sft_trainer.py  dLLMTrainer.compute_loss()
    Labels at unmasked positions are already -100 (as set by the collator).
    """
    B, L, V = logits.shape
    # d1 uses labels=-100 at unmasked positions (same convention as ours)
    masked_labels = labels.clone()
    masked_labels[~diffusion_mask] = -100
    unscaled = F.cross_entropy(
        logits.view(-1, V),
        masked_labels.view(-1),
        reduction="none",
    ).view(B, L)
    # d1 divides by t (scalar broadcast per sequence)
    loss = unscaled / t
    # normalise by total number of tokens (d1 style: numel - prompt_tokens)
    n_total = diffusion_mask.sum().clamp_min(1)
    return loss.sum() / n_total


@dataclass
class BenchResult:
    label: str
    B: int
    L: int
    V: int
    time_ms: float      # mean wall time per iteration (ms)
    mem_mb: float       # peak GPU memory delta (MB)
    loss: float         # sanity-check loss value


def _bench_fn(fn, logits, labels, mask, warmup: int, iters: int) -> tuple[float, float]:
    """Return (mean_ms, peak_memory_delta_mb)."""
    # Warmup
    for _ in range(warmup):
        _ = fn(logits, labels, mask)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    start = time.perf_counter()
    for _ in range(iters):
        loss = fn(logits, labels, mask)
    torch.cuda.synchronize()
    end = time.perf_counter()

    peak = torch.cuda.max_memory_allocated()
    mean_ms = (end - start) / iters * 1000
    mem_mb = (peak - mem_before) / 1024 ** 2
    return mean_ms, mem_mb, loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmarks() -> list[BenchResult]:
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    results: list[BenchResult] = []

    for B, L, V in SHAPES:
        logits, labels, mask = _make_inputs(B, L, V, device)

        # --- Triton (unturtle) ---
        gc.collect(); torch.cuda.empty_cache()
        tri_ms, tri_mem, tri_loss = _bench_fn(
            fast_masked_diffusion_loss, logits, labels, mask, WARMUP, ITERS
        )
        results.append(BenchResult("Triton (unturtle)", B, L, V, tri_ms, tri_mem, tri_loss))

        # --- d1/LLaDA reference ---
        gc.collect(); torch.cuda.empty_cache()
        d1_ms, d1_mem, d1_loss = _bench_fn(
            _d1_style_loss, logits, labels, mask, WARMUP, ITERS
        )
        results.append(BenchResult("d1-style (reference)", B, L, V, d1_ms, d1_mem, d1_loss))

        # --- PyTorch masked baseline ---
        gc.collect(); torch.cuda.empty_cache()
        py_ms, py_mem, py_loss = _bench_fn(
            _pytorch_masked_loss, logits, labels, mask, WARMUP, ITERS
        )
        results.append(BenchResult("PyTorch masked", B, L, V, py_ms, py_mem, py_loss))

        # --- Fused (unturtle Phase D4) ---
        gc.collect(); torch.cuda.empty_cache()
        fused_ms, fused_mem, fused_loss = _bench_fn(
            fused_masked_diffusion_loss, logits, labels, mask, WARMUP, ITERS
        )
        results.append(BenchResult("Fused (unturtle D4)", B, L, V, fused_ms, fused_mem, fused_loss))

        print(
            f"[B={B:2d} L={L:4d} V={V:6d}]  "
            f"Triton {tri_ms:6.2f}ms  "
            f"d1-ref {d1_ms:6.2f}ms (x{d1_ms/tri_ms:.2f})  "
            f"PyTorch {py_ms:6.2f}ms (x{py_ms/tri_ms:.2f})  "
            f"Fused {fused_ms:6.2f}ms (x{fused_ms/tri_ms:.2f})  "
            f"| mem Triton={tri_mem:.1f}MB d1={d1_mem:.1f}MB Fused={fused_mem:.1f}MB"
        )

    return results


def _md_table(results: list[BenchResult]) -> str:
    rows = []
    rows.append("| Batch | SeqLen | Vocab | Impl | Time (ms) | Mem (MB) | vs d1-ref | vs PyTorch |")
    rows.append("|------:|-------:|------:|------|----------:|---------:|----------:|-----------:|")

    it = iter(results)
    for tri in it:
        d1 = next(it)
        py = next(it)
        fused = next(it)
        rows.append(
            f"| {tri.B} | {tri.L} | {tri.V:,} | **Triton (unturtle)** |"
            f" {tri.time_ms:7.2f} | {tri.mem_mb:6.1f} |"
            f" **{d1.time_ms/tri.time_ms:.2f}x** |"
            f" **{py.time_ms/tri.time_ms:.2f}x** |"
        )
        rows.append(
            f"| | | | d1-style (reference) |"
            f" {d1.time_ms:7.2f} | {d1.mem_mb:6.1f} | — | — |"
        )
        rows.append(
            f"| | | | PyTorch masked |"
            f" {py.time_ms:7.2f} | {py.mem_mb:6.1f} | — | — |"
        )
        rows.append(
            f"| | | | **Fused (unturtle D4)** |"
            f" {fused.time_ms:7.2f} | {fused.mem_mb:6.1f} |"
            f" **{d1.time_ms/fused.time_ms:.2f}x** |"
            f" **{py.time_ms/fused.time_ms:.2f}x** |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Shapes: {SHAPES}")
    print(f"Warmup={WARMUP} Iters={ITERS} mask_ratio={MASK_RATIO}\n")

    results = run_benchmarks()

    print("\n## Benchmark Results\n")
    print(_md_table(results))
