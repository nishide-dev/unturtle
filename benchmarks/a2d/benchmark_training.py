"""Benchmark: A2D forward/loss/backward training throughput.

Measures forward + masked diffusion loss + backward (no optimizer step)
throughput across representative dLLM training shapes.

Usage:
    python benchmarks/a2d/benchmark_training.py

Output:
    Markdown table suitable for CLAUDE.md / PR comments.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import torch

from unturtle.kernels.masked_diffusion_loss import fast_masked_diffusion_loss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 5
ITERS = 20
HIDDEN = 256
N_HEADS = 4
N_LAYERS = 2
MAX_POS = 256

SHAPES: list[tuple[int, int, int, str]] = [
    # (batch, seq_len, vocab, loss_weight_type)
    (2, 128, 32000, "uniform"),
    (2, 512, 32000, "uniform"),
    (4, 128, 128256, "uniform"),
    (4, 512, 128256, "uniform"),
    (2, 128, 32000, "timestep"),
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _make_tiny_a2d_model(vocab: int) -> torch.nn.Module:
    """Create a tiny A2D LLaMA model for benchmarking (randinit, no download)."""
    from unturtle.models.conversion.a2d.tiny_a2d import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    cfg = TinyA2DLlamaConfig(
        vocab_size=vocab,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 4,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        max_position_embeddings=MAX_POS,
    )
    model = TinyA2DLlamaLMHeadModel(cfg)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    label: str
    B: int
    L: int
    V: int
    weight_type: str
    tokens_per_sec: float
    mem_mb: float


def _bench_forward_backward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    diffusion_mask: torch.Tensor,
    loss_weight_type: str,
    timesteps: torch.Tensor,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    """Return (tokens_per_sec, peak_memory_mb)."""
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    # Warmup
    for _ in range(warmup):
        out = model(input_ids)
        logits = out.logits
        weights = None
        if loss_weight_type == "timestep":
            weights = (
                (1.0 / timesteps.clamp_min(1e-6)).unsqueeze(1).expand_as(logits[..., 0])
            )
        loss = fast_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_weights=weights,
        )
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = model(input_ids)
        logits = out.logits
        weights = None
        if loss_weight_type == "timestep":
            weights = (
                (1.0 / timesteps.clamp_min(1e-6)).unsqueeze(1).expand_as(logits[..., 0])
            )
        loss = fast_masked_diffusion_loss(
            logits=logits,
            labels=labels,
            diffusion_mask=diffusion_mask,
            loss_weights=weights,
        )
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    end = time.perf_counter()

    B, L = input_ids.shape
    total_tokens = B * L * iters
    elapsed = end - start
    tokens_per_sec = total_tokens / elapsed

    peak = torch.cuda.max_memory_allocated()
    mem_mb = (peak - mem_before) / 1024**2
    return tokens_per_sec, mem_mb


def run_benchmarks() -> list[BenchResult]:
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    results: list[BenchResult] = []

    for B, L, V, weight_type in SHAPES:
        gc.collect()
        torch.cuda.empty_cache()

        model = _make_tiny_a2d_model(V).to(device).to(torch.float32)

        torch.manual_seed(0)
        input_ids = torch.randint(0, V, (B, L), device=device)
        labels = torch.randint(0, V, (B, L), device=device)
        diffusion_mask = torch.rand(B, L, device=device) < 0.5
        diffusion_mask[:, 0] = True
        timesteps = torch.rand(B, device=device) * 0.9 + 0.1

        tp, mem = _bench_forward_backward(
            model,
            input_ids,
            labels,
            diffusion_mask,
            weight_type,
            timesteps,
            WARMUP,
            ITERS,
        )

        label = f"unturtle (A2D-{N_LAYERS}L-{HIDDEN}H, {weight_type})"
        results.append(BenchResult(label, B, L, V, weight_type, tp, mem))
        print(
            f"[B={B:2d} L={L:4d} V={V:7d} {weight_type}]  "
            f"{tp:10,.0f} tokens/sec  "
            f"mem={mem:.1f} MB"
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    return results


def _md_table(results: list[BenchResult]) -> str:
    rows = []
    rows.append("| Batch | SeqLen | Vocab | Weight | Tokens/sec | Mem (MB) |")
    rows.append("|------:|-------:|------:|--------|-----------:|---------:|")
    for r in results:
        rows.append(
            f"| {r.B} | {r.L} | {r.V:,} | {r.weight_type} | {r.tokens_per_sec:,.0f} | {r.mem_mb:6.1f} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: A2D-{N_LAYERS}L-{HIDDEN}H (tiny)")
    print(f"Warmup={WARMUP} Iters={ITERS}\n")

    results = run_benchmarks()

    print("\n## Training Throughput Results\n")
    print(_md_table(results))
