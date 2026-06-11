#!/usr/bin/env python3
"""Benchmark block-decode KV cache performance (Phase M.1 baseline).

Compares:
- origin: No caching, full forward every step
- block_decode: Tuple cache with trimming (Phase M.1)

Expected Phase M.1 baseline: ~1.0x speedup (infrastructure only, no optimization yet)
Phase M target: ≥2.0x speedup (block-decode + parallel sampling)
"""

import time

import torch

from unturtle.models.conversion.a2d.tiny_a2d import (
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
)
from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationConfig,
)


def benchmark_generation(model, input_ids, gen_config, warmup=3, iters=10):
    """Benchmark generation with warmup."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(inputs=input_ids, generation_config=gen_config)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = model.generate(inputs=input_ids, generation_config=gen_config)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    return elapsed / iters


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tiny model for fast benchmarking
    config = TinyA2DLlamaConfig(
        vocab_size=256,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
    )
    model = TinyA2DLlamaLMHeadModel(config)
    model = model.to(device)
    model.eval()

    # Test parameters
    batch_size = 2
    prompt_len = 16
    max_new_tokens = 32
    num_steps = 4
    block_length = 8
    mask_token_id = 1

    input_ids = torch.randint(
        low=3,
        high=config.vocab_size,
        size=(batch_size, prompt_len),
        device=device,
    )

    print("\nModel: A2D-4L-256H (tiny)")
    print(f"Batch: {batch_size}, Prompt: {prompt_len}, Max new: {max_new_tokens}")
    print(f"Steps: {num_steps}, Block length: {block_length}")
    print("-" * 60)

    # Benchmark origin (no cache)
    gen_config_origin = MaskedDiffusionGenerationConfig(
        max_new_tokens=max_new_tokens,
        steps=num_steps,
        alg="origin",
        use_cache=False,
        mask_token_id=mask_token_id,
    )
    time_origin = benchmark_generation(
        model, input_ids, gen_config_origin, warmup=3, iters=10
    )

    # Benchmark block_decode (tuple cache)
    gen_config_block = MaskedDiffusionGenerationConfig(
        max_new_tokens=max_new_tokens,
        steps=num_steps,
        alg="origin",
        use_cache=True,
        block_length=block_length,
        mask_token_id=mask_token_id,
    )
    time_block = benchmark_generation(
        model, input_ids, gen_config_block, warmup=3, iters=10
    )

    # Results
    speedup = time_origin / time_block
    print(f"\nOrigin (no cache):     {time_origin * 1000:.2f} ms")
    print(f"Block-decode (tuple):  {time_block * 1000:.2f} ms")
    print(f"Speedup:               {speedup:.2f}x")
    print()
    print(
        "Phase M.1 baseline: ~1.0x (infrastructure only, no block-decode optimization)"
    )
    print("Phase M target: ≥2.0x (block-decode + parallel sampling)")


if __name__ == "__main__":
    main()
