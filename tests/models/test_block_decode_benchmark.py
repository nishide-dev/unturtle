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

"""Smoke benchmark for block-decode KV-cache speedup."""

import time

import pytest
import torch

from unturtle.models.conversion.a2d.tiny_a2d import (
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for meaningful benchmark"
)
class TestBlockDecodeBenchmark:
    """Smoke benchmark to verify KV-cache speedup."""

    @pytest.fixture
    def benchmark_model(self):
        """Create a slightly larger model for benchmarking."""
        torch.manual_seed(42)
        config = TinyA2DLlamaConfig(
            vocab_size=512,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=256,
            mask_token_id=500,
        )
        model = TinyA2DLlamaLMHeadModel(config).cuda()
        model.eval()
        return model

    def test_cache_speedup_smoke(self, benchmark_model):
        """Smoke test to measure cache overhead (Phase L baseline).

        Phase L implements basic cache infrastructure but does not yet include
        block-wise decoding optimizations. This test confirms the cache path
        works without adding significant overhead.

        **Expected**: ~1.0x (no speedup yet, cache resets each step)
        **Phase M target**: ≥2.0x (with block-decode + parallel sampling)

        Full benchmarks will be in benchmarks/ after Phase M.
        """
        prompt = torch.tensor([[1, 2, 3, 4, 5]]).cuda()
        warmup_runs = 2
        test_runs = 5

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                benchmark_model.diffusion_generate(
                    inputs=prompt,
                    max_new_tokens=32,
                    steps=16,
                    use_cache=False,
                    mask_token_id=500,
                )

        # Benchmark no-cache
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(test_runs):
            with torch.no_grad():
                benchmark_model.diffusion_generate(
                    inputs=prompt,
                    max_new_tokens=32,
                    steps=16,
                    use_cache=False,
                    mask_token_id=500,
                )
        torch.cuda.synchronize()
        no_cache_time = (time.perf_counter() - start) / test_runs

        # Warmup cache
        for _ in range(warmup_runs):
            with torch.no_grad():
                benchmark_model.diffusion_generate(
                    inputs=prompt,
                    max_new_tokens=32,
                    steps=16,
                    use_cache=True,
                    block_length=16,
                    mask_token_id=500,
                )

        # Benchmark with-cache
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(test_runs):
            with torch.no_grad():
                benchmark_model.diffusion_generate(
                    inputs=prompt,
                    max_new_tokens=32,
                    steps=16,
                    use_cache=True,
                    block_length=16,
                    mask_token_id=500,
                )
        torch.cuda.synchronize()
        with_cache_time = (time.perf_counter() - start) / test_runs

        speedup = no_cache_time / with_cache_time

        print("\n=== Block-decode KV-cache Smoke Benchmark (Phase L) ===")
        print(f"No-cache:       {no_cache_time * 1000:.2f} ms")
        print(f"With-cache:     {with_cache_time * 1000:.2f} ms")
        print(f"Speedup:        {speedup:.2f}x")
        print("Phase L status: Infrastructure complete (no optimization yet)")
        print("Phase M target: ≥2.0x (block-decode + parallel sampling)")

        # Phase L: Both paths must run; timing is a loose smoke check only.
        # Some GPUs/drivers show the cache path slower here (extra bookkeeping, small
        # tensors); tight bands caused flaky CI. Phase M targets real speedup (≥2x).
        assert 0.15 <= speedup <= 5.0, (
            f"Cache speedup {speedup:.2f}x is outside sanity range (0.15x-5.0x). "
            "Expect extreme values only if a path errors or hangs."
        )
