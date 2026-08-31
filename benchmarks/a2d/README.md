# benchmarks/a2d

## `benchmark_a2d_aligned.py` — cross-backend inference benchmark

Latency/NFE comparison of **unturtle vs the external `dllm` backend** on the
pinned A2D checkpoint `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`. Moved
verbatim from `examples/` in #205 PR 2 — defaults, output schema
(`outputs/a2d_aligned_benchmark/` JSON records + summary) and measurement
conditions are unchanged; only the repo-root derivation was adapted to the
new location (`parents[2]`).

**Resource requirements**: CUDA GPU; TWO virtualenvs — `.venv` (unturtle) and
`.venvDllm` (`dllm`, transformers 4.x; see `benchmarks/qwen3/benchmark_dllm.py`
header for the setup recipe); checkpoint download on first run.

**Warmup / build / compile handling** (`--mode`, default `aligned-warm`):

- `aligned-warm` — both backends measured after `--warmup-iters` (default 2)
  in-process warmup generations: steady-state kernels, caches and model warm;
  the apples-to-apples number.
- `validator-warm` — warmup mirrors the validation tool's call pattern instead
  of the benchmark's own, for comparing against
  `tools/validation/validate_a2d_real_inference.py` timings.
- `cold-start` — each run in a FRESH subprocess per backend env: includes
  import, model load and first-call compile/build costs. No warmup by design.

Per-mode/per-backend model caches are process-local; cold-start explicitly
bypasses them.

```bash
.venv/bin/python benchmarks/a2d/benchmark_a2d_aligned.py --mode aligned-warm
```

Not superseded by `benchmarks/qwen3/benchmark_dllm.py` (that is an SFT
training-loss benchmark of the reference backend, not an inference
comparison) — the PR 1 delete-if-superseded check concluded MOVE.

## Other files

- `benchmark_block_decode.py`, `benchmark_training.py`,
  `hybrid_vs_bidirectional.py` — pre-existing A2D benchmarks (unchanged).

Unit tests: `tests/benchmarks/test_benchmark_a2d_aligned.py` (18 tests,
CPU-only; exercise record/summary schema, mode selection and subprocess
plumbing with stubs — no GPU or checkpoint needed).
