# tools/validation

Maintainer validation tools — not user examples (moved from `examples/` in
#205 PR 2, verbatim: defaults, output schema and checkpoint semantics
unchanged).

## `validate_a2d_real_inference.py`

Real-checkpoint output validation of **unturtle vs the external `dllm`
backend**.

**Checkpoint revisions under validation** (pinned in-module):

- MDLM: `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`
- BD3LM: `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`

**What is validated**: for each checkpoint, both backends load the same
revision and generate from the same prompts; the tool records per-backend
outputs, timing and environment metadata into a JSON report and compares the
decoded results across backends — a release/regression gate for the A2D
inference path, not a performance benchmark (use
`benchmarks/a2d/benchmark_a2d_aligned.py` for timing).

**Resource requirements**: CUDA GPU; the same TWO virtualenvs as the
benchmark (`.venv` + `.venvDllm`); checkpoint downloads.

```bash
.venv/bin/python tools/validation/validate_a2d_real_inference.py
```

Unit tests: `tests/tools/test_validate_a2d_real_inference.py` (7 tests,
CPU-only, stub-based).
