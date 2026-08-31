# Examples

Current state (#205 complete). The disposition history for every file is in
[docs/maintenance/examples-inventory.md](../docs/maintenance/examples-inventory.md). Everything here is short, uses the current public API, and states its dependencies and resource requirements.

## Supported examples

### Diffu-GRPO GPU smoke — `grpo_diffu_train_smoke.py`

End-to-end `DiffuGRPOTrainer.train()` for a few steps on a tiny random masked
LM (no checkpoint download). Public API only (`unturtle.diffusion`).

**Requires:** CUDA; `uv pip install -e ".[huggingface,grpo]"`.

```bash
.venv/bin/python examples/grpo_diffu_train_smoke.py   # options: --wd1 / --wd1++
```

Design note: [docs/diffu-grpo-d1-notes.md](../docs/diffu-grpo-d1-notes.md).

### Tested CLI recipes — `configs/*.yaml`

Public recipes for `unturtle train --config …`, all load-validated by
`tests/test_cli_smoke.py`:

| Config | Task | Model (downloaded at run time) |
|---|---|---|
| `configs/llada_sft.yaml` | SFT | `GSAI-ML/LLaDA-8B-Instruct` |
| `configs/a2d_llama_sft.yaml` | SFT | `nishide-dev/A2D-Llama3-8B-dLLM` |
| `configs/dream_sft.yaml` | SFT | `nishide-dev/Dream-7B-dLLM` |
| `configs/llada_grpo.yaml` | diffu-GRPO / wd1 (needs `[grpo]` extras) | LLaDA-8B |

Running the recipes needs CUDA and the listed checkpoint downloads.

## Everything else

The former maintainer tools and drafts are gone from this directory (#205):
the cross-backend benchmark lives at `benchmarks/a2d/benchmark_a2d_aligned.py`,
the release-validation tool at `tools/validation/validate_a2d_real_inference.py`
(PR 2), the headless SFT launcher is archived under `experiments/` (superseded
by the `unturtle train` recipe above), and the marimo demo was deleted with
its lint carve-out (PR 3). `tests/test_examples_surface.py` keeps this README
and the directory bidirectionally consistent.
