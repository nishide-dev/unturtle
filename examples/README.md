# Examples

Honest current state (#205 PR 1). Dispositions for every file are frozen in
[docs/maintenance/examples-inventory.md](../docs/maintenance/examples-inventory.md);
PR 2 moves the benchmark/validation tools out, PR 3 settles the rest. Until
then, this README describes what actually exists here today.

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

## Present but NOT user examples (PR 3 dispositions pending)

- `training/run_training.py` — headless LLaDA SFT launcher; not
  README-supported; overlaps the `unturtle train` CLI path. **REWRITE-or-
  archive decision in #205 PR 3.**
- `demos/llada_sft_demo.py` — marimo notebook, currently alive mainly via a
  lint carve-out. **Re-validation / delete-or-archive in #205 PR 3.**

Moved out in #205 PR 2: the cross-backend benchmark now lives at
`benchmarks/a2d/benchmark_a2d_aligned.py` and the release-validation tool at
`tools/validation/validate_a2d_real_inference.py` (see their READMEs for
resource requirements).
