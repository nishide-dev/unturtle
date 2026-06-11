# Examples

## Diffu-GRPO (GPU smoke)

End-to-end `DiffuGRPOTrainer.train()` for a few steps on a tiny random masked LM (no checkpoint download).

**Dependencies:** `uv pip install -e ".[huggingface,grpo]"` (or `trl`, `mergekit`, `tokenizers` on top of the Hugging Face extra). **CUDA** is required for this script.

```bash
.venv/bin/python examples/grpo_diffu_train_smoke.py
```

Options: `--wd1` (wd1 loss), `--wd1++` (wd1++ loss). See the module docstring in `grpo_diffu_train_smoke.py` for details.

Design note: [docs/diffu-grpo-d1-notes.md](../docs/diffu-grpo-d1-notes.md) (d1 alignment, `scale_rewards`, Gumbel).

## Configs

YAML under `examples/configs/` (e.g. `llada_grpo.yaml`) for CLI-oriented experiments.
