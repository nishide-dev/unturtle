# experiments/

Archived, non-public scripts kept for provenance (#205 PR 3). Nothing here is
a supported example: no README-level promise, no CI, may rot. The public
surface lives in `examples/` (see its README) and the `unturtle` CLI.

| File | What it was | Why archived | Superseded by |
|---|---|---|---|
| `run_training_llada_sft.py` | Headless LLaDA-8B SFT launcher (current public API: `FastDiffusionModel` + `DiffusionTrainer`) | No distinct value over the tested CLI recipe — hardcoded parameters, no CLI surface | `unturtle train --config examples/configs/llada_sft.yaml` (load-validated by `tests/test_cli_smoke.py`) |
