# examples/ inventory and disposition (#205 PR 1 — policy freeze)

Frozen BEFORE any file moves: every disposition below is the reviewed decision
that PR 2 (benchmark/validation separation) and PR 3 (public examples/recipes)
execute. Moving first and rationalizing afterwards is explicitly out of
process. Runtime code is not touched by any #205 PR.

Legend: **KEEP** stays in `examples/` · **MOVE** relocates verbatim (defaults,
output schema and measurement conditions unchanged) · **REWRITE** replaced by
a thin, current-API version · **DELETE** removed · **ARCHIVE** moved out of
the public surface without deletion (e.g. `experiments/`).

| File | Purpose | Intended user | Resources | Owner issue | CI / tests | Disposition | Destination (PR) |
|---|---|---|---|---|---|---|---|
| `benchmark_a2d_aligned.py` (818 L) | Cross-backend latency/NFE benchmark: unturtle vs external `dllm`, aligned/validator-warm/cold-start modes | maintainers comparing backends | 2 venvs (`.venv`, `.venvDllm`), CUDA, `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` download | #205 (perf history: #127/#182) | `tests/examples/test_benchmark_a2d_aligned.py` (18 unit tests, import `examples.…`) | **MOVE** — delete only if PR 2 review finds `benchmarks/qwen3/benchmark_dllm.py` fully supersedes it (overlap unproven today) | `benchmarks/a2d/` (PR 2); tests move to `tests/benchmarks/`, imports repointed |
| `validate_a2d_real_inference.py` (402 L) | Real-checkpoint output validation: unturtle vs `dllm` on pinned MDLM/BD3LM checkpoints | maintainers gating releases/regressions | 2 venvs, CUDA, `dllm-hub/Qwen3-0.6B-diffusion-{mdlm,bd3lm}-v0.1` | #205 | `tests/examples/test_validate_a2d_real_inference.py` (7 unit tests) | **MOVE** | `tools/validation/` (PR 2); tests move, imports repointed; checkpoint revisions + validated properties documented in-module |
| `grpo_diffu_train_smoke.py` (222 L) | End-to-end `DiffuGRPOTrainer.train()` smoke on a tiny random model (no downloads) | users verifying the GRPO stack | CUDA, `[huggingface,grpo]` extras | #205 | README-listed; imports only public `unturtle.diffusion` API | **KEEP** — meets every PR 3 criterion (public API, README-discoverable, explicit command/deps, copyable) | stays (PR 3 re-verifies executability) |
| `training/run_training.py` (198 L) | Headless LLaDA-8B SFT launcher mirroring the demo notebook | script users | CUDA, 8B checkpoint download, HF datasets | #205 | none; not README-listed | **REWRITE** to a thin current-API public example — or ARCHIVE to `experiments/` if PR 3 finds `unturtle train --config` already covers it (likely: the CLI + `llada_sft.yaml` is the supported path) | PR 3 |
| `demos/llada_sft_demo.py` (384 L) | LLaDA-8B SFT marimo notebook (tulu-3 subset) | notebook users | CUDA, 8B checkpoint, marimo | #205 | none; survives via a broad lint carve-out (`F821 by design`) — the exact red flag PR 3 screens for | **DELETE/ARCHIVE** unless PR 3 re-validation on the current API shows distinct value over the CLI path; if archived, the `examples/demos` lint exclusion is removed with it | PR 3 |
| `configs/a2d_llama_sft.yaml` | `unturtle train` SFT recipe (A2D-Llama3-8B) | CLI users | CUDA + checkpoint at run time | #205 | `tests/test_cli_smoke.py` loads/validates it | **KEEP** — tested public recipe | stays |
| `configs/dream_sft.yaml` | `unturtle train` SFT recipe (Dream-7B) | CLI users | 〃 | #205 | 〃 | **KEEP** | stays |
| `configs/llada_sft.yaml` | `unturtle train` SFT recipe (LLaDA-8B) | CLI users | 〃 | #205 | 〃 (incl. builder test) | **KEEP** | stays |
| `configs/llada_grpo.yaml` | `unturtle train` diffu-GRPO/wd1 recipe | CLI users | CUDA + `[grpo]` extras | #205 | 〃 (task/builder tests) | **KEEP** | stays |
| `README.md` | index | everyone | — | #205 | — | **REWRITE** (this PR: honest current state; PR 2/3 update it again after moves) | stays |

End state (after PR 3): `examples/` contains only short, current-public-API,
README-listed material with explicit dependencies and resource requirements —
today that is `grpo_diffu_train_smoke.py` + the four tested `configs/*.yaml`,
plus whatever the `run_training.py` rewrite decision produces.

Lint posture: `examples/**` keeps only the `E402` carve-out (documented:
patching before model imports). The `examples/demos` `F821` carve-out is
deleted together with the notebook's disposition in PR 3 — no file survives
solely because lint excludes it.
