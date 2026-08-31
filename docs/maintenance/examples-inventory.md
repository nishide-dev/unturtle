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
| `benchmark_a2d_aligned.py` (818 L) | Cross-backend latency/NFE benchmark: unturtle vs external `dllm`, aligned/validator-warm/cold-start modes | maintainers comparing backends | 2 venvs (`.venv`, `.venvDllm`), CUDA, `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` download | #205 (perf history: #127/#182) | `tests/examples/test_benchmark_a2d_aligned.py` (18 unit tests, import `examples.…`) | **MOVED** (PR 2): supersession check concluded NOT superseded — `benchmarks/qwen3/benchmark_dllm.py` is an SFT training-loss benchmark, this is an inference comparison | `benchmarks/a2d/benchmark_a2d_aligned.py`; tests at `tests/benchmarks/` |
| `validate_a2d_real_inference.py` (402 L) | Real-checkpoint output validation: unturtle vs `dllm` on pinned MDLM/BD3LM checkpoints | maintainers gating releases/regressions | 2 venvs, CUDA, `dllm-hub/Qwen3-0.6B-diffusion-{mdlm,bd3lm}-v0.1` | #205 | `tests/examples/test_validate_a2d_real_inference.py` (7 unit tests) | **MOVED** (PR 2) | `tools/validation/validate_a2d_real_inference.py`; tests at `tests/tools/`; checkpoint revisions + validated properties in `tools/validation/README.md` |
| `grpo_diffu_train_smoke.py` (222 L) | End-to-end `DiffuGRPOTrainer.train()` smoke on a tiny random model (no downloads) | users verifying the GRPO stack | CUDA, `[huggingface,grpo]` extras | #205 | README-listed; imports only public `unturtle.diffusion` API | **KEPT** (PR 3): re-verified end-to-end on the current environment — 8 GRPO steps to completion on CUDA | stays |
| `training/run_training.py` (198 L) | Headless LLaDA-8B SFT launcher mirroring the demo notebook | script users | CUDA, 8B checkpoint download, HF datasets | #205 | none; not README-listed | **ARCHIVED** (PR 3): current-API but zero distinct value over the tested CLI recipe (`unturtle train --config examples/configs/llada_sft.yaml`) — hardcoded params, no CLI surface | `experiments/run_training_llada_sft.py` |
| `demos/llada_sft_demo.py` (384 L) | LLaDA-8B SFT marimo notebook (tulu-3 subset) | notebook users | CUDA, 8B checkpoint, marimo | #205 | none; survives via a broad lint carve-out (`F821 by design`) — the exact red flag PR 3 screens for | **DELETED** (PR 3): current-API but same SFT flow as the tested CLI recipe; archiving would have carried the F821 lint carve-out to a new path — the exact延命 pattern being screened; the exclusion is removed with it | git history |
| `configs/a2d_llama_sft.yaml` | `unturtle train` SFT recipe (A2D-Llama3-8B) | CLI users | CUDA + checkpoint at run time | #205 | `tests/test_cli_smoke.py` loads/validates it | **KEEP** — tested public recipe | stays |
| `configs/dream_sft.yaml` | `unturtle train` SFT recipe (Dream-7B) | CLI users | 〃 | #205 | 〃 | **KEEP** | stays |
| `configs/llada_sft.yaml` | `unturtle train` SFT recipe (LLaDA-8B) | CLI users | 〃 | #205 | 〃 (incl. builder test) | **KEEP** | stays |
| `configs/llada_grpo.yaml` | `unturtle train` diffu-GRPO/wd1 recipe | CLI users | CUDA + `[grpo]` extras | #205 | 〃 (task/builder tests) | **KEEP** | stays |
| `README.md` | index | everyone | — | #205 | — | **REWRITE** (this PR: honest current state; PR 2/3 update it again after moves) | stays |

## Final surface (executed, PR 3)

`examples/` now contains exactly: `grpo_diffu_train_smoke.py` (re-verified
end-to-end) + the four cli-smoke-tested `configs/*.yaml` + `README.md` —
short, current-public-API, README-listed, with explicit dependencies and
resource requirements. `tests/test_examples_surface.py` enforces the
directory↔README bidirectional consistency from now on.

Lint posture: `examples/**` keeps only the documented `E402` carve-out; the
`examples/demos` `F821` exclusion is GONE (deleted with the notebook — no
file survives solely because lint excludes it).

Related surfaces after #205: `benchmarks/a2d/` + `tools/validation/`
(maintainer tools, PR 2), `experiments/` (archived provenance, no support
promise), `packs/` (lifecycle-documented reference packs).
