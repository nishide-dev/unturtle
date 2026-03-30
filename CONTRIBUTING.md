# Contributing to Unturtle

Unturtle is a fork of [unslothai/unsloth](https://github.com/unslothai/unsloth) focused on
**Diffusion Language Model (dLLM)** training with Triton-optimized kernels.

## Ways to Contribute

- **Report bugs** — open an issue with a minimal reproduction
- **Propose features** — dLLM algorithms (LLaDA, MDLM, BD3LM, d1), new schedulers, RL stages
- **Submit PRs** — see the workflow below
- **Improve docs** — `dev/` design documents, `CLAUDE.md`

## Development Setup

See [CLAUDE.md](CLAUDE.md) for the full setup guide (Python 3.12, uv, CUDA 12.4).

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install "setuptools==80.9.0" && uv pip install -e ".[huggingface]"
uv pip install pytest ruff
```

## PR Workflow

1. Open an Issue first (`[Phase N] <verb> <target>`)
2. Branch: `<type>/<issue#>-<short-description>`
3. Write tests first (see `tests/diffusion/`)
4. Run `pytest tests/ -v` — all tests must pass
5. For Triton kernel changes: paste numerical test output in the PR comment
6. Submit PR — 1 PR per Issue, Squash and merge

## Commit Format

```
<emoji> <type>(<scope>): <description> (#<issue>)
```

Examples:
```
✨ feat(kernel): add fast_masked_diffusion_loss (#42)
🐛 fix(collator): fix Bernoulli mask rate (#55)
⚡ perf(kernel): chunk loss for vocab>65K (#61)
```

Full emoji/type table and PR rules: [CLAUDE.md § Git / Issue / PR](CLAUDE.md)

## Key Constraint

**Do not modify existing AR-LLM functionality.** All dLLM additions are new files only.
Upstream unsloth changes are merged via `git fetch upstream && git merge upstream/main`.

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).
