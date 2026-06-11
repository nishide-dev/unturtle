# Unturtle

**dLLM method layer on top of [unsloth](https://github.com/unslothai/unsloth)**

Unturtle adds diffusion language model (dLLM) capabilities — conversion recipes,
training objectives, and inference acceleration — as a thin layer on top of the
unsloth fast-training stack.

## Layer architecture

```
transformers          model implementations + loss primitives
TRL                   objective trainers (SFT, GRPO, …)
unsloth               hardware acceleration patches (LoRA fast paths, kernels)
unturtle              dLLM method layer (this repo)
                        ├── models/backbones/     native diffusion backbones (LLaDA, Dream, ModernBERT-diffusion)
                        ├── models/conversion/    AR→Diffusion methods (Tiny-A2D)
                        ├── models/generation/    shared cache / block-decode / generation mixin
                        ├── diffusion/            MDLM / BD3LM trainer, collator, GRPO
                        ├── kernels/              Triton kernels, fast LoRA paths
                        └── eval/                 smoke evaluators + lm-evaluation-harness adapter
```

## Quick start

```bash
./install.sh          # uv venv + CUDA-matched torch + editable install (+ verification)
./install.sh --eval   # additionally install the lm-eval-harness extra
```

See the header of [`install.sh`](install.sh) for `TORCH_INDEX` / `PYTHON_VERSION`
overrides. Plain `pip` is not supported — use `uv` (the script handles ordering:
torch must be installed before the editable install so the CUDA-matched build
is preserved).

## Legacy repository

This repository is a clean rebuild of the archived
[unturtle-legacy](https://github.com/nishide-dev/unturtle-legacy) monorepo
(originally a fork of unsloth; see `NOTICE` for the vendored-code provenance).
The legacy repository includes Unturtle Studio and lighteval integration;
those components are not carried forward here. Its issue/PR history remains
browsable read-only.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
