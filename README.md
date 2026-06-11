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
uv venv .venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[huggingface]"
```

## Legacy repository

This repository is a clean rebuild forked from the legacy
[unturtle monorepo](https://github.com/nishide-dev/unturtle) at commit
`a6c1f893fc87c0973f9c32e59ca3d7d54ffb9724` (2026-03-28).
The legacy repository includes Unturtle Studio and lighteval integration;
those components are not carried forward here.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
