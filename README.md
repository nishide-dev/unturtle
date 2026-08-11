# Unturtle

**The systems/method layer for rapidly researching diffusion/flow language
models — training, post-training, and generation — on top of the existing
Transformers / TRL / Unsloth ecosystem.**

Unturtle does not own a training loop and does not replace upstream. It adds
the diffusion/flow-specific pieces — forward processes, objective trainers,
conversion recipes, generation algorithms, and a decision-grade evaluation
surface — as a composable layer, and keeps a strict line between *implemented
mechanism* and *promoted capability*: research methods ship as explicit
opt-ins until gate experiments earn them a capability claim.

## Layer architecture

```
transformers          model implementations + in-model loss / generation primitives
TRL                   objective trainers (DPOTrainer, GRPOTrainer, … as peers)
unsloth               hardware-acceleration patches (fast LoRA, Triton kernels)
unturtle              diffusion/flow method layer (this repo)
  ├── processes/            forward (noising) processes: masked, discrete-flow, continuous-flow
  ├── diffusion/            objective trainers (MDLM, BD3LM), collators, GRPO
  ├── post_training/        on-policy distillation (OPD): rollouts, teacher divergence
  ├── models/
  │   ├── backbones/        native diffusion backbones (LLaDA, Dream, ModernBERT-diffusion,
  │   │                     DiffusionGemma, MDLM-DiT + mdlm-owt checkpoint conversion)
  │   ├── conversion/       AR→Diffusion methods (Tiny-A2D; PreDiff-style hybrid attention)
  │   ├── integrations/     per-family loading / PEFT / capability registry
  │   ├── latent/           latent & continuous track (FlowLM, LaDiff codec/prior, MeanFlow)
  │   └── generation/       algorithm registry — families: masked_discrete (mdlm /
  │                         block_decode / bd3lm), canvas (block_ar), continuous_flow
  │                         (flowlm), latent_guided (ladiff), discrete_flow (dfm)
  ├── kernels/              Triton kernels, fast LoRA paths, sparse masked LM-head loss
  ├── eval/                 canonical generation metrics (MAUVE + collapse guards),
  │                         smoke evaluators, lm-evaluation-harness adapter
  └── cli/                  unturtle CLI (train / generate / export / eval)
```

Inference acceleration (KV-cache block decode, Triton fast paths) is one
important axis of the layer — it is not the project's sole reason to exist.

## Research state

The research roadmap and its evidence — positive results, negative
gate outcomes, and deliberately-unpromoted mechanisms alike — are tracked in
[`docs/dllm-gap-map.md`](docs/dllm-gap-map.md). Highlights: PreDiff-style
hybrid AR→diffusion conversion is landed with topology-matched evidence on
both masked NLL and free-generation MAUVE; DFM/FS-DFM and the latent (LaDiff)
track are implemented end-to-end but remain research-only after decision-grade
gates returned negative/undecidable results at the tested budgets.

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
