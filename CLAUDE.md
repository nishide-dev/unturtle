# Unturtle — operational guide

## Purpose

Unturtle is a dLLM-focused method layer built on top of `unsloth`. Work here should
preserve `unsloth` integration and concentrate on diffusion-specific value:
- masked diffusion loss
- bidirectional attention fast paths
- dLLM generation / block decode / cache behavior
- dLLM training and the canonical evaluation harness

## Layering — who owns what

Unturtle does **not** own the training loop. It composes four layers:

```text
transformers   model implementations + in-model loss / generation primitives
TRL            objective trainers (DPOTrainer, GRPOTrainer, … as peers)
unsloth        hardware-acceleration patches (fast LoRA, Triton kernels) — not a loop
unturtle       dLLM method layer: conversion, objective trainers, inference accel, eval
```

`DiffusionTrainer` sits in the TRL objective-trainer tier — it is a peer of TRL's
`DPOTrainer`, supplying the masked-diffusion objective. The line:

- **Models that carry their own loss / generation** (e.g. DiffusionGemma) train through the
  standard `transformers` / `FastModel` path; `DiffusionTrainer` is **not** used for them.
- **Backbones without a built-in diffusion objective** are trained by `DiffusionTrainer`,
  which supplies the masked-diffusion loss as an objective layer.

## Core invariants

- dLLM attention is bidirectional. Do not reintroduce causal masking; preserve `is_causal=False`.
- Diffusion loss is computed only on masked positions (`labels == -100` elsewhere).
- Loss normalization aligns with `n_maskable`, not `n_masked`, to match MDLM/d1 semantics.
- Canonical imports for this stack are under `unturtle.diffusion` (or `from unturtle import …`).
- Unturtle depends on `unsloth`; do not spend effort trying to fully decouple it.
- For algorithm changes, check the reference implementations in `dev/repos/` before deciding
  behavior is wrong.

## Model taxonomy — orthogonal axes

A concrete dLLM is a point in several independent axes. Place new code on the right axis:

- **Backbone architecture** (`unturtle.models.backbones.{llada,dream,modernbert,diffusion_gemma,mdlm_dit}`):
  native diffusion backbones Unturtle implements. LLaDA/Dream are full from-scratch
  implementations; ModernBERT-diffusion wraps the upstream bidirectional encoder;
  DiffusionGemma wraps the upstream `transformers` implementation — self-conditioned canvas
  block diffusion, NOT masked diffusion (no mask token). MDLM-DiT is a native, time-agnostic
  adaLN-Zero Diffusion Transformer baseline (kuleshov-group/mdlm DiT) trained via
  `DiffusionTrainer`'s SUBS objective. It supports transformers-standard gradient checkpointing.
- **Conversion method** (`unturtle.models.conversion`): how a non-diffusion backbone
  becomes a dLLM — a *method*, not a model. `a2d` is the AR→Diffusion family; the
  implemented recipe is **Tiny-A2D** (`unturtle.models.conversion.a2d.tiny_a2d`, classes
  `TinyA2D*`, model_types `tiny-a2d-{llama,qwen2,qwen3}`). These are thin adapters over
  `transformers` Qwen/Llama backbones.
- **Forward process** (`unturtle.processes`): how a clean batch becomes a noised
  training state. `MaskedDiffusionProcess` is the masked-discrete implementation;
  `ForwardProcess` does not imply one tensor contract for future DFM/continuous
  methods. Applied device-side by the trainer/evaluator, not by the collator.
- **Training objective** (`unturtle.diffusion`): MDLM, BD3LM.
- **Shared infra** (`unturtle.models.generation`): cache, block-decode, and the
  masked-diffusion generation mixin used by all families (neither backbone nor method).

`AutoConfig.register(...)` fires once per model_type.

See `docs/dllm-gap-map.md` for the implemented-vs-missing method map and the roadmap.

## Minimal repo map

```text
.
├── unturtle/
│   ├── diffusion/          # trainer, collator, scheduler, GRPO
│   ├── processes/          # forward (noising) processes — training-state construction
│   ├── kernels/            # Triton kernels / fast LoRA / sparse masked LM-head loss
│   ├── models/
│   │   ├── backbones/      # native diffusion backbones: llada / dream / modernbert / diffusion_gemma / mdlm_dit
│   │   ├── conversion/     # methods: a2d/ (family) → tiny_a2d/ (recipe)
│   │   ├── integrations/   # per-model-family loading / PEFT / capability registry
│   │   └── generation/     # shared infra: cache / block-decode / generation mixins
│   ├── eval/               # smoke evaluators + lm-evaluation-harness adapter
│   ├── utils/              # shared helpers
│   ├── cli/                # unturtle CLI (train / generate / export / eval)
│   └── fast_diffusion_model.py
├── tests/
│   ├── diffusion/
│   ├── models/
│   ├── examples/
│   ├── test_cli_smoke.py
│   ├── test_fast_diffusion_model.py
│   └── test_e2e_*.py
├── benchmarks/             # tracked benchmark scripts and result helpers
├── examples/               # runnable training / inference examples + configs
└── dev/
    ├── repos/              # local cloned reference repos (ignored)
    ├── papers/             # local papers / notes (ignored)
    ├── local/              # local archival notes / exploratory scripts (ignored)
    └── *.md                # local-only design notes (ignored)
```

## Reference implementations

Clone locally when needed:

```bash
mkdir -p dev/repos
git clone https://github.com/dllm-reasoning/d1.git dev/repos/d1
git clone https://github.com/zhziszz/dllm.git dev/repos/dllm
git clone --depth=1 https://github.com/huggingface/transformers.git dev/repos/transformers
git clone --depth=1 https://github.com/NVlabs/Fast-dLLM.git dev/repos/fast-dllm
```

Most important reference files:
- `dev/repos/d1/SFT/sft_trainer.py`
- `dev/repos/d1/diffu-grpo/diffu_grpo_trainer.py`
- `dev/repos/dllm/dllm/core/trainers/mdlm.py`
- `dev/repos/transformers/src/transformers/modeling_utils.py`
- `dev/repos/transformers/src/transformers/integrations/bitsandbytes.py`
- `dev/repos/fast-dllm/dream/model/generation_utils_block.py`
- `dev/repos/fast-dllm/llada/model/modeling_llada.py`

Deep reference material and benchmark archives can live in ignored local docs under `dev/local/`.

## Environment

`./install.sh` is the supported setup path (uv venv + CUDA-matched torch + editable install,
with verification). See the header of `install.sh` for `TORCH_INDEX` / `PYTHON_VERSION`
overrides.

```bash
./install.sh          # base install
./install.sh --eval   # additionally install the lm-eval-harness extra
```

Notes:
- Plain `pip` is **not** supported — use `uv` (torch must be installed before the editable
  install so the CUDA-matched build is preserved; the script handles ordering).
- Unsloth is pinned to `>=2026.6.2`.
- After setup, use `uv run python` (or `.venv/bin/python`) to execute scripts.

## Common commands

```bash
# focused fast tests
uv run python -m pytest tests/diffusion/ tests/models/ tests/utils/ tests/test_fast_diffusion_model.py tests/test_e2e_integration.py -m "not slow" -v

# full suite
uv run python -m pytest tests/ -v

# lint / format
uv run ruff check .
uv run ruff format .
```

## Testing expectations

- All relevant fast tests must pass before opening or updating a PR.
- Triton / Flash changes must be checked on both CPU fallback and CUDA paths.
- Triton kernel correctness must be compared against `F.cross_entropy` or the appropriate
  reference behavior, not just shape checks.
- Real-checkpoint tests should remain `@pytest.mark.slow` and `@pytest.mark.gpu`.
- For generation / cache work, run the targeted model regressions instead of only broad
  smoke tests.

## Evaluation

Two tiers, different guarantees:

- **Smoke / in-loop** (`unturtle.eval` — `GSM8KEvaluator`, `MaskedDiffusionEvaluator`,
  `GenerationEvaluator`): fast local sanity checks during training/CI. NOT authoritative.
- **Canonical benchmark** (`unturtle.eval.harness`, lm-evaluation-harness): the
  authoritative, paper-comparable path. Uses `FastDiffusionModel.from_pretrained` then calls
  `model.generate(..., algorithm="mdlm")` directly in the adapter (no forwarder). Every run
  pins an explicit per-(model_family, task) `DecodingConfig` and records it with the score
  (dLLM scores are highly sensitive to decoding hyperparameters such as max_new_tokens,
  eos-suppression, steps, temperature).

`lm_eval` is an optional dependency (`./install.sh --eval`); `import unturtle.eval`
must not require it (the adapter/runner import `lm_eval` lazily).

## High-level generation

`model.generate(inputs, algorithm="auto", **kwargs)` is the unified dLLM inference entry
(transformers-standard name; diffusion is the default behavior). `algorithm` is explicit:
`"auto"` (default — fastest discrete path the model supports: `"block_ar"` when the model
supports it, else BD3LM when requested via the `use_block_diffusion=True` KWARG, else
block-decode when available, else MDLM), or force `"mdlm"` / `"block_decode"` /
`"bd3lm"` / `"block_ar"`. `"block_ar"` is self-conditioned canvas block diffusion
(DiffusionGemma-style; no mask token); it is distinct from `"bd3lm"`, which is Unturtle's
masked block diffusion (requires a mask token, TinyA2D family). The resolved algorithm's
flags (`use_cache` / `use_block_diffusion`) are injected into kwargs and override
`generation_config` fields — pin `algorithm="mdlm"` explicitly when the no-cache MDLM
path is the intent on a block-decode-capable model. Explicit algorithm choices are
capability-checked and raise `ValueError` immediately for unsupported combinations (e.g.
`"bd3lm"` on DiffusionGemma, or `"block_ar"` on a masked-diffusion model). `FastDiffusionModel.generate(model, inputs, algorithm=..., **kwargs)`
remains as a thin forwarder (unsloth-style facade, behaviorally identical to calling
`model.generate` directly). CLI `generate` is masked-dLLM-only (derives `mdlm` /
`block_decode` from `--use-cache`); it cannot drive DiffusionGemma (no mask token — it
fails at mask-token resolution) — use `model.generate(algorithm="block_ar")` directly for
block-AR inference. Decoding algorithms are registered in
`unturtle/models/generation/sampler.py` (masked loops mdlm/block_decode/bd3lm are
discrete-masked-only; `block_ar` covers the canvas family; the registry is open for
future continuous-diffusion algorithms).

## Issue, branch, commit, PR workflow

### Issues

Create an issue before implementation.

Title pattern:

```text
[Phase N] <verb> <target>
```

Issue body should include:
- background / goal
- acceptance criteria
- links to relevant issue context, local notes if useful, or reference code

Typical labels:
- `type: feat`, `type: fix`, `type: docs`, `type: test`, `type: perf`, `type: refactor`, `type: chore`
- `diffusion`, `triton`

### Branches

```text
<type>/<issue-number>-<short-description>
```

Examples:
- `feat/42-masked-diffusion-loss`
- `fix/55-collator-masking-bug`
- `docs/124-slim-claude-md`

Rules:
- do not push directly to `main`
- upstream sync should keep history explicit

### Commits

Format:

```text
<emoji> <type>(<scope>): <description> (#<issue>)
```

Common types:
- `✨ feat`
- `🐛 fix`
- `📚 docs`
- `✅ test`
- `⚡ perf`
- `♻️ refactor`
- `🔧 chore`

### Pull requests

- Prefer 1 PR = 1 issue.
- Use Draft PRs while work is in progress.
- Reconcile with `main` before merge.
- Default merge strategy is **Squash and merge**.
- If docs or workflow changed, update the relevant tracked docs in the same PR.

## PR review process

Before marking a PR ready or merging it, run the current PR review tooling.

Use the repo's review agents / PR review toolkit for review.
Focus review on:
1. reference implementation alignment
2. transformers API compatibility
3. CUDA guards on Triton / Flash paths
4. preservation of bidirectional attention
5. packed-sequence / varlen correctness
6. meaningful regression coverage

Fix all critical/high findings before merge.

## Mandatory review checks

For any dLLM algorithm or generation change, explicitly compare against:
- `d1` for SFT / Diffu-GRPO behavior
- `dllm` for MDLM / LLaDA behavior
- `transformers` for init / tie-weights / quantization compatibility
- `Fast-dLLM` for KV-cache / block-decode / replace-position behavior

Do not call a difference a bug until you verify whether it was an intentional design choice.

## Model-specific path reminders

### A2D

```text
PeftModel
 └─ base_model.model.model.layers[i]
     ├─ self_attn.{q_proj,k_proj,v_proj,o_proj}
     └─ mlp.{gate_proj,up_proj,down_proj}
```

### LLaDA

```text
PeftModel
 └─ base_model.model
     └─ model
         └─ transformer.blocks[i]
```

Use runtime fallback when resolving the transformer path:

```python
inner = model.base_model.model
transformer = (
    inner.model.transformer
    if hasattr(inner, "model") and hasattr(inner.model, "transformer")
    else inner.transformer
)
```

### Dream

```text
PeftModel
 └─ base_model.model.model.layers[i]
     ├─ self_attn.{q_proj,k_proj,v_proj}  # bias=True
     ├─ self_attn.o_proj
     └─ mlp.{gate_proj,up_proj,down_proj}
```

Dream q/k/v uses bias, so standard bias-free QKV patching rules do not apply.

## Fast-path patching rules

- Guard on the actual parameter device, not just `torch.cuda.is_available()`.
- Skip Triton patching on CPU.
- Standard `apply_lora_qkv` requires `bias=False`.
- Dream q/k/v requires `apply_lora_qkv_with_bias`.
- `lora_dropout != 0` disables the Triton LoRA path.
- Preserve bidirectional attention fast-forward behavior even when LoRA is absent.

## High-signal gotchas

- Adding a model family means a `BackboneIntegration` registration in
  `unturtle/models/integrations/registry.py`, not a new `elif model_type` branch.
  Registrations are declared centrally there rather than by backbones self-registering:
  `models/backbones/__init__` is eager, so self-registration would create a
  `fast_diffusion_model → backbones → registry → fast_diffusion_model` cycle.
- Integration resolvers are zero-arg callables so the registry imports no backbone.
  Keep it that way; eager class references reintroduce the import cost and lose the
  per-family `except ImportError` degradation.
- A family's PEFT `model_type`s are NOT its load `model_type`s: a PEFT-wrapped Tiny-A2D
  model reports plain `llama`/`qwen2`/`qwen3`, and ModernBERT is patchable without being
  natively loadable.
- On a `PeftModel`, `.model` is the *LM-head model*, not the backbone. Use `get_decoder()`
  to reach hidden states, or the output head runs anyway and "hidden states" are logits.
- The sparse masked LM-head path (`unturtle.kernels.sparse_masked_loss`) saves memory only
  below roughly a 40% mask ratio; MDLM-style `t ~ U(0,1)` averages ~50%, where it costs
  more than dense. Never upcast `[M, V]` logits — the dense Triton kernel upcasts per tile
  and never materializes fp32 logits.
- `MaskedDiffusionDataCollator` defaults to `noise=True` (legacy in-collator corruption).
  `DiffusionTrainer` / `BlockDiffusionTrainer` / `MaskedDiffusionEvaluator` inject it with
  `noise=False` and apply the process device-side; the packed collator still noises. A
  batch carrying only one of `diffusion_mask`/`timesteps` is rejected, not guessed at.
- Pass `processing_class=tokenizer` explicitly to `DiffusionTrainer` in tests/custom setups.
- Do not confuse `packed_seq_lengths` with `cu_seqlens`; packed fast paths read `packed_seq_lengths`.
- Flash varlen must guard on `Q.device.type == "cuda"`; package availability alone is not enough.
- `build_sdpa_packed_attention_mask()` is causal and must not be reused for dLLM packed attention.
- For real checkpoints, `mask_token_id` may need to come from `model.config`, not the tokenizer.
- `LLaDAModelLM` must preserve HF compatibility requirements such as `post_init()`, `tie_weights(**kwargs)`, and tolerant `forward(..., **kwargs)` behavior.
- `load_in_4bit=True` should usually pair with `device_map="auto"`.
- Loss normalization should align with `n_maskable`, not `n_masked`, to match MDLM/d1 reference semantics.
- A2D CUDA RoPE must not re-index pre-aligned `position_embeddings` with `position_ids`.
- Use `unturtle.diffusion` for trainers and collators; do not rely on non-existent third-party diffusion package paths.

## Local archival notes

The following are intentionally **not** kept inline here:
- long benchmark result tables
- historical phase summaries
- extended troubleshooting writeups
- one-off investigation notes

Put that material in ignored local docs under `dev/local/` when it is useful for future work.
