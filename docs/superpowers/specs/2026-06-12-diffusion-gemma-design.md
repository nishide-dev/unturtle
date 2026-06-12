# DiffusionGemma Backbone + FastModel Delegation — Design

**Date:** 2026-06-12
**Status:** Approved (design)
**Scope:** Two PRs. Roadmap item 1 after the clean migration; consumes the generate-API
unification (#22 / PR #23). Inference-first: training-path verification and GGUF export are
explicitly out of scope.

## Goal

1. **PR 1 (issue #15):** delegate loading of model_types outside the native dict to
   `unsloth.FastModel.from_pretrained`, so HF-registered dLLM backbones receive unsloth's
   load / quantization / patch chain.
2. **PR 2 (new issue):** add DiffusionGemma as a wrapped native diffusion backbone with the
   unified `generate(inputs, *, algorithm=..., ...)` contract, plus eval-harness support.

## Upstream facts (transformers 5.11, verified empirically)

- `transformers.models.diffusion_gemma` ships `DiffusionGemmaForBlockDiffusion`,
  `DiffusionGemmaGenerationMixin`, `DiffusionGemmaGenerationConfig`, `EntropyBoundSampler`.
- `generate` is defined by `DiffusionGemmaGenerationMixin` (transformers `GenerationMixin`
  is **not** in the MRO — same shape as Unturtle's own mixins). Signature:
  `generate(input_ids=None, past_key_values=None, streamer=None, generation_config=None,
  logits_processor=None, stopping_criteria=None, **kwargs)`.
- Generation-config vocabulary: `max_denoising_steps`, `confidence_threshold`,
  `stability_threshold`, `t_min` / `t_max`, `sampler_config`, `cache_implementation` —
  disjoint from the masked-diffusion vocabulary (`steps`, `mask_token_id`, `block_length`).
- **DiffusionGemma is NOT a masked diffusion LM.** The generation source contains zero
  `mask_token` references and the config has no `mask_token_id`. Mechanism: prompt KV cache
  + per-block "canvas" denoising with self-conditioning, resampling all canvas positions per
  step under entropy/confidence/stability acceptance (`EntropyBoundSampler`). It is a
  self-conditioned canvas block-diffusion family — the first non-masked entrant to the
  algorithm registry ("discrete-masked-only today; open for future" in `sampler.py`).
- Real checkpoint: `google/diffusiongemma-26B-A4B-it` (unsloth GGUF variants exist; GGUF is
  out of scope). unsloth supports DiffusionGemma fine-tuning; loading rides FastModel.
- Checkpoints carry `model_type = "diffusion_gemma"`, so the wrapper must NOT introduce a
  distinct model_type (unlike the ModernBERT-diffusion precedent).

## PR 1 — FastModel delegation (issue #15)

In `unturtle/fast_diffusion_model.py`, replace `_load_via_automodel` (the fallback carrying
the #15 NOTE) with `_load_via_fastmodel`:

- Non-native model_types are loaded via `unsloth.FastModel.from_pretrained(model_name,
  load_in_4bit=..., device_map=..., dtype=...)`; unsloth owns quantization on this path
  (Unturtle's own `BitsAndBytesConfig` logic remains for the native path).
- **Fallback preserved:** if unsloth is unavailable or `FastModel` fails to load (local stub
  paths, offline CI), fall back to the existing `AutoModel → AutoModelForMaskedLM →
  AutoModelForCausalLM` chain. Existing behavioral contracts (54 `from_pretrained` tests)
  stay green.
- **Return contract:** `from_pretrained` keeps returning `(model, tokenizer)`. The
  FastModel-returned tokenizer is preferred on that path; the fallback path keeps the
  current tokenizer logic.
- **Class swap hook:** after a FastModel load, if `config.model_type == "diffusion_gemma"`,
  swap `model.__class__` to the PR-2 wrapper (field-free subclass, `generate` override
  only — safe for `__class__` assignment). Loading via the native dict instead would bypass
  unsloth's chain, defeating #15. (In PR 1 the hook is a registry/no-op; PR 2 fills it.)
- Tests: monkeypatched-FastModel delegation assertion; fallback contract pin; tokenizer
  preference pin.

## PR 2 — backbone wrapper, `block_ar` algorithm, eval

### Wrapper (`unturtle/models/backbones/diffusion_gemma/`)

`UnturtleDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion)`:

- No new fields, no config subclass, no AutoConfig registration (upstream already
  registered; model_type unchanged).
- Does **not** inherit any masked-diffusion mixin (no mask semantics exist on this family).
- `generate(inputs=None, *, algorithm="auto", generation_config=None, **kwargs)` shim:
  resolve via `resolve_algorithm`; `"block_ar"` delegates verbatim to
  `super().generate(input_ids=inputs, generation_config=generation_config, **kwargs)` —
  **no vocabulary translation**. `mdlm` / `block_decode` / `bd3lm` raise `ValueError` via
  the sampler capability checks (semantically inapplicable: no mask token).

### Sampler (`unturtle/models/generation/sampler.py`)

- New algorithm `"block_ar"` (upstream self-conditioned canvas block diffusion) —
  deliberately distinct from `"bd3lm"` (Unturtle's masked block diffusion).
- `_supports_block_ar(model)`: probes a `DiffusionGemmaGenerationMixin`-specific method.
- `auto` resolution order: `block_ar` (when supported) → bd3lm-if-requested → block_decode →
  mdlm. Explicit `"block_ar"` on non-supporting models raises (existing capability-check
  pattern).
- `algorithm_to_flags("block_ar")` → `{}` (no `use_cache`/`use_block_diffusion` injection;
  upstream config governs itself).
- Explicit `"mdlm"` gains a capability check so mask-free families (DiffusionGemma) reject
  it with an actionable error.

### Eval

- `DecodingConfig` gains `algorithm: str = "mdlm"` (existing entries unchanged); the
  harness adapter's hardcoded `algorithm="mdlm"` pin becomes `config.algorithm`, recorded
  with scores (this consumes the field deferred in PR #23).
- Adapter builds generation kwargs per algorithm: masked families keep
  `steps`/`temperature`/`mask_token_id`; `block_ar` passes `max_new_tokens` +
  `max_denoising_steps=num_steps` only (both are denoising step budgets — faithful at the
  config-author boundary). Entropy/confidence knobs stay at upstream defaults until
  measurement demands DecodingConfig extension.
- New entry: `("diffusion_gemma", "gsm8k")` with `algorithm="block_ar"`.

### Limitations (documented, not built)

- CLI `generate` derives mdlm/block_decode from `--use-cache` and therefore raises on
  DiffusionGemma; the CLI stays masked-dLLM-only (an `--algorithm` flag is a separate
  issue).
- Training-path verification, GGUF, entropy-knob configs: later milestones.

## Test strategy (two tiers)

- **tiny-config (fast):** minimal `DiffusionGemmaConfig` model built directly —
  `generate(algorithm="auto"/"block_ar")` shape smoke; `mdlm`/`block_decode`/`bd3lm`
  ValueErrors; sampler resolution pins (auto→block_ar; explicit block_ar elsewhere raises);
  class-swap identity pin (`type(model).generate is UnturtleDiffusionGemma....generate`);
  loader integration with monkeypatched FastModel.
- **real checkpoint (`@pytest.mark.slow` + `@pytest.mark.gpu`):**
  `google/diffusiongemma-26B-A4B-it` via FastModel `load_in_4bit=True` — short generation
  smoke + a few harness `("diffusion_gemma", "gsm8k")` samples.

## Risks

| Risk | Mitigation |
|---|---|
| FastModel delegation breaks offline/stub tests | Auto* fallback preserved; contracts pinned |
| `__class__` swap on a subclass with new state | Wrapper is field-free by design; identity pin test |
| upstream `generate(**kwargs)` silently ignores stray kwargs | Shim passes through verbatim; harness passes only mapped kwargs |
| 26B-A4B too heavy for CI | Real-weight tests are slow/gpu-gated; tiny-config tier carries CI |
| `block_ar` vs `bd3lm` naming confusion | Both documented side by side in sampler docstring + CLAUDE.md |
