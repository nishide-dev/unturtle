# Generate API Unification — Design

**Date:** 2026-06-11
**Status:** Approved (design)
**Scope:** Single PR. Precursor sub-project to the DiffusionGemma backbone addition.

## Goal

Unify the dLLM inference entry point on the transformers-standard `model.generate()`.
`diffusion_generate` is **removed entirely** (the repository is being rebuilt, so no
backward-compatibility alias is kept). Diffusion generation becomes the default behavior
of `generate()`.

This aligns Unturtle with the HF ecosystem and matches the convention DiffusionGemma adopts
upstream (`model.generate()` returns diffusion output), making the subsequent DiffusionGemma
backbone addition clean.

## Current structure

```
FastDiffusionModel.generate(model, inputs, algorithm="auto", **kw)   # static helper
  └─ resolve_algorithm() → algorithm_to_flags() → model.diffusion_generate(**flags)

MaskedDiffusionGenerationMixin.diffusion_generate(...)   # shared infra (LLaDA / TinyA2D)
DreamGenerationMixin.diffusion_generate(...)             # Dream-specific
LLaDAModelLM.generate(...) → self.diffusion_generate(...)  # already redirects
```

Inheritance:

```
MaskedDiffusionGenerationMixin (defines diffusion_generate)
  ├─ LLaDAGenerationMixin (+ BlockDecodeMixin) → LLaDAModelLM (generate→diffusion_generate redirect)
  ├─ MaskedDiffusionBlockGenerationMixin → TinyA2DGenerationMixin → TinyA2D{Llama,Qwen2,Qwen3}
  └─ DreamGenerationMixin (BlockDecodeMixin only, own diffusion_generate)
```

TinyA2D additionally inherits `transformers.{Llama,Qwen}ForCausalLM`, so it has a working
**AR `generate()`** — this is why `diffusion_generate` historically existed under a separate
name.

## Target structure

```
model.generate(inputs, algorithm="auto", **kw)          # sole generation entry (diffusion default)
  ├─ pure dLLM (LLaDA/Dream): generate = diffusion dispatch
  └─ TinyA2D: generate = diffusion default; algorithm="ar" → transformers AR generate

FastDiffusionModel.generate(model, inputs, algorithm=...)  # thin facade → model.generate
```

`diffusion_generate` disappears from infra, backbones, eval, and tests.

## Section 2 — `generate()` signature and algorithm dispatch

The shared infra method `diffusion_generate` on `MaskedDiffusionGenerationMixin` is
**renamed to `generate`** and accepts `algorithm` as a first-class keyword-only parameter.
This base `generate` handles only diffusion paths — it has **no `ar` branch**, because the
pure-dLLM backbones that inherit it (LLaDA/Dream) do not have a `transformers.GenerationMixin`
in their MRO (verified: `LLaDAModelLM.__mro__` reaches `PreTrainedModel` directly, no
`GenerationMixin`), so `super().generate()` would `AttributeError`. The `ar` branch lives
only on the TinyA2D mixin (Section 3).

```python
def generate(
    self,
    inputs=None,
    *,
    algorithm="auto",          # "auto"|"mdlm"|"block_decode"|"bd3lm"
    generation_config=None,
    **kwargs,
):
    resolved = resolve_algorithm(algorithm, self, bd3lm_requested=...)
    # resolve_algorithm raises ValueError if algorithm=="ar" and the model is
    # not AR-capable, so "ar" never reaches this diffusion-only path.
    flags = algorithm_to_flags(resolved)   # use_cache / use_block_diffusion
    # then: the existing diffusion_generate body (_prepare_*, _sample) unchanged
```

`bd3lm_requested` is computed exactly as the current facade does it:
`bool(kwargs.get("use_block_diffusion", False)) or algorithm == "bd3lm"`.

The algorithm→flags resolution currently living in `FastDiffusionModel.generate` moves
**down into the model's `generate()`**. `sampler.py`'s `resolve_algorithm` /
`algorithm_to_flags` are reused as-is.

`sampler.py` gains the `"ar"` concept: `resolve_algorithm` returns `"ar"` verbatim and the
backbone `generate()` branches before `algorithm_to_flags` is ever called for `"ar"`.

`FastDiffusionModel.generate` becomes a simple forwarder:

```python
@staticmethod
def generate(model, inputs=None, *, algorithm="auto", **kwargs):
    if not callable(getattr(model, "generate", None)):
        raise TypeError(...)
    return model.generate(inputs, algorithm=algorithm, **kwargs)
```

`resolve_algorithm` / `algorithm_to_flags` imports leave the facade and move to the model layer.

### Backbone behavior

| backbone | `generate` behavior |
|---|---|
| LLaDA | Remove the `generate→diffusion_generate` redirect. Inherit `MaskedDiffusionGenerationMixin.generate` (diffusion default). |
| Dream | Rename `DreamGenerationMixin.diffusion_generate` → `generate`. Add `algorithm` acceptance. |
| TinyA2D | `MaskedDiffusionBlockGenerationMixin.generate` (diffusion default). `algorithm="ar"` → `super().generate()` (transformers AR). MRO puts the mixin first, so diffusion wins by default. |

## Section 3 — TinyA2D AR fallback and MRO

The verified TinyA2D Llama MRO (with each class that *defines* `generate`):

```
TinyA2DLlamaLMHeadModel              (no generate)
TinyA2DGenerationMixin               (no generate)
MaskedDiffusionBlockGenerationMixin  (defines generate — Section 3 override, added here)
BlockDecodeMixin                     (no generate)
MaskedDiffusionGenerationMixin       (defines generate — Section 2 base, added here)
LlamaForCausalLM                     (no generate)
LlamaPreTrainedModel / PreTrainedModel / nn.Module / ...
GenerationMixin                      (defines generate — transformers AR)   ← last
ContinuousMixin / object
```

Today, **only `GenerationMixin.generate` exists** (the diffusion entry is `diffusion_generate`,
a different name, so there is no `generate` collision yet). After this change two new `generate`
definitions are inserted *before* `GenerationMixin` in the MRO, so by default a diffusion
`generate` wins — exactly the desired behavior.

`MaskedDiffusionBlockGenerationMixin` (the TinyA2D-only mixin) **overrides `generate`** to add
the `algorithm="ar"` branch:

```python
def generate(self, inputs=None, *, algorithm="auto", generation_config=None, **kwargs):
    if algorithm == "ar":
        # super() walks the MRO past MaskedDiffusionGenerationMixin and LlamaForCausalLM
        # (neither defines generate) down to transformers GenerationMixin.generate (AR).
        return super().generate(inputs, generation_config=generation_config, **kwargs)
    # delegate all diffusion algorithms to the base mixin's generate (Section 2),
    # called explicitly by class so we skip GenerationMixin entirely.
    return MaskedDiffusionGenerationMixin.generate(
        self, inputs, algorithm=algorithm, generation_config=generation_config, **kwargs
    )
```

Why the two call styles differ:

- **`ar` path uses `super()`** — verified to reach `GenerationMixin.generate` (transformers AR),
  since no intervening MRO class defines `generate`.
- **diffusion path calls `MaskedDiffusionGenerationMixin.generate` explicitly by class** — a
  bare `super()` from `MaskedDiffusionBlockGenerationMixin` would *also* land on the base
  diffusion mixin (correct), but routing the `algorithm` kwarg through it explicitly keeps the
  dispatch obvious and independent of future MRO insertions. The AR regression test pins both
  paths.

### Where `algorithm="ar"` is dispatched vs. rejected

The `ar` value is handled at two different layers depending on the backbone:

- **TinyA2D** inherits `MaskedDiffusionBlockGenerationMixin.generate`, whose `ar` branch fires
  *before* `resolve_algorithm` is consulted. So for TinyA2D, `resolve_algorithm` only ever sees
  diffusion algorithm names.
- **LLaDA/Dream** do *not* inherit the TinyA2D mixin (no `ar` branch). Their `generate` is the
  base `MaskedDiffusionGenerationMixin.generate`, which calls `resolve_algorithm(algorithm, self, ...)`
  directly. There, `resolve_algorithm` raises `ValueError`
  ("this model does not support autoregressive generation") when `algorithm == "ar"` and the
  model is not AR-capable.

This keeps a single source of truth: `resolve_algorithm` rejects `ar` for non-AR-capable models;
the TinyA2D mixin short-circuits `ar` before reaching it.

### `_supports_ar`

Added to `sampler.py`, **model_type based** (TinyA2D family → AR-capable). model_type checking
is preferred over MRO traversal for robustness. `resolve_algorithm` uses it: `algorithm == "ar"`
with `_supports_ar(model) is False` → `ValueError`; otherwise `"ar"` is returned verbatim (the
caller's `ar` branch will have already handled it for AR-capable models).

## Section 4 — eval / facade / caller re-wiring

### eval layer

`getattr(model, "diffusion_generate", None)` existence checks (`eval/generation.py:108`,
`eval/gsm8k.py:98`) become direct `model.generate(...)` calls. All dLLMs expose `generate`, so
the existence check is dropped.

### harness adapter

`eval/harness` (lm-eval) paths that reach `diffusion_generate` are likewise pointed at `generate`.

### Tests (12 files / ~99 sites)

Mechanical replacement: `.diffusion_generate(` → `.generate(`. `FastDiffusionModel.generate`
calls keep their arguments (facade name unchanged). **Assertions (expected outputs) are not
changed** — behavioral contract preserved. After replacement, each file's tests are run to
confirm green.

### Removed symbols

- `MaskedDiffusionGenerationMixin.diffusion_generate` (→ `generate`)
- `DreamGenerationMixin.diffusion_generate` (→ `generate`)
- `LLaDAModelLM.generate` redirect implementation (replaced by mixin inheritance)
- `diffusion_generate` references inside `FastDiffusionModel`

## Section 5 — implementation stages, tests, risks

### Stages (1 PR)

1. **infra**: rename `MaskedDiffusionGenerationMixin.diffusion_generate` → `generate`
   (accept `algorithm` + `ar` branch). Same for `MaskedDiffusionBlockGenerationMixin`. Add
   `_supports_ar` (model_type based) to `sampler.py`.
2. **backbones**: rename Dream, remove LLaDA redirect, TinyA2D inherits automatically.
3. **facade**: simplify `FastDiffusionModel.generate` to a forwarder, tidy imports.
4. **eval**: repoint `generation.py` / `gsm8k.py` / harness `diffusion_generate` references to
   `generate`.
5. **tests**: mechanical replace across 12 files, confirm green per-file.
6. **whole suite**: not-slow suite green (transformers 5.11: 424 tests + replaced sites).

### Test strategy

- Existing ~99 sites become `generate`-routed with **unchanged assertions** (behavioral
  contract).
- New: 1–2 regression tests — TinyA2D produces AR output for `algorithm="ar"`; LLaDA/Dream
  raise `ValueError`.
- Update the `FastDiffusionModel.generate` → `model.generate` forwarding unit test (mock).

### Risks

| Risk | Mitigation |
|---|---|
| `algorithm` kwarg collides with the transformers `generate` signature | dLLM `generate` defines `algorithm` keyword-only; AR path does not forward `algorithm` to `super().generate()`. |
| TinyA2D MRO prioritizes AR generate over diffusion | model_type check + `super().generate()` explicit reach; covered by regression test. |
| Dropping the eval existence check lets a non-dLLM slip in | eval is dLLM-only; type errors surface early. |

### YAGNI

- DiffusionGemma itself is **not** in this PR (next sub-project). This PR is the `generate`
  unification only.
- No backward-compatibility alias (repository being rebuilt).

## Invariants preserved

- Output (token IDs / model output) identical to current behavior. Only the generation entry
  is re-wired.
- Bidirectional attention and MDLM loss semantics untouched.
- TinyA2D's AR generation path is retained via `algorithm="ar"` (pre-conversion behavior not lost).
