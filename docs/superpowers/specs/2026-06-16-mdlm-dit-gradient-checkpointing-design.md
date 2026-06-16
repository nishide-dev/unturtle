# MDLM-DiT gradient checkpointing (transformers-standard) — design

> Status: approved (brainstorming). Next: implementation plan (writing-plans).
> Date: 2026-06-16. Issue: #35.

## Goal

Make the MDLM-DiT backbone support **transformers-standard gradient checkpointing**, so that a
default `DiffusionTrainer.train()` run (unsloth defaults `gradient_checkpointing=True`) works
without raising, and large MDLM-DiT training benefits from activation-memory savings.

## Background

`DiffusionTrainingArguments` defaults `gradient_checkpointing=True` (inherited from unsloth's
`UnslothTrainingArguments`, which flips HF's `False` default). But
`MDLMDiTPreTrainedModel.supports_gradient_checkpointing = False` — the backbone never implemented
the checkpointing machinery (unlike sibling LLaDA/Dream, which set it `True`). Consequently a
default `DiffusionTrainer` run on MDLM-DiT raises immediately at `Trainer.train()`:

```
ValueError: MDLMDiTForMaskedDiffusionLM does not support gradient checkpointing.
```

This was found while verifying #33. Issue #35 originally proposed working around it in
`DiffusionTrainer` (auto-disable when unsupported). **This design supersedes that:** rather than
smoothing the conflict in the trainer, MDLM-DiT itself supports the standard mechanism — the
faithful "follow transformers/unsloth standard" path. The trainer-side auto-disable (approach B)
becomes unnecessary, since the conflict disappears once the model supports GC.

## transformers-standard mechanism (verified)

- `PreTrainedModel.gradient_checkpointing_enable()` only proceeds when
  `supports_gradient_checkpointing=True`; otherwise it raises `ValueError`
  (`transformers/modeling_utils.py:3237`). This is intentional standard behavior.
- It calls `_set_gradient_checkpointing(enable, gradient_checkpointing_func)`, which the base
  `PreTrainedModel` implements by walking submodules and, on each module that owns a
  `gradient_checkpointing` attribute, setting `module.gradient_checkpointing = enable` and
  `module._gradient_checkpointing_func = func`.
- A module's `forward` then wraps each layer with:
  `if self.gradient_checkpointing and self.training: self._gradient_checkpointing_func(layer.__call__, *args)`.

Sibling Dream uses exactly this transformers-5.x-native pattern
(`unturtle/models/backbones/dream/modeling_dream.py:923-936`): a flag `self.gradient_checkpointing`
on the inner `nn.Module`, checked in the layer loop. LLaDA uses a heavier custom
`_set_gradient_checkpointing` + an `ActivationCheckpointingStrategy` enum. **MDLM-DiT adopts the
Dream-style native pattern** — its `DDiTBlock` list is simple, so the plain flag-check loop is the
minimal, standard fit.

## Approach (chosen: A — implement GC on the model)

Make MDLM-DiT a first-class GC-supporting backbone:
1. `MDLMDiTPreTrainedModel.supports_gradient_checkpointing = True`.
2. `MDLMDiTModel` holds `self.gradient_checkpointing = False` and, in `forward`, wraps each
   `DDiTBlock` call with `self._gradient_checkpointing_func` when
   `self.gradient_checkpointing and self.training`.
3. Rely on the base `PreTrainedModel._set_gradient_checkpointing` to propagate the flag + func
   into the inner `MDLMDiTModel` (it owns a `gradient_checkpointing` attribute). **This propagation
   must be verified at implementation time** (see Implementation note).

Rejected: approach B (auto-disable in `DiffusionTrainer`) — it papers over the conflict instead of
following the standard, and would leave MDLM-DiT permanently unable to use GC. Not implemented.

## Components

Single file changed: `unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py`.

### `MDLMDiTPreTrainedModel`

```python
supports_gradient_checkpointing = True   # was False
```
No custom `_set_gradient_checkpointing` hook (Dream doesn't write one either — the base
implementation handles propagation). See the implementation-time verification note below.

### `MDLMDiTModel`

```python
def __init__(self, config: MDLMDiTConfig) -> None:
    super().__init__()
    ...
    self.gradient_checkpointing = False   # toggled by the standard hook

def forward(self, input_ids, attn_bias):
    B, L = input_ids.shape
    x = self.vocab_embed(input_ids)
    c = F.silu(self.cond).unsqueeze(0).expand(B, -1)
    cos, sin = self.rotary(L, input_ids.device)
    for block in self.blocks:
        if self.gradient_checkpointing and self.training:
            x = self._gradient_checkpointing_func(
                block.__call__, x, cos, sin, c, attn_bias
            )
        else:
            x = block(x, cos, sin, c, attn_bias)
    return self.output_layer(x, c)
```

`self._gradient_checkpointing_func` is injected by the base `gradient_checkpointing_enable`; the
model does NOT import `torch.utils.checkpoint` itself or handle `use_reentrant` — all delegated to
the standard func (configurable via `gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)`).

### Implementation note (verification-time branch)

The base `_set_gradient_checkpointing` walks submodules and sets the flag on any module owning a
`gradient_checkpointing` attribute. `MDLMDiTModel` is a plain `nn.Module` child of the
`PreTrainedModel` wrapper, with the attribute — matching Dream's `DreamBaseModel` structure, so
propagation should work. **If, at implementation time, `gradient_checkpointing_enable()` does NOT
set `model.model.gradient_checkpointing = True` (propagation fails),** add a minimal LLaDA-style
`_set_gradient_checkpointing(self, enable, gradient_checkpointing_func=None)` on
`MDLMDiTForMaskedDiffusionLM` that explicitly sets `self.model.gradient_checkpointing = enable` and
`self.model._gradient_checkpointing_func = func`. The propagation test (below) is the gate that
decides whether this branch is needed.

## Scope (YAGNI)

In scope: the model GC support + tests + flipping the e2e trainer test to use the unsloth default
`gradient_checkpointing=True`.

Out of scope: trainer-side auto-disable (approach B — no longer needed); custom
`gradient_checkpointing_kwargs`/`use_reentrant` handling (delegated to the standard func); LLaDA's
`ActivationCheckpointingStrategy` enum (overkill for a flat block list); gap-map changes.

## Testing

`tests/models/test_mdlm_dit.py`, CPU, tiny config, `-m "not slow"`:

| Test | Asserts |
|---|---|
| `test_supports_gradient_checkpointing` | `MDLMDiTForMaskedDiffusionLM.supports_gradient_checkpointing is True` |
| `test_gradient_checkpointing_enable_propagates` | after `model.gradient_checkpointing_enable()`, `model.model.gradient_checkpointing is True` and `_gradient_checkpointing_func` is set (locks the propagation assumption / the verification branch) |
| `test_gradient_checkpointing_forward_backward` | with GC enabled + `train()`, forward→backward runs, loss finite, gradients flow to all params (checkpoint path is numerically sound) |
| `test_gradient_checkpointing_disable` | `gradient_checkpointing_disable()` resets `model.model.gradient_checkpointing` to False |
| `test_gc_output_matches_non_gc` | identical input + weights → GC-enabled vs disabled forward outputs match (`torch.allclose`); checkpoint is numerically transparent |

Plus update the existing `TestMDLMDiTTrainerE2E::test_diffusion_trainer_runs`: change the
`DiffusionTrainingArguments` from `gradient_checkpointing=False` to `gradient_checkpointing=True`
(the unsloth default), locking acceptance criterion #5 — a default-config `DiffusionTrainer.train()`
runs without raising.

## Acceptance criteria

1. `MDLMDiTForMaskedDiffusionLM.supports_gradient_checkpointing is True`.
2. `gradient_checkpointing_enable()` sets the inner `MDLMDiTModel.gradient_checkpointing` to True
   (propagation verified; LLaDA-style hook added only if base propagation fails).
3. GC enabled + train → forward/backward runs with finite gradients.
4. GC-enabled vs disabled forward outputs match (numerical transparency).
5. A default `DiffusionTrainer.train()` (with `gradient_checkpointing=True`) runs without raising.
6. All `tests/models/test_mdlm_dit.py` pass; ruff clean; focused fast suite green.

## Issue / branch / PR

- Update issue #35 body/title from "auto-disable in DiffusionTrainer" to "implement
  transformers-standard gradient checkpointing on MDLM-DiT".
- Branch: `feat/35-mdlm-dit-gradient-checkpointing`.
- 1 PR = 1 issue; Draft PR → pr-review → squash merge.

## Implementation order (TDD)

1. Write the 5 GC tests → RED (supports flag False, no propagation, enable raises).
2. Flip `supports_gradient_checkpointing = True`; add `self.gradient_checkpointing = False` to
   `MDLMDiTModel.__init__`; wrap blocks in `forward` → GREEN.
3. Run `test_gradient_checkpointing_enable_propagates` to verify base propagation; add the
   LLaDA-style explicit hook ONLY if it fails.
4. Flip the e2e trainer test to `gradient_checkpointing=True`; confirm a real
   `DiffusionTrainer.train()` runs.
5. CLAUDE.md: one-line note that MDLM-DiT supports gradient checkpointing (optional; gap-map
   unchanged).
6. ruff + focused fast suite.
7. Draft PR → pr-review → squash merge → close #35.
