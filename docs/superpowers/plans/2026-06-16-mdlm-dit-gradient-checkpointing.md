# MDLM-DiT Gradient Checkpointing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MDLM-DiT support transformers-standard gradient checkpointing so a default `DiffusionTrainer.train()` (unsloth default `gradient_checkpointing=True`) runs without raising.

**Architecture:** Flip `MDLMDiTPreTrainedModel.supports_gradient_checkpointing` to `True`, give `MDLMDiTModel` a `self.gradient_checkpointing` flag, and wrap each `DDiTBlock` call in `forward` with the standard `self._gradient_checkpointing_func` when checkpointing is enabled in training. Rely on the base `PreTrainedModel` propagation (Dream-style); add an explicit `_set_gradient_checkpointing` hook only if a test shows base propagation does not reach the inner `MDLMDiTModel`.

**Tech Stack:** PyTorch, transformers `PreTrainedModel.gradient_checkpointing_enable` / `_set_gradient_checkpointing` / `_gradient_checkpointing_func`, pytest (CPU, tiny config).

**Spec:** `docs/superpowers/specs/2026-06-16-mdlm-dit-gradient-checkpointing-design.md`
**Issue:** #35. **Branch:** `feat/35-mdlm-dit-gradient-checkpointing` (already checked out, holds the spec commit).

---

## File Structure

- Modify: `unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py`
  - `MDLMDiTPreTrainedModel`: `supports_gradient_checkpointing = True`; (conditionally) a `_set_gradient_checkpointing` hook.
  - `MDLMDiTModel.__init__`: add `self.gradient_checkpointing = False`.
  - `MDLMDiTModel.forward`: wrap each block with `self._gradient_checkpointing_func` under `gradient_checkpointing and training`.
- Modify: `tests/models/test_mdlm_dit.py`
  - New `TestMDLMDiTGradientCheckpointing` (5 tests).
  - Update `TestMDLMDiTTrainerE2E::test_diffusion_trainer_runs` to use `gradient_checkpointing=True`.
- Modify: `CLAUDE.md` (one-line note that MDLM-DiT supports gradient checkpointing).

The current code (to anchor edits):
- `MDLMDiTModel.__init__` ends at `modeling_mdlm_dit.py:261` (`self.output_layer = DDitFinalLayer(...)`).
- `MDLMDiTModel.forward` is `modeling_mdlm_dit.py:263-272` (the `for block in self.blocks: x = block(x, cos, sin, c, attn_bias)` loop).
- `MDLMDiTPreTrainedModel` is `modeling_mdlm_dit.py:275-279` with `supports_gradient_checkpointing = False`.

---

## Task 1: Model gradient-checkpointing support

**Files:**
- Modify: `unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py`
- Test: `tests/models/test_mdlm_dit.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_mdlm_dit.py` (the `tiny_config` fixture and `_activate_adaln` helper already exist in this file; reuse `tiny_config`):

```python
class TestMDLMDiTGradientCheckpointing:
    def test_supports_gradient_checkpointing(self):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        assert MDLMDiTForMaskedDiffusionLM.supports_gradient_checkpointing is True

    def test_gradient_checkpointing_enable_propagates(self, tiny_config):
        """Standard gradient_checkpointing_enable() must reach the inner MDLMDiTModel."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config)
        assert model.model.gradient_checkpointing is False
        model.gradient_checkpointing_enable()
        assert model.model.gradient_checkpointing is True
        # The standard func must be injected on the inner module.
        assert callable(getattr(model.model, "_gradient_checkpointing_func", None))

    def test_gradient_checkpointing_disable(self, tiny_config):
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        model = MDLMDiTForMaskedDiffusionLM(tiny_config)
        model.gradient_checkpointing_enable()
        assert model.model.gradient_checkpointing is True
        model.gradient_checkpointing_disable()
        assert model.model.gradient_checkpointing is False

    def test_gradient_checkpointing_forward_backward(self, tiny_config):
        """With GC enabled + train(), forward/backward runs with finite grads."""
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        torch.manual_seed(0)
        model = MDLMDiTForMaskedDiffusionLM(tiny_config)
        model.gradient_checkpointing_enable()
        model.train()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))
        out = model(input_ids=input_ids)
        loss = out.logits.float().log_softmax(-1).mean().neg()
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(torch.isfinite(g).all() for g in grads)

    def test_gc_output_matches_non_gc(self, tiny_config):
        """Checkpointing is numerically transparent: same input+weights -> same output.

        Compared in train() mode (GC only activates under self.training) with dropout=0
        (tiny_config sets dropout=0.0), so the forward is deterministic.
        """
        from unturtle.models.backbones.mdlm_dit import MDLMDiTForMaskedDiffusionLM

        torch.manual_seed(0)
        model = MDLMDiTForMaskedDiffusionLM(tiny_config)
        model.train()
        input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8))

        with torch.no_grad():
            ref = model(input_ids=input_ids).logits

        model.gradient_checkpointing_enable()
        with torch.no_grad():
            got = model(input_ids=input_ids).logits

        assert torch.allclose(ref, got, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTGradientCheckpointing -v`
Expected: FAIL — `test_supports_gradient_checkpointing` fails (currently False); `test_gradient_checkpointing_enable_propagates` fails (either `gradient_checkpointing_enable()` raises `ValueError` because `supports_gradient_checkpointing` is False, or `model.model` has no `gradient_checkpointing` attribute).

- [ ] **Step 3: Flip the support flag and add the inner flag**

In `unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py`, change `MDLMDiTPreTrainedModel` (currently at lines 275-279):

```python
class MDLMDiTPreTrainedModel(PreTrainedModel):
    config_class = MDLMDiTConfig
    base_model_prefix = "model"
    _no_split_modules = ["DDiTBlock"]
    supports_gradient_checkpointing = True
```

In `MDLMDiTModel.__init__`, add the flag as the last line of `__init__` (right after `self.output_layer = DDitFinalLayer(...)` at line 261):

```python
        self.output_layer = DDitFinalLayer(dim, config.vocab_size, config.cond_dim)
        # Toggled by PreTrainedModel.gradient_checkpointing_enable() via the standard
        # _set_gradient_checkpointing propagation. self._gradient_checkpointing_func is
        # injected by the same call.
        self.gradient_checkpointing = False
```

- [ ] **Step 4: Wrap the block loop in forward**

Replace the `MDLMDiTModel.forward` block loop (lines 263-272) so each `DDiTBlock` is checkpointed when enabled in training:

```python
    def forward(
        self, input_ids: torch.Tensor, attn_bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.vocab_embed(input_ids)
        c = F.silu(self.cond).unsqueeze(0).expand(B, -1)  # [B, cond_dim]
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

- [ ] **Step 5: Run tests; verify propagation; add explicit hook ONLY if needed**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py::TestMDLMDiTGradientCheckpointing -v`

Expected: PASS (all 5).

If `test_gradient_checkpointing_enable_propagates` still FAILS because `model.model.gradient_checkpointing` is not set to True (base `_set_gradient_checkpointing` did not reach the inner `MDLMDiTModel`), add this explicit hook to `MDLMDiTPreTrainedModel` (mirrors LLaDA's pattern at `unturtle/models/backbones/llada/modeling_llada.py:1368-1401`):

```python
    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func=None
    ) -> None:
        from torch.utils.checkpoint import checkpoint

        if gradient_checkpointing_func is None:
            gradient_checkpointing_func = checkpoint
        # Reach the inner MDLMDiTModel (the wrapper holds it as self.model).
        target = getattr(self, "model", self)
        if isinstance(target, MDLMDiTModel):
            target._gradient_checkpointing_func = gradient_checkpointing_func
            target.gradient_checkpointing = enable
            return
        for module in self.modules():
            if isinstance(module, MDLMDiTModel):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                break
```

Note: define this method AFTER `MDLMDiTModel` is defined OR reference it lazily — `MDLMDiTPreTrainedModel` is currently defined after `MDLMDiTModel` in the file (MDLMDiTModel ends at line 272, MDLMDiTPreTrainedModel starts at 275), so `isinstance(target, MDLMDiTModel)` resolves fine. Re-run the tests after adding the hook; expected PASS.

- [ ] **Step 6: Run the full file + ruff**

Run: `.venv/bin/python -m pytest tests/models/test_mdlm_dit.py -v`
Expected: all prior tests + the 5 new ones PASS.

Run: `.venv/bin/python -m ruff format unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py tests/models/test_mdlm_dit.py`
Run: `.venv/bin/python -m ruff check unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py tests/models/test_mdlm_dit.py`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add unturtle/models/backbones/mdlm_dit/modeling_mdlm_dit.py tests/models/test_mdlm_dit.py
git commit -m "✨ feat(backbones): transformers-standard gradient checkpointing on MDLM-DiT (#35)"
```

---

## Task 2: Flip the E2E trainer test to the unsloth default

**Files:**
- Modify: `tests/models/test_mdlm_dit.py` (the `TestMDLMDiTTrainerE2E::test_diffusion_trainer_runs` method added in #33)

This test currently sets `gradient_checkpointing=False` in its `DiffusionTrainingArguments` (a workaround needed before this feature). Now that MDLM-DiT supports GC, flip it to `True` to lock acceptance criterion #5: a default-config `DiffusionTrainer.train()` runs.

- [ ] **Step 1: Update the test argument**

In `tests/models/test_mdlm_dit.py`, find the `DiffusionTrainingArguments(...)` constructed inside `TestMDLMDiTTrainerE2E::test_diffusion_trainer_runs`. It contains a line `gradient_checkpointing=False`. Change it to:

```python
            gradient_checkpointing=True,  # unsloth default; MDLM-DiT now supports GC (#35)
```

Also add an assertion after `result = trainer.train()` (right before or after the existing finite-loss assertions) that the model actually ran with GC enabled, to give the test teeth:

```python
        # GC was active during training (the default path that previously raised).
        assert model.model.gradient_checkpointing is True
```

(Note: `model` here is the `MDLMDiTForMaskedDiffusionLM` instance the test built. If the test variable is named differently, use that name. The Trainer enables GC on the model it holds; assert on the same instance.)

- [ ] **Step 2: Run the e2e test**

Run: `.venv/bin/python -m pytest "tests/models/test_mdlm_dit.py::TestMDLMDiTTrainerE2E::test_diffusion_trainer_runs" -v`
Expected: PASS — a real `DiffusionTrainer.train()` with `gradient_checkpointing=True` runs end-to-end without the `ValueError`.

If the GC assertion fails because the Trainer wraps/copies the model (e.g. accelerate prepares a new object), drop the `assert model.model.gradient_checkpointing is True` line — the meaningful assertion is that `train()` no longer raises and `result.training_loss` is finite. Do NOT weaken the finite-loss assertion. Document in the commit if you had to drop the GC-state assertion and why.

- [ ] **Step 3: ruff + commit**

Run: `.venv/bin/python -m ruff check tests/models/test_mdlm_dit.py`
Expected: clean.

```bash
git add tests/models/test_mdlm_dit.py
git commit -m "✅ test(backbones): MDLM-DiT e2e trains under default gradient_checkpointing=True (#35)"
```

---

## Task 3: Docs + regression guard

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Note GC support in CLAUDE.md**

In `CLAUDE.md`, find the MDLM-DiT sentence added in #31 (in the "Backbone architecture" bullet of the "Model taxonomy" section): "MDLM-DiT is a native, time-agnostic adaLN-Zero Diffusion Transformer baseline (kuleshov-group/mdlm DiT) trained via `DiffusionTrainer`'s SUBS objective." Append: " It supports transformers-standard gradient checkpointing."

- [ ] **Step 2: Run the focused fast suite (regression guard)**

Run: `.venv/bin/python -m pytest tests/models/ tests/diffusion/ tests/test_fast_diffusion_model.py -m "not slow" -q`
Expected: all PASS — confirms the GC change did not break the backbone, sibling backbones, the trainer, or the loader.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "📚 docs: note MDLM-DiT gradient-checkpointing support (#35)"
```

---

## Task 4: PR + review

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/35-mdlm-dit-gradient-checkpointing
```

- [ ] **Step 2: Open a Draft PR**

```bash
gh pr create --draft \
  --title "✨ feat(backbones): transformers-standard gradient checkpointing on MDLM-DiT (#35)" \
  --body "Implements #35. Makes MDLM-DiT support transformers-standard gradient checkpointing (Dream-style: supports_gradient_checkpointing=True + a gradient_checkpointing flag on MDLMDiTModel checked in the block loop, using the injected _gradient_checkpointing_func). A default DiffusionTrainer.train() (unsloth default gradient_checkpointing=True) now runs without ValueError. See docs/superpowers/specs/2026-06-16-mdlm-dit-gradient-checkpointing-design.md."
```

- [ ] **Step 3: Run pr-review-toolkit**

Run the repo PR review (code-reviewer + pr-test-analyzer). Focus: alignment with the sibling Dream GC pattern, that checkpointing is numerically transparent (the output-match test), that the e2e test genuinely exercises the default-trainer path, and bidirectional attention is unaffected. Fix all critical/high findings.

- [ ] **Step 4: Mark ready + squash merge**

After CI is green and review findings are addressed, mark ready and squash-merge. Confirm #35 closes and the branch is deleted.

---

## Notes for the implementer

- **Always use `.venv/bin/python`**, never `uv run python`.
- CPU-only tests. `tiny_config` (vocab=512, hidden=64, cond=32, layers=2, heads=4, dropout=0.0, mask_token_id=511) and `_activate_adaln` already exist near the top of `tests/models/test_mdlm_dit.py` — reuse them.
- The whole point: `gradient_checkpointing_enable()` must reach the INNER `MDLMDiTModel`. The `test_gradient_checkpointing_enable_propagates` test is the gate that tells you whether the base propagation suffices or the explicit `_set_gradient_checkpointing` hook (Task 1 Step 5) is needed. Do not skip it.
- Reference: sibling `unturtle/models/backbones/dream/modeling_dream.py:923-936` (the native flag-check loop) and `unturtle/models/backbones/llada/modeling_llada.py:1368-1401` (the explicit hook, for the fallback only).
- Do NOT add `gradient_checkpointing_kwargs`/`use_reentrant` handling or LLaDA's `ActivationCheckpointingStrategy` enum (YAGNI — delegated to the standard func).
- Do NOT touch the trainer (no auto-disable; this design supersedes that approach).
```
