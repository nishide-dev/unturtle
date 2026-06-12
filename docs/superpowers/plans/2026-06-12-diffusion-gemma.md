# DiffusionGemma Backbone + FastModel Delegation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delegate non-native model loading to `unsloth.FastModel` (issue #15) and add DiffusionGemma as a wrapped native diffusion backbone with the unified `generate(algorithm=...)` contract plus eval-harness support.

**Architecture:** Two PRs. PR 1 inserts a FastModel attempt between the native-class path and the Auto* fallback chain in `fast_diffusion_model.py`, returning the FastModel tokenizer when available, and adds a post-load class-swap registry. PR 2 adds `unturtle/models/backbones/diffusion_gemma/` (a field-free subclass of upstream `DiffusionGemmaForBlockDiffusion` with a `generate` shim), the `"block_ar"` algorithm in `sampler.py` (capability-checked, no flag injection), registers the class swap, and threads `DecodingConfig.algorithm` through the harness adapter.

**Tech Stack:** Python 3.12, transformers 5.11 (`models/diffusion_gemma`), unsloth `FastModel`, pytest. Venv: `.venv/bin/python`.

**Design spec:** `docs/superpowers/specs/2026-06-12-diffusion-gemma-design.md` (read it first — esp. the "DiffusionGemma is NOT a masked diffusion LM" section).

---

## File Structure

**PR 1 (branch `feat/15-fastmodel-delegation`):**
- Modify: `unturtle/fast_diffusion_model.py` — `_load_via_fastmodel` helper, `_load_model_auto` ordering, tokenizer plumbing, `_POST_LOAD_CLASS_SWAPS` registry
- Modify: `tests/test_fast_diffusion_model.py` — delegation/fallback/tokenizer tests

**PR 2 (branch `feat/<issue>-diffusion-gemma-backbone`):**
- Modify: `unturtle/models/generation/sampler.py` — `"block_ar"`, `_supports_block_ar`, `_supports_mdlm`, auto order
- Create: `unturtle/models/backbones/diffusion_gemma/{__init__.py,modeling.py}`
- Modify: `unturtle/models/backbones/__init__.py` — exports
- Modify: `unturtle/fast_diffusion_model.py` — register class swap
- Modify: `unturtle/eval/harness/configs.py` + `unturtle/eval/harness/model_adapter.py` — `algorithm` field
- Create: `tests/models/test_diffusion_gemma.py`, `tests/test_e2e_diffusion_gemma_real.py` (slow/gpu)
- Modify: `tests/models/test_sampler.py`, `tests/eval/test_harness_adapter.py`, `CLAUDE.md`, `docs/dllm-gap-map.md`

---

## Task 0: Issues + branch (PR 1)

- [ ] **Step 1:** Issue #15 already exists. Create the PR-2 issue now so the plan can reference it:

```bash
gh issue create --label "type: feat" --label "diffusion" \
  --title "[P1] add DiffusionGemma backbone wrapper + block_ar algorithm + harness support" \
  --body "Roadmap item 1. Design: docs/superpowers/specs/2026-06-12-diffusion-gemma-design.md. Depends on #15 (FastModel delegation). Inference-first: wrapper shim with unified generate(algorithm=...), sampler block_ar capability checks, DecodingConfig.algorithm, tiny-config + slow/gpu real-checkpoint tests."
```

Record the issue number — referred to below as `#G`.

- [ ] **Step 2:** `git checkout main && git pull && git checkout -b feat/15-fastmodel-delegation`

---

## Task 1: `_load_via_fastmodel` + delegation ordering (PR 1)

**Files:**
- Modify: `unturtle/fast_diffusion_model.py` (`_load_via_automodel` at ~688-727, `_load_model_auto` at ~730+, `from_pretrained` tokenizer block at ~1008-1014)
- Test: `tests/test_fast_diffusion_model.py`

- [ ] **Step 1: Write the failing tests.** Add to `tests/test_fast_diffusion_model.py` (mirror the file's existing monkeypatch style — read neighboring `TestFromPretrained` tests first):

```python
class TestFastModelDelegation:
    def test_non_native_model_type_delegates_to_fastmodel(self, monkeypatch, tmp_path):
        """A model_type outside the native dict goes through unsloth FastModel."""
        calls = {}

        class _FakeFastModel:
            @staticmethod
            def from_pretrained(model_name, **kwargs):
                calls["model_name"] = model_name
                calls["kwargs"] = kwargs
                return "FM_MODEL", "FM_TOKENIZER"

        from unturtle import fast_diffusion_model as fdm

        monkeypatch.setattr(fdm, "_import_fastmodel", lambda: _FakeFastModel)
        # _load_via_fastmodel returns (model, tokenizer) or None
        out = fdm._load_via_fastmodel("some/hub-model", {"torch_dtype": "bf16"})
        assert out == ("FM_MODEL", "FM_TOKENIZER")
        assert calls["model_name"] == "some/hub-model"

    def test_fastmodel_failure_falls_back_to_automodel(self, monkeypatch):
        """unsloth unavailable / load failure -> Auto* chain still used."""
        from unturtle import fast_diffusion_model as fdm

        def _boom():
            raise ImportError("no unsloth")

        monkeypatch.setattr(fdm, "_import_fastmodel", _boom)
        assert fdm._load_via_fastmodel("x", {}) is None
```

- [ ] **Step 2:** Run: `.venv/bin/python -m pytest tests/test_fast_diffusion_model.py::TestFastModelDelegation -v` — Expected: FAIL (`_import_fastmodel` / `_load_via_fastmodel` undefined).

- [ ] **Step 3: Implement.** In `unturtle/fast_diffusion_model.py`:

```python
def _import_fastmodel() -> Any:
    """Import hook for unsloth FastModel (separate function for testability)."""
    from unsloth import FastModel

    return FastModel


def _load_via_fastmodel(model_name: str, load_kwargs: dict) -> tuple[Any, Any] | None:
    """Load a non-native model_type via ``unsloth.FastModel.from_pretrained``.

    Returns ``(model, tokenizer)`` on success, or ``None`` when unsloth is
    unavailable or the load fails — the caller then falls back to the Auto*
    chain (offline / local-stub paths keep working).
    """
    try:
        fast_model = _import_fastmodel()
    except Exception as exc:  # noqa: BLE001
        _logger.debug("FastDiffusionModel: unsloth FastModel unavailable: %s", exc)
        return None
    try:
        # FastModel handles quantization itself; map our kwargs onto its API.
        fm_kwargs: dict[str, Any] = {}
        if "torch_dtype" in load_kwargs:
            fm_kwargs["dtype"] = load_kwargs["torch_dtype"]
        for key in ("token", "device_map", "trust_remote_code"):
            if key in load_kwargs:
                fm_kwargs[key] = load_kwargs[key]
        fm_kwargs["load_in_4bit"] = "quantization_config" in load_kwargs
        model, tokenizer = fast_model.from_pretrained(model_name, **fm_kwargs)
        return model, tokenizer
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001
        _logger.debug("FastDiffusionModel: FastModel load failed: %s", exc)
        return None
```

Read the actual `FastModel.from_pretrained` signature in the installed unsloth (`.venv/bin/python -c "import inspect, unsloth; print(inspect.signature(unsloth.FastModel.from_pretrained))"`) and reconcile `fm_kwargs` names (e.g. `dtype` vs `torch_dtype`, `load_in_4bit`) — report any deviation.

- [ ] **Step 4: Wire into `_load_model_auto` + tokenizer plumbing.** Read `_load_model_auto` (just after `_load_via_automodel`). Change the orchestration to: native → `_load_via_fastmodel` → `_load_via_automodel`, propagating the FastModel tokenizer:
  - `_load_model_auto` returns `(model, tokenizer_or_none)`.
  - In `from_pretrained` (~line 1004-1014): `model, fm_tokenizer = _load_model_auto(...)`; keep `_patch_for_diffusion(model, max_seq_length)`; then `tokenizer = fm_tokenizer if fm_tokenizer is not None else _load_tokenizer(model_name, trust_remote_code, token)`.
  - Update the `_load_via_automodel` docstring: remove the #15 NOTE (it is now implemented), state it is the offline/unsloth-unavailable fallback.

- [ ] **Step 5: Class-swap registry.** Add near `_native_model_classes`:

```python
#: model_type → callable returning the wrapper class to swap in after a
#: FastModel load (FastModel loads upstream classes; wrappers add only a
#: `generate` shim, so `__class__` swap is safe). Filled by backbone modules.
_POST_LOAD_CLASS_SWAPS: dict[str, Any] = {}


def _apply_post_load_class_swap(model: Any) -> None:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    resolver = _POST_LOAD_CLASS_SWAPS.get(model_type)
    if resolver is None:
        return
    wrapper_cls = resolver()
    if not isinstance(model, wrapper_cls):
        model.__class__ = wrapper_cls
```

Call `_apply_post_load_class_swap(model)` in `from_pretrained` right after `_load_model_auto` returns (before `_patch_for_diffusion`). Registry is empty in PR 1 — add a unit test that a registered fake resolver swaps the class and an unregistered model_type is untouched.

- [ ] **Step 6:** Run: `.venv/bin/python -m pytest tests/test_fast_diffusion_model.py -q` — Expected: all pass (existing 54 + new).
- [ ] **Step 7:** `.venv/bin/python -m ruff check unturtle/fast_diffusion_model.py tests/test_fast_diffusion_model.py && .venv/bin/python -m ruff format --check unturtle/fast_diffusion_model.py tests/test_fast_diffusion_model.py`
- [ ] **Step 8:** Commit: `git add unturtle/fast_diffusion_model.py tests/test_fast_diffusion_model.py && git commit -m "✨ feat(loader): delegate non-native model loading to unsloth FastModel with Auto* fallback (#15)"`

---

## Task 2: PR 1 verification + PR

- [ ] **Step 1:** Full suite: `.venv/bin/python -m pytest tests/ -m "not slow" -q` — Expected: 0 failures (≥563 passed).
- [ ] **Step 2:** `.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .`
- [ ] **Step 3:** Push + PR: `git push -u origin feat/15-fastmodel-delegation`, then `gh pr create` (base main, title `✨ feat(loader): delegate non-native model loading to unsloth FastModel (#15)`, body: summary, fallback contract, tokenizer preference, class-swap registry, test plan). Run the PR review tooling before marking ready; squash-merge after review per repo convention. PR 2 branches from the merged main.

---

## Task 3: sampler `block_ar` + capability checks (PR 2)

**Files:**
- Modify: `unturtle/models/generation/sampler.py`
- Test: `tests/models/test_sampler.py`

- [ ] **Step 0:** `git checkout main && git pull && git checkout -b feat/G-diffusion-gemma-backbone` (use the real `#G` number).

- [ ] **Step 1: Failing tests** in `tests/models/test_sampler.py` (reuse the file's stub style):

```python
class _BlockArCapable:
    """Stub mimicking DiffusionGemmaGenerationMixin capability."""

    def _denoising_step(self, *a, **k): ...


def test_resolve_auto_prefers_block_ar():
    assert (
        resolve_algorithm("auto", _BlockArCapable(), bd3lm_requested=False)
        == "block_ar"
    )


def test_resolve_explicit_block_ar_on_masked_model_raises():
    with pytest.raises(ValueError, match="block_ar"):
        resolve_algorithm("block_ar", _PlainDiffusion(), bd3lm_requested=False)


def test_resolve_explicit_mdlm_on_block_ar_model_raises():
    # block_ar families have no mask semantics -> mdlm is inapplicable.
    with pytest.raises(ValueError, match="masked"):
        resolve_algorithm("mdlm", _BlockArCapable(), bd3lm_requested=False)


def test_algorithm_to_flags_block_ar_is_empty():
    assert algorithm_to_flags("block_ar") == {}
```

- [ ] **Step 2:** Run: `.venv/bin/python -m pytest tests/models/test_sampler.py -v` — Expected: new tests FAIL ("Unknown decoding algorithm 'block_ar'").

- [ ] **Step 3: Implement** in `sampler.py`:
  - `_supports_block_ar(model)`: `callable(getattr(model, "_denoising_step", None))` (DiffusionGemmaGenerationMixin-specific; document the probe choice).
  - `_supports_mdlm(model)`: `callable(getattr(model, "_sample", None))` (the masked-mixin denoising loop; LLaDA/Dream/TinyA2D/ModernBERT all have it, DiffusionGemma does not).
  - Register `"block_ar"` as a known algorithm; `algorithm_to_flags("block_ar")` returns `{}` (no flag injection — the upstream generation config governs itself; document the bd3lm-vs-block_ar distinction: bd3lm = Unturtle masked block diffusion, block_ar = upstream self-conditioned canvas block diffusion).
  - `resolve_algorithm`: in the `auto` branch, check `_supports_block_ar(model)` FIRST (before bd3lm_requested/block_decode/mdlm). For explicit names add capability checks: `"block_ar"` requires `_supports_block_ar`; `"mdlm"` requires `_supports_mdlm` (error message must mention "masked"); keep the existing block_decode/bd3lm checks.

- [ ] **Step 4:** Run: `.venv/bin/python -m pytest tests/models/test_sampler.py -q` — Expected: all pass (existing tests unaffected — `_PlainDiffusion`/`_CacheCapable` stubs need a `_sample` stub method added if `_supports_mdlm` would now fail them; update the stubs, NOT the assertions).
- [ ] **Step 5:** Commit: `git add unturtle/models/generation/sampler.py tests/models/test_sampler.py && git commit -m "✨ feat(generation): block_ar algorithm + mdlm/block_ar capability checks (#G)"`

---

## Task 4: DiffusionGemma wrapper backbone (PR 2)

**Files:**
- Create: `unturtle/models/backbones/diffusion_gemma/__init__.py`, `unturtle/models/backbones/diffusion_gemma/modeling.py`
- Modify: `unturtle/models/backbones/__init__.py`
- Test: `tests/models/test_diffusion_gemma.py` (create)

- [ ] **Step 1: Failing tests.** Create `tests/models/test_diffusion_gemma.py`:

```python
from __future__ import annotations

import pytest
import torch


def _tiny_model():
    from transformers.models.diffusion_gemma import DiffusionGemmaConfig

    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    config = DiffusionGemmaConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        ),
    )
    model = UnturtleDiffusionGemmaForBlockDiffusion(config)
    model.eval()
    return model


def test_generate_auto_runs_block_ar():
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=8)
    seq = out.sequences if hasattr(out, "sequences") else out
    assert seq.shape[0] == 1
    assert seq.shape[-1] >= prompt.shape[-1]


@pytest.mark.parametrize("algorithm", ["mdlm", "block_decode", "bd3lm"])
def test_generate_masked_algorithms_raise(algorithm):
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(ValueError):
        model.generate(prompt, algorithm=algorithm, max_new_tokens=4)


def test_generate_is_shim_not_upstream():
    from transformers.models.diffusion_gemma import DiffusionGemmaGenerationMixin

    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    assert (
        UnturtleDiffusionGemmaForBlockDiffusion.generate
        is not DiffusionGemmaGenerationMixin.generate
    )
```

> The tiny-config kwargs above are best-guess. FIRST verify `DiffusionGemmaTextConfig`'s actual `__init__` parameters (`.venv/bin/python -c "import inspect; from transformers.models.diffusion_gemma import DiffusionGemmaTextConfig; print(inspect.signature(DiffusionGemmaTextConfig.__init__))"`) and the minimal kwargs `DiffusionGemmaForBlockDiffusion(config)` + `generate(max_new_tokens=...)` need on CPU (canvas_length, generation-config defaults like `max_denoising_steps`). Adapt the fixture/kwargs to make a REAL tiny CPU generation run; if a default `generate()` cannot run under ~30s CPU, cap via a small `DiffusionGemmaGenerationConfig(max_denoising_steps=4, max_new_tokens=8)` passed explicitly. Report deviations.

- [ ] **Step 2:** Run: `.venv/bin/python -m pytest tests/models/test_diffusion_gemma.py -v` — Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement** `unturtle/models/backbones/diffusion_gemma/modeling.py`:

```python
"""DiffusionGemma backbone wrapper.

DiffusionGemma is NOT a masked diffusion LM: it denoises a per-block "canvas"
with self-conditioning under entropy/confidence acceptance (no mask token).
This wrapper adds ONLY the unified ``generate(algorithm=...)`` shim — no
masked-diffusion mixins, no config subclass, and the upstream
``model_type = "diffusion_gemma"`` is unchanged (real checkpoints carry it).
The class is deliberately field-free so the loader can ``__class__``-swap a
FastModel-loaded upstream instance (see ``_POST_LOAD_CLASS_SWAPS``).
"""

from __future__ import annotations

from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion


class UnturtleDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion):
    def generate(self, inputs=None, *, algorithm: str = "auto", generation_config=None, **kwargs):
        """Generate via the upstream block-AR canvas diffusion.

        ``algorithm`` accepts ``"auto"`` / ``"block_ar"`` (both delegate to the
        upstream loop verbatim — no vocabulary translation). Masked-diffusion
        algorithms (mdlm / block_decode / bd3lm) raise ``ValueError`` via
        ``resolve_algorithm`` — this family has no mask semantics.
        """
        from unturtle.models.generation.sampler import resolve_algorithm

        resolve_algorithm(algorithm, self, bd3lm_requested=False)  # raises unless block_ar-compatible
        return super().generate(
            input_ids=inputs, generation_config=generation_config, **kwargs
        )
```

`__init__.py` exports the class; register the loader swap (import-light):

```python
from unturtle.models.backbones.diffusion_gemma.modeling import (
    UnturtleDiffusionGemmaForBlockDiffusion,
)

__all__ = ["UnturtleDiffusionGemmaForBlockDiffusion"]
```

Add the export to `unturtle/models/backbones/__init__.py` following the existing pattern (guarded try/except ImportError like the others if that is the file's style — read it first).

- [ ] **Step 4:** Run: `.venv/bin/python -m pytest tests/models/test_diffusion_gemma.py -v` — Expected: PASS.
- [ ] **Step 5:** Commit: `git add unturtle/models/backbones/ tests/models/test_diffusion_gemma.py && git commit -m "✨ feat(backbones): DiffusionGemma wrapper with unified generate shim (#G)"`

---

## Task 5: loader class-swap registration (PR 2)

**Files:**
- Modify: `unturtle/fast_diffusion_model.py` (`_POST_LOAD_CLASS_SWAPS`)
- Test: `tests/models/test_diffusion_gemma.py`, `tests/test_fast_diffusion_model.py`

- [ ] **Step 1: Failing test** (in `tests/models/test_diffusion_gemma.py`):

```python
def test_class_swap_registered_for_diffusion_gemma():
    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    resolver = fdm._POST_LOAD_CLASS_SWAPS.get("diffusion_gemma")
    assert resolver is not None
    assert resolver() is UnturtleDiffusionGemmaForBlockDiffusion


def test_post_load_swap_installs_shim():
    from unturtle import fast_diffusion_model as fdm
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    model = _tiny_upstream_model()  # build via upstream DiffusionGemmaForBlockDiffusion
    fdm._apply_post_load_class_swap(model)
    assert type(model) is UnturtleDiffusionGemmaForBlockDiffusion
```

(`_tiny_upstream_model` mirrors `_tiny_model` but constructs the UPSTREAM class.)

- [ ] **Step 2:** Run + expect FAIL (no registration).
- [ ] **Step 3: Implement.** Register in `fast_diffusion_model.py` next to `_POST_LOAD_CLASS_SWAPS` (lazy import to keep module load light):

```python
def _resolve_diffusion_gemma_wrapper() -> Any:
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    return UnturtleDiffusionGemmaForBlockDiffusion


_POST_LOAD_CLASS_SWAPS["diffusion_gemma"] = _resolve_diffusion_gemma_wrapper
```

- [ ] **Step 4:** Run the two test files; expect PASS. Commit: `git add unturtle/fast_diffusion_model.py tests/ && git commit -m "✨ feat(loader): swap FastModel-loaded diffusion_gemma onto the unturtle wrapper (#G)"`

---

## Task 6: eval — `DecodingConfig.algorithm` (PR 2)

**Files:**
- Modify: `unturtle/eval/harness/configs.py` (~lines 30-77), `unturtle/eval/harness/model_adapter.py` (~lines 114-125)
- Test: `tests/eval/test_harness_adapter.py`

- [ ] **Step 1: Failing tests** in `tests/eval/test_harness_adapter.py`: extend the existing routing test pattern — build the adapter with `algorithm="block_ar"` and assert the recorded call has `algorithm == "block_ar"`, `max_denoising_steps == <num_steps>`, `max_new_tokens` set, and does NOT contain `steps`/`mask_token_id`/`temperature`. Also assert the default (masked) path still records `algorithm == "mdlm"` with `steps`/`temperature`/`mask_token_id` (existing assertions stay).

- [ ] **Step 2:** Run + expect FAIL (adapter has no algorithm parameter).

- [ ] **Step 3: Implement.**
  - `configs.py`: add `algorithm: str = "mdlm"` to `DecodingConfig`; update the NOTE comment (the field now exists — recorded with scores). Add the new entry:

```python
    ("diffusion_gemma", "gsm8k"): DecodingConfig(
        model_family="diffusion_gemma",
        task="gsm8k",
        max_new_tokens=256,
        num_steps=48,
        temperature=0.0,
        use_chat_template=True,
        fewshot=0,
        algorithm="block_ar",
    ),
```

(`num_steps=48` mirrors the published `max_denoising_steps: 48` default; `temperature` is recorded but NOT forwarded on the block_ar path.)
  - `model_adapter.py`: thread `algorithm` from the config into the adapter (constructor param like `num_steps`); replace the pinned `algorithm="mdlm"` call with per-algorithm kwargs:

```python
        if self._algorithm == "block_ar":
            sequences = self._model.generate(
                input_ids,
                algorithm="block_ar",
                max_new_tokens=self._max_new_tokens,
                max_denoising_steps=self._num_steps,
            )
        else:
            sequences = self._model.generate(
                input_ids,
                algorithm=self._algorithm,
                max_length=max_length,
                mask_token_id=mask_token_id,
                steps=self._num_steps,
                temperature=self._temperature,
            )
```

Read the actual adapter code first and preserve its existing kwarg derivation (max_length, mask_token_id lookup) on the masked branch; `build_harness_lm`/runner must pass `algorithm=config.algorithm` through — follow how `num_steps` flows today. The block_ar branch result handling: upstream returns `DiffusionGemmaGenerationOutput` — unwrap `.sequences` if present (mirror the masked branch's handling).

- [ ] **Step 4:** Run: `.venv/bin/python -m pytest tests/eval/ -m "not slow" -q` — Expected: all pass.
- [ ] **Step 5:** Commit: `git add unturtle/eval/ tests/eval/ && git commit -m "✨ feat(eval): DecodingConfig.algorithm + block_ar harness path for diffusion_gemma (#G)"`

---

## Task 7: real-checkpoint slow/gpu test (PR 2)

**Files:**
- Create: `tests/test_e2e_diffusion_gemma_real.py`

- [ ] **Step 1:** Create the file (marked, not run in CI):

```python
from __future__ import annotations

import pytest
import torch

CHECKPOINT = "google/diffusiongemma-26B-A4B-it"


@pytest.mark.slow
@pytest.mark.gpu
def test_real_checkpoint_loads_via_fastmodel_and_generates():
    from unturtle.fast_diffusion_model import FastDiffusionModel
    from unturtle.models.backbones.diffusion_gemma import (
        UnturtleDiffusionGemmaForBlockDiffusion,
    )

    model, tokenizer = FastDiffusionModel.from_pretrained(
        CHECKPOINT, load_in_4bit=True
    )
    assert type(model) is UnturtleDiffusionGemmaForBlockDiffusion
    prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids.to(
        model.device
    )
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=16)
    seq = out.sequences if hasattr(out, "sequences") else out
    text = tokenizer.decode(seq[0], skip_special_tokens=True)
    assert len(text) > 0
```

- [ ] **Step 2:** Verify it is deselected by default: `.venv/bin/python -m pytest tests/test_e2e_diffusion_gemma_real.py -m "not slow" -q` → `1 deselected`. If a CUDA GPU with ≥18GB is available in this environment, run it once (`-m "slow and gpu"`) and report; otherwise mark as not-run in the report.
- [ ] **Step 3:** Commit: `git add tests/test_e2e_diffusion_gemma_real.py && git commit -m "✅ test(e2e): slow/gpu real-checkpoint smoke for diffusion_gemma (#G)"`

---

## Task 8: docs + full verification + PR (PR 2)

- [ ] **Step 1: CLAUDE.md.** Update: (a) "Model taxonomy" backbone axis — add diffusion_gemma to the native-backbone list with a one-line "self-conditioned canvas block diffusion (NOT masked; wraps upstream transformers)" note; (b) "High-level generation" — add `"block_ar"` to the algorithm list with the bd3lm-vs-block_ar distinction and the CLI limitation (CLI generate is masked-dLLM-only and raises on DiffusionGemma); (c) repo-map line for `backbones/` (llada / dream / modernbert / diffusion_gemma).
- [ ] **Step 2: `docs/dllm-gap-map.md`.** Mark roadmap item 1 (DiffusionGemma backbone) as done with PR reference; update the gap-map row if one exists.
- [ ] **Step 3:** Full suite: `.venv/bin/python -m pytest tests/ -m "not slow" -q` → 0 failures. Lint: `.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .`
- [ ] **Step 4:** Commit docs: `git add CLAUDE.md docs/dllm-gap-map.md && git commit -m "📚 docs: diffusion_gemma backbone + block_ar algorithm references (#G)"`
- [ ] **Step 5:** Push + `gh pr create` (base main; title `✨ feat(backbones): DiffusionGemma backbone + block_ar algorithm + harness support (#G)`; body: summary, the NOT-masked clarification, class-swap mechanism, DecodingConfig.algorithm, limitations, test plan incl. whether the slow/gpu test was executed). Run the PR review tooling before marking ready; squash-merge.

---

## Self-Review Notes

- **Spec coverage:** PR 1 (delegation/fallback/tokenizer/swap registry) → Tasks 1-2. Wrapper + shim → Task 4. Sampler block_ar/_supports_mdlm → Task 3. Class swap registration → Task 5. Eval algorithm field + entry → Task 6. Two-tier tests → Tasks 4/7. Docs/limitations → Task 8.
- **Verify-and-adapt points** (upstream API may differ from best-guess): FastModel kwargs (Task 1 Step 3), tiny DiffusionGemma config/generation kwargs (Task 4 Step 1), adapter kwarg flow (Task 6 Step 3). Each instructs the implementer to introspect and report deviations rather than guess.
- **Type consistency:** `UnturtleDiffusionGemmaForBlockDiffusion`, `_POST_LOAD_CLASS_SWAPS`, `_apply_post_load_class_swap`, `_load_via_fastmodel`, `_supports_block_ar`, `_supports_mdlm` used identically across tasks.
