# Generate API Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `diffusion_generate` entirely and unify the dLLM inference entry on transformers-standard `model.generate()`, with diffusion as the default behavior and TinyA2D retaining AR via `algorithm="ar"`.

> **Amendment 2026-06-12 (issue #22):** `algorithm="ar"` / `_supports_ar` are **dropped**
> (see the design spec amendment). Deltas: Task 1 ships only the sampler test coverage +
> docstring update (no `_supports_ar`, no `"ar"` branch); Task 3 adds **no** `generate`
> override — only an MRO regression test that TinyA2D `generate` runs diffusion by default;
> Task 5's "ar raises" test asserts the unknown-algorithm `ValueError` instead; Tasks 6/11
> do not document an `"ar"` algorithm choice.

**Architecture:** The algorithm→flags resolution moves out of the `FastDiffusionModel.generate` facade and down into the model's `generate()` method (renamed from `diffusion_generate`). `MaskedDiffusionGenerationMixin.generate` handles diffusion-only paths; `MaskedDiffusionBlockGenerationMixin` (TinyA2D-only) overrides `generate` to add an `algorithm="ar"` branch that delegates to transformers' AR generate. `sampler.py` gains `_supports_ar` and an `"ar"` concept in `resolve_algorithm`. No backward-compat alias is kept (repository is being rebuilt).

**Tech Stack:** Python 3.12, PyTorch, transformers 5.11, pytest. Venv at `.venv/`, run tests with `.venv/bin/python -m pytest`.

---

## File Structure

**Modified:**
- `unturtle/models/generation/sampler.py` — add `"ar"` handling + `_supports_ar`; update module docstring (drop `diffusion_generate` mention)
- `unturtle/models/generation/diffusion_generation_utils.py` — rename `diffusion_generate` → `generate`, accept `algorithm`, resolve flags internally
- `unturtle/models/generation/masked_diffusion_block_mixin.py` — override `generate` with `algorithm="ar"` branch
- `unturtle/models/backbones/dream/generation_utils.py` — rename `diffusion_generate` → `generate`, accept `algorithm`
- `unturtle/models/backbones/llada/modeling_llada.py` — remove the `generate`→`diffusion_generate` redirect (inherit mixin `generate`)
- `unturtle/fast_diffusion_model.py` — simplify `FastDiffusionModel.generate` to a thin forwarder
- `unturtle/eval/generation.py` — call `model.generate` directly
- `unturtle/eval/gsm8k.py` — call `model.generate` directly
- `unturtle/eval/harness/model_adapter.py` — call `model.generate` directly

**Test files (rewrite `diffusion_generate` → `generate`, assertions unchanged unless noted):**
- `tests/test_fast_diffusion_generate.py` — **redesigned** (facade no longer resolves flags; flag-resolution tests move to model `generate`)
- `tests/models/test_llada.py`, `tests/models/test_a2d.py`, `tests/models/test_dream.py`
- `tests/models/test_block_decode.py`, `tests/models/test_parallel_decode.py`, `tests/models/test_block_decode_benchmark.py`
- `tests/diffusion/test_block_diffusion_generator.py`
- `tests/eval/test_evaluators.py`, `tests/eval/test_gsm8k.py`, `tests/eval/test_harness_adapter.py`
- `tests/examples/test_benchmark_a2d_aligned.py`
- **New:** AR regression test for TinyA2D + `ValueError` for pure dLLM (placed in `tests/models/test_a2d.py` and `tests/models/test_llada.py`)

---

## Task 1: Add `"ar"` support and `_supports_ar` to sampler.py

**Files:**
- Modify: `unturtle/models/generation/sampler.py`
- Test: `tests/models/test_sampler.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_sampler.py`:

```python
from __future__ import annotations

import pytest

from unturtle.models.generation.sampler import (
    _supports_ar,
    algorithm_to_flags,
    resolve_algorithm,
)


class _BlockCapable:
    """Stub exposing the block-decode cache hook."""

    def _model_forward_with_cache(self, *a, **k): ...


class _PlainDiffusion:
    """Stub without the cache hook."""


class _FakeConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class _ARModel:
    """Stub whose config model_type marks it AR-capable (TinyA2D family)."""

    def __init__(self, model_type):
        self.config = _FakeConfig(model_type)


@pytest.mark.parametrize(
    "model_type,expected",
    [
        ("tiny-a2d-llama", True),
        ("tiny-a2d-qwen2", True),
        ("tiny-a2d-qwen3", True),
        ("llada", False),
        ("dream", False),
        ("modernbert-diffusion", False),
    ],
)
def test_supports_ar_by_model_type(model_type, expected):
    assert _supports_ar(_ARModel(model_type)) is expected


def test_supports_ar_missing_config_is_false():
    assert _supports_ar(_PlainDiffusion()) is False


def test_resolve_ar_for_ar_capable_returns_ar():
    model = _ARModel("tiny-a2d-llama")
    assert resolve_algorithm("ar", model, bd3lm_requested=False) == "ar"


def test_resolve_ar_for_non_ar_capable_raises():
    model = _ARModel("llada")
    with pytest.raises(ValueError, match="autoregressive"):
        resolve_algorithm("ar", model, bd3lm_requested=False)


def test_resolve_auto_block_decode_still_works():
    assert (
        resolve_algorithm("auto", _BlockCapable(), bd3lm_requested=False)
        == "block_decode"
    )


def test_resolve_auto_mdlm_fallback_still_works():
    assert (
        resolve_algorithm("auto", _PlainDiffusion(), bd3lm_requested=False) == "mdlm"
    )


def test_algorithm_to_flags_unchanged():
    assert algorithm_to_flags("mdlm") == {
        "use_cache": False,
        "use_block_diffusion": False,
    }
    assert algorithm_to_flags("block_decode") == {
        "use_cache": True,
        "use_block_diffusion": False,
    }
    assert algorithm_to_flags("bd3lm") == {
        "use_cache": False,
        "use_block_diffusion": True,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_sampler.py -v`
Expected: FAIL — `ImportError: cannot import name '_supports_ar'`

- [ ] **Step 3: Implement `_supports_ar` and `"ar"` handling in `resolve_algorithm`**

In `unturtle/models/generation/sampler.py`, add the AR-capable model_type set near the top (after `DISCRETE_ALGORITHMS`):

```python
#: model_types whose backbone retains a usable transformers autoregressive generate.
_AR_CAPABLE_MODEL_TYPES: frozenset[str] = frozenset(
    {"tiny-a2d-llama", "tiny-a2d-qwen2", "tiny-a2d-qwen3"}
)


def _supports_ar(model: Any) -> bool:
    """True if the model exposes a usable autoregressive ``generate`` (TinyA2D family)."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    return model_type in _AR_CAPABLE_MODEL_TYPES
```

Update `resolve_algorithm` to accept and validate `"ar"`:

```python
def resolve_algorithm(algorithm: str, model: Any, *, bd3lm_requested: bool) -> str:
    """Resolve ``algorithm`` to a concrete algorithm name.

    ``auto`` picks the fastest discrete path the model supports:
      - BD3LM if requested,
      - else block-decode (Fast-dLLM) when the model supports the cache hook,
      - else plain MDLM.
    ``ar`` is returned verbatim for AR-capable models and raises for others.
    An explicit discrete algorithm name is validated and returned as-is.
    """
    if algorithm == "auto":
        if bd3lm_requested:
            return "bd3lm"
        if _supports_block_decode(model):
            return "block_decode"
        return "mdlm"
    if algorithm == "ar":
        if not _supports_ar(model):
            raise ValueError(
                f"{type(model).__name__} does not support autoregressive "
                "generation (algorithm='ar'); it is a pure diffusion model."
            )
        return "ar"
    if algorithm not in DISCRETE_ALGORITHMS:
        raise ValueError(
            f"Unknown decoding algorithm {algorithm!r}. "
            f"Supported: {sorted(DISCRETE_ALGORITHMS)} (or 'auto'/'ar')."
        )
    return algorithm
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/models/test_sampler.py -v`
Expected: PASS (all 11 cases)

- [ ] **Step 5: Update the module docstring**

In `unturtle/models/generation/sampler.py`, change the docstring line that references `diffusion_generate`:

```python
flag set that the model's ``generate`` dispatch understands — so this is
```

(was: "the existing ``diffusion_generate`` dispatch already understands")

- [ ] **Step 6: Commit**

```bash
git add unturtle/models/generation/sampler.py tests/models/test_sampler.py
git commit -m "✨ feat(generation): add ar algorithm + _supports_ar to sampler"
```

---

## Task 2: Rename `MaskedDiffusionGenerationMixin.diffusion_generate` → `generate` with algorithm dispatch

**Files:**
- Modify: `unturtle/models/generation/diffusion_generation_utils.py:815-879`
- Test: `tests/models/test_a2d.py` (existing diffusion tests exercise this path)

- [ ] **Step 1: Write the failing test**

Add these methods **inside the `TestA2DGeneration` class** in `tests/models/test_a2d.py` (it has `MASK_TOKEN_ID = 999` and fixtures `llama_config` / `llama_model`). The class is at line ~454; insert after `test_has_diffusion_generate`:

```python
    def test_generate_accepts_algorithm_kwarg(self, llama_model):
        B, L_prompt, L_new = 1, 4, 4
        L_total = L_prompt + L_new
        prompt_ids = torch.tensor([[1, 2, 3, 4]])
        mask_fill = torch.full((B, L_new), self.MASK_TOKEN_ID, dtype=torch.long)
        input_ids_full = torch.cat([prompt_ids, mask_fill], dim=1)
        with torch.no_grad():
            out = llama_model.generate(
                input_ids_full,
                algorithm="mdlm",
                steps=3,
                mask_token_id=self.MASK_TOKEN_ID,
                max_length=L_total + 1,
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape == (B, L_total + 1)

    def test_generate_auto_matches_block_decode(self, llama_model):
        B, L_prompt, L_new = 1, 4, 4
        L_total = L_prompt + L_new
        prompt_ids = torch.tensor([[1, 2, 3, 4]])
        mask_fill = torch.full((B, L_new), self.MASK_TOKEN_ID, dtype=torch.long)
        input_ids_full = torch.cat([prompt_ids, mask_fill], dim=1)
        gen = dict(steps=3, mask_token_id=self.MASK_TOKEN_ID, max_length=L_total + 1)

        torch.manual_seed(0)
        out_auto = llama_model.generate(input_ids_full, **gen)
        torch.manual_seed(0)
        out_block = llama_model.generate(input_ids_full, algorithm="block_decode", **gen)

        s_auto = out_auto.sequences if hasattr(out_auto, "sequences") else out_auto
        s_block = out_block.sequences if hasattr(out_block, "sequences") else out_block
        assert torch.equal(s_auto, s_block)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest "tests/models/test_a2d.py::TestA2DGeneration::test_generate_accepts_algorithm_kwarg" -v`
Expected: FAIL — `TinyA2DLlamaLMHeadModel` inherits transformers' AR `generate`, which does not accept the `algorithm` kwarg (TypeError) / does not run the diffusion loop.

- [ ] **Step 3: Rename and add algorithm dispatch in the base mixin**

In `unturtle/models/generation/diffusion_generation_utils.py`, change the method (line ~815):

```python
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        *,
        algorithm: str = "auto",
        generation_config: Optional[MaskedDiffusionGenerationConfig] = None,
        **kwargs,
    ) -> Union[MaskedDiffusionModelOutput, torch.LongTensor]:
        """Generate sequences via masked diffusion.

        ``algorithm`` selects the discrete decoding path ("auto"|"mdlm"|
        "block_decode"|"bd3lm"). The resolved algorithm's flags
        (``use_cache`` / ``use_block_diffusion``) are injected before the
        denoising loop runs. Pure-diffusion backbones do not support
        ``algorithm="ar"`` and will raise ``ValueError`` via ``resolve_algorithm``.
        """
        from unturtle.models.generation.sampler import (
            algorithm_to_flags,
            resolve_algorithm,
        )

        bd3lm_requested = bool(kwargs.get("use_block_diffusion", False)) or (
            algorithm == "bd3lm"
        )
        resolved = resolve_algorithm(algorithm, self, bd3lm_requested=bd3lm_requested)
        # "ar" never reaches this mixin for pure dLLM (resolve_algorithm raises);
        # TinyA2D short-circuits "ar" in its own override before delegating here.
        flags = algorithm_to_flags(resolved)
        kwargs = {**kwargs, **flags}

        generation_config = self._prepare_generation_config(generation_config, **kwargs)

        assert inputs is not None, "`inputs` (input_ids) must be provided"
        input_ids = inputs
        attention_mask = kwargs.pop("attention_mask", None)

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )
        self._validate_generated_length(
            generation_config, input_ids_length, has_default_max_length
        )

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with `input_ids` on a different device type than the model."
                f" `input_ids` is on {input_ids.device.type}, model is on {self.device.type}.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
```

> The `_prepare_generation_config` call already consumes `use_cache`/`use_block_diffusion` from kwargs (they are config fields), so injecting flags into kwargs before it runs is correct and mirrors what the facade previously did.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest "tests/models/test_a2d.py::TestA2DGeneration::test_generate_accepts_algorithm_kwarg" "tests/models/test_a2d.py::TestA2DGeneration::test_generate_auto_matches_block_decode" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add unturtle/models/generation/diffusion_generation_utils.py tests/models/test_a2d.py
git commit -m "♻️ refactor(generation): rename base mixin diffusion_generate to generate with algorithm dispatch"
```

---

## Task 3: TinyA2D `generate` override with `algorithm="ar"` branch

**Files:**
- Modify: `unturtle/models/generation/masked_diffusion_block_mixin.py:68-81` (add `generate` override to the class)
- Test: `tests/models/test_a2d.py`

- [ ] **Step 1: Write the failing test**

Add this method **inside the `TestA2DGeneration` class** in `tests/models/test_a2d.py`:

```python
    def test_generate_ar_delegates_to_transformers(self, llama_model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        # AR path produces input + up to max_new_tokens via transformers GenerationMixin.
        out = llama_model.generate(prompt, algorithm="ar", max_new_tokens=3, do_sample=False)
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape[-1] >= prompt.shape[-1]
        assert seq.shape[-1] <= prompt.shape[-1] + 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest "tests/models/test_a2d.py::TestA2DGeneration::test_generate_ar_delegates_to_transformers" -v`
Expected: FAIL — without the `ar` branch yet, `algorithm="ar"` reaches the base diffusion mixin's `generate`, whose `resolve_algorithm("ar", ...)` raises `ValueError` ("does not support autoregressive").

- [ ] **Step 3: Add `generate` override to the TinyA2D block mixin**

In `unturtle/models/generation/masked_diffusion_block_mixin.py`, add a `generate` method to `MaskedDiffusionBlockGenerationMixin` (after the class docstring, before `_model_forward_with_cache`). Also add the import at the top of the file:

```python
from unturtle.models.generation.diffusion_generation_utils import (
    MaskedDiffusionGenerationMixin,
)
```

```python
    def generate(
        self,
        inputs=None,
        *,
        algorithm: str = "auto",
        generation_config=None,
        **kwargs,
    ):
        """Generate via diffusion (default) or transformers AR (``algorithm="ar"``).

        TinyA2D backbones inherit a usable autoregressive ``generate`` from
        ``transformers.*ForCausalLM``. ``algorithm="ar"`` short-circuits to it
        via ``super().generate()`` (which walks the MRO down to
        ``transformers.GenerationMixin.generate``). All diffusion algorithms are
        routed explicitly to :meth:`MaskedDiffusionGenerationMixin.generate`.
        """
        if algorithm == "ar":
            return super().generate(
                inputs, generation_config=generation_config, **kwargs
            )
        return MaskedDiffusionGenerationMixin.generate(
            self,
            inputs,
            algorithm=algorithm,
            generation_config=generation_config,
            **kwargs,
        )
```

> `super().generate(...)` here resolves through the MRO past `BlockDecodeMixin`, `MaskedDiffusionGenerationMixin`, and `LlamaForCausalLM` (none define `generate`) to `transformers.GenerationMixin.generate`. Verified MRO order during design.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest "tests/models/test_a2d.py::TestA2DGeneration::test_generate_ar_delegates_to_transformers" -v`
Expected: PASS

- [ ] **Step 5: Run the full TinyA2D test file**

Run: `.venv/bin/python -m pytest tests/models/test_a2d.py -v`
Expected: FAIL only on the not-yet-rewritten `diffusion_generate` calls (handled in Task 8). The new `generate`/`ar` tests PASS.

- [ ] **Step 6: Commit**

```bash
git add unturtle/models/generation/masked_diffusion_block_mixin.py tests/models/test_a2d.py
git commit -m "✨ feat(generation): TinyA2D generate override with algorithm=ar AR fallback"
```

---

## Task 4: Rename Dream `diffusion_generate` → `generate` with algorithm dispatch

**Files:**
- Modify: `unturtle/models/backbones/dream/generation_utils.py:364-370`
- Test: `tests/models/test_dream.py`

- [ ] **Step 1: Write the failing test**

Add this method **inside the `TestDreamGenerationUtils` class** in `tests/models/test_dream.py` (it has a `config` fixture and builds `DreamModel(config)` inline per test — mirror that pattern):

```python
    def test_dream_generate_accepts_algorithm(self, config):
        from unturtle.models.backbones.dream import DreamGenerationConfig, DreamModel

        model = DreamModel(config).cpu().eval()
        inputs = torch.tensor([[2, 3, 4, 5]])
        generation_config = DreamGenerationConfig(
            max_new_tokens=4,
            steps=4,
            block_length=2,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
        )
        with torch.no_grad():
            out = model.generate(
                inputs=inputs, algorithm="mdlm", generation_config=generation_config
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape == (1, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest "tests/models/test_dream.py::TestDreamGenerationUtils::test_dream_generate_accepts_algorithm" -v`
Expected: FAIL — `DreamModel.generate` is the transformers AR `generate` (Dream defines only `diffusion_generate`), which does not accept the `algorithm` kwarg (TypeError) and does not run the diffusion loop.

- [ ] **Step 3: Rename Dream method and add algorithm dispatch**

In `unturtle/models/backbones/dream/generation_utils.py`, change the method signature (line ~364) from `diffusion_generate` to `generate` and inject flags. Add the `algorithm` parameter and resolution at the top of the body:

```python
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        *,
        algorithm: str = "auto",
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        from unturtle.models.generation.sampler import (
            algorithm_to_flags,
            resolve_algorithm,
        )

        bd3lm_requested = bool(kwargs.get("use_block_diffusion", False)) or (
            algorithm == "bd3lm"
        )
        resolved = resolve_algorithm(algorithm, self, bd3lm_requested=bd3lm_requested)
        kwargs = {**kwargs, **algorithm_to_flags(resolved)}

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        ...  # REST OF THE EXISTING METHOD BODY UNCHANGED
```

> Keep the entire remaining body identical (from `generation_tokens_hook_func = ...` onward). Only the signature and the inserted algorithm-resolution block at the top change. Dream is a pure dLLM (not AR-capable), so `algorithm="ar"` raises in `resolve_algorithm`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest "tests/models/test_dream.py::TestDreamGenerationUtils::test_dream_generate_accepts_algorithm" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add unturtle/models/backbones/dream/generation_utils.py tests/models/test_dream.py
git commit -m "♻️ refactor(dream): rename diffusion_generate to generate with algorithm dispatch"
```

---

## Task 5: Remove LLaDA `generate`→`diffusion_generate` redirect

**Files:**
- Modify: `unturtle/models/backbones/llada/modeling_llada.py:1963-1977`
- Test: `tests/models/test_llada.py`

- [ ] **Step 1: Write the failing test**

Add these methods **inside the `TestLLaDAGeneration` class** in `tests/models/test_llada.py` (it has a `model` fixture and `TINY_MASK_ID = 511`; the config uses `mask_token_id=511`):

```python
    def test_llada_generate_runs_diffusion(self, model):
        B, L_prompt, L_new = 1, 4, 4
        L_total = L_prompt + L_new
        prompt_ids = torch.tensor([[1, 2, 3, 4]])
        mask_fill = torch.full((B, L_new), self.TINY_MASK_ID, dtype=torch.long)
        input_ids_full = torch.cat([prompt_ids, mask_fill], dim=1)
        with torch.no_grad():
            out = model.generate(
                input_ids_full,
                algorithm="mdlm",
                steps=3,
                mask_token_id=self.TINY_MASK_ID,
                max_length=L_total + 1,
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        assert seq.shape == (B, L_total + 1)

    def test_llada_generate_ar_raises(self, model):
        prompt = torch.tensor([[1, 2, 3, 4]])
        with pytest.raises(ValueError, match="autoregressive"):
            model.generate(prompt, algorithm="ar", max_new_tokens=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest "tests/models/test_llada.py::TestLLaDAGeneration::test_llada_generate_runs_diffusion" "tests/models/test_llada.py::TestLLaDAGeneration::test_llada_generate_ar_raises" -v`
Expected: FAIL — the current `LLaDAModelLM.generate` redirect forwards to `diffusion_generate` (no `algorithm` param), so `algorithm="mdlm"` is an unexpected kwarg and `algorithm="ar"` does not raise the expected `ValueError`.

- [ ] **Step 3: Remove the redirect method**

In `unturtle/models/backbones/llada/modeling_llada.py`, delete the entire `generate` override (lines ~1963-1977):

```python
    def generate(self, inputs=None, generation_config=None, **kwargs):
        """Redirect HF autoregressive ``generate()`` to ``diffusion_generate()``.
        ...
        """
        return self.diffusion_generate(
            inputs, generation_config=generation_config, **kwargs
        )
```

After removal, `LLaDAModelLM` inherits `generate` from `MaskedDiffusionGenerationMixin` (via `LLaDAGenerationMixin`), which now handles `algorithm` and raises `ValueError` for `"ar"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest "tests/models/test_llada.py::TestLLaDAGeneration::test_llada_generate_runs_diffusion" "tests/models/test_llada.py::TestLLaDAGeneration::test_llada_generate_ar_raises" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add unturtle/models/backbones/llada/modeling_llada.py tests/models/test_llada.py
git commit -m "♻️ refactor(llada): drop generate->diffusion_generate redirect, inherit mixin generate"
```

---

## Task 6: Simplify `FastDiffusionModel.generate` to a thin forwarder

**Files:**
- Modify: `unturtle/fast_diffusion_model.py:1211-1258`
- Test: `tests/test_fast_diffusion_generate.py` (redesigned in Task 7)

- [ ] **Step 1: Replace the facade body**

In `unturtle/fast_diffusion_model.py`, replace the `generate` staticmethod (lines ~1211-1258) with:

```python
    @staticmethod
    def generate(
        model: Any,
        inputs: Any = None,
        *,
        algorithm: str = "auto",
        **kwargs: Any,
    ) -> Any:
        """Generate from a dLLM via its unified ``generate`` entry point.

        Thin facade that forwards to ``model.generate(inputs, algorithm=...)``.
        Algorithm resolution (auto/mdlm/block_decode/bd3lm/ar) happens inside the
        model's ``generate``. Output is whatever ``model.generate`` returns.

        Args:
            model: A dLLM model exposing ``generate`` (e.g. from
                ``FastDiffusionModel.from_pretrained``).
            inputs: Prompt token IDs (``[B, L]``).
            algorithm: ``"auto"`` | ``"mdlm"`` | ``"block_decode"`` | ``"bd3lm"`` | ``"ar"``.
            **kwargs: Forwarded to ``model.generate`` / the generation config.

        Returns:
            Whatever ``model.generate`` returns (token IDs or model output).
        """
        if not callable(getattr(model, "generate", None)):
            raise TypeError(
                f"{type(model).__name__} has no `generate` method; "
                "FastDiffusionModel.generate requires a dLLM model."
            )
        return model.generate(inputs, algorithm=algorithm, **kwargs)
```

- [ ] **Step 2: Remove now-unused imports**

In `unturtle/fast_diffusion_model.py`, remove the `resolve_algorithm` / `algorithm_to_flags` imports if they are no longer referenced anywhere else in the file. Verify with:

Run: `grep -n "resolve_algorithm\|algorithm_to_flags" unturtle/fast_diffusion_model.py`
Expected: no matches after the edit (remove the import line if present).

- [ ] **Step 3: Run lint to confirm no unused imports**

Run: `.venv/bin/python -m ruff check unturtle/fast_diffusion_model.py`
Expected: PASS (no F401)

- [ ] **Step 4: Commit**

```bash
git add unturtle/fast_diffusion_model.py
git commit -m "♻️ refactor(fast-diffusion): simplify FastDiffusionModel.generate to thin forwarder"
```

---

## Task 7: Redesign `tests/test_fast_diffusion_generate.py`

**Files:**
- Modify: `tests/test_fast_diffusion_generate.py` (full rewrite)

The facade no longer resolves flags, so the old stub-records-flags tests are invalid. The facade's contract is now: "forward `inputs` + `algorithm` + kwargs to `model.generate`, and raise `TypeError` if the model has no `generate`." Flag-resolution parity is covered by the model-level tests (Tasks 2-5).

- [ ] **Step 1: Rewrite the test file**

Replace the entire contents of `tests/test_fast_diffusion_generate.py` with:

```python
from __future__ import annotations

import pytest
import torch

from unturtle.fast_diffusion_model import FastDiffusionModel


class _RecordingModel:
    """Stub dLLM model: records the args its generate() receives."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, inputs=None, *, algorithm="auto", **kwargs):  # noqa: ANN001, ANN003
        self.calls.append({"inputs": inputs, "algorithm": algorithm, **kwargs})
        return "GENERATED"


def test_facade_forwards_inputs_and_algorithm() -> None:
    model = _RecordingModel()
    out = FastDiffusionModel.generate(model, inputs="X", algorithm="mdlm", steps=8)
    assert out == "GENERATED"
    call = model.calls[-1]
    assert call["inputs"] == "X"
    assert call["algorithm"] == "mdlm"
    assert call["steps"] == 8


def test_facade_default_algorithm_is_auto() -> None:
    model = _RecordingModel()
    FastDiffusionModel.generate(model, inputs="X")
    assert model.calls[-1]["algorithm"] == "auto"


def test_facade_passes_through_gen_kwargs() -> None:
    model = _RecordingModel()
    FastDiffusionModel.generate(
        model, inputs="X", algorithm="block_decode", temperature=0.7, max_new_tokens=32
    )
    call = model.calls[-1]
    assert call["temperature"] == 0.7
    assert call["max_new_tokens"] == 32


def test_facade_requires_generate() -> None:
    class _NotADLLM:
        pass

    with pytest.raises(TypeError, match="generate"):
        FastDiffusionModel.generate(_NotADLLM(), inputs="X")


def _tiny_a2d_model():
    from unturtle.models.conversion.a2d.tiny_a2d import (
        TinyA2DLlamaConfig,
        TinyA2DLlamaLMHeadModel,
    )

    config = TinyA2DLlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        mask_token_id=999,
    )
    model = TinyA2DLlamaLMHeadModel(config)
    model.eval()
    return model


@pytest.mark.parametrize("algorithm", ["mdlm", "block_decode"])
def test_facade_parity_with_direct_generate(algorithm) -> None:
    """Facade output equals calling model.generate directly with the same algorithm."""
    model = _tiny_a2d_model()
    prompt = torch.tensor([[1, 2, 3, 4]])
    gen_kwargs = dict(
        steps=4, max_new_tokens=4, temperature=0.0, mask_token_id=999, block_length=4
    )

    torch.manual_seed(0)
    out_direct = model.generate(inputs=prompt, algorithm=algorithm, **gen_kwargs)

    torch.manual_seed(0)
    out_facade = FastDiffusionModel.generate(
        model, inputs=prompt, algorithm=algorithm, **gen_kwargs
    )

    seq_direct = out_direct.sequences if hasattr(out_direct, "sequences") else out_direct
    seq_facade = out_facade.sequences if hasattr(out_facade, "sequences") else out_facade
    assert torch.equal(seq_direct, seq_facade)
```

- [ ] **Step 2: Run the redesigned test file**

Run: `.venv/bin/python -m pytest tests/test_fast_diffusion_generate.py -v`
Expected: PASS (all cases)

- [ ] **Step 3: Commit**

```bash
git add tests/test_fast_diffusion_generate.py
git commit -m "✅ test(fast-diffusion): redesign facade tests for thin-forwarder contract"
```

---

## Task 8: Rewrite model test files (`diffusion_generate` → `generate`)

**Files:**
- Modify: `tests/models/test_llada.py`, `tests/models/test_a2d.py`, `tests/models/test_dream.py`, `tests/models/test_block_decode.py`, `tests/models/test_parallel_decode.py`, `tests/models/test_block_decode_benchmark.py`, `tests/diffusion/test_block_diffusion_generator.py`

These call `model.diffusion_generate(...)` directly. Replace the method name with `generate`. **Assertions are unchanged.** Where a call passes `use_cache=True` / `use_block_diffusion=True` flags directly, those still work (the model `generate` injects flags, and explicit flags in kwargs are honored by `_prepare_generation_config`). However, calls that pass `use_cache`/`use_block_diffusion` as the *only* selector and rely on default `algorithm="auto"` will still resolve correctly because auto picks block_decode for cache-capable models — verify each file's tests after replacement.

- [ ] **Step 1: Replace `diffusion_generate` references with `generate` in each model test file**

This replaces method-call sites, `callable(...)` assertions, and `hasattr(..., "diffusion_generate")` assertions (e.g. `test_generation_mixin_importable` in test_dream.py checks `hasattr(DreamGenerationMixin, "diffusion_generate")`).

```bash
.venv/bin/python - <<'PY'
import pathlib
files = [
    "tests/models/test_llada.py",
    "tests/models/test_a2d.py",
    "tests/models/test_dream.py",
    "tests/models/test_block_decode.py",
    "tests/models/test_parallel_decode.py",
    "tests/models/test_block_decode_benchmark.py",
    "tests/diffusion/test_block_diffusion_generator.py",
]
for f in files:
    p = pathlib.Path(f)
    s = p.read_text()
    s = s.replace(".diffusion_generate(", ".generate(")
    s = s.replace("callable(model.diffusion_generate)", "callable(model.generate)")
    s = s.replace("callable(llama_model.diffusion_generate)", "callable(llama_model.generate)")
    # hasattr(...Mixin, "diffusion_generate") assertions
    s = s.replace('"diffusion_generate"', '"generate"')
    p.write_text(s)
    print("rewrote", f)
PY
```

- [ ] **Step 2: Fix `test_has_diffusion_generate` assertions**

In `tests/models/test_llada.py` and `tests/models/test_a2d.py`, the existing `test_has_diffusion_generate` tests assert `callable(model.diffusion_generate)`. After the rewrite they assert `callable(model.generate)` (handled by Step 1's extra replacements). Rename these test functions for clarity:

In `tests/models/test_llada.py`: rename `test_has_diffusion_generate` → `test_has_generate`.
In `tests/models/test_a2d.py`: rename `test_has_diffusion_generate` → `test_has_generate`.

Run to find them: `grep -rn "test_has_diffusion_generate" tests/models/`

- [ ] **Step 3: Handle calls passing explicit `use_cache`/`use_block_diffusion`**

Some tests (e.g. `test_block_decode.py`, `test_parallel_decode.py`) call `diffusion_generate(..., use_cache=True)`. After rename these become `generate(..., use_cache=True)`. The model `generate` resolves `algorithm="auto"` → for cache-capable models that yields `block_decode` (use_cache=True), and the explicit `use_cache=True` in kwargs is consistent. For BD3LM tests passing `use_block_diffusion=True`, `bd3lm_requested` is True so auto resolves to bd3lm. No assertion changes needed — verify in Step 4.

- [ ] **Step 4: Run each rewritten file**

Run: `.venv/bin/python -m pytest tests/models/test_llada.py tests/models/test_a2d.py tests/models/test_dream.py tests/models/test_block_decode.py tests/models/test_parallel_decode.py tests/diffusion/test_block_diffusion_generator.py -m "not slow" -v`
Expected: PASS. If any FAIL on flag conflicts, the failing test passed `use_cache=False` while also relying on auto block_decode — set an explicit `algorithm=` in that call to match its original intent (document which in the commit).

- [ ] **Step 5: Run the benchmark test file**

Run: `.venv/bin/python -m pytest tests/models/test_block_decode_benchmark.py -m "not slow" -v`
Expected: PASS (or skip if benchmark-gated)

- [ ] **Step 6: Commit**

```bash
git add tests/models/ tests/diffusion/test_block_diffusion_generator.py
git commit -m "✅ test(models): route generation tests through unified generate()"
```

---

## Task 9: Re-wire eval layer to `model.generate`

**Files:**
- Modify: `unturtle/eval/generation.py:108-113`, `unturtle/eval/gsm8k.py:98-110`, `unturtle/eval/harness/model_adapter.py:115-127`
- Test: `tests/eval/test_evaluators.py`, `tests/eval/test_gsm8k.py`, `tests/eval/test_harness_adapter.py`

- [ ] **Step 1: Update `eval/generation.py`**

Replace the `getattr`-guarded block (lines ~108-113):

```python
        sequences = self.model.generate(prompt_ids, **generation_kwargs)
```

(removes the `diffusion_generate` getattr and the else-branch; all dLLMs now expose `generate`).

- [ ] **Step 2: Update `eval/gsm8k.py`**

Replace the `getattr`-guarded block (lines ~98 onward) with:

```python
        max_length = prompt_len + self.max_new_tokens
        sequences = self.model.generate(
            input_ids,
            max_length=max_length,
            steps=self.num_steps,
            temperature=self.temperature,
        )
```

(removes the `diffusion_generate` getattr, the `if callable` guard, and the warning else-branch).

- [ ] **Step 3: Update `eval/harness/model_adapter.py`**

Replace the `getattr`-guarded block (lines ~115-127). Change the docstring on line 77 from "through ``diffusion_generate``" to "through ``generate``", and the dispatch:

```python
            sequences = self._model.generate(
                ...  # same kwargs as before, formerly passed to diffusion_generate
            )
```

Keep the exact kwargs that were passed to `diffusion_generate` (read lines 117-125 and preserve them verbatim, only changing the method name to `generate`). Remove the `if callable(...)` guard and its `else` (the TypeError-raising branch on lines ~126).

- [ ] **Step 4: Run eval tests**

Run: `.venv/bin/python -m pytest tests/eval/ -m "not slow" -v`
Expected: Some tests still reference `diffusion_generate` in stubs (handled next step). Production-code-driven tests PASS.

- [ ] **Step 5: Rewrite eval test stubs**

In `tests/eval/test_evaluators.py`, `tests/eval/test_gsm8k.py`, `tests/eval/test_harness_adapter.py`, any stub model that defines `diffusion_generate` must define `generate` instead. Replace:

```bash
.venv/bin/python - <<'PY'
import pathlib
files = [
    "tests/eval/test_evaluators.py",
    "tests/eval/test_gsm8k.py",
    "tests/eval/test_harness_adapter.py",
]
for f in files:
    p = pathlib.Path(f)
    s = p.read_text()
    s = s.replace("def diffusion_generate(", "def generate(")
    s = s.replace(".diffusion_generate(", ".generate(")
    p.write_text(s)
    print("rewrote", f)
PY
```

> If a stub's `generate` signature must accept `algorithm` (because the production code now passes it), check whether the eval call sites pass `algorithm`. The eval code in Steps 1-3 does NOT pass `algorithm` (defaults to auto inside the model), so stub `generate(self, inputs=None, **kwargs)` is sufficient. Verify after rewrite.

- [ ] **Step 6: Run eval tests again**

Run: `.venv/bin/python -m pytest tests/eval/ -m "not slow" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add unturtle/eval/ tests/eval/
git commit -m "♻️ refactor(eval): route generation/gsm8k/harness through unified generate()"
```

---

## Task 10: Rewrite remaining example test + full-suite verification

**Files:**
- Modify: `tests/examples/test_benchmark_a2d_aligned.py`
- Verify: whole not-slow suite

- [ ] **Step 1: Rewrite the example test**

Replace `.diffusion_generate(` with `.generate(` in `tests/examples/test_benchmark_a2d_aligned.py`:

```bash
.venv/bin/python - <<'PY'
import pathlib
p = pathlib.Path("tests/examples/test_benchmark_a2d_aligned.py")
s = p.read_text()
s = s.replace(".diffusion_generate(", ".generate(")
p.write_text(s)
print("rewrote", p)
PY
```

- [ ] **Step 2: Confirm no `diffusion_generate` references remain anywhere**

Run: `grep -rn "diffusion_generate" unturtle/ tests/ examples/ benchmarks/`
Expected: NO matches. If any remain (e.g. in docstrings or benchmark scripts), update them to `generate`.

- [ ] **Step 3: Run the focused fast suite**

Run: `.venv/bin/python -m pytest tests/diffusion/ tests/models/ tests/test_fast_diffusion_model.py tests/test_fast_diffusion_generate.py tests/test_e2e_integration.py tests/eval/ -m "not slow" -v`
Expected: PASS

- [ ] **Step 4: Run the full not-slow suite**

Run: `.venv/bin/python -m pytest tests/ -m "not slow" -q`
Expected: PASS (≥ 424 + new tests)

- [ ] **Step 5: Lint and format**

Run: `.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/examples/test_benchmark_a2d_aligned.py
git commit -m "✅ test(examples): route benchmark test through unified generate()"
```

---

## Task 11: Update docs and CLAUDE.md references

**Files:**
- Modify: `CLAUDE.md`, `docs/dllm-gap-map.md` (if they reference `diffusion_generate`)

- [ ] **Step 1: Find doc references**

Run: `grep -rn "diffusion_generate" CLAUDE.md AGENTS.md docs/ README.md 2>/dev/null`

- [ ] **Step 2: Update each reference**

For each match, change `diffusion_generate` to `generate`. In `CLAUDE.md`, the "High-level generation" section describes `FastDiffusionModel.generate` delegating to `model.diffusion_generate(...)` — update it to say it delegates to `model.generate(..., algorithm=...)`. Update the gotcha line "Loss normalization should align with..." region if it mentions `diffusion_generate` (it does not, but verify). Update the algorithm list to include `"ar"`.

Specifically, in `CLAUDE.md` replace:

> It delegates to `model.diffusion_generate(...)` with the flag set the named algorithm implies

with:

> It delegates to `model.generate(..., algorithm=...)`, which resolves the named
> algorithm to its decoding flags internally

And add `"ar"` (TinyA2D autoregressive fallback) to the documented algorithm choices.

- [ ] **Step 3: Verify no stale references**

Run: `grep -rn "diffusion_generate" CLAUDE.md AGENTS.md docs/ README.md 2>/dev/null`
Expected: NO matches (except possibly the design/plan specs themselves, which are historical records — leave those).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/ README.md AGENTS.md 2>/dev/null
git commit -m "📚 docs: update generation references to unified generate() API"
```

---

## Self-Review Notes

- **Spec coverage:** Section 2 (signature/dispatch) → Tasks 2, 6. Section 3 (TinyA2D AR + MRO) → Tasks 1, 3. Section 4 (eval/facade/tests rewiring) → Tasks 6, 8, 9, 10. Section 5 (stages/tests/risks) → all tasks; AR regression + ValueError tests → Tasks 3, 5. `_supports_ar` → Task 1.
- **`ar` dispatch single-source-of-truth:** TinyA2D short-circuits `ar` before `resolve_algorithm` (Task 3); pure dLLM reaches `resolve_algorithm` which raises (Tasks 2, 5). Matches spec.
- **No backward-compat alias:** `diffusion_generate` fully removed; Task 10 Step 2 asserts zero remaining references.
- **Type consistency:** `generate(inputs, *, algorithm="auto", generation_config=None, **kwargs)` used identically in base mixin (Task 2), TinyA2D override (Task 3), Dream (Task 4), and facade forwarder (Task 6).
