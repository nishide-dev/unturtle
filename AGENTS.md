# AGENTS.md — Unturtle Project

> This file is read by OpenAI Codex before performing code review or implementation tasks.
> Keep it concise and actionable. See also: `CLAUDE.md` for full project context.

---

## Project Overview

**Unturtle** is a fork of [unslothai/unsloth](https://github.com/unslothai/unsloth) adding
Triton-optimized training support for **Diffusion Language Models (dLLMs)** such as LLaDA and Dream.

Key differences from standard autoregressive LLM training:
- Tokens are randomly **masked** (not predicted autoregressively)
- Loss is computed only at **masked positions** (`label == -100` elsewhere)
- Attention is **bidirectional** (`is_causal=False`, no causal mask)
- Timestep `t ∈ (0, 1]` controls mask rate; loss may be weighted by `1/t` (d1-style)

### Package Structure

```
unturtle/                     # canonical public API (Phase B+)
├── __init__.py               # re-exports unsloth.* + dLLM symbols
├── fast_diffusion_model.py   # FastDiffusionModel (analogous to FastLanguageModel)
├── kernels/
│   ├── masked_diffusion_loss.py   # core dLLM loss (reuses Fast_CrossEntropyLoss)
│   └── fast_lora.py               # LoRA_QKV_Bias kernel for bias-aware QKV
├── diffusion/
│   ├── trainer.py            # DiffusionTrainer (DiffusionTrainingArguments)
│   ├── collator.py           # MaskedDiffusionDataCollator
│   ├── schedulers.py         # LinearAlphaScheduler, CosineAlphaScheduler
│   └── grpo_trainer.py       # DiffuGRPOTrainer
└── models/
    ├── a2d/                  # Autoregressive→Diffusion adapters (LLaMA/Qwen2/Qwen3)
    ├── llada/                # LLaDA native dLLM (non-standard block structure)
    └── dream/                # Dream dLLM (bias=True on q/k/v_proj)

unsloth/                      # upstream + backward-compat shims
tests/
├── diffusion/                # loss / scheduler / collator / GRPO tests (98 tests)
│   ├── test_packed_collator.py   # PackedMaskedDiffusionDataCollator (23 tests)
│   └── ...                       # other diffusion tests
├── models/                   # A2D / LLaDA / Dream model tests (39 tests)
│   └── test_a2d.py           # includes packed forward + flash varlen compaction (23 tests)
├── test_fast_diffusion_model.py  # LoRA patching + save/load (23 tests)
├── test_e2e_integration.py   # fast CPU E2E tests (2 tests)
└── test_e2e_real_checkpoint.py  # slow GPU + real-checkpoint E2E tests (4 tests)
dev/
├── repos/                    # reference implementations (gitignore'd, clone manually)
│   ├── d1/                   # dllm-reasoning/d1
│   └── dllm/                 # zhziszz/dllm
└── *.md                      # design documents
```

---

## Testing Requirements

### Commands

```bash
# Activate virtualenv first
source .venv/bin/activate

# Fast tests only (required to pass before every PR merge)
python -m pytest tests/diffusion/ tests/models/ tests/test_fast_diffusion_model.py tests/test_e2e_integration.py -m "not slow" -v

# Full suite including slow E2E (GPU + real HF checkpoints required)
python -m pytest tests/ -v

# Race-condition check (if modifying shared state)
python -m pytest tests/ -v  # (no -race for Python, but run on GPU if kernel changes)
```

### Rules

1. **All fast tests must pass** before a PR is approved — both CPU and GPU paths.
2. Tests that invoke Triton kernels **must** be marked `@pytest.mark.skipif(not cuda)` and
   call `model.cuda()` before `get_peft_model()`.
3. `@pytest.mark.slow` + `@pytest.mark.gpu` for tests that download real HF checkpoints.
4. Numerical accuracy tests must compare against `F.cross_entropy` (not just shape checks).
5. Triton kernel changes require running `tests/diffusion/test_gpu_accuracy.py` explicitly.

---

## Code Review Process

### Mandatory Checks for Every PR

1. **Reference implementation alignment** — For any dLLM algorithm change, compare against:
   - `dev/repos/d1/SFT/sft_trainer.py` — d1 SFT reference
   - `dev/repos/d1/diffu-grpo/diffu_grpo_trainer.py` — Diffu-GRPO reference
   - `dev/repos/dllm/dllm/core/trainers/mdlm.py` — MDLM/LLaDA reference
   - For transformers API compatibility (post_init, tie_weights, BnB quantizer, init_weights):
     `dev/repos/transformers/src/transformers/modeling_utils.py`,
     `dev/repos/transformers/src/transformers/integrations/bitsandbytes.py`
   - If `dev/repos/` is not cloned, run:
     ```bash
     git clone https://github.com/dllm-reasoning/d1.git dev/repos/d1
     git clone https://github.com/zhziszz/dllm.git dev/repos/dllm
     git clone --depth=1 https://github.com/huggingface/transformers.git dev/repos/transformers
     ```

2. **Behavioral differences from reference must be intentional** — Do not classify a
   deviation as a "bug" without first checking if it was an explicit design choice.

3. **CPU/GPU correctness** — Verify the change works on CPU (no Triton) and GPU (with Triton).

4. **No unintended causal masking** — dLLM attention must remain bidirectional. Check that
   `is_causal=False` is preserved across any attention path changes.

5. **Packed sequence path** — When `seq_info` is not None and `past_key_values` is None,
   the varlen path must be taken. Verify `packed_seq_lengths` (not `cu_seqlens`) is in batch,
   and that the Flash varlen CUDA guard (`Q.device.type == "cuda"`) is present.

### Review Criteria Priority

| Priority | Category | Description |
|----------|----------|-------------|
| CRITICAL | Correctness | Reference impl divergence, wrong loss computation, wrong masking |
| CRITICAL | Safety | CUDA/CPU dtype mismatch that causes runtime error |
| HIGH | Kernel patching | Missing CUDA guard, wrong layer path for patching |
| HIGH | Test coverage | Missing GPU guard on Triton tests |
| MEDIUM | Performance | Unnecessary tensor copies, bf16/float32 conversions |
| LOW | Style | Naming, docstrings, log message clarity |

---

## Model-Specific Layer Hierarchies

Different dLLM models have different layer structures. Always use runtime attribute checks:

### A2D (LLaMA / Qwen2 / Qwen3)
```
PeftModel
 └─ base_model.model.model.layers[i]
     ├─ self_attn.{q_proj, k_proj, v_proj, o_proj}
     └─ mlp.{gate_proj, up_proj, down_proj}
```

### LLaDA
```
PeftModel
 └─ base_model.model          ← LLaDAModelLM
     └─ model                 ← LLaDAModel (extra nesting vs LLaMA)
         └─ transformer.blocks[i]  ← LLaDALlamaBlock or LLaDASequentialBlock
             ├─ {q_proj, k_proj, v_proj, attn_out}
             └─ {ff_proj, up_proj}
```
**Note**: `LLaDASequentialBlock` uses fused `att_proj` — not patchable with split QKV kernel.
Use the two-path fallback:
```python
inner = model.base_model.model
transformer = (inner.model.transformer if hasattr(inner, "model")
               and hasattr(inner.model, "transformer")
               else inner.transformer)
```

### Dream
```
PeftModel
 └─ base_model.model.model.layers[i]
     ├─ self_attn.{q_proj, k_proj, v_proj}  ← bias=True → use apply_lora_qkv_with_bias
     ├─ self_attn.o_proj                    ← bias=False → standard apply_lora_o
     └─ mlp.{gate_proj, up_proj, down_proj}
```

---

## Triton Kernel Patching Rules

These rules apply to `_patch_a2d_peft`, `_patch_dream_peft`, `_patch_llada_peft`:

1. **CUDA guard first** — Skip all patching if model is on CPU:
   ```python
   first_param = next(iter(model.parameters()), None)
   if first_param is None or first_param.device.type != "cuda":
       return 0, 0, 0
   ```

2. **Bias check** — Standard `apply_lora_qkv` requires `bias=False` on all of q/k/v_proj.
   Use `apply_lora_qkv_with_bias` for Dream's q/k/v_proj (bias=True).

3. **Dropout check** — Skip Triton patching if `lora_dropout != 0` (kernel doesn't support it).

4. **lora_magnitude_vector check** — Skip if PEFT magnitude scaling is active (`_no_lora_mag()`).

5. **Emit `_warn_once()`** — Log a warning (not an error) when patching is skipped.

6. **Bidirectional attention** — Always patch `self_attn.forward` with `A2DAttention_fast_forward`
   on CUDA (not just when LoRA is present); this ensures `is_causal=False` even without LoRA.

---

## Common Gotchas

| Gotcha | Symptom | Fix |
|--------|---------|-----|
| `unsloth_zoo` forces bf16 on activations | `RuntimeError: BFloat16 != Float` in matmul_lora | Add CUDA guard; don't patch float32 CPU models |
| `DiffusionTrainer` without tokenizer | `OSError: Repo id must be alphanumeric: ''` | Pass `processing_class=tokenizer` explicitly |
| save/reload logits mismatch | `assert torch.allclose(...)` fails | Snapshot base weights before PEFT wrap; restore in fresh_base |
| Dream q/k/v_proj bias | QKV kernel silently skipped | Use `apply_lora_qkv_with_bias` from `unturtle.kernels.fast_lora` |
| LLaDA extra model nesting | `AttributeError: 'LLaDAModelLM' has no attribute 'transformer'` | Use two-path fallback (see layer hierarchy above) |
| `lora_dropout > 0` | Triton kernels not applied, slow training | Use `lora_dropout=0` for Triton path |
| SDPA auto-causal | Attention silently becomes causal when `q_len == k_len` | Set `sdpa_kwargs={"is_causal": False}` in AttentionConfig |
| `packed_seq_lengths` vs `cu_seqlens` naming | `get_packed_info_from_kwargs()` returns `None`; packed path silently disabled | Key must be `"packed_seq_lengths"` (flat 1D int32); `"cu_seqlens"` is a different list-of-tensors field |
| Flash varlen without CUDA guard | `flash_attn_varlen_func` crashes on CPU even when `HAS_FLASH_ATTENTION=True` | Check `Q.device.type == "cuda"` before calling flash varlen — `HAS_FLASH_ATTENTION` only means the package is installed |
| Flash varlen compaction requires prefix-contiguous layout | Metadata mismatch / wrong attention output | `PackedMaskedDiffusionDataCollator` places real tokens at `[0:sum(seq_lengths[b])]` with padding at end — compaction slices `Q_t[b, :real_counts[b]]`; never use uncompacted `[B, L]` tensors directly with `flash_attn_varlen_func` |
| `build_sdpa_packed_attention_mask()` is causal | Packed SDPA silently reintroduces causal masking for dLLM | Never use the upstream `build_sdpa_packed_attention_mask()` for A2D/packed path — it builds upper-triangular causal blocks; use `block_attention_mask` from collator or `effective_mask=None` |
| real-checkpoint tokenizer may lack `mask_token_id` | `MaskedDiffusionDataCollator` init fails even though the model supports masking | Use `tokenizer.mask_token_id or model.config.mask_token_id` and pass `mask_token_id=` explicitly in slow E2E tests |
| prompt/full tokenization mismatch in completion-only datasets | prompt tokens become maskable or completion boundary shifts for override checkpoints | Build prompt/completion ids with the same tokenizer settings (e.g. both `add_special_tokens=False`) and concatenate them explicitly |
| `LLaDAModelLM` lacks `all_tied_weights_keys` (Hub class) | `AttributeError: 'LLaDAModelLM' has no attribute 'all_tied_weights_keys'` during 4-bit load | `FastDiffusionModel._load_model_auto` routes `model_type="llada"` to unturtle's own class; unturtle class calls `self.post_init()` and accepts `tie_weights(**kwargs)` |
| `load_in_4bit` without `device_map` OOM on crowded GPU | OOM caught silently → falls through to Hub class → `all_tied_weights_keys` error | `FastDiffusionModel.from_pretrained` sets `device_map="auto"` automatically when `load_in_4bit=True` |

---

## Commit & PR Conventions

```
<emoji> <type>(<scope>): <description> (#<issue>)
```

| Emoji | Type | Use |
|-------|------|-----|
| ✨ | feat | New feature |
| 🐛 | fix | Bug fix |
| 📚 | docs | Documentation |
| ✅ | test | Tests |
| ⚡ | perf | Performance (Triton kernel optimization) |
| ♻️ | refactor | Refactoring |
| 🔧 | chore | Config / dependency updates |

**Triton kernel changes** require test output (`pytest tests/diffusion/test_gpu_accuracy.py`)
to be pasted in the PR description.

---

## Environment Setup (Quick Reference)

```bash
# First time
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install "setuptools==80.9.0" "setuptools-scm==9.2.0"
uv pip install -e ".[huggingface]"
uv pip install pytest ruff bitsandbytes

# Every session
source .venv/bin/activate

# Lint
ruff check . && ruff format .

# Verify installation
python -c "
import torch; print('torch:', torch.__version__, '/ CUDA:', torch.cuda.is_available())
import unturtle; print('unturtle:', unturtle.__version__)
from unturtle.diffusion import DiffusionTrainer; print('DiffusionTrainer: OK')
from unturtle import FastDiffusionModel; print('FastDiffusionModel: OK')
"
```

**Known compatibility constraints:**
- Python 3.12 (not 3.13 — xformers incompatible)
- TRL 0.29.1+ (do NOT install `llm_blender` or `mergekit`)
- CUDA 12.4 (tested on NVIDIA RTX 6000 Ada)
