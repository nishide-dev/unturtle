# AGENTS.md — Unturtle Project

> Keep this file concise and actionable.
> See also: `CLAUDE.md` for the main project workflow and invariants.

---

## Project Overview

**Unturtle** is a dLLM method layer built on `unsloth`, with Triton-optimized support for
diffusion language models such as LLaDA and Dream.

Core behavioral differences from autoregressive LLM training:
- tokens are randomly masked
- loss is computed only at masked positions
- attention is bidirectional (`is_causal=False`)
- timestep-dependent masking / weighting may be used

Layering (Unturtle does not own the training loop):

```text
transformers  model implementations + in-model loss / generation
TRL           objective trainers (DiffusionTrainer is a peer of DPOTrainer)
unsloth       acceleration patches (fast LoRA, Triton kernels)
unturtle      dLLM method layer
```

Models that carry their own loss/generation (e.g. DiffusionGemma) train through the standard
`transformers`/`FastModel` path; `DiffusionTrainer` supplies the objective only for backbones
that lack a built-in diffusion objective.

## Model taxonomy — three orthogonal axes

- **Backbone architecture** (`unturtle.models.backbones.{llada,dream,modernbert}`):
  native diffusion backbones Unturtle implements.
- **Conversion method** (`unturtle.models.conversion`): a *method*, not a model. `a2d` is
  the AR→Diffusion family; the recipe is **Tiny-A2D** (`conversion.a2d.tiny_a2d`, classes
  `TinyA2D*`, model_types `tiny-a2d-*`) — thin adapters over `transformers` Qwen/Llama.
- **Training objective** (`unturtle.diffusion`): MDLM, BD3LM.
- **Shared infra** (`unturtle.models.generation`): cache / block-decode / generation mixin.

New code goes on the correct axis. `AutoConfig.register(...)` fires once per model_type.

## Important Paths

```text
.
├── unturtle/
│   ├── diffusion/          # trainer / collator / scheduler / GRPO
│   ├── kernels/            # masked diffusion loss / fast LoRA
│   ├── models/
│   │   ├── backbones/      # llada / dream / modernbert (native backbones)
│   │   ├── conversion/     # a2d/ (family) → tiny_a2d/ (recipe)
│   │   └── generation/     # cache / block-decode / generation mixins
│   ├── eval/               # smoke evaluators + lm-eval-harness adapter
│   ├── cli/                # unturtle CLI (train / generate / export / eval)
│   └── fast_diffusion_model.py
├── tests/                  # incl. test_cli_smoke.py, tests/examples/
├── benchmarks/             # tracked benchmark scripts
├── examples/               # runnable examples + configs
└── dev/repos/              # local reference repos (ignored)
```

## Testing Requirements

```bash
uv run python -m pytest tests/diffusion/ tests/models/ tests/test_fast_diffusion_model.py tests/test_e2e_integration.py -m "not slow" -v
uv run python -m pytest tests/ -v
```

Rules:
1. Relevant fast tests must pass before PR merge.
2. Triton / Flash changes must be checked on CPU fallback and CUDA.
3. Triton kernel numerical checks should compare against reference behavior, not only shapes.
4. Real-checkpoint tests should stay `slow` + `gpu`.
5. Root `conftest.py` autouse clears distributed launcher env vars (`WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, …) before each test. Tests that call `torch.distributed.init_process_group` must set those vars again for that test (e.g. `monkeypatch.setenv`).

## Code Review Checklist

For algorithmic changes, compare against:
- `dev/repos/d1/SFT/sft_trainer.py`
- `dev/repos/d1/diffu-grpo/diffu_grpo_trainer.py`
- `dev/repos/dllm/dllm/core/trainers/mdlm.py`
- `dev/repos/transformers/src/transformers/modeling_utils.py`
- `dev/repos/transformers/src/transformers/integrations/bitsandbytes.py`
- `dev/repos/fast-dllm/dream/model/generation_utils_block.py`
- `dev/repos/fast-dllm/llada/model/modeling_llada.py`

Always check:
1. reference alignment
2. CPU/GPU correctness
3. CUDA guards on Triton / Flash paths
4. bidirectional attention preserved
5. packed-sequence / varlen behavior preserved

Use the current PR review tooling / review agents for review.

## Model Path Reminders

### A2D

```text
PeftModel
 └─ base_model.model.model.layers[i]
```

### LLaDA

```text
PeftModel
 └─ base_model.model
     └─ model
         └─ transformer.blocks[i]
```

### Dream

```text
PeftModel
 └─ base_model.model.model.layers[i]
```

## Common Gotchas

| Gotcha | Symptom | Fix |
|--------|---------|-----|
| `DiffusionTrainer` without tokenizer | `Repo id must be alphanumeric` | pass `processing_class=tokenizer` |
| `packed_seq_lengths` vs `cu_seqlens` | packed path silently disabled | use `packed_seq_lengths` for packed fast path |
| `build_sdpa_packed_attention_mask()` is causal | packed dLLM attention silently regresses | do not reuse it for dLLM packed attention |
| Flash varlen on CPU | crash despite package installed | guard on `Q.device.type == "cuda"` |
| A2D CUDA RoPE reuses pre-aligned `position_ids` | packed / reset positions break CUDA outputs | flatten pre-aligned cos/sin and index by flat row ids, not reused `position_ids` |
| Dream q/k/v has bias | QKV Triton patch skipped | use `apply_lora_qkv_with_bias` |
| LLaDA extra nesting | transformer path lookup fails | use runtime attribute fallback |
| loss normalized by `n_masked` | reference-aligned loss scale drifts | normalize by `n_maskable` |
| `load_in_4bit` without `device_map` | OOM / wrong fallback path | use `device_map="auto"` |
| wrong package for dLLM trainers | `ModuleNotFoundError` / wrong API | `from unturtle.diffusion import …` |
