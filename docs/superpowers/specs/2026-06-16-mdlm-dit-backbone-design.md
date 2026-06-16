# MDLM-DiT native diffusion backbone — design

> Status: approved (brainstorming). Next: implementation plan (writing-plans).
> Date: 2026-06-16

## Goal

Add the **MDLM DiT** (Diffusion Transformer from
[kuleshov-group/mdlm](https://github.com/kuleshov-group/mdlm)) as a native
diffusion **backbone** in Unturtle, so that it can be trained from scratch with
the existing `DiffusionTrainer` (MDLM/SUBS masked-diffusion objective) and
sampled via `model.generate(algorithm="mdlm")`. This provides a lightweight,
research-comparable **baseline backbone** alongside LLaDA / Dream / ModernBERT /
DiffusionGemma.

### What this is — and is not

- This is the **backbone axis** of Unturtle's three-axis taxonomy
  (backbone × conversion method × training objective). The MDLM *objective*
  (SUBS-parameterized masked-CE) already exists in `DiffusionTrainer`; only the
  *backbone* (DiT) is new.
- This is a **native re-implementation baseline**, NOT a weight-compatible port
  of kuleshov's published checkpoints. It reproduces the architecture as a
  from-scratch-trainable model, equivalent to the paper's
  `time_conditioning=False` setting (see "Time-agnostic design" below). It is
  **not** bit-identical to the kuleshov time-conditioned (sigma-input) variant.

## Background

kuleshov-group/mdlm ships three backbones selected by `config.backbone`
(`dev/repos/mdlm/diffusion.py:92`): `dit` (the main DiT, `models/dit.py`),
`dimamba` (Mamba), and `ar` (autoregressive baseline), plus an `hf_dit` loader.
MDLM as a *method* = the SUBS objective + samplers; DiT is the **carrier
architecture** (the one that achieves the paper's SOTA perplexity).

Unturtle already implements the MDLM objective (`unturtle/diffusion/trainer.py`,
`fast_masked_diffusion_loss`, schedulers, CART reweighting) and the `mdlm`
sampling algorithm (`unturtle/models/generation/sampler.py`). It does NOT have a
DiT backbone. This design adds exactly that.

## Key contract discovery

Unturtle's backbone contract is **time-agnostic**: `forward(input_ids,
attention_mask, ...) -> CausalLMOutputWithPast(logits=...)`. Timesteps are
sampled by the collator and used **only for loss weighting** in
`DiffusionTrainer.compute_loss` (`unturtle/diffusion/trainer.py:308-326`) — they
are NEVER passed into `model.forward`.

kuleshov DiT, by contrast, is **time-conditioned**: `forward(indices, sigma)`,
where `sigma` drives the adaLN-Zero modulation. The two must be reconciled.

Unturtle reconciles diverse conditioning needs not by unifying `forward`
signatures, but by **per-algorithm capability contracts** in the sampler
registry:

| Algorithm | Capability probe | Model I/O | Conditioning |
|---|---|---|---|
| `mdlm` / `block_decode` / `bd3lm` | `_sample` / `_model_forward_with_cache` / `_sample_block_diffusion` | `forward(input_ids) -> logits` (time-agnostic) | none (masked family) |
| `block_ar` (DiffusionGemma) | `_denoising_step` | upstream canvas forward | yes, **internal to the model** |

So the time-agnostic contract is local to the **masked-diffusion family**, not a
global invariant. DiffusionGemma already coexists by riding the separate
`block_ar` lane with its conditioning sealed inside its own `generate` loop.
A future time-conditioned masked model would add its own algorithm lane; it does
not force sigma plumbing into the shared masked path. MDLM-DiT rides the `mdlm`
lane.

## Approach (chosen: A — time-agnostic port)

Port the DiT as a **time-agnostic** model that conforms to the masked-diffusion
backbone contract, riding the existing `mdlm` algorithm with no changes to the
collator / trainer / sampler / harness.

Rejected alternatives:
- **B (faithful time-conditioned port):** keep `forward(input_ids, sigma)` and
  add a sigma path through collator/trainer/sampler/mixin. Breaks the
  time-agnostic contract shared by LLaDA/Dream/ModernBERT; large blast radius;
  would become a liability on those time-agnostic paths. Rejected.
- **C (load HF weights only):** no new architecture; load kuleshov's published
  weights via `trust_remote_code`. Doesn't deliver a from-scratch-trainable
  native backbone, and conflicts with Unturtle's "native avoids
  trust_remote_code" loading policy. Rejected.

### Time-agnostic design (the core decision)

kuleshov's conditioning flow:
`sigma -> TimestepEmbedder -> c -> SiLU -> each DDiTBlock.adaLN_modulation(c) ->
shift/scale/gate`. The `adaLN_modulation` linear is **zero-initialized**
(adaLN-Zero), so at init `modulate(x,0,0)=x` and gate=0 kills the residual
branch until training grows the weights.

**Decision (2A):** replace `TimestepEmbedder(sigma)` with a single **learnable
constant conditioning vector** `cond = nn.Parameter(torch.zeros(cond_dim))`;
in forward, `c = F.silu(cond).expand(batch_size, -1)`. The adaLN-Zero modulation
machinery (shift/scale/gate) is **retained** — only the time input is removed.

This is structurally and computationally **equivalent to MDLM's
`time_conditioning=False`** setting: kuleshov's `_process_sigma`
(`dev/repos/mdlm/diffusion.py:307`) feeds `0` into `TimestepEmbedder` when
`time_conditioning=False`; our variant feeds a single learnable vector through
adaLN. Both stream one batch-shared conditioning vector into adaLN-Zero. The
paper reports `time_conditioning` has negligible effect under SUBS, consistent
with this equivalence.

Consequence: `scale_by_sigma` (sigma-dependent output scaling) becomes
meaningless and is **dropped**.

## Components

New module tree (templated on LLaDA — the closest native from-scratch backbone):

```
unturtle/models/backbones/mdlm_dit/
├── __init__.py                  # public symbols + (re-export); registration lives in modeling
├── configuration_mdlm_dit.py    # MDLMDiTConfig(PretrainedConfig), model_type="mdlm-dit"
└── modeling_mdlm_dit.py         # MDLMDiTModel / MDLMDiTForMaskedDiffusionLM + Auto* registration
```

### Config (`MDLMDiTConfig`)

`model_type = "mdlm-dit"` (unique; no upstream HF collision). Flat HF-standard
fields (no OmegaConf nesting). Field mapping from kuleshov `model.*`:

| kuleshov | MDLMDiTConfig | note |
|---|---|---|
| `hidden_size` | `hidden_size` | HF-standard |
| `cond_dim` | `cond_dim` | adaLN conditioning dim (constant-vector dim) |
| `n_blocks` | `num_hidden_layers` | mapped to HF-standard name |
| `n_heads` | `num_attention_heads` | mapped to HF-standard name |
| `dropout` | `dropout` | |
| `length` | `max_position_embeddings` | RoPE max length |
| `tie_word_embeddings` | `tie_word_embeddings` | DiT default False |
| — | `vocab_size` | required |
| — | `mask_token_id` | required (mdlm generation + collator) |
| — | `pad_token_id` | required |
| — | `eos_token_id` | optional (generation stop) |
| — | `use_cache=False` | bidirectional, no KV cache |
| `scale_by_sigma` | — | **dropped** (meaningless under time-agnostic) |

HF properties (`hidden_size`/`num_attention_heads`/`num_hidden_layers`) are held
as direct fields (names already HF-standard), so no `@property` mapping needed.

### Model (`MDLMDiTForMaskedDiffusionLM`)

Class hierarchy (LLaDA-style):

```
MDLMDiTPreTrainedModel(PreTrainedModel)               # config_class, _no_split_modules
  └─ MDLMDiTModel                                      # embed + blocks + final layer
MDLMDiTForMaskedDiffusionLM(
    MDLMDiTPreTrainedModel, MaskedDiffusionGenerationMixin
)
  └─ self.model = MDLMDiTModel(...)                    # forward -> CausalLMOutputWithPast
```

Ported from kuleshov `dit.py`, retained: `DDiTBlock` (adaLN-Zero, zero-init),
`Rotary`/`apply_rotary_pos_emb`, `EmbeddingLayer`, `DDitFinalLayer`, `LayerNorm`,
`modulate`. **Removed:** `TimestepEmbedder` (→ learnable constant `cond`).

`forward` (time-agnostic, Unturtle contract):

```python
def forward(
    self,
    input_ids=None,
    attention_mask=None,        # [B,L] 1=attend, 0=pad
    inputs_embeds=None,
    labels=None,                # accepted for compat; loss computed by trainer
    output_hidden_states=None,
    return_dict=None,
    **kwargs,                   # absorbs past_key_values etc. (no KV cache)
) -> CausalLMOutputWithPast:    # .logits [B, L, vocab_size]; past_key_values=None
```

No `sigma`/`timesteps` parameter; absorbed via `**kwargs` if passed.

### Attention (the main porting change)

kuleshov uses `flash_attn_varlen_qkvpacked_func(..., causal=False)` only.
Replace with the **LLaDA pattern** (flash when available + CUDA + no attn_mask,
else SDPA):

```python
if self.flash_attn_func is not None and q.device.type == "cuda" and attn_mask is None:
    out = self.flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
else:
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
```

- `is_causal=False` fixed (invariant: preserve bidirectional attention).
- CUDA guard (`q.device.type == "cuda"`); CPU tests fall to SDPA.
- **Add padding support** (kuleshov ignored padding): convert `attention_mask`
  `[B,L]` to an additive SDPA bias `[B,1,L,L]`. Bidirectional — no causal bias
  added.

**jit fusion:** do NOT port kuleshov's global `torch._C._jit_set_profiling_*`
side-effect flags (avoid affecting other backbones). The fused
`bias_dropout_add_scale` helpers are kept as plain Python functions; Triton
fast-path optimization is deferred to a follow-up PR.

### Generation

Inherit `MaskedDiffusionGenerationMixin` → gets `_sample` (mdlm loop) →
`_supports_mdlm()` passes → `resolve_algorithm("auto", model) == "mdlm"`.
`mask_token_id` resolved from config.

### Loader integration

Add to `_native_model_classes()` in `unturtle/fast_diffusion_model.py`:

```python
"mdlm-dit": MDLMDiTForMaskedDiffusionLM,
```

`from_pretrained` then loads `model_type=="mdlm-dit"` natively (avoids
`trust_remote_code`); existing `_patch_for_diffusion()` post-load passes
unchanged.

## Scope (YAGNI)

In scope: time-agnostic DiT port, HF-compatible config + registration,
bidirectional attention with padding + CUDA guard, mdlm generation, native
loader registration, tests, doc updates.

Out of scope (separate PRs / deliberately excluded):
- SEDD / D3PM / AR parameterizations (MDLM = SUBS only).
- Mamba backbone (different axis).
- sigma time-conditioning path (approach B).
- kuleshov HF weight-load compatibility (this is a native re-implementation).
- Triton fast-LoRA / packed-varlen optimization (correctness first;
  `PackedMaskedDiffusionDataCollator` combination is a future PR — unpacked
  collator only for now).
- Real-checkpoint tests (no published weights for this native baseline).

## Testing

`tests/models/test_mdlm_dit.py`, LLaDA test pattern, CPU + tiny config
(hidden≈128, layers=2, heads=4), `-m "not slow"`:

| Test | Asserts |
|---|---|
| `test_config_defaults` | field defaults, `model_type=="mdlm-dit"` |
| `test_config_hf_properties` | `hidden_size`/`num_attention_heads`/`num_hidden_layers` match |
| `test_config_has_mask_token_id` | `mask_token_id` present |
| `test_forward_logits_shape` | `out.logits.shape == (B, L, vocab_size)`; `hasattr(out,"logits")` |
| `test_forward_is_time_agnostic` | forward succeeds with no sigma/timesteps (contract lock) |
| `test_bidirectional_attention` | output at i depends on tokens after i (not causal) |
| `test_attention_mask_respected` | masking a padding position changes output (padding support) |
| `test_resolve_algorithm_auto_mdlm` | `resolve_algorithm("auto", model) == "mdlm"` |
| `test_generate_output_shape` | `model.generate(input_ids, steps=2, mask_token_id=...)` shape |
| `test_save_reload_forward` | save_pretrained → from_pretrained forward parity (registration round-trip) |
| `test_adaln_zero_init` | adaLN_modulation weight/bias zero-initialized (identity modulation at init) |

Plus a `DiffusionTrainer` one-step training smoke (SUBS loss, existing collator).

## Mandatory review-check alignment (DiT-specific)

- `dllm` (MDLM/SUBS): loss supplied by existing `DiffusionTrainer`; backbone
  only emits logits. Aligned.
- `transformers`: init / tie-weights / registration covered by `save_reload`;
  `tie_word_embeddings=False` (DiT default).
- Bidirectional preserved: `is_causal=False` fixed + `test_bidirectional_attention`.
- CUDA guards: flash branch gated on `device.type=="cuda"`; CPU → SDPA.
- packed-varlen: NOT supported this PR (YAGNI); unpacked collator only.

## Docs (same PR)

- `docs/dllm-gap-map.md`: add MDLM-DiT native backbone row (✅).
- `CLAUDE.md`: add `mdlm_dit` to the backbones list in the model taxonomy.
- Optional `examples/` minimal training snippet (may follow later).

## Issue / branch / PR

Issue title: `[Phase N] add MDLM-DiT native diffusion backbone (time-agnostic baseline)`
Labels: `type: feat`, `diffusion`. Branch: `feat/<issue#>-mdlm-dit-backbone`.

### Acceptance criteria

1. `MDLMDiTConfig` (`model_type="mdlm-dit"`, HF properties, `mask_token_id`)
   registers.
2. `MDLMDiTForMaskedDiffusionLM.forward(input_ids)` returns
   `CausalLMOutputWithPast(logits=[B,L,V])` **with no sigma** (time-agnostic).
3. adaLN-Zero structure preserved (zero-init, constant conditioning vector).
4. Bidirectional (`is_causal=False`) + padding support + CUDA-guarded flash/SDPA.
5. `resolve_algorithm("auto", model) == "mdlm"`; `model.generate(...)` works.
6. One `DiffusionTrainer` training step passes (SUBS loss, existing collator).
7. save/reload round-trip; all `-m "not slow"` tests green.
8. gap-map / CLAUDE.md updated.

### Implementation order (TDD)

1. Config + registration → config tests.
2. Internal modules (time-agnostic DiT, attention replacement) → forward shape /
   bidirectional / time-agnostic tests.
3. Mixin inheritance + generation → algorithm-resolution / generate tests.
4. Loader integration → save/reload test.
5. DiffusionTrainer one-step training smoke.
6. gap-map / CLAUDE.md updates.
7. `ruff check` / `ruff format`; full fast-test suite.
8. Draft PR → pr-review → fix critical/high → squash merge.
