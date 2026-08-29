# Acceleration ledger — Unturtle-specific optimized paths

Issue #166 Stage 0. Parent #151.

## What this is

One row per Unturtle-specific optimized or reference-only code path, with the
regime where it is valid, its correctness fallback, and the evidence that
exists **today**. It answers a single question:

> Which Unturtle-specific hot path now consumes enough end-to-end time or
> memory to justify the next specialized optimization?

Scope boundary with #157: that issue owns generation topology, parallel
decoding, KV-cache and commitment order. This ledger owns training objective
and output-head costs, family-specific attention/conditioning, continuous /
flow solver and state update, codec / endpoint projection, and pack-local
reference paths.

**This document records measurements, not decisions.** Stage 2 selects at most
one target; nothing here preselects it. Negative and neutral results are
retained deliberately — an optimization that did not pay off is the most
useful row in a ledger, because it stops the same idea being re-proposed.

## Reading the evidence column

Two evidence grades are kept apart, because conflating them is how a kernel
win becomes an unfounded end-to-end claim:

- **end-to-end** — measured through a real training step or generation call,
  including the model forward it has to compete with;
- **kernel-only** — measured on the operation in isolation. Never sufficient
  to justify a dispatch default.

A third state, **not measured**, is stated explicitly rather than left blank.

---

## Row 1 — dense masked CE / fused-mask wrapper

| field | value |
|---|---|
| family/method | masked diffusion (all families) |
| operation | loss (LM head + cross-entropy) |
| backend | Unsloth Triton CE (`Fast_CrossEntropyLoss`) plus an Unturtle allocation fusion: one `torch.where` replacing `labels.clone()` + masked scatter |
| semantic scope | masked-diffusion loss on masked positions only; `labels == -100` elsewhere |
| valid regime | CUDA; falls back on CPU |
| fallback | in-function `else` branch (`fused_masked_diffusion_loss.py:117-137`) using `F.cross_entropy`, mirroring the kernel's op order (scaling before softcapping); guard is `logits.device.type == "cuda"` |
| dispatch | default |
| parity evidence | kernel correctness compared against `F.cross_entropy`, not shape-only (project testing rule) |
| end-to-end evidence | **not measured as a share of step time** |
| provenance | `unturtle/kernels/`, `benchmarks/kernels/benchmark_loss.py` |

Gap: no end-to-end wall share. This is the default path for every masked
family, so its share is a Stage-1 requirement rather than an optional cell.

## Row 2 — sparse masked LM-head

| field | value |
|---|---|
| family/method | masked diffusion |
| operation | output projection + loss |
| backend | **plain PyTorch** — gather → project → `F.cross_entropy` (`unturtle/kernels/sparse_masked_loss.py`); no Triton in this file |
| semantic scope | projects only masked positions; saving scales with `1 - M/(B*L)` |
| valid regime | **mask ratio is the decisive variable, not vocabulary size**; memory sign flips around ~40% masking. Capability-gated: Tiny-A2D (llama/qwen2/qwen3) only; `logit_softcapping`/`logit_scaling` rejected |
| fallback | dense masked CE (row 1). **Raises rather than falling back silently**; `supports_sparse_masked_loss()` is the sanctioned probe, and `DiffusionTrainer` raises at construction so an opt-in cannot degrade into a no-op |
| dispatch | **default-off, opt-in** |
| parity evidence | differential against dense |
| end-to-end evidence | **end-to-end, regime-swept** — see below |
| provenance | #77, `benchmarks/sparse_lm_head_training.py` |

Measured through `DiffusionTrainer.compute_loss` (forward + loss + backward),
RTX 6000 Ada, fp32, B=2 L=512 H=512, 2 layers, 3 interleaved trials of 10
timed steps; negative = sparse better. `d activ` subtracts model weights,
which are identical in both arms and dilute the total:

| vocab | mask | step time | d peak | d activ |
|---|---|---|---|---|
| 32000 | 0.15 | **−32.6%** | −25.7% | **−40.9%** |
| 32000 | 0.50 | **−12.6%** | +0.2% | +0.4% |
| 32000 | 0.75 | +1.3% | +21.4% | +34.1% |
| 128256 | 0.15 | **−62.1%** | −22.7% | **−37.4%** |
| 128256 | 0.50 | **−25.7%** | +1.0% | +1.7% |
| 128256 | 0.75 | −4.3% | +27.7% | +45.5% |

Kernel-level peak memory measured earlier (#77; RTX 6000 Ada, bf16, B=4 L=1024
H=1024, forward + loss + backward) agrees on the sign and the crossover:

| vocab | mask 15% | mask 50% | mask 75% |
|---|---|---|---|
| 32000 | −28% | +8% | +35% |
| 128256 | −41% | +10% | +49% |

Why dense is harder to beat than the `[B, L, V]` shape suggests:
`Fast_CrossEntropyLoss` upcasts per tile in registers and never materializes an
fp32 logits tensor, so dense holds one bf16 `[B, L, V]` while sparse holds a
bf16 `[M, V]` **plus its autograd graph**. Past `M/(B·L) ≈ 0.4` the gather
stops paying for itself.

LoRA, 128256 vocab, same setup:

| mask | step time | d peak | d activ |
|---|---|---|---|
| 0.15 | **−49.3%** | −25.6% | **−50.3%** |
| 0.50 | −6.0% | +24.2% | +47.5% |
| 0.75 | +16.1% | +57.5% | +112.9% |

Why the flag stays off — and the precise reason matters. At ~50% masking the
**step-time win survives** under full fine-tuning (−12.6% at 32K vocab, −25.7%
at 128K); what disappears is the *memory* benefit (+0.4% / +1.7% activations),
and under LoRA memory becomes an outright cost (+47.5% activations at 0.50,
+112.9% at 0.75) because the frozen backbone's activations already dominate, so
the `[M, V]` projection and its autograd graph are close to pure overhead.

So the accurate statement is **not** "sparse stops paying at ~50% masking" —
that would deny a real speedup. It is: *at the ~50% masking MDLM-style training
averages, this is not a consistent speed **and** memory win, so it is opt-in
rather than default.* MDLM samples `t ~ U(0,1)`, which lands exactly there.

## Row 3 — device-side masked noising

| field | value |
|---|---|
| family/method | masked diffusion |
| operation | noising (training-state construction) |
| backend | device-side process (`unturtle/processes`) vs CPU collator |
| semantic scope | `MaskedDiffusionProcess`; trainer/evaluator inject the collator with `noise=False` and apply the process device-side |
| valid regime | any; the packed collator still noises |
| fallback | in-collator corruption (`noise=True`, legacy default) |
| dispatch | default for `DiffusionTrainer` / `BlockDiffusionTrainer` / `MaskedDiffusionEvaluator` |
| parity evidence | architecture/RNG correctness |
| end-to-end evidence | **end-to-end — NO measurable difference** |
| provenance | #62, `benchmarks/collator_vs_process_noising.py` |

Measured through `DiffusionTrainer.compute_loss`; B=4, L=512, V=32000, H=512,
2 layers, 5 interleaved trials of 40 timed steps, single GPU: median
**+0.42%**, range −0.61% to +1.12%, device path slower in 4 of 5 trials.

**The sign is not consistent, so this is "no measurable difference".** The
path is justified by architecture and RNG correctness, not speed. Recorded so
that no future work claims a latency win here.

## Row 4 — hybrid prefix attention fast path

| field | value |
|---|---|
| family/method | Tiny-A2D (llama / qwen2 / qwen3) |
| operation | attention |
| backend | mask-free two-call split |
| semantic scope | 2-D all-ones attention mask, no packed kwargs; the dense mask is **always built** |
| valid regime | `seq_len >= hybrid_fast_min_seq_len` (default **2048**) |
| fallback | dense masked attention — the semantic reference; the gate only ever trades speed |
| dispatch | **gated on a declared config field**, not a buried constant |
| parity evidence | mutation-verified eligibility rules (`_hybrid.py`) |
| end-to-end evidence | **end-to-end, full forward, both sides of the crossover** |
| provenance | #63, #99, `_hybrid.py:179`, `modeling_qwen3.py:72` |

Measured on an 8-layer bf16 model, **full forward** (not kernel-only):
**0.90× at L=1024**, 1.50× at L=2048, 1.92× at L=4096 (h512/8-head; the
h1024/16-head crossover sits in the same range).

The sub-crossover slowdown is the load-bearing number: below the threshold the
extra kernel launch, `cat` and output transpose outweigh the attention win, so
an ungated fast path would make forwards *slower*. 2048 is the conservative
side of the measured crossover.

## Row 5 — bias-aware fast LoRA

| field | value |
|---|---|
| family/method | Dream (q/k/v use `bias=True`) |
| operation | fused QKV LoRA |
| backend | custom `autograd.Function` over Unsloth primitives (`matmul_lora`, `fast_dequantize`); extends Unsloth `LoRA_QKV`. **Authors no Triton itself** — the fusion is at the autograd/GEMM level (`unturtle/kernels/fast_lora.py`) |
| semantic scope | adds bias in forward and the bias gradient (`dQ.sum(0)`) in backward; standard `apply_lora_qkv` requires `bias=False` |
| valid regime | CUDA + `lora_dropout == 0` + `bias == "none"` + LoRA present + no DoRA + **activations in the quantization compute dtype** (bf16/fp16). The module itself carries **no guards**; per-layer gating lives in `_patch_dream_peft`, and the dtype constraint is enforced model-wide in `patch_peft_model` (#177). The documented 4-bit + PEFT flow satisfies it because `unturtle/save.py::prepare_model_for_kbit_training` uses unsloth semantics (frozen params keep their loaded dtype) — peft's own prepare upcasts them to fp32, which no fused path (`matmul_lora`) can execute |
| fallback | PEFT's default `LoraLinear.forward` — silent, with a one-shot warning. A model whose hidden-state dtype cannot feed the fused kernels (e.g. fp32-upcasted) skips **all** fast paths uniformly with the typed reason `incompatible_compute_dtype` (#177) — never a partially-fast model |
| dispatch | default where the family matches |
| parity evidence | differential against the unfused path; since #177, a 4-bit tiny fixture **executes** forward+backward with output/gradient parity against a genuinely unfused standard-PEFT arm (`tests/test_4bit_peft_fast_lora.py`). At 7B the retained `parity_preflight` now runs both arms end-to-end; its frozen scale-blind atol/rtol=2e-2 flags bf16 rounding on QKV outputs of magnitude ~50–140 (single-layer fp32 ground truth: the fast arm's error is *smaller* than the peft reference arm's own — 0.16 vs 0.25 on Q, 0.50 vs 0.83 on K). Tolerance recalibration is a #166 measurement decision |
| end-to-end evidence | **not measured as a share of step time** |
| provenance | `unturtle/kernels/fast_lora.py` |

## Row 6 — ELF generation

| field | value |
|---|---|
| family/method | ELF (pack `unturtle-elf`) |
| operation | denoiser forward, SC-CFG, solver, endpoint projection |
| backend | reference PyTorch (pack-local `_reference/`) |
| semantic scope | frozen #153 checkpoint/config cells |
| valid regime | cfg=1, SC-CFG=3, γ=1.5@32 / 1.0@64, logit-normal grid, L=1024 |
| fallback | reference path IS the semantic oracle |
| dispatch | reference-only (not optimized) |
| parity evidence | decision-grade parity (#153) |
| end-to-end evidence | end-to-end wall + peak memory per frozen cell; **no per-operation breakdown** |
| provenance | `benchmarks/results/elf_b_owt_{32,64}/frontier_record.jsonl` |

| cell | NFE | bs1 samples/s | bs8 | bs32 | peak |
|---|---|---|---|---|---|
| `elf_b_owt_32` | 32 | 3.147 | 4.518 | — | 4.9 GiB |
| `elf_b_owt_64` | 64 | 2.419 | 4.415 | 3.628 | 8.8 GiB |

**SC-CFG adds no extra forward in these cells.** Executed NFE equals the step
count (32 and 64), and SC-CFG is passed as a per-batch *scale tensor*
(`generation_utils.py:124`), not a doubled batch — it is in-context
conditioning.

The scope of that claim is deliberately narrow: **there is no extra forward, so
forward *sharing* has nothing to recover** at cfg=1, and Stage 2 should not
pursue it on the strength of the issue's candidate list alone. This is **not**
a claim that SC-CFG is free — the prefix/attention overhead of the SC-CFG
tokens themselves has not been decomposed, and Stage 1 has not measured it. The
training path is a different matter (row 8).

## Row 7 — FMLM / FLM generation

| field | value |
|---|---|
| family/method | FMLM flow-map, FLM (pack `unturtle-flm`) |
| operation | model forward, state construction/update, endpoint projection |
| backend | reference PyTorch (`_reference/`) |
| semantic scope | frozen #155 cells |
| valid regime | L=1024; FMLM 1-step and 32-step; FLM 1024-step |
| fallback | reference path IS the oracle |
| dispatch | reference-only |
| parity evidence | decision-grade parity (#155) |
| end-to-end evidence | end-to-end wall + peak memory; **no per-operation breakdown, and peak is not attributable** |
| provenance | `benchmarks/results/fmlm_owt_{1,32}`, `flm_owt_1024` |

| cell | NFE | bs1 samples/s | bs8 | bs32 | peak |
|---|---|---|---|---|---|
| `fmlm_owt_1` | 1 | 35.54 | 61.58 | 69.49 | 16.0 GiB |
| `fmlm_owt_32` | 32 | 1.035 | 1.481 | — | 25.2 GiB |
| `flm_owt_1024` | 1024 | 0.0401 | 0.0560 | — | 25.2 GiB |

State shape: the flow-map state `z` is `[B, L, V]` — a full simplex tensor,
not token ids (`sampler.py:144`) — and each step allocates further `[B, L, V]`
intermediates (`D_st_pred`, `z_tilde`, the two weighted terms).

**A caution against the obvious inference.** One `[B, L, V]` tensor at B=1,
L=1024 is only 0.06–0.49 GiB depending on vocab and dtype, so the state tensor
alone does **not** explain a 16–25 GiB peak. `peak_memory_bytes` is recorded
once per record while each record ran batch 1, 8 and 32, so the figure belongs
to the largest batch that ran and **cannot be attributed to any operation or
batch size**. Whether state construction/update is actually the memory driver
is a Stage-1 measurement, not something this ledger may assert.

## Row 8 — ELF training (#154 smoke path)

| field | value |
|---|---|
| family/method | ELF |
| operation | T5 encoding, denoiser/decoder objective, **self-conditioning target forwards**, backward, Muon/aux optimizer |
| backend | reference PyTorch + `muon_with_aux_adam` |
| semantic scope | disposable #154 smoke path; non-quality-bearing |
| valid regime | `self_cond_prob=0.5`, `num_self_cond_cfg_tokens=4` (both **defaults**) |
| fallback | reference path |
| dispatch | reference-only |
| parity evidence | Stage-1/2 accepted (#154) |
| end-to-end evidence | **not measured** |
| provenance | `packs/unturtle-elf/src/unturtle_elf/training.py:176-268` |

Structural finding, from reading the call graph rather than a profile: at the
default config a single training step performs **up to two extra `no_grad`
model forwards** beyond the trained forward —

- `compute_shared_uncond` (`training.py:248`), taken when
  `self_cond_prob > 0 or num_self_cond_cfg_tokens > 0`;
- `net_out_cond` inside `get_sc_cond_and_uncond` (`training.py:204`), taken
  when `self_cond_prob > 0`;
- then the single grad-enabled forward (`training.py:266`) plus backward.

Both conditions hold at the shipped defaults, so this is the default training
path, not an exotic setting, and both auxiliary forwards are genuinely executed
— unlike ELF generation at cfg=1 (row 6).

On **call count** that makes self-conditioning likely to be a material share of
the ELF training step. Deliberately not stated more strongly than that: a
forward count is not a wall share (the auxiliary forwards are `no_grad`, so
they carry no backward and need not cost what the trained forward costs), and
Stage 1 must measure it before any kernel is proposed. It is a strong prior,
not a finding.

---

## Incidental findings from the survey

Not optimization targets — recorded so they are not rediscovered as mysteries.

**Evidence asymmetry is the headline.** Two paths carry measured numbers
in-code: the sparse LM-head (both the #77 kernel table and the end-to-end
sweep) and device-side noising, whose module docstring preserves all five
trials individually, not just the summary.

The asymmetry is on the other side. **Row 1 (dense masked CE) and row 5
(bias-aware fast LoRA) are enabled by default with no in-code measurement at
all**, and neither has an attributed share of step time: `fast_lora.py`
contains no performance comment of any kind, and
`fused_masked_diffusion_loss.py`'s justification is purely mechanical ("saves
one allocation and one kernel launch", `fused_masked_diffusion_loss.py:28`)
even though `benchmarks/kernels/benchmark_loss.py` exists to measure it and no
checked-in result references it. `masked_diffusion_loss_from_timesteps` is also
unmeasured but is opt-in and unused internally, so it carries no default-path
risk.

**Two terminology drifts.** `fast_lora.py:15` calls the file "Triton-fused LoRA
kernel extensions" and the runtime warnings say "Triton kernel", but the file
authors no Triton — verified by grep. Likewise
`masked_diffusion_loss.py:52-55` still describes a Phase-1 design where the
`-100` write and Python-level weighting are its own work; the function now
delegates entirely (`:82-92`). Both are accurate about the *backend* and
misleading about *who implements it*, which matters for a ledger whose whole
purpose is attributing cost.

**One dead branch.** `fused_masked_diffusion_loss.py:158-160`: both arms of
`if loss_weights is None:` return the identical expression
`per_token_loss.sum() / n_maskable`. Harmless, but it reads as though weighting
were handled in the `"token"` branch when it is not.

**No `torch.compile` anywhere** in the kernels, the trainer, or
`fast_diffusion_model.py`.

**A soundness hole worth knowing** (documented in-code, not a defect):
`masked_diffusion_loss_from_timesteps` cannot distinguish a `(B, L)` timesteps
tensor from its transpose when `B == L`, so a transposed input is accepted and
silently yields a different loss (`masked_diffusion_loss.py:126-130`).
Orientation is the caller's responsibility.

## Stage-0 conclusion

Evidence status across the required rows:

| row | end-to-end evidence |
|---|---|
| 2 sparse LM-head | **yes**, regime-swept at two levels (kernel #77 + end-to-end), negatives retained |
| 3 device noising | **yes**, negative result |
| 4 hybrid attention | **yes**, both sides of the crossover |
| 1 dense masked CE | no |
| 5 bias-aware LoRA | no |
| 6 ELF generation | wall/memory only, no per-operation split |
| 7 FMLM/FLM | wall/memory only, peak not attributable |
| 8 ELF training | none |

The three paths with end-to-end evidence are all **already correctly
dispatched**: sparse is opt-in because ~50% masking is not a consistent
speed-*and*-memory win (the step-time win survives there; the memory benefit
does not), noising is justified on correctness with no speed claim, and hybrid
is gated on the conservative side of its measured crossover. There is no open
decision in those rows.

Every path that lacks end-to-end evidence is a *reference* path (rows 6–8) or a
default whose share has never been attributed (rows 1, 5). That is the Stage-1
target, and no existing artifact can substitute for it: the frozen cells record
one wall figure and one peak per record, which is the right granularity for a
frontier comparison and the wrong granularity for choosing a kernel.

Stage 1 therefore needs per-operation instrumentation that does not exist yet
for these paths — the only timing infrastructure in the repo today
(`output_timing` in the block-decode and Dream generation loops) belongs to the
#157 axis.

No target is selected here. Stage 2 owns that, and the strongest prior from
Stage 0 is row 8 — but it is a prior from a call graph, not a measurement, and
the issue's own rule applies: choose what consumes wall time, not what is easy
to write.
