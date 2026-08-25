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
| backend | Triton via Unsloth CE reuse (`unturtle/kernels/masked_diffusion_loss.py`, `fused_masked_diffusion_loss.py`) |
| semantic scope | masked-diffusion loss on masked positions only; `labels == -100` elsewhere |
| valid regime | CUDA; falls back on CPU |
| fallback | reference `F.cross_entropy` path; guards on parameter device, not `torch.cuda.is_available()` |
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
| backend | Triton (`unturtle/kernels/sparse_masked_loss.py`) |
| semantic scope | projects only masked positions; saving scales with `1 - M/(B*L)` |
| valid regime | **mask ratio is the decisive variable, not vocabulary size**; memory sign flips around ~40% masking |
| fallback | dense masked CE (row 1) |
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

LoRA, 128256 vocab, same setup:

| mask | step time | d peak | d activ |
|---|---|---|---|
| 0.15 | **−49.3%** | −25.6% | **−50.3%** |
| 0.50 | −6.0% | +24.2% | +47.5% |
| 0.75 | +16.1% | +57.5% | +112.9% |

Why the flag stays off: MDLM-style training samples `t ~ U(0,1)`, averaging
~50% masking — the column where memory is neutral-to-worse. The step-time win
is real and larger than the kernel benchmark suggested, but it is **not**
automatically a memory optimization. Under LoRA above ~15% masking the frozen
backbone's activations already dominate, so the `[M, V]` projection and its
autograd graph are close to pure overhead.

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
| backend | Triton, extends Unsloth `LoRA_QKV` (`unturtle/kernels/fast_lora.py`) |
| semantic scope | adds bias in forward and the bias gradient (`dQ.sum(0)`) in backward; standard `apply_lora_qkv` requires `bias=False` |
| valid regime | CUDA; `lora_dropout != 0` disables the Triton LoRA path |
| fallback | non-fused PEFT path |
| dispatch | default where the family matches |
| parity evidence | differential against the unfused path |
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
conditioning. So "SC-CFG forward sharing" in **generation** has no cost to
recover at cfg=1, and Stage 2 should not pursue it on the strength of the
issue's candidate list alone. The training path is a different matter (row 8).

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
path, not an exotic setting. That makes self-conditioning a large share of the
ELF training step **by construction** — but "large by construction" is a
hypothesis about wall share, and Stage 1 must measure it before any kernel is
proposed. Unlike ELF generation at cfg=1 (row 6), here the extra forwards are
genuinely executed.

---

## Stage-0 conclusion

Evidence status across the required rows:

| row | end-to-end evidence |
|---|---|
| 2 sparse LM-head | **yes**, regime-swept, negatives retained |
| 3 device noising | **yes**, negative result |
| 4 hybrid attention | **yes**, both sides of the crossover |
| 1 dense masked CE | no |
| 5 bias-aware LoRA | no |
| 6 ELF generation | wall/memory only, no per-operation split |
| 7 FMLM/FLM | wall/memory only, peak not attributable |
| 8 ELF training | none |

The three paths with end-to-end evidence are all **already correctly
dispatched**: sparse is opt-in because 50% masking is where it stops paying,
noising is justified on correctness with no speed claim, and hybrid is gated on
the conservative side of its measured crossover. There is no open decision in
those rows.

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
