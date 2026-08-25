# Stage-1 measurement protocol — profile before proposing a kernel

Issue #166 Stage 1. Design frozen before any instrumentation is written; the
implementation PR is separate and contains **measurement infrastructure plus
profile artifacts only** — no optimization.

Stage 0 (`docs/acceleration-ledger.md`) established that every path with
end-to-end evidence is already dispatched correctly, and that the paths without
it are either reference-only or defaults whose share was never attributed. This
protocol says how that attribution is measured.

## 0. What this is not

**No general-purpose profiler API is introduced.** The measurement contract is
benchmark-local: a record envelope shared across families, and an operation
taxonomy that is family-local. A universal `forward_time` field across ELF,
FMLM and masked training would force incomparable things into one name, which
is how a profile stops answering the question it was built for.

## 1. The verdict is the outer wall-clock

The sum of per-operation timings is **never** used as a proxy for step time.
Per-operation values explain *where* time went; only the outer measurement
decides *how much* there is.

Every cell therefore records:

| field | meaning |
|---|---|
| `wall_seconds_instrumented_off` | steady-state end-to-end, instrumentation disabled — **this is the verdict** |
| `wall_seconds_instrumented_on` | same cell with instrumentation active |
| `instrumentation_overhead` | the difference, reported not hidden |
| `operation_sum_seconds` | sum of per-operation inclusive times |
| `unattributed_seconds` | `wall_instrumented_on − operation_sum` |

`unattributed_seconds` is mandatory. A taxonomy that accounts for 60% of the
step is a useful partial result, but only if the 40% is visible; silently
normalizing the operation shares to 100% would manufacture confidence.

Rules, each one a defence against a specific way this goes wrong:

- **warmup and compile excluded from steady state**, and recorded separately —
  charging a one-time cost to one arm is one of the mutation targets;
- **CUDA synchronization at consistent boundaries.** Async kernel launches
  otherwise attribute time to whichever operation happens to synchronize;
- **reference and candidate arms interleaved with alternating order**, so
  thermal drift does not land entirely on whichever runs second (the discipline
  `benchmarks/sparse_lm_head_training.py` already uses);
- **replicated trials, median reported with range.** No performance claim from
  a single trial;
- **identical inputs and RNG across arms** — the same pre-noised batch, not two
  draws from the same distribution;
- **peak allocated and reserved memory per cell**, at a granularity that can be
  attributed. Stage 0 could not use the frozen cells' memory figures because
  one peak per record covered batch 1, 8 and 32; this protocol records peak
  **per (cell, batch)** so that failure is not repeated;
- **sequence length, dtype, batch and hardware recorded on every cell.**

## 2. Inclusive time, call count, and exclusive time where it is meaningful

Each operation event records `inclusive_seconds`, `call_count`, and
`exclusive_seconds` where nesting makes it well-defined. Call count is not
decoration: Stage 0's strongest prior is a *count* (ELF training's two
auxiliary forwards), and the whole point of Stage 1 is to find out whether that
count converts into wall share. A `no_grad` forward carries no backward, so it
need not cost what the trained forward costs — only measurement settles it.

## 3. Family-local operation taxonomies

### ELF training — the two auxiliary forwards are separate events

Recording both as one `self_conditioning` event would destroy the finding.
Stage 0 identified them as *distinct* call sites under *different* conditions
(`self_cond_prob > 0 or num_self_cond_cfg_tokens > 0` for the shared one,
`self_cond_prob > 0` for the conditional one), so they can be independently
material:

- `data_collation` (including H2D transfer)
- `t5_encoding` (frozen encoder)
- `sc_shared_uncond_forward` — `compute_shared_uncond`, `no_grad`
- `sc_conditional_forward` — `net_out_cond` inside `get_sc_cond_and_uncond`, `no_grad`
- `trained_forward` — the single grad-enabled forward
- `objective_loss`
- `backward`
- `optimizer_step` — Muon/aux Adam

### Masked training

- `attention`
- `lm_head_projection`
- `loss` (dense CE; sparse when opt-in is the arm under test)
- `noising` (device-side process)
- `backward`
- `optimizer_step`

### FMLM / FLM

- `model_forward`
- `state_construction` — the initial `[B, L, V]` allocation
- `state_update` — the weighted combination per step
- `flow_map_composition`
- `endpoint_projection`

State-related allocation is explicitly its own event because omitting it is a
named mutation target, and because Stage 0 deliberately refused to attribute
FMLM's 16–25 GiB peak without measurement.

### Hybrid attention

- `dense_mask_build` — always executed, so always charged
- `attention_path` (fast split vs dense)
- `full_model_forward`

The dense mask is built regardless of which path runs, so it belongs in the
accounting even when the fast path is taken.

### ELF generation

- `denoiser_forward`
- `solver_state_update`
- `endpoint_projection`

Included to confirm Stage 0's reading that SC-CFG adds no extra forward at
cfg=1, and to decompose the SC-CFG token prefix/attention overhead that Stage 0
explicitly left unmeasured.

## 4. Profiling order

ELF training is a strong prior, **not** a preselected target. Stage 1 measures
it first because it is the largest unmeasured default path, not to confirm it:

1. **ELF training smoke** (#154 disposable path; non-quality-bearing — must not
   inspect or reinterpret Stage-3 generation results)
2. **dense masked CE / LM-head / loss** — the default for every masked family
3. **bias-aware fast LoRA** — default-on, share never attributed
4. **FMLM 1-step and 32-step**
5. **ELF generation** — model / solver / projection confirmation

### Sanity and regression cells, not target candidates

Sparse LM-head, device-side noising and hybrid attention are reused to validate
the instrumentation itself: each has a known end-to-end result, so a harness
that cannot reproduce the sign and rough magnitude is not yet trustworthy. They
**do not return to the target pool** — their dispatch decisions are closed.

This gives the harness a falsifiable acceptance test rather than assuming it
measures what it claims: the sparse arm must reproduce a step-time win at 15%
masking and a memory penalty at 75%; hybrid must reproduce a sub-crossover
slowdown at L=1024.

## 5. Cells

Batch 1 / 8 / 32 where semantics and memory permit; typed `oom` or
`unsupported` cells are data, not gaps (#152). Sequence lengths follow the
family: masked and ELF/FMLM at their frozen lengths, hybrid on both sides of
its 2048 crossover including 1024 and ≥2048.

## 6. What Stage 1 may and may not conclude

Stage 1 produces profiles and nothing else. It may not select a target, propose
a kernel, or change any dispatch default — Stage 2 owns selection against its
six criteria, and a candidate must be a material share of a *frozen
representative cell*, not of a microbenchmark.

The available verdicts are fixed in advance: `TARGET GO — <operation>`,
`EXISTING PATHS ADEQUATE`, `UPSTREAM/BACKEND FIRST`, or `UNDECIDABLE`. A GO must
name family, checkpoint/config, batch, length, dtype and hardware.

## 7. Mutation targets for the instrumentation

The harness is not trusted until these are killed:

- kernel-only gain reported as end-to-end gain;
- one-trial timing accepted;
- warmup/compile charged to only one arm;
- candidate and reference given different inputs or RNG;
- `operation_sum` substituted for the outer wall-clock;
- `unattributed_seconds` dropped, or shares normalized to 100%;
- the two ELF auxiliary forwards merged into one event;
- FMLM state allocation omitted from accounting;
- endpoint/codec cost omitted;
- SC-CFG forward count under-reported;
- peak memory recorded per record rather than per (cell, batch);
- instrumentation overhead unreported.
