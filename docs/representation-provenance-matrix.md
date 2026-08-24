# #156 Part A — representation provenance / confound matrix

**Status: observational. This matrix cannot produce a representation
winner and does not attempt one.** Its purpose is to make the confounds
between the three representation families explicit *before* Part B
diagnostics and the Part C matched causal gate are designed.

Every value is labelled by provenance:

| label | meaning |
|---|---|
| **canonical** | measured by Unturtle under the #152 protocol v1 (1000 samples, seed 42, corpus-pooled GenPPL under a pinned evaluator). **"canonical" is a FORM, not a guarantee of cross-row comparability** — see the status column below |
| **official** | measured by Unturtle using the *paper's own* metric code/semantics, which differ from canonical (see "Why the columns differ") |
| **paper-only** | quoted from a publication; NOT reproduced here, protocol unknown or known to differ |
| **frozen readout** | an Unturtle gate result frozen in an earlier issue, under that issue's own protocol |

Sources: `benchmarks/results/*/frontier_record.jsonl` (#153, #155, #165),
issue #130 (LaDiff gates), and the cited papers.

### Comparability typing (two orthogonal fields, not one enum)

A cell being "canonical" says it was produced by the #152 protocol. It does
NOT say two canonical cells are comparable, because the protocol itself was
amended mid-programme (the evaluator pin in #165, the ragged guards in
#167).

Comparability is therefore **two fields**, because the reasons a cell fails
to be comparable are independent and can co-occur — a single five-level
enum cannot express "canonical form, unknown evaluator snapshot, old guard
semantics" without inventing a rank ordering that does not exist.

**Field 1 — `measurement_status`** (exactly one, mutually exclusive):

| value | meaning |
|---|---|
| `EXACT` | the value is exact under the current frozen protocol |
| `BOUNDED_APPROXIMATE` | the value is stated with an arithmetic error bound |
| `NOT_COMPARABLE` | the value cannot enter a cross-row comparison as it stands |

**Field 2 — `comparability_flags`** (zero or more, orthogonal):

| flag | meaning |
|---|---|
| `PROTOCOL_COMPATIBLE` | produced by the #152 protocol (form, not identity) |
| `IDENTITY_UNRESOLVED` | evaluator and/or tokenizer revision cannot be resolved to a commit (e.g. measured under `main`) |
| `GUARD_INCOMPLETE` | guard values missing, or computed under the pre-#167 padded semantics |
| `PROTOCOL_DEVIATION` | a frozen condition (samples, seed, length, steps) differs |
| `TOKENIZER_MISMATCH` | the generation tokenizer differs from the comparison set's |

**Decision rule, with precedence** — applied top to bottom, first match wins:

1. `IDENTITY_UNRESOLVED` or `PROTOCOL_DEVIATION` present → `NOT_COMPARABLE`
   (an unnamed evaluator snapshot cannot be compared, however close the
   number probably is);
2. else `GUARD_INCOMPLETE` present → `NOT_COMPARABLE` **for the affected
   metrics only**; the unaffected metrics keep their own status;
3. else any metric stated with a bound → `BOUNDED_APPROXIMATE`;
4. else → `EXACT`.

Rule 2 is per-metric on purpose: Sumi's GenPPL is exact while its guards are
bounded, and collapsing that into one row-level label would either discard a
usable number or overstate a bounded one.

---

## A1 — structural provenance (no quality values)

| | ELF-B | FLM-B | FMLM-B | MDLM-OWT | Sumi-7B | GPT-2 medium | #130 LaDiff |
|---|---|---|---|---|---|---|---|
| family | embedding flow | one-hot flow | flow map | masked discrete | uniform discrete | AR | learned latent |
| denoiser state | continuous, T5 embedding space | continuous over one-hot simplex | continuous over one-hot simplex | discrete token ids + MASK | discrete token ids | discrete token ids | continuous latent |
| state dim / position | **128** (bottleneck) | **50 258** (vocab) | **50 258** (vocab) | 1 (id) | 1 (id) | 1 (id) | latent dim (see #130) |
| **mathematical** state dim @ L=1024 | 131 072 reals | 51.5 M reals (**392× ELF**) | 51.5 M reals | 1024 ids | 1024 ids | 1024 ids | compressed, see #130 |
| **materialized** activation bytes | not measured | not measured | not measured | not measured | not measured | not measured | not measured |
| **peak runtime memory** (from the records) | 4.89 GB (32 steps) / 8.80 GB (64) | 25.2 GB | 16.0 GB (1 step) / 25.2 GB (32) | 19.2 GB | 0.008 GB † | 3.74 GB | not recorded |
| sequence compression | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **< 1.0** (only compressing arm) |
| tokenizer / encoder | **t5-small encoder**, T5 vocab 32 100 | gpt2 BPE 50 258 | gpt2 BPE 50 258 | gpt2 BPE 50 258 (+MASK) | own, 100 278 | gpt2 BPE 50 257 | gpt2 BPE 50 258 |
| denoiser params | **104 579 940** | **169 627 250** | **169 676 658** ‖ | **169 627 250** | ~7 B | 355 M | see #130 |
| backbone | DiT, width 768, depth 12 | DiT, width 768, depth 12 | DiT, width 768, depth 12 | DiT, width 768, depth 12 | 36 layers, hidden 4096 | GPT-2, 24 layers | MDLM-DiT + codec |
| separate decoder / prior | endpoint token projection only (**no learned prior**) | none (state IS vocab space) | none | none | none | none | **yes: codec + latent prior** |
| training data | OpenWebText-T5 | OpenWebText | OpenWebText | OpenWebText | ~1.5 T tokens, other corpus | WebText | OpenWebText (#130 split) |
| training budget | **49.9 B token-pres** (95 085 steps × 512 × 1024, 5 epochs; read from the checkpoint) | not published in the repo | not published in the repo | not published in the repo | ~1.5 T tokens (card) | not comparable (GPT-2 era) | **AE 1.0 B + prior 2.0 B token-pres / seed** (frozen gate budget, ~1–2 % of the paper's ~105 B) |
| provenance of the above | config.yml + checkpoint | checkpoint | checkpoint | checkpoint | config.json | config | #130 frozen |
| **pinned revision** | `146f84133c1389bfd4ef47f14ec7a955da22faa7` | `624471b9` ‡ | `483ea1b3` ‡ | `d0958fa851335ece6c15260ce0025f030673c0fb` | `0d20f7becf84340b8a8d71a8dda577a502a5c8dd` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | as recorded in #130 |

‖ FMLM-B carries +49 408 parameters over FLM-B — exactly its flow map's
second time-conditioning MLP (`sigma_map_prime`), see the decomposition
section. FLM-B and MDLM-OWT are byte-identical to each other.

‡ The FLM/FMLM revisions are the abbreviated forms the #155 records carry.
Part B requires a full SHA at extraction time and treats an unresolvable
revision as a **blocked cell**, so these must be expanded before any Part B
measurement — an abbreviated revision is not an identity for the same reason
`main` is not (#165).

† Sumi's 0.008 GB is an artefact of WHERE its producer sampled
`max_memory_allocated`, not a property of the model — a 7 B bf16 model
cannot run in 8 MB. The counter was reset after warmup and read after the
model had been deleted and the cache emptied, so it measures the wrong
interval. Treat it as `NOT_COMPARABLE`.

The other peak figures are real but still not comparable to each other:
they were captured at different points by different producers, at different
generation batch sizes (ELF/FLM/FMLM at their pack defaults, MDLM at 16,
AR at 16, Sumi at 12), and peak memory scales with batch. They are recorded
because they are what the runs measured — not because they support a
representation comparison.

**Materialized activation bytes were never measured for any arm.** The
mathematical state dimensionality (row 1) is a property of the
representation; rows 2 and 3 are properties of an implementation, a batch
size and a measurement point. This matrix keeps them apart rather than
letting "392× larger state" be read as "392× more memory" — an
implementation may never materialize the dense one-hot tensor at all.

### What can be said about the trunks

FLM-B, FMLM-B and MDLM-OWT are **169.6 M-class DiT checkpoints with the same
width (768), depth (12) and conditioning size (128)** — measured from the
three checkpoints, not assumed. The precise claim is:

> **denoiser-trunk architecture matched; training budget and objective
> unmatched.**

Their objectives differ (one-hot Euclidean flow / flow map / masked discrete
diffusion) and only ELF's budget is even known, so "matched" covers the
trunk shape and nothing else.

### Module-level parameter decomposition (measured; every column reconciles to its total)

Exact tensor counts, read from the checkpoint shards. Every group is listed
so the columns sum to the totals with no residual:

| module group | FLM-B / MDLM-OWT | ELF-B |
|---|---|---|
| denoiser trunk (blocks) | 92 132 352 | 85 049 856 |
| vocabulary-directed: input embedding | 38 598 144 | 0 |
| vocabulary-directed: output head | 38 847 314 | 16 467 300 |
| state input projection | 0 | 393 728 |
| state output projection | 0 | 394 496 |
| encoder-side text projection | 0 | 164 608 |
| timestep / conditioning / self-cond | 49 408 | 2 109 952 |
| positional | 32 | 0 |
| unclassified | 0 | 0 |
| **total** | **169 627 250** | **104 579 940** |
| **buckets sum to total** | ✅ | ✅ |

**FLM-B and MDLM-OWT decompose byte-identically** — measured: their tensor
name → size maps are equal as sets, so every group above matches exactly.
That is what "same trunk architecture" means concretely here.

**FMLM-B is NOT byte-identical to them**, and the difference is exactly the
flow-map machinery:

| | count |
|---|---|
| FLM-B total | 169 627 250 |
| MDLM-OWT total | 169 627 250 (identical to FLM-B) |
| FMLM-B total | **169 676 658** |
| FMLM-B − FLM-B | **+49 408** |

The +49 408 is four tensors present only in FMLM:
`sigma_map_prime.mlp.{0,2}.{weight,bias}` (32 768 + 16 384 + 128 + 128) —
the **second time-conditioning path** a flow map needs, since FMLM is
conditioned on two times rather than one (#155). No shared tensor differs in
size and no FLM tensor is missing from FMLM.

So the accurate structural claim across the three is:

> **FLM-B and MDLM-OWT are byte-identical in their parameter decomposition.
> FMLM-B differs from them by exactly +49 408 parameters, all of it the
> flow map's second time-conditioning MLP.** Training budget and objective
> remain unmatched for all three.

**Reconciled difference accounting** (exact, no rounding residual):

| | count |
|---|---|
| total difference (FLM/MDLM − ELF) | **65 047 310** |
| vocabulary-directed modules (77 445 458 − 16 467 300) | **+60 978 158** |
| denoiser trunk (92 132 352 − 85 049 856) | **+7 082 496** |
| ELF-only modules (state projections, text projection, extra conditioning) | **−3 013 344** |
| sum of the three lines | **65 047 310** ✅ |

An earlier version of this table reported ELF's output head as 0.39 M. That
was wrong: it counted only `final_layer` and missed `unembed_kernel` +
`unembed_bias` (16 467 300 = T5 vocab 32 100 × 512). ELF *does* carry a
vocabulary-directed head; it is simply 2.4× smaller than FLM/MDLM's because
the T5 vocabulary is 32 100 rather than 50 258 and the hidden width feeding
it is 512 rather than 768.

The safe claim is therefore:

> **The bulk of the total parameter difference is located in the
> vocabulary-directed input/output modules (≈61.0 M of 65.0 M). The
> remainder is distributed across the trunk (+7.1 M) and ELF-only
> state/conditioning modules (−3.0 M). This is a decomposition of parameter
> LOCATION, not a causal attribution of any quality difference** — the two
> checkpoints were trained by different groups, on different objectives,
> with only one of the two budgets even known (C1).

---

## A2 — quality and compute, by provenance column

### Unturtle canonical (#152 protocol v1)

| method | NFE | GenPPL | MAUVE | entropy | distinct | pooled-H | unique rows | `measurement_status` | `comparability_flags` |
|---|---|---|---|---|---|---|---|---|---|
| ELF-B | 32 | 24.27 | 0.9308 | 6.926 | — | — | — | `NOT_COMPARABLE` | `PROTOCOL_COMPATIBLE`, `IDENTITY_UNRESOLVED`, `GUARD_INCOMPLETE` |
| ELF-B | 64 | 19.01 | 0.9149 | 7.015 | — | — | — | `NOT_COMPARABLE` | same |
| FLM-B | 1024 | 62.08 | 0.9127 | 6.494 | — | — | — | `NOT_COMPARABLE` | same |
| FMLM-B | 32 | 45.01 | 0.9545 | 6.120 | — | — | — | `NOT_COMPARABLE` | same |
| FMLM-B | 1 | 166.41 | 0.2342 | 5.932 | — | — | — | `NOT_COMPARABLE` | same |
| MDLM-OWT | **129** | 122.84 | 0.8744 | 7.561 | 0.5314 | 7.553 | 1.000 | `EXACT` | `PROTOCOL_COMPATIBLE` |
| Sumi-7B | 128 | 56.56 | 0.7478 | 7.189 | 0.4553 | 7.773 | 1.000 | `EXACT` (GenPPL/MAUVE/entropy) · `BOUNDED_APPROXIMATE` (guard trio) | `PROTOCOL_COMPATIBLE` |
| GPT-2 medium | 1024 | 202.31 | 0.8999 | 8.436 | 0.6933 | 8.432 | 1.000 | `EXACT` | `PROTOCOL_COMPATIBLE` |

**Why the flow rows are `NOT_COMPARABLE`.** Two separate reasons, and
either alone is disqualifying:

1. *evaluator identity unresolved* — they were measured with
   `hf_causal_evaluator("gpt2-large", revision="main", ...)`. `main` moves,
   so the evaluator snapshot that produced 24.27 cannot be named. The
   current frozen evaluator is `gpt2-large@32b71b12589c2f8d625668d2335a01cac3249519`
   (pinned in the #165 Stage-0 amendment). The GenPPL gap between the two
   is probably small, but "probably small" is not an identity, and GenPPL is
   defined only relative to its evaluator.
2. *guards incomplete* — those records' `distinct` / `pooled-H` /
   `unique-rows` exist but under the pre-#167 padded semantics. On the AR
   arm that difference moved `distinct_fraction` from 0.4289 to 0.6933
   (+62 %), so listing them beside corrected values would invite exactly the
   comparison the status vocabulary exists to prevent. They are left blank
   rather than shown with a footnote.

Moving the flow rows to `measurement_status = EXACT` with only
`PROTOCOL_COMPATIBLE` set requires re-running the #153/#155
cells against the pinned evaluator with ragged guards. That is a
measurement, not a matrix edit, and it is **not** part of Part A.

Sumi's guards are `BOUNDED_APPROXIMATE`: measured before the ragged guards,
with an absolute error bound of 7.83e-05 on `distinct_fraction` (see
`PADDING_BIAS_BOUND.json`; the true value lies in [0.455191, 0.455347]). Its
GenPPL/MAUVE/entropy are `measurement_status = EXACT` — only the guard trio
is bounded, which is what per-metric precedence (rule 2) is for.

### Official (paper's own metric code, run by Unturtle)

| method | NFE | GenPPL official | entropy official | seeds | semantics |
|---|---|---|---|---|---|
| ELF-B | 32 | 24.315 | 5.163 | 42 | ELF Metrics: first-EOS masking, **mean per-sample** unigram entropy over gpt2-large re-tokenized ids |
| ELF-B | 64 | 19.058 | 5.071 | 42 | same |
| FLM-B | 1024 | 61.612 | 5.333 | median of 42/43/44 | FLM Metrics: first-EOS masking, entropy = mean per-sample unigram over **NATIVE gpt2 ids** (pre-decode) |
| FMLM-B | 32 | 44.758 | 5.178 | median of 42/43/44 | same |
| FMLM-B | 1 | 165.976 | 5.167 | median of 42/43/44 | same |

The FLM/FMLM official cells are **per-seed medians** over seeds 42/43/44
(derived as `seed * 1_000_003 + offset`, after the #155 review found that
`seed + offset` made the arms overlap at small batch sizes). The ELF cells
are single-seed. That asymmetry is a protocol difference between the two
packs, not a property of the methods.

### Paper-only (not reproduced here)

| source | claim | why it cannot enter the canonical column |
|---|---|---|
| LDLM, arXiv:2605.07933 | naive joint encoder/diffusion/decoder training is poor; MSE decoder loss, diffusion→encoder warmup, adaptive timestep sampling and decoder-input noise materially change generation | no checkpoint reproduced in Unturtle; its metric protocol is not the #152 one. Enters #156 as **hypothesis material for H2**, not as a comparable number |
| Cola DLM, arXiv:2605.06548 | hierarchical Text-VAE → continuous latent prior → conditional decoder scales, with matched ~2 B AR/LLaDA evidence | ~2 B scale and a different corpus/tokenizer; useful as external scaling evidence, not as a small-scale causal control |
| ELF, arXiv:2605.10938 | embedding-space FM avoids the learned latent-prior → decoder decomposition | the *mechanism claim* is what #156 tests; the paper's numbers are superseded here by the reproduced official + canonical columns |
| FLM/FMLM, arXiv:2602.16813 | one-hot Euclidean flow; FMLM distils sequence transport into a few-step flow map | same — reproduced columns exist, so paper numbers are not quoted |

### Frozen readouts (#130 LaDiff, real text)

| gate | result | measured values |
|---|---|---|
| Gate A (mechanism) | **PASS** | true latent improves masked-position NLL/recovery over latent-dropout at t ∈ {0.75, 0.9, 1.0}; benefit vanishes with wrong/shuffled latents; benefit grows with mask ratio |
| Gate B (generation) | **FAIL**, decidable, same sign on both seeds, no collapse | MAUVE at the decision cells: seed 0 N=64/128 and seed 1 N=64/128 all favour `latent_off`; e.g. seed 1 N=128 LaDiff **0.575** vs latent_off **0.806**; full grid N=32 seed 0: 0.385 / 0.770 / 0.814 (LaDiff / off / gaussian) |
| localization | prior-sample / off-manifold gap | NOT decoder liveness (Gate A passed), and NOT a claim that latent methods never work |

#130 stays frozen. #156 is a new hypothesis/regime, not a retry.

---

## A3 — why the official and canonical columns differ

Three entropy semantics are in play and must never be averaged together:

1. **ELF official** — mean per-sample unigram entropy over **gpt2-large
   re-tokenized** ids, after first-EOS masking;
2. **FLM official** — mean per-sample unigram entropy over the model's
   **native gpt2 ids**, before decoding;
3. **#152 canonical** — corpus-**pooled** entropy over decoded text under
   one common tokenizer.

GenPPL differs for the same reason: the canonical column is corpus
token-weighted `exp(total_nll / total_tokens)` under a pinned evaluator,
while each paper's harness applies its own masking and aggregation. The
gaps are small here (ELF 24.27 vs 24.315; FLM 62.08 vs 61.61) but they are
*not* noise — they are different estimators, and mixing them would silently
average two definitions.

---

## A4 — confounds that block any representation claim from this matrix

| # | confound | why it blocks a causal read |
|---|---|---|
| C1 | **training budget unknown or unequal** | only ELF (49.9 B) and #130 (1–2 B) have known budgets. FLM/FMLM/MDLM publish none in their repos. A representation difference cannot be separated from a budget difference that is not even measured. |
| C2 | **tokenizer / encoder differs** | ELF transports in a t5-small embedding space with a 32 100 vocab; the others use gpt2 BPE 50 258; Sumi uses its own 100 278. GenPPL under a common evaluator partially normalizes the *output* comparison but not the *task* the model was trained on. |
| C3 | **parameter count differs with the interface** | ELF 104.6 M vs 169.6 M is itself a consequence of the representation (128-dim bottleneck vs 50 258-wide state). Matching parameters and matching representation are in tension — Part C must report the decomposition rather than claim exact matching. |
| C4 | **NFE is not one scalar across families** | AR bills 1024 token-forwards; MDLM 129 denoiser calls (128 + upstream `noise_removal`); ELF/FMLM 32 or 1 solver steps. Equal NFE does not mean equal compute, and the throughput cells carry per-cell executed work for that reason. |
| C5 | **scale outliers** | Sumi (~7 B, ~1.5 T tokens) and GPT-2 medium (355 M, WebText) are anchors, not controls. Their rows exist to bound "what a competent model of that family does", not to rank representations. |
| C6 | **#130's arms are not this matrix's arms** | #130 measured a *specific* codec+prior pipeline at a 1–2 B gate budget, ~1–2 % of the paper's ~105 B. Its FAIL localizes to the prior-sample/off-manifold gap in that configuration. Reading it as "learned latents lose to embedding flow" would compare a 1 % budget arm against a 49.9 B arm. |
| C7 | **guard semantics changed mid-programme** | the flow-family guard values predate the ragged content-only guards (#167 review 2), where the AR arm moved `distinct_fraction` 0.4289 → 0.6933. Guard columns are therefore not comparable across the canonical rows until #153/#155 cells are re-run. |

---

## A5 — what Part A licenses, and what it does not

**Licensed:**

- the observation that **FLM-B and MDLM-OWT are byte-identical** in their
  parameter decomposition (169 627 250 each), which makes that specific pair
  the cheapest to extend into a matched Part C arm; FMLM-B sits +49 408
  away, all of it its flow map's second time-conditioning MLP, so it is a
  near-match rather than an exact one;
- the observation that the embedding arm reaches decision-grade quality at a
  **392× smaller mathematical state dimensionality** and a smaller parameter
  count, with **60 978 158 of the 65 047 310 difference (≈61.0 M of 65.0 M)
  located in the vocabulary-directed modules** — this is the concrete form
  H1's "interface burden" question takes, and it is a location, not a cause;
- the observation that FMLM's 1-step cell (GenPPL 166.41, MAUVE 0.2342)
  versus its 32-step cell (45.01, 0.9545) is a *within-method* NFE effect,
  free of cross-family confounds — the one comparison in this matrix that no
  confound above touches;
- the identification of C1 (unknown budgets) as the confound Part C must fix
  by construction, since it cannot be corrected after the fact.

**Not licensed:**

- any statement of the form "representation X beats representation Y";
- any aggregate score across families, or any ranking;
- treating #130's Gate B FAIL as evidence about learned latents in general,
  as opposed to that pipeline at that budget;
- comparing an official-column number with a canonical-column number.

## Next — Part B

Part B is frozen in
[`representation-interface-diagnostics-protocol.md`](representation-interface-diagnostics-protocol.md):
diagnostic formulas, units, sample ids, the disjoint reference split, the
normalized corruption axis per family, the perturbation grid, controls,
primary/secondary split, CI method, failure semantics, the amendment rule,
and the fake-state test design — all committed before any measurement.

The confounds in §A4 are what shaped it. In particular C1 (unknown training
budgets) is why the Part B diagnostics are all computable *without*
generation: they characterize the interface a checkpoint learned, which is
observable, rather than trying to infer a causal effect from budgets that are
not recorded.
