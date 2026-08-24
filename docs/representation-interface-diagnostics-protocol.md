# #156 Part B — preregistered interface-diagnostics protocol

**Frozen before any diagnostic is computed.** Part B's completion condition
is not a good result; it is this document existing, committed, with every
formula, grid, control, split, statistic and failure rule fixed *before* the
first measurement. #156 forbids adding a diagnostic because it correlates
with a desired answer after the run, and this document is what makes that
rule checkable: anything not listed here is `exploratory` and cannot feed a
Part C gate.

Depends on: `docs/representation-provenance-matrix.md` (Part A) for the
confounds these diagnostics are designed around.

**Status: protocol only. No diagnostic code, no measurement.**

---

## 1. Frozen common inputs

### 1.1 Checkpoints and revisions

| family | repo | revision | tokenizer / encoder revision |
|---|---|---|---|
| embedding flow | `embedded-language-flows/ELF-B-owt-torch` | `146f84133c1389bfd4ef47f14ec7a955da22faa7` | `t5-small` encoder, revision pinned at extraction time and recorded |
| one-hot flow | `david3684/FLM-B-OWT` | `624471b9` | `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| flow map | `david3684/FMLM-B-OWT` | `483ea1b3` | same as FLM |
| masked discrete (anchor) | `kuleshov-group/mdlm-owt` | `d0958fa851335ece6c15260ce0025f030673c0fb` | same as FLM |
| learned latent | #130 frozen AE + prior checkpoints, both seeds | as recorded in #130 | `openai-community/gpt2@607a30d783df…` |

Any revision not resolvable to a full SHA at extraction time is a **blocked
cell**, not a cell measured against `main`. (This is the #165 lesson: `main`
moves, so a value recorded against it has no identity.)

### 1.2 Held-out data

- source: the #130 OWT held-out artifact `dev/local/owt/heldout_1024`
  (110 520 rows × 1024 gpt2 ids, OWT snapshot
  `79d93d786212f7344586290adb811d4ae6a1762c`, `mdlm _group_texts` packing);
- **artifact SHA-256 is recorded in every diagnostic record** — the file is
  gitignored, so its identity travels with the measurement, not with the
  repo;
- **diagnostic rows: the FIRST 1024 rows, row ids `[0, 1024)`**, taken in
  file order (no shuffle, no sampling) so the set is reproducible from the
  artifact alone;
- sequence length **1024**, no truncation, no re-packing;
- these 1024 rows are the *only* rows any diagnostic may read.

### 1.3 Reference / whitening split — disjoint from the diagnostic rows

Whitening statistics and the nearest-latent reference bank are computed from
rows **`[1024, 5120)`** of the same artifact (4096 rows), never from the
diagnostic rows. Using held-out rows for both the statistic and the
measurement leaks, and the leak flatters exactly the latent diagnostics this
study is meant to test.

- reference bank size: **4096 latents** (one per reference row);
- whitening: mean and per-dimension standard deviation over the reference
  rows, computed once, recorded in the record, and reused unchanged for every
  arm;
- if an arm cannot produce latents for the reference rows, its latent
  diagnostics are `blocked`, not computed against a different bank.

### 1.4 Numerics

- dtype: **fp32 for every diagnostic**, regardless of the checkpoint's
  training dtype (a diagnostic that changes with autocast is not measuring
  the model);
- device: recorded per record; a diagnostic must be device-independent to
  within 1e-4 relative, and that is asserted on one CPU/CUDA pair per family;
- attention backend recorded (`sdpa` / `flash` / `math`);
- `torch.use_deterministic_algorithms(True)` where the family's ops support
  it; where they do not, the record says so.

### 1.5 Corruption axis — normalized, not raw `t`

Raw `t` is not comparable across families: ELF uses logit-normal time with
`P_mean = -1.5`, FLM/FMLM use a tau↔t LUT reparameterization, MDLM uses a
loglinear schedule. Every family therefore reports on a **normalized
corruption axis**, and the mapping is declared here:

**`q = t` is a fact about MDLM's loglinear masking schedule, not a general
rule.** #165 verified that under loglinear noise MDLM's `move_chance_t` *is*
`t`, so for that family the masking probability and the time variable
coincide. Nothing licenses transplanting that identity to ELF's
logit-normal embedding-space time, to FLM/FMLM's LUT-reparameterized `tau`,
or to a learned-latent prior's timestep. Each family gets its own axis
definition, and a family whose mapping cannot be derived keeps its native
axis.

**Sumi is not a Part B family.** It has no diagnostic section here; it
appears in Part A as the observational `uniform_state` anchor and nothing in
this protocol measures it. Bringing it into Part B would require freezing
its corruption-severity definition first (its state is a uniform-noise
canvas, not a masked or embedding one), which is a protocol amendment, not
an implementation detail.

| family | native axis | normalized axis | mapping | status |
|---|---|---|---|---|
| **MDLM** | loglinear `t` | corruption quantile `q` | **`q = move_chance_t = t`** — verified for this schedule (#165) | `DERIVED` |
| **ELF** | `t ∈ (t_eps, 1)`, logit-normal sampling | corruption quantile `q` | only if the pack's own `alpha(t)` yields a monotone severity map: `q = 1 - alpha(t)`, computed from the pack's schedule code and recorded numerically per point. **If that derivation cannot be shown from the code, ELF keeps `native_t`.** | `TO BE DERIVED BEFORE MEASUREMENT` |
| **FLM / FMLM** | `tau` → `t` via the LUT | corruption quantile `q` | only if the LUT-mapped `t` gives a monotone severity map: the state is `(1-t)·ε + t·onehot`, so `q = 1 - t` is the mass on noise — but the LUT must be read and the monotonicity checked, not assumed. **Otherwise `native_tau`.** | `TO BE DERIVED BEFORE MEASUREMENT` |
| **learned latent** | prior/diffusion timestep | corruption quantile `q` | must be defined on **standardized latent corruption** (the whitened-space signal-to-noise ratio), never by calling the training timestep `q`. A timestep is an index into a schedule; corruption severity is a property of the state. | `TO BE DERIVED BEFORE MEASUREMENT` |

`DERIVED` means the mapping is established and citable now. `TO BE DERIVED
BEFORE MEASUREMENT` means the derivation is part of the work authorized
*before* the first diagnostic runs — and if it fails for a family, that
family falls back to `native_*` rather than borrowing another family's
identity.

**Frozen grid, for families with a derived quantile mapping:**

```
q ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}
```

The grid points are recorded numerically in each record together with the
native value they came from, so the mapping is auditable rather than
asserted.

**A family on a native axis is excluded from every cross-family AUC.** Its
curve is still recorded — under `axis.kind = "native"` with the native
variable named — and it may be compared *within* that family across
checkpoints or step counts. It never shares an AUC column with a family
whose x-axis means something else.

### 1.6 Controlled perturbation grid — ELF / FLM / FMLM only

Perturbation applies to the three continuous-state families (see the scope
table under "Perturbation robustness" in §2.1). MDLM has no continuous state
to perturb, Sumi has no Part B diagnostic section, and the learned-latent
arm's interface stress is measured by its `wrong_latent` / `nll_prior`
controls rather than by a perturbation curve.

Perturbation magnitude is **relative to the clean state's own standard
deviation**, per family, computed over the reference split:

```
{0, 0.01, 0.03, 0.10, 0.30}   (relative sigma)
```

- perturbation is isotropic Gaussian in the family's state space;
- `0` is the unperturbed control and must be present in every curve;
- one fixed generator per (family, grid point), seeded `42`; the seed rule
  is `42` for the corruption draw and `9000 + round(1000 * rel_sigma)` for
  the perturbation draw, so the two are independent and reproducible;
- the same perturbation directions are reused across families where the
  state shapes allow, and where they do not, the record says the directions
  are family-local.

### 1.7 Stochastic seeds

- corruption / masking seed: **42**, one generator per (family, axis point),
  advanced per row so two rows never share a draw;
- perturbation seed: as above;
- no diagnostic may reseed globally; the seed rule is recorded per record.

---

## 2. Direct-state diagnostics

### 2.1 Common to ELF / FLM / FMLM

**Normalized denoising prediction error.** For a clean state `x0`, a
corrupted state `xq` at normalized level `q`, and the model's prediction
`x̂0`:

```
nde(q) = || x̂0 - x0 ||_2  /  || x0 ||_2
```

Frozen conventions:

- denominator is the **clean state norm**, per sequence, not per token;
- the reduction is **mean over sequences** of the per-sequence ratio (a
  per-token mean would weight short sequences differently, and every row
  here is exactly 1024 tokens, but the convention is fixed anyway);
- **zero denominator**: if `||x0|| == 0` for a row (possible only for a
  degenerate state), that row is reported as `nan` with
  `failure_reason: "zero_clean_norm"` and is **excluded from the mean but
  counted in `n_failed`**. It is never silently dropped.
- unit: dimensionless ratio.

**Endpoint token recovery.** Project the predicted state to tokens with the
family's own endpoint rule (ELF: nearest embedding / its decode head;
FLM/FMLM: argmax over the vocabulary axis) and compare to the clean tokens:

- `top1_recovery(q)` = fraction of positions where the projected token
  equals the clean token; unit: fraction;
- `endpoint_nll(q)` = mean per-position negative log-likelihood of the clean
  token under the family's endpoint distribution, in **nats**; families with
  no normalized endpoint distribution report `null`, not a substitute;
- `margin_top1_top2(q)` = mean over positions of
  `logit_top1 - logit_top2` in the endpoint distribution, in **nats**;
  reported alongside `top1_recovery` because a high recovery at a razor-thin
  margin is a different regime from a high recovery held confidently.

**Perturbation robustness.** For each relative sigma `s` in the perturbation
grid, at a **fixed** corruption level:

- `top1_recovery(s)` — the recovery curve under perturbation;
- `robustness_auc` = trapezoidal area of `top1_recovery(s)` over
  `s ∈ [0, 0.30]`, divided by `0.30` so it is a mean recovery over the
  perturbation range, in the same unit as recovery (fraction);
**Scope: `robustness_auc` is defined for ELF, FLM and FMLM only.** These are
the three families in this study with a continuous state that can be
perturbed by an isotropic Gaussian and then re-read through an endpoint
token projection. The others are deliberately excluded rather than given an
invented perturbation:

| family | perturbation diagnostics | why |
|---|---|---|
| ELF, FLM, FMLM | **in scope** | continuous state + endpoint token readout |
| MDLM | **out of scope** | no continuous state to perturb; contributes the discrete anchor of §2.4 (§2.4's own rule) |
| Sumi | **out of scope for Part B** | it appears in Part A as an observational `uniform_state` anchor and has no Part B diagnostic section; adding a perturbation for it would require freezing its perturbed object and readout first |
| learned latent | **out of scope for `robustness_auc`** | §3 defines no perturbation-recovery readout. Its interface stress is measured by the `wrong_latent` and `nll_prior` controls instead, which are already frozen there. A latent perturbation curve would need its perturbed object (whitened latent), decoder readout, metric and control frozen — that is a protocol amendment, not an implementation detail |

- **the fixed corruption level is `q = 0.50` for the in-scope families that
  have a derived corruption-quantile mapping** (§1.5). An in-scope family on
  a native axis uses a **pre-specified native midpoint**, declared here
  before any measurement, and its `robustness_auc` goes in a **separate
  column** — never averaged or ranked against quantile-mapped families:

  | in-scope family | fixed level if quantile-mapped | pre-specified native fallback |
  |---|---|---|
  | ELF | `q = 0.50` | `t = 0.5` on the pack's own time variable |
  | FLM / FMLM | `q = 0.50` | `tau = 0.5` on the LUT axis |

- both the level and the fallback are fixed *now*, so the robustness number
  cannot be quoted at whichever corruption level happens to separate the
  arms.

### 2.2 ELF-specific manifold proxy

- `nearest_embed_distance` — L2 from the predicted state to the nearest
  token embedding in the frozen T5 embedding table;
- `nearest_embed_distance_normalized` — the above divided by the **local
  embedding spacing**, defined as the mean L2 distance from that nearest
  embedding to its own `k = 8` nearest neighbours in the table (a fixed
  `k`, chosen now, not tuned);
- `nearest_token_margin` — L2 distance to the second-nearest embedding minus
  the distance to the nearest, normalized the same way;
- `clean_norm_deviation` — `(||x̂0|| - ||x0||) / ||x0||`, signed, so
  over- and under-shooting the embedding shell are distinguishable.

### 2.3 FLM / FMLM-specific manifold proxy

Their state is Euclidean over one-hot encodings, so the simplex is a proxy
for the clean manifold:

- `simplex_sum_residual` — `mean_positions | sum_v state[v] - 1 |`;
- `negative_mass` — `mean_positions sum_v max(0, -state[v])`;
- `simplex_projection_distance` — L2 from the state to its Euclidean
  projection onto the probability simplex;
- `nearest_vertex_distance` — L2 to the nearest one-hot vertex;
- `state_entropy` — entropy in **nats** of the simplex-projected state,
  per position, averaged (a state near a vertex has low entropy);
- `endpoint_token_margin` — as in §2.1.

### 2.4 MDLM anchor — deliberately NOT pushed into continuous-state form

MDLM has no continuous state to perturb; forcing it into §2.1 would invent a
quantity. It contributes an anchor row of discrete diagnostics only:

- `masked_position_ce` — mean CE in **nats** at masked positions;
- `masked_position_top1` — argmax accuracy at masked positions;
- `residual_mask_rate` — fraction of positions still masked at the end of
  the loop (structurally 0 under the frozen `noise_removal` config; recorded
  so a change is visible);
- `commit_accuracy` — of positions committed at step `k`, the fraction still
  equal to the clean token at the end;
- `revision_events` — as defined in `net_revision_stats` (#165), which
  measured 16 256 events over 16 256 revised positions for this checkpoint.

MDLM rows are never entered into a cross-family AUC over a continuous axis.

---

## 3. Learned-latent diagnostics

All computable without free generation. `NLL` values are mean per-position
negative log-likelihood of the clean token, in **nats**, at masked
positions, under the frozen #130 decoder.

| name | definition |
|---|---|
| `nll_true` | decoder NLL given the **true** latent (encoder output for that row) |
| `nll_off` | decoder NLL with the latent pathway **off** (the #130 latent-dropout control) |
| `nll_wrong` | decoder NLL given another row's latent (fixed derangement of the 1024 diagnostic rows, seed 42, so no row gets its own) |
| `nll_prior` | decoder NLL given a latent **sampled from the prior** |
| `true_latent_benefit` | `nll_off - nll_true` — positive means the true latent helps |
| `wrong_latent_discrimination` | `nll_wrong - nll_true` — positive means the decoder actually uses *which* latent it got |
| `prior_decoder_gap` | `nll_prior - nll_true` — **the #130 Gate B failure mode, as a pre-generation number** |
| `paired_whitened_latent_mse` | MSE between predicted and true latent after whitening with the reference-split statistics, paired per row |
| `latent_norm_mean` / `latent_norm_var` | over the diagnostic rows, whitened space |
| `covariance_spectrum` | eigenvalues of the whitened latent covariance, descending, recorded as a vector |
| `prior_true_spectrum_error` | relative L1 between the prior-sample and true-latent spectra, `sum|λp - λt| / sum λt` |
| `nearest_latent_distance_ratio` | for each row, L2 to its nearest neighbour in the **frozen reference bank**, divided by the mean nearest-neighbour distance *within* the bank — a manifold proxy that is ≈1 when a latent sits as close to the data as real latents do, and >1 when it is off-manifold |

Frozen choices that would otherwise be tunable after the fact: the
derangement seed (42), the reference bank rows (`[1024, 5120)`), the
whitening source (reference split), `k = 1` for nearest-latent distance, and
the masking used for every NLL (the §1.5 grid, same masks across all four
latent conditions — paired by construction).

---

## 4. Primary vs secondary

**Primary** (may inform a Part C gate):

1. normalized prediction-error curve and its AUC over the frozen grid;
2. endpoint token `top1_recovery` and `endpoint_nll`;
3. perturbation `robustness_auc` **(ELF / FLM / FMLM only)** at `q = 0.50`,
   or at the pre-specified native midpoint for an in-scope family without a
   derived quantile mapping — kept in a separate column from the mapped ones;
4. `prior_decoder_gap`;
5. `wrong_latent_discrimination`;
6. the family's fixed manifold-distance proxy
   (`nearest_embed_distance_normalized` / `nearest_vertex_distance` /
   `nearest_latent_distance_ratio`).

**Secondary** (recorded, reported, but not gate-bearing): norms, covariance
spectra, calibration, `margin_top1_top2`, simplex residuals, `state_entropy`,
MDLM anchor rows, and everything in §2.2/§2.3 not named primary above.

**No single interface score.** The primaries are kept as a **vector**. There
is no weighting that would make "one interface number" meaningful across an
embedding state, a simplex state and a learned latent, and inventing one
would smuggle in exactly the representation verdict Part A refuses to give.

---

## 5. Statistics

- **paired** comparisons only, over the same 1024 diagnostic rows;
- the **independent unit is the row** (one packed 1024-token sequence), not
  the token and not the position. Positions within a row are correlated;
  treating them as independent would shrink every interval by ~30×;
- **paired bootstrap, 10 000 resamples**, resampling rows with replacement;
- bootstrap seed **`20261156`** (fixed here);
- **95 % CI**, percentile method, reported for every primary;
- pre-specified summaries: **mean, median, p90** of the per-row values — all
  three, always, so a distributional shift cannot be reported as a mean
  shift or hidden by one;
- **failed / NaN / collapsed cells are never excluded.** Each record carries
  `n_total`, `n_failed`, and a `failure_reason` histogram; a summary computed
  over fewer rows than `n_total` states that explicitly. A cell that is
  entirely failed is reported as failed, not omitted;
- any metric added after results are seen is marked `exploratory: true` and
  **cannot** be used in a Part C gate. This is enforced by the record schema
  carrying the field, not by memory.

---

## 6. Architecture boundary

No shared `ContinuousState`, no shared `LatentState`. #156's own
architecture rule keeps representations method-specific and limits sharing
to diagnostics and records.

Implementation order, when implementation is authorized:

1. **pack-local extractor** — each pack computes its own diagnostics from its
   own state, with no cross-pack abstraction;
2. **plain JSONL diagnostic records** — one line per (family, method,
   checkpoint, sample, diagnostic, axis point);
3. **shared reducer only when two consumers need the same aggregation** — and
   then only the reducer, never the state.

The record schema is deliberately small:

| field | note |
|---|---|
| `family`, `method`, `checkpoint` | checkpoint includes the pinned revision |
| `sample_id` | the held-out row id |
| `diagnostic` | name from this document |
| `axis` | `{"kind": "corruption_quantile" \| "log_snr" \| "rel_sigma" \| "native", "value": float, "native_variable": str, "native_value": float, "mapping_status": "derived" \| "native_only"}` — a quantile-mapped point carries the native value it came from, so the mapping can be re-checked; a native-only point says so explicitly |
| `value`, `unit` | unit is mandatory; `nats`, `fraction`, `ratio`, `l2`, or `bytes` |
| `provenance` | dataset artifact SHA, tokenizer revision, dtype, device, backend, seeds |
| `status` | `ok` \| `failed` \| `blocked` \| `exploratory` |
| `failure_reason` | required when status ≠ `ok` |

No state objects are shared. No family's state type appears in another
family's code path.

---

## 7. Fake-state test design (design only; implementation deferred)

Metric semantics are testable without a checkpoint, and should be tested
that way first — a diagnostic that only ever runs on a real model cannot be
shown to measure what it claims.

Planned cases, each an assertion about a *known* answer:

| diagnostic | fake input | expected |
|---|---|---|
| `nde` | `x̂0 = x0` | exactly 0 |
| `nde` | `x̂0 = 0`, `x0 ≠ 0` | exactly 1 |
| `nde` | `||x0|| = 0` | `nan`, `failure_reason: zero_clean_norm`, counted in `n_failed` |
| `top1_recovery` | prediction = clean tokens | exactly 1 |
| `top1_recovery` | prediction = shifted tokens | exactly 0 |
| `robustness_auc` | recovery constant at `c` over the grid | exactly `c` |
| `simplex_sum_residual` | exact one-hot rows | 0 |
| `simplex_sum_residual` | rows summing to 1.5 | 0.5 |
| `negative_mass` | a state with one −0.2 entry per position | 0.2 |
| `nearest_vertex_distance` | state = a vertex | 0 |
| `state_entropy` | uniform over V | `ln(V)` |
| `nearest_embed_distance_normalized` | synthetic table with unit spacing | equals the raw distance |
| `wrong_latent_discrimination` | decoder that ignores the latent | 0 (and the test asserts the diagnostic can *detect* an indifferent decoder) |
| `prior_decoder_gap` | prior sample = true latent | 0 |
| `nearest_latent_distance_ratio` | diagnostic latents drawn from the bank's own distribution | ≈1 |
| `nearest_latent_distance_ratio` | latents shifted far off-manifold | ≫1 |
| paired bootstrap | identical arms | CI containing 0 |
| paired bootstrap | arms offset by a constant `d` | CI centred on `d`, not containing 0 |
| failure accounting | half the rows `nan` | `n_failed` = half, summaries state the reduced n |

Each case names the mutation it would catch — e.g. the constant-recovery
case kills an AUC that forgets to divide by the range, and the
`zero_clean_norm` case kills a silent `dropna`.

---

## 8. Amendment rule

This protocol may be amended **only** before the corresponding measurement
runs, and only by appending a dated amendment section that states what
changed and why — never by editing a frozen value in place. This mirrors the
#165 Stage-0 amendment discipline, where the evaluator pin was corrected
*before* any quality output and recorded as an amendment rather than a
silent edit.

An amendment made after seeing results marks every affected metric
`exploratory`, permanently, for this study.

---

## 9. Completion condition (this document)

Part B is frozen when all of the following are committed, before any
measurement:

- [x] diagnostic formulas, with denominators and reductions fixed
- [x] units for every metric
- [x] sample ids (`[0, 1024)`) and the disjoint reference split (`[1024, 5120)`)
- [x] corruption grid, with the per-family axis definition table and the
      explicit rule that `q = t` is MDLM-specific (mappings for the other
      families are `TO BE DERIVED BEFORE MEASUREMENT`, with native-axis
      fallbacks and exclusion from cross-family AUC)
- [x] perturbation grid, defined relative to clean-state sigma
- [x] controls (`latent off`, `wrong latent`, `s = 0`, MDLM anchor)
- [x] primary / secondary split, with no aggregate score
- [x] CI method, resamples, seed, and the pre-specified summaries
- [x] failure semantics (nothing excluded, `n_failed` mandatory)
- [x] amendment rule
- [x] fake-state test design

Still **NO-GO**: diagnostic code, real-checkpoint measurement, and any Part C
arm implementation.
