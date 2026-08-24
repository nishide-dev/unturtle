# #157 Part 0 — parallel-decoding reference audit

**Docs only. No implementation, no measurement, no candidate selection.**

#157 forbids porting before this audit identifies, per candidate, a
**reproducible reference** and the **exact falsifiable claim** it could test
against a frozen baseline. This document is that gate. It states what each
candidate is, what can actually be obtained today, who would own it in
Unturtle, and one claim that a later baseline run could refute.

Scope is #157's: generation topology, parallel commit, KV-cache semantics.
General kernel profiling and ELF/FMLM optimization belong to #166 and are
out of scope here.

Revisions pinned at audit time — an unresolvable revision makes a candidate
`BLOCKED`, not "measured against latest":

The **canonical identity is the upstream repository and commit**;
`dev/repos/` is a local untracked clone (gitignored, absent from this PR's
tree) and is named only as where the audit read the code:

| artifact | canonical identity | read locally at |
|---|---|---|
| Fast-dLLM v1 + v2 code | `NVlabs/Fast-dLLM@a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`, paths `v1/` and `v2/` | `dev/repos/fast-dllm` (untracked clone) |
| Fast-dLLM v2 weights | `Efficient-Large-Model/Fast_dLLM_v2_7B@0661abf5f9f0ee338970d091052a26c8efa51974` | HF cache |
| v1 host model | `GSAI-ML/LLaDA-8B-Instruct@08b83a6feb34df1a6011b80c3c00c7563e963b07` | HF cache |
| v1 host model | `Dream-org/Dream-v0-Instruct-7B@05334cb9faaf763692dcf9d8737c642be2b2a6ae` | HF cache |
| dKV-Cache code | `horseee/dKV-Cache@49a76fcc43b744ec2d960137f216e419317138b1` | not cloned |

---

## Summary

| candidate | state | why |
|---|---|---|
| 1. Unturtle `block_decode` | **REFERENCE_READY** | in-tree, already exercises block cache + confidence-aware parallel decode; the natural frozen baseline |
| 2. Set Diffusion (arXiv:2607.01775) | **BLOCKED** | no official code or weights locatable at audit time; nothing to reproduce |
| 3. Fast-dLLM v2 (arXiv:2509.26328) | **REFERENCE_READY** | Apache-2.0 code at a pinned upstream commit + official 7B checkpoint at a pinned SHA; hierarchical cache is implemented, not just described |
| 4. Fast-dLLM v1 training-free cache (arXiv:2505.22618) | **REFERENCE_READY** | Apache-2.0 code at a pinned upstream commit; training-free, so its "artifact" is code + a host model (LLaDA / Dream, both pinned) |

Additional named caches from the same slot, audited separately:

| candidate | state | why |
|---|---|---|
| dKV-Cache (arXiv:2505.15781) | **BLOCKED — LICENSE UNRESOLVED** | the artifact **does** exist: `horseee/dKV-Cache` (xML Lab / NUS, first author Xinyin Ma), head `49a76fcc43b744ec2d960137f216e419317138b1`, with Dream and LLaDA implementations and a `cache_steps` refresh interval. But the repository declares **no license** — no `LICENSE` file and no GitHub license metadata (checked via the API). Availability is not the blocker; redistribution and modification rights are |
| dLLM-Cache | **BLOCKED** | no locatable official artifact under that name at audit time |

**Count for the four original slots: 3 REFERENCE_READY, 1 BLOCKED** (Set
Diffusion). Including the two additional named caches: 3 ready, 3 blocked.
The training-free slot resolves to Fast-dLLM v1, with dKV-Cache blocked on
licensing rather than on existence.

---

## 1. Unturtle `block_decode` / cache path — REFERENCE_READY

**Code.** `unturtle/models/generation/diffusion_generation_utils.py`,
`_sample_with_cache`; registered as the `block_decode` algorithm in
`unturtle/models/generation/sampler.py` (`auto_priority` 30, flags
`use_cache=True, use_block_diffusion=False`).

**License / redistribution.** Repository core; no external condition.

**Official artifact.** Not applicable — this is the in-tree baseline. It runs
on any masked-diffusion backbone whose capability probe accepts it.

**Topology.** Fixed-size blocks over a masked-diffusion canvas. Generation
length must be divisible by `block_length` (the code rejects otherwise).
Bidirectional within the sequence; no set-causal or complementary mask.

**Training / conversion required.** None. It is a decode-time path over an
existing masked-diffusion checkpoint.

**Cache state, refresh, invalidation.** Read from the implementation, not
from its docstring's aspiration:

- one full forward per block produces `past_key_values`;
- the cache is then **trimmed to `current_block_start`** (`_trim_kv_cache`),
  i.e. only strictly-previous blocks are retained;
- **the cache is not updated during the denoising steps of a block** — it is
  built at the block boundary and held constant while that block denoises;
- invalidation is therefore positional and coarse: entering block *k+1*
  rebuilds from a forward over the whole sequence and re-trims.

**Commit / unmask policy.** `alg='origin'` commits by the
`p_transfer = 1 - s/t` schedule; the confidence-ordered variants
(`maskgit_plus`, `topk_margin`, `entropy`) commit the top-`n_transfer`
positions by confidence. `parallel_decode=True` with
`confidence_threshold` commits every position above the threshold in one
step — the Fast-dLLM v1 mechanism, already present. The code warns that
`alg='entropy'` + threshold degenerates (negative-entropy confidences never
reach a `[0,1]` threshold), which is a real trap and already guarded.

**Supported context / output length.** Whatever the host backbone supports;
`max_length` is caller-set. `output_history` is **not** supported on this
path.

**Reference metrics.** Native to Unturtle: this is where a frozen baseline
would be measured, so it has no external number to cite.

**Ownership boundary in Unturtle.** Core (`unturtle/models/generation/`). Any
change here affects every masked-diffusion family, so it is the one candidate
whose modification is not pack-local.

**Which checkpoints can actually run this path.** Verified against the
capability probe, not assumed: `_supports_block_decode` requires a callable
`_model_forward_with_cache` and refuses `hybrid_attention` models. Only
**LLaDA** and **Dream** implement that forward
(`unturtle/models/backbones/{llada,dream}/generation_utils.py`). MDLM-DiT and
ModernBERT set `supports_block_decode = False` explicitly.

Checked on the #165 checkpoint: `kuleshov-group/mdlm-owt` reports
`supports_block_decode = False`, has no `_model_forward_with_cache`, and
`find_algorithm("block_decode").supports(model)` returns **False** while
`mdlm` returns True. So a `mdlm` vs `block_decode` comparison on that
checkpoint is **not possible** — an earlier draft of this audit proposed
exactly that and was wrong.

**Paired comparison is available on LLaDA / Dream**, where both algorithms
are capability-valid on one checkpoint: `mdlm` (no cache) and `block_decode`
(cache + optional `parallel_decode`) differ only in the decode path, so the
comparison is same-checkpoint and same-weights. `GSAI-ML/LLaDA-8B-Instruct`
@ `08b83a6feb34…` is the natural pin, since it is already required for the
Fast-dLLM v1 slot.

**Falsifiable claim it can test.** *At batch 8 and 32 with 1024-token output
on one LLaDA checkpoint, `block_decode` with `parallel_decode` at a fixed
confidence threshold gives no steady-state wall-clock throughput advantage
over plain `mdlm` at matched quality and dependency correctness.*

The speed coordinate is **steady-state wall-clock latency and throughput at
fixed quality/dependency constraints** — not NFE-normalized throughput.
Per-forward efficiency is a different quantity and can move opposite to
wall-clock: a path that needs more forwards but cheaper ones can be faster in
seconds and worse per forward. **Executed NFE is reported alongside as an
explanatory variable, never as the denominator of the verdict.**

Refutable by the baseline producer: if wall-clock throughput is higher at
equal quality and equal `coupled_token_accuracy`, the claim is false. The
claim is worth stating because arXiv:2510.18480 reports cache gains shrinking
with batch size, and #165's Tier-A cells show MDLM's throughput nearly flat
from batch 1 to 32 (0.429 → 0.414 samples/s) — though on a checkpoint that
cannot run the cached path at all, which is precisely why the baseline must
be run on LLaDA/Dream.

---

## 2. Set Diffusion (arXiv:2607.01775) — BLOCKED

**Code / weights.** No official repository or checkpoint locatable at audit
time: no HF model search hit for the method, and nothing in `dev/repos/`.
The paper's claims (flexible-position/flexible-length token sets, set-causal
attention permitting a KV-cache update after **every** inference step,
arbitrary set orderings and sliding windows, better infilling than block
diffusion) are therefore **paper-only**.

**License.** Unknown — nothing to license.

**Topology.** Set-causal over flexible token sets, per the paper. This is
exactly the axis #63/#127 showed to be load-bearing: train/decode topology
consistency. A Set path could not be a sampler flag on a
block-trained backbone; it would need its own topology end to end.

**Training / conversion.** Almost certainly model-specific — set-causal
attention is a training-time property, not a decode-time option.

**Cache semantics.** Per-step cache update (the paper's central systems
claim). Unverifiable without code.

**Supported lengths / reference metrics.** Paper-only; not recorded here as
comparable numbers, because #152's rules forbid entering a paper number into
a canonical column.

**Ownership boundary.** Would be a new backbone or method pack
(`unturtle.models.backbones.*` or a `packs/` entry), never a flag on an
existing sampler.

**Falsifiable claim — deferred.** A claim can only be frozen once a
reproducible reference exists. The claim this candidate *would* test is:
*set-causal per-step cache updates beat fixed-block cache trimming on the
speed-quality frontier at matched executed NFE, and its infilling advantage
survives a dependency-sensitive task.* **Not measurable now.**

**Unblock condition.** An official code release or checkpoint with a
resolvable revision and a license permitting local use. Until then this枠
stays BLOCKED and no reduced re-implementation is attempted: a
self-written "Set-style" sampler would test our reading of a paper, not the
method.

---

## 3. Fast-dLLM v2 (arXiv:2509.26328) — REFERENCE_READY

**Code.** `NVlabs/Fast-dLLM@a9b81e4`, path `v2/`, Apache-2.0 (read from an
untracked local clone). `generation_functions.py` implements the hierarchical cache
directly: `use_block_cache`, a separate `block_past_key_values`, and a
`replace_position` argument on the forward — i.e. block-level historical
context plus a sub-block cache, as claimed.

**Weights.** `Efficient-Large-Model/Fast_dLLM_v2_7B` at
`0661abf5f9f0ee338970d091052a26c8efa51974`, license `apache-2.0`.

**License / redistribution.** Apache-2.0 on both code and weights:
local use, modification and redistribution with attribution are permitted.
No blocker.

**Topology.** Block diffusion with a **complementary attention mask** and a
token-shift mechanism that retains AR characteristics — a co-designed
training + decode topology, not a decode-time trick.

**Training / conversion required.** **Yes** — an AR→block-diffusion
fine-tune, ~1B tokens per the paper. That is the load-bearing fact for
Unturtle: adopting v2's *cache* without its *training recipe* would
reproduce the #125/#127 failure mode (topology mismatch), because the
hierarchical cache assumes the complementary-mask topology.

**Cache state, refresh, invalidation.** Two tiers, read from
`generation_functions.py` at the pinned commit:

- **historical block tier** (`past_key_values`): built once per block over
  completed context, passed with `update_past_key_values=False` so it is
  **read-only during the block**. Present in BOTH branches of the
  `use_block_cache` toggle;
- **sub-block tier** (`block_past_key_values`): created by a full
  block-width forward, then **reused for narrow forwards** over
  `[start:end]` with `replace_position=small_block_start_idx`. Refresh is
  conditional: the full-width forward reruns (and the tier is rebuilt) when
  `block_past_key_values is None` **or** the first position of the current
  sub-block is still masked; otherwise the narrow path reuses the tier and
  writes into it at `replace_position`. So invalidation is **positional
  overwrite at the sub-block offset**, not eviction, and the rebuild trigger
  is the sub-block's own leading-mask state.

**Commit / unmask policy.** Threshold parallel commit with a guaranteed
minimum: masked-position confidences below `threshold` stay masked, but the
**argmax position is force-unmasked every step**
(`unmask_idx[arange, max_prob_idx] = True`), so the loop cannot stall. Logits
are shifted by one position before use (the token-shift mechanism that
retains AR characteristics), and a row finishes when its committed set
contains `stop_token`.

**Supported context / output length.** From the pinned config:
`max_position_embeddings = 32768`, `vocab_size = 152064`, 28 layers,
hidden 3584, `model_type = Fast_dLLM_Qwen`. Block and sub-block sizes are
generation arguments, not config fields, so they are per-run settings rather
than model limits.

**Official quality / systems coordinates.** The paper reports up to ~2.5×
speedup over standard AR decoding without quality loss *in its evaluated
regime*. That regime is not #152's, so this number is **paper-only** and may
not be compared with a canonical cell.

**Ownership boundary.** A **conversion + systems path**, parallel to
`unturtle.models.conversion.a2d` — an AR→block-diffusion recipe with its own
cache, most naturally a `packs/` entry so its topology stays with its
sampler. Not a modification of core `block_decode`.

**Falsifiable claim it can test — scoped to what the toggle actually
changes.** `use_block_cache=False` does **not** disable the hierarchy: the
historical block tier (`past_key_values`) is used in both branches. What the
toggle removes is the **sub-block tier** (`block_past_key_values` +
`replace_position`) and, with it, the narrow `[start:end]` forwards — the
`False` branch always forwards the full block width.

So the claim is: *the incremental sub-block tier, with the historical block
cache present in both arms, delivers a steady-state wall-clock speedup at
matched quality — and that speedup does not shrink to nothing at batch 8 and
32.* Refutable by v2 against itself at `use_block_cache ∈ {True, False}`,
same checkpoint, same block/sub-block sizes, same threshold.

This is the sharpest available test because the ablation lives **inside one
checkpoint** and toggles **one tier**, so it is free of the cross-family
confounds catalogued in #156 Part A. What it cannot test is the hierarchy as
a whole, nor whether the speedup is attributable to the 7B scale or the AR
initialization — those need a second model and are not claimed here.

---

## 4. Training-free cache — resolves to Fast-dLLM v1 — REFERENCE_READY

The slot was written as "Fast-dLLM / dLLM-Cache / dKV-Cache style". Of the
three, **Fast-dLLM v1 is the only one that is `REFERENCE_READY`**, but for
different reasons per candidate (see the Summary):

- **Fast-dLLM v1** — Apache-2.0 code at a pinned upstream commit: ready;
- **dKV-Cache** — the artifact exists (`horseee/dKV-Cache@49a76fcc…`) but
  declares no license, so it is `BLOCKED — LICENSE UNRESOLVED`. Its unblock
  condition is a **license declaration**, not a release — unlike Set
  Diffusion, which needs an artifact to exist at all;
- **dLLM-Cache** — no locatable artifact under that name; same unblock
  condition as Set Diffusion.

**Code.** `NVlabs/Fast-dLLM@a9b81e4`, path `v1/`, Apache-2.0, with `llada/`
and `dream/` host integrations (read from an untracked local clone).

**Weights.** None of its own — it is training-free. Its host models, both
pinned: `GSAI-ML/LLaDA-8B-Instruct` @ `08b83a6feb34…` (MIT) and
`Dream-org/Dream-v0-Instruct-7B` @ `05334cb9faaf…` (Apache-2.0).

**License.** Apache-2.0 (method code) over MIT / Apache-2.0 hosts. No
blocker.

**Topology.** None of its own: it runs on an existing bidirectional /
block-decoding masked-diffusion model.

**Training / conversion required.** **No** — this is the枠's defining
property and the reason it is the cheapest candidate to evaluate.

**Cache semantics — three distinct paths, not one.** Read from
`v1/llada/generate.py` at the pinned commit. The repo's own flags expose
them independently (`--use_cache`, `--if_cache_position`, `--threshold`),
which is what makes the cache axis and the commit axis separable:

| path | what the cache holds | when it is rebuilt | trim / overwrite rule |
|---|---|---|---|
| `generate` (no cache) | nothing | n/a — a full forward every step | n/a |
| `generate_with_prefix_cache` | prefix KV for positions **before** the current block | one **full** forward per block, at the block boundary | after the block's first commit, every layer/tensor is sliced to `[:, :, :current_block_start]`; the trimmed prefix is then **held constant for the rest of the block** |
| `generate_with_dual_cache` | prefix KV **and** the current block's own KV | one full forward per block (`out_full`), then narrow forwards over `x[:, s:e]` | a boolean `replace_position` marks `[s:e]`; subsequent steps pass only the block slice with that mask, so current-block entries are **overwritten in place at the marked positions** while the prefix stays |

**The three paths are not all approximate.** `generate` recomputes a fresh
full forward at every step and reuses no stale KV, so it is the **exact
reference path** — the arm every cache claim is measured against:

| path | reuse | classification |
|---|---|---|
| `generate` | none | **exact** |
| `generate_with_prefix_cache` | prefix entries computed under an earlier masked context | **approximate reuse**, scope = positions before the current block |
| `generate_with_dual_cache` | prefix **and** current-block entries | **approximate reuse**, scope = prefix **plus** the block being actively unmasked |

Both cache paths are approximate in the same sense — an entry computed under
one masked context is reused under a later one, which is not an identity.
Dual cache has the **wider approximation scope** because its reuse extends
into the block currently being unmasked; whether that yields a *larger error*
is an empirical question the source does not answer, and this audit does not
assert it.

**Separating the cache effect from the parallel-commit effect.** The two axes
are independent in the reference and must be measured that way, so the
frozen ablation shape is a **2-D grid**, not a single "with/without cache"
comparison:

| | schedule/quota commit (`threshold=None`) | threshold parallel commit |
|---|---|---|
| no cache (exact) | baseline cell | commit effect alone |
| prefix cache | cache effect alone | both |
| dual cache | wider-scope cache effect alone | both |

The commit axis is `threshold=None` versus a fixed threshold; the cache axis
is the three paths above. Reporting only the diagonal would confound the two,
which is precisely the trap the v1 paper's own dependency-violation finding
warns about.

> ### ERRATUM — 2026-08-25: the grid above is not fully realizable
>
> The preregistered text above (kept verbatim, not rewritten) asserted that
> the cache and commit axes are independent and therefore measurable as a
> complete 2-D grid. **That is false for the current Unturtle
> implementation**, discovered when the baseline producer's wiring smoke hit
> `ValueError: parallel_decode=True does not support alg='origin'`.
>
> Cache reuse and commit policy are conceptually separable effects, but the
> current Unturtle implementation does not expose a complete Cartesian
> product. Supported arms must be represented as explicit
> `(cache_path, alg, commit_policy)` tuples. Unsupported combinations are
> typed data. In particular, the no-cache × threshold corner is unavailable
> because threshold parallel decoding requires the cached block path and a
> compatible confidence-ordering algorithm.
>
> Measured against the config validator:
>
> | cache | alg | commit | state |
> |---|---|---|---|
> | no cache | `origin` | quota | **supported** |
> | no cache | `origin` | threshold | **unsupported** — `parallel_decode` requires `use_cache=True` |
> | prefix cache | `origin` | quota | **supported** |
> | prefix cache | `maskgit_plus` | quota / top-k | **supported** |
> | prefix cache | `maskgit_plus` | threshold 0.9 | **supported** |
>
> Consequence for the ablation: the commit axis cannot be varied while
> holding `alg` fixed, because quota commit exists only on `origin` and
> threshold commit only on a confidence-ordered alg. The baseline therefore
> runs **four explicit tuples** rather than a product, so that the `alg`
> change and the commit change stay separable:
>
> 1. `(no_cache, origin, quota)` — the exact reference arm;
> 2. `(prefix_cache, origin, quota)` — 1→2 isolates the **cache** effect;
> 3. `(prefix_cache, maskgit_plus, quota)` — 2→3 isolates the **alg** effect;
> 4. `(prefix_cache, maskgit_plus, threshold 0.9)` — 3→4 isolates the
>    **commit-policy** effect.
>
> Without arm 3 a 1→4 comparison would compound two changes. The
> `(no_cache, threshold)` corner is emitted as a typed `unsupported` cell.

**`threshold=None` is a quota policy, not one token per step.**
`get_num_transfer_tokens` allocates `floor(masked_in_block / steps)` per step
and distributes the remainder as `+1` over the first `masked mod steps`
steps, so a step commits **several** tokens whenever `masked_in_block >
steps`. Every quota equals 1 only in the boundary case
`steps == masked_in_block` (and quotas of 0 appear when `steps >
masked_in_block`).

A genuine **one-token-per-step control** is therefore a separate thing, and
if a later ablation needs it, it must be frozen as one of:

- a `(steps, block_length)` setting where `steps == masked_in_block` for
  every block — exact but only reachable at specific lengths; or
- an explicit commit policy that transfers the single highest-confidence
  masked position per step, declared as its own arm rather than obtained by
  tuning `threshold`.

Neither is claimed here, and no cell in the grid above is labelled
"one token per step".

**Commit / unmask policy.** Confidence-threshold parallel commit — **already
present in Unturtle** as `parallel_decode` + `confidence_threshold`, with the
`alg='entropy'` degeneration already guarded. So for this枠 Unturtle's
baseline is not a strawman: part of v1's mechanism already exists in
Unturtle core.

**Supported lengths.** Host-model governed, resolved from the pinned
configs: LLaDA-8B-Instruct `max_position_embeddings = 4096` (vocab 126464);
Dream-v0-Instruct-7B `131072` (vocab 152064). The v1 method adds no length
limit of its own.

**Official quality / systems coordinates.** The paper identifies **dependency
violation from the conditional-independence assumption** as the major quality
failure mode of parallel decoding. That is a *mechanism* claim this audit
carries forward, and it is why the baseline protocol below requires
dependency-correctness cells rather than PPL alone (also ParallelBench,
arXiv:2510.04767).

**Ownership boundary.** Cache/commit policy inside the existing
`block_decode` path — the only枠 that would touch core. Its confidence-aware
half already does.

**Falsifiable claim it can test.** *Training-free approximate KV reuse
preserves dependency correctness at the thresholds where it is fast — and the
speedup survives once the parallel-commit effect is held fixed.* The second
clause is what the 2-D grid above tests: a gain that appears only on the
diagonal is a commit gain wearing a cache label.
Refutable by a dependency-sensitive cell (`unturtle/eval/dependency_slice.py`
already provides copy / reverse / kv_recall with `exact_match` and
`coupled_token_accuracy`): if `coupled_token_accuracy` drops as the threshold
loosens while throughput rises, the claim is false and the speedup is being
paid for in dependency violations — exactly the failure the v1 paper names.

---

## 5. Commitment order as an independent axis

`Answer First, Reason Later` is a **failure mode of commit order**, not of
throughput, so it needs its own coordinates. Any candidate sampler whose
per-position commit step is observable contributes these. They are frozen
here as baseline-protocol candidates, **before** any measurement:

| metric | definition | unit |
|---|---|---|
| `normalized_commit_step` | for each output position, the step at which it was first committed, divided by the executed step count | fraction in [0,1] |
| `answer_before_reasoning_rate` | fraction of samples where the mean `normalized_commit_step` of the answer/suffix span is **less** than that of the reasoning/prefix span | fraction |
| `tokens_committed_per_step` | count committed at each step, with its position distribution (mean and standard deviation of committed positions per step) | count + position stats |
| `dependency_correctness_under_commit_constraint` | `coupled_token_accuracy` and `exact_match` measured at ≥2 commit-constraint settings (e.g. one-token-per-step vs threshold parallel commit) | fraction |

**Which metrics are defined on which fixture.** `answer_before_reasoning_rate`
needs two spans *inside the output*, and `dependency_slice`'s `DependencyTask`
carries only `prompt`, `source` and `target` — its boundary is between the
prompt and the output, not within the output. So the metric is **not
computable there**, and an empty-span exclusion rule would not fix that: it
would mark every fixture unsupported.

| metric | `dependency_slice` fixtures | a reasoning fixture with declared output spans |
|---|---|---|
| `normalized_commit_step` | **supported** | supported |
| `tokens_committed_per_step` (+ position distribution) | **supported** | supported |
| `dependency_correctness_under_commit_constraint` | **supported** | n/a |
| `answer_before_reasoning_rate` | **UNSUPPORTED — reported as such, never as 0** | supported |

`answer_before_reasoning_rate` is therefore scoped to a fixture that
**declares its own reasoning and answer spans as part of the task
definition** — the span boundary is task-provided data, not something the
producer infers by inspecting generated text. Defining that fixture is part
of the baseline work; until it exists the metric has no cells, and a missing
cell is recorded as `unsupported`, which #152 already treats as data rather
than an omission.

Frozen conventions so these cannot be reshaped after seeing results:

- spans, where they exist, come from the **task definition**, never from
  inspecting the output;
- a position committed once and later revised counts at its **first** commit
  for `normalized_commit_step`, with revisions reported separately via the
  `revision_events` counter that #165 added — the two answer different
  questions and must not be merged;
- on a fixture that does declare spans, `answer_before_reasoning_rate`
  requires both to be non-empty; a sample where either is empty is reported
  as excluded with a reason, never dropped silently.

**This is not authorization for a general trace API.** The first
implementation is experiment-local: whatever the baseline producer needs to
emit these columns for the samplers it runs. A shared abstraction is
justified only if a second consumer needs the same reduction — the same rule
#156 Part B fixed for diagnostics.

---

## 6. What this audit does NOT do

- no candidate implementation or porting;
- no cache-threshold tuning;
- no performance measurement;
- no winner selection — three `REFERENCE_READY` slots (Unturtle's own
  baseline plus **two external implementation candidates**, Fast-dLLM v1 and
  v2) is not a ranking;
- no universal cache or trace abstraction;
- no relaxation of the PreDiff hybrid's capability guard. #125/#127 rejected
  `block_decode` for hybrid PreDiff because train/decode topology mismatch
  was catastrophic, and the capability probes encode that. Nothing here
  changes it.

## 7. Next step, and what it must produce

After this audit is reviewed: a **baseline producer** over the existing
AR / MDLM / `block_decode` paths — no candidate code. Its required cells,
so the requirement is fixed before the numbers exist:

- batch **1 / 8 / 32**, output length **1024** plus a long-output cell;
- warmup outside every timed region (#165's producers pay ~160 s of
  first-call compile, which would otherwise be billed to generation);
- **executed** NFE, not requested — with the per-family accounting #165
  established (AR bills token-forwards; MDLM 128 steps **+1** for upstream
  `noise_removal`);
- throughput and peak memory per cell, with the per-cell executed work #165
  added so batch scaling and generation-length effects stay separable;
- **dependency correctness** from `dependency_slice`, not PPL alone;
- **commitment order** per §5;
- typed OOM / unsupported cells as data (#152), never omissions.

The frozen baseline is what every candidate claim in §1–§4 is refuted or
confirmed against. Until it exists, the claims stay claims.
