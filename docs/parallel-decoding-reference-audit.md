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

| artifact | revision |
|---|---|
| local `dev/repos/fast-dllm` clone | `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` |
| `Efficient-Large-Model/Fast_dLLM_v2_7B` | `0661abf5f9f0ee338970d091052a26c8efa51974` |
| `GSAI-ML/LLaDA-8B-Instruct` | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| `Dream-org/Dream-v0-Instruct-7B` | `05334cb9faaf763692dcf9d8737c642be2b2a6ae` |

---

## Summary

| candidate | state | why |
|---|---|---|
| 1. Unturtle `block_decode` | **REFERENCE_READY** | in-tree, already exercises block cache + confidence-aware parallel decode; the natural frozen baseline |
| 2. Set Diffusion (arXiv:2607.01775) | **BLOCKED** | no official code or weights locatable at audit time; nothing to reproduce |
| 3. Fast-dLLM v2 (arXiv:2509.26328) | **REFERENCE_READY** | Apache-2.0 code in-tree at a pinned commit + official 7B checkpoint at a pinned SHA; hierarchical cache is implemented, not just described |
| 4. Fast-dLLM v1 training-free cache (arXiv:2505.22618) | **REFERENCE_READY** | Apache-2.0 code in-tree; training-free, so its "artifact" is code + a host model (LLaDA / Dream, both pinned) |
| dLLM-Cache / dKV-Cache | **BLOCKED** | no locatable official artifact under those names at audit time |

Two of the four枠 are ready, one枠 is blocked on artifact availability, and
the training-free枠 resolves to Fast-dLLM v1 rather than the other named
caches.

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

**Falsifiable claim it can test.** *At batch 8 and 32 with 1024-token output,
`block_decode` with `parallel_decode` at a fixed confidence threshold gives
no throughput advantage over plain `mdlm` once NFE is counted as executed
denoiser calls rather than loop steps.* Refutable by the baseline producer:
if executed-NFE-normalized throughput is higher, the claim is false. This
matters because arXiv:2510.18480 reports cache gains shrinking with batch
size, and #165's Tier-A cells already show MDLM's throughput being nearly
flat from batch 1 to 32 (0.429 → 0.414 samples/s).

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

**Code.** In-tree at `dev/repos/fast-dllm/v2/` (clone commit `a9b81e4`),
Apache-2.0. `generation_functions.py` implements the hierarchical cache
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

**Cache state, refresh, invalidation.** Two levels. The block level holds
completed-block context; the sub-block level (`block_past_key_values` with
`replace_position`) is refreshed within a block. Whether sub-block entries
are invalidated or overwritten in place is a code-reading question this
audit deliberately leaves open — answering it requires running the code,
which is out of scope for Part 0.

**Commit / unmask policy.** Block-diffusion parallel commit within a block;
exact policy to be read from `generation_functions.py` when the baseline
work starts.

**Supported context / output length.** To be recorded from the model config
at baseline time; the 7B model's own context limit governs.

**Official quality / systems coordinates.** The paper reports up to ~2.5×
speedup over standard AR decoding without quality loss *in its evaluated
regime*. That regime is not #152's, so this number is **paper-only** and may
not be compared with a canonical cell.

**Ownership boundary.** A **conversion + systems path**, parallel to
`unturtle.models.conversion.a2d` — an AR→block-diffusion recipe with its own
cache, most naturally a `packs/` entry so its topology stays with its
sampler. Not a modification of core `block_decode`.

**Falsifiable claim it can test.** *Fast-dLLM v2's hierarchical cache
delivers its speedup because of the cache hierarchy, not because of the
7B-vs-169M scale and the AR-derived initialization.* Refutable by comparing
v2 against its own no-block-cache setting (`use_block_cache=False`, which
the code exposes) at matched batch and executed NFE: if the intra-model
delta is small while the delta against Unturtle's `block_decode` is large,
the speedup is attributable to the model, not the hierarchy.

This is the sharpest available test because the ablation lives **inside one
checkpoint**, so it is free of the cross-family confounds catalogued in
#156 Part A.

---

## 4. Training-free cache — resolves to Fast-dLLM v1 — REFERENCE_READY

The枠 was written as "Fast-dLLM / dLLM-Cache / dKV-Cache style". At audit
time **only Fast-dLLM v1 has a locatable artifact**; HF searches for
`dLLM-Cache` and `dKV-Cache` return nothing under those names. Those two are
recorded as **BLOCKED** with the same unblock condition as Set Diffusion.

**Code.** `dev/repos/fast-dllm/v1/` (clone commit `a9b81e4`), Apache-2.0,
with `llada/` and `dream/` host integrations.

**Weights.** None of its own — it is training-free. Its host models, both
pinned: `GSAI-ML/LLaDA-8B-Instruct` @ `08b83a6feb34…` (MIT) and
`Dream-org/Dream-v0-Instruct-7B` @ `05334cb9faaf…` (Apache-2.0).

**License.** Apache-2.0 (method code) over MIT / Apache-2.0 hosts. No
blocker.

**Topology.** None of its own: it runs on an existing bidirectional /
block-decoding masked-diffusion model.

**Training / conversion required.** **No** — this is the枠's defining
property and the reason it is the cheapest candidate to evaluate.

**Cache semantics.** Block-wise **approximate** KV reuse: entries computed
under one masked context are reused under a later one, which is an
approximation, not an identity. Combined with confidence-aware parallel
decoding.

**Commit / unmask policy.** Confidence-threshold parallel commit — **already
present in Unturtle** as `parallel_decode` + `confidence_threshold`, with the
`alg='entropy'` degeneration already guarded. So for this枠 Unturtle's
baseline is not a strawman: part of v1 is in-tree.

**Supported lengths.** Host-model governed (LLaDA 8B / Dream 7B).

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
preserves dependency correctness at the thresholds where it is fast.*
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

Frozen conventions so these cannot be reshaped after seeing results:

- spans are defined by the **task**, not by inspection of the output: for
  `dependency_slice` tasks the prefix/suffix boundary is the task's own input
  boundary; for a reasoning task it is the marker the task specifies;
- a position committed once and later revised counts at its **first** commit
  for `normalized_commit_step`, with revisions reported separately via the
  `revision_events` counter that #165 added — the two answer different
  questions and must not be merged;
- `answer_before_reasoning_rate` requires both spans non-empty; a sample
  where either is empty is reported as excluded with a reason, never dropped
  silently.

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
- no winner selection — two `REFERENCE_READY` candidates is not a ranking;
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
