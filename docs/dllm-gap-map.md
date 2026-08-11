# Diffusion/Flow LM Gap-Map and Roadmap

> **Living document.** Update the status and priority columns as Unturtle changes and as
> the diffusion-LM field moves. The axes below are validated against the dLLM paper
> (`dev/papers/dllm.md`) and the community surveys
> [Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) and
> [Awesome-Diffusion-LLM](https://github.com/AIDASLab/Awesome-Diffusion-LLM).

## North star

Unturtle is **the systems/method layer for rapidly researching diffusion/flow language
models — training, post-training, and generation — on top of the existing
Transformers / TRL / Unsloth ecosystem** (#67). It adopts the field-standard
decomposition (forward process × backbone architecture × conversion method × training
objective × generation algorithm) instead of inventing its own, and leans on upstream
for evaluation reproducibility (`lm-evaluation-harness`) plus its own canonical
free-generation surface (MAUVE + collapse guards).

Inference acceleration (Triton kernels, fast LoRA, bidirectional/packed-varlen fast
paths, KV-cache block decode) is an important axis of that layer — **not the sole
reason the project exists**. Priority is driven by fit with the research north star.

**Evidence hygiene.** This document distinguishes *implementation* from *capability
promotion*: a method being mechanically composable does not make it a supported
capability. Negative and undecidable gate outcomes are first-class evidence and are
recorded as such, frozen on their issues.

## Status legend

- ✅ implemented & validated (regression-covered, part of the supported surface)
- 🧪 research-only / explicit opt-in (implemented end-to-end; **not** a promoted capability)
- 🟡 partial
- ❌ missing
- 🔭 experimental extension track (in scope; maturity varies by row)

## Gap-map

| Category | Representative methods | Unturtle status | Where | Evidence |
|---|---|---|---|---|
| Forward processes | masked (MDLM-style), discrete-flow, continuous-flow | ✅ | `unturtle/processes/` | process boundary (#62); device-side noising by trainers |
| Training objectives | MDLM, BD3LM | ✅ | `unturtle/diffusion/` | core; loss normalization matches MDLM/d1 semantics |
| AR→Diffusion conversion | Tiny-A2D recipe (DiffuLLaMA / TESS-2 / SDAR family) | ✅ | `unturtle/models/conversion/a2d/tiny_a2d/` | core |
| Hybrid-attention conversion | **PreDiff-style eq.(3)** (prompt-causal + target-bidirectional) | ✅ | `_hybrid.py` + generation threading | **Positive** (#63→#114→#125→#127): beats uniform bidirectional on masked NLL (2.11 vs 2.53) AND free-generation MAUVE (4.5–7×) when decoded topology-matched; #125's reversed sign was the measured cost of a train/decode topology mismatch (49–84×), kept as historical evidence. See `docs/a2d-attention-topology.md` |
| Post-training: on-policy distillation | OPD (rollouts, teacher divergence) | ✅ | `unturtle/post_training/` | #64; e2e-tested incl. Qwen3 student |
| Post-training: RL | d1-style GRPO (+wd1) | 🟡 | `unturtle/diffusion/grpo_trainer.py` | TRL-style reward-model scoring landed; VRPO/MDPO not adopted |
| Encoder backbones | ModernBERT-chat | ✅ | `unturtle/models/backbones/modernbert/` | maintain |
| Block-AR diffusion backbone | DiffusionGemma (`block_ar` canvas algorithm) | ✅ | `unturtle/models/backbones/diffusion_gemma/` | #25 |
| MDLM DiT backbone | kuleshov-group/mdlm DiT (time-agnostic adaLN-Zero) **+ published-checkpoint conversion** | ✅ | `unturtle/models/backbones/mdlm_dit/` (+ `convert_mdlm_owt.py`) | #31; #130-PR0 proved fp32-exact parity + seeded-sampling identity against mdlm-owt |
| Generation registry | explicit runners/capabilities per algorithm | ✅ | `unturtle/models/generation/sampler.py` | families: masked_discrete (mdlm/block_decode/bd3lm), canvas (block_ar), continuous_flow (flowlm), latent_guided (ladiff), discrete_flow (dfm) |
| Inference: block decode / KV cache | Fast-dLLM (have); dLLM-Cache, dKV-Cache (candidates) | 🟡 | `unturtle/models/generation/` | still a priority axis; see P-queue |
| Discrete flow matching | **DFM / FS-DFM** (kinetic-optimal, few-step) | 🧪 **implemented, deliberately unpromoted** | `dfm_mixin`, `dfm_solver`, `fs_dfm`, `processes/discrete_flow` | #65/#120–#122: solver + shortcut distillation run end-to-end; the real-backbone eligibility gate was undecidable at prototype budget and the canonical-scale gate FAILED its frozen thresholds — `supports_dfm_generation` stays an explicit research opt-in. A retry is a NEW experiment in a different regime, not a reopened #65 |
| Continuous / latent diffusion | FlowLM; **LaDiff / DiLaDiff-style codec+prior**; MeanFlow distillation | 🔭 prototypes 🧪 real-text gated | `unturtle/models/latent/` | #66 (FlowLM), #117/#118 (trainable codec → latent prior → latent-guided decode), #119 (MeanFlow/JVP; honest-negative at prototype scale). **#130 real-text LaDiff existence gate: Gate A PASS** (true latents help, wrong latents hurt, benefit grows with mask ratio — mechanism works on the real pretrained backbone) **but Gate B FAIL at the tested budget (~1–2% of paper scale)**: prior-sampled latents (MAUVE 0.37–0.58) lost to every unconditional mode (0.71–0.94) on both seeds with no collapse — failure localized to the learned prior, NOT a claim that LaDiff never works. No real-text DiLaDiff follow-up per the frozen stop/go |
| Canonical generation evaluation | MAUVE + diversity/collapse guard trio + record schema | ✅ | `unturtle/eval/generation_metrics.py` | #123/#124; decision rules stay experiment-local by design; two proven consumers (dfm canonical gate, hybrid readouts) |
| Evaluation discipline | lm-eval-harness, pinned DecodingConfig | ✅ | `unturtle/eval/harness/` | authoritative path |
| Data tooling | mdlm-convention OWT packing (streaming, atomic, audited sidecars) | ✅ | `unturtle/utils/packed_text.py` | #132 |
| Tri-mode AR+diffusion backbone | Nemotron-Labs-Diffusion | ❌ (candidate) | — | evaluate |
| Hybrid faster-than-AR | D2F, E2D2, Fast-dLLM v2 | ❌ | — | P2 |
| Few-step distillation (masked) | SDTT, CDLM | ❌ | — | P2 (FS-DFM's flow-side sibling is 🧪 above) |
| Quantization / sparsity | DLLMQuant, SparseD | ❌ | — | P2 |
| Inference frameworks | dInfer, dlmserve | ❌ | — | defer (reference) |
| Multimodal / VLA / agentic | LaViDa, MMaDA, LLaDA-VLA | ⛔ out of scope | — | defer |

## Roadmap

### Landed (evidence-complete arcs)

- **#63→#127 hybrid conversion**: PreDiff eq.(3) hybrid attention, training + generation
  threading, topology-matched readout. Positive on both metrics; the topology-mismatch
  magnitude (49–84× on MAUVE) is itself a frozen, quantified result.
- **#64 OPD**: on-policy distillation post-training path.
- **#65 DFM/FS-DFM**: full implementation + two pre-registered gates ending in a
  decision-grade negative; capability deliberately unpromoted.
- **#66/#117–#119 continuous & latent prototypes**: FlowLM, DiLaDiff-style codec/prior
  slices, MeanFlow (honest-negative at prototype scale).
- **#123/#124 canonical eval surface** with measurement/verdict separation.
- **#130 real-text LaDiff existence gate**: checkpoint parity (PR0), OWT data layer,
  real-backbone codec (Gate A PASS), latent prior (Gate B FAIL, prior-localized).
  Frozen verdict; retry only as a new experiment in a different regime (candidates:
  larger prior budget/capacity, regularization re-calibration, latent geometry).
- Infrastructure arcs: taxonomy axis-split, process boundary (#62), integration
  registry, generation registry, `FastDiffusionModel.generate` facade, DiffusionGemma
  (#25), MDLM-DiT (#31) + published-checkpoint conversion (#130-PR0).

### P-queue (inference-acceleration axis, unchanged in substance)

1. **dLLM-Cache / dKV-Cache** — training-free caching composing with Fast-dLLM block
   decode; benchmark and ship the winner as default.
2. **Adaptive parallel decode** (APD), **D2F-style hybrid faster-than-AR**.
3. **Masked few-step distillation** (SDTT / CDLM), **RL beyond GRPO** (VRPO lead),
   **quantization/sparsity**.

### Deferred / out of scope

- Inference-framework architecture (mine dInfer's decomposition when consolidating the
  sampler layer).
- Multimodal / VLA / agentic; decoupling from `unsloth`; rewriting the MDLM/BD3LM
  objectives (YAGNI).

## How to update this doc

- When a category's status changes, update its **status**, **where**, and **evidence**
  cells in the same PR.
- Record gate outcomes (positive, negative, undecidable) with a pointer to the frozen
  issue record; never rewrite a negative as missing or a mechanism as a capability.
- Keep the priority logic tied to the research north star, not to novelty.
