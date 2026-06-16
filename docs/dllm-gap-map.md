# dLLM Gap-Map and Roadmap

> **Living document.** Update the status and priority columns as Unturtle changes and as
> the diffusion-LM field moves. The axes below are validated against the dLLM paper
> (`dev/papers/dllm.md`) and the community surveys
> [Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs) and
> [Awesome-Diffusion-LLM](https://github.com/AIDASLab/Awesome-Diffusion-LLM).

## North star

Unturtle is **the unsloth-accelerated dLLM layer**. It adopts the field-standard
decomposition (backbone architecture × conversion method × training objective) instead of
inventing its own, leans on upstream for evaluation reproducibility
(`lm-evaluation-harness`), and differentiates on Triton kernels, fast LoRA,
bidirectional / packed-varlen fast paths, and inference acceleration.

**Priority is driven by fit with that north star.** Inference acceleration ranks highest
because it is Unturtle's reason to exist; multimodal/VLA/agentic are deferred — not
unimportant, but not this library's job.

## Status legend

- ✅ implemented
- 🟡 partial
- ❌ missing
- ⛔ out of scope (deliberately not Unturtle's lane)

## Gap-map

> Sharpened 2026-05-31 from the literature survey (`.references/survey-matrix.md` +
> `.references/adoption-queue.md`).

| Category | Representative methods | Unturtle status | Where | Fit | Priority |
|---|---|---|---|---|---|
| Training objectives | MDLM, BD3LM | ✅ | `unturtle/diffusion/trainer.py`, `block_diffusion_trainer.py` | core | maintain |
| AR→Diffusion conversion | A2D family; Tiny-A2D recipe (DiffuLLaMA / TESS-2 / SDAR family) | ✅ | `unturtle/models/conversion/a2d/tiny_a2d/` | core | maintain |
| Encoder backbones | ModernBERT-chat | ✅ | `unturtle/models/backbones/modernbert/` | medium | maintain |
| Block-AR diffusion backbone | **DiffusionGemma** (Google, 26B-A4B MoE, block-AR diffusion; `transformers` `models/diffusion_gemma` wrapper) | ✅ (#25) | `unturtle/models/backbones/diffusion_gemma/` | high | maintain |
| MDLM DiT backbone | **MDLM-DiT** (kuleshov-group/mdlm DiT; adaLN-Zero, time-agnostic native baseline) | ✅ (#31) | `unturtle/models/backbones/mdlm_dit/` | medium | maintain |
| Tri-mode AR+diffusion backbone | **Nemotron-Labs-Diffusion** (NVIDIA 3B/8B/14B, AR + diffusion + self-spec; HF weights public) | ❌ (candidate) | — | medium | evaluate |
| Evaluation discipline | lm-eval-harness, hyperparam sensitivity | ✅ | `unturtle/eval/harness/` | high | maintain |
| High-level generation entry | `FastDiffusionModel.generate(algorithm=…)` | ✅ | `unturtle/models/generation/sampler.py` + `fast_diffusion_model.py` | high | maintain |
| Inference: block decode / KV cache | Fast-dLLM (have); **dLLM-Cache** (training-free adaptive, 9×), **dKV-Cache** (delayed KV) | 🟡 | `unturtle/models/generation/{block_decode_mixin,cache,cache_utils}.py` | high | **P1** |
| Inference: parallel / adaptive decode | Fast-dLLM parallel (partial); **APD** (adaptive parallel) | 🟡 | `unturtle/models/generation/diffusion_generation_utils.py` | high | **P1→P2** |
| Hybrid faster-than-AR | **D2F** (discrete diffusion forcing), E2D2, Fast-dLLM v2 | ❌ | — | high | P2 |
| Distillation / few-step | **SDTT** (self-distill, 32–64× steps), CDLM (consistency), FS-DFM | ❌ | — | high (speed) | P2 |
| Quantization / sparsity | DLLMQuant (PTQ), SparseD / Sparse-dLLM | ❌ | — | medium | P2 |
| Post-training / RL | d1 (GRPO) + wd1 (have); **VRPO** (LLaDA-1.5), MDPO, DiFFPO, DARE | 🟡 GRPO+wd1 | `unturtle/diffusion/grpo_trainer.py` | medium | P2 |
| Inference frameworks | dInfer (modular, >1100 TPS), dlmserve | ❌ | — | medium | defer (reference) |
| Continuous / latent diffusion | Diffusion-LM, TESS, CCDD | ⛔ | — | future extension point | defer |
| Multimodal / VLA / driving / agentic | LaViDa, MMaDA, LLaDA-VLA, Dream-VLA | ⛔ | — | out of scope | defer |

## Roadmap

> Detailed per-technique rationale: `.references/adoption-queue.md` (deep-read evaluations).

### Done

- Taxonomy axis-split (PR #292), eval canonicalization (PR #294), gap-map (PR #296),
  physical taxonomy migration (PR #298), high-level `FastDiffusionModel.generate` +
  algorithm registry (PR #300).
- DiffusionGemma backbone wrapper + `block_ar` algorithm (self-conditioned canvas block
  diffusion; no mask token) + `("diffusion_gemma","gsm8k")` DecodingConfig entry (#25).
- MDLM-DiT native diffusion backbone — time-agnostic adaLN-Zero Diffusion
  Transformer (kuleshov-group/mdlm DiT, `time_conditioning=False` equivalent),
  rides the existing `mdlm` algorithm; native re-implementation baseline, not
  weight-compatible with published checkpoints (#31).

### First roadmap after the clean migration

Now that the clean port (CLI / benchmarks / examples / docs) is complete, the immediate
sequence is:

1. **DiffusionGemma backbone** — wrap the upstream `transformers` `models/diffusion_gemma`
   implementation as a native block-AR diffusion backbone. Pairs with the `FastModel`
   delegation work (#15): models that carry their own loss/generation route through the
   standard path, so this lands as a backbone wrapper, not a new objective. — done (#25)
2. **Nemotron evaluation** — evaluate Nemotron-Labs-Diffusion (tri-mode AR+diffusion+self-spec)
   on the canonical harness to decide whether to add it as a backbone.
3. **unsloth CLI plugin mechanism** — propose the `unturtle` CLI integration as a plugin
   mechanism upstream to unsloth.
4. **dLLM-Cache (P1)** — the highest-value training-free inference speedup (see below).

### P1 — now / immediate next

- **Inference caching (training-free):** adopt **dLLM-Cache** (training-free adaptive
  caching, ~9× on LLaDA/Dream, no quality loss) as a cache strategy in
  `unturtle/models/generation/`, composing with the existing Fast-dLLM block decode. This is
  the single highest-value training-free speedup. Evaluate **dKV-Cache** alongside and ship
  whichever benchmarks best as default.

### P2 — after P1

- **Adaptive parallel decode:** APD (auxiliary-AR-guided adaptive parallel width).
- **Hybrid faster-than-AR:** D2F (discrete diffusion forcing; distillation + pipelined decode).
- **Few-step distillation:** SDTT / CDLM / FS-DFM (training-based; cluster as one workstream).
- **RL beyond GRPO:** VRPO (LLaDA-1.5; variance-reduced DPO) as the lead, then MDPO / DiFFPO.
- **Quantization / sparsity:** DLLMQuant, SparseD (lower priority).

### Deferred (reference / future)

- **Inference-framework architecture:** mine dInfer's 4-component decomposition (model /
  iteration manager / decoding strategy / KV-cache manager) when consolidating the
  generation/sampler layer — it validates the `FastDiffusionModel.generate` direction.
- **Continuous / latent-space diffusion:** future extension point (the `generate` algorithm
  registry is already open for it; current loops are discrete-masked-only).

### Out of scope (YAGNI)

- Multimodal / VLA / agentic dLLMs.
- Decoupling from `unsloth`.
- Rewriting the MDLM / BD3LM objectives.

## How to update this doc

- When a category's status changes (e.g. a P2 item lands), update its **status**, **where**,
  and **priority** cells in the same PR.
- When the field adds a category the surveys track and it fits the north star, add a row.
- Keep the priority logic tied to fit with the unsloth-acceleration north star, not to
  novelty.
