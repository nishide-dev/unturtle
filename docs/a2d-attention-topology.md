# AR→dLLM conversion: choosing the attention topology (#63)

Unturtle's Tiny-A2D recipe converts a pretrained causal LM (Llama / Qwen2 /
Qwen3) into a masked-diffusion model. The conversion preserves every
checkpoint tensor bit-for-bit; the *choice* is what happens to attention:

- **Uniform bidirectional** (`hybrid_attention=False`, the default): every
  position attends everywhere. The classic A2D adaptation.
- **Hybrid** (`hybrid_attention=True`): PreDiff-LM eq. (3) — prompt tokens
  keep the causal pattern they were pretrained with, target tokens denoise
  bidirectionally, and the corrupted target cannot write into prompt
  representations.

## When to choose hybrid

**Default to hybrid for prompt/response (SFT-style) adaptation from a
pretrained AR checkpoint** — on the evidence measured so far: one backbone
(Qwen3-0.6B), one dataset (GSM8K), and a masked-CE NLL proxy rather than
downstream task accuracy. Within that scope the matched benchmark
(`benchmarks/a2d/hybrid_vs_bidirectional.py` — frozen reference run in its
docstring; same init / data / noise / compute) is unambiguous: hybrid ahead
at every checkpoint, every masking rate, on both seeds — final NLL 2.11 vs
2.53 — at identical training cost (~same steps/s, same peak memory). The
step-0 gap (7.8 vs 14.8 before any gradient step) measures *topology fit*,
not learned quality: the causal prompt keeps the AR pretraining usable
instead of asking training to relearn it under a topology the weights never
saw. Note the two arms score under their own inference topologies (the
models a user would actually run), so the hybrid arm conditions on strictly
more structure by design.

Choose **uniform bidirectional** when:

- there is no prompt/target split — unconditional generation, or corruption
  over the whole sequence (`completion_only=False`), where eq. (3) has no
  observed region to preserve;
- batches are **packed**: one boundary per row cannot express per-sample
  splits, so `HybridPromptCollator` rejects packed collators/batches loudly.
  Packed hybrid training needs a segment-aware mask that does not exist yet;
- you cannot supply prompt boundaries at inference time — a hybrid-trained
  model is meant to be *run* hybrid (`prompt_lengths` at generation), and the
  train/inference topology should match.

## How to train hybrid

```python
from unturtle.diffusion import DiffusionTrainer, MaskedDiffusionDataCollator
from unturtle.models.conversion.a2d.tiny_a2d import (
    HybridPromptCollator,
    load_tiny_a2d_from_ar,
)

model = load_tiny_a2d_from_ar(
    "Qwen/Qwen3-0.6B", hybrid_attention=True, torch_dtype=torch.bfloat16
)
collator = HybridPromptCollator(
    MaskedDiffusionDataCollator(
        tokenizer=tokenizer,
        mask_token_id=model.config.mask_token_id,
        completion_only=True,
    )
)
# DiffusionTrainer(..., data_collator=collator) — the trainer ships
# `prompt_lengths` to the model via model(**inputs); nothing else changes.
```

The boundary is derived from the SFT labels convention (first `labels != -100`
position per row), so any prompt-masked dataset works unchanged. LoRA via
`FastDiffusionModel.get_peft_model` composes with both topologies without
restoring causal-only attention (regression-tested).

## Operational notes

- The hybrid flag is **inert without `prompt_lengths`**: a converted model
  that never receives the boundary behaves exactly like the bidirectional
  one. This is fail-safe (missing plumbing degrades to a *different valid
  topology*, not a crash), which is why the e2e tests pin the threading
  explicitly.
- Hybrid attention rejects KV caches during training (`use_cache=False`);
  eq. (3) is defined over one square sequence.
- Below `hybrid_fast_min_seq_len` (default 2048) the dense eq.-(3) mask is
  used; at L ≥ 2048 the mask-free split gives 1.3–1.9× attention speedups
  (#101). The crossover is hardware-dependent — it is a config field, not a
  constant.
