# RFC: continuous / latent diffusion and flow-matching backend

Status: **draft, no implementation**
Issue: [#66](https://github.com/nishide-dev/unturtle/issues/66)
Date: 2026-08-08

## What this answers

The smallest boundary Unturtle needs so that continuous and latent methods can
be added without turning the masked-discrete code into a mode-flag switchboard.

It is grounded in the five papers #66 cites, all read rather than summarized:

| paper | codec | diffusion target | notable |
|---|---|---|---|
| [LDLM](https://arxiv.org/abs/2605.07933) | trained **jointly** with the diffusion model | `x_0` | needs MSE decoder loss, diffusion-to-encoder warmup, adaptive timestep sampling, decoder-input noise |
| [TextLDM](https://arxiv.org/abs/2605.07748) | **separate** Transformer VAE + REPA alignment to a frozen LM | flow matching | reconstruction fidelity alone is *insufficient*; alignment is the critical piece |
| [AURORA-LM](https://arxiv.org/abs/2608.02602) | prefix-aligned, **deliberately uncompressed** | ablates `x_0` vs velocity | block-causal DiT over full-width latents |
| [DiLaDiff](https://arxiv.org/abs/2605.23605) | auto-encoder **fine-tuned from an existing masked dLLM** | latent prior + consistency distillation | latent *guides* discrete decoding rather than replacing it |
| [FlowLM](https://arxiv.org/abs/2605.20199) | none — operates on an existing diffusion LM | **`x_0`** | straightens trajectories of a pretrained model; validates clean-data prediction over velocity |

## The constraint that shapes everything

These five disagree about the codec in a way that is not a detail:

- LDLM's codec is a **training participant** with its own losses and a warmup
  schedule relative to the diffusion model.
- TextLDM's is **separately pretrained** but carries an auxiliary alignment
  loss against a frozen LM.
- DiLaDiff's is **derived from an existing masked dLLM** — the asset Unturtle
  already has.
- FlowLM has **no codec at all**.

So a `LatentCodec` that is only `encode()`/`decode()` is wrong. It would force
LDLM's recipe and TextLDM's REPA term into the trainer as special cases —
precisely the flag explosion #66 exists to prevent. The protocol has to admit
codec-owned losses and a trainability signal:

```python
class LatentCodec(Protocol):
    def encode(self, input_ids, attention_mask=None, **kwargs) -> LatentBatch: ...
    def decode(self, latents, attention_mask=None, **kwargs) -> Any: ...

    # Non-negotiable additions, from LDLM / TextLDM:
    @property
    def trainable(self) -> bool: ...
    def auxiliary_losses(self, batch, latents) -> dict[str, Tensor]: ...
```

`auxiliary_losses` returns a **dict**, not a scalar, so the trainer can log and
weight terms it does not need to understand — TextLDM's REPA and LDLM's MSE
decoder loss are different things that must not be summed behind the trainer's
back. `trainable` is what lets one trainer serve a frozen VAE and a jointly
optimized encoder without branching on model type.

Deliberately **not** in the base protocol: any VAE or KL assumption. Only
TextLDM is a VAE; FlowLM has no codec, and AURORA-LM's central claim is that
the decoder-facing latent should *not* be compressed.

## Target parameterization is an axis, not a default

AURORA-LM notes `x_0` and velocity "correspond" and then ablates all four
combinations of {`x_0`, velocity} × {loss space}, finding it matters. LDLM
predicts `x_0`. FlowLM states its own finding explicitly — "predicting clean
data to consistently guide the sampling process towards the true data
distribution" is *more effective* for flow matching.

Two of three that take a position choose `x_0`, but the disagreement is live
enough that hardcoding either silently picks a side in an open question. This
must be an explicit process/objective parameter.

## Where each piece attaches

### Process — the piece with a real anchor today

[#62](https://github.com/nishide-dev/unturtle/issues/62) established
`unturtle.processes`, and [#65](https://github.com/nishide-dev/unturtle/issues/65)
added the second implementation of it. `base.py` already says so:

> `ForwardProcess` is a structural contract, not a universal tensor schema.
> Masked-discrete diffusion, discrete flow matching, and continuous/latent
> methods each produce different `model_inputs`/`objective_inputs` keys.

A continuous process is a third implementation of a proven interface, not a new
abstraction. It differs from both existing ones in that `x_t` is continuous and
the supervision is a target tensor rather than token ids — which the protocol
already permits, since it fixes only *how* a process is called.

**Do not reuse `alpha(t)`.** Masked `alpha` is an absorbing-state survival
probability. Continuous paths interpolate; `kappa` in `DiscreteFlowProcess`
(#65) is the closer analogue and is already a separate protocol.

### Generation — already open

`unturtle/models/generation/sampler.py` was built for this. Its docstring:

> The registry is open to future families (discrete flow matching,
> continuous/latent). A family needs a name, a capability probe, and a runner —
> nothing else. [...] a continuous or flow family is never handed a concept
> [it does not have].

`GenerationAlgorithm` already carries a `family` field (`"masked_discrete"`
today) and `auto_priority` is explicit rather than registration-ordered, so a
new family cannot accidentally outrank the masked ones. **No change needed
here** — a continuous solver registers like any other algorithm.

## Acceleration inventory

What survives into continuous space, verified against this repo rather than
assumed:

**Does not apply — token/vocabulary-coupled by construction:**

| component | why |
|---|---|
| `kernels/masked_diffusion_loss.py` | cross-entropy over a vocabulary; `ignore_index=-100` semantics |
| `kernels/fused_masked_diffusion_loss.py` | same, plus `diffusion_mask` selection |
| `kernels/sparse_masked_loss.py` | gathers *masked token positions* to skip an LM-head GEMM; there is no LM head on a continuous denoiser |
| block-decode cache (`models/generation/`) | caches around progressively-unmasked token blocks |
| `MaskedDiffusionDataCollator` mask-token logic | requires a mask token |

The sparse LM-head path deserves a note: it is not merely inapplicable, it is
*pointless* in latent space. Its whole value is avoiding a `[B, L, V]`
projection, and a continuous denoiser's output is `[B, L, H]` — already the
small tensor. The analogous optimization would be on the **decoder**, at the
final token projection, which is a different place.

**Applies unchanged — agnostic to what the denoiser consumes:**

| component | condition |
|---|---|
| Unsloth LoRA / QLoRA | denoiser is a supported Transformer; a DiT with standard attention/MLP qualifies |
| Flash / SDPA / xFormers dispatch (`utils/attention_dispatch.py`) | operates on Q/K/V, indifferent to their origin |
| gradient checkpointing | transformers-standard, already used across backbones |
| `torch.compile` | no discrete-specific graph breaks in the process layer |

**Needs new work:**

- Triton fused objectives — the existing ones are CE kernels. An MSE/flow-matching
  objective over `[B, L, H]` is a different kernel, and probably not worth one:
  it is memory-bound elementwise work, not a vocabulary reduction.
- Packing for fixed-length latent blocks. The current packing utilities are
  built around `cu_seqlens`/`packed_seq_lengths` over token counts.
  AURORA-LM's block-causal structure may want something else; **open**.

## Prototype

Per #66, deliberately tiny and after this RFC settles, not before:

1. small learned codec (embedding + linear, `trainable=True`)
2. tiny DiT-style continuous denoiser
3. one objective, `x_0` prediction (the majority choice above)
4. Euler solver registered as a generation family
5. decode back to logits
6. `PreTrainedModel`-compatible save/reload

Its purpose is to test these interfaces, not to claim quality.

## Recommended first experiment

**FlowLM**, and the case is stronger than #66 states. It needs **no codec at
all** — it fine-tunes an existing diffusion LM toward straighter trajectories.
That means it exercises the continuous *process*, *objective* and *solver*
while sidestepping the one part of the boundary these papers disagree about.
It is also the cheapest: it reports reaching saturation in half the epochs of
training from scratch.

**DiLaDiff** is the natural second: its codec is fine-tuned from an existing
masked dLLM, which is exactly the asset Unturtle already has, and its latent
guides discrete decoding rather than replacing it — so the existing generation
path stays live.

## Open questions

| question | what closes it |
|---|---|
| Does `LatentBatch` need a mask distinct from `attention_mask`? | AURORA-LM's prefix-aligned layout in detail (§3.3) |
| Packing semantics for fixed-length latent blocks | a concrete block-causal implementation; may be "none" |
| Should the codec's auxiliary losses be weighted by the trainer or the codec? | LDLM's warmup schedule interacts with this; needs the prototype |
| Is a Triton objective kernel worth it? | measure first — likely memory-bound |

## Non-goals

Reproducing any of these papers at scale; a generic generative-model framework;
image-diffusion infrastructure.
