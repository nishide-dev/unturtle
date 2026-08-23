# Copyright 2025-present nishide-dev & the Unturtle team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tier-A control-producer helpers (#165).

The #152 protocol owns measurement; #153/#155 own the flow families.  This
module owns only the small, SHARED discipline the three remaining control
roles (`ar_control`, `masked_discrete`, `uniform_state`) need, so each
producer script cannot re-invent it differently:

- an AR control must be COMPETENT — a no-cache configuration is refused
  rather than quietly measured (the issue's headline mutation target);
- AR NFE is one forward per generated token, and must be the EXECUTED
  count;
- a control record carries its role, its official column kept separate
  from the canonical one, and its **mandatory** confound labels (scale,
  training data, tokenizer) — the protocol forbids calling an unmatched
  control "matched", so the record cannot omit them;
- the cell's single generator threads through every batch;
- iterative samplers report OBSERVED net revision, never revision
  capability inferred from theory.

Deliberately NOT here: any universal AR/discrete model abstraction, or any
model loading.  Producers stay thin scripts under `benchmarks/`.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable

from unturtle.eval.frontier import (
    FRONTIER_PROTOCOL,
    frontier_record,
    generative_perplexity,
    measure_throughput_cells,
    text_unigram_entropy,
)

__all__ = [
    "ar_generation_config",
    "canonical_quality_column",
    "derive_device_generator",
    "global_rng_from",
    "mdlm_nfe",
    "mdlm_noise_removal",
    "subs_parameterization",
    "uniform_state_compute_scope",
    "uniform_state_nfe",
    "pinned_global_rng",
    "ar_nfe",
    "build_control_record",
    "measure_control_throughput",
    "net_revision_stats",
]


def ar_generation_config(
    *,
    use_cache: bool = True,
    attn_implementation: str = "sdpa",
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """The frozen competent-AR settings (Stage-0 freeze).

    KV cache is mandatory: #152 requires the AR control to be a competent
    optimized path, and the protocol explicitly forbids comparing a
    compiled/cached diffusion path against a naive AR loop.  Sampling
    carries NO truncation (top-k/top-p) because the diffusion anchors use
    none either.
    """
    if not use_cache:
        raise ValueError(
            "the AR control must run with the KV cache enabled — a no-cache "
            "loop is not a competent optimized control (#152 protocol / "
            "#165 mutation target)"
        )
    return {
        "use_cache": True,
        "attn_implementation": attn_implementation,
        "do_sample": True,
        "temperature": temperature,
        "top_k": None,
        "top_p": None,
        "max_new_tokens": max_new_tokens,
    }


def ar_nfe(*, generated_tokens: int | None) -> int:
    """AR denoiser-call accounting: one forward per generated token.

    Recorded from what was GENERATED, never from a requested length — an
    early EOS shortens the run and the record must follow.  (This number is
    not comparable to a diffusion step count; the producer notes that in
    the record.)
    """
    if generated_tokens is None:
        raise ValueError(
            "AR NFE requires the executed generated-token count; a requested "
            "length is not evidence of what ran (#165 mutation target)"
        )
    return int(generated_tokens)


def build_control_record(
    *,
    role: str,
    family: str,
    method: str,
    checkpoint: str,
    seed: int,
    quality: dict[str, Any],
    systems: dict[str, Any],
    confounds: list[str],
    official: dict[str, Any],
    decoding: Any = None,
    provider: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A protocol-v1 record for one Tier-A control role.

    Adds the two producer-level guards `frontier_record` does not own:
    `confounds` must be non-empty, and the official column is forced into
    `extra` so it can never be conflated with the canonical `quality`
    fields.  Role validity and the DFM-as-uniform_state refusal stay with
    `frontier_record` (#152) — duplicating them here was verified inert by
    the #165 mutation battery (mutant M3 could not be killed because the
    protocol layer already rejects both cases).
    """
    if not confounds:
        raise ValueError(
            "confounds must be recorded explicitly (scale / training data / "
            "tokenizer): the protocol forbids presenting an unmatched "
            "control as matched (#165). Pass ['none'] only when the control "
            "is genuinely matched on every axis."
        )
    overlap = set(quality) & {"genppl_official", "entropy_official_native"}
    if overlap:
        raise ValueError(
            f"official-evaluator keys {sorted(overlap)} must not appear in "
            "the canonical quality column — the two evaluator columns stay "
            "separate (#152/#165)"
        )
    merged_extra = dict(extra or {})
    merged_extra["official_column"] = official
    merged_extra["confounds"] = list(confounds)
    return frontier_record(
        family=family,
        method=method,
        checkpoint=checkpoint,
        seed=seed,
        tier_a_role=role,
        provider=provider,
        quality=quality,
        systems=systems,
        decoding=decoding,
        extra=merged_extra,
    )


def measure_control_throughput(
    run_batch: Callable[[int, Any], Any],
    *,
    seed: int,
    warmup: Callable[[], Any] | None = None,
    unsupported: dict[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Protocol throughput cells for a control producer — a thin pass-through
    to `measure_throughput_cells` so every producer inherits the same
    one-generator / warmup-outside / typed-cell discipline."""
    return measure_throughput_cells(
        run_batch, seed=seed, warmup=warmup, unsupported=unsupported
    )


def net_revision_stats(trajectory: list[Any]) -> dict[str, Any]:
    """Measure how much an iterative sampler ACTUALLY revises.

    ``trajectory`` is a list of committed-token snapshots (identical shape,
    one per observed step).  Reports the number of positions whose value
    changed at least once after its first snapshot — evidence about real
    revision, as opposed to the theoretical claim that a uniform/masked
    sampler "can" revise (#152 Sumi note, #165 mutation target).
    """
    if len(trajectory) < 2:
        raise ValueError(
            "net revision needs at least two snapshots; a single state "
            "cannot show whether any token changed"
        )
    import torch

    first = trajectory[0]
    changed = torch.zeros_like(first, dtype=torch.bool)
    events = 0
    previous = first
    for snapshot in trajectory[1:]:
        if snapshot.shape != first.shape:
            raise ValueError(
                f"snapshot shape {tuple(snapshot.shape)} != "
                f"{tuple(first.shape)}; net revision compares aligned states"
            )
        step_changed = snapshot != previous
        changed |= step_changed
        events += int(step_changed.sum())
        previous = snapshot
    total = int(changed.numel())
    revised = int(changed.sum())
    return {
        "revised_positions": revised,
        "total_positions": total,
        "revision_fraction": revised / total if total else 0.0,
        # How many times a committed token was OVERWRITTEN, summed over
        # positions and steps.  This is the quantity `revised_positions`
        # cannot express: a token that flips away and back counts twice.
        # (Cumulative "differs from predecessor" and "differs from the
        # first state" select provably identical position sets, so the
        # event count is the only observable difference between them.)
        "revision_events": events,
        "steps_observed": len(trajectory),
    }


def canonical_quality_column(
    texts: list[str],
    *,
    evaluator: Callable[[str], tuple[float, int]],
    evaluator_identity: dict[str, str],
    tokenize: Callable[[str], list[int]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The #152 canonical quality cells for one generation run.

    #153 and #155 each hand-rolled this column; the AR and MDLM producers
    share this one instead so the canonical column cannot drift a third
    time.  `evaluator`/`tokenize` are injected — the caller supplies the
    frozen `hf_causal_evaluator("gpt2-large", ...)` and the matching
    tokenizer, and tests supply fakes.  Corpus-pooled entropy under the
    common tokenizer (NOT the ELF or FLM official entropy semantics).
    """
    # No empty-input guard here: `generative_perplexity` already refuses a
    # zero-text run, and the #165 battery could not kill a duplicate
    # (mutant M11 survived because the lower layer raises first).
    genppl = generative_perplexity(
        texts, evaluator=evaluator, evaluator_identity=evaluator_identity
    )
    quality = {
        "genppl": genppl["genppl"],
        "genppl_evaluator": dict(evaluator_identity),
        "unigram_entropy": text_unigram_entropy(texts, tokenize=tokenize),
        "sample_count": len(texts),
        "collapse_flags": [],
    }
    quality.update(extra or {})
    return quality


def global_rng_from(generator: Any) -> int:
    """Draw one seed from the cell's generator, advancing its stream.

    `transformers.generate()` has no `generator=` parameter — passing one
    raises `ValueError: The following model_kwargs are not used by the
    model` — and its sampling reads the GLOBAL torch RNG.  The protocol's
    "one cell-owned generator" is therefore honoured indirectly: every
    batch's global seed is DRAWN from the cell generator, so the stream
    advances exactly as it would if the sampler took the generator
    directly, and a per-batch reset cannot masquerade as compliance.
    """
    import torch

    return int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())


@contextlib.contextmanager
def pinned_global_rng(seed: int):
    """Pin the global torch RNG to `seed`, then restore the caller's state.

    Without the restore, a generation call would silently reposition the
    global stream that everything else in the producer (MAUVE subsampling,
    a later throughput cell) draws from.
    """
    import torch

    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        yield seed
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


#: Upstream MDLM's stand-in for -inf; kept as a finite value so
#: `logsumexp` stays well-behaved (dev/repos/mdlm/diffusion.py).
_NEG_INFINITY = -1_000_000.0


def subs_parameterization(logits: Any, xt: Any, *, mask_index: int) -> Any:
    """Upstream MDLM SUBS parameterization, ported verbatim.

    Reference: `MDLM._subs_parameterization`
    (dev/repos/mdlm/diffusion.py:261-277).  Three steps, in order:

    1. the mask column gets -inf, so the model can never re-emit MASK;
    2. renormalize to log-probabilities;
    3. UNMASKED positions are pinned to the token they already hold
       (-inf everywhere else, 0 at the held token) — a committed token is
       never revised.

    On the published mdlm-owt checkpoint step 1 is nearly a no-op — the
    trained model puts P(mask) ~ 3.9e-08 on an all-masked input — but it is
    load-bearing for the argmax in :func:`mdlm_noise_removal`, where a
    single -inf decides whether a literal mask token can be committed.
    """
    import torch

    logits = logits.clone()
    logits[:, :, mask_index] += _NEG_INFINITY
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked = xt != mask_index
    logits[unmasked] = _NEG_INFINITY
    logits[unmasked, xt[unmasked]] = 0.0
    return logits


def mdlm_noise_removal(
    x: Any,
    *,
    forward: Callable[[Any], Any],
    mask_index: int,
) -> Any:
    """Upstream MDLM's `sampling.noise_removal` final step (config default).

    Reference: `MDLM._sample`, the `if self.config.sampling.noise_removal`
    tail (dev/repos/mdlm/diffusion.py:690-696).  ONE extra forward at
    t = eps whose SUBS argmax replaces the whole sequence — deterministic,
    no temperature, no sampling.  Unturtle's `alg="origin"` loop has no
    equivalent (its last step samples the remaining masks instead), so the
    producer adds this rather than editing the core loop.

    SUBS runs INSIDE this step: without it a raw argmax could commit the
    mask id itself, and a literal mask token would reach the evaluator.
    """
    logits = forward(x)
    return subs_parameterization(logits, x, mask_index=mask_index).argmax(dim=-1)


def mdlm_nfe(*, steps_executed: int, noise_removal: bool) -> int:
    """Denoiser calls for one MDLM sample: the executed loop steps plus the
    noise-removal forward when it runs.  Upstream's default (steps 128 +
    noise_removal) is 129 calls, not 128 — a compute cell that reports 128
    understates the cost of the official configuration."""
    return int(steps_executed) + (1 if noise_removal else 0)


def uniform_state_nfe(*, steps_executed: int | None) -> int:
    """Denoiser calls for one uniform-state sample.

    Sumi's ancestral sampler runs exactly one forward per denoising step and
    adds no tail step (audited: `SumiGenerationMixin.generate`, the
    `for step in range(num_denoising_steps)` loop in generation_sumi.py @
    0d20f7becf84).  Contrast MDLM, whose official configuration adds a
    noise-removal forward — see :func:`mdlm_nfe`.
    """
    if steps_executed is None:
        raise ValueError(
            "uniform-state NFE requires the executed step count; a requested "
            "step count is not evidence of what ran"
        )
    return int(steps_executed)


def uniform_state_compute_scope(
    *,
    canvas_length: int,
    content_budget: int,
    prompt_length: int,
) -> dict[str, Any]:
    """What a uniform-state cell actually forwarded, versus what it kept.

    Sumi is trained on a packed fixed-length canvas and denoises the WHOLE
    canvas every step (default `canvas_length=2048`, ceiling 4864 from
    `max_position_embeddings`); `max_new_tokens` is only the content budget
    before the anchored EOS,BOS delimiter, and decoding is cut at the first
    EOS.  A cell that reports the content budget as its sequence length
    understates its own compute — so the forwarded canvas is the recorded
    `sequence_length`, and the #152 context-1024 condition is reported as
    matched or not rather than assumed.
    """
    if content_budget > canvas_length:
        raise ValueError(
            f"content_budget {content_budget} exceeds canvas_length "
            f"{canvas_length}: the canvas bounds what can be generated"
        )
    return {
        "sequence_length": int(canvas_length),
        "forwarded_tokens": int(canvas_length),
        "content_budget": int(content_budget),
        "prompt_length": int(prompt_length),
        "protocol_context_match": int(canvas_length)
        == int(FRONTIER_PROTOCOL["context_length"]),
        "note": "Sumi denoises the full canvas every step; sequence_length "
        "is the forwarded canvas, not the content budget",
    }


def derive_device_generator(generator: Any, *, device: Any) -> Any:
    """A generator on `device`, seeded from the cell's generator.

    `measure_throughput_cells` (#152) owns one CPU generator, but some
    native samplers require the generator to live on the model's device —
    Sumi's `_ancestral_step` raises `RuntimeError: Expected a 'cuda' device
    type for generator but found 'cpu'`.  Deriving keeps the protocol's
    single-stream property: the cell generator advances exactly once per
    derivation, so consecutive batches cannot share an RNG stream.

    Never short-circuits when the devices already match — skipping the draw
    would stop the cell stream from advancing.
    """
    import torch

    seed = global_rng_from(generator)
    derived = torch.Generator(device=device)
    derived.manual_seed(seed)
    return derived
