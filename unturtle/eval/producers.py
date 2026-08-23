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
    "ar_batch_forwards",
    "canvas_diagnostics",
    "content_rows",
    "guard_rows",
    "guard_scope_note",
    "revision_diagnostics",
    "canonical_evaluator_identity",
    "decision_preflight",
    "stack_sample_ids",
    "ar_generation_config",
    "ar_nfe_from_batches",
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
    steps_requested: int | None = None,
    steps_executed: int | None = None,
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
        # The protocol's own step fields, so its "requested without
        # executed" validator can see them — stashing them in `extra` made
        # that check structurally unreachable (#165 review F3).
        steps_requested=steps_requested,
        steps_executed=steps_executed,
        extra=merged_extra,
    )


def measure_control_throughput(
    run_batch: Callable[[int, Any], Any],
    *,
    seed: int,
    warmup: Callable[[], Any] | None = None,
    unsupported: dict[int, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Protocol throughput cells, each carrying its own EXECUTED work.

    Wraps `measure_throughput_cells` so every producer inherits the same
    one-generator / warmup-outside / typed-cell discipline, and merges
    whatever `run_batch` returns into that batch's cell.  Natural EOS makes
    forward counts differ by batch size, and the record's top-level NFE
    comes from the quality run's batch — so without per-cell work, batch
    scaling and generation-length differences cannot be separated (#167
    review 5).

    `run_batch` may return a mapping of work counters (e.g.
    `forwards_executed`, `content_length_mean`); `token_work` (forwards x
    batch size) is derived when the forward count is present.  A
    `run_batch` that returns nothing leaves the cell with timings only —
    work is never fabricated.
    """
    work: dict[int, Any] = {}

    def run_and_capture(batch_size: int, generator: Any) -> Any:
        result = run_batch(batch_size, generator)
        if isinstance(result, dict):
            work[batch_size] = result
        return result

    cells = measure_throughput_cells(
        run_and_capture, seed=seed, warmup=warmup, unsupported=unsupported
    )
    for batch_size, counters in work.items():
        cell_entry = cells.get(f"batch_{batch_size}")
        if cell_entry is None or cell_entry.get("status") != "ok":
            continue
        cell_entry["value"].update(counters)
        forwards = counters.get("forwards_executed")
        if forwards is not None:
            cell_entry["value"]["token_work"] = int(forwards) * int(batch_size)
    return cells


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
    sample_ids: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The #152 canonical quality cells for one generation run.

    #153 and #155 each hand-rolled this column; the AR and MDLM producers
    share this one instead so the canonical column cannot drift a third
    time.  `evaluator`/`tokenize` are injected — the caller supplies the
    frozen `hf_causal_evaluator("gpt2-large", ...)` and the matching
    tokenizer, and tests supply fakes.  Corpus-pooled entropy under the
    common tokenizer (NOT the ELF or FLM official entropy semantics).

    `sample_ids` (a `[N, L]` token tensor in the model's OWN vocabulary) is
    MANDATORY: `diversity_guards` rides in the canonical column of every
    existing frontier record, and `_KNOWN_QUALITY_KEYS` reserves its three
    slots.  Since GenPPL is entropy-sensitive, an arm without the guards is
    the arm where a collapsed but low-GenPPL sample set goes unflagged
    (#165 review F2).
    """
    from unturtle.eval.generation_metrics import diversity_guards

    if sample_ids is None:
        raise ValueError(
            "sample_ids is required: the canonical column carries "
            "diversity_guards (distinct_fraction / pooled_unigram_entropy / "
            "unique_rows_fraction) in every frontier record, and dropping "
            "them silently is the drift this helper exists to prevent"
        )
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
        **diversity_guards(sample_ids),
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

    # The draw must happen on the GENERATOR's device: torch.randint requires
    # the two to match, and a cell generator may itself be device-side.
    device = getattr(generator, "device", None)
    return int(
        torch.randint(0, 2**31 - 1, (1,), generator=generator, device=device).item()
    )


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


def ar_batch_forwards(*, generated_width: int) -> int:
    """Denoiser calls one AR batch actually executed.

    `generate()` does not stop when ONE row emits EOS — it keeps forwarding
    the whole batch until every row finishes or `max_new_tokens` is hit, and
    each forward processes the full batch (#165 review F1: 32 forwards
    executed against a 24.75 mean of truncated lengths).

    The executed count is the WIDTH of the generated tensor, taken directly
    rather than reconstructed from content lengths: EOS is excluded from a
    content length, so `max(lengths)` is off by one whenever any row stops.
    Measured on gpt2 with EOS forced on the first generated token: one
    forward ran while every content length was 0 (#167 review 2).
    """
    if generated_width < 1:
        raise ValueError(
            f"generated width {generated_width} implies no forwards; a batch "
            "that ran produced at least one column"
        )
    return int(generated_width)


def ar_nfe_from_batches(batches: list[tuple[int, int]]) -> float:
    """Per-sample NFE across a cell's batches.

    `batches` is `[(forwards_executed, batch_size), ...]`.  Every sample in
    a batch paid for every forward that batch executed, so the per-sample
    figure is the sample-weighted mean of the per-batch forward counts.
    """
    if not batches:
        raise ValueError("no batches — the cell generated nothing")
    total_samples = sum(size for _, size in batches)
    if total_samples == 0:
        raise ValueError("no samples across the recorded batches")
    return sum(forwards * size for forwards, size in batches) / total_samples


def stack_sample_ids(
    rows: list[list[int]], *, pad_id: int
) -> tuple[Any, dict[str, Any]]:
    """Stack per-sample id lists into the `[N, L]` tensor the guards need.

    Batches can end at different widths — a batch whose rows all stop early
    is narrower than one that runs to the budget — so `torch.tensor(rows)`
    raises a ragged-tensor error.  Padding is explicit and REPORTED: the
    filler inflates `distinct_fraction`-style guards toward whatever token
    `pad_id` is, so a record must be able to say how much of its guard
    input was filler (#167 review 2).
    """
    import torch

    if not rows:
        raise ValueError("no rows to stack — the cell generated nothing")
    width = max(len(row) for row in rows)
    padded_rows = sum(1 for row in rows if len(row) < width)
    stacked = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for index, row in enumerate(rows):
        if row:
            stacked[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return stacked, {
        "width": int(width),
        "pad_id": int(pad_id),
        "padded_rows": int(padded_rows),
        "row_count": len(rows),
    }


#: The frozen decision-run conditions (#165 Stage-0 freeze, amended by the
#: #167 review: the canonical evaluator is pinned to a commit, not `main`).
DECISION_SAMPLE_COUNT = 1000
DECISION_SEED = 42
CANONICAL_EVALUATOR_MODEL = "gpt2-large"
CANONICAL_EVALUATOR_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"

_FLOATING_REVISIONS = frozenset({"main", "master", "refs/heads/main", "HEAD", ""})


def canonical_evaluator_identity(
    *,
    model: str,
    revision: str | None,
    tokenizer_revision: str | None,
) -> dict[str, str]:
    """The canonical evaluator's IMMUTABLE identity.

    `hf_causal_evaluator` records whatever revision it is handed, so passing
    `main` produces an identity that cannot name the commit that scored the
    run — and `main` moves.  GenPPL is not comparable across evaluator
    identities, so a floating revision silently breaks the one property the
    protocol relies on (#167 review 4).

    The entropy tokenizer is pinned separately because it moves
    independently of the scorer, and the `transformers` version rides along:
    a scoring change in the library is not visible in either SHA.
    """
    import transformers

    for label, value in (
        ("revision", revision),
        ("tokenizer revision", tokenizer_revision),
    ):
        if value is None or str(value).strip() in _FLOATING_REVISIONS:
            raise ValueError(
                f"canonical evaluator {label} {value!r} is not an identity — "
                "pin an immutable commit SHA (a branch name moves, so "
                "GenPPL values recorded under it are not comparable)"
            )
    return {
        "model": model,
        "revision": str(revision),
        "tokenizer_revision": str(tokenizer_revision),
        "transformers_version": transformers.__version__,
    }


def decision_preflight(
    *,
    mode: str,
    role: str,
    num_samples: int,
    seed: int,
    mauve_available: bool,
    evaluator_revision: str | None,
) -> str | None:
    """Whether this run may claim its Tier-A role.

    Returns the role in `decision` mode once every frozen condition holds,
    and `None` in `smoke` mode — a record with `tier_a_role=None` cannot
    close a gap in `tier_a_gaps()`.

    Before this, `--num-samples 4` produced a role-claiming record and a
    missing MAUVE reference only left a note, so a wiring smoke satisfied
    the coverage check exactly like a decision run (#167 review 1).
    """
    if mode not in ("smoke", "decision"):
        raise ValueError(
            f"unknown mode {mode!r}: use 'smoke' (no role claim) or "
            "'decision' (verified frozen conditions)"
        )
    if mode == "smoke":
        return None
    if num_samples != DECISION_SAMPLE_COUNT:
        raise ValueError(
            f"decision mode requires the frozen sample budget "
            f"{DECISION_SAMPLE_COUNT}, got {num_samples}"
        )
    if seed != DECISION_SEED:
        raise ValueError(
            f"decision mode requires the frozen seed {DECISION_SEED}, got {seed}"
        )
    if not mauve_available:
        raise ValueError(
            "decision mode requires the MAUVE reference (#130 OWT held-out); "
            "a missing reference is a blocked run, not a note on a valid one"
        )
    canonical_evaluator_identity(
        model=CANONICAL_EVALUATOR_MODEL,
        revision=evaluator_revision,
        tokenizer_revision=evaluator_revision,
    )
    return role


def content_rows(rows: list[list[int]], *, eos_id: int) -> list[list[int]]:
    """Each row cut at its first EOS — the tokens a reader actually sees.

    The canonical guards must run on these, not on the full canvas: a
    denoised tail nobody reads contributes its own diversity and can hide a
    collapsed decoded region (#167 review 3).  An empty content row is
    RETURNED as empty rather than dropped, so the guard denominator cannot
    shrink silently.
    """
    cut = []
    for row in rows:
        if eos_id in row:
            cut.append(row[: row.index(eos_id)])
        else:
            cut.append(list(row))
    return cut


def canvas_diagnostics(canvas: Any, *, content_widths: list[int]) -> dict[str, Any]:
    """Full-canvas entropy/diversity, under CANVAS-prefixed names.

    Kept deliberately distinct from the canonical guard keys so a
    canvas-wide number can never be read as the canonical column's collapse
    detection (#167 review 3).
    """
    from unturtle.eval.generation_metrics import (
        distinct_fraction,
        pooled_unigram_entropy,
    )

    return {
        "canvas_width": int(canvas.shape[-1]),
        "canvas_pooled_unigram_entropy": pooled_unigram_entropy(canvas),
        "canvas_distinct_fraction": distinct_fraction(canvas),
        "content_width_mean": (
            sum(content_widths) / len(content_widths) if content_widths else 0.0
        ),
    }


def revision_diagnostics(trajectory: list[Any]) -> dict[str, Any]:
    """Measured revision for the record, or an explicit "not captured".

    `net_revision_stats` existed and was tested but reached no record — the
    Sumi producer kept step NUMBERS only, so no cell ever carried measured
    revision (#167 review 3).  A trajectory shorter than two snapshots is
    reported as uncaptured rather than as zero revision, which would read as
    a measurement.
    """
    if len(trajectory) < 2:
        return {
            "status": "not_captured",
            "reason": "fewer than two committed-state snapshots were kept; "
            "revision cannot be measured from a single state",
        }
    stats = net_revision_stats(trajectory)
    stats["status"] = "measured"
    return stats


#: What an EOS token means for a family, which decides the guard scope.
_EOS_SEMANTICS = {
    # AR: EOS ends generation, everything after it is padding the model
    # never produced as content.
    "end_of_generation": "cut each row at its first EOS",
    # Masked / uniform diffusion on a fixed canvas: the model was trained
    # on packed text where EOS delimits documents, so an early EOS is
    # ordinary content.  The canvas IS the output — it is what gets decoded
    # and scored — so the guards must see all of it.
    "document_delimiter": "keep the whole canvas row",
}


def guard_rows(
    rows: list[list[int]], *, eos_id: int, eos_means: str
) -> list[list[int]]:
    """Guard input rows under the family's EOS semantics.

    The guards must measure what the EVALUATOR scored.  Getting this wrong
    is not a cosmetic mismatch: the first #165 decision run cut MDLM rows at
    the first gpt2 EOS and reported `distinct_fraction 0.0047` /
    `pooled_unigram_entropy 0.086` off an average of 6.9 tokens per row,
    while GenPPL and entropy scored the full ~1024-token decoded canvas.
    MDLM trains on packed OWT, so its EOS is a document delimiter, and the
    frozen ELF/FMLM precedent passes ALL generated ids to the guards.
    """
    if eos_means not in _EOS_SEMANTICS:
        raise ValueError(
            f"unknown eos_means {eos_means!r}; choose "
            f"{sorted(_EOS_SEMANTICS)} — the guard scope has to match what "
            "the evaluator scored"
        )
    if eos_means == "document_delimiter":
        return [list(row) for row in rows]
    return content_rows(rows, eos_id=eos_id)


def guard_scope_note(*, eos_means: str) -> str:
    """The record's own statement of which guard scope it used."""
    if eos_means not in _EOS_SEMANTICS:
        raise ValueError(f"unknown eos_means {eos_means!r}")
    return f"{eos_means}: {_EOS_SEMANTICS[eos_means]}"
