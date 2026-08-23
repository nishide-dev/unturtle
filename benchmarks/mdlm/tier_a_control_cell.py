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


"""#165: the Tier-A `masked_discrete` record — the PUBLISHED MDLM-OWT model.

Stage-0 freeze (issue #165):
  checkpoint kuleshov-group/mdlm-owt @ d0958fa851335ece6c15260ce0025f030673c0fb
             (Apache-2.0), loaded through the existing native conversion
             (`convert_mdlm_owt.load_mdlm_owt`) — the upstream remote code's
             flash-attn hard dependency never executes
  canonical  steps 128 (upstream `sampling.steps` default) + noise_removal,
             so NFE = 129; steps 1024 as a curve-only secondary
  samples    1000, seed 42, one cell-owned generator
  quality    hf_causal_evaluator("gpt2-large", revision="main",
             max_length=1024), corpus-pooled GenPPL + paired unigram entropy

Alignment with the upstream sampler, audited verbatim against
dev/repos/mdlm/diffusion.py (#165, recorded before any measurement):

  * `ddpm` / `ddpm_cache` update  ==  Unturtle `alg="origin"`.  Proven
    equidistributional, not assumed: under loglinear noise upstream's
    `move_chance_t` IS t, so its q_xs assigns P(stay masked) = s/t and
    P(token k) = (1 - s/t)*p_k — exactly the Bernoulli-then-categorical
    split `alg="origin"` performs.  Measured over 200k draws:
    P(mask) 0.749 (upstream) vs 0.753 (unturtle), theory 0.750.
    `ddpm_cache` is a caching optimization of `ddpm`, distribution-identical.
  * `sampling.noise_removal=True` (upstream config default) has NO
    equivalent in Unturtle's loop, so the producer supplies it —
    `mdlm_noise_removal`, one extra deterministic SUBS-argmax forward.
    This is why NFE is 129 and not 128.
  * SUBS is applied inside that final step only.  Unturtle's loop samples
    from raw logits; on this checkpoint that is numerically almost the same
    (measured P(mask token) = 3.9e-08 on an all-masked input), and the
    unmasked-pinning branch cannot differ because `alg="origin"` never
    rewrites a committed position.  The residual is recorded, not hidden.

Confounds are LABELLED: this is a 169M-param model trained by another group
on OpenWebText with its own budget.  It anchors "what published masked
discrete diffusion does", not an equal-compute control.
"""

import argparse
import json
import pathlib
import time

from unturtle.eval.producers import CANONICAL_EVALUATOR_REVISION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="kuleshov-group/mdlm-owt")
    parser.add_argument(
        "--revision", default="d0958fa851335ece6c15260ce0025f030673c0fb"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument(
        "--no-noise-removal",
        action="store_true",
        help="drop the upstream noise-removal step (NOT the official default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("smoke", "decision"),
        default="smoke",
        help="smoke: wiring only, NO Tier-A role claim. decision: verifies "
        "the frozen conditions and claims the role",
    )
    parser.add_argument("--evaluator-revision", default=CANONICAL_EVALUATOR_REVISION)
    parser.add_argument("--out", required=True)
    parser.add_argument("--owt-heldout", default="data/owt/heldout.bin")
    parser.add_argument("--eval-device", default=None)
    return parser.parse_args()


def load_model(args):
    import torch

    from unturtle.models.backbones.mdlm_dit.convert_mdlm_owt import load_mdlm_owt

    model = load_mdlm_owt(
        repo_id=args.repo, revision=args.revision, dtype=torch.float32
    )
    model.to(args.device).eval()
    mask_id = getattr(model.config, "mask_token_id", None)
    if mask_id is None:
        raise RuntimeError(
            f"{args.repo} config carries no mask_token_id — the masked "
            "diffusion loop cannot be pinned without it"
        )
    return model, int(mask_id)


def sample_batch(
    model,
    mask_id,
    batch_size,
    args,
    generator,
    step_counter=None,
    capture_trajectory=False,
    trajectory_out=None,
):
    """One batch through the Unturtle `mdlm` loop, then the upstream
    noise-removal step.  Returns the committed token ids."""
    import torch

    from unturtle.eval.producers import (
        global_rng_from,
        mdlm_noise_removal,
        pinned_global_rng,
    )

    # The loop needs a starting tensor; `_sample` pads to max_length with
    # the mask token, so a single-column all-mask prompt yields a fully
    # masked canvas of `sequence_length` — upstream's `_sample_prior`.
    prompt = torch.full((batch_size, 1), mask_id, dtype=torch.long, device=args.device)
    # The loop's own step_callback fires once per executed step, so the
    # recorded count is OBSERVED, not the request echoed back (#165 F3).
    observed = []
    # Committed-state snapshots, so the record can report MEASURED revision
    # rather than the theoretical capability (#167 review 3).  The loop's
    # stream_callback hands over a clone per step.
    trajectory = []
    with torch.no_grad(), pinned_global_rng(global_rng_from(generator)):
        out = model.generate(
            prompt,
            algorithm="mdlm",
            steps=args.steps,
            step_callback=lambda i, total: observed.append(i),
            stream_callback=(
                (lambda i, total, x: trajectory.append(x.detach().cpu().clone()))
                if capture_trajectory
                else None
            ),
            max_length=args.sequence_length,
            mask_token_id=mask_id,
            alg="origin",
            temperature=1.0,
            top_p=None,
            top_k=None,
            return_dict=False,
        )
        x = out if isinstance(out, torch.Tensor) else out.sequences
        if not args.no_noise_removal:
            x = mdlm_noise_removal(
                x,
                forward=lambda ids: model(input_ids=ids).logits,
                mask_index=mask_id,
            )
    if step_counter is not None:
        step_counter.append(max(observed) if observed else 0)
    if trajectory_out is not None:
        trajectory_out.extend(trajectory)
    return x


def generate_samples(model, mask_id, tokenizer, args):
    import torch

    generator = torch.Generator().manual_seed(args.seed)
    texts: list[str] = []
    ids_all = []
    step_counts: list[int] = []
    trajectory: list = []
    while len(texts) < args.num_samples:
        batch = min(args.batch_size, args.num_samples - len(texts))
        # Snapshots only on the FIRST batch: a full trajectory is
        # steps x batch x length and would dominate memory.
        first_batch = not texts
        x = sample_batch(
            model,
            mask_id,
            batch,
            args,
            generator,
            step_counter=step_counts,
            capture_trajectory=first_batch,
            trajectory_out=trajectory if first_batch else None,
        )
        for row in x:
            ids = row.tolist()
            ids_all.append(ids)
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        print(f"  generated {len(texts)}/{args.num_samples}", flush=True)
    if len(set(step_counts)) != 1:
        raise RuntimeError(
            f"batches executed differing step counts {sorted(set(step_counts))} "
            "— the cell would have no single executed step count to record"
        )
    return (
        texts[: args.num_samples],
        ids_all[: args.num_samples],
        step_counts[0],
        trajectory,
    )


def canonical_column(texts, token_ids, eos_id, args, device):
    from transformers import AutoTokenizer

    from unturtle.eval.frontier import hf_causal_evaluator
    from unturtle.eval.producers import (
        canonical_evaluator_identity,
        canonical_quality_column,
        guard_rows,
        guard_scope_note,
        stack_sample_ids,
    )

    evaluator, _raw_identity = hf_causal_evaluator(
        "gpt2-large",
        revision=args.evaluator_revision,
        device=device,
        max_length=1024,
    )
    identity = canonical_evaluator_identity(
        model="gpt2-large",
        revision=args.evaluator_revision,
        tokenizer_revision=args.evaluator_revision,
    )
    identity["max_length"] = 1024
    gpt2_tokenizer = AutoTokenizer.from_pretrained(
        "gpt2-large", revision=args.evaluator_revision
    )
    # MDLM trains on packed OWT, so EOS delimits documents rather than
    # ending generation: the whole 1024 canvas is the output and is what
    # gets decoded and scored.  Cutting at the first EOS measured 6.9
    # tokens per row against a ~1024-token scored text (#165 run 2).
    eos_means = "document_delimiter"
    content = guard_rows(token_ids, eos_id=eos_id, eos_means=eos_means)
    sample_ids, pad_meta = stack_sample_ids(content, pad_id=eos_id)
    quality = canonical_quality_column(
        texts,
        evaluator=evaluator,
        evaluator_identity=identity,
        tokenize=gpt2_tokenizer.encode,
        sample_ids=sample_ids,
    )
    quality_scope = {
        "guard_input": guard_scope_note(eos_means=eos_means),
        **pad_meta,
    }
    mauve_note = None
    heldout = pathlib.Path(args.owt_heldout)
    heldout_meta = heldout.parent / f"{heldout.name}.json"
    if heldout.is_file() and heldout_meta.exists():
        import numpy as np

        from unturtle.eval import mauve_score

        meta = json.loads(heldout_meta.read_text())
        memmap = np.memmap(
            heldout,
            dtype=np.uint16,
            mode="r",
            shape=(meta["num_rows"], meta["block_size"]),
        )
        reference = [
            gpt2_tokenizer.decode(
                np.asarray(row, dtype=np.int64).tolist(), skip_special_tokens=True
            )
            for row in memmap[: len(texts)]
        ]
        quality["mauve"] = mauve_score(
            reference, texts, featurize_model_name="gpt2", max_text_length=256
        )
        quality["mauve_settings"] = {
            "featurize_model_name": "gpt2",
            "max_text_length": 256,
            "num_buckets": "auto",
            "reference": "#130 OWT held-out (gpt2-decoded packed rows)",
        }
    else:
        mauve_note = f"reference texts not found at {heldout}; MAUVE omitted"
    return quality, mauve_note, quality_scope


def throughput_cells(model, mask_id, args):
    import torch

    from unturtle.eval.producers import mdlm_nfe, measure_control_throughput

    def run_batch(batch_size, generator):
        cell_steps: list[int] = []
        sample_batch(
            model, mask_id, batch_size, args, generator, step_counter=cell_steps
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        executed = cell_steps[0] if cell_steps else 0
        return {
            "forwards_executed": mdlm_nfe(
                steps_executed=executed,
                noise_removal=not args.no_noise_removal,
            ),
            "steps_executed": executed,
            "sequence_length": args.sequence_length,
        }

    def warmup():
        run_batch(1, torch.Generator().manual_seed(args.seed))

    return measure_control_throughput(run_batch, seed=args.seed, warmup=warmup)


def main():
    import torch
    from transformers import AutoTokenizer

    from unturtle.eval.frontier import write_jsonl
    from unturtle.eval.producers import (
        build_control_record,
        canvas_diagnostics,
        decision_preflight,
        mdlm_nfe,
        revision_diagnostics,
        stack_sample_ids,
    )

    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_device = args.eval_device or args.device
    noise_removal = not args.no_noise_removal

    heldout = pathlib.Path(args.owt_heldout)
    claimed_role = decision_preflight(
        mode=args.mode,
        role="masked_discrete",
        num_samples=args.num_samples,
        seed=args.seed,
        mauve_available=heldout.is_file()
        and (heldout.parent / f"{heldout.name}.json").exists(),
        evaluator_revision=args.evaluator_revision,
    )

    model, mask_id = load_model(args)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Warm the kernels outside every timed region (the AR producer's #165
    # smoke showed a fresh process pays ~160 s of compile on its first call).
    warm_args = argparse.Namespace(**vars(args))
    warm_args.steps = 2
    warm_args.sequence_length = 64
    sample_batch(model, mask_id, 1, warm_args, torch.Generator().manual_seed(0))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    texts, ids_all, observed_steps, trajectory = generate_samples(
        model, mask_id, tokenizer, args
    )
    generation_seconds = time.perf_counter() - started
    (out / "samples.json").write_text(json.dumps(texts, ensure_ascii=False))

    residual_masks = sum(row.count(mask_id) for row in ids_all)

    throughput = throughput_cells(model, mask_id, args)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    quality, mauve_note, quality_scope = canonical_column(
        texts, ids_all, tokenizer.eos_token_id, args, eval_device
    )
    full_width_ids, _ = stack_sample_ids(ids_all, pad_id=tokenizer.eos_token_id)
    content_widths = [
        len(row)
        if tokenizer.eos_token_id not in row
        else row.index(tokenizer.eos_token_id)
        for row in ids_all
    ]

    record = build_control_record(
        role=claimed_role,
        family="mdlm",
        method="mdlm-owt",
        checkpoint=f"{args.repo}@{args.revision}",
        seed=args.seed,
        steps_requested=args.steps,
        steps_executed=observed_steps,
        quality=quality,
        systems={
            "nfe": mdlm_nfe(steps_executed=observed_steps, noise_removal=noise_removal),
            "sequence_length": args.sequence_length,
            "solver": "ddpm-ancestral" + ("+noise-removal" if noise_removal else ""),
            "throughput": throughput,
            "peak_memory_bytes": peak,
        },
        decoding={
            "algorithm": "mdlm",
            "alg": "origin",
            "steps": args.steps,
            "noise_removal": noise_removal,
            "temperature": 1.0,
            "top_p": None,
            "top_k": None,
            "dtype": "float32",
            "mask_token_id": mask_id,
        },
        confounds=[
            "scale/budget: 169M-param model trained by another group with "
            "its own compute budget — a published anchor, not an "
            "equal-compute control",
            "tokenizer: gpt2 BPE with an appended mask id (50258 vs 50257)",
            "sampler residual: SUBS is applied only in the noise-removal "
            "step; the loop samples raw logits (measured P(mask) = 3.9e-08 "
            "on this checkpoint, so numerically near-identical)",
        ],
        official={
            "status": "not_measured_here",
            "reason": "the MDLM paper reports GenPPL under its own "
            "evaluator and sample budget; this cell reports the #152 "
            "canonical column only, and the two must not be compared "
            "directly (#165 Stage-0 freeze)",
        },
        extra={
            "generation_seconds": generation_seconds,
            "steps_requested": args.steps,
            "steps_executed": observed_steps,
            # Counted AFTER noise removal, which scrubs the mask column by
            # construction — so 0 is expected under the default and only
            # informative with --no-noise-removal (#165 review F5).
            "residual_mask_tokens": residual_masks,
            "residual_mask_scope": (
                "post-noise-removal (structurally 0: SUBS gives the mask "
                "column -inf before the argmax)"
                if noise_removal
                else "post-loop, no noise removal — a non-zero count here "
                "means the loop left masks uncommitted"
            ),
            "upstream_alignment": {
                "reference": "dev/repos/mdlm/diffusion.py",
                "ddpm_vs_origin": "equidistributional (measured 0.749 vs "
                "0.753 P(mask) over 200k draws; theory 0.750)",
                "noise_removal": "supplied by the producer "
                "(mdlm_noise_removal), one deterministic SUBS-argmax forward",
                "ddpm_cache": "omitted — a caching optimization of ddpm, "
                "distribution-identical",
            },
            "mauve_note": mauve_note,
            "quality_scope": quality_scope,
            "mode": args.mode,
            "full_width_diagnostics": canvas_diagnostics(
                full_width_ids, content_widths=content_widths
            ),
            # MEASURED revision from the first batch's committed states —
            # `net_revision_stats` previously reached no record (#167 F3).
            "revision_diagnostics": revision_diagnostics(trajectory),
        },
    )
    write_jsonl([record], out / "frontier_record.jsonl")
    print(json.dumps(record, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
