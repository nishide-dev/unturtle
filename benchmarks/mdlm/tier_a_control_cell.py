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


def sample_batch(model, mask_id, batch_size, args, generator):
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
    with torch.no_grad(), pinned_global_rng(global_rng_from(generator)):
        out = model.generate(
            prompt,
            algorithm="mdlm",
            steps=args.steps,
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
    return x


def generate_samples(model, mask_id, tokenizer, args):
    import torch

    generator = torch.Generator().manual_seed(args.seed)
    texts: list[str] = []
    ids_all = []
    while len(texts) < args.num_samples:
        batch = min(args.batch_size, args.num_samples - len(texts))
        x = sample_batch(model, mask_id, batch, args, generator)
        for row in x:
            ids = row.tolist()
            ids_all.append(ids)
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        print(f"  generated {len(texts)}/{args.num_samples}", flush=True)
    return texts[: args.num_samples], ids_all[: args.num_samples]


def canonical_column(texts, args, device):
    from transformers import AutoTokenizer

    from unturtle.eval.frontier import hf_causal_evaluator
    from unturtle.eval.producers import canonical_quality_column

    evaluator, identity = hf_causal_evaluator(
        "gpt2-large", revision="main", device=device, max_length=1024
    )
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    quality = canonical_quality_column(
        texts,
        evaluator=evaluator,
        evaluator_identity=identity,
        tokenize=gpt2_tokenizer.encode,
    )
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
    return quality, mauve_note


def throughput_cells(model, mask_id, args):
    import torch

    from unturtle.eval.producers import measure_control_throughput

    def run_batch(batch_size, generator):
        sample_batch(model, mask_id, batch_size, args, generator)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def warmup():
        run_batch(1, torch.Generator().manual_seed(args.seed))

    return measure_control_throughput(run_batch, seed=args.seed, warmup=warmup)


def main():
    import torch
    from transformers import AutoTokenizer

    from unturtle.eval.frontier import write_jsonl
    from unturtle.eval.producers import build_control_record, mdlm_nfe

    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_device = args.eval_device or args.device
    noise_removal = not args.no_noise_removal

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
    texts, ids_all = generate_samples(model, mask_id, tokenizer, args)
    generation_seconds = time.perf_counter() - started
    (out / "samples.json").write_text(json.dumps(texts, ensure_ascii=False))

    residual_masks = sum(row.count(mask_id) for row in ids_all)

    throughput = throughput_cells(model, mask_id, args)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    quality, mauve_note = canonical_column(texts, args, eval_device)

    record = build_control_record(
        role="masked_discrete",
        family="mdlm",
        method="mdlm-owt",
        checkpoint=f"{args.repo}@{args.revision}",
        seed=args.seed,
        quality=quality,
        systems={
            "nfe": mdlm_nfe(steps_executed=args.steps, noise_removal=noise_removal),
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
            "steps_executed": args.steps,
            "residual_mask_tokens": residual_masks,
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
        },
    )
    write_jsonl([record], out / "frontier_record.jsonl")
    print(json.dumps(record, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
