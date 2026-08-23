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


"""#165: the Tier-A `ar_control` record — a COMPETENT autoregressive control.

Stage-0 freeze (issue #165):
  primary    openai-community/gpt2-medium @ 6dcaa7a952f72f9298047fd5137cd6e4f05f41da (MIT)
  secondary  openai-community/gpt2 @ 607a30d783dfa663caf39e06633721c8d4cfcd7e
  path       transformers GPT2LMHeadModel + .generate(), KV cache ON, SDPA,
             bf16, temperature 1.0, no top-k/top-p, max_new_tokens 1024
  NFE        1024 forwards (one per generated token) — EXECUTED, not requested
  samples    1000, seed 42, one cell-owned generator
  canonical  hf_causal_evaluator("gpt2-large", revision="main",
             max_length=1024), corpus-pooled GenPPL + paired unigram entropy

There is no "official evaluator" column for this control: GPT-2's paper
numbers are conditional-LM perplexities on held-out corpora, not generative
perplexity of unconditional samples, so quoting them beside a GenPPL cell
would be a category error.  `official_column` therefore records that
absence explicitly rather than inventing a number.

Confounds are LABELLED, never called matched: gpt2-medium is 355M params
trained on WebText, against 105M-class diffusion anchors trained on
OpenWebText.  This control bounds "what a competent AR of similar era does",
not "what an equal-compute AR does".
"""

import argparse
import json
import pathlib
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai-community/gpt2-medium")
    parser.add_argument(
        "--revision", default="6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--owt-heldout",
        default="data/owt/heldout.bin",
        help="#130 packed OWT held-out rows, for the MAUVE reference",
    )
    parser.add_argument(
        "--eval-device",
        default=None,
        help="device for gpt2-large scoring (defaults to --device)",
    )
    return parser.parse_args()


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from unturtle.eval.producers import ar_generation_config

    config = ar_generation_config(max_new_tokens=args.max_new_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation=config["attn_implementation"],
    )
    model.to(args.device).eval()
    # The competent path is a claim about the RUNNING model, so assert it on
    # the instance rather than trusting the config dict.
    if not model.config.use_cache:
        raise RuntimeError(
            f"{args.model} loaded with use_cache=False — the AR control must "
            "run the cached path (#165)"
        )
    return model, tokenizer, config


def generate_samples(model, tokenizer, config, args):
    """Unconditional samples from BOS, one cell-owned generator.

    Returns (texts, generated_token_counts).  The token counts are what NFE
    is derived from: an early EOS shortens the run and the record follows.
    """
    import torch

    from unturtle.eval.producers import global_rng_from, pinned_global_rng

    generator = torch.Generator().manual_seed(args.seed)
    bos = tokenizer.bos_token_id
    if bos is None:
        raise RuntimeError(f"{args.model} has no bos_token_id to start from")
    texts: list[str] = []
    lengths: list[int] = []
    while len(texts) < args.num_samples:
        batch = min(args.batch_size, args.num_samples - len(texts))
        prompt = torch.full((batch, 1), bos, dtype=torch.long, device=args.device)
        # `generate()` takes no `generator=` (it raises on the unused kwarg)
        # and samples from the global RNG, so the cell generator supplies
        # each batch's global seed instead — see `global_rng_from`.
        with torch.no_grad(), pinned_global_rng(global_rng_from(generator)):
            out = model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=config["do_sample"],
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                max_new_tokens=config["max_new_tokens"],
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Drop the BOS prompt column: only GENERATED tokens count.
        generated = out[:, 1:]
        for row in generated:
            ids = row.tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            lengths.append(len(ids))
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        print(f"  generated {len(texts)}/{args.num_samples}", flush=True)
    return texts[: args.num_samples], lengths[: args.num_samples]


def warm_generation(model, tokenizer, config, args):
    """One short throwaway generation, outside every timed region."""
    import torch

    from unturtle.eval.producers import pinned_global_rng

    prompt = torch.full(
        (1, 1), tokenizer.bos_token_id, dtype=torch.long, device=args.device
    )
    with torch.no_grad(), pinned_global_rng(0):
        model.generate(
            prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=True,
            temperature=config["temperature"],
            top_k=config["top_k"],
            top_p=config["top_p"],
            max_new_tokens=8,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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


def throughput_cells(model, tokenizer, config, args):
    import torch

    from unturtle.eval.producers import (
        global_rng_from,
        measure_control_throughput,
        pinned_global_rng,
    )

    bos = tokenizer.bos_token_id

    def run_batch(batch_size, generator):
        prompt = torch.full((batch_size, 1), bos, dtype=torch.long, device=args.device)
        with torch.no_grad(), pinned_global_rng(global_rng_from(generator)):
            model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=True,
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"],
                max_new_tokens=config["max_new_tokens"],
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def warmup():
        run_batch(1, torch.Generator().manual_seed(args.seed))

    return measure_control_throughput(run_batch, seed=args.seed, warmup=warmup)


def main():
    import torch

    from unturtle.eval.frontier import write_jsonl
    from unturtle.eval.producers import ar_nfe, build_control_record

    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_device = args.eval_device or args.device

    model, tokenizer, config = load_model(args)

    # Warm the CUDA/SDPA kernels BEFORE the timed sampling run: the first
    # generate() call on a fresh process paid ~160 s of compile/autotune in
    # the #165 smoke, which would otherwise be billed to
    # `generation_seconds` and read as AR slowness.
    warm_generation(model, tokenizer, config, args)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    texts, lengths = generate_samples(model, tokenizer, config, args)
    generation_seconds = time.perf_counter() - started
    (out / "samples.json").write_text(json.dumps(texts, ensure_ascii=False))

    throughput = throughput_cells(model, tokenizer, config, args)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    quality, mauve_note = canonical_column(texts, args, eval_device)

    executed_tokens = sum(lengths) / len(lengths)
    record = build_control_record(
        role="ar_control",
        family="ar",
        method="gpt2-lmhead",
        checkpoint=f"{args.model}@{args.revision}",
        seed=args.seed,
        quality=quality,
        systems={
            # NFE from the MEAN executed generated length, not from
            # --max-new-tokens: early EOS makes the request an upper bound.
            "nfe": ar_nfe(generated_tokens=round(executed_tokens)),
            "sequence_length": args.max_new_tokens,
            "solver": "ar-cached-sampling",
            "throughput": throughput,
            "peak_memory_bytes": peak,
        },
        decoding={
            "use_cache": True,
            "attn_implementation": config["attn_implementation"],
            "do_sample": True,
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "top_p": config["top_p"],
            "max_new_tokens": args.max_new_tokens,
            "dtype": "bfloat16",
        },
        confounds=[
            "scale: gpt2-medium is 355M params vs ~105M-class diffusion anchors",
            "training data: WebText (GPT-2) vs OpenWebText (anchors)",
            "tokenizer: gpt2 BPE 50257 vs T5 32100 (ELF) / gpt2 (FLM, MDLM)",
            "objective: this control bounds a competent AR of the same era, "
            "NOT an equal-compute AR",
        ],
        official={
            "status": "not_applicable",
            "reason": "GPT-2's published perplexities are conditional-LM "
            "scores on held-out corpora, not generative perplexity of "
            "unconditional samples; there is no official GenPPL column to "
            "quote here (#165 Stage-0 freeze)",
        },
        extra={
            "generation_seconds": generation_seconds,
            "generated_tokens_mean": executed_tokens,
            "generated_tokens_min": min(lengths),
            "generated_tokens_max": max(lengths),
            "nfe_note": "AR NFE counts token forwards; it is not comparable "
            "to a diffusion step count",
            "mauve_note": mauve_note,
        },
    )
    write_jsonl([record], out / "frontier_record.jsonl")
    print(json.dumps(record, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
