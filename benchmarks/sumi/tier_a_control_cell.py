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


"""#165: the Tier-A `uniform_state` record — Sumi, the external anchor.

Stage-0 freeze (issue #165) + the verbatim audit of the native sampler
(`generation_sumi.py` @ 0d20f7becf84340b8a8d71a8dda577a502a5c8dd), recorded
here BEFORE any measurement:

  checkpoint tohoku-nlp/sumi-7b @ 0d20f7becf84340b8a8d71a8dda577a502a5c8dd
             (Apache-2.0), `trust_remote_code=True` required — the family
             has no in-tree implementation
  role fit   GENUINE uniform state: the canvas starts as
             `randint(0, vocab_size)`, there is NO mask token, and the
             ancestral posterior is categorical on the one-hot simplex.
             This is exactly what #152 forbids DFM from standing in for.
  sampler    class defaults `sampler="ancestral"`, `schedule="linear"`,
             `min/max_log_snr = -9/+9`, `num_denoising_steps=128`,
             `temperature=1.0`.  The README's example uses 64 steps and
             temperature 0.7 — an EXAMPLE, not the default; this cell runs
             the class defaults and records the README variant separately
             if asked for.
  NFE        == num_denoising_steps.  One forward per step, no tail step
             (contrast MDLM, whose official config adds a noise-removal
             forward).
  RNG        the native `generate()` DOES take `generator=`, so the cell's
             generator is handed over directly — no global-RNG pinning
             needed here (the AR producer needs it because
             `transformers.generate()` rejects the kwarg).

Canvas decision (fixed by the #167 review, recorded before any quality
output): **the main decision cell is canvas 1024**, matching the #152
protocol context; **the model's native 2048 is a protocol-deviating
secondary curve point**.  The main table has to satisfy the protocol
context, and there is no reason to swap the main table to 2048.

The deviation is recorded, never hidden: Sumi is trained on a packed
fixed-length canvas and denoises the WHOLE canvas every step
(`canvas_length` default 2048, ceiling 4864 = `max_position_embeddings`),
while `max_new_tokens` is only the content budget before the anchored
EOS,BOS delimiter.  `uniform_state_compute_scope` records whether the
forwarded canvas matches the protocol context, so a 2048 secondary cell
carries `protocol_context_match: False` in its own record.

Confounds are LABELLED, and they are large: ~7B params, ~1.5T training
tokens, a different tokenizer (100,278) and a different corpus.  This is an
EXTERNAL SCALING ANCHOR for the uniform-state role — not an equal-compute
control, and not comparable head-to-head with the 105M-class anchors.
"""

import argparse
import json
import pathlib
import time

from unturtle.eval.producers import CANONICAL_EVALUATOR_REVISION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="tohoku-nlp/sumi-7b")
    parser.add_argument(
        "--revision", default="0d20f7becf84340b8a8d71a8dda577a502a5c8dd"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--canvas-length",
        type=int,
        default=1024,
        help="forwarded canvas; 1024 matches the #152 protocol context, the "
        "model's own default is 2048 (protocol-deviating)",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampler", default="ancestral")
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
    from transformers import AutoModelForMaskGeneration, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.repo, revision=args.revision, trust_remote_code=True
    )
    # `AutoModelForCausalLM` does NOT recognize SumiConfig — the repo's
    # auto_map registers `AutoModelForMaskGeneration` (architecture
    # `SumiForMaskGeneration`), matching the model card's own example.
    model = AutoModelForMaskGeneration.from_pretrained(
        args.repo,
        revision=args.revision,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(args.device).eval()
    # The role claim is structural, so verify it on the loaded object rather
    # than trusting the card: a uniform-state model has no mask token and
    # exposes the native diffusion generate().
    if getattr(model.config, "mask_token_id", None) is not None:
        raise RuntimeError(
            f"{args.repo} reports a mask_token_id — that is masked discrete "
            "diffusion, not the uniform_state role (#152)"
        )
    if not hasattr(model, "generate"):
        raise RuntimeError(f"{args.repo} exposes no generate()")
    return model, tokenizer


def sample_batch(
    model,
    batch_size,
    args,
    generator,
    step_counter=None,
    capture_trajectory=False,
    trajectory_out=None,
):
    """One batch through the NATIVE Sumi sampler.

    `generation_sumi.generate` accepts `generator=` and threads it into
    both the initial uniform draw and every ancestral `multinomial` — but
    those run on CUDA tensors, so a CPU generator raises `RuntimeError:
    Expected a 'cuda' device type for generator`.  The cell's generator is
    therefore DERIVED onto the model's device (one draw per batch, so the
    cell stream still advances exactly once).
    """
    import torch

    from unturtle.eval.producers import derive_device_generator

    generator = derive_device_generator(generator, device=args.device)
    bos = model.config.bos_token_id
    # The sampler's own progress_callback fires once per executed step, so
    # the recorded step count is OBSERVED rather than the request echoed
    # back (#165 review F3).
    observed = []
    trajectory = []

    def progress_callback(state):
        observed.append(int(state["step"]))
        # Committed-state snapshots, so revision is MEASURED rather than
        # inferred from the sampler's capability (#167 review 3).
        if capture_trajectory:
            trajectory.append(state["z"].detach().cpu().clone())

    prompt = torch.full((batch_size, 1), bos, dtype=torch.long, device=args.device)
    with torch.no_grad():
        out = model.generate(
            prompt,
            num_denoising_steps=args.steps,
            sampler=args.sampler,
            schedule="linear",
            min_log_snr=-9.0,
            max_log_snr=9.0,
            temperature=args.temperature,
            canvas_length=args.canvas_length,
            max_new_tokens=args.canvas_length - 3,
            generator=generator,
            progress_callback=progress_callback,
        )
    if step_counter is not None:
        step_counter.append(max(observed) if observed else 0)
    if trajectory_out is not None:
        trajectory_out.extend(trajectory)
    return out.sequences, out.canvas


def generate_samples(model, tokenizer, args):
    import torch

    generator = torch.Generator().manual_seed(args.seed)
    pad_id = model.config.pad_token_id
    if pad_id is None:
        pad_id = model.config.eos_token_id
    texts: list[str] = []
    lengths: list[int] = []
    content_ids: list[list[int]] = []
    canvas_ids: list[list[int]] = []
    step_counts: list[int] = []
    trajectory: list = []
    while len(texts) < args.num_samples:
        batch = min(args.batch_size, args.num_samples - len(texts))
        first_batch = not texts
        sequences, canvas = sample_batch(
            model,
            batch,
            args,
            generator,
            step_counter=step_counts,
            capture_trajectory=first_batch,
            trajectory_out=trajectory if first_batch else None,
        )
        for row, canvas_row in zip(sequences, canvas, strict=True):
            ids = row.tolist()
            # `_trim_at_eos` cuts each row at its first EOS and then RIGHT-PADS
            # the batch back into a rectangle, so len(row) is the batch's
            # longest row, not this sample's content length.  Strip the pad
            # filler to get the real length (#165 review F4).
            content = ids
            while content and content[-1] == pad_id:
                content = content[:-1]
            lengths.append(len(content))
            # Canonical guards need the CONTENT tokens; the full canvas is
            # kept separately for the secondary diagnostics (#167 review 3).
            content_ids.append(content)
            canvas_ids.append(canvas_row.tolist())
            texts.append(tokenizer.decode(ids, skip_special_tokens=True))
        print(f"  generated {len(texts)}/{args.num_samples}", flush=True)
    if len(set(step_counts)) != 1:
        raise RuntimeError(
            f"batches executed differing step counts {sorted(set(step_counts))} "
            "— the cell would have no single executed step count to record"
        )
    return (
        texts[: args.num_samples],
        lengths[: args.num_samples],
        content_ids[: args.num_samples],
        canvas_ids[: args.num_samples],
        step_counts[0],
        trajectory,
    )


def canonical_column(texts, content_ids, args, device):
    from transformers import AutoTokenizer

    from unturtle.eval.frontier import hf_causal_evaluator
    from unturtle.eval.producers import (
        canonical_evaluator_identity,
        canonical_quality_column,
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
    # Guards on CONTENT tokens: the canvas tail is denoised context nobody
    # reads, and its diversity would mask a collapsed decoded region (#167
    # review 3).
    sample_ids, pad_meta = stack_sample_ids(content_ids, pad_id=0)
    quality = canonical_quality_column(
        texts,
        evaluator=evaluator,
        evaluator_identity=identity,
        tokenize=gpt2_tokenizer.encode,
        sample_ids=sample_ids,
    )
    quality_scope = {
        "guard_input": "content tokens (canvas trimmed at the first EOS)",
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


def throughput_cells(model, args):
    import torch

    from unturtle.eval.producers import measure_control_throughput

    def run_batch(batch_size, generator):
        cell_steps: list[int] = []
        sample_batch(model, batch_size, args, generator, step_counter=cell_steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        executed = cell_steps[0] if cell_steps else 0
        return {
            "forwards_executed": executed,
            "steps_executed": executed,
            "sequence_length": args.canvas_length,
        }

    def warmup():
        run_batch(1, torch.Generator().manual_seed(args.seed))

    # A 7B model at batch 32 on a 1024 canvas may not fit; an OOM becomes a
    # typed cell rather than a missing row (#152).
    return measure_control_throughput(run_batch, seed=args.seed, warmup=warmup)


def main():
    import torch

    from unturtle.eval.frontier import write_jsonl
    from unturtle.eval.producers import (
        build_control_record,
        canvas_diagnostics,
        decision_preflight,
        revision_diagnostics,
        stack_sample_ids,
        uniform_state_compute_scope,
        uniform_state_nfe,
    )

    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_device = args.eval_device or args.device

    heldout = pathlib.Path(args.owt_heldout)
    claimed_role = decision_preflight(
        mode=args.mode,
        role="uniform_state",
        num_samples=args.num_samples,
        seed=args.seed,
        mauve_available=heldout.is_file()
        and (heldout.parent / f"{heldout.name}.json").exists(),
        evaluator_revision=args.evaluator_revision,
    )

    model, tokenizer = load_model(args)

    # Warm outside every timed region.
    warm_args = argparse.Namespace(**vars(args))
    warm_args.steps = 2
    warm_args.canvas_length = 128
    sample_batch(
        model, 1, warm_args, torch.Generator(device=args.device).manual_seed(0)
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    texts, lengths, content_ids, canvas_ids, observed_steps, trajectory = (
        generate_samples(model, tokenizer, args)
    )
    generation_seconds = time.perf_counter() - started
    (out / "samples.json").write_text(json.dumps(texts, ensure_ascii=False))

    throughput = throughput_cells(model, args)
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    quality, mauve_note, quality_scope = canonical_column(
        texts, content_ids, args, eval_device
    )
    canvas_tensor, _ = stack_sample_ids(canvas_ids, pad_id=0)

    scope = uniform_state_compute_scope(
        canvas_length=args.canvas_length,
        content_budget=args.canvas_length - 3,
        prompt_length=1,
    )
    record = build_control_record(
        role=claimed_role,
        family="uniform_diffusion",
        method="sumi",
        checkpoint=f"{args.repo}@{args.revision}",
        seed=args.seed,
        steps_requested=args.steps,
        steps_executed=observed_steps,
        quality=quality,
        systems={
            "nfe": uniform_state_nfe(steps_executed=observed_steps),
            "sequence_length": scope["sequence_length"],
            "solver": f"{args.sampler}-uniform-diffusion",
            "throughput": throughput,
            "peak_memory_bytes": peak,
        },
        decoding={
            "num_denoising_steps": args.steps,
            "sampler": args.sampler,
            "schedule": "linear",
            "min_log_snr": -9.0,
            "max_log_snr": 9.0,
            "temperature": args.temperature,
            "canvas_length": args.canvas_length,
            "max_new_tokens": args.canvas_length - 3,
            "trim_at_eos": True,
            "anchor_eosbos": True,
            "dtype": "bfloat16",
        },
        confounds=[
            "scale: ~7B params (36 layers, hidden 4096) vs ~105M-class "
            "anchors — an EXTERNAL SCALING ANCHOR, not an equal-compute "
            "control, and not comparable head-to-head",
            "training data: ~1.5T tokens from a different corpus, not OpenWebText",
            "tokenizer: 100,278-entry vocabulary vs gpt2 50257 / T5 32100",
            "canvas: the model denoises a packed fixed-length canvas; this "
            f"cell forwards {args.canvas_length} tokens (protocol context "
            f"match: {scope['protocol_context_match']})",
        ],
        official={
            "status": "not_measured_here",
            "reason": "Sumi's card reports its own benchmark suite, not "
            "OpenWebText generative perplexity; this cell reports the #152 "
            "canonical column only",
        },
        extra={
            "generation_seconds": generation_seconds,
            "steps_requested": args.steps,
            # OBSERVED, via the sampler's own progress_callback — not the
            # requested count re-labelled (#165 review F3).
            "steps_executed": observed_steps,
            "compute_scope": scope,
            "content_length_mean": sum(lengths) / len(lengths),
            "content_length_min": min(lengths),
            "content_length_max": max(lengths),
            "role_fit": "true uniform state — canvas initialized from "
            "randint(0, vocab_size), no mask token, ancestral posterior on "
            "the one-hot simplex (audited: generation_sumi.py)",
            "sampler_defaults_note": "class defaults used (128 steps, "
            "temperature 1.0); the model card's example (64 steps, "
            "temperature 0.7) is an example, not the default",
            "mauve_note": mauve_note,
            "quality_scope": quality_scope,
            "mode": args.mode,
            # The canvas is the model's real compute scope, so its
            # entropy/diversity ride as CANVAS-named diagnostics — never as
            # the canonical guards (#167 review 3).
            "canvas_diagnostics": canvas_diagnostics(
                canvas_tensor, content_widths=lengths
            ),
            "revision_diagnostics": revision_diagnostics(trajectory),
        },
    )
    write_jsonl([record], out / "frontier_record.jsonl")
    print(json.dumps(record, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
