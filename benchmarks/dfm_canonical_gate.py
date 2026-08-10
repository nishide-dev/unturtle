#!/usr/bin/env python3
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
"""Canonical-scale DFM eligibility gate (#65, step 3 — pre-registered).

The decision surface was frozen on the issue BEFORE any run:

- **Gate A (NFE is load-bearing)**: on the ordinary arm,
  ``MAUVE(32) - MAUVE(1) > 0.05`` on EVERY seed.
- **Gate B (step-aware helps at low NFE)**: the shortcut arm's MAUVE at
  S in {2, 4} beats BOTH the step-matched and the FLOP-matched ordinary
  controls on EVERY seed (S=1 reported, not gated — the tiny control's
  exclusion carried over).
- **Undecidable**: a gate whose relevant points are >25% collapsed
  (``unique_rows < 0.5`` or pooled entropy < 3.0) freezes as undecidable.
- A missing/unstable direction freezes as the result — no rerunning until
  positive.

Design, per the frozen protocol and the #121 lessons:

- MAUVE is PRIMARY (repetition is not rewarded); judge NLL is auxiliary and
  only comparable at matched diversity (pooled entropy within 1.0 nat).
  MAUVE features: gpt2 (base) — the paper uses gpt2-large; recorded
  deviation (not cached here).
- Three branches fork from ONE phase-1 checkpoint per seed, so arm inits
  are hermetic BY CONSTRUCTION (#121's fuse confound cannot recur), and
  every arm draws from its own derived generators (``seed*1000 + arm_id``,
  CPU/device split included) — complete RNG separation.
- BOTH controls: step-matched (same phase-2 steps) and FLOP-matched
  (phase-2 steps scaled by the shortcut's measured per-step cost ratio,
  computed from the grid/tau composition below — the #106 lesson).
- Data: GSM8K packed to fixed 256-token rows (the paper's own
  "packed to length L" protocol; #121's truncation filter yields too few
  long rows), 512 held-out rows as the MAUVE reference.
- Collapse guards on every point: per-row distinct fraction, corpus-pooled
  unigram entropy, unique-rows fraction.
- Every sample flows through the PUBLIC ``generate(algorithm="dfm")`` on a
  gate-local mixin adoption; the family declares nothing.

Usage (one seed per GPU; run seeds 0, 1, 2):
    uv run python benchmarks/dfm_canonical_gate.py --seed 0 --smoke
    uv run python benchmarks/dfm_canonical_gate.py --seed 0

Frozen verdict (2026-08-10, 3 seeds on RTX 6000 Ada, defaults above; raw
JSONs archived under dev/local/): **both gates NOT passed — capability
promotion stays blocked, per the pre-registered surface.**

    gate A (NFE load-bearing):    seed 0 undecidable (50% collapsed),
                                  seed 1 FAIL, seed 2 undecidable (100%)
    gate B (step-aware low-NFE):  seed 0 undecidable, seed 1 FAIL,
                                  seed 2 undecidable

MAUVE sat at 0.02-0.15 everywhere against a 0.979 held-out ceiling
(sanity: random tokens 0.092, repetitive 0.025): at this budget the
generated distribution is far from the reference regardless of arm or NFE,
with widespread collapse flags.  The one clean seed (1) DID show the NFE
direction on the ordinary arm — MAUVE(32) - MAUVE(1) = 0.112 - 0.068 =
+0.044 — but below the pre-frozen 0.05 margin, so it records as FAIL;
the threshold was frozen before the run and is not moved after it.

Reading: a 0.6B AR-initialized conversion fine-tuned for ~5k steps on a
7.5k-example corpus does not reach a distribution where the quality-vs-NFE
claim is testable, and neither directional result from the tiny control is
demonstrable here.  Consequence for #65: `supports_dfm_generation` stays
unpromoted; the public `dfm` family remains an explicit research opt-in
path, which is exactly the honest posture #120 built.  A future rerun that
wants decidability needs a different regime (longer training on a larger
corpus, per the FS-DFM paper's own scale), not another pass at this one.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
from unturtle.diffusion.fs_dfm import (
    StepAwareWrapper,
    clip_step_to_path,
    few_step_distillation_loss,
    rk_teacher_logits,
)
from unturtle.models.conversion.a2d.tiny_a2d import load_tiny_a2d_from_ar
from unturtle.models.conversion.a2d.tiny_a2d.modeling_qwen3 import (
    TinyA2DQwen3LMHeadModel,
)
from unturtle.models.generation.dfm_mixin import DiscreteFlowGenerationMixin
from unturtle.processes.discrete_flow import DiscreteFlowProcess, LinearKappa

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

MASK_ID = 151669
TAU = 2.0**-5
GRID = [2.0**k for k in range(-6, 1)]
H_PRETRAIN = 2.0**-6

# Per-step cost ratio of the shortcut objective vs a plain DFM step, in
# forward-equivalents (backward ~ 2 forwards): a plain step is fwd+bwd = 3;
# a distillation step adds 4 no-grad teacher forwards = 7.  The grid puts
# 1/7 of draws below tau (plain branch) and 6/7 in distillation.
_P_DISTILL = sum(1 for h in GRID if h >= TAU) / len(GRID)
FLOP_RATIO = ((1 - _P_DISTILL) * 3 + _P_DISTILL * 7) / 3


class DFMGateModel(DiscreteFlowGenerationMixin, TinyA2DQwen3LMHeadModel):
    """Gate-local opt-in: the family itself declares nothing (#120)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase1-steps", type=int, default=4000)
    parser.add_argument("--phase2-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--reference-rows", type=int, default=512)
    parser.add_argument("--nfe-grid", type=int, nargs="+", default=[1, 2, 4, 8, 32])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scratch", default="/tmp/dfm_canonical_gate")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.phase1_steps = 30
        args.phase2_steps = 10
        args.eval_samples = 16
        args.reference_rows = 64
        args.nfe_grid = [1, 4]
    return args


def packed_rows(tokenizer, seq_len, seed):
    """Pack the whole corpus into fixed-length rows (the paper's protocol)."""
    dataset = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=seed)
    eos = tokenizer.eos_token_id
    stream: list[int] = []
    for example in dataset:
        stream.extend(
            tokenizer(
                f"Question: {example['question']}\nAnswer: {example['answer']}",
                add_special_tokens=False,
            )["input_ids"]
        )
        stream.append(eos)
    rows = len(stream) // seq_len
    return torch.tensor(stream[: rows * seq_len]).view(rows, seq_len)


def assert_bf16(module, label):
    dtypes = {p.dtype for p in module.parameters()}
    assert dtypes == {torch.bfloat16}, f"{label} silently widened: {dtypes}"


def build_gate_model(args, init_seed):
    torch.manual_seed(init_seed)  # pins the fresh StepAwareWrapper.fuse head
    converted = load_tiny_a2d_from_ar(
        args.model, mask_token_id=MASK_ID, torch_dtype=torch.bfloat16
    )
    model = DFMGateModel(converted.config).to(torch.bfloat16)
    model.load_state_dict(converted.state_dict(), strict=True)
    del converted
    model = model.to(args.device)
    assert_bf16(model, "converted gate model")
    wrapper = StepAwareWrapper(model).to(device=args.device, dtype=torch.bfloat16)
    assert_bf16(wrapper, "step-aware wrapper")
    return model, wrapper


def arm_generators(seed, arm_id, device):
    """Complete per-arm RNG separation: derived, never shared."""
    base = seed * 1000 + arm_id
    cpu = torch.Generator().manual_seed(base)
    dev = (
        torch.Generator(device=device).manual_seed(base)
        if str(device).startswith("cuda")
        else cpu
    )
    return cpu, dev


def train_phase(wrapper, corpus, *, steps, lr, shortcut, args, cpu_gen, dev_gen):
    process = DiscreteFlowProcess(
        vocab_size=wrapper.base.config.vocab_size,
        mask_token_id=MASK_ID,
        source="mask",
    )
    scheduler = LinearKappa()
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=lr)
    ema = None
    if shortcut:
        ema = copy.deepcopy(wrapper).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)

    losses = []
    for _ in range(steps):
        picks = torch.randint(0, corpus.shape[0], (args.batch_size,), generator=cpu_gen)
        clean = corpus[picks].to(args.device)
        out = process({"input_ids": clean.cpu()}, generator=cpu_gen)
        x_t = out.model_inputs["input_ids"].to(args.device)
        timesteps = out.objective_inputs["timesteps"].to(args.device)

        if not shortcut:
            loss = discrete_flow_matching_loss(
                wrapper(x_t, timesteps, H_PRETRAIN),
                clean,
                x_t,
                timesteps,
                scheduler=scheduler,
            )
        else:
            h = GRID[int(torch.randint(0, len(GRID), (1,), generator=cpu_gen))]
            scaled_t, h_eff = clip_step_to_path(timesteps, h)
            if h < TAU:
                loss = discrete_flow_matching_loss(
                    wrapper(x_t, scaled_t, h),
                    clean,
                    x_t,
                    scaled_t,
                    scheduler=scheduler,
                    step_size=h_eff,
                )
            else:
                teacher = rk_teacher_logits(
                    lambda x, t, hh: ema(x, t, float(hh)),
                    x_t,
                    scaled_t,
                    h_eff,
                    generator=dev_gen,
                )
                loss = few_step_distillation_loss(wrapper(x_t, scaled_t, h), teacher)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if shortcut:
            with torch.no_grad():
                for ema_param, param in zip(
                    ema.parameters(), wrapper.parameters(), strict=True
                ):
                    ema_param.mul_(0.99).add_(param, alpha=0.01)
    ema = None  # release the teacher before the next branch loads
    torch.cuda.empty_cache()
    return losses


@torch.no_grad()
def judge_nll(judge, samples):
    logits = judge(input_ids=samples).logits[:, :-1].float()
    targets = samples[:, 1:]
    return float(
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
    )


def evaluate_arm(model, wrapper, judge, tokenizer, reference_texts, args, *, label):
    import mauve  # benchmark-only optional dependency (mauve-text)

    model.eval()
    wrapper.eval()
    model.dfm_denoiser = lambda x_t, t, h: wrapper(x_t, t, float(h))
    curve = {}
    for steps in args.nfe_grid:
        start = time.perf_counter()
        samples = model.generate(
            algorithm="dfm",
            batch_size=args.eval_samples,
            steps=steps,
            seq_len=args.seq_len,
            generator=torch.Generator(device=args.device).manual_seed(1234),
        )
        elapsed = time.perf_counter() - start

        distinct = (
            float(
                torch.tensor(
                    [row.unique().numel() for row in samples], dtype=torch.float32
                ).mean()
            )
            / samples.shape[1]
        )
        counts = torch.bincount(samples.reshape(-1)).float()
        frequencies = counts[counts > 0] / counts.sum()
        entropy = float(-(frequencies * frequencies.log()).sum())
        unique_rows = len({tuple(r.tolist()) for r in samples}) / samples.shape[0]
        collapsed = unique_rows < 0.5 or entropy < 3.0

        generated_texts = tokenizer.batch_decode(samples, skip_special_tokens=True)
        score = mauve.compute_mauve(
            p_text=reference_texts,
            q_text=generated_texts,
            featurize_model_name="gpt2",
            device_id=torch.device(args.device).index or 0,
            max_text_length=args.seq_len,
            verbose=False,
        ).mauve

        curve[steps] = {
            "mauve": float(score),
            "judge_nll": judge_nll(judge, samples),
            "distinct_fraction": distinct,
            "pooled_unigram_entropy": entropy,
            "unique_rows_fraction": unique_rows,
            "collapsed": collapsed,
            "sample_seconds": elapsed,
        }
        print(
            f"  {label} S={steps}: MAUVE {score:.3f} NLL "
            f"{curve[steps]['judge_nll']:.2f} entropy {entropy:.2f} "
            f"unique {unique_rows:.2f}{' [COLLAPSED]' if collapsed else ''}",
            flush=True,
        )
    assert_bf16(wrapper, f"{label} after generation")
    return curve


def main() -> None:
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    rows = packed_rows(tokenizer, args.seq_len, seed=args.seed)
    reference = rows[: args.reference_rows]
    corpus = rows[args.reference_rows :]
    reference_texts = tokenizer.batch_decode(reference, skip_special_tokens=True)
    print(
        f"seed {args.seed}: corpus {corpus.shape[0]} x {args.seq_len} "
        f"(reference {len(reference_texts)}); FLOP ratio {FLOP_RATIO:.3f}",
        flush=True,
    )

    judge = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )

    # Phase 1 once; every arm forks from this checkpoint (hermetic init).
    model, wrapper = build_gate_model(args, init_seed=args.seed)
    cpu_gen, dev_gen = arm_generators(args.seed, arm_id=0, device=args.device)
    phase1_losses = train_phase(
        wrapper,
        corpus,
        steps=args.phase1_steps,
        lr=args.lr,
        shortcut=False,
        args=args,
        cpu_gen=cpu_gen,
        dev_gen=dev_gen,
    )
    checkpoint = scratch / f"phase1_seed{args.seed}.pt"
    torch.save(wrapper.state_dict(), checkpoint)
    print(f"phase 1 done (loss {phase1_losses[-1]:.3f}) -> {checkpoint}", flush=True)

    arms = {
        "step_matched": dict(steps=args.phase2_steps, shortcut=False, arm_id=1),
        "flop_matched": dict(
            steps=round(args.phase2_steps * FLOP_RATIO), shortcut=False, arm_id=2
        ),
        "shortcut": dict(steps=args.phase2_steps, shortcut=True, arm_id=3),
    }
    curves = {}
    for name, spec in arms.items():
        print(f"== arm {name} ({spec['steps']} steps)", flush=True)
        wrapper.load_state_dict(torch.load(checkpoint, weights_only=True))
        cpu_gen, dev_gen = arm_generators(
            args.seed, arm_id=spec["arm_id"], device=args.device
        )
        train_phase(
            wrapper,
            corpus,
            steps=spec["steps"],
            lr=args.lr,
            shortcut=spec["shortcut"],
            args=args,
            cpu_gen=cpu_gen,
            dev_gen=dev_gen,
        )
        curves[name] = evaluate_arm(
            model, wrapper, judge, tokenizer, reference_texts, args, label=name
        )

    # --- Pre-registered gate evaluation (this seed's contribution).
    ordinary = curves["step_matched"]
    lo, hi = args.nfe_grid[0], args.nfe_grid[-1]
    gate_a_points = [ordinary[lo], ordinary[hi]]
    gate_a_collapsed = sum(p["collapsed"] for p in gate_a_points) / len(gate_a_points)
    gate_a = ordinary[hi]["mauve"] - ordinary[lo]["mauve"] > 0.05

    gate_b_grid = [s for s in (2, 4) if s in args.nfe_grid]
    gate_b_points = [curves[a][s] for a in curves for s in gate_b_grid]
    gate_b_collapsed = (
        sum(p["collapsed"] for p in gate_b_points) / len(gate_b_points)
        if gate_b_points
        else 1.0
    )
    gate_b = bool(gate_b_grid) and all(
        curves["shortcut"][s]["mauve"] > curves["step_matched"][s]["mauve"]
        and curves["shortcut"][s]["mauve"] > curves["flop_matched"][s]["mauve"]
        for s in gate_b_grid
    )

    verdict = {
        "gate_a_nfe_load_bearing": "undecidable"
        if gate_a_collapsed > 0.25
        else ("pass" if gate_a else "fail"),
        "gate_b_step_aware_low_nfe": "undecidable"
        if gate_b_collapsed > 0.25
        else ("pass" if gate_b else "fail"),
        "gate_a_collapsed_fraction": gate_a_collapsed,
        "gate_b_collapsed_fraction": gate_b_collapsed,
    }
    print(f"\nseed {args.seed} verdict: {verdict}", flush=True)

    payload = {
        "config": {
            **vars(args),
            "mask_token_id": MASK_ID,
            "tau": TAU,
            "grid": GRID,
            "flop_ratio": FLOP_RATIO,
            "mauve_features": "gpt2 (base; paper uses gpt2-large — deviation)",
        },
        "phase1_final_loss": phase1_losses[-1],
        "curves": curves,
        "verdict": verdict,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"dfm_canonical_gate_seed{args.seed}_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"results -> {out}", flush=True)


if __name__ == "__main__":
    main()
