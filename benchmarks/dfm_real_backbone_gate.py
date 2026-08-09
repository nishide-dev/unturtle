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
"""Real-backbone DFM eligibility gate (#65, step 1 of the promotion order).

`supports_dfm_generation` is a capability claim, not "the solver runs" —
so before any concrete family declares it, this gate measures whether the
tiny-control's two directional results survive on a real backbone
(Qwen3-0.6B through the #112-#115 loader):

1. **NFE is load-bearing**: sample quality improves with the step budget
   on an ordinary-DFM fine-tune;
2. **step-aware helps at low NFE**: an FS-DFM shortcut fine-tune
   (#105/#106 recipe) beats the step-matched ordinary control at 1-4 steps.

Protocol mirrors the tiny control (`tests/test_e2e_fs_dfm_shortcut.py`):
BOTH arms train the `StepAwareWrapper`-wrapped converted model from the
same init — arm A keeps ``h`` pinned in both phases (constant conditioning
carries no step information: ordinary DFM), arm B switches to the shortcut
objective (RK-4 teacher over EMA weights, eq. 4.5 blend) in phase 2.
Step-matched on purpose; the shortcut branch's RK-4 teacher makes it
~1.64x FLOPs per step (#106), so a win here is per-step, and the
FLOP-matched question belongs to the full-scale reproduction (step 3).

Quality judge: the ORIGINAL frozen Qwen3-0.6B AR model's mean per-token
NLL over unconditional samples — no external dependency, direction-readable.
Every sample flows through the PUBLIC ``generate(algorithm="dfm")`` path on
a gate-local mixin adoption (`DFMGateModel`); the family itself gains no
capability flag here.

Recorded limitations: fixed-length rows only (`StepAwareWrapper` carries no
attention_mask); the Unsloth `FastDiffusionModel` load path is out of scope
(tokenizer-side mask resolution) — the gate rides the #112 convert path.

Usage:
    uv run python benchmarks/dfm_real_backbone_gate.py --smoke
    uv run python benchmarks/dfm_real_backbone_gate.py

Frozen verdict (2026-08-10, RTX 6000 Ada, 4 runs; raw JSON archived under
dev/local/): **the directional gate is NOT decidable at this budget, and the
run measured why.**

- Judge NLL alone is gamed by degeneracy — the paper's own GenPPL caveat,
  now measured here: the best NLL points are repetition (shortcut 1-step
  NLL 3.02 at distinct 0.22 / entropy 2.07; ordinary 32-step NLL 3.08 at
  distinct 0.14 / entropy 1.86), while the healthiest samples (entropy ~5,
  distinct ~0.55) score the WORST NLL (~7.0).  At this budget NLL
  anti-correlates with diversity, so neither "NFE is load-bearing" nor
  "step-aware helps at low NFE" is readable from it.
- Seed instability confirms it: seed 0 (1500/600) showed ordinary
  7.52->3.08 over NFE and a shortcut 1-step "win"; seed 1 inverted both.
  Every low-NLL point the sweeps produced was a low-entropy point.
- Teacher-quality dependence: with an undertrained base (600/300) the
  shortcut arm degraded at every NFE; longer pretraining (1500/600) moved
  it — the distillation is gated by base quality, as in the tiny control.

What DID pass, frozen as infra facts: the full public path runs on the
real backbone end-to-end (convert -> StepAwareWrapper -> shortcut recipe ->
`generate(algorithm="dfm")`), bf16 pins hold at every stage (no #112-style
widening), the LoRA path runs (base bf16; peft keeps adapters fp32 by its
own default), and per-NFE sampling cost is measured (0.2s/2 to 6.5s/32 at
batch 64 x seq 128).

Consequence for the #65 promotion order: capability promotion (step 2)
stays blocked; the decidable version of this gate needs the step-3 budget
and a joint quality-diversity metric (NLL at matched entropy, or MAUVE) —
this script now records both guards for that run.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import UTC, datetime, timezone
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

MASK_ID = 151669  # the reference's <|MASK|> padded-vocab slot (#115)
TAU = 2.0**-5
GRID = [2.0**k for k in range(-6, 1)]
H_PRETRAIN = 2.0**-6


class DFMGateModel(DiscreteFlowGenerationMixin, TinyA2DQwen3LMHeadModel):
    """Gate-local opt-in: the family itself declares nothing (#120)."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--phase1-steps", type=int, default=600)
    parser.add_argument("--phase2-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--judge-samples", type=int, default=64)
    parser.add_argument("--nfe-grid", type=int, nargs="+", default=[1, 2, 4, 8, 32])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.phase1_steps = 20
        args.phase2_steps = 10
        args.train_rows = 128
        args.judge_samples = 8
        args.nfe_grid = [1, 4]
    return args


def gsm8k_rows(tokenizer, rows, seq_len, seed):
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.shuffle(seed=seed)
    out = []
    for example in dataset:
        ids = tokenizer(
            f"Question: {example['question']}\nAnswer: {example['answer']}",
            add_special_tokens=False,
        )["input_ids"]
        if len(ids) >= seq_len:
            out.append(ids[:seq_len])
        if len(out) >= rows:
            break
    return torch.tensor(out)


def assert_bf16(module, label):
    dtypes = {p.dtype for p in module.parameters()}
    assert dtypes == {torch.bfloat16}, f"{label} silently widened: {dtypes}"


def build_gate_model(args):
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


def train_phase(wrapper, corpus, *, steps, lr, shortcut, batch_size, seed, device):
    """One training phase; ``shortcut=False`` keeps h pinned (ordinary DFM)."""
    # Two generators, one per device (the dfm_structured_baseline precedent):
    # data picks and process noising draw on CPU; the RK-4 teacher's jump
    # draws follow x_t's device and need a device-local generator.
    generator = torch.Generator().manual_seed(seed)
    device_generator = (
        torch.Generator(device=device).manual_seed(seed)
        if str(device).startswith("cuda")
        else generator
    )
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
        picks = torch.randint(0, corpus.shape[0], (batch_size,), generator=generator)
        clean = corpus[picks].to(device)
        out = process({"input_ids": clean.cpu()}, generator=generator)
        x_t = out.model_inputs["input_ids"].to(device)
        timesteps = out.objective_inputs["timesteps"].to(device)

        if not shortcut:
            loss = discrete_flow_matching_loss(
                wrapper(x_t, timesteps, H_PRETRAIN),
                clean,
                x_t,
                timesteps,
                scheduler=scheduler,
            )
        else:
            h = GRID[int(torch.randint(0, len(GRID), (1,), generator=generator))]
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
                    generator=device_generator,
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
    return losses


@torch.no_grad()
def judge_nll(judge, samples):
    logits = judge(input_ids=samples).logits[:, :-1].float()
    targets = samples[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean"
    )
    return float(nll)


def quality_curve(model, wrapper, judge, args, *, label):
    """Judge NLL per NFE, every sample through the PUBLIC dfm path."""
    model.eval()
    wrapper.eval()
    model.dfm_denoiser = lambda x_t, t, h: wrapper(x_t, t, float(h))
    curve = {}
    for steps in args.nfe_grid:
        start = time.perf_counter()
        # The solver draws on x_t's device, so the generator must live there
        # too (documented contract; a CPU generator raises on a CUDA model).
        samples = model.generate(
            algorithm="dfm",
            batch_size=args.judge_samples,
            steps=steps,
            seq_len=args.seq_len,
            generator=torch.Generator(device=args.device).manual_seed(1234),
        )
        elapsed = time.perf_counter() - start
        # Degeneracy guards (the DiLaDiff paper's own GenPPL caveat):
        # repetitive text scores LOW judge NLL, so the NLL is only readable
        # next to a diversity measure.
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
        curve[steps] = {
            "judge_nll": judge_nll(judge, samples),
            "distinct_fraction": distinct,
            "unigram_entropy": entropy,
            "sample_seconds": elapsed,
        }
        print(
            f"  {label} steps={steps}: judge NLL {curve[steps]['judge_nll']:.3f} "
            f"distinct {distinct:.2f} entropy {entropy:.2f} ({elapsed:.1f}s)",
            flush=True,
        )
    assert_bf16(wrapper, f"{label} after generation")
    return curve


def lora_path_smoke(args):
    """The #65 gate's path pin: LoRA-wrapped convert path runs and stays bf16."""
    from peft import LoraConfig, get_peft_model

    converted = load_tiny_a2d_from_ar(
        args.model, mask_token_id=MASK_ID, torch_dtype=torch.bfloat16
    )
    model = DFMGateModel(converted.config).to(torch.bfloat16)
    model.load_state_dict(converted.state_dict(), strict=True)
    del converted
    model = model.to(args.device)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=8,
            lora_dropout=0.0,
        ),
    )
    base = peft_model.get_base_model()
    ids = base.generate(
        algorithm="dfm",
        batch_size=2,
        steps=2,
        seq_len=32,
        generator=torch.Generator(device=args.device).manual_seed(3),
    )
    dtypes = {p.dtype for p in peft_model.parameters()}
    print(f"LoRA path smoke: ids {tuple(ids.shape)}, dtypes {dtypes}", flush=True)
    del peft_model, model
    torch.cuda.empty_cache()
    return sorted(str(d) for d in dtypes)


def main() -> None:
    args = parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    corpus = gsm8k_rows(tokenizer, args.train_rows, args.seq_len, seed=args.seed)
    print(f"corpus rows: {corpus.shape[0]} x {corpus.shape[1]}", flush=True)

    judge = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        .to(args.device)
        .eval()
    )

    lora_dtypes = lora_path_smoke(args)

    # --- Arm A: ordinary DFM (h pinned in both phases).
    print("== arm A: ordinary DFM", flush=True)
    model_a, wrapper_a = build_gate_model(args)
    train_phase(
        wrapper_a,
        corpus,
        steps=args.phase1_steps,
        lr=args.lr,
        shortcut=False,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    train_phase(
        wrapper_a,
        corpus,
        steps=args.phase2_steps,
        lr=args.lr,
        shortcut=False,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        device=args.device,
    )
    curve_a = quality_curve(model_a, wrapper_a, judge, args, label="ordinary")
    state_a = {
        name: parameter.detach().clone()
        for name, parameter in wrapper_a.named_parameters()
    }
    del model_a, wrapper_a
    torch.cuda.empty_cache()

    # --- Arm B: same phase 1 (recomputed with the same seed — identical
    # trajectory), shortcut objective in phase 2.
    print("== arm B: FS-DFM shortcut", flush=True)
    model_b, wrapper_b = build_gate_model(args)
    train_phase(
        wrapper_b,
        corpus,
        steps=args.phase1_steps,
        lr=args.lr,
        shortcut=False,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    train_phase(
        wrapper_b,
        corpus,
        steps=args.phase2_steps,
        lr=args.lr,
        shortcut=True,
        batch_size=args.batch_size,
        seed=args.seed + 1,
        device=args.device,
    )
    curve_b = quality_curve(model_b, wrapper_b, judge, args, label="shortcut")

    payload = {
        "config": {
            **vars(args),
            "mask_token_id": MASK_ID,
            "tau": TAU,
            "grid": GRID,
            "h_pretrain": H_PRETRAIN,
            "protocol": (
                "both arms wrapped from the same converted init; arm A pins h "
                "(ordinary DFM), arm B runs the #105/#106 shortcut recipe in "
                "phase 2; step-matched (shortcut ~1.64x FLOPs/step, #106); "
                "judge = frozen original AR model mean NLL; all samples via "
                "public generate(algorithm='dfm')"
            ),
            "lora_smoke_dtypes": lora_dtypes,
        },
        "ordinary": curve_a,
        "shortcut": curve_b,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"dfm_real_backbone_gate_{stamp}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nresults -> {out}", flush=True)
    print("\nsteps  ordinary  shortcut")
    for steps in args.nfe_grid:
        print(
            f"{steps:5d}  {curve_a[steps]['judge_nll']:8.3f}  "
            f"{curve_b[steps]['judge_nll']:8.3f}"
        )
    del state_a


if __name__ == "__main__":
    main()
