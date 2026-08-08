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

"""
FS-DFM shortcut fine-tuning vs a matched-compute DFM control (#65 Phase B).

The Phase B claim — step-aware training recovers few-step quality — evaluated
against the frozen structured control of ``dfm_structured_baseline.py``.
Both arms share the model, the seed, the data stream shape, and the fully
seeded pipeline (no draw rides on the global RNG).

**Compute accounting, stated honestly.**  "Same 2700 optimizer steps" is NOT
the same compute: with ``TAU = 2^-5`` on this grid, 6 of 7 h values take the
distillation branch (~86% of fine-tune steps), and each of those runs the
RK-4 teacher — 4 extra no-grad forwards.  Counting backward as 2 forwards,
the shortcut arm spends ~1.64x the control's FLOPs at equal steps.  Two
control arms are therefore frozen below: **step-matched** (2700 steps) and
**FLOP-matched** (4414 steps ~= 2700 x 1.64).  The shortcut wins every
paired comparison against BOTH — plain DFM is near-saturated by 2700 steps
on this task, so the extra 64% of FLOPs buys the control almost nothing.

**Frozen result** (CUDA, seeds 0-3, adjacent-chain consistency; the
step-matched arms regenerate with this script, the FLOP-matched control by
raising the control's step count to 4414):

    ====  ========  =======  =======  =======
    seed  arm        2-step   4-step   8-step
    ====  ========  =======  =======  =======
       0  control     0.436    0.717    0.864
       0  shortcut    0.475    0.778    0.921
       1  control     0.419    0.669    0.827
       1  shortcut    0.477    0.769    0.888
       2  control     0.426    0.687    0.848
       2  shortcut    0.488    0.746    0.882
       3  control     0.406    0.642    0.804
       3  shortcut    0.530    0.763    0.868
    ====  ========  =======  =======  =======

    Paired deltas vs the STEP-matched control (2700 steps), all 12 positive:
      2-step  +0.039 +0.058 +0.062 +0.124   mean +0.071
      4-step  +0.060 +0.100 +0.059 +0.122   mean +0.085
      8-step  +0.058 +0.062 +0.034 +0.065   mean +0.055

    Paired deltas vs the FLOP-matched control (4414 steps), all 12 positive:
      2-step  +0.029 +0.051 +0.039 +0.135   mean +0.064
      4-step  +0.061 +0.084 +0.038 +0.131   mean +0.079
      8-step  +0.055 +0.053 +0.028 +0.076   mean +0.053

    FLOP-matched control raw (seeds 0-3):
      2-step  0.446 0.426 0.449 0.395
      4-step  0.717 0.685 0.708 0.632
      8-step  0.866 0.835 0.854 0.792

**1-step is not in the table on purpose.**  At one call the terminal draw
samples positions independently from their marginals, which this task makes
uniform by construction — ~1/V adjacent consistency is the *correct* 1-step
ceiling, and beating it requires distorting the marginals.  The measurable
few-step win on this control is at 2-8 steps.

Config pinned in code; the self-paired CPU-sized regression form lives in
``tests/test_e2e_fs_dfm_shortcut.py``.

Usage::

    .venv/bin/python benchmarks/fs_dfm_shortcut.py [--seeds N] [--cpu]
"""

from __future__ import annotations

import argparse
import copy

import torch

from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
from unturtle.diffusion.fs_dfm import (
    StepAwareWrapper,
    clip_step_to_path,
    few_step_distillation_loss,
    rk_teacher_logits,
)
from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
)
from unturtle.models.generation.dfm_solver import solve_discrete_flow
from unturtle.processes.discrete_flow import DiscreteFlowProcess, LinearKappa

DATA_VOCAB = 8
MASK_ID = DATA_VOCAB
LENGTH = 16
HIDDEN = 64
PRETRAIN_STEPS = 1200
FINETUNE_STEPS = 1500
BATCH = 64
H_PRETRAIN = 2.0**-6
TAU = 2.0**-5
GRID = [2.0**k for k in range(-6, 1)]
EMA_BETA = 0.99
STEP_BUDGETS = (2, 4, 8)


def _config():
    return TinyA2DLlamaConfig(
        vocab_size=DATA_VOCAB + 1,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=LENGTH,
    )


def _corpus(n, generator):
    start = torch.randint(0, DATA_VOCAB, (n, 1), generator=generator)
    return (start + torch.arange(LENGTH)) % DATA_VOCAB


def _adjacent(samples):
    return float(((samples[:, 1:] - samples[:, :-1]) % DATA_VOCAB == 1).float().mean())


def _measure(denoise, seed, device):
    results = {}
    for steps in STEP_BUDGETS:
        generator = torch.Generator(device=device).manual_seed(seed)
        x_0 = torch.full((256, LENGTH), MASK_ID, dtype=torch.long, device=device)
        out = solve_discrete_flow(denoise, x_0, steps=steps, generator=generator)
        results[steps] = _adjacent(out)
    return results


def run_control(seed, device):
    """Plain DFM, time-agnostic (the #104 recipe), same total compute."""
    torch.manual_seed(seed)
    model = TinyA2DLlamaLMHeadModel(_config()).to(device).train()
    process = DiscreteFlowProcess(
        vocab_size=DATA_VOCAB + 1, mask_token_id=MASK_ID, source="mask"
    )
    scheduler = LinearKappa()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data_gen = torch.Generator().manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed)

    for _ in range(PRETRAIN_STEPS + FINETUNE_STEPS):
        clean = _corpus(BATCH, data_gen).to(device)
        out = process({"input_ids": clean}, generator=noise_gen)
        x_t = out.model_inputs["input_ids"]
        timesteps = out.objective_inputs["timesteps"]
        loss = discrete_flow_matching_loss(
            model(input_ids=x_t).logits, clean, x_t, timesteps, scheduler=scheduler
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    def denoise(x_t, t, h):
        with torch.no_grad():
            return model(input_ids=x_t).logits

    return _measure(denoise, seed, device)


def run_shortcut(seed, device):
    torch.manual_seed(seed)
    model = StepAwareWrapper(TinyA2DLlamaLMHeadModel(_config()).to(device)).to(device)
    model.train()
    process = DiscreteFlowProcess(
        vocab_size=DATA_VOCAB + 1, mask_token_id=MASK_ID, source="mask"
    )
    scheduler = LinearKappa()
    data_gen = torch.Generator().manual_seed(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(PRETRAIN_STEPS):
        clean = _corpus(BATCH, data_gen).to(device)
        out = process({"input_ids": clean}, generator=noise_gen)
        x_t = out.model_inputs["input_ids"]
        timesteps = out.objective_inputs["timesteps"]
        loss = discrete_flow_matching_loss(
            model(x_t, timesteps, H_PRETRAIN),
            clean,
            x_t,
            timesteps,
            scheduler=scheduler,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    for _ in range(FINETUNE_STEPS):
        clean = _corpus(BATCH, data_gen).to(device)
        out = process({"input_ids": clean}, generator=noise_gen)
        x_t = out.model_inputs["input_ids"]
        timesteps = out.objective_inputs["timesteps"]
        h = GRID[int(torch.randint(0, len(GRID), (1,), generator=data_gen))]
        scaled_t, h_eff = clip_step_to_path(timesteps, h)

        if h < TAU:
            loss = discrete_flow_matching_loss(
                model(x_t, scaled_t, h),
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
                generator=noise_gen,
            )
            loss = few_step_distillation_loss(model(x_t, scaled_t, h), teacher)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for ema_param, param in zip(
                ema.parameters(), model.parameters(), strict=True
            ):
                ema_param.mul_(EMA_BETA).add_(param, alpha=1 - EMA_BETA)

    model.eval()

    def denoise(x_t, t, h):
        with torch.no_grad():
            return model(x_t, t, h)

    return _measure(denoise, seed, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"device={device}")

    header = " ".join(f"{s}-step".rjust(8) for s in STEP_BUDGETS)
    print(f"{'seed':>4} {'arm':>10} {header}")
    deltas = {s: [] for s in STEP_BUDGETS}
    for seed in range(args.seeds):
        control = run_control(seed, device)
        shortcut = run_shortcut(seed, device)
        for name, res in (("control", control), ("shortcut", shortcut)):
            row = " ".join(f"{res[s]:8.3f}" for s in STEP_BUDGETS)
            print(f"{seed:4d} {name:>10} {row}")
        for s in STEP_BUDGETS:
            deltas[s].append(shortcut[s] - control[s])

    print("\npaired deltas (shortcut - control):")
    for s, values in deltas.items():
        joined = " ".join(f"{v:+.3f}" for v in values)
        print(f"  {s}-step  {joined}   mean {sum(values) / len(values):+.3f}")


if __name__ == "__main__":
    main()
