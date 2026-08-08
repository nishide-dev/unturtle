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
Structured DFM baseline: the frozen quality-vs-NFE control for Phase B (#65).

Phase B (FS-DFM few-step / step-aware training) claims to recover low-NFE
quality.  Evaluating that claim needs a control where ordinary DFM genuinely
degrades as steps are reduced — the earlier position-independent task cannot
provide one, since a single denoising call solves it in principle and its
curve is flat.

The task here makes NFE load-bearing by construction.  Sequences follow
``x_i = (s + i) mod V`` for a random start ``s``: every position's marginal
is uniform, so a one-step sample — positions drawn independently from their
marginals — cannot beat chance (``1/V``) on the chain rule, however well the
model fits.  Recovering consistency requires committing tokens across rounds
so later denoiser calls condition on earlier commitments, which is exactly
what step count buys.

**Frozen control curve** (seed 0, CUDA, config below — this table is the
Phase B reference; a step-aware objective is measured by how far it lifts the
low-step entries toward the 64-step ceiling at matched model/data/compute):

    ======  ============  ============  =========
    steps   adjacent-ok   full-seq ok   mask-left
    ======  ============  ============  =========
         1         0.124         0.000      0.008
         2         0.443         0.000      0.000
         4         0.706         0.070      0.000
         8         0.851         0.316      0.000
        16         0.944         0.660      0.000
        64         0.978         0.855      0.000
    ======  ============  ============  =========

    Seed robustness (training seeds 0-7, fully seeded pipeline): 1-step
    0.125-0.130, 4-step 0.617-0.705, 64-step 0.953-0.977.  Scores are not
    bit-stable across CPU thread counts, and ``--cpu`` does **not** reproduce
    the CUDA table (e.g. 16-step 0.908 vs 0.944) -- the frozen reference is
    the CUDA seed-0 run above.

Config is pinned in code, not flags, so "regenerate the control" cannot
silently drift from what the numbers above describe.  The guarded regression
form of the same property lives in
``tests/test_e2e_discrete_flow_structured.py``.

Usage::

    .venv/bin/python benchmarks/dfm_structured_baseline.py [--seed N] [--cpu]
"""

from __future__ import annotations

import argparse

import torch

from unturtle.diffusion.dfm_loss import discrete_flow_matching_loss
from unturtle.models.conversion.a2d.tiny_a2d.modeling_llama import (
    TinyA2DLlamaConfig,
    TinyA2DLlamaLMHeadModel,
)
from unturtle.models.generation.dfm_solver import solve_discrete_flow
from unturtle.processes.discrete_flow import DiscreteFlowProcess, LinearKappa

# The frozen control config.  Changing any of these invalidates the table in
# the module docstring; regenerate and update both together.
DATA_VOCAB = 8
MASK_ID = DATA_VOCAB
LENGTH = 16
HIDDEN = 64
LAYERS = 2
HEADS = 4
TRAIN_STEPS = 1200
BATCH = 64
LR = 1e-3
SAMPLE_ROWS = 256
STEP_BUDGETS = (1, 2, 4, 8, 16, 64)


def _corpus(n: int, generator: torch.Generator) -> torch.Tensor:
    start = torch.randint(0, DATA_VOCAB, (n, 1), generator=generator)
    return (start + torch.arange(LENGTH)) % DATA_VOCAB


def train(seed: int, device: torch.device) -> TinyA2DLlamaLMHeadModel:
    config = TinyA2DLlamaConfig(
        vocab_size=DATA_VOCAB + 1,
        hidden_size=HIDDEN,
        intermediate_size=HIDDEN * 2,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        max_position_embeddings=LENGTH,
    )
    torch.manual_seed(seed)
    model = TinyA2DLlamaLMHeadModel(config).to(device).train()
    process = DiscreteFlowProcess(
        vocab_size=DATA_VOCAB + 1, mask_token_id=MASK_ID, source="mask"
    )
    scheduler = LinearKappa()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # Two generators: `_corpus` draws on CPU (randint with a CPU generator),
    # while the process noises device-side and torch requires the generator's
    # device to match.  Both are seeded, so the run is self-contained -- no
    # draw rides on the global RNG, whose state any unrelated import or model
    # init could shift (review finding on the first draft: one extra global
    # rand moved the 64-step score by 0.01).
    generator = torch.Generator().manual_seed(seed)
    noise_generator = torch.Generator(device=device).manual_seed(seed)

    for step in range(TRAIN_STEPS):
        clean = _corpus(BATCH, generator).to(device)
        out = process({"input_ids": clean}, generator=noise_generator)
        logits = model(input_ids=out.model_inputs["input_ids"]).logits
        loss = discrete_flow_matching_loss(
            logits,
            clean,
            out.model_inputs["input_ids"],
            out.objective_inputs["timesteps"],
            scheduler=scheduler,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 400 == 0:
            print(f"  train step {step:5d}  loss {float(loss.detach()):.4f}")

    return model.eval()


def measure(model, seed: int, device: torch.device) -> None:
    def denoise(x_t, t, h):
        with torch.no_grad():
            return model(input_ids=x_t).logits

    print(f"\n{'steps':>6} {'adjacent-ok':>12} {'full-seq ok':>12} {'mask-left':>10}")
    for steps in STEP_BUDGETS:
        generator = torch.Generator(device=device).manual_seed(seed)
        x_0 = torch.full((SAMPLE_ROWS, LENGTH), MASK_ID, dtype=torch.long)
        out = solve_discrete_flow(
            denoise, x_0.to(device), steps=steps, generator=generator
        )
        ok = (out[:, 1:] - out[:, :-1]) % DATA_VOCAB == 1
        print(
            f"{steps:6d} {float(ok.float().mean()):12.3f} "
            f"{float(ok.all(dim=1).float().mean()):12.3f} "
            f"{float((out == MASK_ID).float().mean()):10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"device={device} seed={args.seed}")
    model = train(args.seed, device)
    measure(model, args.seed, device)


if __name__ == "__main__":
    main()
