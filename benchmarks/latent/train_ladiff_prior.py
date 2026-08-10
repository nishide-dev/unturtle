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

"""LaDiff latent-prior training run (#130) — protocol frozen on the issue
BEFORE runs (issuecomment-5246782747).

Prior seed K pairs with AE seed K throughout (encoder, statistics, decoder).
The AE is FROZEN; the prior lives entirely in standardized latent space
(the AE's latent_standardizer supplies mu_z/sigma_z, its eval forward IS the
standardization). Budget: 2.0B token presentations = 30,518 steps of 64x1024
effective rows. LR 2e-4 cosine, warmup 200 (paper 1000/150k scaled).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

MICRO_BATCH = 32
GRAD_ACCUM = 2
ROWS_PER_STEP = MICRO_BATCH * GRAD_ACCUM
TOTAL_STEPS = math.ceil(2e9 / (ROWS_PER_STEP * 1024))  # 30,518
LR = 2e-4
WARMUP = 200
GRAD_CLIP = 1.0
EVAL_EVERY = 2000
EVAL_ROWS = slice(768, 1024)  # heldout smoke rows (Gate A used [0:512])
EVAL_GEN_SEED = 4243


def load_frozen_autoencoder(ae_checkpoint: Path, device: str):
    spec = importlib.util.spec_from_file_location(
        "train_ladiff_ae", Path(__file__).parent / "train_ladiff_ae.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    autoencoder = mod.build_autoencoder(device)
    state = torch.load(ae_checkpoint, map_location=device, weights_only=True)
    autoencoder.load_state_dict(state)  # fingerprint enforces trunk identity
    autoencoder.eval()
    autoencoder.requires_grad_(False)
    return autoencoder


def build_prior(device: str):
    from unturtle.models.latent import LaDiffPriorConfig, LatentPriorDenoiser

    return LatentPriorDenoiser(LaDiffPriorConfig()).to(device)


@torch.no_grad()
def standardized_latents(autoencoder, ids: torch.Tensor) -> torch.Tensor:
    """Frozen encode + standardization (the AE's eval forward normalizes
    with its frozen running statistics)."""
    features = autoencoder.feature_standardizer(autoencoder.features(ids))
    return autoencoder.latent_standardizer(autoencoder.encoder(features))


def lr_lambda(step: int) -> float:
    if step < WARMUP:
        return (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument(
        "--ae-dir", type=Path, default=REPO_ROOT / "dev/local/ladiff_ae"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "dev/local/ladiff_prior"
    )
    parser.add_argument(
        "--train-corpus", type=Path, default=REPO_ROOT / "dev/local/owt/train_3b_1024"
    )
    parser.add_argument(
        "--heldout-corpus",
        type=Path,
        default=REPO_ROOT / "dev/local/owt/heldout_1024",
    )
    args = parser.parse_args()

    from unturtle.models.latent import ladiff_prior_loss
    from unturtle.utils.packed_text import read_packed

    out_dir = args.out_dir / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    torch.manual_seed(args.seed)
    autoencoder = load_frozen_autoencoder(
        args.ae_dir / f"seed{args.seed}" / "ae_final.pt", args.device
    )
    prior = build_prior(args.device)
    prior.train()

    optimizer = torch.optim.AdamW(prior.parameters(), lr=LR, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    rows, _ = read_packed(args.train_corpus)
    take = args.steps * ROWS_PER_STEP
    g = torch.Generator().manual_seed(args.seed + 1000)
    assert take <= rows.shape[0], (
        f"corpus has {rows.shape[0]} rows but the run needs {take}; an "
        "exhausted order would crash mid-run with an opaque reshape error"
    )
    order = torch.randperm(rows.shape[0], generator=g)[:take]
    heldout, _ = read_packed(args.heldout_corpus)
    heldout_ids = torch.from_numpy(heldout[EVAL_ROWS].astype("int64"))

    loss_generator = torch.Generator().manual_seed(args.seed * 1_000_003 + 37)

    def micro_batch(index: int) -> torch.Tensor:
        sel = order[index * MICRO_BATCH : (index + 1) * MICRO_BATCH]
        return torch.from_numpy(rows[sel.numpy()].astype("int64")).to(args.device)

    start = time.time()
    micro_index = 0
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            ids = micro_batch(micro_index)
            micro_index += 1
            with torch.autocast(args.device, dtype=torch.bfloat16):
                z = standardized_latents(autoencoder, ids)
                losses = ladiff_prior_loss(prior, z, generator=loss_generator)
            loss = losses["total"] / GRAD_ACCUM
            loss.backward()
            step_loss += float(losses["total"]) / GRAD_ACCUM
        torch.nn.utils.clip_grad_norm_(prior.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == args.steps - 1:
            record = {
                "step": step,
                "loss": round(step_loss, 5),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - start, 1),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(record, flush=True)

        if (step + 1) % EVAL_EVERY == 0:
            prior.eval()
            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                z = standardized_latents(autoencoder, heldout_ids.to(args.device))
                eval_losses = ladiff_prior_loss(
                    prior, z, generator=torch.Generator().manual_seed(EVAL_GEN_SEED)
                )
            prior.train()
            record = {
                "step": step,
                "heldout_prior_mse_smoke": round(float(eval_losses["total"]), 5),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(record, flush=True)

        if (step + 1) in (args.steps // 2, args.steps):
            tag = "half" if (step + 1) == args.steps // 2 else "final"
            torch.save(prior.state_dict(), out_dir / f"prior_{tag}.pt")
            (out_dir / f"prior_{tag}.json").write_text(
                json.dumps(
                    {
                        "seed": args.seed,
                        "step": step + 1,
                        "protocol": "issue130 issuecomment-5246782747",
                        "ae_checkpoint": str(
                            args.ae_dir / f"seed{args.seed}" / "ae_final.pt"
                        ),
                        "total_steps": args.steps,
                        "rows_per_step": ROWS_PER_STEP,
                    },
                    indent=2,
                )
            )

    print(f"done: prior seed {args.seed}, {args.steps} steps")


if __name__ == "__main__":
    main()
