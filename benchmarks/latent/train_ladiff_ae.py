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

"""LaDiff AE training run (#130) — protocol frozen on the issue BEFORE runs.

All decision-relevant hyperparameters mirror the pre-registered protocol
(issuecomment-5242035162): 1.0B token presentations/seed as 15,259 steps of
64x1024 effective batch (micro 32 x accum 2), AdamW lr 5e-5 cosine, encoder
warmup 80 / decoder 800 (the paper's 0.5%/5% of total steps), best-aug
regularizers with p_zdropout=0.1, open_latent_channel(1e-3) as the recorded
deviation. In-loop held-out loss is smoke only, never a verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

BLOCK = 1024
MICRO_BATCH = 32
GRAD_ACCUM = 2
EFFECTIVE_TOKENS = MICRO_BATCH * GRAD_ACCUM * BLOCK
TOTAL_STEPS = math.ceil(1e9 / EFFECTIVE_TOKENS)  # 15,259
LR = 5e-5
ENCODER_WARMUP = 80
DECODER_WARMUP = 800
OPEN_CHANNEL_STD = 1e-3
GRAD_CLIP = 1.0
EVAL_EVERY = 1000
EVAL_ROWS = 128


def build_autoencoder(device: str):
    from unturtle.models.backbones.mdlm_dit.convert_mdlm_owt import load_mdlm_owt
    from unturtle.models.latent import (
        LaDiffAutoencoder,
        LaDiffDiTConfig,
        LatentConditionedMDLMDiT,
    )

    base = load_mdlm_owt()
    config = LaDiffDiTConfig(
        vocab_size=base.config.vocab_size,
        hidden_size=base.config.hidden_size,
        cond_dim=base.config.cond_dim,
        num_hidden_layers=base.config.num_hidden_layers,
        num_attention_heads=base.config.num_attention_heads,
        dropout=base.config.dropout,
        max_position_embeddings=base.config.max_position_embeddings,
        mask_token_id=base.config.mask_token_id,
        num_latents=512,
        latent_dim=base.config.hidden_size,
        encoder_layers=4,
    )
    decoder = LatentConditionedMDLMDiT(config)
    decoder.model.load_state_dict(base.model.state_dict())
    autoencoder = LaDiffAutoencoder(config, decoder)  # copies the trunk FIRST
    decoder.open_latent_channel(std=OPEN_CHANNEL_STD)  # then the deviation
    return autoencoder.to(device)


def data_order(num_rows: int, take: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(num_rows, generator=g)[:take]


def lr_lambda(step: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, TOTAL_STEPS - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "dev/local/ladiff_ae"
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

    from unturtle.models.latent import ladiff_autoencoder_loss
    from unturtle.utils.packed_text import read_packed

    out_dir = args.out_dir / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    torch.manual_seed(args.seed)
    autoencoder = build_autoencoder(args.device)
    autoencoder.train()

    encoder_params = list(autoencoder.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    decoder_params = [
        p
        for p in autoencoder.decoder.parameters()
        if p.requires_grad and id(p) not in encoder_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "name": "encoder"},
            {"params": decoder_params, "name": "decoder"},
        ],
        lr=LR,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda s: lr_lambda(s, ENCODER_WARMUP),
            lambda s: lr_lambda(s, DECODER_WARMUP),
        ],
    )

    rows, _ = read_packed(args.train_corpus)
    take = args.steps * MICRO_BATCH * GRAD_ACCUM
    order = data_order(rows.shape[0], min(take, rows.shape[0]), args.seed)
    heldout, _ = read_packed(args.heldout_corpus)
    heldout_ids = torch.from_numpy(heldout[:EVAL_ROWS].astype("int64"))

    loss_generator = torch.Generator().manual_seed(args.seed * 1_000_003 + 17)
    eval_generator_seed = 4242  # fixed across seeds: identical eval corruption

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
                losses = ladiff_autoencoder_loss(
                    autoencoder, ids, generator=loss_generator
                )
            loss = losses["total"] / GRAD_ACCUM
            loss.backward()
            step_loss += float(losses["total"]) / GRAD_ACCUM
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            GRAD_CLIP,
        )
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == args.steps - 1:
            record = {
                "step": step,
                "loss": round(step_loss, 5),
                "lr_encoder": scheduler.get_last_lr()[0],
                "lr_decoder": scheduler.get_last_lr()[1],
                "elapsed_s": round(time.time() - start, 1),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(record, flush=True)

        if (step + 1) % EVAL_EVERY == 0:
            autoencoder.eval()
            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                eval_losses = ladiff_autoencoder_loss(
                    autoencoder,
                    heldout_ids.to(args.device),
                    generator=torch.Generator().manual_seed(eval_generator_seed),
                )
            autoencoder.train()
            record = {
                "step": step,
                "heldout_loss_smoke": round(float(eval_losses["total"]), 5),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(record, flush=True)

        if (step + 1) in (args.steps // 2, args.steps):
            tag = "half" if (step + 1) == args.steps // 2 else "final"
            torch.save(autoencoder.state_dict(), out_dir / f"ae_{tag}.pt")
            (out_dir / f"ae_{tag}.json").write_text(
                json.dumps(
                    {
                        "seed": args.seed,
                        "step": step + 1,
                        "protocol": "issue130 issuecomment-5242035162",
                        "total_steps": args.steps,
                        "effective_tokens_per_step": EFFECTIVE_TOKENS,
                        "open_channel_std": OPEN_CHANNEL_STD,
                    },
                    indent=2,
                )
            )

    print(f"done: seed {args.seed}, {args.steps} steps")


if __name__ == "__main__":
    main()
