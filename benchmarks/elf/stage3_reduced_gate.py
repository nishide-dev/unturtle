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

"""#154 Stage 3: ELF reduced-scale OWT eligibility gate.

Frozen (Stage-0 comment, unchanged): **300M token presentations per seed**,
seeds **42 and 43**, effective batch **64 x 1024** via grad accumulation,
**lr 2.5e-4** (= blr 1e-3 x 64/256), warmup 10% of steps, clip 1.0, EMA
0.9999, bf16 autocast with fp32 master params, heldout = the deterministic
tail of the materialized rows.

Data: only ~293k rows are needed for 300M presentations, so this fetches
the first N shards of `openwebtext-t5@0a8443e8` (129,829 rows each) rather
than the full 37.4GB — a disk decision, not a protocol change; the rows are
a deterministic PREFIX of the official split.

Two modes:

* ``--smoke``  — the mandated DISPOSABLE integration check: real dataset
  revision, ELF-B, corrected optimizer, bf16, grad accumulation, a
  checkpoint/resume round trip and token accounting.  Writes into a
  ``smoke/`` subdirectory, and its checkpoint/optimizer/RNG state is NEVER
  reused by a seeded run.  Generation quality is not evaluated.
* default — one seeded decision run (fresh init).

Usage:
    .venv/bin/python benchmarks/elf/stage3_reduced_gate.py --smoke \
        --device cuda:0 --out benchmarks/results/elf_stage3
    .venv/bin/python benchmarks/elf/stage3_reduced_gate.py --seed 42 \
        --device cuda:0 --out benchmarks/results/elf_stage3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

DATASET = "embedded-language-flows/openwebtext-t5"
DATASET_REVISION = "0a8443e847ee6206e4737a6b9a93218347eabc08"
ROWS_PER_SHARD = 129_829

# --- frozen Stage-0 numbers -------------------------------------------------
TOKENS_PER_SEED = 300_000_000
EFFECTIVE_BATCH = 64
SEQUENCE_LENGTH = 1024
LEARNING_RATE = 2.5e-4
WARMUP_FRACTION = 0.10
GRAD_CLIP = 1.0
EMA_DECAY = 0.9999
HELDOUT_ROWS = 2_048


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=6,
        help="optimizer steps for the disposable integration check",
    )
    return parser.parse_args()


def materialize_rows(shards: int) -> list[str]:
    """Download the first ``shards`` arrow files at the frozen revision."""
    from huggingface_hub import hf_hub_download

    paths = []
    for index in range(shards):
        name = f"data-{index:05d}-of-00075.arrow"
        paths.append(
            hf_hub_download(
                DATASET,
                name,
                revision=DATASET_REVISION,
                repo_type="dataset",
            )
        )
        print(f"  shard {index}: {name}", flush=True)
    return paths


def load_dataset_rows(paths: list[str]):
    """Memory-map the arrow shards as one table (no full materialization)."""
    import pyarrow as pa

    tables = []
    for path in paths:
        with pa.memory_map(path, "rb") as source:
            tables.append(pa.ipc.open_stream(source).read_all())
    return pa.concat_tables(tables)


def make_batch(table, indices, device):
    """One microbatch in the oracle's train_step contract (uncond OWT:
    every position valid, no conditioning prefix)."""
    import torch

    rows = [table["input_ids"][int(i)].as_py() for i in indices]
    input_ids = torch.tensor([row[:SEQUENCE_LENGTH] for row in rows], dtype=torch.long)
    if input_ids.shape[1] < SEQUENCE_LENGTH:
        raise ValueError(
            f"row shorter than {SEQUENCE_LENGTH} tokens; the official split "
            "is pre-packed, so this indicates a wrong shard/revision"
        )
    ones = torch.ones(input_ids.shape, dtype=torch.float32)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": ones.to(device),
        "encoder_attention_mask": ones.to(device),
        "cond_seq_mask": torch.zeros(input_ids.shape, device=device),
    }


def build_config(raw_config):
    """The frozen training config: the checkpoint's own training fields with
    the Stage-0 reduced-gate overrides applied explicitly."""

    class Config:
        pass

    config = Config()
    for key, value in raw_config.items():
        setattr(config, key, value)
    config.grad_accum_steps = 1  # accumulation is driven by this script
    config.ema_decay1 = EMA_DECAY
    config.use_bf16 = True
    return config


def main():
    args = parse_args()
    out = pathlib.Path(args.out) / ("smoke" if args.smoke else f"seed_{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    if not args.smoke and args.seed is None:
        raise SystemExit("--seed is required unless --smoke")

    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from transformers import T5EncoderModel
    from unturtle_elf.loader import (
        DEFAULT_CHECKPOINT,
        DEFAULT_REVISION,
        build_elf_model,
    )
    from unturtle_elf.training import (
        build_muon_optimizer,
        elf_training_loss,
        ema_update,
        init_ema,
        muon_parameter_partition,
    )

    print("[1/5] materializing dataset shards ...", flush=True)
    paths = materialize_rows(args.shards)
    table = load_dataset_rows(paths)
    total_rows = table.num_rows
    heldout_start = total_rows - HELDOUT_ROWS
    print(f"  rows: {total_rows:,} (heldout tail: {HELDOUT_ROWS})", flush=True)

    print("[2/5] building ELF-B + frozen T5 encoder ...", flush=True)
    config_path = hf_hub_download(
        DEFAULT_CHECKPOINT, "config.yml", revision=DEFAULT_REVISION
    )
    with open(config_path) as handle:
        raw_config = yaml.safe_load(handle)
    config = build_config(raw_config)

    seed = args.seed if args.seed is not None else 0
    torch.manual_seed(seed)
    model = build_elf_model(raw_config).to(args.device)
    model.train()
    encoder = T5EncoderModel.from_pretrained(raw_config["encoder_model_name"]).to(
        args.device
    )
    encoder.eval().requires_grad_(False)

    class EncoderShim(torch.nn.Module):
        """Adapts HF T5EncoderModel to the oracle's encoder contract."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask=None, deterministic=True):
            del deterministic
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    encoder_shim = EncoderShim(encoder)

    optimizer = build_muon_optimizer(model, lr=LEARNING_RATE)
    partition = muon_parameter_partition(model)
    ema = init_ema(model)
    generator = torch.Generator().manual_seed(seed)

    tokens_target = (
        args.smoke_steps * EFFECTIVE_BATCH * SEQUENCE_LENGTH
        if args.smoke
        else TOKENS_PER_SEED
    )
    total_steps = tokens_target // (EFFECTIVE_BATCH * SEQUENCE_LENGTH)
    accum = EFFECTIVE_BATCH // args.microbatch
    warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))
    print(
        f"[3/5] {'SMOKE' if args.smoke else f'seed {seed}'}: "
        f"{total_steps:,} optimizer steps x {EFFECTIVE_BATCH} seqs "
        f"(microbatch {args.microbatch} x accum {accum}), "
        f"warmup {warmup_steps}, lr {LEARNING_RATE}",
        flush=True,
    )
    print(
        f"  Muon params: {len(partition['muon'])} | aux Adam: {len(partition['adam'])}",
        flush=True,
    )

    def set_lr(step):
        scale = min(1.0, (step + 1) / warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * scale

    @torch.no_grad()
    def heldout_loss(n_batches=4):
        model.eval()
        losses = []
        local = torch.Generator().manual_seed(12345)  # fixed heldout stream
        for index in range(n_batches):
            start = heldout_start + index * args.microbatch
            batch = make_batch(
                table, range(start, start + args.microbatch), args.device
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, metrics, _ = elf_training_loss(
                    model, encoder_shim, batch, config, dropout_generator=local
                )
            losses.append(float(loss))
        model.train()
        return sum(losses) / len(losses)

    history = []
    row_cursor = 0
    tokens_seen = 0
    start_time = time.perf_counter()
    checkpoint_every = max(1, total_steps // 5)

    baseline = heldout_loss()
    print(f"  heldout@init: {baseline:.4f}", flush=True)
    history.append({"step": 0, "tokens": 0, "heldout": baseline})

    for step in range(total_steps):
        set_lr(step)
        for _ in range(accum):
            indices = [
                (row_cursor + offset) % heldout_start
                for offset in range(args.microbatch)
            ]
            row_cursor += args.microbatch
            batch = make_batch(table, indices, args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, metrics, _ = elf_training_loss(
                    model,
                    encoder_shim,
                    batch,
                    config,
                    dropout_generator=generator,
                )
            (loss / accum).backward()
            tokens_seen += args.microbatch * SEQUENCE_LENGTH
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
        )
        optimizer.step()
        ema_update(ema, model, EMA_DECAY)
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % checkpoint_every == 0 or step == total_steps - 1:
            heldout = heldout_loss()
            elapsed = time.perf_counter() - start_time
            record = {
                "step": step + 1,
                "tokens": tokens_seen,
                "train_loss": float(loss.detach()),
                "train_ce": float(metrics["ce_loss"]),
                "train_l2": float(metrics["l2_loss"]),
                "heldout": heldout,
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            history.append(record)
            print(
                f"  step {step + 1}/{total_steps} tokens {tokens_seen:,} "
                f"heldout {heldout:.4f} ({elapsed:.0f}s)",
                flush=True,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema,
                    "optimizer": optimizer.state_dict(),
                    "step": step + 1,
                    "tokens": tokens_seen,
                    "seed": seed,
                    "smoke": args.smoke,
                },
                out / "checkpoint_last.pt",
            )

    print("[4/5] checkpoint/resume round trip ...", flush=True)
    payload = torch.load(out / "checkpoint_last.pt", weights_only=True)
    fresh = build_elf_model(raw_config).to(args.device)
    fresh.load_state_dict(payload["model"], strict=True)
    fresh.eval()
    model.eval()
    probe = make_batch(table, range(heldout_start, heldout_start + 2), args.device)
    x0 = encoder_shim(probe["input_ids"], probe["attention_mask"])
    x0 = (x0 - config.latent_mean) / config.latent_std
    with torch.no_grad():
        pair = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
        t_probe = torch.full((x0.shape[0],), 0.4, device=args.device)
        scale = torch.full((x0.shape[0],), 3.0, device=args.device)
        before, _ = model(pair, t_probe, deterministic=True, self_cond_cfg_scale=scale)
        after, _ = fresh(pair, t_probe, deterministic=True, self_cond_cfg_scale=scale)
    roundtrip_ok = bool(torch.equal(before, after))
    print(f"  resume output identical: {roundtrip_ok}", flush=True)
    model.train()

    print("[5/5] writing report ...", flush=True)
    report = {
        "mode": "smoke" if args.smoke else "decision",
        "seed": seed,
        "frozen": {
            "tokens_per_seed": TOKENS_PER_SEED,
            "effective_batch": EFFECTIVE_BATCH,
            "sequence_length": SEQUENCE_LENGTH,
            "learning_rate": LEARNING_RATE,
            "warmup_fraction": WARMUP_FRACTION,
            "ema_decay": EMA_DECAY,
        },
        "executed": {
            "optimizer_steps": total_steps,
            "microbatch": args.microbatch,
            "grad_accum": accum,
            "tokens_seen": tokens_seen,
            "tokens_target": tokens_target,
            "warmup_steps": warmup_steps,
            "shards": args.shards,
            "rows": total_rows,
            "heldout_rows": HELDOUT_ROWS,
        },
        "dataset": {"repo": DATASET, "revision": DATASET_REVISION},
        "checkpoint": {"repo": DEFAULT_CHECKPOINT, "revision": DEFAULT_REVISION},
        "partition": {
            "muon": len(partition["muon"]),
            "adam": len(partition["adam"]),
        },
        "roundtrip_identical": roundtrip_ok,
        "history": history,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": time.perf_counter() - start_time,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    if args.smoke:
        print(
            "\nSMOKE COMPLETE — this checkpoint/optimizer/RNG state is "
            "DISPOSABLE and must not seed a decision run.",
            flush=True,
        )


if __name__ == "__main__":
    main()
