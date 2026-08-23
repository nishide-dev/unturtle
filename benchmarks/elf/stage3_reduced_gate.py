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

"""#154 Stage 3: ELF reduced-scale OWT training SCAFFOLD.

STATUS (#164 review): this is a training scaffold, NOT a decision-ready
gate runner.  It trains under the frozen budget and records the training /
heldout curve, but it does NOT yet emit the measurements the frozen Stage-3
verdict needs (official + canonical #152 evaluator columns, GenPPL /
entropy / MAUVE, collapse guards, 50%/100% decision checkpoints, sign
agreement).  Do not read a GO / FAIL / UNDECIDABLE verdict from its output.

Data handling matches the Stage-0 freeze (#164 review, finding 1): the
heldout split is the tail 2,048 rows of the FULL official split, and rows
are visited in a SHUFFLED order like the oracle's DataLoader
(`shuffle=True`, data_utils.py:77 / train.py:287).  `--shards` therefore
only bounds how much TRAINING data is materialized; the heldout indices are
computed against the full 9,737,184-row split and their shards are always
fetched.

Frozen (Stage-0 comment, unchanged): **300M token presentations per seed**,
seeds **42 and 43**, effective batch **64 x 1024** via grad accumulation,
**lr 2.5e-4** (= blr 1e-3 x 64/256), warmup 10% of steps, clip 1.0, EMA
0.9999, bf16 autocast with fp32 master params, heldout = the deterministic
tail of the materialized rows.

Data: 300M presentations need ~293k rows, so training materializes the
first `--shards` arrow files of `openwebtext-t5@0a8443e8` (129,829 rows
each); the frozen heldout tail lives in the LAST shard, which is fetched
separately.  Disk stays bounded without moving the evaluation split.

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
TOTAL_SHARDS = 75
TOTAL_ROWS = 9_737_184  # dataset_info.json at the frozen revision

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
        "--resume-from",
        default=None,
        help="checkpoint to continue from; restores model/EMA/optimizer/RNG/"
        "cursor so the trajectory continues rather than restarting",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=6,
        help="optimizer steps for the disposable integration check",
    )
    return parser.parse_args()


def _shard_path(index: int) -> str:
    from huggingface_hub import hf_hub_download

    name = f"data-{index:05d}-of-{TOTAL_SHARDS:05d}.arrow"
    path = hf_hub_download(
        DATASET, name, revision=DATASET_REVISION, repo_type="dataset"
    )
    print(f"  shard {index}: {name}", flush=True)
    return path


def materialize_rows(shards: int) -> tuple[list[str], list[str]]:
    """Fetch the TRAINING shards (a prefix) and, separately, the LAST shard
    which contains the frozen heldout tail of the full official split."""
    train_paths = [_shard_path(index) for index in range(shards)]
    heldout_paths = (
        train_paths[-1:] if shards >= TOTAL_SHARDS else [_shard_path(TOTAL_SHARDS - 1)]
    )
    return train_paths, heldout_paths


def load_dataset_rows(paths: list[str]):
    """Memory-map the arrow shards as one table (no full materialization)."""
    import pyarrow as pa

    tables = []
    for path in paths:
        with pa.memory_map(path, "rb") as source:
            tables.append(pa.ipc.open_stream(source).read_all())
    return pa.concat_tables(tables)


def make_batch(table, indices, device, pad_token_id=0):
    """One microbatch built with the ORACLE's collation semantics
    (data_utils.py:61-115): rows are VARIABLE length, so pad/truncate to
    `SEQUENCE_LENGTH` and derive the masks from the true lengths —

    - `attention_mask` = position < true length (NOT all ones: padded
      positions must not enter the loss);
    - `encoder_attention_mask` = `build_self_attn_cond_masks` with no
      conditioning prefix (uncond OWT), i.e. every row attends to the
      valid positions;
    - `cond_seq_mask` = zeros (no conditioning tokens).

    Correction #3 (Stage-3): an earlier version assumed the official split
    was pre-packed to 1024 and passed all-ones masks; the smoke run proved
    rows are 846/987/... tokens, so that would have trained on padding.
    """
    import numpy as np
    import torch
    from unturtle_elf._reference.encoder_utils import build_self_attn_cond_masks

    rows = [np.asarray(table["input_ids"][int(i)].as_py()) for i in indices]
    padded, lengths = [], []
    for ids in rows:
        true_len = min(len(ids), SEQUENCE_LENGTH)
        ids = ids[:SEQUENCE_LENGTH]
        if true_len < SEQUENCE_LENGTH:
            ids = np.concatenate(
                [
                    ids,
                    np.full(SEQUENCE_LENGTH - true_len, pad_token_id, dtype=ids.dtype),
                ]
            )
        padded.append(ids)
        lengths.append(true_len)

    ids = np.stack(padded)
    total_lens = np.array(lengths)
    positions = np.arange(SEQUENCE_LENGTH)[None, :]
    is_cond = positions < np.zeros((len(rows), 1), dtype=np.int32)  # uncond
    is_valid = positions < total_lens[:, None]
    encoder_attn, attn, cond = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
    return {
        "input_ids": torch.from_numpy(ids).long().to(device),
        "attention_mask": torch.from_numpy(attn).to(device),
        "encoder_attention_mask": torch.from_numpy(encoder_attn).to(device),
        "cond_seq_mask": torch.from_numpy(cond).to(device),
    }


def build_config(raw_config):
    """The frozen training config: the checkpoint's own training fields with
    the Stage-0 reduced-gate overrides applied explicitly."""

    class Config:
        pass

    config = Config()
    # The oracle's Config CLASS supplies defaults for fields the checkpoint
    # config.yml omits (configs/config.py).  Copy the ones the objective
    # reads, explicitly, so a missing key can never become a silent default
    # (Stage-3 correction #5 — config.yml has no `pad_token`).
    oracle_defaults = {
        "pad_token": "pad",
        "label_drop_prob": 0.0,
        "t_eps": 5e-2,
        "self_cond_prob": 0.5,
        "self_cond_cfg_min": 0.5,
        "self_cond_cfg_max": 5.0,
        "time_schedule": "logit_normal",
        "denoiser_noise_scale": 1.0,
        "decoder_noise_scale": 1.0,
        "num_self_cond_cfg_tokens": 4,
        "latent_mean": 0.0,
        "latent_std": 1.0,
    }
    for key, value in oracle_defaults.items():
        setattr(config, key, value)
    for key, value in raw_config.items():  # checkpoint values win
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
    train_paths, heldout_paths = materialize_rows(args.shards)
    table = load_dataset_rows(train_paths)
    heldout_table = load_dataset_rows(heldout_paths)
    train_rows = table.num_rows
    # The FROZEN heldout split: the last HELDOUT_ROWS rows of the full
    # official split, which live at the end of the final shard.
    heldout_start = heldout_table.num_rows - HELDOUT_ROWS
    assert heldout_start >= 0, "final shard smaller than the heldout tail"
    print(
        f"  train rows: {train_rows:,} from {args.shards} shard(s) | "
        f"heldout: last {HELDOUT_ROWS} rows of the full "
        f"{TOTAL_ROWS:,}-row split",
        flush=True,
    )

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
            """Correction #4 (Stage-3): the oracle hands its 3D float
            self-attention mask straight to `T5EncoderModel` (t5_encoder.py:
            66-80), which the transformers version it targets accepted;
            transformers 5.x rejects a float mask (`bitwise_and` on Float).

            For the UNCONDITIONAL scope every query row of that 3D mask is
            identical and equals the 2D validity mask (proven directly:
            `enc[b][q] == attn[b]` for all q), so collapsing to the 2D long
            mask preserves the oracle's semantics exactly here.  A
            conditional run would NOT be reducible and needs a different
            adapter.
            """
            del deterministic
            if attention_mask is not None and attention_mask.dim() == 3:
                attention_mask = attention_mask[:, 0, :]
            if attention_mask is not None:
                attention_mask = attention_mask.long()
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    encoder_shim = EncoderShim(encoder)

    optimizer = build_muon_optimizer(model, lr=LEARNING_RATE)
    partition = muon_parameter_partition(model)
    ema = init_ema(model)
    generator = torch.Generator().manual_seed(seed)
    # Oracle parity: the official DataLoader shuffles (data_utils.py:77,
    # train.py:287).  A seed-derived permutation gives the same property
    # while staying exactly reproducible and resumable via `row_cursor`.
    row_order = torch.randperm(
        train_rows, generator=torch.Generator().manual_seed(seed + 7919)
    )

    tokens_per_step = EFFECTIVE_BATCH * SEQUENCE_LENGTH
    if args.smoke:
        total_steps = args.smoke_steps
    else:
        # A fixed batch cannot hit 300,000,000 exactly (#164 review,
        # finding 5).  The FROZEN choice is the CEIL step count, so the run
        # never presents FEWER tokens than the budget:
        #   4,578 steps x 65,536 = 300,023,808 >= 300,000,000
        total_steps = -(-TOKENS_PER_SEED // tokens_per_step)
    tokens_target = total_steps * tokens_per_step
    if not args.smoke:
        assert total_steps == 4578 and tokens_target == 300_023_808, (
            f"frozen Stage-3 budget drift: {total_steps} steps / "
            f"{tokens_target} tokens (expected 4578 / 300023808)"
        )
        assert tokens_target >= TOKENS_PER_SEED
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

    HELDOUT_EVAL_SEED = 12345

    @torch.no_grad()
    def heldout_loss(n_batches=4):
        """Hermetic heldout evaluation (#164 review, finding 3).

        The objective draws t / noise / decoder lambda+noise / SC mask+scale
        from the GLOBAL RNG, and the oracle's `model.train()` would re-enable
        dropout after our `eval()`.  So this: (a) snapshots and restores the
        global CPU and CUDA RNG around the whole evaluation, so a checkpoint
        eval cannot shift the training trajectory; (b) reseeds the global
        stream to a FIXED eval seed so every checkpoint sees the identical
        heldout stream; (c) passes `training_mode=False` so dropout stays off.
        """
        cpu_state = torch.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        was_training = model.training
        try:
            model.eval()
            torch.manual_seed(HELDOUT_EVAL_SEED)
            local = torch.Generator().manual_seed(HELDOUT_EVAL_SEED)
            losses, l2_values = [], []
            for index in range(n_batches):
                start = heldout_start + index * args.microbatch
                batch = make_batch(
                    heldout_table,
                    range(start, start + args.microbatch),
                    args.device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, metrics, _ = elf_training_loss(
                        model,
                        encoder_shim,
                        batch,
                        config,
                        dropout_generator=local,
                        training_mode=False,
                    )
                losses.append(float(loss))
                l2_values.append(float(metrics["l2_loss"]))
            return {
                "combined": sum(losses) / len(losses),
                "denoiser_l2": sum(l2_values) / len(l2_values),
            }
        finally:
            if was_training:
                model.train()
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    history = []
    row_cursor = 0
    tokens_seen = 0
    first_step = 0
    if args.resume_from:
        state = torch.load(args.resume_from, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer = build_muon_optimizer(model, lr=LEARNING_RATE)
        optimizer.load_state_dict(state["optimizer"])
        ema = {name: tensor.clone() for name, tensor in state["ema"].items()}
        torch.set_rng_state(state["cpu_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["dropout_generator"])
        row_cursor = state["row_cursor"]
        tokens_seen = state["tokens"]
        history = state.get("history", [])
        first_step = state["step"]
        print(
            f"  resumed from {args.resume_from}: step {first_step}, "
            f"{tokens_seen:,} tokens, cursor {row_cursor}",
            flush=True,
        )
    start_time = time.perf_counter()
    checkpoint_every = max(1, total_steps // 5)

    baseline = heldout_loss()
    print(
        f"  heldout@init: {baseline['combined']:.4f} "
        f"(denoiser L2 {baseline['denoiser_l2']:.4f})",
        flush=True,
    )
    history.append(
        {
            "step": 0,
            "tokens": 0,
            "heldout": baseline["combined"],
            "heldout_denoiser_l2": baseline["denoiser_l2"],
        }
    )

    for step in range(first_step, total_steps):
        set_lr(step)
        for _ in range(accum):
            indices = [
                int(row_order[(row_cursor + offset) % train_rows])
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
                "heldout": heldout["combined"],
                "heldout_denoiser_l2": heldout["denoiser_l2"],
                "elapsed_seconds": elapsed,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            history.append(record)
            print(
                f"  step {step + 1}/{total_steps} tokens {tokens_seen:,} "
                f"heldout {heldout['combined']:.4f} "
                f"L2 {heldout['denoiser_l2']:.4f} ({elapsed:.0f}s)",
                flush=True,
            )
            # FULL training state (#164 review, finding 4): model + EMA +
            # optimizer + both RNG streams + the data cursor, so a resumed
            # run can continue the SAME trajectory rather than merely
            # reproducing a forward pass.
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema,
                    "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "dropout_generator": generator.get_state(),
                    "row_cursor": row_cursor,
                    "history": history,
                    "step": step + 1,
                    "tokens": tokens_seen,
                    "seed": seed,
                    "smoke": args.smoke,
                },
                out / f"checkpoint_step{step + 1}.pt",
            )

    print("[4/5] model-state save/load forward identity ...", flush=True)
    # NAMED PRECISELY (#164 review, finding 4): this checks that reloading
    # the MODEL state reproduces the forward pass.  It is NOT a training-
    # resume identity claim; that requires the N-vs-(N save/restore +1)
    # comparison, which `tests/test_elf_training_resume.py` covers.
    payload = torch.load(out / f"checkpoint_step{total_steps}.pt", weights_only=False)
    fresh = build_elf_model(raw_config).to(args.device)
    fresh.load_state_dict(payload["model"], strict=True)
    fresh.eval()
    model.eval()
    probe = make_batch(
        heldout_table, range(heldout_start, heldout_start + 2), args.device
    )
    x0 = encoder_shim(probe["input_ids"], probe["attention_mask"])
    x0 = (x0 - config.latent_mean) / config.latent_std
    with torch.no_grad():
        pair = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
        t_probe = torch.full((x0.shape[0],), 0.4, device=args.device)
        scale = torch.full((x0.shape[0],), 3.0, device=args.device)
        before, _ = model(pair, t_probe, deterministic=True, self_cond_cfg_scale=scale)
        after, _ = fresh(pair, t_probe, deterministic=True, self_cond_cfg_scale=scale)
    roundtrip_ok = bool(torch.equal(before, after))
    print(f"  model-state forward identical: {roundtrip_ok}", flush=True)
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
            "train_rows": train_rows,
            "total_split_rows": TOTAL_ROWS,
            "data_order": "seed-derived permutation (oracle shuffle=True)",
            "heldout_split": "tail of the FULL official split (Stage-0 freeze)",
            "heldout_rows": HELDOUT_ROWS,
            "resumed_from": args.resume_from,
            "first_step": first_step,
        },
        "dataset": {"repo": DATASET, "revision": DATASET_REVISION},
        "checkpoint": {"repo": DEFAULT_CHECKPOINT, "revision": DEFAULT_REVISION},
        "partition": {
            "muon": len(partition["muon"]),
            "adam": len(partition["adam"]),
        },
        "model_state_forward_identical": roundtrip_ok,
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
