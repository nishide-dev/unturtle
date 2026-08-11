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

"""Gate B readout (#130) — frozen BEFORE the prior runs
(issue comment 5246782747).

Arms (same decoder checkpoint = the seed's AE decoder):
  ladiff      z ~ prior (N_cont=200, gamma=0) -> denormalize -> decode
  latent_off  latents=None (adapters skipped; the pure finetuned-MDLM path)
  gaussian    latents = mu_z + sigma_z * eta (trained unconditional mode;
              AUXILIARY, recorded only — not part of the verdict)

Per (arm, N_disc, seed): 256 generations, prompt [BOS], max_new_tokens 1023,
bf16 autocast; discrete RNG seed 5000 + N_disc*10 + seed IDENTICAL across
arms; latent RNG in the 6000 block. N_disc grid {32, 64, 128, 256} + 1024
anchor. MAUVE: gpt2 features, max_text_length 256, num_buckets 12,
reference = held-out rows [512:768] detokenized (disjoint from Gate A's
[0:512] and the training-smoke rows [768:1024]).

Collapse guards per ARM per cell (a contrast is valid iff BOTH its arms
pass): distinct_fraction > 0.3, pooled_unigram_entropy > 4.0 nats,
unique_rows_fraction > 0.99.

VERDICT (registered): LaDiff MAUVE > latent_off MAUVE at N_disc=64 AND 128,
same sign on BOTH seeds, no collapsed arm at those cells -> PASS.
Anything else -> FAIL/undecidable; #130 closes with that verdict.

Latency: per arm, latent-prior sampling and discrete decode timed
separately at batch 1 and 32 (model construction and warm-up excluded
uniformly); facts only, no acceptance threshold.

FROZEN RESULTS (2026-08-12, raw JSONs dev/local/ladiff_gate_b/seed{0,1}/):

VERDICT: FAIL — decidable, both seeds, all four verdict cells valid (no
collapse), same sign everywhere:

    seed  N_disc  LaDiff   latent_off   winner
    0     64      0.366    0.812        latent_off
    0     128     0.393    0.839        latent_off
    1     64      0.456    0.842        latent_off
    1     128     0.575    0.806        latent_off

Full grid (MAUVE, LaDiff / latent_off / gaussian):
    N=32    s0 0.385/0.770/0.814   s1 0.399/0.839/0.710
    N=64    s0 0.366/0.812/0.777   s1 0.456/0.842/0.714
    N=128   s0 0.393/0.839/0.846   s1 0.575/0.806/0.829
    N=256   s0 0.460/0.857/0.876   s1 0.467/0.893/0.821
    N=1024  s0 0.442/0.939/0.837   s1 0.495/0.848/0.892

The failure localizes to the PRIOR: Gate A showed true latents help and
wrong latents hurt; here prior-sampled latents (0.37-0.58) are consistently
worse than EVERY unconditional mode (0.71-0.94).  The decoder and the
latent channel work; the prior's samples are off-manifold at this budget
(~1-2% of the paper's).  LaDiff improves with N_disc but plateaus far
below.  Latency (LaDiff arm, N_cont=200): batch1 prior 1.15s / decode
0.51s @64; batch32 36.4s / 10.8s — the prior also dominates cost at
N_disc <= 128.

Two mechanics fixes were recorded on the issue BEFORE the verdict
(temperature=1.0 reverse-kernel semantics; cell-owned RNG after the guard
trio caught 32/256 duplicated rows); verdict rule and thresholds unchanged.
Per the frozen stop/go this negative is the real-text LaDiff verdict for
#130; no DiLaDiff issue is opened.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

N_GENERATIONS = 256
GEN_BATCH = 32
N_DISC_GRID = (32, 64, 128, 256, 1024)
VERDICT_POINTS = (64, 128)
N_CONT = 200
GAMMA = 0.0
MAX_NEW = 1023
BOS = 50256
REFERENCE_ROWS = slice(512, 768)
GUARDS = {
    "distinct_fraction": 0.3,
    "pooled_unigram_entropy": 4.0,
    "unique_rows_fraction": 0.99,
}
MAUVE_SETTINGS = {"max_text_length": 256, "num_buckets": 12}


def _load(module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).parent / f"{module_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_models(seed: int, device: str):
    autoencoder = _load("train_ladiff_prior").load_frozen_autoencoder(
        REPO_ROOT / f"dev/local/ladiff_ae/seed{seed}/ae_final.pt", device
    )
    from unturtle.models.latent import LaDiffPriorConfig, LatentPriorDenoiser

    prior = LatentPriorDenoiser(LaDiffPriorConfig()).to(device)
    state = torch.load(
        REPO_ROOT / f"dev/local/ladiff_prior/seed{seed}/prior_final.pt",
        map_location=device,
        weights_only=True,
    )
    prior.load_state_dict(state)
    return autoencoder, prior.eval()


def sample_arm_latents(
    arm: str, autoencoder, prior, batch: int, n_disc: int, seed: int, device: str, g
):
    """Latent construction per arm; the 6000-block generator ``g`` is owned
    by the CELL and advances across batches — re-seeding it per batch made
    every batch identical (the unique_rows=32/256 mechanics bug)."""
    from unturtle.models.latent import sample_latent_prior

    if arm == "latent_off":
        return None, 0.0
    std = autoencoder.latent_standardizer
    start = time.perf_counter()
    if arm == "ladiff":
        z = sample_latent_prior(
            prior, batch=batch, steps=N_CONT, gamma=GAMMA, generator=g
        )
        z = std.std * z + std.mean  # denormalize (Algorithm 3 line 18)
    elif arm == "gaussian":
        eta = torch.randn(
            (batch, prior.config.num_latents, prior.config.latent_dim), generator=g
        ).to(device)
        z = std.mean + std.std * eta
    else:
        raise ValueError(arm)
    return z, time.perf_counter() - start


def generate_cell(arm, autoencoder, prior, n_disc, seed, device):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    decoder = autoencoder.decoder
    texts, token_batches, prior_s, decode_s = [], [], 0.0, 0.0
    # Seed ONCE per cell (identical across arms), never per batch: a per-batch
    # reset replayed the same randomness 8x and produced 32x8 copied rows.
    torch.manual_seed(5000 + n_disc * 10 + seed)
    latent_g = torch.Generator().manual_seed(6000 + n_disc * 10 + seed)
    for start in range(0, N_GENERATIONS, GEN_BATCH):
        batch = min(GEN_BATCH, N_GENERATIONS - start)
        latents, dt_prior = sample_arm_latents(
            arm, autoencoder, prior, batch, n_disc, seed, device, latent_g
        )
        prior_s += dt_prior
        prompt = torch.full((batch, 1), BOS, dtype=torch.long, device=device)
        t0 = time.perf_counter()
        with torch.autocast(device, dtype=torch.bfloat16):
            kwargs = dict(
                algorithm="mdlm",
                max_new_tokens=MAX_NEW,
                steps=n_disc,
                # Reverse-kernel semantics (reference _sample_categorical);
                # the unpinned default (0 = argmax) collapsed ALL arms to
                # marginal-mode text in the mechanics smoke — amended on the
                # issue before any decision output.
                temperature=1.0,
            )
            if latents is not None:
                kwargs["latents"] = latents
            out = decoder.generate(prompt, **kwargs)
        decode_s += time.perf_counter() - t0
        generated = out[:, 1:].cpu()  # strip the BOS prompt column
        token_batches.append(generated)
        texts.extend(tok.decode(row.tolist()) for row in generated)
    tokens = torch.cat(token_batches, dim=0)
    return texts, tokens, {"prior_s": round(prior_s, 2), "decode_s": round(decode_s, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "dev/local/ladiff_gate_b"
    )
    parser.add_argument(
        "--heldout-corpus", type=Path, default=REPO_ROOT / "dev/local/owt/heldout_1024"
    )
    parser.add_argument(
        "--arms", nargs="+", default=["ladiff", "latent_off", "gaussian"]
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from unturtle.eval import (
        distinct_fraction,
        mauve_score,
        pooled_unigram_entropy,
        unique_rows_fraction,
    )
    from unturtle.utils.packed_text import read_packed

    out_dir = args.out_dir / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("gpt2")
    heldout, _ = read_packed(args.heldout_corpus)
    reference = [
        tok.decode(row[1:].tolist())
        for row in torch.from_numpy(heldout[REFERENCE_ROWS].astype("int64"))
    ]

    autoencoder, prior = load_models(args.seed, args.device)

    results = {}
    for n_disc in N_DISC_GRID:
        for arm in args.arms:
            cell_path = out_dir / f"{arm}_ndisc{n_disc}.json"
            if cell_path.exists():
                results[f"{arm}_{n_disc}"] = json.loads(cell_path.read_text())
                continue
            texts, tokens, latency = generate_cell(
                arm, autoencoder, prior, n_disc, args.seed, args.device
            )
            # Guards on the RAW generated token tensor (uniform width; avoids
            # retokenization drift).
            guards = {
                "distinct_fraction": distinct_fraction(tokens),
                "pooled_unigram_entropy": pooled_unigram_entropy(tokens),
                "unique_rows_fraction": unique_rows_fraction(tokens),
            }
            collapsed = any(guards[k] <= v for k, v in GUARDS.items())
            mauve = mauve_score(
                reference_texts=reference, generated_texts=texts, **MAUVE_SETTINGS
            )
            cell = {
                "arm": arm,
                "n_disc": n_disc,
                "seed": args.seed,
                "mauve": mauve,
                "guards": guards,
                "collapsed": collapsed,
                "latency": latency,
                "n_generations": len(texts),
                "mauve_settings": MAUVE_SETTINGS,
                "protocol": "issue130 issuecomment-5246782747",
            }
            cell_path.write_text(json.dumps(cell, indent=2))
            (out_dir / f"{arm}_ndisc{n_disc}_texts.json").write_text(json.dumps(texts))
            results[f"{arm}_{n_disc}"] = cell
            print(
                {k: cell[k] for k in ("arm", "n_disc", "mauve", "collapsed")},
                flush=True,
            )

    verdict_cells = {}
    for n in VERDICT_POINTS:
        la, off = results.get(f"ladiff_{n}"), results.get(f"latent_off_{n}")
        if la and off:
            verdict_cells[str(n)] = {
                "ladiff_mauve": la["mauve"],
                "latent_off_mauve": off["mauve"],
                "ladiff_wins": la["mauve"] > off["mauve"],
                "valid": not (la["collapsed"] or off["collapsed"]),
            }
    summary = {
        "seed": args.seed,
        "verdict_cells": verdict_cells,
        "gate_b_this_seed": bool(verdict_cells)
        and all(c["ladiff_wins"] and c["valid"] for c in verdict_cells.values()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
