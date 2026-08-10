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

"""Gate A readout (#130) — frozen BEFORE the training runs.

Protocol (issue #130, issuecomment-5242035162):

- held-out rows [0:512], fp32 eval, standardizer statistics frozen from
  training;
- t in {0.75, 0.9, 1.0}; the mask for each t comes from a FIXED generator
  (seed 9000 + t*100) and is shared by every arm;
- arms (identical mask, identical forward path):
    true      z = encoder(standardized clean features)
    dropout   z = mu_z + sigma_z * eta        (fixed generator 7000 + t*100)
    wrong     z = true z of row (i + 257) mod 512
    shuffled  z with its position axis permuted by a fixed permutation
- metrics: masked-position NLL (mean CE) and recovery (argmax match);
  benefit(x) := NLL(dropout) - NLL(x);
- PASS (all, on BOTH seeds):
    1. every t: NLL(true) < NLL(dropout) and recovery(true) > recovery(dropout)
    2. every t: benefit(wrong)/benefit(true) < 0.25
                and benefit(shuffled)/benefit(true) < 0.25
- auxiliary (recorded, NOT gated): monotonicity of benefit(true) in t.

Stop/go: FAIL => no prior-side rescue; the negative is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
ROWS = 512
T_VALUES = (0.75, 0.9, 1.0)
WRONG_OFFSET = 257
CHUNK = 16
BENEFIT_LEAK_CUTOFF = 0.25


def load_trained_autoencoder(checkpoint: Path, device: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_ladiff_ae", Path(__file__).parent / "train_ladiff_ae.py"
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    autoencoder = train_mod.build_autoencoder(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    autoencoder.load_state_dict(state)  # fingerprint enforces trunk identity
    return autoencoder.eval()


@torch.no_grad()
def arm_latents(autoencoder, clean_ids: torch.Tensor, t: float) -> dict:
    chunks = []
    for start in range(0, clean_ids.shape[0], CHUNK):
        ids = clean_ids[start : start + CHUNK]
        features = autoencoder.feature_standardizer(autoencoder.features(ids))
        chunks.append(autoencoder.encoder(features))
    true = torch.cat(chunks, dim=0)

    g = torch.Generator().manual_seed(7000 + int(t * 100))
    eta = torch.randn(true.shape, generator=g).to(true.device)
    dropout = (
        autoencoder.latent_standardizer.mean + autoencoder.latent_standardizer.std * eta
    )

    wrong = true.roll(-WRONG_OFFSET, dims=0)

    perm_g = torch.Generator().manual_seed(7100)
    perm = torch.randperm(true.shape[1], generator=perm_g).to(true.device)
    shuffled = true[:, perm, :]
    return {"true": true, "dropout": dropout, "wrong": wrong, "shuffled": shuffled}


@torch.no_grad()
def masked_metrics(autoencoder, corrupted, clean, masked, latents) -> dict:
    nll_sum, hit_sum, count = 0.0, 0.0, 0
    for start in range(0, corrupted.shape[0], CHUNK):
        sl = slice(start, start + CHUNK)
        logits = autoencoder.decoder(
            input_ids=corrupted[sl], latents=latents[sl]
        ).logits.float()
        m = masked[sl]
        target = clean[sl][m]
        chunk_logits = logits[m]
        nll_sum += float(F.cross_entropy(chunk_logits, target, reduction="sum"))
        hit_sum += float((chunk_logits.argmax(-1) == target).sum())
        count += int(m.sum())
    return {"nll": nll_sum / count, "recovery": hit_sum / count, "masked": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-label", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--heldout-corpus",
        type=Path,
        default=REPO_ROOT / "dev/local/owt/heldout_1024",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from unturtle.utils.packed_text import read_packed

    heldout, _ = read_packed(args.heldout_corpus)
    clean = torch.from_numpy(heldout[:ROWS].astype("int64")).to(args.device)

    autoencoder = load_trained_autoencoder(args.checkpoint, args.device)
    mask_id = autoencoder.config.mask_token_id

    results: dict = {
        "seed": args.seed_label,
        "checkpoint": str(args.checkpoint),
        "protocol": "issue130 issuecomment-5242035162",
        "rows": ROWS,
        "per_t": {},
    }
    for t in T_VALUES:
        g = torch.Generator().manual_seed(9000 + int(t * 100))
        masked = (torch.rand(clean.shape, generator=g) < t).to(args.device)
        assert bool(masked.any(dim=1).all()), "dead row at high mask ratio"
        corrupted = clean.masked_fill(masked, mask_id)

        latents = arm_latents(autoencoder, clean, t)
        arms = {
            name: masked_metrics(autoencoder, corrupted, clean, masked, z)
            for name, z in latents.items()
        }
        benefit_true = arms["dropout"]["nll"] - arms["true"]["nll"]
        entry = {
            "arms": arms,
            "benefit_true": benefit_true,
            "benefit_wrong": arms["dropout"]["nll"] - arms["wrong"]["nll"],
            "benefit_shuffled": arms["dropout"]["nll"] - arms["shuffled"]["nll"],
        }
        entry["criterion1"] = (
            arms["true"]["nll"] < arms["dropout"]["nll"]
            and arms["true"]["recovery"] > arms["dropout"]["recovery"]
        )
        entry["criterion2"] = (
            benefit_true > 0
            and entry["benefit_wrong"] / benefit_true < BENEFIT_LEAK_CUTOFF
            and entry["benefit_shuffled"] / benefit_true < BENEFIT_LEAK_CUTOFF
        )
        results["per_t"][str(t)] = entry

    results["gate_a_this_seed"] = all(
        e["criterion1"] and e["criterion2"] for e in results["per_t"].values()
    )
    benefits = [results["per_t"][str(t)]["benefit_true"] for t in T_VALUES]
    results["monotone_benefit_auxiliary"] = benefits == sorted(benefits)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "per_t"}, indent=2))
    for t, entry in results["per_t"].items():
        print(
            t,
            {a: round(m["nll"], 4) for a, m in entry["arms"].items()},
            "c1",
            entry["criterion1"],
            "c2",
            entry["criterion2"],
        )


if __name__ == "__main__":
    main()
