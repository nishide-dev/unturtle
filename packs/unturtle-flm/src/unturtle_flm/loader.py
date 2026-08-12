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

"""FLM/FMLM checkpoint loader (#155 Stage 1) — Unturtle ADAPTATION.

The oracle's DiT backbone is a `huggingface_hub.PyTorchModelHubMixin`, and
the official HF repos (`david3684/FLM-B-OWT` / `FMLM-B-OWT`) carry exactly
its `config.json` (full hydra config) + `model.safetensors`.  Loading goes
through the VERBATIM `DIT.from_pretrained` — no key remapping of ours can
drift.  Frozen behaviors (Stage-0):

- revisions pinned to the frozen commits;
- the wrapper marker matches the checkpoint's algo (`flm` -> denoiser,
  `fmlm` -> flow map with double time conditioning) — loading an FLM
  checkpoint can never produce an `is_fmlm_flow_map` model;
- weights are the HF export used AS-IS (the export's relationship to the
  Lightning .ckpt EMA is recorded, not assumed — see the Stage-1 audit);
- dtype preserved as stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FLM_CHECKPOINT = "david3684/FLM-B-OWT"
FLM_REVISION = "624471b934fdd0421757d62290f7e639f32566d3"
FMLM_CHECKPOINT = "david3684/FMLM-B-OWT"
FMLM_REVISION = "483ea1b38bba56632cd40dc5a3c70a2340bb4946"


@dataclass(frozen=True)
class FlmCheckpointInfo:
    repo_id: str
    revision: str
    algo_name: str
    model_name: str


def _load(repo_id: str, revision: str, device: str) -> Any:
    from unturtle_flm._reference.dit import DIT
    from unturtle_flm.model import FlmInferenceModel

    backbone = DIT.from_pretrained(repo_id, revision=revision)
    backbone.eval().to(device)
    config = backbone.config
    algo_name = str(config.algo.name)
    model = FlmInferenceModel(
        backbone,
        vocab_size=int(backbone.vocab_size),
        length=int(config.model.length),
    )
    model.eval().to(device)
    model.flm_config = config
    model.flm_checkpoint = FlmCheckpointInfo(
        repo_id=repo_id,
        revision=revision,
        algo_name=algo_name,
        model_name=str(config.model.name),
    )
    return model, algo_name


def load_flm_model(
    *,
    checkpoint: str = FLM_CHECKPOINT,
    revision: str = FLM_REVISION,
    device: str = "cpu",
) -> Any:
    """The multi-step FLM denoiser (one-time conditioning)."""
    model, algo_name = _load(checkpoint, revision, device)
    # STRUCTURAL contract check, not a name check: the real FLM-B-OWT HF
    # config carries the historical algo.name 'dos' (Stage-0 correction #1),
    # so only the presence of the double time embedding decides.
    if getattr(model.backbone, "sigma_map_prime", None) is not None:
        raise ValueError(
            f"checkpoint {checkpoint!r} is a flow map (algo={algo_name!r}, "
            "double time conditioning present) — load it with "
            "load_fmlm_model() instead; the two contracts are not "
            "interchangeable (#155)"
        )
    model.is_flm_denoiser = True
    return model


def load_fmlm_model(
    *,
    checkpoint: str = FMLM_CHECKPOINT,
    revision: str = FMLM_REVISION,
    device: str = "cpu",
) -> Any:
    """The distilled flow map (double time conditioning, one/few-step)."""
    model, algo_name = _load(checkpoint, revision, device)
    if getattr(model.backbone, "sigma_map_prime", None) is None:
        raise ValueError(
            f"checkpoint {checkpoint!r} has no double time conditioning "
            f"(algo={algo_name!r}) — it is not a flow map; load it with "
            "load_flm_model() instead (#155)"
        )
    model.is_fmlm_flow_map = True
    return model


__all__ = [
    "FLM_CHECKPOINT",
    "FLM_REVISION",
    "FMLM_CHECKPOINT",
    "FMLM_REVISION",
    "FlmCheckpointInfo",
    "load_flm_model",
    "load_fmlm_model",
]
