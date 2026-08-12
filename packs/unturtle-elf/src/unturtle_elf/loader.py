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

"""ELF checkpoint loader (#153 Stage 1) — Unturtle ADAPTATION, not reference.

Loads the official converted PyTorch checkpoint
(`embedded-language-flows/ELF-B-owt-torch`) into the verbatim-ported
reference model.  Frozen behaviors (Stage-0):

- **EMA weights** (`ema_params1`) are the evaluation parameters; a
  checkpoint without them falls back to `params` LOUDLY (warning recorded
  on the model), matching `checkpoint_utils._restore` semantics;
- key coverage is STRICT: unexpected or missing state-dict keys raise —
  never silently dropped;
- dtype is preserved as stored; any cast is the caller's explicit choice;
- the model is marked `is_elf_denoiser = True` (the pack's supports probe)
  and carries `elf_config` + `elf_checkpoint` provenance for records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_CHECKPOINT = "embedded-language-flows/ELF-B-owt-torch"
DEFAULT_REVISION = "146f84133c1389bfd4ef47f14ec7a955da22faa7"
_CHECKPOINT_FILE = "checkpoint_95085"
_CONFIG_FILE = "config.yml"

_MODEL_FACTORIES = {"ELF-B": "ELF_B", "ELF-M": "ELF_M", "ELF-L": "ELF_L"}

_ENCODER_DIMS = {"t5-small": 512, "t5-base": 768, "t5-large": 1024}


@dataclass(frozen=True)
class ElfCheckpointInfo:
    """Provenance riding with the loaded model (into #152 records)."""

    repo_id: str
    revision: str
    checkpoint_file: str
    model_name: str
    used_ema: bool


def load_elf_model(
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    revision: str = DEFAULT_REVISION,
    device: str = "cpu",
) -> Any:
    """Build the reference ELF model and load the official checkpoint.

    Heavyweight (downloads ~840MB on first use); returns the eval-mode
    reference `ELF` module with the pack marker and provenance attached.
    """
    import yaml
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(checkpoint, _CONFIG_FILE, revision=revision)
    with open(config_path) as handle:
        raw_config = yaml.safe_load(handle)

    checkpoint_path = hf_hub_download(checkpoint, _CHECKPOINT_FILE, revision=revision)
    return load_elf_model_from_files(
        checkpoint_path,
        raw_config,
        device=device,
        provenance=ElfCheckpointInfo(
            repo_id=checkpoint,
            revision=revision,
            checkpoint_file=_CHECKPOINT_FILE,
            model_name=str(raw_config["model"]),
            used_ema=True,  # corrected below if the fallback fires
        ),
    )


def build_elf_model(raw_config: dict[str, Any]) -> Any:
    """Construct the verbatim reference architecture from a checkpoint
    config dict (no weights) — shared by the loader and the parity tests."""
    from unturtle_elf._reference import model as reference_model

    model_name = str(raw_config["model"])
    if model_name not in _MODEL_FACTORIES:
        raise ValueError(
            f"unknown ELF model {model_name!r}; known: {sorted(_MODEL_FACTORIES)}"
        )
    encoder_name = str(raw_config.get("encoder_model_name", "t5-small"))
    if encoder_name not in _ENCODER_DIMS:
        raise ValueError(
            f"unknown encoder {encoder_name!r}; known: {sorted(_ENCODER_DIMS)}"
        )
    factory = getattr(reference_model, _MODEL_FACTORIES[model_name])
    return factory(
        text_encoder_dim=_ENCODER_DIMS[encoder_name],
        max_length=int(raw_config["max_length"]),
        bottleneck_dim=int(raw_config.get("bottleneck_dim", 128)),
        num_time_tokens=int(raw_config.get("num_time_tokens", 4)),
        num_self_cond_cfg_tokens=int(raw_config.get("num_self_cond_cfg_tokens", 4)),
        num_model_mode_tokens=int(raw_config.get("num_model_mode_tokens", 0)),
        vocab_size=32128,  # t5 tokenizer family (Stage-0: t5-small vocab)
    )


def load_elf_model_from_files(
    checkpoint_path: str,
    raw_config: dict[str, Any],
    *,
    device: str = "cpu",
    provenance: ElfCheckpointInfo | None = None,
) -> Any:
    """Load from an already-downloaded checkpoint file (test seam)."""
    import torch

    model = build_elf_model(raw_config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "params" not in payload:
        raise ValueError(
            f"checkpoint {checkpoint_path!r} has no 'params' entry; found "
            f"keys {sorted(payload)} — refusing to guess the layout"
        )
    used_ema = "ema_params1" in payload and bool(payload["ema_params1"])
    state = payload["ema_params1"] if used_ema else payload["params"]
    # STRICT: unexpected/missing keys are loud (issue mutation target:
    # "wrong checkpoint key silently dropped").
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    model.is_elf_denoiser = True
    model.elf_config = dict(raw_config)
    if provenance is not None and provenance.used_ema != used_ema:
        provenance = ElfCheckpointInfo(
            repo_id=provenance.repo_id,
            revision=provenance.revision,
            checkpoint_file=provenance.checkpoint_file,
            model_name=provenance.model_name,
            used_ema=used_ema,
        )
    model.elf_checkpoint = provenance
    if not used_ema:
        import warnings

        warnings.warn(
            "ELF checkpoint carries no EMA params; falling back to raw "
            "'params' — evaluation numbers will NOT match the official "
            "EMA-based reference.",
            stacklevel=2,
        )
    return model


__all__ = [
    "DEFAULT_CHECKPOINT",
    "DEFAULT_REVISION",
    "ElfCheckpointInfo",
    "build_elf_model",
    "load_elf_model",
    "load_elf_model_from_files",
]
