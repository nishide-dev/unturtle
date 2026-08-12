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

"""ELF sampling entry (#153 Stage 2) — Unturtle ADAPTATION over the verbatim
reference rollout.

The actual math lives in `_reference.generation_utils` /
`_reference.sampling_utils` (ported operation-for-operation).  This module
only translates an Unturtle `GenerationRequest` into the reference call and
returns the executed-configuration record alongside the tokens.

Frozen semantics honored here (Stage-0):

- the requested solver ("sde"/"ode") is the solver EXECUTED — γ=0 makes an
  SDE step an ODE step mathematically, but the request's solver name is
  recorded from what actually ran, never echoed from the input;
- executed step count = len(t_grid) - 1 = the number of denoiser forwards
  (NFE; in-context SC-CFG adds no extra forward at cfg=1);
- RNG scope is ONE CALL: the reference CUDA paths draw from global torch
  RNG (oracle behavior, Stage-0 caveat), so this function seeds the global
  stream from the request's seed — inside torch.random.fork_rng, so a
  caller's RNG state is never polluted (#160 review F5).  A multi-batch
  evaluation cell therefore runs N derived per-call seeds, a DISCLOSED
  deviation from the oracle script's single sequential stream (#160 F4);
- the logit-normal time grid uses the CHECKPOINT's denoiser_p_mean/p_std
  (generation.py:151), never the function defaults (#160 F1 — the trap the
  Stage-0 freeze flagged); the executed grid is recorded verbatim;
- everything after the first EOS is masked to pad BEFORE returning, exactly
  as the oracle does before decoding (generation.py:184; #160 F2);
- discretization happens ONLY at the endpoint via the reference
  `_dlm_decode_batch` (decoder head + argmax);
- unconditional OWT: cfg=1, SC-CFG=3, γ=1.5@32 / 1.0@64, logit-normal grid.
"""

from __future__ import annotations

from typing import Any


def _sampling_config(reference_config_module, kwargs: dict[str, Any]):
    config = reference_config_module.SamplingConfig()
    config.sampling_method = str(kwargs.get("solver", "sde"))
    if config.sampling_method not in ("sde", "ode"):
        raise ValueError(
            f"unknown ELF solver {config.sampling_method!r}; expected 'sde' or 'ode'"
        )
    config.num_sampling_steps = [int(kwargs.get("steps", 32))]
    config.cfgs = [float(kwargs.get("cfg_scale", 1.0))]
    config.self_cond_cfg_scales = [float(kwargs.get("self_cond_cfg_scale", 3.0))]
    config.time_schedule = str(kwargs.get("time_schedule", "logit_normal"))
    config.sde_gamma = float(kwargs.get("sde_gamma", 1.5))
    return config


def run_generation_request(model: Any, request: Any) -> dict[str, Any]:
    """Sample token ids from an ELF denoiser for an Unturtle request.

    Recognized ``request.kwargs`` (all reference-native, none universal):
    ``solver`` ("sde"|"ode"), ``steps``, ``sde_gamma``,
    ``self_cond_cfg_scale``, ``cfg_scale``, ``time_schedule``, ``seed``,
    ``num_samples``.  Returns tokens plus the EXECUTED configuration.
    """
    import torch

    from unturtle_elf._reference import config as reference_config
    from unturtle_elf._reference.generation_utils import (
        _dlm_decode_batch,
        _generate_samples_single_batch,
        mask_after_eos,
    )
    from unturtle_elf._reference.sampling_utils import get_sampling_steps

    if not getattr(model, "is_elf_denoiser", False):
        raise ValueError(f"{type(model).__name__} is not a pack-loaded ELF denoiser")

    kwargs = dict(getattr(request, "kwargs", None) or {})
    sampling = _sampling_config(reference_config, kwargs)
    raw = model.elf_config

    config = reference_config.Config()
    for key, value in raw.items():
        setattr(config, key, value)

    seed = int(kwargs.get("seed", 42))
    num_samples = int(kwargs.get("num_samples", 1))
    steps = sampling.num_sampling_steps[0]
    # T5 family: eos=1, pad=0 (overridable for other tokenizers).
    eos_token_id = int(kwargs.get("eos_token_id", 1))
    pad_token_id = int(kwargs.get("pad_token_id", 0))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        # The reference CUDA branches draw from the global stream (oracle
        # behavior, Stage-0 caveat), so the seed is applied there too; the
        # fork keeps that from polluting the caller's RNG state (#160 F5).
        generator = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)

        t_steps = get_sampling_steps(
            steps,
            time_schedule=sampling.time_schedule,
            # The CHECKPOINT's schedule parameters, never the function
            # defaults: ELF-B has denoiser_p_mean=-1.5 vs the -0.8 default,
            # ~5x fewer low-t points (#160 F1; oracle generation.py:151).
            P_mean=float(getattr(config, "denoiser_p_mean", -0.8)),
            P_std=float(getattr(config, "denoiser_p_std", 0.8)),
            device=device,
            dtype=dtype,
        )

        d_model = model.text_encoder_dim
        if device.type == "cuda":
            z = (
                torch.randn(
                    (num_samples, config.max_length, d_model),
                    dtype=dtype,
                    device=device,
                )
                * config.denoiser_noise_scale
            )
        else:
            z = (
                torch.randn(
                    (num_samples, config.max_length, d_model),
                    generator=generator,
                    dtype=dtype,
                )
                * config.denoiser_noise_scale
            ).to(device)

        latent = _generate_samples_single_batch(
            model=model,
            generator=generator,
            z=z,
            t_steps=t_steps,
            cond_seq=None,
            cond_seq_mask=None,
            config=config,
            sampling_config=sampling,
            cfg_scale=sampling.cfgs[0],
            self_cond_cfg_scale=sampling.self_cond_cfg_scales[0],
        )
        tokens = _dlm_decode_batch(
            latent,
            model,
            t_steps[-1],
            config,
            self_cond_cfg_scale=sampling.self_cond_cfg_scales[0],
        )
        # Oracle post-processing (generation.py:184): everything after the
        # first EOS is pad — unmasked ids leak junk into evaluation (#160 F2).
        tokens = mask_after_eos(
            tokens, eos_token_id=eos_token_id, pad_token_id=pad_token_id
        )

    executed_steps = int(t_steps.shape[0]) - 1
    return {
        "method": "elf",
        "tokens": tokens,
        "executed": {
            "solver": sampling.sampling_method,
            "steps_requested": steps,
            "steps_executed": executed_steps,
            "nfe": executed_steps,
            "t_grid": [float(value) for value in t_steps],
            "sde_gamma": sampling.sde_gamma,
            "self_cond_cfg_scale": sampling.self_cond_cfg_scales[0],
            "cfg_scale": sampling.cfgs[0],
            "time_schedule": sampling.time_schedule,
            "schedule_p_mean": float(getattr(config, "denoiser_p_mean", -0.8)),
            "schedule_p_std": float(getattr(config, "denoiser_p_std", 0.8)),
            "seed": seed,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "max_length": int(config.max_length),
        },
    }


__all__ = ["run_generation_request"]
