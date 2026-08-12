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

"""ELF training objective (#154 Stage 1) — Unturtle ADAPTATION, line-cited
mirror of the official ``train_step.py`` LOSS computation (commit b29d8833).

Separated from backward/optimizer so the objective is testable against the
oracle differentially (identical loss AND gradients under frozen RNG) and
analytically (minimizer tests).  The RNG DRAW ORDER matches the oracle
exactly — that is load-bearing for the differential and is itself frozen
reference behavior:

  1. t             (global randn, sample_timesteps)     train_step.py:78
  2. noise         (global randn)                       :86
  3. decoder_step_active (dropout_generator bernoulli)  :106
  4. decoder_z_vals (global randn)                      :111
  5. decoder_noise (global randn)                       :116
  6. use_self_cond_mask (global rand)                   :123
  7. self_cond_cfg_scale (global rand)                  :131

Scope (per the Stage-0 training freeze): unconditional OWT —
``label_drop_prob=0`` and conditional-sequence machinery are inactive; the
cond-mask plumbing is kept only where it shapes the loss mask.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def encode_text(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    encoder: Any,
    latent_mean: float,
    latent_std: float,
) -> torch.Tensor:
    """Frozen-encoder latents with normalization (encoder_utils.py:6-19;
    the bf16-autocast wrapper is a GPU concern handled by the caller)."""
    latents = encoder(
        input_ids=input_ids, attention_mask=attention_mask, deterministic=True
    )
    return (latents - latent_mean) / latent_std


def elf_training_loss(
    model: Any,
    encoder: Any,
    batch: dict[str, torch.Tensor],
    config: Any,
    *,
    dropout_generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """The ELF objective for one batch — train_step.py:31-261 minus
    backward/optimizer.  Returns ``(loss, metrics, aux)`` where ``aux``
    exposes the tensors the Stage-1 tests pin (t, detached v target,
    per-token losses, masks)."""
    from unturtle_elf._reference.sampling_utils import (
        add_noise,
        net_out_to_v_x,
        restore_cond,
        sample_cfg_scale,
        sample_timesteps,
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob

    input_ids = batch["input_ids"].to(device).long()
    encoder_attention_mask = batch["encoder_attention_mask"].to(
        device, dtype=torch.float32
    )
    cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.float32)
    assert getattr(config, "label_drop_prob", 0.0) == 0.0, (
        "conditional label-drop is out of the frozen #154 scope"
    )

    x0 = encode_text(
        input_ids=input_ids,
        attention_mask=encoder_attention_mask,
        encoder=encoder,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
    ).to(dtype)

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    # train_step.py:76-83
    t = sample_timesteps(
        batch_size,
        P_mean=config.denoiser_p_mean,
        P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device,
        dtype=dtype,
    )
    # train_step.py:86
    noise = torch.randn(x0.shape, dtype=dtype, device=device)

    # train_step.py:88-92
    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - cond_seq_mask)

    cond_seq_mask = cond_seq_mask.unsqueeze(-1)
    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)

    decoder_targets = input_ids

    # train_step.py:104-109 — per-example CE/L2 branch draw on the
    # dedicated generator.
    decoder_step_active = torch.bernoulli(
        torch.full((batch_size,), config.decoder_prob, dtype=torch.float32),
        generator=dropout_generator,
    ).to(device=device, dtype=dtype)
    decoder_mask_B11 = decoder_step_active.view(-1, 1, 1)
    decoder_mask_B1 = decoder_step_active.view(-1, 1)

    # train_step.py:110-117 — decoder-branch logit-normal-noised latent.
    decoder_z_vals = (
        torch.randn((batch_size * seq_length,), dtype=dtype, device=device)
        * config.decoder_p_std
        + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = (
        torch.randn(x0.shape, dtype=dtype, device=device) * config.decoder_noise_scale
    )
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    # train_step.py:119-120 — the v target.
    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_expanded, min=t_eps)

    # train_step.py:122-133
    if self_cond_prob > 0:
        use_self_cond_mask = (
            (torch.rand((batch_size,), dtype=dtype, device=device) < self_cond_prob)
            .reshape(-1, 1, 1)
            .to(dtype)
        )
    else:
        use_self_cond_mask = None
    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            batch_size,
            cfg_min=config.self_cond_cfg_min,
            cfg_max=config.self_cond_cfg_max,
            dtype=dtype,
            device=device,
        )
    else:
        self_cond_cfg_scale = None

    def compute_shared_uncond(z, t_input, x_tokens):
        # train_step.py:143-153
        z_uncond = restore_cond(torch.zeros_like(z), x_tokens, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        with torch.no_grad():
            return model(
                z_input_uncond,
                t_input,
                deterministic=True,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )

    def get_sc_cond_and_uncond(z, t_input, cond_mask, x_tokens, shared_net_out_uncond):
        # train_step.py:155-175
        if config.self_cond_prob == 0:
            with torch.no_grad():
                net_out_uncond = model(
                    z,
                    t_input,
                    deterministic=True,
                    self_cond_cfg_scale=self_cond_cfg_scale,
                )
            v_uncond, _ = net_out_to_v_x(net_out_uncond, z, t_input, t_eps)
            return v_uncond, v_uncond
        v_uncond, x_uncond = net_out_to_v_x(shared_net_out_uncond, z, t_input, t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)
        z_input_cond = torch.cat([z, x_uncond], dim=-1)
        with torch.no_grad():
            net_out_cond = model(
                z_input_cond,
                t_input,
                deterministic=True,
                self_cond_cfg_scale=self_cond_cfg_scale,
            )
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t_input, t_eps)
        return v_cond, v_uncond

    def get_v_target(z, t_input, base_v_target, x_tokens, shared_net_out_uncond):
        # train_step.py:177-196 — SC-CFG guidance applied to the TARGET,
        # detached.
        if config.num_self_cond_cfg_tokens > 0 and config.self_cond_prob > 0:
            v_cond, v_uncond = get_sc_cond_and_uncond(
                z,
                t_input,
                cond_mask=cond_seq_mask,
                x_tokens=x_tokens,
                shared_net_out_uncond=shared_net_out_uncond,
            )
            sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
            sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
            sc_guidance = torch.where(
                use_self_cond_mask.bool(),
                sc_guidance,
                torch.zeros_like(sc_guidance),
            )
            return (base_v_target + sc_guidance).detach()
        return base_v_target

    model.train()

    # train_step.py:200-206 — mixed input, one forward for both heads.
    denoiser_t = t
    t_mixed = decoder_step_active * torch.ones_like(t) + (1.0 - decoder_step_active) * t
    z_mixed = decoder_mask_B11 * decoder_z + (1.0 - decoder_mask_B11) * denoiser_z

    if self_cond_prob > 0 or config.num_self_cond_cfg_tokens > 0:
        shared_net_out_uncond = compute_shared_uncond(denoiser_z, denoiser_t, x0)
    else:
        shared_net_out_uncond = None

    if config.self_cond_prob > 0:
        # train_step.py:213-221
        _, x_pred_init = net_out_to_v_x(
            shared_net_out_uncond, denoiser_z, denoiser_t, t_eps
        )
        x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask.to(dtype)
        x_pred_cond = restore_cond(x_pred_cond, x0, cond_seq_mask)
        sc_half = x_pred_cond * (1.0 - decoder_mask_B11)
        model_input = torch.cat([z_mixed, sc_half], dim=-1)
    else:
        model_input = z_mixed

    # train_step.py:225-230 — the single grad-enabled forward.
    net_out, decoder_logits = model(
        model_input,
        t_mixed,
        deterministic=False,
        self_cond_cfg_scale=self_cond_cfg_scale,
        decoder_step_active=decoder_step_active,
    )

    # train_step.py:232-245
    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, decoder_targets.unsqueeze(-1)).squeeze(-1)

    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
    v_final_target = get_v_target(
        denoiser_z,
        denoiser_t,
        base_v_target=v_target,
        x_tokens=x0,
        shared_net_out_uncond=shared_net_out_uncond,
    )
    l2_per_token = ((v_pred - v_final_target) ** 2).mean(dim=-1)

    # train_step.py:247-261 — masks and the single denominator.
    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_mask_B1
    l2_mask = loss_mask_f * (1.0 - decoder_mask_B1)
    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)

    metrics = {
        "loss": loss.detach(),
        "ce_loss": (
            (ce_per_token * ce_mask).sum() / torch.clamp(ce_mask.sum(), min=1.0)
        ).detach(),
        "l2_loss": (
            (l2_per_token * l2_mask).sum() / torch.clamp(l2_mask.sum(), min=1.0)
        ).detach(),
    }
    aux = {
        "t": t,
        "v_final_target": v_final_target,
        "ce_per_token": ce_per_token.detach(),
        "l2_per_token": l2_per_token.detach(),
        "ce_mask": ce_mask,
        "l2_mask": l2_mask,
        "loss_mask": loss_mask_f,
    }
    return loss, metrics, aux


def init_ema(model: Any) -> dict[str, torch.Tensor]:
    """EMA shadow of the trainable parameters (train_utils.TrainState)."""
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def ema_update(ema_state: dict[str, torch.Tensor], model: Any, decay: float) -> None:
    """train_utils.py:137-142 verbatim logic: ema.lerp_(param, 1-decay)."""
    for name, param in model.named_parameters():
        if name in ema_state:
            ema_state[name].lerp_(param.detach(), 1.0 - decay)


def build_muon_optimizer(model: Any, lr: float) -> Any:
    """The official optimizer, via the VERBATIM-ported `muon_with_aux_adam`.

    Stage-2 correction #1: an earlier adaptation called bare upstream
    `SingleDeviceMuonWithAuxAdam`, which is a DIFFERENT optimizer — the
    oracle layers four patches (Nesterov-Adam aux update, fp32
    Newton-Schulz, Nesterov bias correction, layout-aware `sqrt(fan_out/
    fan_in)` scaling) plus a missing-grad safety wrapper the alternating
    CE/L2 branches require.  Distribution name is `muon-optimizer`
    (import name `muon`); the PyPI project literally called `muon` is an
    unrelated single-cell library.
    """
    from unturtle_elf._reference.muon_utils import muon_with_aux_adam

    return muon_with_aux_adam(model, lr=lr)


def muon_parameter_partition(model: Any) -> dict[str, list[str]]:
    """The partition, by NAME, from an immutable snapshot of the model's
    parameters — the #154 Stage-2 entry guard (ndim==2 -> Muon, else aux
    Adam), checkable without stepping the optimizer."""
    muon_names, adam_names = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (muon_names if param.ndim == 2 else adam_names).append(name)
    return {"muon": muon_names, "adam": adam_names}


__all__ = [
    "build_muon_optimizer",
    "elf_training_loss",
    "ema_update",
    "encode_text",
    "init_ema",
    "muon_parameter_partition",
]
