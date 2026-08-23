# Ported VERBATIM (imports only adjusted) from the official ELF PyTorch
# reference — https://github.com/lillian039/ELF branch pytorch_elf
# @ b29d8833609e9ab7f67cd9da39435ac5cea04837 (MIT License, (c) 2026 ELF
# authors).  #154 parity oracle: DO NOT edit computation here.
import torch
import numpy as np


@torch.no_grad()
def encode_text(
    input_ids,
    attention_mask,
    encoder,
    latent_mean,
    latent_std,
    use_bf16=True,
):
    """Encoder pass from text to latent with normalization."""
    autocast_enabled = bool(use_bf16) and input_ids.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
        latents = encoder(input_ids=input_ids, attention_mask=attention_mask, deterministic=True)
    return (latents - latent_mean) / latent_std


def build_self_attn_cond_masks(is_cond, is_valid, xp=np):
    """Build self-attention conditioning masks from cond/valid token flags."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
