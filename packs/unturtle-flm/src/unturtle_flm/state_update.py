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

"""FMLM flow-map state update, with a scope-guarded fast path (#166 Stage 2).

The reference expression is

    z_tilde = weight_z * z + weight_d * D
    z       = z_tilde + mean_adjustment * D + noise_std * eps

which dispatches SEVEN full-size ATen ops, each materializing a temporary. The
fast path computes the SAME RESULT with one `mul` and three `addcmul`, reducing
how many intermediates reach global memory. Measured locally at 1.56-1.61x on
[B, 1024, 50258] fp32; the end-to-end effect is a separate question answered by
the paired benchmark, not by this module.

WHY BIT IDENTITY, NOT CLOSENESS
-------------------------------
An algebraically equivalent but reassociated form of this update was measured at
2.4x and REJECTED. Under teacher forcing its per-step error was only ~1 fp32 ULP
— the algebra was correct — but the sampler feeds each state back into the model,
and that error amplified across 31 steps: 1.19e-07 at step 0, 3.97e-02 at step 1,
8.61e-01 at step 30, ending in 476/1024 changed endpoint tokens. There is no
usable tolerance band here. A candidate is either bit-identical or it changes
what the model generates.

SCOPE
-----
Bit identity was MEASURED under one configuration and is claimed for no other:

    CUDA device, contiguous inputs, autocast disabled, eager execution
    float32 accumulator (`z`, weights, `eps`); `d_pred` float32 OR bfloat16
    torch 2.10.0+cu128 / CUDA 12.8 / NVIDIA RTX 6000 Ada Generation

The split between accumulator and prediction dtype is not cosmetic. An earlier
version of this guard demanded float32 everywhere; because the model emits
`d_pred` in bfloat16, it rejected all 31 calls of every real request and the
specialization was silently inert — a paired outer-wall benchmark measured 0.0%
before the cause was found. Identity in the mixed configuration was then
measured over 144 combinations of seed, shape and weight extremes.

`torch.addcmul` is NOT universally equivalent to a separate mul followed by an
add. On CPU it contracts to an FMA, leaving the product unrounded, and the result
differs from the reference by ~2.4e-07 — enough to amplify. The guards below are
therefore a correctness mechanism, not an optimization heuristic: anything
outside the measured scope executes the reference sequence.

Evidence: `docs/artifacts/166-fmlm-state-update-agreement.json`.
"""

from __future__ import annotations

from typing import Any

import torch

#: The accumulator dtype. `z`, the weights and `eps` set the precision the
#: additions are carried out in, and identity was measured in float32 only.
_ACCUMULATOR_DTYPE = torch.float32

#: `d_pred` is only ever multiplied INTO the fp32 accumulator, never accumulated
#: in its own precision, so it may also arrive in bfloat16 — which is what the
#: model actually emits. Identity was measured for both; float16 was not.
_PREDICTION_DTYPES = (torch.float32, torch.bfloat16)


def fast_path_applies(
    z: Any,
    d_pred: Any,
    weight_z: Any,
    weight_d: Any,
    mean_adjustment: Any,
    noise_std: Any,
    eps: Any,
) -> bool:
    """Whether the measured-scope conditions hold for these inputs.

    Deliberately conservative: a false negative costs a little speed, while a
    false positive changes generated tokens.
    """
    tensors = (z, d_pred, weight_z, weight_d, mean_adjustment, noise_std, eps)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    if any(t.device.type != "cuda" for t in tensors):
        return False
    # One device: a cross-device set would not have been measured together, and
    # `.device.type` alone would not catch cuda:0 mixed with cuda:1.
    if len({t.device for t in tensors}) != 1:
        return False
    accumulator = (z, weight_z, weight_d, mean_adjustment, noise_std, eps)
    if any(t.dtype is not _ACCUMULATOR_DTYPE for t in accumulator):
        return False
    if d_pred.dtype not in _PREDICTION_DTYPES:
        return False
    if any(not t.is_contiguous() for t in tensors):
        return False
    # Autocast changes the dtype the kernels actually execute in, which is
    # outside the configuration where identity was measured.
    return not torch.is_autocast_enabled("cuda")


def _reference_update(z, d_pred, weight_z, weight_d, mean_adjustment, noise_std, eps):
    """The original sequence, unchanged. This is what runs outside the measured
    scope, and it remains the definition of correct behaviour."""
    z_tilde = weight_z * z + weight_d * d_pred
    return z_tilde + mean_adjustment * d_pred + noise_std * eps


def _fast_update(z, d_pred, weight_z, weight_d, mean_adjustment, noise_std, eps):
    """Same result, fewer full-size materializations.

    `addcmul(base, a, b)` computes `base + a * b` in one pass. Under the scope
    above this rounds identically to a separate mul and add — VERIFIED against
    the reference over full 32-step rollouts at batch 1, 8 and 32, plus five
    additional seeds, with matching endpoint tokens and RNG state.
    """
    z_tilde = torch.addcmul(weight_z * z, d_pred, weight_d)
    return torch.addcmul(
        torch.addcmul(z_tilde, d_pred, mean_adjustment), eps, noise_std
    )


def apply_state_update(
    z: Any,
    d_pred: Any,
    weight_z: Any,
    weight_d: Any,
    mean_adjustment: Any,
    noise_std: Any,
    eps: Any,
) -> Any:
    """Compute the next latent, taking the fast path only inside the measured
    scope.

    Mutates nothing the caller owns and consumes no randomness: `eps` is
    supplied, so the RNG stream advances exactly as the reference's does.
    """
    chosen = (
        _fast_update
        if fast_path_applies(
            z, d_pred, weight_z, weight_d, mean_adjustment, noise_std, eps
        )
        else _reference_update
    )
    return chosen(z, d_pred, weight_z, weight_d, mean_adjustment, noise_std, eps)


__all__ = ["apply_state_update", "fast_path_applies"]
