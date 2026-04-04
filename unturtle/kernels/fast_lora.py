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

"""Triton-fused LoRA kernel extensions for dLLM models.

Provides ``apply_lora_qkv_with_bias``: the bias-aware variant of unsloth's
``apply_lora_qkv`` kernel needed for Dream's ``q/k/v_proj`` layers which have
``bias=True``.

Mathematics::

    Q = X @ Wq.T + bq + (X @ Aq.T) @ Bq.T * Sq
    K = X @ Wk.T + bk + (X @ Ak.T) @ Bk.T * Sk
    V = X @ Wv.T + bv + (X @ Av.T) @ Bv.T * Sv

The weight-update part is identical to ``LoRA_QKV``; this class adds the bias
addition in forward and the bias gradient (``dQ.sum(0)`` etc.) in backward.
"""

from __future__ import annotations

import torch

from unsloth.kernels.fast_lora import (
    LoRA_QKV,
    apply_lora_qkv,
    apply_lora_o,
    apply_lora_mlp_swiglu,
    _maybe_fake_quantize_activations,
    fast_dequantize,
    get_lora_parameters_bias,
    matmul_lora,
    torch_amp_custom_bwd,
    torch_amp_custom_fwd,
)

__all__ = [
    "LoRA_QKV",
    "LoRA_QKV_Bias",
    "apply_lora_qkv",
    "apply_lora_qkv_with_bias",
    "apply_lora_o",
    "apply_lora_mlp_swiglu",
]


class LoRA_QKV_Bias(torch.autograd.Function):
    """Fused QKV LoRA with bias support.

    Identical to :class:`unturtle.kernels._fast_lora_core.LoRA_QKV` except each
    projection's bias is added after the matmul (forward) and the bias
    gradients are accumulated in backward.

    Args order matches ``LoRA_QKV`` with three extra bias args appended:
    ``Qbias, Kbias, Vbias``.
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(
        ctx,
        X: torch.Tensor,
        QW,
        QW_quant,
        QA,
        QB,
        QS,
        QBias,
        KW,
        KW_quant,
        KA,
        KB,
        KS,
        KBias,
        VW,
        VW_quant,
        VA,
        VB,
        VS,
        VBias,
        inplace: bool = True,
    ):
        orig_shape = X.shape
        X_for_matmul = X.view(-1, X.shape[-1]) if X.dim() == 3 else X

        Q = matmul_lora(X_for_matmul, QW, QW_quant, QA, QB, QS)
        K = matmul_lora(X_for_matmul, KW, KW_quant, KA, KB, KS)
        V = matmul_lora(X_for_matmul, VW, VW_quant, VA, VB, VS)

        if len(orig_shape) == 3:
            Q = Q.view(orig_shape[0], orig_shape[1], -1)
            K = K.view(orig_shape[0], orig_shape[1], -1)
            V = V.view(orig_shape[0], orig_shape[1], -1)

        # Add biases (broadcast over batch and sequence dimensions)
        if QBias is not None:
            Q = Q + QBias
        if KBias is not None:
            K = K + KBias
        if VBias is not None:
            V = V + VBias

        ctx.custom_saved_tensors = (
            QW, QW_quant, QS,
            KW, KW_quant, KS,
            VW, VW_quant, VS,
        )
        ctx.save_for_backward(X, QA, QB, KA, KB, VA, VB)
        ctx.inplace = inplace
        ctx.has_bias = (QBias is not None, KBias is not None, VBias is not None)
        return Q, K, V

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dQ, dK, dV):
        QW, QW_quant, QS, KW, KW_quant, KS, VW, VW_quant, VS = ctx.custom_saved_tensors
        X, QA, QB, KA, KB, VA, VB = ctx.saved_tensors
        has_Qb, has_Kb, has_Vb = ctx.has_bias

        batch, seq_len, hd = X.shape
        dQ = dQ.reshape(-1, dQ.shape[-1])
        dK = dK.reshape(-1, dK.shape[-1])
        dV = dV.reshape(-1, dV.shape[-1])
        X = X.reshape(-1, X.shape[-1])
        dtype = X.dtype

        QA, QB = QA.to(dtype).t(), QB.to(dtype).t()
        KA, KB = KA.to(dtype).t(), KB.to(dtype).t()
        VA, VB = VA.to(dtype).t(), VB.to(dtype).t()

        d_QA = torch.empty_like(QA)
        d_QB = torch.empty_like(QB)
        d_KA = torch.empty_like(KA)
        d_KB = torch.empty_like(KB)
        d_VA = torch.empty_like(VA)
        d_VB = torch.empty_like(VB)

        d_QA.addmm_(X.t(), dQ @ QB.t(), alpha=QS, beta=0)
        d_QB.addmm_(QA.t() @ X.t(), dQ, alpha=QS, beta=0)

        d_KA.addmm_(X.t(), dK @ KB.t(), alpha=KS, beta=0)
        d_KB.addmm_(KA.t() @ X.t(), dK, alpha=KS, beta=0)

        d_VA.addmm_(X.t(), dV @ VB.t(), alpha=VS, beta=0)
        d_VB.addmm_(VA.t() @ X.t(), dV, alpha=VS, beta=0)

        # dX
        QW = fast_dequantize(QW.t(), QW_quant)
        dX = torch.matmul(dQ, QW.t(), out=X if ctx.inplace else None)
        del QW
        dX.addmm_(dQ @ QB.t(), QA.t(), alpha=QS)

        KW = fast_dequantize(KW.t(), KW_quant)
        dX.addmm_(dK, KW.t())
        del KW
        dX.addmm_(dK @ KB.t(), KA.t(), alpha=KS)

        VW = fast_dequantize(VW.t(), VW_quant)
        dX.addmm_(dV, VW.t())
        del VW
        dX.addmm_(dV @ VB.t(), VA.t(), alpha=VS)

        # Bias gradients: sum over batch*seq dims
        d_QBias = dQ.sum(0) if has_Qb else None
        d_KBias = dK.sum(0) if has_Kb else None
        d_VBias = dV.sum(0) if has_Vb else None

        # X, QW, QW_quant, QA, QB, QS, QBias,
        # KW, KW_quant, KA, KB, KS, KBias,
        # VW, VW_quant, VA, VB, VS, VBias, inplace
        return (
            dX.view(batch, seq_len, hd),
            None, None, d_QA.t(), d_QB.t(), None, d_QBias,
            None, None, d_KA.t(), d_KB.t(), None, d_KBias,
            None, None, d_VA.t(), d_VB.t(), None, d_VBias,
            None,
        )


def apply_lora_qkv_with_bias(self, X, inplace: bool = True):
    """Bias-aware drop-in for ``apply_lora_qkv``.

    Use this as ``self_attn.apply_qkv`` for models where ``q/k/v_proj``
    have ``bias=True`` (e.g. Dream).

    Reads parameters via ``get_lora_parameters_bias`` which returns a
    6-tuple ``(W, W_quant, A, B, S, bias)``.
    """
    X = _maybe_fake_quantize_activations(X, self.q_proj)
    QW, QW_quant, QA, QB, QS, QBias = get_lora_parameters_bias(self.q_proj)
    KW, KW_quant, KA, KB, KS, KBias = get_lora_parameters_bias(self.k_proj)
    VW, VW_quant, VA, VB, VS, VBias = get_lora_parameters_bias(self.v_proj)
    Q, K, V = LoRA_QKV_Bias.apply(
        X,
        QW, QW_quant, QA, QB, QS, QBias,
        KW, KW_quant, KA, KB, KS, KBias,
        VW, VW_quant, VA, VB, VS, VBias,
        inplace,
    )
    return Q, K, V
