# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
#
# Vendored from unsloth/kernels/utils.py for issue #67.

from __future__ import annotations

import ctypes
import importlib

try:
    import bitsandbytes as bnb
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "unturtle.kernels.fast_lora requires bitsandbytes to be importable"
    ) from exc

import torch

from ._fp8 import fp8_linear, weight_dequant
from ._triton_utils import DEVICE_COUNT, torch_device_stream, torch_gpu_device


def _version_tuple(version: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in version.replace("+", ".").replace("-", ".").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            out.append(int(digits))
        else:
            break
    return tuple(out)


if _version_tuple(torch.__version__) < (2, 4, 0):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")


HAS_CUDA_STREAM = _version_tuple(bnb.__version__) > (0, 43, 3)
get_ptr = bnb.functional.get_ptr

c_void_p = ctypes.c_void_p
ctypes_c_int = ctypes.c_int
ctypes_c_int32 = ctypes.c_int32
cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4
cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemm_4bit_inference_naive_fp16
cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemm_4bit_inference_naive_bf16

torch_mm = torch.mm
torch_mv = torch.mv
torch_matmul = torch.matmul
torch_empty = torch.empty
torch_float32 = torch.float32
torch_float16 = torch.float16
torch_bfloat16 = torch.bfloat16

if importlib.util.find_spec("torchao") is not None:
    try:
        from torchao.quantization import Float8Tensor
    except Exception:
        Float8Tensor = type(None)
else:
    Float8Tensor = type(None)


def QUANT_STATE(W):
    return getattr(W, "quant_state", None)


def get_lora_parameters(proj):
    base_layer = getattr(proj, "base_layer", proj)
    W = base_layer.weight

    if hasattr(base_layer, "weight_fake_quantizer"):
        weight_fake_quantizer = getattr(base_layer, "weight_fake_quantizer", None)
        if weight_fake_quantizer is not None:
            W = weight_fake_quantizer(W)

    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    if getattr(base_layer, "quant_method", None) == "fp8":
        W.block_size = getattr(base_layer, "block_size", [128, 128])
        W_quant.block_size = W.block_size

    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]

    lora_A_linear = proj.lora_A[adapter]
    lora_B_linear = proj.lora_B[adapter]
    A = lora_A_linear.weight
    B = lora_B_linear.weight
    if hasattr(lora_A_linear, "weight_fake_quantizer"):
        lora_A_fake_quantizer = getattr(lora_A_linear, "weight_fake_quantizer", None)
        if lora_A_fake_quantizer is not None:
            A = lora_A_fake_quantizer(A)
    if hasattr(lora_B_linear, "weight_fake_quantizer"):
        lora_B_fake_quantizer = getattr(lora_B_linear, "weight_fake_quantizer", None)
        if lora_B_fake_quantizer is not None:
            B = lora_B_fake_quantizer(B)

    return W, W_quant, A, B, proj.scaling[adapter]


def get_lora_parameters_bias(proj):
    base_layer = getattr(proj, "base_layer", proj)
    W = base_layer.weight

    if hasattr(base_layer, "weight_fake_quantizer"):
        weight_fake_quantizer = getattr(base_layer, "weight_fake_quantizer", None)
        if weight_fake_quantizer is not None:
            W = weight_fake_quantizer(W)

    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    if getattr(base_layer, "quant_method", None) == "fp8":
        W.block_size = getattr(base_layer, "block_size", [128, 128])
        W_quant.block_size = W.block_size

    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None, base_layer.bias

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]

    lora_A_linear = proj.lora_A[adapter]
    lora_B_linear = proj.lora_B[adapter]
    A = lora_A_linear.weight
    B = lora_B_linear.weight
    if hasattr(lora_A_linear, "weight_fake_quantizer"):
        lora_A_fake_quantizer = getattr(lora_A_linear, "weight_fake_quantizer", None)
        if lora_A_fake_quantizer is not None:
            A = lora_A_fake_quantizer(A)
    if hasattr(lora_B_linear, "weight_fake_quantizer"):
        lora_B_fake_quantizer = getattr(lora_B_linear, "weight_fake_quantizer", None)
        if lora_B_fake_quantizer is not None:
            B = lora_B_fake_quantizer(B)

    return (
        W,
        W_quant,
        A,
        B,
        proj.scaling[adapter],
        base_layer.bias,
    )


def _maybe_fake_quantize_activations(
    X: torch.Tensor, proj: torch.nn.Module
) -> torch.Tensor:
    base_layer = getattr(proj, "base_layer", proj)
    activation_fake_quantizer = getattr(base_layer, "activation_fake_quantizer", None)
    if activation_fake_quantizer is not None:
        X = activation_fake_quantizer(X)
    return X


if DEVICE_COUNT > 0 and HAS_CUDA_STREAM:
    _CUDA_STREAMS = {
        (index := torch.cuda.device(i).idx): ctypes.c_void_p(
            torch._C._cuda_getCurrentRawStream(index)
        )
        for i in range(DEVICE_COUNT)
    }
    CUDA_STREAMS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
    WEIGHT_BUFFERS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
    ABSMAX_BUFFERS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
    for k, v in _CUDA_STREAMS.items():
        CUDA_STREAMS[k] = v
    CUDA_STREAMS = tuple(CUDA_STREAMS)
    del _CUDA_STREAMS
else:
    CUDA_STREAMS = ()
    WEIGHT_BUFFERS = []
    ABSMAX_BUFFERS = []


@torch.inference_mode
def fast_dequantize(W, quant_state=None, out=None, use_global_buffer=False):
    if isinstance(W, Float8Tensor):
        return W.dequantize()
    if quant_state is None:
        return W
    if W.dtype == torch.float8_e4m3fn:
        return weight_dequant(W, quant_state)

    if type(quant_state) is not list:
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2

    device = W.device
    device_index = device.index
    CUDA_STREAM = CUDA_STREAMS[device_index]
    n_elements_absmax = absmax.numel()

    if use_global_buffer:
        size = shape[0] * shape[1]
        WEIGHT_BUFFER = WEIGHT_BUFFERS[device_index]
        ABSMAX_BUFFER = ABSMAX_BUFFERS[device_index]
        if WEIGHT_BUFFER is None or WEIGHT_BUFFER.dtype != dtype:
            WEIGHT_BUFFERS[device_index] = WEIGHT_BUFFER = torch_empty(
                size, dtype=dtype, device=device, requires_grad=False
            )
            ABSMAX_BUFFERS[device_index] = ABSMAX_BUFFER = torch_empty(
                n_elements_absmax,
                dtype=torch_float32,
                device=device,
                requires_grad=False,
            )

        if size > WEIGHT_BUFFER.numel():
            WEIGHT_BUFFER.resize_(size)
        if n_elements_absmax > ABSMAX_BUFFER.numel():
            ABSMAX_BUFFER.resize_(n_elements_absmax)

        out = WEIGHT_BUFFER[:size].view(shape)
        out_absmax = ABSMAX_BUFFER[:n_elements_absmax]
    else:
        if out is None:
            out = torch_empty(shape, dtype=dtype, device=device, requires_grad=False)
        else:
            assert out.shape == shape
            assert out.dtype == dtype
        out_absmax = torch_empty(
            n_elements_absmax,
            dtype=torch_float32,
            device=device,
            requires_grad=False,
        )

    ptr_out_absmax = get_ptr(out_absmax)
    with torch_gpu_device(device):
        cdequantize_blockwise_fp32(
            get_ptr(code2),
            get_ptr(absmax),
            get_ptr(absmax2),
            ptr_out_absmax,
            ctypes_c_int(blocksize2),
            ctypes_c_int(n_elements_absmax),
            CUDA_STREAM,
        )
        out_absmax += offset

        fx = (
            cdequantize_blockwise_fp16_nf4
            if dtype == torch_float16
            else cdequantize_blockwise_bf16_nf4
        )
        fx(
            get_ptr(None),
            get_ptr(W),
            ptr_out_absmax,
            get_ptr(out),
            ctypes_c_int(blocksize),
            ctypes_c_int(out.numel()),
            CUDA_STREAM,
        )

    is_transposed = W.shape[0] == 1
    return out.t() if is_transposed else out


def matmul_lora(X, W, W_quant, A, B, s, out=None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    if isinstance(W, Float8Tensor):
        assert W.ndim == 2
        if W.block_size[0] == W.shape[0] and W.block_size[1] == 1:
            W = W.dequantize()
        else:
            W = W.contiguous()
        out = torch_matmul(X, W.t(), out=out)
    elif W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant)
    else:
        W = fast_dequantize(W, W_quant, use_global_buffer=True)
        out = torch_matmul(X, W.t(), out=out)
    if W_quant is not None:
        del W

    if A is not None:
        A, B = A.t(), B.t()
        XA = torch_matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha=s)

    return out.view(batch, seq_len, -1) if reshape else out
