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

from __future__ import annotations

import functools
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE: int = 65536


def _version_tuple(version: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in version.replace("+", ".").replace("-", ".").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            out.append(int(digits))
        else:
            break
    return tuple(out)


@functools.lru_cache(1)
def is_hip() -> bool:
    return bool(getattr(getattr(torch, "version", None), "hip", None))


@functools.lru_cache(1)
def _device_type() -> str:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "hip" if is_hip() else "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


@functools.lru_cache(1)
def _device_count() -> int:
    device_type = _device_type()
    if device_type in ("cuda", "hip"):
        return torch.cuda.device_count()
    if device_type == "xpu":
        return torch.xpu.device_count()
    return 0


DEVICE_COUNT: int = _device_count()
_DEVICE_TYPE = _device_type()
_DEVICE_TYPE_TORCH = "cuda" if _DEVICE_TYPE == "hip" else _DEVICE_TYPE

if _DEVICE_TYPE == "xpu":
    torch_device_stream = torch.xpu.current_stream
elif _DEVICE_TYPE in ("cuda", "hip"):
    torch_device_stream = torch.cuda.current_stream
else:

    def torch_device_stream(device):
        raise RuntimeError("Triton stream access requires CUDA, HIP, or XPU.")

if DEVICE_COUNT > 1:
    if _DEVICE_TYPE in ("cuda", "hip"):
        torch_gpu_device = torch.cuda.device
    elif _DEVICE_TYPE == "xpu":
        torch_gpu_device = torch.xpu.device
    else:

        def torch_gpu_device(device):
            return nullcontext()
else:

    def torch_gpu_device(device):
        return nullcontext()


if _version_tuple(triton.__version__) >= (3, 0, 0):
    if _DEVICE_TYPE == "xpu":
        triton_tanh = tl.extra.intel.libdevice.tanh
    else:
        from triton.language.extra import libdevice

        triton_tanh = libdevice.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh

    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)


@functools.lru_cache(1)
def is_cdna() -> bool:
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx950",
    )


def calculate_settings(n: int) -> tuple[int, int]:
    block_size: int = triton.next_power_of_2(n)
    if block_size > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if block_size >= 32768:
        num_warps = 32
    elif block_size >= 8192:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    return block_size, num_warps
