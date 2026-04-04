# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Vendored from unsloth/models/_utils.py (Flash Attention + xFormers detection
# sections only) for issue #67.

"""Runtime hardware-capability detection for unturtle.

Provides:
  HAS_FLASH_ATTENTION  — True when flash_attn >= 2.x is usable on the current GPU
  xformers             — the xformers.ops.fmha module or None
  xformers_attention   — xformers.ops.fmha.memory_efficient_attention or None
"""

from __future__ import annotations

import torch
from packaging.version import Version
from transformers.utils.import_utils import _is_package_available

# ---------------------------------------------------------------------------
# Device type
# ---------------------------------------------------------------------------
try:
    _device_type: str = torch.device(torch.cuda.current_device()).type
except Exception:
    _device_type = "cpu"

if torch.cuda.is_available():
    _device_type = "cuda"

# HIP / ROCm
try:
    if torch.version.hip is not None:
        _device_type = "hip"
except Exception:
    pass

# ---------------------------------------------------------------------------
# Flash Attention
# ---------------------------------------------------------------------------
HAS_FLASH_ATTENTION = False
HAS_FLASH_ATTENTION_SOFTCAPPING = False

if _device_type in ("cuda", "hip"):
    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version >= 8 or _device_type == "hip":
        if _is_package_available("flash_attn"):
            try:
                try:
                    from flash_attn.flash_attn_interface import flash_attn_gpu  # noqa: F401
                except Exception:
                    from flash_attn.flash_attn_interface import flash_attn_cuda  # noqa: F401
                HAS_FLASH_ATTENTION = True

                from flash_attn import __version__ as _flash_attn_version

                HAS_FLASH_ATTENTION_SOFTCAPPING = Version(_flash_attn_version) >= Version(
                    "2.6.3"
                )
            except Exception:
                HAS_FLASH_ATTENTION = False

# ---------------------------------------------------------------------------
# xFormers
# ---------------------------------------------------------------------------
try:
    from xformers import __version__ as _xformers_version

    torch_version = torch.__version__.split("+")[0]
    if Version(torch_version) < Version("2.2.0") and Version(
        _xformers_version
    ) >= Version("0.0.24"):
        raise ImportError(
            f"unturtle: torch={torch_version} but xformers={_xformers_version}; "
            "please install xformers < 0.0.24 for this torch version."
        )
    if Version(torch_version) < Version("2.3.0") and Version(
        _xformers_version
    ) >= Version("0.0.26"):
        raise ImportError(
            f"unturtle: torch={torch_version} but xformers={_xformers_version}; "
            "please install xformers < 0.0.26 for this torch version."
        )
    if Version(torch_version) < Version("2.4.0") and Version(
        _xformers_version
    ) > Version("0.0.27"):
        raise ImportError(
            f"unturtle: torch={torch_version} but xformers={_xformers_version}; "
            "please install xformers <= 0.0.27 for this torch version."
        )

    # xformers FA3 dispatch broken on Blackwell+ before 0.0.33
    if torch.cuda.is_available():
        _cc = torch.cuda.get_device_capability()
        if f"{_cc[0]}.{_cc[1]}" in ("10.0", "11.0", "12.0") and Version(
            _xformers_version
        ) <= Version("0.0.32.post2"):
            raise ImportError(
                f"unturtle: xformers {_xformers_version} has a broken FA3 dispatch on "
                f"SM {_cc[0]}.{_cc[1]}; please upgrade to >= 0.0.33."
            )

    from xformers._cpp_lib import _register_extensions

    try:
        _register_extensions()
    except Exception as _e:
        raise ImportError(
            "unturtle: xformers was not installed correctly.\n"
            "Please run: python -m xformers.info\n\n"
            "Longer error: " + str(_e)
        ) from _e

    import xformers.ops.fmha as xformers  # type: ignore[assignment]

    xformers_attention = xformers.memory_efficient_attention
except ModuleNotFoundError:
    xformers = None
    xformers_attention = None
except Exception:
    xformers = None
    xformers_attention = None

SUPPORTS_BFLOAT16: bool = False
if _device_type in ("cuda", "hip") and torch.cuda.is_available():
    _major, _ = torch.cuda.get_device_capability()
    SUPPORTS_BFLOAT16 = _major >= 8


def is_bfloat16_supported() -> bool:
    return SUPPORTS_BFLOAT16


__all__ = [
    "HAS_FLASH_ATTENTION",
    "HAS_FLASH_ATTENTION_SOFTCAPPING",
    "SUPPORTS_BFLOAT16",
    "is_bfloat16_supported",
    "xformers",
    "xformers_attention",
]
