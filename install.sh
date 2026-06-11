#!/usr/bin/env bash
# Unturtle setup script (uv-based).
#
# Usage:
#   ./install.sh           # core + huggingface extras
#   ./install.sh --eval    # additionally install the lm-eval-harness extra
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - NVIDIA GPU + driver. The torch CUDA build is selected via TORCH_INDEX.
#     The default cu128 runs on CUDA 12.x drivers via CUDA minor-version
#     compatibility (empirically verified on driver 555 / CUDA 12.5). If the
#     final verification step reports "CUDA not available", retry with an
#     older build matching your driver, e.g.:
#     TORCH_INDEX=https://download.pytorch.org/whl/cu124 ./install.sh
#     If your driver supports CUDA 13, the default PyPI wheels also work:
#     TORCH_INDEX=https://pypi.org/simple ./install.sh
#
# NOTE: plain `pip install -e ".[huggingface]"` is NOT supported — pip's
# resolver has been observed to pick a broken ancient `regex` build from
# unsloth's dependency graph. Use uv (this script).

set -euo pipefail
cd "$(dirname "$0")"

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

command -v uv >/dev/null || { echo "error: uv not found — install from https://docs.astral.sh/uv/"; exit 1; }

echo "==> creating venv (.venv, python ${PYTHON_VERSION})"
uv venv .venv --python "${PYTHON_VERSION}" --allow-existing

# torch must be installed BEFORE the editable install so that uv keeps the
# CUDA-matched build instead of re-resolving torch from PyPI (which may ship
# a CUDA major version your driver does not support).
echo "==> installing torch stack from ${TORCH_INDEX}"
uv pip install torch torchvision torchaudio xformers --index-url "${TORCH_INDEX}"

echo "==> installing build/test tooling"
uv pip install "setuptools==80.9.0" "setuptools-scm==9.2.0" pytest ruff bitsandbytes

echo "==> installing unturtle (editable) + huggingface extras"
# unsloth 2026.6.x pins transformers<=5.5.0, but DiffusionGemma and upcoming models
# require transformers>=5.8.0.  Install transformers first so uv keeps the newer
# build; unsloth/unsloth_zoo have no runtime version enforcement (verified 2026-06-11).
uv pip install "transformers>=5.8.0"
uv pip install -e ".[huggingface]"

if [[ "${1:-}" == "--eval" ]]; then
    echo "==> installing eval extra (lm-eval-harness)"
    uv pip install -e ".[eval]"
fi

echo "==> verifying installation"
.venv/bin/python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA not available — check TORCH_INDEX vs your driver"
import unsloth  # noqa: F401  (requires a GPU at import time)
import unturtle
print(f"unturtle {unturtle.__version__} — OK")
PY

echo "done. run tests with: uv run python -m pytest tests/ -m 'not slow' -v"
