#!/usr/bin/env bash
# Unturtle setup script (uv-based).
#
# Usage:
#   ./install.sh           # core + huggingface extras
#   ./install.sh --eval    # additionally install the lm-eval-harness extra
#
# Overrides:
#   TORCH_INDEX=...        # torch wheel index (see Requirements below)
#   PYTHON_VERSION=3.12    # interpreter for the venv
#   (a user-level UV_PYTHON pin is ignored — this script manages its own venv)
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

# A user-level UV_PYTHON pin makes `uv pip` ignore ./.venv whenever the pinned
# version differs from the venv's interpreter ("No virtual environment found
# for Python X.Y.Z"). This script manages its own venv, so drop the pin and
# target .venv explicitly.
unset UV_PYTHON UV_PROJECT_ENVIRONMENT 2>/dev/null || true
export VIRTUAL_ENV="$PWD/.venv"

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
uv pip install -e ".[huggingface]"

if [[ "${1:-}" == "--eval" ]]; then
    echo "==> installing eval extra (lm-eval-harness)"
    uv pip install -e ".[eval]"
fi

echo "==> re-pinning transformers and the CUDA torch stack"
# The editable install above re-resolves the full dependency graph and has been
# observed (uv 0.11.x) to downgrade transformers to unsloth's <=5.5.0 pin and to
# replace the CUDA-matched torch build with a plain PyPI wheel. DiffusionGemma
# and upcoming models require transformers>=5.8.0 (unsloth/unsloth_zoo have no
# runtime version enforcement, verified 2026-06-11), and torch must stay on the
# TORCH_INDEX build — so re-pin both AFTER the editable install.
uv pip install "transformers>=5.8.0,<6"
# Re-pin the exact torch version the resolver settled on (it already satisfies
# unsloth's constraints) instead of an open-ended --upgrade, which resolves only
# the listed packages' requirements and can jump past unsloth's supported torch
# range to a brand-new release on the index.
TORCH_VER=$(.venv/bin/python -c "import torch; print(torch.__version__.split('+')[0])")
uv pip install --upgrade "torch==${TORCH_VER}" torchvision torchaudio xformers --index-url "${TORCH_INDEX}"

echo "==> checking dependency consistency"
uv pip check || echo "warning: dependency conflicts reported above — review before relying on this env"

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
