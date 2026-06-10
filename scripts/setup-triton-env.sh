#!/bin/bash
# Create a Python venv with Triton-XDNA for kernel compilation.
# C++ / XRT builds do not need this venv — only kernel builds do.
#
# Usage:
#   bash scripts/setup-triton-env.sh              # ~/triton-env
#   bash scripts/setup-triton-env.sh .venv-triton # custom path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${1:-$HOME/triton-env}"

FIND_LINKS=(
    "https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels"
    "https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2"
    "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
    "https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti"
)

echo "=== GGNPU Triton-XDNA environment setup ==="
echo "Venv: $VENV_DIR"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found"
    echo "  sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip wheel

# Point at local XRT headers when system libxrt-dev is unavailable
if [ -z "${XILINX_XRT:-}" ] && [ -d "${SCRIPT_DIR}/third_party/xrt-dev/usr/include/xrt" ]; then
    export XILINX_XRT="${SCRIPT_DIR}/third_party/xrt-dev/usr"
    echo "Using XRT headers from: $XILINX_XRT"
fi

PIP_ARGS=(install -r "$SCRIPT_DIR/requirements-triton.txt")
for url in "${FIND_LINKS[@]}"; do
    PIP_ARGS+=(--find-links "$url")
done

echo "Installing Triton-XDNA (mlir-aie/mlir-air/llvm-aie come as transitive wheels)..."
python -m pip "${PIP_ARGS[@]}"

echo "Installing torch (CPU) for kernel compile scripts..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Verifying Triton-XDNA..."
python -c "
from triton.backends.amd_triton_npu.driver import NPUDriver
import triton
print('  triton:', getattr(triton, '__version__', 'unknown'))
print('  NPUDriver: OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate before building kernels:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Build kernels:"
echo "  ./scripts/build-kernels.sh npu6"
echo ""
echo "Or set GGNPU_TRITON_VENV=$VENV_DIR and build-kernels.sh will auto-activate."
