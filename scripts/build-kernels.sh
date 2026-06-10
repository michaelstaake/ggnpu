#!/bin/bash
# Build NPU kernels for GGNPU
# Compiles Triton Python kernels into .xclbin files using Triton-XDNA
#
# Usage:
#   ./scripts/build-kernels.sh              # Build with available tools
#   ./scripts/build-kernels.sh npu6         # Build only for npu6 (Krackan)
#   ./scripts/build-kernels.sh matmul       # Build only matmul kernel
#
# Kernels built:
#   - matmul: INT8/BF16 matrix multiplication (core bottleneck)
#   - rmsnorm: RMS normalization
#   - rope: Rotary positional embeddings
#   - softmax: Softmax activation
#   - silu: SiLU/Swish activation
#   - flash_attn: FlashAttention v1 (decomposed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Auto-activate Triton venv when present
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -n "${GGNPU_TRITON_VENV:-}" ] && [ -f "${GGNPU_TRITON_VENV}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${GGNPU_TRITON_VENV}/bin/activate"
    elif [ -f "${SCRIPT_DIR}/.venv-triton/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${SCRIPT_DIR}/.venv-triton/bin/activate"
    elif [ -f "${HOME}/triton-env/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/triton-env/bin/activate"
    fi
fi

CACHE_DIR="${GGNPU_CACHE_DIR:-$HOME/.cache/ggnpu}"
XCLBIN_DIR="$CACHE_DIR/xclbin"
COMPILE_SCRIPT="$SCRIPT_DIR/kernels/triton/compile_kernels.py"

# NPU profiles: 4=Strix Point, 5=Strix Point rev, 6=Krackan
ALL_PROFILES=("4" "5" "6")

# Parse args: profile (6, npu6) and/or kernel name (matmul)
KERNEL_FILTER=""
PROFILES=("${ALL_PROFILES[@]}")

parse_profile_arg() {
    local arg="$1"
    case "$arg" in
        npu4|4) echo "4" ;;
        npu5|5) echo "5" ;;
        npu6|6) echo "6" ;;
        *) echo "" ;;
    esac
}

while [ $# -gt 0 ]; do
    profile="$(parse_profile_arg "$1")"
    if [ -n "$profile" ]; then
        PROFILES=("$profile")
    else
        KERNEL_FILTER="$1"
    fi
    shift
done

mkdir -p "$XCLBIN_DIR"

echo "=== GGNPU NPU Kernel Builder (Triton-XDNA) ==="
echo "Output directory: $XCLBIN_DIR"
echo "NPU profiles: ${PROFILES[*]}"
if [ -n "${KERNEL_FILTER:-}" ]; then
    echo "Kernel filter: $KERNEL_FILTER"
fi
echo ""

#====//
# Check for python3 and Triton-XDNA
#====//
PYTHON3_FOUND=false
PYTHON3_BIN=""

if command -v python3 >/dev/null 2>&1; then
    PYTHON3_BIN="$(command -v python3)"
    PYTHON3_FOUND=true
elif command -v python >/dev/null 2>&1; then
    PYTHON3_BIN="$(command -v python)"
    PYTHON3_FOUND=true
fi

if [ "$PYTHON3_FOUND" = false ]; then
    echo "ERROR: python3 not found"
    echo ""
    echo "  Install Python 3.10+ and Triton-XDNA:"
    echo "    sudo apt install python3 python3-pip"
    echo "    pip install triton-xdna"
    exit 1
fi

# Check if Triton is importable
if ! $PYTHON3_BIN -c "from triton.backends.amd_triton_npu.driver import NPUDriver" 2>/dev/null; then
    echo "ERROR: Triton-XDNA not installed"
    echo ""
    echo "  Install with:"
    echo "    bash scripts/setup-triton-env.sh"
    echo "    source ~/triton-env/bin/activate   # or .venv-triton"
    echo ""
    echo "  Or use prebuilt xclbins in:"
    echo "    $XCLBIN_DIR"
    exit 1
fi

echo "Triton-XDNA: $PYTHON3_BIN"

# XRT dev headers required for Triton-XDNA compile-only launcher build
if [ -n "${XILINX_XRT:-}" ] && [ -d "${XILINX_XRT}/include/xrt" ] && [ -d "${XILINX_XRT}/lib" ]; then
    echo "XRT SDK: $XILINX_XRT"
elif [ -d /opt/xilinx/xrt/include/xrt ] && [ -d /opt/xilinx/xrt/lib ]; then
    export XILINX_XRT=/opt/xilinx/xrt
    echo "XRT SDK: $XILINX_XRT"
elif [ -d "$SCRIPT_DIR/third_party/xrt-dev/usr/include/xrt" ]; then
    export XILINX_XRT="$SCRIPT_DIR/third_party/xrt-dev/usr"
    echo "XRT SDK: $XILINX_XRT (third_party)"
elif [ -d /usr/include/xrt ]; then
    echo "XRT SDK: will use system headers via build/xrt-sdk-shim (compile_kernels.py)"
else
    echo "ERROR: XRT development files not found"
    echo ""
    echo "  Triton-XDNA kernel builds need libxrt-dev headers + libraries."
    echo "  Install: sudo apt install libxrt-dev uuid-dev"
    echo "  Or set:  export XILINX_XRT=/opt/xilinx/xrt"
    echo ""
    exit 1
fi

if [ ! -f "$COMPILE_SCRIPT" ]; then
    echo "ERROR: Triton compile script not found: $COMPILE_SCRIPT"
    echo "  Make sure kernels/triton/compile_kernels.py exists"
    exit 1
fi

echo "Compile script: $COMPILE_SCRIPT"
echo ""

#====//
# Build kernels
#====//
TOTAL=0
SUCCESS=0
FAILED=0

# Kernels to build: name:compile_script_args
Kernels=(
    "matmul:--M 64 --N 64 --K 64"
    "rmsnorm:--N 2048"
    "rope:--N 2048 --dims 64"
    "softmax:--rows 1 --cols 1024"
    "silu:--N 8192"
    "flash_attn:--n_head 8 --head_dim 128 --ctx_len 2048"
)

for kernel_def in "${Kernels[@]}"; do
    IFS=':' read -r kernel_name kernel_args <<< "$kernel_def"

    # Apply kernel filter if set
    if [ -n "${KERNEL_FILTER:-}" ] && [ "$kernel_name" != "$KERNEL_FILTER" ]; then
        continue
    fi

    echo "Building kernel: $kernel_name"

    for profile in "${PROFILES[@]}"; do
        TOTAL=$((TOTAL + 1))
        output_xclbin="$XCLBIN_DIR/${kernel_name}_npu${profile}.xclbin"

        if [ -f "$output_xclbin" ]; then
            echo "  [npu$profile] already exists, skipping"
            SUCCESS=$((SUCCESS + 1))
            continue
        fi

        echo -n "  [npu$profile] "

        if $PYTHON3_BIN "$COMPILE_SCRIPT" \
            --op "$kernel_name" \
            --profile "npu${profile}" \
            --output-dir "$XCLBIN_DIR" \
            $kernel_args 2>&1; then
            
            if [ -f "$output_xclbin" ]; then
                size=$(stat -c%s "$output_xclbin" 2>/dev/null || stat -f%z "$output_xclbin" 2>/dev/null || echo "?")
                echo "OK (${size} bytes)"
                SUCCESS=$((SUCCESS + 1))
            else
                echo "FAILED (xclbin not produced)"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "FAILED"
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
done

#====//
# Summary
#====//
echo "=== Build Summary ==="
echo "  Total: $TOTAL"
echo "  Success: $SUCCESS"
echo "  Failed: $FAILED"
echo ""

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No kernels were built."
    if [ -n "${KERNEL_FILTER:-}" ]; then
        echo "  Unknown kernel filter: $KERNEL_FILTER"
        echo "  Valid kernels: matmul rmsnorm rope softmax silu flash_attn"
    fi
    exit 1
fi

if [ "$FAILED" -gt 0 ]; then
    echo "Some kernels failed to compile. Check the output above for errors."
    echo "Common issues:"
    echo "  - Triton-XDNA not properly installed"
    echo "  - Insufficient memory during compilation"
    echo ""
    echo "You can still run ggnpu with prebuilt xclbins."
    exit 1
fi

echo "All kernels built successfully!"
echo "Output: $XCLBIN_DIR"
echo ""
echo "To use these xclbins with ggnpu, ensure they are in:"
echo "  $XCLBIN_DIR"
echo ""
echo "Or set GGNPU_CACHE_DIR to the directory containing xclbins."
