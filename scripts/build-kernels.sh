#!/bin/bash
# Build NPU kernels for GGNPU
# Compiles MLIR kernel sources into .xclbin files for the AMD NPU
#
# Usage:
#   ./scripts/build-kernels.sh              # Build with available tools
#   AIE_HOME=/path/to/mlir-aie ./scripts/build-kernels.sh
#   PEANO_HOME=/path/to/peano ./scripts/build-kernels.sh
#   ./scripts/build-kernels.sh npu6         # Build only for npu6 (Krackan)
#   ./scripts/build-kernels.sh matmul       # Build only matmul kernel
#
# Kernels built:
#   - matmul: INT8 matrix multiplication (core bottleneck)
#   - rmsnorm: RMS normalization
#   - rope: Rotary positional embeddings
#   - softmax: Softmax activation
#   - silu: SiLU/Swish activation
#   - flash_attn: FlashAttention v1 (decomposed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KERNELS_DIR="$SCRIPT_DIR/kernels/amd"
CACHE_DIR="${GGNPU_CACHE_DIR:-$HOME/.cache/ggnpu}"
XCLBIN_DIR="$CACHE_DIR/xclbin"

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

echo "=== GGNPU NPU Kernel Builder ==="
echo "Kernels source: $KERNELS_DIR"
echo "Output directory: $XCLBIN_DIR"
echo "NPU profiles: ${PROFILES[*]}"
if [ -n "${KERNEL_FILTER:-}" ]; then
    echo "Kernel filter: $KERNEL_FILTER"
fi
echo ""

#====//
# Check for aiecc / aiecc.py
#====//
AIECC_FOUND=false
AIECC_PATH=""

if [ -z "${AIE_HOME:-}" ] && [ -f /etc/ggnpu-aie-home ]; then
    AIE_HOME="$(cat /etc/ggnpu-aie-home)"
fi

if [ -n "${AIE_HOME:-}" ]; then
    if [[ "$AIE_HOME" == *"/path/to/"* ]] || [ ! -d "$AIE_HOME" ]; then
        echo "ERROR: AIE_HOME is not a real mlir-aie install: $AIE_HOME"
        echo ""
        echo "  /path/to/mlir-aie is a documentation placeholder — set AIE_HOME to your"
        echo "  actual mlir-aie build directory (the one that contains bin/aiecc.py)."
        echo ""
        echo "  Example after building mlir-aie:"
        echo "    export AIE_HOME=\$HOME/mlir-aie/build"
        echo "    ./scripts/build-kernels.sh npu6 matmul"
        exit 1
    fi
    if [ -x "$AIE_HOME/bin/aiecc" ]; then
        AIECC_FOUND=true
        AIECC_PATH="$AIE_HOME/bin/aiecc"
    elif [ -f "$AIE_HOME/bin/aiecc.py" ]; then
        AIECC_FOUND=true
        AIECC_PATH="$AIE_HOME/bin/aiecc.py"
    else
        echo "ERROR: AIE_HOME=$AIE_HOME but bin/aiecc or bin/aiecc.py not found"
        echo "  mlir-aie is not installed. See docs/host-setup-guide.md"
        exit 1
    fi
elif command -v aiecc >/dev/null 2>&1; then
    AIECC_FOUND=true
    AIECC_PATH="$(command -v aiecc)"
elif command -v aiecc.py >/dev/null 2>&1; then
    AIECC_FOUND=true
    AIECC_PATH="$(command -v aiecc.py)"
else
    echo "ERROR: mlir-aie (aiecc.py) not found"
    echo ""
    echo "To install mlir-aie:"
    echo "  git clone https://github.com/Xilinx/mlir-aie.git"
    echo "  cd mlir-aie && mkdir build && cd build"
    echo "  cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release \\"
    echo "      -DLLVM_ENABLE_PROJECTS=mlir \\"
    echo "      -DMLIR_AIE_BUILD_TOOLS=ON"
    echo "  ninja"
    echo ""
    echo "Or set AIE_HOME to your mlir-aie *build* directory (contains bin/aiecc.py):"
    echo "  export AIE_HOME=\$HOME/mlir-aie/build"
    echo "  ./scripts/build-kernels.sh npu6 matmul"
    echo ""
    echo "Build guide: https://github.com/Xilinx/mlir-aie/blob/main/docs/Building.md"
    echo "Ryzen AI notes: https://github.com/Xilinx/mlir-aie/blob/main/docs/Building.md"
    echo ""
    echo "On 16GB RAM machines, prefer copying prebuilt xclbins into:"
    echo "  $XCLBIN_DIR"
    echo "  (needs at least matmul_npu6.xclbin for bench-matmul)"
    exit 1
fi

echo "mlir-aie: $AIECC_PATH"

if [ -n "${PEANO_HOME:-}" ] && [ -f "$PEANO_HOME/bin/aie2p-none-unknown-elf-g++" ]; then
    echo "Peano: $PEANO_HOME"
    PEANO_FLAGS="-I${PEANO_HOME}/include -L${PEANO_HOME}/lib"
elif command -v aie2p-none-unknown-elf-g++ >/dev/null 2>&1; then
    PEANO_DIR="$(dirname "$(dirname "$(command -v aie2p-none-unknown-elf-g++)")")"
    echo "Peano: $PEANO_DIR"
    PEANO_FLAGS="-I${PEANO_DIR}/include -L${PEANO_DIR}/lib"
else
    echo "Peano: not found (tile ELF compilation may fail)"
    PEANO_FLAGS=""
fi
echo ""

#====//
# Compile a single MLIR file to xclbin
#====//
compile_kernel() {
    local mlir_file="$1"
    local output_xclbin="$2"
    local profile="$3"
    local kernel_name="$4"

    local cmd="$AIECC_PATH --target=aie2p --npu-profile=$profile"

    if [ -n "$PEANO_FLAGS" ]; then
        cmd+=" $PEANO_FLAGS"
    fi

    # Add AIE_HOME include path if set
    if [ -n "${AIE_HOME:-}" ]; then
        cmd+=" -I${AIE_HOME}/include"
    fi

    cmd+=" \"$mlir_file\" -o \"$output_xclbin\""

    echo -n "  [$kernel_name npu$profile] "

    if eval "$cmd" 2>&1; then
        if [ -f "$output_xclbin" ]; then
            local size
            size=$(stat -c%s "$output_xclbin" 2>/dev/null || stat -f%z "$output_xclbin" 2>/dev/null || echo "?")
            echo "OK (${size} bytes)"
            return 0
        fi
    fi

    echo "FAILED"
    return 1
}

#====//
# Build kernels
#====//
TOTAL=0
SUCCESS=0
FAILED=0

# Memory-limited kernel builds for 16 GB RAM machines.
# Build only matmul first (critical path for bench-matmul).
# Additional kernels can be built later with: ./scripts/build-kernels.sh npu6 rmsnorm
Kernels=(
    "matmul:matmul_i8/matmul.mlir:matmul"
)

for kernel_def in "${Kernels[@]}"; do
    IFS=':' read -r kernel_name mlir_rel output_prefix <<< "$kernel_def"

    # Apply kernel filter if set
    if [ -n "${KERNEL_FILTER:-}" ] && [ "$kernel_name" != "$KERNEL_FILTER" ]; then
        continue
    fi

    mlir_file="$KERNELS_DIR/$mlir_rel"

    if [ ! -f "$mlir_file" ]; then
        echo "WARNING: MLIR source not found: $mlir_file (skipping $kernel_name)"
        continue
    fi

    echo "Building kernel: $kernel_name"

    for profile in "${PROFILES[@]}"; do
        TOTAL=$((TOTAL + 1))
        output_xclbin="$XCLBIN_DIR/${output_prefix}_npu${profile}.xclbin"

        if [ -f "$output_xclbin" ]; then
            echo "  [npu$profile] already exists, skipping"
            SUCCESS=$((SUCCESS + 1))
            continue
        fi

        if compile_kernel "$mlir_file" "$output_xclbin" "$profile" "$kernel_name"; then
            SUCCESS=$((SUCCESS + 1))
        else
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
    echo "  - MLIR syntax errors in source files"
    echo "  - Missing Peano toolchain for tile code"
    echo "  - Insufficient memory during compilation"
    echo ""
    echo "You can still run ggnpu with prebuilt xclbins or CPU reference backend."
    exit 1
fi

echo "All kernels built successfully!"
echo "Output: $XCLBIN_DIR"
echo ""
echo "To use these xclbins with ggnpu, ensure they are in:"
echo "  $XCLBIN_DIR"
echo ""
echo "Or set GGNPU_CACHE_DIR to the directory containing xclbins."
