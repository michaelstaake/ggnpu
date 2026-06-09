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

# Default: build all profiles unless specified
if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    PROFILES=("$1")
elif [ $# -ge 1 ]; then
    # Kernel name filter (e.g., "matmul")
    KERNEL_FILTER="$1"
    PROFILES=("${ALL_PROFILES[@]}")
else
    PROFILES=("${ALL_PROFILES[@]}")
    KERNEL_FILTER=""
fi

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
# Check for aiecc.py
#====//
AIECC_FOUND=false
AIECC_PATH=""

if [ -n "${AIE_HOME:-}" ]; then
    if [ -f "$AIE_HOME/bin/aiecc.py" ]; then
        AIECC_FOUND=true
        AIECC_PATH="$AIE_HOME/bin/aiecc.py"
    else
        echo "ERROR: AIE_HOME=$AIE_HOME but aiecc.py not found"
        exit 1
    fi
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
    echo "Or set AIE_HOME to an existing mlir-aie build."
    echo "Alternatively, use prebuilt xclbins in $XCLBIN_DIR"
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

# Define kernels to build: name, mlir_file, output_prefix
Kernels=(
    "matmul:matmul_i8/matmul.mlir:matmul"
    "rmsnorm:rmsnorm/rmsnorm.mlir:rmsnorm"
    "rope:rope/rope.mlir:rope"
    "softmax:softmax/softmax.mlir:softmax"
    "silu:silu/silu.mlir:silu"
    "flash_attn:fused_attn/flash_attn.mlir:flash_attn"
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
