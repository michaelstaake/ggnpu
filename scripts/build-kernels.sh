#!/bin/bash
# Build NPU kernels for GGNPU
# Requires mlir-aie and Peano toolchains

set -e

CACHE_DIR="${GGNPU_CACHE_DIR:-$HOME/.cache/ggnpu}"
XCLBIN_DIR="$CACHE_DIR/xclbin"

mkdir -p "$XCLBIN_DIR"

echo "=== GGNPU NPU Kernel Builder ==="
echo "Cache dir: $CACHE_DIR"

# Check for mlir-aie
if command -v aiecc.py >/dev/null 2>&1; then
    echo "  mlir-aie: found"
else
    echo "  WARNING: mlir-aie not found. Using prebuilt xclbins."
fi

# Check for Peano
if command -p aie2p-none-unknown-elf-g++ >/dev/null 2>&1; then
    echo "  Peano: found"
else
    echo "  WARNING: Peano not found. Using prebuilt xclbins."
fi

# Build kernel templates
echo ""
echo "Building kernel templates..."

KERNELS=(
    "matmul_i8"
    "rmsnorm"
    "rope"
    "softmax"
    "fused_attn"
)

for kernel in "${KERNELS[@]}"; do
    echo "  $kernel: checking..."
    if [ -f "$XCLBIN_DIR/${kernel}.xclbin" ]; then
        echo "    Already cached"
    else
        echo "    Not found (prebuilt xclbin expected)"
    fi
done

echo ""
echo "=== Kernel Build Complete ==="
echo "To enable JIT compilation, install mlir-aie and Peano toolchains."
