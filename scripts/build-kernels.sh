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
#   - matmul: INT8 matrix multiplication (core bottleneck)
#   - rmsnorm: RMS normalization
#   - softmax: Softmax activation
#   - silu: SiLU/Swish activation
# Experimental (GGNPU_EXPERIMENTAL=1; no working transform recipe yet):
#   - rope: Rotary positional embeddings
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
resolve_xrt_sdk() {
    if [ -n "${XILINX_XRT:-}" ] && [ -d "${XILINX_XRT}/include/xrt" ] && [ -d "${XILINX_XRT}/lib" ]; then
        echo "$XILINX_XRT"
        return 0
    fi
    if [ -d /opt/xilinx/xrt/include/xrt ] && [ -d /opt/xilinx/xrt/lib ]; then
        echo /opt/xilinx/xrt
        return 0
    fi
    if [ -d "$SCRIPT_DIR/third_party/xrt-dev/usr/include/xrt" ] \
        && [ -d "$SCRIPT_DIR/third_party/xrt-dev/usr/lib" ]; then
        echo "$SCRIPT_DIR/third_party/xrt-dev/usr"
        return 0
    fi
    if [ -d /usr/include/xrt ] && [ -f /usr/include/uuid/uuid.h ]; then
        echo "shim"
        return 0
    fi
    return 1
}

XRT_SDK="$(resolve_xrt_sdk || true)"
if [ -z "$XRT_SDK" ]; then
    echo "XRT/uuid dev headers not found — fetching libxrt-dev + uuid-dev into third_party/ ..."
    if bash "$SCRIPT_DIR/scripts/fetch-xrt-dev.sh"; then
        XRT_SDK="$(resolve_xrt_sdk || true)"
    fi
fi

if [ -z "$XRT_SDK" ]; then
    echo "ERROR: XRT development files not found"
    echo ""
    echo "  Triton-XDNA kernel builds need libxrt-dev headers + libraries."
    echo "  Option A: sudo apt install libxrt-dev uuid-dev"
    echo "  Option B: bash scripts/fetch-xrt-dev.sh   # extract debs to third_party/"
    echo "  Option C: export XILINX_XRT=/opt/xilinx/xrt"
    echo ""
    exit 1
fi

if [ "$XRT_SDK" = "shim" ]; then
    echo "XRT SDK: system headers via ~/.cache/ggnpu/xrt-sdk-shim (compile_kernels.py)"
    if grep -qi microsoft /proc/version 2>/dev/null && [[ "$SCRIPT_DIR" == /mnt/* ]]; then
        echo "WARNING: WSL repo on /mnt/c — move to ~/ggnpu (native Linux FS) if builds fail."
        echo "  Run: bash scripts/verify-kernel-build.sh"
    fi
else
    export XILINX_XRT="$XRT_SDK"
    echo "XRT SDK: $XILINX_XRT"
fi

if [ ! -f "$COMPILE_SCRIPT" ]; then
    echo "ERROR: Triton compile script not found: $COMPILE_SCRIPT"
    echo "  Make sure kernels/triton/compile_kernels.py exists"
    exit 1
fi

echo "Compile script: $COMPILE_SCRIPT"
echo ""

# xclbinutil from third_party needs boost .so (no sudo apt install required)
XCLBINUTIL="$SCRIPT_DIR/third_party/xrt-tools/usr/bin/xclbinutil"
if [ -x "$XCLBINUTIL" ]; then
    export PATH="$SCRIPT_DIR/third_party/xrt-tools/usr/bin:$PATH"
    BOOST_LIB="$SCRIPT_DIR/third_party/boost-lib/usr/lib/x86_64-linux-gnu"
    if [ -d "$BOOST_LIB" ]; then
        export LD_LIBRARY_PATH="${BOOST_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
elif ! command -v xclbinutil >/dev/null 2>&1; then
    echo "Fetching xclbinutil (libxrt-utils) into third_party/ ..."
    bash "$SCRIPT_DIR/scripts/fetch-xrt-dev.sh" --tools || true
    if [ -x "$XCLBINUTIL" ]; then
        export PATH="$SCRIPT_DIR/third_party/xrt-tools/usr/bin:$PATH"
    fi
fi

#====//
# Build kernels
#====//
TOTAL=0
SUCCESS=0
FAILED=0

# Kernels to build: name:compile_script_args
# Sizes must match what the transform scripts in kernels/triton/transforms/
# were authored for (they tile fixed block shapes).
Kernels=(
    "matmul:--M 256 --N 256 --K 256"
    "rmsnorm:--M 32 --N 256"
    "softmax:--rows 256 --cols 256"
    "silu:--N 8192"
)

# rope and flash_attn have no working Triton-XDNA transform recipe yet
# (gather loads / loop-carried accumulators). Build with GGNPU_EXPERIMENTAL=1.
if [ -n "${GGNPU_EXPERIMENTAL:-}" ]; then
    Kernels+=(
        "rope:--N 2048 --dims 64"
        "flash_attn:--n_head 8 --head_dim 128 --ctx_len 2048"
    )
fi

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
                # Llama hidden=2048: also install shape-specific name for backend lookup
                if [ "$kernel_name" = "rmsnorm" ] && [[ "$kernel_args" == *"--N 2048"* ]]; then
                    shaped="$XCLBIN_DIR/rmsnorm_2048_npu${profile}.xclbin"
                    shaped_seq="$XCLBIN_DIR/rmsnorm_2048_npu${profile}_sequence.bin"
                    cp -f "$output_xclbin" "$shaped"
                    seq_src="$XCLBIN_DIR/${kernel_name}_npu${profile}_sequence.bin"
                    if [ -f "$seq_src" ]; then
                        cp -f "$seq_src" "$shaped_seq"
                    fi
                    echo "  Also installed $shaped"
                fi
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
        echo "  Valid kernels: matmul rmsnorm softmax silu"
        echo "  Experimental (need GGNPU_EXPERIMENTAL=1): rope flash_attn"
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
