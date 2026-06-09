# AMD NPU Kernel Sources

This directory contains NPU kernel sources for the GGNPU inference runtime.
Kernels are compiled into `.xclbin` files using the mlir-aie + Peano toolchain.

## Directory Structure

```
kernels/amd/
├── matmul_i8/     # INT8 matrix multiplication (Phase 2 priority)
│   ├── matmul.mlir       # MLIR source for mlir-aie compilation
│   ├── matmul_tile.h     # C++ tile code for Peano compilation
│   ├── matmul_control.h  # Control code for DMA/kernel launch
│   └── README.md         # This file
├── rmsnorm/         # RMS normalization (Phase 4)
├── rope/            # RoPE positional encoding (Phase 4)
├── softmax/         # Softmax activation (Phase 4)
└── fused_attn/      # Fused attention v2 (Phase 6)
```

## Building Kernels

### Prerequisites

1. **mlir-aie**: https://github.com/Xilinx/mlir-aie
   ```bash
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie && mkdir build && cd build
   cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_TARGETS_TO_BUILD="X86;AArch64;AMDGPU" \
       -DMLIR_AIE_BUILD_TOOLS=ON
   ninja
   ```

2. **Peano** (aie2p toolchain): https://github.com/Xilinx/peano
   ```bash
   git clone https://github.com/Xilinx/peano.git
   cd peano && mkdir build && cd build
   cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
   ninja
   ```

3. **Triton-XDNA** (for AIE2P transforms): https://github.com/amd/Triton-XDNA
   ```bash
   pip install triton-xdna
   ```

### Build Script

```bash
# Set toolchain paths
export AIE_HOME=/path/to/mlir-aie/build/lib
export PEANO_HOME=/path/to/peano/install

# Build all kernels
./scripts/build-kernels.sh
```

### Manual Compilation

```bash
# Compile MLIR to xclbin for npu6 (Krackan)
aiecc.py --target=aie2p --npu-profile=4 \
    kernels/amd/matmul_i8/matmul.mlir \
    -o ~/.cache/ggnpu/xclbin/matmul_npu6.xclbin

# Compile tile code with Peano
aie2p-none-unknown-elf-g++ -target=aie2p \
    kernels/amd/matmul_i8/matmul_tile.h \
    -o kernels/amd/matmul_i8/matmul_tile.elf
```

## Kernel Design

All kernels follow the four AIE2P guardrails:

1. **Memory-first**: Block-based DMA, L2-aware tiling, overlap compute with streaming
2. **Vector intrinsics only**: INT8/BF16 types, 512-bit vector registers, no scalar math
3. **No branches**: Fully unrolled loops, predication only, zero conditional logic
4. **Two-layer**: Control code (DMA/launch) never contains tensor math; kernel code never handles DMA

## Runtime Behavior

At runtime, `ggnpu` will:

1. Check for prebuilt `.xclbin` in `~/.cache/ggnpu/xclbin/`
2. If not found and mlir-aie is available, JIT-compile on first use
3. Cache compiled xclbin for subsequent runs
4. Fall back to CPU reference if no xclbin available (development only)

## Current Status

- `matmul_i8`: Source files created, requires mlir-aie to compile
- `rmsnorm`: Placeholder (Phase 4)
- `rope`: Placeholder (Phase 4)
- `softmax`: Placeholder (Phase 4)
- `fused_attn`: Placeholder (Phase 6)
