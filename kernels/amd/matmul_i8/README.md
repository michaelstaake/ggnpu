# INT8 Matrix Multiplication Kernel for AMD XDNA2 (AIE2P)

## Operation

`C = A x B` where:
- A: M x K (INT8, row-major)
- B: K x N (INT8, row-major)
- C: M x N (INT32 accumulator, row-major)

This is the core bottleneck of LLM inference (70-80% of compute time).

## Files

| File | Purpose |
|------|---------|
| `matmul.mlir` | MLIR source for mlir-aie compilation to .xclbin |
| `matmul_tile.h` | AIE2P tile compute code (Peano-compiled ELF) |
| `matmul_control.h` | Control tile code (DMA setup, synchronization) |

## Building

```bash
# Compile MLIR to xclbin for npu6 (Krackan)
aiecc.py --target=aie2p --npu-profile=6 \
    kernels/amd/matmul_i8/matmul.mlir \
    -o ~/.cache/ggnpu/xclbin/matmul_npu6.xclbin

# For other NPU profiles
aiecc.py --target=aie2p --npu-profile=4 \
    kernels/amd/matmul_i8/matmul.mlir \
    -o ~/.cache/ggnpu/xclbin/matmul_npu4.xclbin
```

## Architecture

- **Control tile (0,0)**: DMA setup, lock signaling, synchronization
- **Compute tiles (1,0)-(1,7)**: INT8 MAC operations, 16x16 output blocks
- **DMA channels**: 0 (A matrix), 1 (B matrix), 2 (C matrix)
- **Locks**: 3 (start), 4 (done)
- **L2 memory**: 16x16 INT32 accumulator per tile + A/B tile buffers

## Tiling Strategy

For large matrices (M > 128, N > 128), the host splits the work:
- Each compute tile handles a 16x16 block
- Up to 8 tiles work in parallel (columns 0-7 of row 1)
- Larger outputs require multiple rounds or host-side assembly

## Guardrail Compliance

1. **Memory-first**: Block-based DMA, L2-aware tiling
2. **Vector intrinsics only**: INT8 -> INT32 MAC via `aie.mlacc`
3. **No branches**: Fully unrolled K-dimension loop
4. **Two-layer**: Control tile handles DMA; compute tiles handle math
