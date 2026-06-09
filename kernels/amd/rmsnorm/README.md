# RMS Normalization Kernel for AMD XDNA2 (AIE2P)

## Operation

RMSNorm: `y[i] = x[i] / sqrt(mean(x^2) + eps)`

Used in every transformer layer (attention and FFN paths).

## Files

| File | Purpose |
|------|---------|
| `rmsnorm.mlir` | MLIR source for mlir-aie compilation to .xclbin |
| `rmsnorm_tile.h` | AIE2P tile compute code (Peano-compiled ELF) |
| `rmsnorm_control.h` | Control tile code (DMA setup, reduction, factor computation) |

## Building

```bash
# Compile MLIR to xclbin for npu6 (Krackan)
aiecc.py --target=aie2p --npu-profile=6 \
    kernels/amd/rmsnorm/rmsnorm.mlir \
    -o ~/.cache/ggnpu/xclbin/rmsnorm_npu6.xclbin
```

## Architecture

- **Control tile (0,0)**: DMA setup, sum-of-squares reduction, normalization factor computation
- **Compute tile (1,0)**: Element-wise vector scaling `y = x * norm_inv`
- **DMA channels**: 0 (input), 1 (output)
- **Lock**: 0 (control -> compute)

## Guardrail Compliance

1. **Memory-first**: Block-based DMA transfers
2. **Vector intrinsics only**: Vector square, reduce-add, vector-scalar multiply
3. **No branches**: Fixed-size vector operations
4. **Two-layer**: Control tile handles reduction; compute tile handles scaling
