# SiLU (Sigmoid Linear Unit) Kernel for AMD XDNA2 (AIE2P)

## Operation

SiLU: `out[i] = x[i] / (1 + exp(-x[i])) = x[i] * sigmoid(x[i])`

Also known as Swish. Used in FFN paths.

## Files

| File | Purpose |
|------|---------|
| `silu.mlir` | MLIR source for mlir-aie compilation to .xclbin |
| `silu_tile.h` | AIE2P tile compute code (Peano-compiled ELF) |

## Building

```bash
aiecc.py --target=aie2p --npu-profile=6 \
    kernels/amd/silu/silu.mlir \
    -o ~/.cache/ggnpu/xclbin/silu_npu6.xclbin
```

## Architecture

- **Control tile (0,0)**: DMA
- **Compute tile (1,0)**: Element-wise SiLU
- **DMA channels**: 0 (input), 1 (output)
- **Lock**: 4 (control -> compute)
