# Softmax Kernel for AMD XDNA2 (AIE2P)

## Operation

Softmax: `out[r][c] = exp(in[r][c] - max_r) / sum(exp(in[r][c] - max_r))`

Computes softmax row-by-row for numerical stability.

## Files

| File | Purpose |
|------|---------|
| `softmax.mlir` | MLIR source for mlir-aie compilation to .xclbin |
| `softmax_tile.h` | AIE2P tile compute code (Peano-compiled ELF) |

## Building

```bash
aiecc.py --target=aie2p --npu-profile=6 \
    kernels/amd/softmax/softmax.mlir \
    -o ~/.cache/ggnpu/xclbin/softmax_npu6.xclbin
```

## Architecture

- **Control tile (0,0)**: DMA, per-row max reduction
- **Compute tile (1,0)**: exp computation, divide by sum
- **DMA channels**: 0 (input), 1 (output)
- **Lock**: 2 (control -> compute)
