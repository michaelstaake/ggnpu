# RoPE (Rotary Positional Embeddings) Kernel for AMD XDNA2 (AIE2P)

## Operation

RoPE: `out[i] = v0*cos(freq) - v1*sin(freq)`, `out[i+1] = v0*sin(freq) + v1*cos(freq)`

Applied to query and key vectors in attention layers.

## Files

| File | Purpose |
|------|---------|
| `rope.mlir` | MLIR source for mlir-aie compilation to .xclbin |
| `rope_tile.h` | AIE2P tile compute code (Peano-compiled ELF) |

## Building

```bash
# Compile MLIR to xclbin for npu6 (Krackan)
aiecc.py --target=aie2p --npu-profile=6 \
    kernels/amd/rope/rope.mlir \
    -o ~/.cache/ggnpu/xclbin/rope_npu6.xclbin
```

## Architecture

- **Control tile (0,0)**: DMA setup
- **Compute tile (1,0)**: Rotary embedding rotation
- **DMA channels**: 0 (input), 1 (output)
- **Lock**: 1 (control -> compute)

## Notes

RoPE is computationally lightweight compared to matmul operations.
Currently implemented on CPU in the inference loop; NPU kernel is provided
for future optimization when NPU utilization is higher.
