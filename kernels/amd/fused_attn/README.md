# FlashAttention Kernel for AMD XDNA2 (AIE2P)

## Operation

FlashAttention v1 (decomposed):
```
attn = softmax(Q @ K^T / sqrt(head_dim)) @ V
```

Three stages: QK^T matmul, softmax per head, weighted V sum.

## Files

| File | Purpose |
|------|---------|
| `flash_attn.mlir` | MLIR source for mlir-aie compilation to .xclbin |
| `flash_attn_tile.h` | AIE2P tile compute code (Peano-compiled ELF) |

## Building

```bash
aiecc.py --target=aie2p --npu-profile=6 \
    kernels/amd/fused_attn/flash_attn.mlir \
    -o ~/.cache/ggnpu/xclbin/flash_attn_npu6.xclbin
```

## Architecture

- **Control tile (0,0)**: DMA for Q, K, V, output
- **Compute tile (1,0)**: QK^T matmul, softmax, weighted V sum
- **DMA channels**: 0 (Q), 1 (K), 2 (V), 3 (output)
- **Lock**: 5 (control -> compute)

## Notes

This is a decomposed v1 implementation. A fully fused v2 would combine
all operations into a single kernel pass for better performance.
