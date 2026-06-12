// PLACEHOLDER — not a working flash-attention transform.
//
// flash_attn Triton kernels use scf.for with loop-carried online-softmax state.
// This file was previously a copy of matmul_aie2p.mlir and cannot lower them.
//
// Production inference uses host f32 decomposed attention (see amd_xdna.cpp).
// To attempt a fused build: GGNPU_EXPERIMENTAL=1 ./scripts/build-kernels.sh npu6 flash_attn_32x64x2048
// Runtime fused path (optional): GGNPU_FLASH_ATTN_FUSED=1
//
// A real recipe needs mlir-air support for attention-style reductions + K/V tiling.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    transform.yield
  }
}
