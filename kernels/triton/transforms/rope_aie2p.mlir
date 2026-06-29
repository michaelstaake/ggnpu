// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for RoPE (AIE2P)
// rope_kernel(in1_ptr, in2_ptr, out_ptr, n):
//   in1_ptr: [t1[0..n-1]]   (n BF16, own buffer)
//   in2_ptr: [t2[0..n-1]]   (n BF16, own buffer)
//   out_ptr: [out[0..n-1]]  (n BF16)
//   out[i] = t1[i] + t2[i]
//
// HOST precomputes the per-element products:
//   even call: t1 = x_e*cos, t2 = -x_o*sin  → out_e = x_e*cos - x_o*sin
//   odd  call: t1 = x_e*sin, t2 =  x_o*cos  → out_o = x_e*sin + x_o*cos
//
// This is the SAME pipeline the proven silu kernel uses (the only difference is
// @pad_and_promote_binary_bf16 vs unary, because rope is a 2-input add). The
// host pads the real n_pairs (=32) up to ROPE_PAD (=512) so flatten_tile_forall
// (tile_sizes=[256]) yields 512/256 = 2 trips → a 2-core herd, each core working
// a standard 256-element vectorized tile. The earlier inline tile_sizes=[16]
// produced 16-element tiles that the npu6 firmware rejected at execution
// ("unexpected command state").
////////////////////////////////////////////////////////////////////////////////

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {

    transform.include @canonicalize_with_fold_dims failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @fuse_elementwise_and_canonicalize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @flatten_tile_forall failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @pad_and_promote_binary_bf16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    transform.include @vectorize_generics_at_16 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    %vh = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1) : (!transform.any_op) -> !transform.any_op
    transform.include @cast_bf16_only_ops failures(propagate)
        (%vh) : (!transform.any_op) -> ()

    transform.yield
  }
}
