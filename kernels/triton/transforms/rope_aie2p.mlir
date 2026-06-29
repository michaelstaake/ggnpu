// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

////////////////////////////////////////////////////////////////////////////////
// Transform Script for RoPE (AIE2P)
// rope_kernel(x_ptr, cs_ptr, out_ptr, n_pairs):
//   x_ptr:  [x_even[0..n-1] | x_odd[0..n-1]]   (2*n_pairs BF16, stride-1)
//   cs_ptr: [cos[0..n-1]    | sin[0..n-1]]      (2*n_pairs BF16, stride-1)
//   out_ptr:[out_e[0..n-1]  | out_o[0..n-1]]    (2*n_pairs BF16, stride-1)
//
// Binary op (x_ptr + cs_ptr → out_ptr). Fuse elementwise first to collapse
// 4 slice-loads into one linalg.generic, then apply binary BF16 promotion.
// Vec tile = 16 (BF16 x 32 = 512-bit AIE2P vector width).
// Uses shared library sequences from transform_library.mlir (auto-injected).
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
