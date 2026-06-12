// FlashAttention transform for AIE2P.
// Kernel uses tl.dot for QK^T and AV matmuls, softmax as elementwise ops.
// Transform: canonicalize → fuse_elementwise → tile_output → bufferize → vectorize → herd

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg1: !transform.any_op {transform.readonly}) {

    // PHASE 1: Canonicalization + elementwise fusion
    transform.include @canonicalize_with_fold_dims failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @fuse_elementwise_and_canonicalize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // Transpose reduces for AIE vectorization
    %reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.air.transpose_reduce %reduces : (!transform.any_op) -> !transform.any_op

    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // PHASE 2: Match fill output, navigate to producer generic
    %fill_output = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %output_generic = transform.get_producer_of_operand %fill_output[0] : (!transform.any_op) -> !transform.any_op

    // L2 allocation for output (memory_space = 1)
    %output_l2_buf, %output_l2_new = transform.structured.bufferize_to_allocation %output_generic
        {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Tile output with forall [1] for head_dim parallelism  
    %tiled_output, %forall_out = transform.structured.tile_using_forall %output_generic tile_sizes [1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse fill into forall (like rmsnorm/softmax do)
    %fused_fill, %fill_loop = transform.structured.fuse_into_containing_op %fill_output into %forall_out
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // PHASE 3: Generalize reduces and fuse their producers into forall
    %all_reduces = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %reduce_qk, %reduce_av =
        transform.split_handle %all_reduces {overflow_result = 1}
        : (!transform.any_op<"linalg.reduce">) -> (!transform.any_op, !transform.any_op)

    %reduce_qk_gen = transform.structured.generalize %reduce_qk : (!transform.any_op) -> !transform.any_op
    %reduce_av_gen = transform.structured.generalize %reduce_av : (!transform.any_op) -> !transform.any_op

    // Navigate from generalized reduces to find producers (POST-fusion IR)
    %mul_qk = transform.get_producer_of_operand %reduce_qk_gen[0] : (!transform.any_op) -> !transform.any_op
    %mul_av = transform.get_producer_of_operand %reduce_av_gen[0] : (!transform.any_op) -> !transform.any_op

    // Fuse QK^T chain: reduce → producer into forall
    %fused_qk_r, %qk_l1 = transform.structured.fuse_into_containing_op %reduce_qk_gen into %forall_out
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_qk_m, %qk_l2 = transform.structured.fuse_into_containing_op %mul_qk into %fused_qk_r
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Fuse AV chain: reduce → producer into forall
    %fused_av_r, %av_l1 = transform.structured.fuse_into_containing_op %reduce_av_gen into %forall_out
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_av_m, %av_l2 = transform.structured.fuse_into_containing_op %mul_av into %fused_av_r
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // PHASE 4: L1 allocation for intermediates
    %fills_l1 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fill_l1_buf, %fill_l1_new = transform.structured.bufferize_to_allocation %fills_l1
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    // Promote input tensors to L1
    %generics_l1 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled_gens, %num_gens = transform.split_handle %generics_l1 : (!transform.any_op<"linalg.generic">) -> (!transform.any_op, !transform.any_op)
    %gen_operand0 = transform.get_operand %tiled_gens[0] : (!transform.any_op) -> !transform.any_value
    transform.structured.promote_tensor to 2 %gen_operand0 : !transform.any_value

    %gen_l1_buf, %gen_l1_new = transform.structured.bufferize_to_allocation %tiled_gens
        {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op

    transform.include @canonicalize_with_cse failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // PHASE 5: Bufferization
    transform.include @one_shot_bufferize failures(propagate)
        (%arg1) : (!transform.any_op) -> ()
    transform.include @post_bufferize_cleanup failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // PHASE 6: Vectorization at 32 (AIE2P vector width)
    transform.include @vectorize_generics_at_32 failures(propagate)
        (%arg1) : (!transform.any_op) -> ()

    // PHASE 7: AIR herd mapping and vectorization
    %vh = transform.include @air_herd_mapping_and_vectorize
        failures(propagate) (%arg1) : (!transform.any_op) -> !transform.any_op

    // PHASE 8: Type casting for AIE2P — bf16-only ops cast from f32 to bf16
    %vector_reductions = transform.structured.match ops{["vector.multi_reduction"]} in %vh : (!transform.any_op) -> !transform.any_op
    %cast_reductions = transform.air.vector_type_cast %vector_reductions {target_element_type = bf16}
        : (!transform.any_op) -> !transform.any_op

    %vector_exps = transform.structured.match ops{["math.exp"]} in %vh : (!transform.any_op) -> !transform.any_op
    %cast_exps = transform.air.vector_type_cast %vector_exps {target_element_type = bf16}
        : (!transform.any_op) -> !transform.any_op

    // PHASE 9: Post-vectorize cleanup for reductions
    %func5 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_scld = transform.air.convert_size1_vector_to_scalar %func5 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_scld {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.vector.cast_away_vector_leading_one_dim
        transform.apply_patterns.vector.reorder_multi_reduction_dims lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.multi_reduction_flattening lowering_strategy = "innerreduction"
        transform.apply_patterns.vector.multi_reduction_unrolling lowering_strategy = "innerreduction"
    } : !transform.any_op
    transform.apply_cse to %func_scld : !transform.any_op

    // PHASE 10: Final optimization
    %func6 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func6 {
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
        transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    %func_folded = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %func_final = transform.air.fold_unit_extent_dims %func_folded : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
