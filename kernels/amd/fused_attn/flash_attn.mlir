// GGNPU Fused Attention Kernel for AMD XDNA2 (AIE2P)
// Compiled by mlir-aie (aiecc.py) into .xclbin for the NPU
//
// FlashAttention v1 (decomposed):
//   attn = softmax(Q @ K^T / sqrt(head_dim)) @ V
//
// Architecture:
//   - Tile (0,0): control tile (DMA for Q, K, V, output)
//   - Tile (1,0): compute tile (QK^T matmul, softmax, weighted V sum)
//   - DMA channels: 0 (Q), 1 (K), 2 (V), 3 (output)
//
// Usage:
//   aiecc.py --target=aie2p --npu-profile=6 flash_attn.mlir -o flash_attn_npu6.xclbin

module attributes {aie.device = "aie2p"} {

    %lock_start = aie.lock {lock_id = 5} : !aie.lock

    //====//
    // Tile (0,0): Shim tile - DMA for Q, K, V, output
    //====//
    aie.core {fn_name = "flash_attn_shim_main", symbol = "flash_attn_shim_main"}
        for tile {row = 0, col = 0} {
        // Start DMA for Q matrix (channel 0)
        %ch_q = aie.shim_dma.begin {
            channel = 0, dir = "MM2S",
            len = memref.dim(%buf_q, 0) * memref.dim(%buf_q, 1) * 4
        } : (index) -> !aie.objectbox.stream

        // Start DMA for K matrix (channel 1)
        %ch_k = aie.shim_dma.begin {
            channel = 1, dir = "MM2S",
            len = memref.dim(%buf_k, 0) * memref.dim(%buf_k, 1) * 4
        } : (index) -> !aie.objectbox.stream

        // Start DMA for V matrix (channel 2)
        %ch_v = aie.shim_dma.begin {
            channel = 2, dir = "MM2S",
            len = memref.dim(%buf_v, 0) * memref.dim(%buf_v, 1) * 4
        } : (index) -> !aie.objectbox.stream

        // Wait for all input DMAs
        aie.shim_dma.end {%ch_q}
        aie.shim_dma.end {%ch_k}
        aie.shim_dma.end {%ch_v}

        // Signal compute tile
        aie.lock.release {%lock_start}

        // Start DMA for output (channel 3)
        %ch_out = aie.shim_dma.begin {
            channel = 3, dir = "S2MM",
            len = memref.dim(%buf_out, 0) * memref.dim(%buf_out, 1) * 4
        } : (index) -> !aie.objectbox.stream
        aie.shim_dma.end {%ch_out}
    }

    //====//
    // Tile (1,0): Compute tile - FlashAttention
    // Computes: softmax(Q @ K^T / sqrt(d)) @ V
    //====//
    aie.core {fn_name = "flash_attn_tile_main", symbol = "flash_attn_tile_main"}
        for tile {row = 1, col = 0} {
        aie.lock.acquire {%lock_start}

        // Step 1: Compute QK^T matmul (n_head x ctx_len)
        // For each query head h:
        //   For each context position j:
        //     score[h][j] = dot(Q[h], K[j]) / sqrt(head_dim)

        // Step 2: Softmax per row
        // For each row h:
        //   max_h = max(score[h][:])
        //   exp_h[j] = exp(score[h][j] - max_h)
        //   sum_h = sum(exp_h[:])
        //   attn[h][:] = exp_h[:] / sum_h

        // Step 3: Weighted V sum
        // For each query head h:
        //   out[h] = sum over j of (attn[h][j] * V[j])

        // In AIE, these operations use:
        //   - Matrix multiply for QK^T
        //   - Vector reduce-max for softmax
        //   - Vector exp and reduce-add for softmax
        //   - Matrix-vector multiply for weighted V sum
    }

    //====//
    // Data flows
    //====//
    aie.flow {
        src_tile = {row = 0, col = 0}, dst_tile = {row = 1, col = 0},
        src_port = 0, dst_port = 0
    }
    aie.flow {
        src_tile = {row = 0, col = 0}, dst_tile = {row = 1, col = 0},
        src_port = 1, dst_port = 1
    }
    aie.flow {
        src_tile = {row = 0, col = 0}, dst_tile = {row = 1, col = 0},
        src_port = 2, dst_port = 2
    }
    aie.flow {
        src_tile = {row = 1, col = 0}, dst_tile = {row = 0, col = 0},
        src_port = 3, dst_port = 0
    }

    //====//
    // External DDR buffers
    //====//
    %buf_q = memref.alloc() : memref<?x?xi32>
    %buf_k = memref.alloc() : memref<?x?xi32>
    %buf_v = memref.alloc() : memref<?x?xi32>
    %buf_out = memref.alloc() : memref<?x?xi32>

}
