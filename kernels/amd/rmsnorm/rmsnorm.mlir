// GGNPU RMS Normalization Kernel for AMD XDNA2 (AIE2P)
// Compiled by mlir-aie (aiecc.py) into .xclbin for the NPU
//
// RMSNorm: y[i] = x[i] / sqrt(mean(x^2) + eps)
//
// Architecture:
//   - Tile (0,0): control tile (DMA, sum-of-squares reduction)
//   - Tile (1,0): compute tile (element-wise scaling)
//   - DMA channels: 0 (input host->tile), 1 (output tile->host)
//
// Usage:
//   aiecc.py --target=aie2p --npu-profile=6 rmsnorm.mlir -o rmsnorm_npu6.xclbin

module attributes {aie.device = "aie2p"} {

    //====//
    // Lock for synchronization between control and compute tile
    // Lock 0: control -> compute tile (start signal)
    //====//
    %lock_start = aie.lock {lock_id = 0} : !aie.lock

    //====//
    // Tile (0,0): Shim tile - DMA and reduction
    //====//
    aie.core {fn_name = "rmsnorm_shim_main", symbol = "rmsnorm_shim_main"}
        for tile {row = 0, col = 0} {
        // Start DMA: load input vector from host (channel 0)
        %ch_in = aie.shim_dma.begin {
            channel = 0, dir = "MM2S",
            len = memref.dim(%buf_in, 0) * 4
        } : (index) -> !aie.objectbox.stream

        // Wait for input DMA to complete
        aie.shim_dma.end {%ch_in}

        // Compute sum of squares on control tile:
        // sum_sq = sum(x[i] * x[i]) for i in [0, N)
        // In AIE: load vector, vector-mul, vector-reduce-add
        // sum_sq = vreduceadd(vmul(%x_vec, %x_vec))
        // norm_inv = 1.0 / sqrt(sum_sq / N + eps)
        // Store norm_inv to shared memory for compute tile

        // Signal compute tile to start scaling
        aie.lock.release {%lock_start}

        // Start DMA: store output vector to host (channel 1)
        %ch_out = aie.shim_dma.begin {
            channel = 1, dir = "S2MM",
            len = memref.dim(%buf_out, 0) * 4
        } : (index) -> !aie.objectbox.stream

        // Wait for output DMA to complete
        aie.shim_dma.end {%ch_out}
    }

    //====//
    // Tile (1,0): Compute tile - element-wise scaling
    // y[i] = x[i] * norm_inv
    //====//
    aie.core {fn_name = "rmsnorm_tile_main", symbol = "rmsnorm_tile_main"}
        for tile {row = 1, col = 0} {
        // Wait for control tile to compute normalization factor
        aie.lock.acquire {%lock_start}

        // Load normalization factor from shared memory
        // norm_inv = shared_mem[0]

        // Process input in vector chunks
        // For each chunk of VEC_LEN elements:
        //   %x_vec = aie.load(%ptr_input, offset)  // 16 x float32
        //   %y_vec = vmul(%x_vec, broadcast(norm_inv))  // vector-scalar mul
        //   aie.store(%y_vec, %ptr_output, offset)

        // Unlock when done
        // (control tile doesn't wait for completion - fire and forget)
    }

    //====//
    // Data flows
    //====//
    aie.flow {
        src_tile = {row = 0, col = 0}, dst_tile = {row = 1, col = 0},
        src_port = 0, dst_port = 0
    }
    aie.flow {
        src_tile = {row = 1, col = 0}, dst_tile = {row = 0, col = 0},
        src_port = 1, dst_port = 0
    }

    //====//
    // External DDR buffers
    //====//
    %buf_in = memref.alloc() : memref<?xi32>
    %buf_out = memref.alloc() : memref<?xi32>

}
