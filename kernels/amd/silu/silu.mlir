// GGNPU SiLU (Sigmoid Linear Unit) Kernel for AMD XDNA2 (AIE2P)
// Compiled by mlir-aie (aiecc.py) into .xclbin for the NPU
//
// SiLU: out[i] = x[i] / (1 + exp(-x[i])) = x[i] * sigmoid(x[i])
//
// Architecture:
//   - Tile (0,0): control tile (DMA)
//   - Tile (1,0): compute tile (element-wise SiLU)
//   - DMA channels: 0 (input host->tile), 1 (output tile->host)
//
// Usage:
//   aiecc.py --target=aie2p --npu-profile=6 silu.mlir -o silu_npu6.xclbin

module attributes {aie.device = "aie2p"} {

    %lock_start = aie.lock {lock_id = 4} : !aie.lock

    //====//
    // Tile (0,0): Shim tile - DMA
    //====//
    aie.core {fn_name = "silu_shim_main", symbol = "silu_shim_main"}
        for tile {row = 0, col = 0} {
        %ch_in = aie.shim_dma.begin {
            channel = 0, dir = "MM2S",
            len = memref.dim(%buf_in, 0) * 4
        } : (index) -> !aie.objectbox.stream
        aie.shim_dma.end {%ch_in}

        aie.lock.release {%lock_start}

        %ch_out = aie.shim_dma.begin {
            channel = 1, dir = "S2MM",
            len = memref.dim(%buf_out, 0) * 4
        } : (index) -> !aie.objectbox.stream
        aie.shim_dma.end {%ch_out}
    }

    //====//
    // Tile (1,0): Compute tile - element-wise SiLU
    // out[i] = x[i] / (1 + exp(-x[i]))
    //====//
    aie.core {fn_name = "silu_tile_main", symbol = "silu_tile_main"}
        for tile {row = 1, col = 0} {
        aie.lock.acquire {%lock_start}

        // Process in vector chunks
        // For each chunk:
        //   %x_vec = load(%input, offset)  // 16 x float32
        //   %neg_x = vneg(%x_vec)
        //   %exp_neg_x = vexp(%neg_x)     // vector exp
        //   %ones = broadcast(1.0f)
        //   %denom = vadd(%exp_neg_x, %ones)
        //   %y_vec = vdiv(%x_vec, %denom)
        //   store(%y_vec, %output, offset)
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
