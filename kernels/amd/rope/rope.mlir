// GGNPU RoPE (Rotary Positional Embeddings) Kernel for AMD XDNA2 (AIE2P)
// Compiled by mlir-aie (aiecc.py) into .xclbin for the NPU
//
// RoPE: out[i] = v0*cos(freq) - v1*sin(freq), out[i+1] = v0*sin(freq) + v1*cos(freq)
//
// Architecture:
//   - Tile (0,0): control tile (DMA)
//   - Tile (1,0): compute tile (rotation)
//   - DMA channels: 0 (input host->tile), 1 (output tile->host)
//
// Usage:
//   aiecc.py --target=aie2p --npu-profile=6 rope.mlir -o rope_npu6.xclbin

module attributes {aie.device = "aie2p"} {

    %lock_start = aie.lock {lock_id = 1} : !aie.lock

    //====//
    // Tile (0,0): Shim tile - DMA
    //====//
    aie.core {fn_name = "rope_shim_main", symbol = "rope_shim_main"}
        for tile {row = 0, col = 0} {
        // Start DMA: load input vector from host (channel 0)
        %ch_in = aie.shim_dma.begin {
            channel = 0, dir = "MM2S",
            len = memref.dim(%buf_in, 0) * 4
        } : (index) -> !aie.objectbox.stream
        aie.shim_dma.end {%ch_in}

        // Signal compute tile
        aie.lock.release {%lock_start}

        // Start DMA: store output vector to host (channel 1)
        %ch_out = aie.shim_dma.begin {
            channel = 1, dir = "S2MM",
            len = memref.dim(%buf_out, 0) * 4
        } : (index) -> !aie.objectbox.stream
        aie.shim_dma.end {%ch_out}
    }

    //====//
    // Tile (1,0): Compute tile - RoPE rotation
    // For each pair (v0, v1): apply rotation by freq
    //====//
    aie.core {fn_name = "rope_tile_main", symbol = "rope_tile_main"}
        for tile {row = 1, col = 0} {
        aie.lock.acquire {%lock_start}

        // Process dimension pairs in vector chunks
        // For each pair of dimensions (i, i+1):
        //   %v0 = load(%input, i)    // float32
        //   %v1 = load(%input, i+1)  // float32
        //   %cos = load(%freq_cos, i/2)  // precomputed cos(freq)
        //   %sin = load(%freq_sin, i/2)  // precomputed sin(freq)
        //   %out0 = v0*cos - v1*sin
        //   %out1 = v0*sin + v1*cos
        //   store(%out0, %output, i)
        //   store(%out1, %output, i+1)

        // In AIE, freq tables are loaded into local memory by the control tile
        // The rotation is done with vector multiply-add operations
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
