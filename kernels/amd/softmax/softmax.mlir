// GGNPU Softmax Kernel for AMD XDNA2 (AIE2P)
// Compiled by mlir-aie (aiecc.py) into .xclbin for the NPU
//
// Softmax: out[r][c] = exp(in[r][c] - max_r) / sum(exp(in[r][c] - max_r))
// Computes softmax row-by-row for numerical stability
//
// Architecture:
//   - Tile (0,0): control tile (DMA, per-row max reduction)
//   - Tile (1,0): compute tile (exp, divide)
//   - DMA channels: 0 (input host->tile), 1 (output tile->host)
//
// Usage:
//   aiecc.py --target=aie2p --npu-profile=6 softmax.mlir -o softmax_npu6.xclbin

module attributes {aie.device = "aie2p"} {

    %lock_start = aie.lock {lock_id = 2} : !aie.lock

    //====//
    // Tile (0,0): Shim tile - DMA
    //====//
    aie.core {fn_name = "softmax_shim_main", symbol = "softmax_shim_main"}
        for tile {row = 0, col = 0} {
        // Start DMA: load input matrix from host (channel 0)
        %ch_in = aie.shim_dma.begin {
            channel = 0, dir = "MM2S",
            len = memref.dim(%buf_in, 0) * memref.dim(%buf_in, 1) * 4
        } : (index) -> !aie.objectbox.stream
        aie.shim_dma.end {%ch_in}

        // Compute per-row max and sum on control tile
        // For each row r:
        //   max_r = max(in[r][c] for c in [0, cols))
        //   sum_r = sum(exp(in[r][c] - max_r) for c in [0, cols))
        // Store (max_r, sum_r) to shared memory

        // Signal compute tile
        aie.lock.release {%lock_start}

        // Start DMA: store output matrix to host (channel 1)
        %ch_out = aie.shim_dma.begin {
            channel = 1, dir = "S2MM",
            len = memref.dim(%buf_out, 0) * memref.dim(%buf_out, 1) * 4
        } : (index) -> !aie.objectbox.stream
        aie.shim_dma.end {%ch_out}
    }

    //====//
    // Tile (1,0): Compute tile - exp and divide
    // out[r][c] = exp(in[r][c] - max_r) / sum_r
    //====//
    aie.core {fn_name = "softmax_tile_main", symbol = "softmax_tile_main"}
        for tile {row = 1, col = 0} {
        aie.lock.acquire {%lock_start}

        // Load max_r, sum_r from shared memory
        // For each row r:
        //   %max_r = load(shared_mem, 2*r)
        //   %sum_r = load(shared_mem, 2*r+1)
        // For each column c:
        //   %x = load(%input, r*cols+c)
        //   %exp_x = exp(x - max_r)
        //   store(%exp_x, %output, r*cols+c)

        // Second pass: divide by sum
        // For each element:
        //   %val = load(%output, idx)
        //   %out = val / sum_r
        //   store(%out, %output, idx)
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
    %buf_in = memref.alloc() : memref<?x?xi32>
    %buf_out = memref.alloc() : memref<?x?xi32>

}
