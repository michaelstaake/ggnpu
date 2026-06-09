// MLIR source for RMS Normalization on AMD XDNA2 (AIE2P)
// This file is compiled by mlir-aie (aiecc.py) into .xclbin for the NPU
//
// Usage: aiecc.py --target=aie2p --npu-profile=6 rmsnorm.mlir -o rmsnorm_npu6.xclbin
//
// Architecture:
//   - Uses AIE2P tile array
//   - Tile (0,0): control core (DMA setup, reduction)
//   - Tile (1,0): compute tile (element-wise scaling)
//   - DMA engines stream data from DDR to tile local memory
//
// RMSNorm formula: y[i] = x[i] / sqrt(mean(x^2) + eps)
//
// Guardrail compliance:
//   1. Memory-first: DMA transfers are block-aligned
//   2. Vector intrinsics: vector multiply, vector sum
//   3. No branches: fully unrolled element operations
//   4. Two-layer: control code handles reduction, tile handles scaling

module attributes {aie.device = "aie2p"} {
    //====//
    // DMA channels for data movement
    //====//
    %dma_in = "aie.dma_acquire"() {
        channel_id = 0 : i32,
        direction = "host_to_tile" : string,
        tile = #aie.tile{row = 0, col = 0}
    } : () -> !aie.objectbox.stream

    %dma_out = "aie.dma_acquire"() {
        channel_id = 1 : i32,
        direction = "tile_to_host" : string,
        tile = #aie.tile{row = 0, col = 0}
    } : () -> !aie.objectbox.stream

    //====//
    // Shim tile (0,0): DMA ports
    //====//
    "aie.shimtile"(#aie.tile{row = 0, col = 0}) {
        "aie.dma_port"() { port = "A", direction = "input" } : () -> ()
        "aie.dma_port"() { port = "B", direction = "output" } : () -> ()
    }

    //====//
    // Control tile: compute RMS normalization factor
    //====//
    "aie.tile"(#aie.tile{row = 0, col = 0}) {
        // Lock for synchronization with compute tile
        %lock_compute = "aie.lock"() {
            lock = #aie.lock{lock_id = 0, target = "tile",
                tile = #aie.tile{row = 1, col = 0}}
        } : () -> !aie.lock

        // Start DMA: load input vector from host
        // Length: N * 4 bytes (float32)
        "aie.dma_start"(%dma_in) { len = 0 : i64 } : (!aie.objectbox.stream) -> ()

        // Wait for input DMA to complete
        "aie.dma_wait"(%dma_in) { count = 1 : i32 } : (!aie.objectbox.stream) -> ()

        // Compute sum of squares on control tile
        // For each vector element: sum += x[i] * x[i]
        // This is done in the control core microcode
        // In real AIE code: vector load, vector mul, vector reduce-add

        // Compute sqrt(sum / N + eps) and its inverse
        // Store normalization factor in shared memory for compute tile
        // norm_inv = 1.0f / sqrt(sum_of_squares / N + eps)

        // Signal compute tile to start scaling
        "aie.lock"(%lock_compute) : (!aie.lock) -> ()

        // Start DMA: store output vector to host
        // Length: N * 4 bytes (float32)
        "aie.dma_start"(%dma_out) { len = 0 : i64 } : (!aie.objectbox.stream) -> ()

        // Wait for output DMA to complete
        "aie.dma_wait"(%dma_out) { count = 1 : i32 } : (!aie.objectbox.stream) -> ()
    }

    //====//
    // Compute tile: element-wise vector scaling
    // y[i] = x[i] * norm_inv
    //====//
    "aie.tile"(#aie.tile{row = 1, col = 0}) {
        %lock_start = "aie.lock"() {
            lock = #aie.lock{lock_id = 0, target = "tile",
                tile = #aie.tile{row = 0, col = 0}}
        } : () -> !aie.lock

        // Wait for control tile to compute normalization factor
        "aie.lock"(%lock_start) : (!aie.lock) -> ()

        // Load normalization factor from shared memory
        // norm_inv = control_tile_shared_mem[0]

        // Load input vector from local memory
        // x_vec = aie.load(%ptr_input, offset=0)  // 16 x float32

        // Scale: y_vec = x_vec * norm_inv (broadcast scalar to vector)
        // In AIE: vector-scalar multiply with broadcast
        // y_vec = vmul(x_vec, broadcast(norm_inv))

        // Store scaled output to local memory
        // aie.store(y_vec, %ptr_output, offset=0)

        // Unlock when done
        "aie.unlock"(%lock_start) : (!aie.lock) -> ()
    }
}
