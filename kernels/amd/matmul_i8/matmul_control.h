// GGNPU Control Code for INT8 Matrix Multiplication
// Runs on control tile (tile 0,0) - handles DMA and synchronization
//
// Architecture:
//   - Control tile manages DMA transfers and compute tile launch
//   - Sets up kernel arguments via shared memory
//   - Coordinates data movement and computation
//
// Guardrail compliance:
//   1. Two-layer: DMA setup here, tensor math in matmul_tile.h
//   2. No branches in compute path: DMA lengths set once, launched sequentially

#pragma once

#include <cstdint>
#include <cstring>

//====//
// Kernel argument structure (passed via shared memory from host)
// Layout matches what amd_xdna.cpp sets up via XRT kernel args
//====//
struct MatmulKernelArgs {
    // Input pointers (DDR addresses, set by host)
    uint64_t ptr_a;   // A matrix: M x K (INT8, row-major)
    uint64_t ptr_b;   // B matrix: K x N (INT8, row-major)
    uint64_t ptr_c;   // C matrix: M x N (INT32, row-major, accumulator)

    // Dimensions
    uint32_t M;       // Output rows
    uint32_t N;       // Output columns
    uint32_t K;       // Reduction dimension

    // Tile configuration
    uint32_t k_chunks;  // K / VEC_LEN (rounded up)
    uint32_t tile_col;   // This tile's column (0-7)
    uint32_t tile_row;   // This tile's row (always 1 for compute)
    uint32_t padding;    // Alignment padding
};

//====//
// DMA channel IDs (must match mlir-aie flow configuration)
// Channel 0: A matrix host -> shim tile (MM2S)
// Channel 1: B matrix host -> shim tile (MM2S)
// Channel 2: C matrix shim tile -> host (S2MM)
//====//
static constexpr int DMA_CH_A = 0;
static constexpr int DMA_CH_B = 1;
static constexpr int DMA_CH_C = 2;

//====//
// Lock IDs (must match mlir-aie lock configuration)
// Lock 3: control -> compute tiles (start signal)
// Lock 4: compute tiles -> control (completion signal)
//====//
static constexpr int LOCK_START = 3;
static constexpr int LOCK_DONE = 4;

//====//
// DMA setup for A matrix transfer (host -> tile)
// A matrix: M x K INT8, row-major layout
//====//
static void setup_dma_a(const MatmulKernelArgs* args) {
    // DMA channel 0: A matrix from host to tile (0,0)
    uint64_t len_a = static_cast<uint64_t>(args->M) * static_cast<uint64_t>(args->K);

    // In actual AIE IRON API:
    //   XAie_DmaStart(&xaie, TILE_ROW, TILE_COL, DMA_CH_A,
    //                 XAIE_DMA_DOWN_DIR, args->ptr_a, len_a, 0, 0);
    //
    // The shim tile's DMA engine handles:
    //   1. Reading from host DDR at ptr_a
    //   2. Transferring to tile local memory / L2
    //   3. Signaling completion via lock
    (void)len_a;
}

//====//
// DMA setup for B matrix transfer (host -> tile)
// B matrix: K x N INT8, row-major layout
//====//
static void setup_dma_b(const MatmulKernelArgs* args) {
    // DMA channel 1: B matrix from host to tile (0,0)
    uint64_t len_b = static_cast<uint64_t>(args->K) * static_cast<uint64_t>(args->N);

    // In actual AIE IRON API:
    //   XAie_DmaStart(&xaie, TILE_ROW, TILE_COL, DMA_CH_B,
    //                 XAIE_DMA_DOWN_DIR, args->ptr_b, len_b, 0, 0);
    (void)len_b;
}

//====//
// DMA setup for C matrix transfer (tile -> host)
// C matrix: M x N INT32, row-major layout
//====//
static void setup_dma_c(const MatmulKernelArgs* args) {
    // DMA channel 2: C matrix from tile (0,0) to host
    uint64_t len_c = static_cast<uint64_t>(args->M) * static_cast<uint64_t>(args->N) * 4;

    // In actual AIE IRON API:
    //   XAie_DmaStart(&xaie, TILE_ROW, TILE_COL, DMA_CH_C,
    //                 XAIE_DMA_UP_DIR, args->ptr_c, len_c, 0, 0);
    (void)len_c;
}

//====//
// Synchronization: signal compute tiles to start
// Uses lock mechanism: control tile releases lock 3
// Compute tiles are waiting on lock 3
//====//
static void signal_compute_start() {
    // In actual AIE IRON API:
    //   XAie_LockRelease(&xaie, TILE_ROW, TILE_COL, LOCK_START);
    // This unblocks all compute tiles waiting on lock 3
}

//====//
// Synchronization: wait for compute tiles to complete
// Compute tiles release lock 4 when done
//====//
static void wait_compute_complete() {
    // In actual AIE IRON API:
    //   XAie_LockAcquire(&xaie, TILE_ROW, TILE_COL, LOCK_DONE);
    // Blocks until all compute tiles release lock 4
}

//====//
// Wait for DMA transfer completion
//====//
static void wait_dma_complete(int channel) {
    // In actual AIE IRON API:
    //   XAie_DmaWait(&xaie, TILE_ROW, TILE_COL, channel, 1);
    (void)channel;
}

//====//
// Launch DMA transfers and compute
// Main control flow for matrix multiplication
//====//
static void launch_matmul(const MatmulKernelArgs* args) {
    // Step 1: Set up DMA transfers
    setup_dma_a(args);
    setup_dma_b(args);

    // Step 2: Start DMA (A and B transfer in parallel)
    // In AIE IRON API:
    //   XAie_DmaStart(..., DMA_CH_A, ...)
    //   XAie_DmaStart(..., DMA_CH_B, ...)

    // Step 3: Signal compute tiles to start
    signal_compute_start();

    // Step 4: Wait for compute to complete
    wait_compute_complete();

    // Step 5: Start DMA to read back C matrix
    setup_dma_c(args);
    // In AIE IRON API:
    //   XAie_DmaStart(..., DMA_CH_C, ...)

    // Step 6: Wait for DMA completion
    wait_dma_complete(DMA_CH_C);
}

//====//
// Control tile main entry point
// Called by Peano runtime on control tile (0,0)
//
// Kernel arguments are passed via MMIO registers or
// shared memory location known to both host and tile
//====//
static void matmul_control_main(uintptr_t args_ptr) {
    const MatmulKernelArgs* args =
        reinterpret_cast<const MatmulKernelArgs*>(args_ptr);

    // Launch the matmul kernel
    launch_matmul(args);
}
