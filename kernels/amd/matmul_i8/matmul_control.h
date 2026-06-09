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
//
// Usage:
//   1. Control tile receives kernel arguments from host via MMIO
//   2. Sets up DMA transfers for A and B matrices
//   3. Launches compute tiles via lock signaling
//   4. Waits for compute completion
//   5. Initiates DMA to read back C matrix

#pragma once

//====//
// Kernel argument structure (passed via shared memory from host)
//====//
struct MatmulKernelArgs {
    // Input pointers (DDR addresses)
    uint64_t ptr_a;   // A matrix: M x K (INT8, row-major)
    uint64_t ptr_b;   // B matrix: K x N (INT8, row-major)
    uint64_t ptr_c;   // C matrix: M x N (INT32, row-major, accumulator)

    // Dimensions
    uint32_t M;       // Output rows
    uint32_t N;       // Output columns
    uint32_t K;       // Reduction dimension

    // Tile configuration
    uint32_t tile_col;   // This tile's column (0-7)
    uint32_t tile_row;   // This tile's row (always 1 for compute)

    // Computed
    uint32_t k_chunks;  // K / VEC_LEN (rounded up)
    uint32_t padding;   // Alignment padding
};

//====//
// DMA setup for A matrix transfer (host -> tile)
// A matrix: M x K INT8, row-major layout
//====//
__attribute__((noinline))
void setup_dma_a(const MatmulKernelArgs* args) {
    // DMA channel 0: A matrix from host to tile (0,0)
    // Length: M * K bytes
    uint64_t len_a = static_cast<uint64_t>(args->M) * static_cast<uint64_t>(args->K);

    // In actual AIE code, this would call:
    // aie::dma::start(channel_0, args->ptr_a, len_a)
    // For now, document the parameters:
    //   - Channel: 0 (A matrix)
    //   - Source: args->ptr_a (host DDR)
    //   - Destination: tile (0,0) shim buffer
    //   - Length: len_a bytes
    //   - Direction: host_to_tile
    (void)len_a;
}

//====//
// DMA setup for B matrix transfer (host -> tile)
// B matrix: K x N INT8, row-major layout
//====//
__attribute__((noinline))
void setup_dma_b(const MatmulKernelArgs* args) {
    // DMA channel 1: B matrix from host to tile (0,0)
    // Length: K * N bytes
    uint64_t len_b = static_cast<uint64_t>(args->K) * static_cast<uint64_t>(args->N);

    // In actual AIE code:
    // aie::dma::start(channel_1, args->ptr_b, len_b)
    (void)len_b;
}

//====//
// DMA setup for C matrix transfer (tile -> host)
// C matrix: M x N INT32, row-major layout
//====//
__attribute__((noinline))
void setup_dma_c(const MatmulKernelArgs* args) {
    // DMA channel 2: C matrix from tile (0,0) to host
    // Length: M * N * 4 bytes (INT32)
    uint64_t len_c = static_cast<uint64_t>(args->M) * static_cast<uint64_t>(args->N) * 4;

    // In actual AIE code:
    // aie::dma::start(channel_2, tile_local_c_buffer, len_c)
    // Destination: args->ptr_c (host DDR)
    (void)len_c;
}

//====//
// Synchronization: signal compute tiles to start
// Uses lock mechanism: control tile releases lock 3
// Compute tiles are waiting on lock 3
//====//
__attribute__((noinline))
void signal_compute_start() {
    // Release lock 3 to signal compute tiles (1,0) through (1,7)
    // In actual AIE code:
    // aie::lock::release(lock_3)
}

//====//
// Synchronization: wait for compute tiles to complete
// Compute tiles release their completion lock when done
//====//
__attribute__((noinline))
void wait_compute_complete() {
    // Wait for compute tiles to signal completion
    // In actual AIE code:
    // aie::lock::acquire(lock_completion)
}

//====//
// Launch DMA transfers and compute
// Main control flow for matrix multiplication
//====//
__attribute__((noinline))
void launch_matmul(const MatmulKernelArgs* args) {
    // Step 1: Set up DMA transfers
    setup_dma_a(args);
    setup_dma_b(args);

    // Step 2: Start DMA (A and B transfer in parallel)
    // aie::dma::start(channel_0, ...)
    // aie::dma::start(channel_1, ...)

    // Step 3: Signal compute tiles to start
    signal_compute_start();

    // Step 4: Wait for compute to complete
    wait_compute_complete();

    // Step 5: Start DMA to read back C matrix
    setup_dma_c(args);
    // aie::dma::start(channel_2, ...)

    // Step 6: Wait for DMA completion
    // aie::dma::wait(channel_2)
}

//====//
// Control tile main entry point
// Called by Peano runtime on control tile (0,0)
//
// Kernel arguments are passed via MMIO registers or
// shared memory location known to both host and tile
//====//
__attribute__((noinline))
void matmul_control_main(uintptr_t args_ptr) {
    const MatmulKernelArgs* args = reinterpret_cast<const MatmulKernelArgs*>(args_ptr);

    // Compute K chunks from K dimension
    // VEC_LEN = 16 (defined in matmul_tile.h)
    constexpr int VEC_LEN = 16;
    uint32_t k_chunks = (args->K + VEC_LEN - 1) / VEC_LEN;

    // Store k_chunks in args for compute tiles to read
    const MatmulKernelArgs mutable_args = {
        .ptr_a = args->ptr_a,
        .ptr_b = args->ptr_b,
        .ptr_c = args->ptr_c,
        .M = args->M,
        .N = args->N,
        .K = args->K,
        .tile_col = args->tile_col,
        .tile_row = args->tile_row,
        .k_chunks = k_chunks,
        .padding = 0
    };

    // Launch the matmul kernel
    launch_matmul(&mutable_args);
}
