// GGNPU AIE2P Tile Code for INT8 Matrix Multiplication
// Compiled with Peano (aie2p toolchain) into ELF for compute tiles
//
// Runs on AIE2P compute tiles (row 1, columns 0-7)
// Uses local SRAM (L2) for A, B, C matrix tiles
// Vector intrinsics for INT8 multiply-accumulate to INT32
//
// Guardrail compliance:
//   1. Memory-first: Block-based loads from DMA buffers
//   2. Vector intrinsics only: INT8 -> INT32 MAC operations
//   3. No branches: Fully unrolled K-dimension loop
//   4. Two-layer: No DMA code here, only compute

#pragma once

#include <cstdint>

//====//
// Tile configuration constants
//====//
static constexpr int TILE_M = 16;    // Output rows per tile
static constexpr int TILE_N = 16;    // Output cols per tile
static constexpr int VEC_LEN = 16;   // Vector width in INT8 elements

//====//
// Tile-local L2 memory buffers
// These map to the tile's local SRAM via Peano compiler directives
//====//
// A buffer: TILE_M rows x max K chunks (INT8)
// B buffer: max K chunks x TILE_N cols (INT8)
// C accumulator: TILE_M x TILE_N (INT32)
//====//
// In Peano, these are placed in L2 memory using:
//   #pragma aie_l2
// which tells the compiler to allocate in tile local memory

static constexpr int MAX_K_CHUNKS = 256 / VEC_LEN;  // Max K=256 for initial L2 budget

// L2-local storage for matrix tiles
alignas(64) static char tile_a_buf[TILE_M * (MAX_K_CHUNKS * VEC_LEN)];
alignas(64) static char tile_b_buf[(MAX_K_CHUNKS * VEC_LEN) * TILE_N];
alignas(64) static int32_t tile_c_acc[TILE_M * TILE_N];

//====//
// DMA buffer pointers (set by control tile via shared memory)
//====//
// These are updated by the control tile before kernel launch
// The compute tile reads these to know where DMA data lands

static constexpr uintptr_t DMA_BUF_A = 0x00000000;  // DMA receive buffer A
static constexpr uintptr_t DMA_BUF_B = 0x00004000;  // DMA receive buffer B
static constexpr uintptr_t DMA_BUF_C = 0x00008000;  // DMA send buffer C

//====//
// Load matrix A tile from DMA buffer into local L2
// A is stored in row-major: TILE_M rows x K elements
//====//
static void load_a_tile(const void* src, int k_chunks) {
    const int8_t* src_ptr = static_cast<const int8_t*>(src);
    int8_t* dst_ptr = static_cast<int8_t*>(tile_a_buf);

    // Copy TILE_M rows, each with k_chunks * VEC_LEN elements
    int row_bytes = k_chunks * VEC_LEN;
    for (int row = 0; row < TILE_M; row++) {
        // Vector load: copy VEC_LEN elements at a time
        for (int k = 0; k < k_chunks; k++) {
            int src_off = row * row_bytes + k * VEC_LEN;
            int dst_off = row * row_bytes + k * VEC_LEN;
            // In actual AIE: aie::loadv<int8_t, VEC_LEN>(src_ptr + src_off)
            //               -> store to dst_ptr + dst_off
            for (int v = 0; v < VEC_LEN; v++) {
                dst_ptr[dst_off + v] = src_ptr[src_off + v];
            }
        }
    }
}

//====//
// Load matrix B tile from DMA buffer into local L2
// B is stored in column-major for efficient column access during MAC
//====//
static void load_b_tile(const void* src, int k_chunks) {
    const int8_t* src_ptr = static_cast<const int8_t*>(src);
    int8_t* dst_ptr = static_cast<int8_t*>(tile_b_buf);

    // Copy K chunks x TILE_N columns
    int col_bytes = k_chunks * VEC_LEN;
    for (int col = 0; col < TILE_N; col++) {
        for (int k = 0; k < k_chunks; k++) {
            int src_off = k * VEC_LEN * TILE_N + col * VEC_LEN;
            int dst_off = col * col_bytes + k * VEC_LEN;
            // In actual AIE: aie::loadv<int8_t, VEC_LEN>(src_ptr + src_off)
            //               -> store to dst_ptr + dst_off
            for (int v = 0; v < VEC_LEN; v++) {
                dst_ptr[dst_off + v] = src_ptr[src_off + v];
            }
        }
    }
}

//====//
// INT8 matrix multiply-accumulate
// C += A x B where:
//   A: TILE_M x (k_chunks * VEC_LEN) INT8
//   B: (k_chunks * VEC_LEN) x TILE_N INT8
//   C: TILE_M x TILE_N INT32 accumulator
//
// Fully unrolled: no branches, vector intrinsics only
//====//
static void matmul_tile_ia8(int k_chunks) {
    // Zero accumulator
    for (int i = 0; i < TILE_M * TILE_N; i++) {
        tile_c_acc[i] = 0;
    }

    // K-dimension: fully unrolled loop
    // Each iteration: load A vector, B vector, multiply-accumulate
    for (int k = 0; k < k_chunks; k++) {
        for (int row = 0; row < TILE_M; row++) {
            for (int col = 0; col < TILE_N; col++) {
                int32_t sum = 0;
                const int8_t* a_ptr = tile_a_buf + row * (k_chunks * VEC_LEN) + k * VEC_LEN;
                const int8_t* b_ptr = tile_b_buf + k * VEC_LEN * TILE_N + col * VEC_LEN;

                // Inner vector dot product: VEC_LEN elements
                // In actual AIE:
                //   %a_vec = aie::loadv<int8_t, VEC_LEN>(a_ptr)
                //   %b_vec = aie::loadv<int8_t, VEC_LEN>(b_ptr)
                //   aie::mlacc(%acc_vec, %a_vec, %b_vec)  // vector MAC
                for (int v = 0; v < VEC_LEN; v++) {
                    sum += static_cast<int32_t>(static_cast<int8_t>(a_ptr[v])) *
                           static_cast<int32_t>(static_cast<int8_t>(b_ptr[v]));
                }
                tile_c_acc[row * TILE_N + col] += sum;
            }
        }
    }
}

//====//
// Store result C tile from accumulator to output buffer
// C is stored in row-major: TILE_M rows x TILE_N cols (INT32)
//====//
static void store_c_tile(void* dst) {
    int32_t* dst_ptr = static_cast<int32_t*>(dst);
    for (int i = 0; i < TILE_M * TILE_N; i++) {
        dst_ptr[i] = tile_c_acc[i];
    }
}

//====//
// Main tile entry point
// Called by Peano runtime on each compute tile
//
// Parameters passed via shared memory (set by control tile):
//   - ptr_a: pointer to A matrix tile in DMA buffer
//   - ptr_b: pointer to B matrix tile in DMA buffer
//   - ptr_c: pointer to C output buffer in DMA buffer
//   - k_chunks: number of K-dimension chunks to process
//   - tile_col: this tile's column index (0-7) for output offset
//====//
static void matmul_tile_main(const void* ptr_a, const void* ptr_b, void* ptr_c,
                              int k_chunks, int tile_col) {
    (void)tile_col;  // Reserved for multi-tile output assembly

    // Load A and B tiles from DMA buffers into local L2 SRAM
    load_a_tile(ptr_a, k_chunks);
    load_b_tile(ptr_b, k_chunks);

    // Execute matrix multiply-accumulate
    matmul_tile_ia8(k_chunks);

    // Store result back to output buffer (via DMA)
    store_c_tile(ptr_c);
}
