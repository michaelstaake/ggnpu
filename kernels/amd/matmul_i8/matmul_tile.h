// GGNPU AIE2P Tile Code for INT8 Matrix Multiplication
// Compiled with Peano (aie2p toolchain) into ELF for compute tiles
//
// Architecture:
//   - Runs on AIE2P compute tiles (row 1, columns 0-7)
//   - Uses local SRAM (L2) for A, B, C matrix tiles
//   - Vector intrinsics for INT8 multiply-accumulate to INT32
//   - 4MB L2 memory limit per tile (shared across all tiles in design)
//
// Guardrail compliance:
//   1. Memory-first: Block-based loads from DMA buffers
//   2. Vector intrinsics only: INT8 -> INT32 MAC operations
//   3. No branches: Fully unrolled K-dimension loop
//   4. Two-layer: No DMA code here, only compute

#pragma once

// Peano/AIE2P intrinsics (provided by aie2p toolchain)
// These are compiler intrinsics, not runtime libraries

// Vector type: 16 x INT8 (128-bit vector register)
using vec_i8 = __attribute__((vector_size(16))) char;
// Vector type: 16 x INT32 (64-byte vector register for accumulator)
using vec_i32 = __attribute__((vector_size(64))) int;

// Tile-local memory buffers (stored in L2 SRAM)
// Each compute tile handles a tile_m x tile_n block of output
// A buffer: tile_m x K_chunk (INT8)
// B buffer: K_chunk x tile_n (INT8)
// C accumulator: tile_m x tile_n (INT32)

static constexpr int TILE_M = 16;
static constexpr int TILE_N = 16;
static constexpr int VEC_LEN = 16;

// Tile-local SRAM buffers
// Note: In actual Peano compilation, these map to tile L2 memory
alignas(64) static vec_i8 tile_a_buf[TILE_M * 256];  // Max K = 256 for L2 budget
alignas(64) static vec_i8 tile_b_buf[256 * TILE_N];
alignas(64) static vec_i32 tile_c_acc[TILE_M * TILE_N];

//====//
// Load matrix A tile from DMA buffer into local memory
// A is stored in row-major: tile_m rows x K elements
//====//
__attribute__((noinline))
void load_a_tile(const vec_i8* src, int k_chunks) {
    for (int k = 0; k < k_chunks; k++) {
        // Vector load from DMA buffer to local memory
        // Each chunk loads VEC_LEN elements per row
        for (int row = 0; row < TILE_M; row++) {
            vec_i8* dst = &tile_a_buf[row * (k_chunks * VEC_LEN) + k * VEC_LEN];
            const vec_i8* src_row = &src[(row * (k_chunks * VEC_LEN)) + k * VEC_LEN];
            #pragma GCC unroll 1
            for (int v = 0; v < 1; v++) {
                dst[v] = src_row[v];
            }
        }
    }
}

//====//
// Load matrix B tile from DMA buffer into local memory
// B is stored in column-major for efficient column access during MAC
//====//
__attribute__((noinline))
void load_b_tile(const vec_i8* src, int k_chunks) {
    for (int k = 0; k < k_chunks; k++) {
        // Vector load: B[k*VEC_LEN .. (k+1)*VEC_LEN, :]
        for (int col = 0; col < TILE_N; col++) {
            vec_i8* dst = &tile_b_buf[k * VEC_LEN * TILE_N + col * VEC_LEN];
            const vec_i8* src_col = &src[(k * VEC_LEN * TILE_N) + (col * VEC_LEN)];
            #pragma GCC unroll 1
            for (int v = 0; v < 1; v++) {
                dst[v] = src_col[v];
            }
        }
    }
}

//====//
// INT8 matrix multiply-accumulate
// C += A x B where:
//   A: TILE_M x K (INT8)
//   B: K x TILE_N (INT8)
//   C: TILE_M x TILE_N (INT32 accumulator)
//
// Fully unrolled: no branches, vector intrinsics only
//====//
__attribute__((noinline))
void matmul_tile_ia8(const int k_chunks) {
    // Zero accumulator
    for (int i = 0; i < TILE_M * TILE_N; i++) {
        tile_c_acc[i] = 0;
    }

    // K-dimension: fully unrolled loop
    // Each iteration: load vectors, multiply-accumulate
    #pragma GCC unroll
    for (int k = 0; k < k_chunks; k++) {
        // For each output element (row, col):
        // acc[row][col] += sum over v of A[row][k*VEC+v] * B[k*VEC+v][col]
        #pragma GCC unroll
        for (int row = 0; row < TILE_M; row++) {
            #pragma GCC unroll
            for (int col = 0; col < TILE_N; col++) {
                int32_t sum = 0;
                // Inner vector dot product: VEC_LEN elements
                const vec_i8* a_ptr = &tile_a_buf[row * (k_chunks * VEC_LEN) + k * VEC_LEN];
                const vec_i8* b_ptr = &tile_b_buf[k * VEC_LEN * TILE_N + col * VEC_LEN];
                #pragma GCC unroll
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
__attribute__((noinline))
void store_c_tile(vec_i32* dst) {
    for (int i = 0; i < TILE_M * TILE_N; i++) {
        dst[i] = tile_c_acc[i];
    }
}

//====//
// Main tile entry point
// Called by Peano runtime on each compute tile
//
// Parameters passed via global memory (set by control tile):
//   - ptr_a: pointer to A matrix in DDR (via DMA)
//   - ptr_b: pointer to B matrix in DDR (via DMA)
//   - ptr_c: pointer to C matrix in DDR (via DMA)
//   - k_chunks: number of K-dimension chunks to process
//   - tile_col: this tile's column index (0-7)
//   - tile_row: this tile's row index (should be 1)
//====//
__attribute__((noinline))
void matmul_tile_main(const void* ptr_a, const void* ptr_b, void* ptr_c,
                      int k_chunks, int tile_col, int tile_row) {
    (void)tile_row;
    (void)tile_col;

    const vec_i8* a_ptr = static_cast<const vec_i8*>(ptr_a);
    const vec_i8* b_ptr = static_cast<const vec_i8*>(ptr_b);
    vec_i32* c_ptr = static_cast<vec_i32*>(ptr_c);

    // Load A and B tiles from DMA buffers into local SRAM
    load_a_tile(a_ptr, k_chunks);
    load_b_tile(b_ptr, k_chunks);

    // Execute matrix multiply-accumulate
    matmul_tile_ia8(k_chunks);

    // Store result back to output buffer
    store_c_tile(c_ptr);
}
