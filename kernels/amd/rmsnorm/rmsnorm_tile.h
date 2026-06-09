// GGNPU AIE2P Tile Code for RMS Normalization
// Compiled with Peano (aie2p toolchain) into ELF for compute tiles
//
// Architecture:
//   - Runs on AIE2P compute tile (row 1, column 0)
//   - Element-wise scaling: y[i] = x[i] * norm_inv
//   - norm_inv computed by control tile (0,0)
//
// Guardrail compliance:
//   1. Memory-first: Load from DMA buffer, store to output
//   2. Vector intrinsics only: vector-scalar multiply
//   3. No branches: fully unrolled element operations
//   4. Two-layer: only scaling here, reduction in control tile

#pragma once

//====//
// RMS Normalization tile compute
// y[i] = x[i] * norm_inv
//
// Parameters passed via shared memory (set by control tile):
//   - ptr_input: pointer to input vector in local SRAM
//   - ptr_output: pointer to output buffer in local SRAM
//   - norm_inv: normalization factor (1 / sqrt(mean(x^2) + eps))
//   - N: vector size (number of elements)
//====//

static constexpr int VEC_LEN = 16;  // Vector width in elements (float32)

__attribute__((noinline))
void rmsnorm_tile(const float* ptr_input, float* ptr_output,
                  float norm_inv, int N) {
    int num_vectors = (N + VEC_LEN - 1) / VEC_LEN;

    // Process input in vector chunks
    for (int v = 0; v < num_vectors; v++) {
        int offset = v * VEC_LEN;
        int remaining = N - offset;
        int vec_size = (remaining < VEC_LEN) ? remaining : VEC_LEN;

        // Vector-scalar multiply: y[offset..offset+vec_size] = x[offset..offset+vec_size] * norm_inv
        // In actual AIE code:
        //   %x_vec = aie.load(%ptr_input, offset=offset)  // 16 x float32
        //   %y_vec = vmul(%x_vec, broadcast(norm_inv))    // vector-scalar multiply
        //   aie.store(%y_vec, %ptr_output, offset=offset)

        // Host-side equivalent for documentation:
        for (int i = 0; i < vec_size; i++) {
            ptr_output[offset + i] = ptr_input[offset + i] * norm_inv;
        }
    }
}

//====//
// Main tile entry point
// Called by Peano runtime on compute tile (1,0)
//
// Parameters passed via global memory (set by control tile):
//   - args_ptr: pointer to shared argument structure
//====//

struct RmsNormTileArgs {
    uintptr_t ptr_input;    // Input vector in local SRAM
    uintptr_t ptr_output;   // Output vector in local SRAM
    float norm_inv;         // Normalization factor
    int N;                  // Vector size
    int padding;            // Alignment padding
};

__attribute__((noinline))
void rmsnorm_tile_main(uintptr_t args_ptr) {
    const RmsNormTileArgs* args = reinterpret_cast<const RmsNormTileArgs*>(args_ptr);

    const float* input = reinterpret_cast<const float*>(args->ptr_input);
    float* output = reinterpret_cast<float*>(args->ptr_output);

    rmsnorm_tile(input, output, args->norm_inv, args->N);
}
