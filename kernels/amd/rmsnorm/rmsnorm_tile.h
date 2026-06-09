// GGNPU AIE2P Tile Code for RMS Normalization
// Compiled with Peano (aie2p toolchain) into ELF for compute tiles
//
// Element-wise scaling: y[i] = x[i] * norm_inv
// norm_inv computed by control tile (0,0)

#pragma once

#include <cstdint>
#include <cmath>

static constexpr int VEC_LEN = 16;  // Vector width in float32 elements

//====//
// RMS Normalization tile compute
// y[i] = x[i] * norm_inv
//====//
static void rmsnorm_tile(const float* ptr_input, float* ptr_output,
                          float norm_inv, int N) {
    int num_vectors = (N + VEC_LEN - 1) / VEC_LEN;

    for (int v = 0; v < num_vectors; v++) {
        int offset = v * VEC_LEN;
        int remaining = N - offset;
        int vec_size = (remaining < VEC_LEN) ? remaining : VEC_LEN;

        // Vector-scalar multiply: y[offset..offset+vec_size] = x[offset..offset+vec_size] * norm_inv
        // In actual AIE:
        //   %x_vec = aie::loadv<float, VEC_LEN>(ptr_input + offset)
        //   %y_vec = vmul(%x_vec, broadcast(norm_inv))
        //   aie::store(%y_vec, ptr_output + offset)
        for (int i = 0; i < vec_size; i++) {
            ptr_output[offset + i] = ptr_input[offset + i] * norm_inv;
        }
    }
}

//====//
// Control tile: compute RMS normalization factor
// norm_inv = 1.0f / sqrt(sum(x[i]^2) / N + eps)
//====//
static float compute_rms_norm_factor(const float* input, int N, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_sq += input[i] * input[i];
    }
    float variance = sum_sq / static_cast<float>(N) + eps;
    float rms = std::sqrt(variance);
    return 1.0f / rms;
}

//====//
// Kernel argument structure (passed via shared memory from host)
//====//
struct RmsNormKernelArgs {
    uint64_t ptr_input;   // Input vector in DDR (float32, N elements)
    uint64_t ptr_output;  // Output vector in DDR (float32, N elements)
    uint32_t N;           // Vector size
    float eps;            // Epsilon for numerical stability
    uint32_t padding;     // Alignment padding
};

//====//
// DMA channel IDs
// Channel 0: input host -> tile (MM2S)
// Channel 1: output tile -> host (S2MM)
//====//
static constexpr int DMA_CH_IN = 0;
static constexpr int DMA_CH_OUT = 1;

//====//
// Lock IDs
// Lock 0: control -> compute tile (start signal)
//====//
static constexpr int LOCK_START = 0;

//====//
// DMA setup for input vector transfer (host -> tile)
//====//
static void setup_dma_input(const RmsNormKernelArgs* args) {
    uint64_t len = static_cast<uint64_t>(args->N) * 4;
    // In actual AIE IRON API:
    //   XAie_DmaStart(&xaie, 0, 0, DMA_CH_IN, XAIE_DMA_DOWN_DIR,
    //                 args->ptr_input, len, 0, 0);
    (void)len;
}

//====//
// DMA setup for output vector transfer (tile -> host)
//====//
static void setup_dma_output(const RmsNormKernelArgs* args) {
    uint64_t len = static_cast<uint64_t>(args->N) * 4;
    // In actual AIE IRON API:
    //   XAie_DmaStart(&xaie, 0, 0, DMA_CH_OUT, XAIE_DMA_UP_DIR,
    //                 args->ptr_output, len, 0, 0);
    (void)len;
}

//====//
// Synchronization: signal compute tile to start
//====//
static void signal_compute_start() {
    // XAie_LockRelease(&xaie, 1, 0, LOCK_START);
}

//====//
// Launch RMS normalization
//====//
static void launch_rmsnorm(const RmsNormKernelArgs* args) {
    // Step 1: Start DMA to load input vector
    setup_dma_input(args);
    // XAie_DmaStart(..., DMA_CH_IN, ...)

    // Step 2: Wait for input DMA
    // XAie_DmaWait(&xaie, 0, 0, DMA_CH_IN, 1);

    // Step 3: Compute normalization factor on control tile
    // Load input from DMA buffer, compute sum of squares,
    // compute norm_inv = 1/sqrt(sum/N + eps), store to shared memory

    // Step 4: Signal compute tile to start
    signal_compute_start();

    // Step 5: Start DMA to store output vector
    setup_dma_output(args);
    // XAie_DmaStart(..., DMA_CH_OUT, ...)

    // Step 6: Wait for output DMA
    // XAie_DmaWait(&xaie, 0, 0, DMA_CH_OUT, 1);
}

//====//
// Control tile main entry point
//====//
static void rmsnorm_control_main(uintptr_t args_ptr) {
    const RmsNormKernelArgs* args =
        reinterpret_cast<const RmsNormKernelArgs*>(args_ptr);
    launch_rmsnorm(args);
}
