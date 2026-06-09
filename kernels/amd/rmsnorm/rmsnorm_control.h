// GGNPU Control Code for RMS Normalization
// Runs on control tile (tile 0,0) - handles DMA, reduction, and scaling factor computation
//
// Architecture:
//   - Control tile receives input vector via DMA
//   - Computes RMS normalization factor: norm_inv = 1 / sqrt(mean(x^2) + eps)
//   - Signals compute tile to scale
//   - Stores output vector via DMA back to host
//
// Guardrail compliance:
//   1. Two-layer: reduction and factor computation here, scaling in tile code
//   2. No branches in compute path: fixed-size vector operations

#pragma once

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
// DMA setup for input vector transfer (host -> tile)
//====//
__attribute__((noinline))
void setup_dma_input(const RmsNormKernelArgs* args) {
    // DMA channel 0: input vector from host to tile (0,0)
    // Length: N * 4 bytes (float32)
    uint64_t len = static_cast<uint64_t>(args->N) * 4;

    // In actual AIE code:
    // aie::dma::start(channel_0, args->ptr_input, len)
    (void)len;
}

//====//
// DMA setup for output vector transfer (tile -> host)
//====//
__attribute__((noinline))
void setup_dma_output(const RmsNormKernelArgs* args) {
    // DMA channel 1: output vector from tile (0,0) to host
    // Length: N * 4 bytes (float32)
    uint64_t len = static_cast<uint64_t>(args->N) * 4;

    // In actual AIE code:
    // aie::dma::start(channel_1, tile_local_output_buffer, len)
    (void)len;
}

//====//
// Compute RMS normalization factor
// norm_inv = 1.0f / sqrt(sum(x[i]^2) / N + eps)
//
// Uses vector operations on control tile:
//   1. Load input vector (vectorized)
//   2. Square each element (vector multiply)
//   3. Reduce-add to get sum of squares
//   4. Divide by N
//   5. Add eps
//   6. Compute reciprocal sqrt
//====//
__attribute__((noinline))
float compute_rms_norm_factor(const float* input, int N, float eps) {
    // Sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_sq += input[i] * input[i];
    }

    // RMS normalization factor inverse
    float variance = sum_sq / static_cast<float>(N) + eps;
    float rms = std::sqrt(variance);
    return 1.0f / rms;
}

//====//
// Synchronization: signal compute tile to start scaling
//====//
__attribute__((noinline))
void signal_compute_start() {
    // Release lock 0 to signal compute tile (1,0)
    // In actual AIE code:
    // aie::lock::release(lock_0)
}

//====//
// Synchronization: wait for compute tile to complete
//====//
__attribute__((noinline))
void wait_compute_complete() {
    // Wait for compute tile to signal completion
    // In actual AIE code:
    // aie::lock::acquire(lock_completion)
}

//====//
// Launch RMS normalization
// Main control flow
//====//
__attribute__((noinline))
void launch_rmsnorm(const RmsNormKernelArgs* args) {
    // Step 1: Start DMA to load input vector
    setup_dma_input(args);
    // aie::dma::start(channel_0, ...)

    // Step 2: Wait for input DMA
    // aie::dma::wait(channel_0)

    // Step 3: Compute normalization factor on control tile
    // In actual AIE code, this uses vector operations:
    //   - Load input from DMA buffer into local SRAM
    //   - Vector square: v_mul(x, x)
    //   - Vector reduce-add: v_reduce_add()
    //   - Scalar: divide by N, add eps, sqrt, reciprocal
    //   - Store norm_inv to shared memory

    // Step 4: Signal compute tile to start
    signal_compute_start();

    // Step 5: Wait for compute tile to finish scaling
    wait_compute_complete();

    // Step 6: Start DMA to store output vector
    setup_dma_output(args);
    // aie::dma::start(channel_1, ...)

    // Step 7: Wait for output DMA
    // aie::dma::wait(channel_1)
}

//====//
// Control tile main entry point
// Called by Peano runtime on control tile (0,0)
//====//
__attribute__((noinline))
void rmsnorm_control_main(uintptr_t args_ptr) {
    const RmsNormKernelArgs* args = reinterpret_cast<const RmsNormKernelArgs*>(args_ptr);

    // Launch the RMS normalization kernel
    launch_rmsnorm(args);
}
