// GGNPU AIE2P Tile Code for Softmax
// Computes: out[r][c] = exp(in[r][c] - max_r) / sum_r

#pragma once

#include <cstdint>
#include <cmath>

static constexpr int VEC_LEN = 16;  // Vector width in float32 elements

//====//
// Softmax tile compute
// Two-pass: first compute exp(x - max), then divide by sum
//====//
static void softmax_tile(const float* ptr_input, float* ptr_output,
                          const float* ptr_max, const float* ptr_sum,
                          int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float max_r = ptr_max[r];
        float sum_r = ptr_sum[r];

        // First pass: compute exp(x - max_r)
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            ptr_output[idx] = std::exp(ptr_input[idx] - max_r);
        }

        // Second pass: divide by sum
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            ptr_output[idx] /= sum_r;
        }
    }
}

//====//
// Control tile: compute per-row max and sum
//====//
static void compute_row_stats(const float* input, float* out_max, float* out_sum,
                               int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float max_r = -1e30f;
        for (int c = 0; c < cols; c++) {
            float val = input[r * cols + c];
            if (val > max_r) max_r = val;
        }
        out_max[r] = max_r;

        float sum_r = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum_r += std::exp(input[r * cols + c] - max_r);
        }
        out_sum[r] = sum_r;
    }
}

//====//
// Kernel argument structure
//====//
struct SoftmaxKernelArgs {
    uint64_t ptr_input;   // Input matrix (float32, rows x cols)
    uint64_t ptr_output;  // Output matrix (float32, rows x cols)
    uint32_t rows;        // Number of rows
    uint32_t cols;        // Number of columns per row
    uint32_t padding[2];  // Alignment padding
};

//====//
// DMA channel IDs and lock IDs
//====//
static constexpr int DMA_CH_IN = 0;
static constexpr int DMA_CH_OUT = 1;
static constexpr int LOCK_START = 2;

//====//
// DMA setup
//====//
static void setup_dma_input(const SoftmaxKernelArgs* args) {
    uint64_t len = static_cast<uint64_t>(args->rows) * static_cast<uint64_t>(args->cols) * 4;
    (void)len;
}

static void setup_dma_output(const SoftmaxKernelArgs* args) {
    uint64_t len = static_cast<uint64_t>(args->rows) * static_cast<uint64_t>(args->cols) * 4;
    (void)len;
}

static void signal_compute_start() {
    // XAie_LockRelease(&xaie, 1, 0, LOCK_START);
}

//====//
// Launch softmax
//====//
static void launch_softmax(const SoftmaxKernelArgs* args) {
    setup_dma_input(args);
    // XAie_DmaStart(..., DMA_CH_IN, ...)
    // XAie_DmaWait(&xaie, 0, 0, DMA_CH_IN, 1);
    signal_compute_start();
    setup_dma_output(args);
    // XAie_DmaStart(..., DMA_CH_OUT, ...)
    // XAie_DmaWait(&xaie, 0, 0, DMA_CH_OUT, 1);
}

static void softmax_control_main(uintptr_t args_ptr) {
    const SoftmaxKernelArgs* args =
        reinterpret_cast<const SoftmaxKernelArgs*>(args_ptr);
    launch_softmax(args);
}
