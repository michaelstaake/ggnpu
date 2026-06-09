// GGNPU AIE2P Tile Code for FlashAttention (decomposed v1)
// Computes: attn = softmax(Q @ K^T / sqrt(d)) @ V

#pragma once

#include <cstdint>
#include <cmath>

static constexpr int VEC_LEN = 16;

//====//
// FlashAttention tile compute (decomposed v1)
// Three stages: QK^T matmul, softmax, weighted V sum
//====//
static void flash_attn_tile(const float* Q, const float* K, const float* V,
                             float* output,
                             int n_head, int head_dim, int ctx_len) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Working buffers for attention scores and attention weights
    // In actual AIE, these are in tile L2 memory
    static float scores[128];  // Max n_head = 128
    static float attn_w[128];  // Max n_head = 128

    for (int h = 0; h < n_head; h++) {
        // Step 1: QK^T - compute attention scores for head h
        for (int j = 0; j < ctx_len; j++) {
            float sum = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum += Q[h * head_dim + d] * K[j * head_dim + d];
            }
            scores[j] = sum * scale;
        }

        // Step 2: Softmax
        float max_s = -1e30f;
        for (int j = 0; j < ctx_len; j++) {
            if (scores[j] > max_s) max_s = scores[j];
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < ctx_len; j++) {
            scores[j] = std::exp(scores[j] - max_s);
            sum_exp += scores[j];
        }

        for (int j = 0; j < ctx_len; j++) {
            attn_w[j] = scores[j] / sum_exp;
        }

        // Step 3: Weighted V sum
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < ctx_len; j++) {
                sum += attn_w[j] * V[j * head_dim + d];
            }
            output[h * head_dim + d] = sum;
        }
    }
}

//====//
// Kernel argument structure
//====//
struct FlashAttnKernelArgs {
    uint64_t ptr_q;       // Q matrix (float32, n_head x head_dim)
    uint64_t ptr_k;       // K matrix (float32, ctx_len x head_dim)
    uint64_t ptr_v;       // V matrix (float32, ctx_len x head_dim)
    uint64_t ptr_output;  // Output (float32, n_head x head_dim)
    uint32_t n_head;      // Number of attention heads
    uint32_t head_dim;    // Dimension per head
    uint32_t ctx_len;     // Context length
    uint32_t padding;     // Alignment padding
};

//====//
// DMA channel IDs and lock IDs
//====//
static constexpr int DMA_CH_Q = 0;
static constexpr int DMA_CH_K = 1;
static constexpr int DMA_CH_V = 2;
static constexpr int DMA_CH_OUT = 3;
static constexpr int LOCK_START = 5;

//====//
// DMA setup
//====//
static void setup_dma_q(const FlashAttnKernelArgs* args) {
    uint64_t len = static_cast<uint64_t>(args->n_head) * static_cast<uint64_t>(args->head_dim) * 4;
    (void)len;
}

static void setup_dma_k(const FlashAttnKernelArgs* args) {
    uint64_t len = static_cast<uint64_t>(args->ctx_len) * static_cast<uint64_t>(args->head_dim) * 4;
    (void)len;
}

static void setup_dma_v(const FlashAttnKernelArgs* args) {
    uint64_t len = static_cast<uint64_t>(args->ctx_len) * static_cast<uint64_t>(args->head_dim) * 4;
    (void)len;
}

static void setup_dma_output(const FlashAttnKernelArgs* args) {
    uint64_t len = static_cast<uint64_t>(args->n_head) * static_cast<uint64_t>(args->head_dim) * 4;
    (void)len;
}

static void signal_compute_start() {
    // XAie_LockRelease(&xaie, 1, 0, LOCK_START);
}

//====//
// Launch FlashAttention
//====//
static void launch_flash_attn(const FlashAttnKernelArgs* args) {
    setup_dma_q(args);
    setup_dma_k(args);
    setup_dma_v(args);
    // XAie_DmaStart(..., DMA_CH_Q, ...)
    // XAie_DmaStart(..., DMA_CH_K, ...)
    // XAie_DmaStart(..., DMA_CH_V, ...)
    // XAie_DmaWait(&xaie, 0, 0, DMA_CH_Q, 1);
    // XAie_DmaWait(&xaie, 0, 0, DMA_CH_K, 1);
    // XAie_DmaWait(&xaie, 0, 0, DMA_CH_V, 1);
    signal_compute_start();
    setup_dma_output(args);
    // XAie_DmaStart(..., DMA_CH_OUT, ...)
    // XAie_DmaWait(&xaie, 0, 0, DMA_CH_OUT, 1);
}

static void flash_attn_control_main(uintptr_t args_ptr) {
    const FlashAttnKernelArgs* args =
        reinterpret_cast<const FlashAttnKernelArgs*>(args_ptr);
    launch_flash_attn(args);
}
