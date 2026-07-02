#include "tensor.h"
#include "quant/quant.h"
#include "quant/kquant.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace ggnpu {

namespace {

void decode_q2_k_flat(const uint8_t* gguf_data, size_t data_size,
                      std::vector<int8_t>& int8_output,
                      std::vector<float>& scales_output) {
    size_t num_blocks = data_size / Q2_K_BLOCK_BYTES;
    size_t n = num_blocks * QK_K;
    std::vector<float> f32(n);

    float max_abs = 0.0f;
    for (size_t i = 0; i < num_blocks; i++) {
        dequant_q2_k_block(gguf_data + i * Q2_K_BLOCK_BYTES, f32.data() + i * QK_K);
        for (size_t j = i * QK_K; j < (i + 1) * QK_K; j++)
            max_abs = std::max(max_abs, std::fabs(f32[j]));
    }

    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    float inv = 1.0f / scale;

    int8_output.resize(n);
    for (size_t j = 0; j < n; j++) {
        float q = std::nearbyint(f32[j] * inv);
        int8_output[j] = static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
    }

    scales_output.assign(1, scale);
}

} // namespace

// Decode Q2_K weights to per-row INT8 + one scale per output row, matching the
// kq_path convention shared by the other k-quants.
void decode_q2_k_for_npu(const uint8_t* gguf_data, size_t data_size,
                         int64_t n_rows, int64_t n_cols,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output) {
    if (n_rows <= 0 || n_cols <= 0) {
        decode_q2_k_flat(gguf_data, data_size, int8_output, scales_output);
        return;
    }

    const size_t blocks_per_row =
        (static_cast<size_t>(n_cols) + QK_K - 1) / QK_K;
    const size_t row_bytes = blocks_per_row * Q2_K_BLOCK_BYTES;
    const size_t expected = static_cast<size_t>(n_rows) * row_bytes;
    if (expected != data_size) {
        decode_q2_k_flat(gguf_data, data_size, int8_output, scales_output);
        return;
    }

    int8_output.assign(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols), 0);
    scales_output.resize(static_cast<size_t>(n_rows));

    std::vector<float> block_f32(QK_K);
    for (int64_t r = 0; r < n_rows; r++) {
        const uint8_t* row_data = gguf_data + static_cast<size_t>(r) * row_bytes;
        float max_abs = 0.0f;
        std::vector<float> row_f32(static_cast<size_t>(n_cols), 0.0f);

        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q2_k_block(row_data + b * Q2_K_BLOCK_BYTES, block_f32.data());
            for (size_t j = 0; j < QK_K; j++) {
                size_t k = b * QK_K + j;
                if (k < static_cast<size_t>(n_cols)) {
                    row_f32[k] = block_f32[j];
                    max_abs = std::max(max_abs, std::fabs(block_f32[j]));
                }
            }
        }

        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        float inv = 1.0f / scale;
        scales_output[static_cast<size_t>(r)] = scale;

        for (int64_t k = 0; k < n_cols; k++) {
            float q = std::nearbyint(row_f32[static_cast<size_t>(k)] * inv);
            int8_output[static_cast<size_t>(r) * static_cast<size_t>(n_cols) + static_cast<size_t>(k)] =
                static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
        }
    }
}

} // namespace ggnpu
