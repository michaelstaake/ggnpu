#include "tensor.h"
#include "quant/q8_0.h"
#include "quant/kquant.h"  // fp16_to_f32
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace ggnpu {

namespace {
constexpr int QK8_0 = 32;                 // values per Q8_0 block
constexpr size_t Q8_0_BLOCK_BYTES = 34;   // fp16 scale (2) + 32 int8
}

// Dequantize a single Q8_0 block (32 values) to float. Scale is fp16 in the
// first two bytes (little-endian), followed by 32 signed int8 quants.
void dequant_q8_0_block(const uint8_t* block, float* out) {
    float d = fp16_to_f32(static_cast<uint16_t>(block[0] | (block[1] << 8)));
    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
    for (int j = 0; j < QK8_0; j++) {
        out[j] = static_cast<float>(qs[j]) * d;
    }
}

// Decode Q8_0 weights for the NPU int8 matmul: per-row symmetric int8 with one
// scale per row (matches the Q4_0/Q4_K/Q6_K convention the kernel expects). GGUF
// 2D layout: n_cols (K) = dims[0], n_rows (N) = dims[1].
void decode_q8_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         int64_t n_rows, int64_t n_cols,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output) {
    const size_t blocks_per_row =
        (static_cast<size_t>(n_cols) + QK8_0 - 1) / QK8_0;
    const size_t row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;

    // Fall back to a flat single-row decode if the shape doesn't match the data.
    if (n_rows <= 0 || n_cols <= 0 ||
        static_cast<size_t>(n_rows) * row_bytes != data_size) {
        size_t num_blocks = data_size / Q8_0_BLOCK_BYTES;
        n_cols = static_cast<int64_t>(num_blocks * QK8_0);
        n_rows = 1;
    }

    int8_output.assign(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols), 0);
    scales_output.assign(static_cast<size_t>(n_rows), 1.0f);

    std::vector<float> block_f32(QK8_0);
    std::vector<float> row_f32(static_cast<size_t>(n_cols));
    for (int64_t r = 0; r < n_rows; r++) {
        const uint8_t* row_data = gguf_data + static_cast<size_t>(r) * row_bytes;
        float max_abs = 0.0f;
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q8_0_block(row_data + b * Q8_0_BLOCK_BYTES, block_f32.data());
            for (int j = 0; j < QK8_0; j++) {
                size_t k = b * QK8_0 + j;
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
