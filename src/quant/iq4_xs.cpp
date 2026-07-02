#include "tensor.h"
#include "quant/iq4_xs.h"
#include "quant/kquant.h"  // fp16_to_f32, QK_K
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace ggnpu {

namespace {
constexpr size_t IQ4_XS_BLOCK_BYTES = 136;  // d(2) scales_h(2) scales_l[4] qs[128]
// Same fixed non-linear 4-bit codebook as IQ4_NL (llama.cpp kvalues_iq4nl).
constexpr int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
}

// Dequantize one 136-byte IQ4_XS super-block (256 values), matching llama.cpp
// dequant_row_iq4_xs.
void dequant_iq4_xs_block(const uint8_t* block, float* out) {
    float d = fp16_to_f32(static_cast<uint16_t>(block[0] | (block[1] << 8)));
    uint16_t scales_h = static_cast<uint16_t>(block[2] | (block[3] << 8));
    const uint8_t* scales_l = block + 4;
    const uint8_t* qs = block + 8;

    float* y = out;
    for (int ib = 0; ib < 8; ++ib) {  // QK_K/32 sub-blocks
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) |
                       (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j]      = dl * kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
        qs += 16;
    }
}

void decode_iq4_xs_for_npu(const uint8_t* gguf_data, size_t data_size,
                           int64_t n_rows, int64_t n_cols,
                           std::vector<int8_t>& int8_output,
                           std::vector<float>& scales_output) {
    const size_t blocks_per_row =
        (static_cast<size_t>(n_cols) + QK_K - 1) / QK_K;
    const size_t row_bytes = blocks_per_row * IQ4_XS_BLOCK_BYTES;

    if (n_rows <= 0 || n_cols <= 0 ||
        static_cast<size_t>(n_rows) * row_bytes != data_size) {
        size_t num_blocks = data_size / IQ4_XS_BLOCK_BYTES;
        n_cols = static_cast<int64_t>(num_blocks * QK_K);
        n_rows = 1;
    }

    int8_output.assign(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols), 0);
    scales_output.assign(static_cast<size_t>(n_rows), 1.0f);

    std::vector<float> block_f32(QK_K);
    std::vector<float> row_f32(static_cast<size_t>(n_cols));
    for (int64_t r = 0; r < n_rows; r++) {
        const uint8_t* row_data = gguf_data + static_cast<size_t>(r) * row_bytes;
        float max_abs = 0.0f;
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_iq4_xs_block(row_data + b * IQ4_XS_BLOCK_BYTES, block_f32.data());
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
