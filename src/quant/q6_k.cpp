#include "tensor.h"
#include "quant/quant.h"
#include "quant/kquant.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace ggnpu {

// Decode Q6_K weights for NPU.
// Real ggml Q6_K layout: 210-byte super-blocks of 256 values (see kquant.h).
//
// Weights are dequantized to float, then requantized symmetrically with one
// per-tensor scale (w ≈ int8 * scale). The backend applies the scale.
void decode_q6_k_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output) {
    size_t num_blocks = data_size / Q6_K_BLOCK_BYTES;
    size_t n = num_blocks * QK_K;
    std::vector<float> f32(n);

    float max_abs = 0.0f;
    for (size_t i = 0; i < num_blocks; i++) {
        dequant_q6_k_block(gguf_data + i * Q6_K_BLOCK_BYTES, f32.data() + i * QK_K);
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

} // namespace ggnpu
