#include "tensor.h"
#include "quant/q8_0.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace ggnpu {

// Dequantize a single Q8_0 block to float
static void dequantize_q8_0_block(const uint8_t* block, float* out) {
    const Q8_0Block* qb = reinterpret_cast<const Q8_0Block*>(block);
    float d = static_cast<float>(qb->d);

    for (int i = 0; i < 32; i++) {
        out[i] = static_cast<float>(qb->qs[i]) * d;
    }
}

// Quantize a block of floats to Q8_0
static void quantize_q8_0_block(const float* input, uint8_t* output, int size) {
    Q8_0Block* qb = reinterpret_cast<Q8_0Block*>(output);
    float max_val = 0.0f;

    for (int i = 0; i < size; i++) {
        float abs_val = std::abs(input[i]);
        if (abs_val > max_val) max_val = abs_val;
    }

    float scale = max_val / 127.0f;
    qb->d = static_cast<int16_t>(scale);

    for (int i = 0; i < size; i++) {
        int val = std::clamp(static_cast<int>(input[i] / scale), -128, 127);
        qb->qs[i] = static_cast<int8_t>(val);
    }
}

// Decode Q8_0 weights for NPU (already INT8, just extract scales)
void decode_q8_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output) {
    int num_blocks = data_size / sizeof(Q8_0Block);
    int8_output.resize(num_blocks * 32);
    scales_output.resize(num_blocks);

    for (int i = 0; i < num_blocks; i++) {
        const uint8_t* block = gguf_data + i * sizeof(Q8_0Block);
        const Q8_0Block* qb = reinterpret_cast<const Q8_0Block*>(block);

        scales_output[i] = static_cast<float>(qb->d);

        std::memcpy(int8_output.data() + i * 32, qb->qs, 32);
    }
}

} // namespace ggnpu
