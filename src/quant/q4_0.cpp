#include "tensor.h"
#include "quant/q4_0.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace ggnpu {

// Dequantize a single Q4_0 block to float
static void dequantize_q4_0_block(const uint8_t* block, float* out) {
    const Q4_0Block* qb = reinterpret_cast<const Q4_0Block*>(block);
    int16_t d_signed = static_cast<int16_t>(qb->d);
    float d = d_signed;

    for (int i = 0; i < 16; i++) {
        uint8_t nibble = (i % 2 == 0) ? (qb->qs[i / 2] & 0x0F) : (qb->qs[i / 2] >> 4);
        int val = static_cast<int>(nibble) - 8;
        out[i] = val * d;
    }
}

// Decode Q4_0 weights for NPU (dequantize to INT8 + scales)
void decode_q4_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output) {
    int num_blocks = data_size / sizeof(Q4_0Block);
    int8_output.resize(num_blocks * 16);
    scales_output.resize(num_blocks);

    for (int i = 0; i < num_blocks; i++) {
        const uint8_t* block = gguf_data + i * sizeof(Q4_0Block);
        const Q4_0Block* qb = reinterpret_cast<const Q4_0Block*>(block);
        int16_t d_signed = static_cast<int16_t>(qb->d);
        scales_output[i] = d_signed;

        for (int j = 0; j < 16; j++) {
            uint8_t nibble = (j % 2 == 0) ? (qb->qs[j / 2] & 0x0F) : (qb->qs[j / 2] >> 4);
            int val = static_cast<int>(nibble) - 8;
            int8_output[i * 16 + j] = static_cast<int8_t>(val);
        }
    }
}

} // namespace ggnpu
