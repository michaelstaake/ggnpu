#include "tensor.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace ggnpu {

// Q4_0 quantization: 32 values per block, 16 bytes
// Block structure:
//   uint16_t d (scale)
//   uint8_t  signs[16] (upper bit is sign)
//   uint8_t  qs[8] (nibbles, each 4 bits is a value)

struct Q4_0Block {
    uint16_t d;
    uint8_t signs[16];
    uint8_t qs[8];
};

// Dequantize a single Q4_0 block to float
static void dequantize_q4_0_block(const uint8_t* block, float* out) {
    Q4_0Block* qb = reinterpret_cast<Q4_0Block*>(const_cast<uint8_t*>(block));
    int16_t d_signed = static_cast<int16_t>(qb->d);
    float d = d_signed;

    for (int i = 0; i < 16; i++) {
        uint8_t nibble = (i % 2 == 0) ? (qb->qs[i / 2] & 0x0F) : (qb->qs[i / 2] >> 4);
        int val = static_cast<int>(nibble) - 8;
        out[i] = val * d;
    }
}

// Quantize a block of floats to Q4_0 (simple uniform quantization)
static void quantize_q4_0_block(const float* input, uint8_t* output, int size = 32) {
    Q4_0Block* qb = reinterpret_cast<Q4_0Block*>(output);
    float max_val = 0.0f;

    for (int i = 0; i < size; i++) {
        float abs_val = std::abs(input[i]);
        if (abs_val > max_val) max_val = abs_val;
    }

    float scale = max_val / 7.0f;
    qb->d = static_cast<uint16_t>(static_cast<int16_t>(scale));

    for (int i = 0; i < size; i += 2) {
        int v0 = std::clamp(static_cast<int>(input[i] / scale + 8), 0, 15);
        int v1 = std::clamp(static_cast<int>(input[i + 1] / scale + 8), 0, 15);
        qb->qs[i / 2] = static_cast<uint8_t>(v0 | (v1 << 4));
        qb->signs[i] = static_cast<uint8_t>((v0 >= 8) ? 0x80 : 0);
        if (i + 1 < size) {
            qb->signs[i + 1] = static_cast<uint8_t>((v1 >= 8) ? 0x80 : 0);
        }
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
        float dequant[32];
        dequantize_q4_0_block(block, dequant);

        Q4_0Block* qb = reinterpret_cast<Q4_0Block*>(const_cast<uint8_t*>(block));
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
