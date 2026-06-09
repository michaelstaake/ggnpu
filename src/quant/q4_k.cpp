#include "tensor.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace ggnpu {

// Q4_K quantization: 256 values per block, 48 bytes
// Mixed quantization: combines Q4_0 + Q8_0 with extra scales
// Block structure (48 bytes):
//   uint16_t d  (min scale)
//   uint16_t c  (min scale multiplier)
//   int8_t   scales[6] (6 bytes for 4 quantized scales)
//   uint8_t  qs[32] (Q4_0 for first 128 values)
//   int8_t   qs[96] (Q8_0 for last 128 values)

struct Q4_KBlock {
    uint16_t d;
    uint16_t c;
    int8_t scales[6];
    uint8_t qs[32];
    int8_t qs_large[96];
};

// Dequantize a single Q4_K block to float
static void dequantize_q4_k_block(const uint8_t* block, float* out) {
    Q4_KBlock* qb = reinterpret_cast<Q4_KBlock*>(const_cast<uint8_t*>(block));
    int16_t d_signed = static_cast<int16_t>(qb->d);
    int16_t c_signed = static_cast<int16_t>(qb->c);

    // Decode 6-byte scale array into 8 scale values
    float scales[8];
    scales[0] = d_signed;
    scales[1] = c_signed;

    for (int i = 0; i < 6; i++) {
        int8_t s = qb->scales[i];
        int8_t s_low = s & 0x0F;
        int8_t s_high = (s >> 4) & 0x0F;
        scales[i * 2 + 2] = static_cast<float>(s_low);
        scales[i * 2 + 3] = static_cast<float>(s_high);
    }

    // First 128 values: Q4_0 with scales[0..3]
    for (int i = 0; i < 128; i += 32) {
        int scale_idx = (i / 32) % 4;
        float scale = scales[scale_idx] * d_signed;
        const uint8_t* qblock = qb->qs + (i / 32) * 16;

        for (int j = 0; j < 16; j++) {
            uint8_t nibble = (j % 2 == 0) ? (qblock[j / 2] & 0x0F) : (qblock[j / 2] >> 4);
            int val = static_cast<int>(nibble) - 8;
            out[i + j] = val * scale;
        }
        for (int j = 16; j < 32; j++) {
            uint8_t nibble = (j % 2 == 0) ? (qblock[j / 2] & 0x0F) : (qblock[j / 2] >> 4);
            int val = static_cast<int>(nibble) - 8;
            out[i + j] = val * scale;
        }
    }

    // Last 128 values: Q8_0 with scales[4..7]
    for (int i = 128; i < 256; i += 32) {
        int scale_idx = ((i - 128) / 32) + 4;
        float scale = scales[scale_idx] * d_signed;
        const int8_t* qblock = qb->qs_large + (i - 128);

        for (int j = 0; j < 32; j++) {
            out[i + j] = static_cast<float>(qblock[j]) * scale;
        }
    }
}

// Decode Q4_K weights for NPU
void decode_q4_k_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output) {
    int num_blocks = data_size / sizeof(Q4_KBlock);
    int8_output.resize(num_blocks * 256);
    scales_output.resize(num_blocks * 8);

    for (int i = 0; i < num_blocks; i++) {
        const uint8_t* block = gguf_data + i * sizeof(Q4_KBlock);
        Q4_KBlock* qb = reinterpret_cast<Q4_KBlock*>(const_cast<uint8_t*>(block));
        int16_t d_signed = static_cast<int16_t>(qb->d);

        // Extract scales
        scales_output[i * 8 + 0] = d_signed;
        scales_output[i * 8 + 1] = static_cast<float>(static_cast<int16_t>(qb->c));

        for (int j = 0; j < 6; j++) {
            int8_t s = qb->scales[j];
            scales_output[i * 8 + j * 2 + 2] = static_cast<float>(s & 0x0F);
            scales_output[i * 8 + j * 2 + 3] = static_cast<float>((s >> 4) & 0x0F);
        }

        // Extract INT8 values (Q4_0 part)
        for (int j = 0; j < 32; j++) {
            uint8_t nibble = (j % 2 == 0) ? (qb->qs[j / 2] & 0x0F) : (qb->qs[j / 2] >> 4);
            int8_output[i * 256 + j] = static_cast<int8_t>(nibble - 8);
        }

        // Q8_0 part
        std::memcpy(int8_output.data() + i * 256 + 128, qb->qs_large, 96);
    }
}

} // namespace ggnpu
