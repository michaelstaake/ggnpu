#include "tensor.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace ggnpu {

// Q6_K quantization: 256 values per block, 64 bytes
// Mixed quantization: combines Q4 + Q6 + extra scales
// Block structure (64 bytes):
//   uint16_t d    (main scale, multiplied by 4)
//   uint16_t d2   (min scale, multiplied by 4)
//   int8_t   scales[12] (12 bytes for 12 quantized scales, each byte = 2 x 4-bit values)
//   uint8_t  qs[32] (Q4 for first 128 values, 16 bytes for 64 Q4 values)
//   uint8_t  high_bits[32] (high 2 bits for last 128 values)
//   int8_t   qs_large[96] (Q6 for last 128 values, 96 bytes for 128 Q6 values)

struct Q6_KBlock {
    uint16_t d;
    uint16_t d2;
    int8_t scales[12];
    uint8_t qs[32];
    uint8_t high_bits[32];
    int8_t qs_large[96];
};

// GGUF block sizes (not sizeof(struct) due to padding)
constexpr size_t Q6_K_BLOCK_SIZE = 64;

// Dequantize a single Q6_K block to float
static void dequantize_q6_k_block(const uint8_t* block, float* out) {
    Q6_KBlock* qb = reinterpret_cast<Q6_KBlock*>(const_cast<uint8_t*>(block));
    int16_t d_signed = static_cast<int16_t>(qb->d);
    int16_t d2_signed = static_cast<int16_t>(qb->d2);

    // Decode 12-byte scale array into 12 scale values
    float scales[12];
    for (int i = 0; i < 6; i++) {
        int8_t s = qb->scales[i];
        int8_t s_low = s & 0x0F;
        int8_t s_high = (s >> 4) & 0x0F;
        scales[i * 2] = s_low;
        scales[i * 2 + 1] = s_high;
    }

    // First 128 values: Q4 with scales[0..5] * d
    for (int i = 0; i < 128; i += 32) {
        int scale_idx = (i / 32) % 6;
        float scale = scales[scale_idx] * d_signed;
        const uint8_t* qblock = qb->qs + (i / 32) * 16;

        for (int j = 0; j < 32; j++) {
            uint8_t nibble = (j % 2 == 0) ? (qblock[j / 2] & 0x0F) : (qblock[j / 2] >> 4);
            int val = static_cast<int>(nibble) - 8;
            out[i + j] = val * scale;
        }
    }

    // Last 128 values: Q6 with scales[6..11] * d2
    for (int i = 128; i < 256; i += 32) {
        int block_idx = (i - 128) / 32;
        float scale = scales[block_idx + 6] * d2_signed;
        const int8_t* qblock = qb->qs_large + (i - 128);
        const uint8_t* hblock = qb->high_bits + (i - 128);

        for (int j = 0; j < 32; j++) {
            int val = qblock[j] | ((hblock[j] & 0x03) << 7);
            if (val >= 64) val -= 128;
            out[i + j] = static_cast<float>(val) * scale;
        }
    }
}

// Decode Q6_K weights for NPU
// Output: 256 INT8 values per block + 12 pre-computed scales per block
// Scales[0-5] = decoded_scale * d (for first 128 values, Q4 part)
// Scales[6-11] = decoded_scale * d2 (for last 128 values, Q6 part)
// Uses direct byte offsets to avoid reading beyond 64-byte GGUF block
void decode_q6_k_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output) {
    int num_blocks = data_size / Q6_K_BLOCK_SIZE;
    int8_output.resize(num_blocks * 256);
    scales_output.resize(num_blocks * 12);

    for (int i = 0; i < num_blocks; i++) {
        const uint8_t* block = gguf_data + i * Q6_K_BLOCK_SIZE;

        // Read d and d2 using direct byte access (no struct pointer)
        int16_t d_signed = static_cast<int16_t>(block[0] | (block[1] << 8));
        int16_t d2_signed = static_cast<int16_t>(block[2] | (block[3] << 8));

        // Decode 12-byte scale array into 12 scale values
        float base_scales[12];
        for (int j = 0; j < 6; j++) {
            int8_t s = static_cast<int8_t>(block[4 + j]);
            base_scales[j * 2] = static_cast<float>(s & 0x0F);
            base_scales[j * 2 + 1] = static_cast<float>((s >> 4) & 0x0F);
        }

        // Pre-compute scales: first 6 * d, last 6 * d2
        for (int j = 0; j < 6; j++) {
            scales_output[i * 12 + j] = base_scales[j] * d_signed;
        }
        for (int j = 0; j < 6; j++) {
            scales_output[i * 12 + 6 + j] = base_scales[6 + j] * d2_signed;
        }

        // Q4 part (first 128) - qs starts at offset 16
        for (int j = 0; j < 32; j++) {
            uint8_t nibble = (j % 2 == 0) ? (block[16 + j / 2] & 0x0F) : (block[16 + j / 2] >> 4);
            int8_output[i * 256 + j] = static_cast<int8_t>(nibble - 8);
        }

        // Q6 part (last 128) - high_bits at offset 32, qs_large at offset 48
        for (int j = 0; j < 96; j++) {
            int8_t q_val = static_cast<int8_t>(block[48 + j]);
            uint8_t h_val = block[32 + j];
            int val = q_val | ((h_val & 0x03) << 7);
            if (val >= 64) val -= 128;
            int8_output[i * 256 + 128 + j] = static_cast<int8_t>(val);
        }
    }
}

} // namespace ggnpu
