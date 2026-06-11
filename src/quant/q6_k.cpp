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
// Output: 256 INT8 values per block (scales pre-applied, clamped to int8)
// The NPU kernel does raw INT8 dot product — scales must be baked in on host.
// Uses direct byte offsets to avoid reading beyond 64-byte GGUF block
void decode_q6_k_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output) {
    int num_blocks = data_size / Q6_K_BLOCK_SIZE;
    int8_output.resize(num_blocks * 256);
    scales_output.clear(); // not needed — scales are pre-applied

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

        // Q4 part (first 128): apply scale, clamp to int8
        // qs starts at offset 16, groups of 32 use scales[0..5]
        for (int g = 0; g < 4; g++) {
            float scale = base_scales[(g * 2) % 12] * d_signed;
            const uint8_t* qblock = block + 16 + g * 16;
            int out_off = i * 256 + g * 32;
            for (int j = 0; j < 32; j++) {
                uint8_t nibble = (j % 2 == 0) ? (qblock[j / 2] & 0x0F) : (qblock[j / 2] >> 4);
                int val = static_cast<int>(nibble) - 8;
                float scaled = static_cast<float>(val) * scale;
                int8_output[out_off + j] = static_cast<int8_t>(std::clamp(scaled, -128.0f, 127.0f));
            }
        }

        // Q6 part (last 128): apply scale, clamp to int8
        // high_bits at offset 32, qs_large at offset 48, groups of 32 use scales[6..11]
        for (int g = 0; g < 4; g++) {
            float scale = base_scales[6 + g] * d2_signed;
            const int8_t* qblock = reinterpret_cast<const int8_t*>(block + 48) + g * 32;
            const uint8_t* hblock = block + 32 + g * 32;
            int out_off = i * 256 + 128 + g * 32;
            for (int j = 0; j < 32; j++) {
                int val = static_cast<int>(qblock[j]) | (static_cast<int>(hblock[j] & 0x03) << 7);
                if (val >= 64) val -= 128;
                float scaled = static_cast<float>(val) * scale;
                int8_output[out_off + j] = static_cast<int8_t>(std::clamp(scaled, -128.0f, 127.0f));
            }
        }
    }
}

} // namespace ggnpu
