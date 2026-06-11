#ifndef GGNPU_KQUANT_H
#define GGNPU_KQUANT_H

#include <cstdint>
#include <cstring>

// Reference dequantization for ggml K-quants (matches llama.cpp layouts).
//
// Q4_K super-block: 144 bytes, 256 values
//   ggml_fp16_t d;        // super-block scale for scales      (offset 0)
//   ggml_fp16_t dmin;     // super-block scale for mins        (offset 2)
//   uint8_t scales[12];   // 8 x (6-bit scale, 6-bit min)      (offset 4)
//   uint8_t qs[128];      // 4-bit quants                      (offset 16)
//
// Q6_K super-block: 210 bytes, 256 values
//   uint8_t ql[128];      // low 4 bits                        (offset 0)
//   uint8_t qh[64];       // high 2 bits                       (offset 128)
//   int8_t  scales[16];   // 8-bit scales per 16 values        (offset 192)
//   ggml_fp16_t d;        // super-block scale                 (offset 208)

namespace ggnpu {

constexpr size_t QK_K = 256;
constexpr size_t Q4_K_BLOCK_BYTES = 144;
constexpr size_t Q6_K_BLOCK_BYTES = 210;

inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal: normalize
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// 6-bit scale/min extraction for Q4_K (same as llama.cpp get_scale_min_k4)
inline void q4k_get_scale_min(int j, const uint8_t* q, uint8_t* sc, uint8_t* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantize one 144-byte Q4_K block to 256 floats
inline void dequant_q4_k_block(const uint8_t* block, float* out) {
    float d = fp16_to_f32(static_cast<uint16_t>(block[0] | (block[1] << 8)));
    float dmin = fp16_to_f32(static_cast<uint16_t>(block[2] | (block[3] << 8)));
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    int idx = 0;
    for (int j = 0; j < 8; j += 2) {
        uint8_t sc1, m1, sc2, m2;
        q4k_get_scale_min(j, scales, &sc1, &m1);
        q4k_get_scale_min(j + 1, scales, &sc2, &m2);
        float d1 = d * sc1, min1 = dmin * m1;
        float d2 = d * sc2, min2 = dmin * m2;
        for (int l = 0; l < 32; l++) out[idx + l] = d1 * (qs[l] & 0xF) - min1;
        for (int l = 0; l < 32; l++) out[idx + 32 + l] = d2 * (qs[l] >> 4) - min2;
        idx += 64;
        qs += 32;
    }
}

// Dequantize one 210-byte Q6_K block to 256 floats
inline void dequant_q6_k_block(const uint8_t* block, float* out) {
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(block + 192);
    float d = fp16_to_f32(static_cast<uint16_t>(block[208] | (block[209] << 8)));

    for (int n = 0; n < 2; n++) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = static_cast<int>((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
            int q2 = static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l] = d * sc[is] * q1;
            out[l + 32] = d * sc[is + 2] * q2;
            out[l + 64] = d * sc[is + 4] * q3;
            out[l + 96] = d * sc[is + 6] * q4;
        }
        out += 128; ql += 64; qh += 32; sc += 8;
    }
}

} // namespace ggnpu

#endif // GGNPU_KQUANT_H
