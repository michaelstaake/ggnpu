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
constexpr size_t Q2_K_BLOCK_BYTES = 84;
constexpr size_t Q3_K_BLOCK_BYTES = 110;
constexpr size_t Q4_K_BLOCK_BYTES = 144;
constexpr size_t Q5_K_BLOCK_BYTES = 176;
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

// Dequantize one 84-byte Q2_K block to 256 floats (matches llama.cpp
// dequant_row_q2_K). Layout: scales[16] (4-bit scale, 4-bit min each) at
// offset 0, qs[64] (2-bit quants) at offset 16, d(fp16) at offset 80,
// dmin(fp16) at offset 82.
inline void dequant_q2_k_block(const uint8_t* block, float* out) {
    const uint8_t* scales = block;
    const uint8_t* qs = block + 16;
    float d = fp16_to_f32(static_cast<uint16_t>(block[80] | (block[81] << 8)));
    float dmin = fp16_to_f32(static_cast<uint16_t>(block[82] | (block[83] << 8)));

    int is = 0;
    float* y = out;
    for (int n = 0; n < 256; n += 128) {
        const uint8_t* q = qs + (n / 128) * 32;
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = scales[is++];
            float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l] >> shift) & 3) - ml;
            sc = scales[is++];
            dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;
            shift += 2;
        }
    }
}

// Dequantize one 110-byte Q3_K block to 256 floats (matches llama.cpp
// dequant_row_q3_K). Layout: hmask[32] at offset 0, qs[64] at offset 32,
// scales[12] (6-bit, packed) at offset 96, d(fp16) at offset 108.
inline void dequant_q3_k_block(const uint8_t* block, float* out) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;

    const uint8_t* hm = block;          // hmask[32]
    const uint8_t* qs = block + 32;     // qs[64]
    float d_all = fp16_to_f32(static_cast<uint16_t>(block[108] | (block[109] << 8)));

    uint32_t aux[4];
    std::memcpy(aux, block + 96, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    int is = 0;
    uint8_t m = 1;
    float* y = out;
    for (int n = 0; n < 256; n += 128) {
        const uint8_t* q = qs + (n / 128) * 32;
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (static_cast<int8_t>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * (static_cast<int8_t>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
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

// Dequantize one 176-byte Q5_K block to 256 floats.
// Layout: d(fp16) dmin(fp16) scales[12] qh[32] qs[128]. Same 6-bit scale/min
// packing as Q4_K, plus a 5th bit per value taken from qh (matches llama.cpp
// dequant_row_q5_K: value = (qs_nibble + (qh_bit ? 16 : 0)) scaled).
inline void dequant_q5_k_block(const uint8_t* block, float* out) {
    float d = fp16_to_f32(static_cast<uint16_t>(block[0] | (block[1] << 8)));
    float dmin = fp16_to_f32(static_cast<uint16_t>(block[2] | (block[3] << 8)));
    const uint8_t* scales = block + 4;
    const uint8_t* qh = block + 16;
    const uint8_t* ql = block + 48;

    int is = 0, idx = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        q4k_get_scale_min(is + 0, scales, &sc, &m);
        float d1 = d * sc, m1 = dmin * m;
        q4k_get_scale_min(is + 1, scales, &sc, &m);
        float d2 = d * sc, m2 = dmin * m;
        for (int l = 0; l < 32; l++)
            out[idx + l] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; l++)
            out[idx + 32 + l] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32; idx += 64; is += 2;
        u1 <<= 2; u2 <<= 2;
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
