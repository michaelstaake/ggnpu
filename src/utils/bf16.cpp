#include "ggnpu/bf16.h"
#include <cstring>

namespace ggnpu {

uint16_t f32_to_bf16(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t frac = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) {
            bits = sign;  // signed zero
        } else {
            // subnormal f16 -> normalized f32
            uint32_t f = frac;
            int e = -1;
            do { f <<= 1; e++; } while ((f & 0x400) == 0);
            f &= 0x3FF;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (f << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (frac << 13);  // inf / nan
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<uint8_t> convert_f32_to_bf16(const void* f32_data, size_t count) {
    const float* f32 = static_cast<const float*>(f32_data);
    std::vector<uint8_t> bf16_buf(count * 2);  // 2 bytes per bf16
    for (size_t i = 0; i < count; i++) {
        uint16_t* bf16_ptr = reinterpret_cast<uint16_t*>(bf16_buf.data() + i * 2);
        *bf16_ptr = f32_to_bf16(f32[i]);
    }
    return bf16_buf;
}

std::vector<float> convert_bf16_to_f32(const void* bf16_data, size_t count) {
    const uint16_t* bf16 = static_cast<const uint16_t*>(bf16_data);
    std::vector<float> f32_buf(count);
    for (size_t i = 0; i < count; i++) {
        f32_buf[i] = bf16_to_f32(bf16[i]);
    }
    return f32_buf;
}

float bf16_roundtrip_f32(float f) {
    uint16_t b = f32_to_bf16(f);
    return bf16_to_f32(b);
}

void bf16_roundtrip_vector(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = bf16_roundtrip_f32(in[i]);
    }
}

} // namespace ggnpu
