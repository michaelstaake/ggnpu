#ifndef GGNPU_BF16_H
#define GGNPU_BF16_H

#include <cstdint>
#include <vector>

namespace ggnpu {

// Convert a single f32 value to bf16 (round to nearest).
uint16_t f32_to_bf16(float f);

// Convert a single bf16 value to f32.
float bf16_to_f32(uint16_t b);

// Convert f32 vector to bf16 vector (writes to separate buffer).
std::vector<uint8_t> convert_f32_to_bf16(const void* f32_data, size_t count);

// Convert bf16 vector to f32 vector.
std::vector<float> convert_bf16_to_f32(const void* bf16_data, size_t count);

// Full f32 -> bf16 -> f32 roundtrip (for test comparisons matching NPU DMA path).
float bf16_roundtrip_f32(float f);

// Vectorized f32 -> bf16 -> f32 roundtrip.
void bf16_roundtrip_vector(const float* in, float* out, int n);

} // namespace ggnpu

#endif // GGNPU_BF16_H
