#ifndef GGNPU_Q8_0_H
#define GGNPU_Q8_0_H

#include <cstdint>
#include <vector>

namespace ggnpu {

// GGUF Q8_0 block: an fp16 scale `d` followed by 32 signed int8 quants.
// (`d` holds IEEE half-precision bits — NOT an integer.)
struct Q8_0Block {
    uint16_t d;      // fp16 scale bits
    int8_t qs[32];
};

// Dequantize one 34-byte Q8_0 block (32 values, fp16 scale) to float.
void dequant_q8_0_block(const uint8_t* block, float* out);

// Decode Q8_0 weights for the NPU int8 matmul: per-row symmetric int8 with one
// scale per row (matches the Q4_0/Q4_K/Q6_K convention the kernel expects). GGUF
// 2D layout: n_cols (K) = dims[0], n_rows (N) = dims[1].
void decode_q8_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         int64_t n_rows, int64_t n_cols,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

} // namespace ggnpu

#endif // GGNPU_Q8_0_H
