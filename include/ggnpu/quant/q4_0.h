#ifndef GGNPU_Q4_0_H
#define GGNPU_Q4_0_H

#include <cstdint>
#include <vector>

namespace ggnpu {

struct Q4_0Block {
    int16_t d;
    uint8_t qs[16];  // 16 bytes for 32 Q4_0 values (2 values per byte)
};

// Dequantize one 18-byte Q4_0 block (32 values, fp16 scale) to float.
void dequant_q4_0_block(const uint8_t* block, float* out);

// Per-row symmetric int8 decode for the NPU matmul (one scale per row).
void decode_q4_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         int64_t n_rows, int64_t n_cols,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

} // namespace ggnpu

#endif // GGNPU_Q4_0_H
