#ifndef GGNPU_Q4_1_H
#define GGNPU_Q4_1_H

#include <cstdint>
#include <vector>

namespace ggnpu {

struct Q4_1Block {
    int16_t d;       // fp16 scale (delta)
    int16_t m;       // fp16 min
    uint8_t qs[16];  // 16 bytes for 32 Q4_1 values (2 values per byte)
};

// Dequantize one 20-byte Q4_1 block (32 values, fp16 scale + fp16 min) to float.
void dequant_q4_1_block(const uint8_t* block, float* out);

// Per-row symmetric int8 decode for the NPU matmul (one scale per row).
void decode_q4_1_for_npu(const uint8_t* gguf_data, size_t data_size,
                         int64_t n_rows, int64_t n_cols,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

} // namespace ggnpu

#endif // GGNPU_Q4_1_H
