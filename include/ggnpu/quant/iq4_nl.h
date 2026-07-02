#ifndef GGNPU_IQ4_NL_H
#define GGNPU_IQ4_NL_H

#include <cstdint>
#include <vector>

namespace ggnpu {

// IQ4_NL block: an fp16 scale `d` followed by 16 bytes of 4-bit indices into a
// fixed 16-entry non-linear codebook (32 values/block, 18 bytes).

// Dequantize one 18-byte IQ4_NL block (32 values) to float.
void dequant_iq4_nl_block(const uint8_t* block, float* out);

// Decode IQ4_NL weights for the NPU int8 matmul: per-row symmetric int8 with one
// scale per row (matches the Q4_0/Q4_K/Q6_K convention the kernel expects). GGUF
// 2D layout: n_cols (K) = dims[0], n_rows (N) = dims[1].
void decode_iq4_nl_for_npu(const uint8_t* gguf_data, size_t data_size,
                           int64_t n_rows, int64_t n_cols,
                           std::vector<int8_t>& int8_output,
                           std::vector<float>& scales_output);

} // namespace ggnpu

#endif // GGNPU_IQ4_NL_H
