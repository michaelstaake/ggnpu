#ifndef GGNPU_QUANT_H
#define GGNPU_QUANT_H

#include "tensor.h"
#include <vector>
#include <cstdint>

namespace ggnpu {

// Decode quantized GGUF weights to INT8 + scales for NPU consumption
// Each function reads from gguf_data (data_size bytes) and writes:
//   int8_output:  dequantized INT8 values (block_size per block)
//   scales_output: per-block or per-group scales
void decode_q4_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

void decode_q8_0_for_npu(const uint8_t* gguf_data, size_t data_size,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

void decode_q4_k_for_npu(const uint8_t* gguf_data, size_t data_size,
                         int64_t n_rows, int64_t n_cols,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

void decode_q6_k_for_npu(const uint8_t* gguf_data, size_t data_size,
                         int64_t n_rows, int64_t n_cols,
                         std::vector<int8_t>& int8_output,
                         std::vector<float>& scales_output);

// Centralized dispatch: pick the right decoder based on GgmlType.
// For 2D weights: n_cols = dims[0] (K), n_rows = dims[1] (N). Pass 0,0 if unknown.
void decode_for_npu(GgmlType type, const uint8_t* gguf_data, size_t data_size,
                    int64_t n_rows, int64_t n_cols,
                    std::vector<int8_t>& int8_output,
                    std::vector<float>& scales_output);

} // namespace ggnpu

#endif // GGNPU_QUANT_H
