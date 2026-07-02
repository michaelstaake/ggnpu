#ifndef GGNPU_IQ4_XS_H
#define GGNPU_IQ4_XS_H

#include <cstdint>
#include <vector>

namespace ggnpu {

// IQ4_XS super-block: 136 bytes, 256 values. Reuses the IQ4_NL 16-entry
// non-linear codebook, but with a 256-value super-block and 8 per-32 6-bit
// scales (4 low bits from scales_l, 2 high bits from scales_h), all times an
// fp16 super-block scale d. Layout: d(2) scales_h(2) scales_l[4] qs[128].

// Dequantize one 136-byte IQ4_XS block (256 values) to float.
void dequant_iq4_xs_block(const uint8_t* block, float* out);

// Decode IQ4_XS weights for the NPU int8 matmul: per-row symmetric int8 with one
// scale per row. GGUF 2D layout: n_cols (K) = dims[0], n_rows (N) = dims[1].
void decode_iq4_xs_for_npu(const uint8_t* gguf_data, size_t data_size,
                           int64_t n_rows, int64_t n_cols,
                           std::vector<int8_t>& int8_output,
                           std::vector<float>& scales_output);

} // namespace ggnpu

#endif // GGNPU_IQ4_XS_H
