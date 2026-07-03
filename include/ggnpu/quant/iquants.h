#ifndef GGNPU_IQUANTS_H
#define GGNPU_IQUANTS_H

#include <cstdint>
#include <vector>

namespace ggnpu {

// Codebook / grid-based i-quant decoders (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S, IQ1_S, IQ1_M). Each decodes GGUF weights to per-row symmetric int8 +
// one scale per output row, matching the NPU matmul kq_path convention.
// GGUF 2D layout: n_cols (K) = dims[0], n_rows (N) = dims[1].
// The exact ggml grid tables live in src/quant/iq_grids.inc.

void decode_iq2_xxs_for_npu(const uint8_t*, size_t, int64_t, int64_t,
                            std::vector<int8_t>&, std::vector<float>&);
void decode_iq2_xs_for_npu(const uint8_t*, size_t, int64_t, int64_t,
                           std::vector<int8_t>&, std::vector<float>&);
void decode_iq2_s_for_npu(const uint8_t*, size_t, int64_t, int64_t,
                          std::vector<int8_t>&, std::vector<float>&);
void decode_iq3_xxs_for_npu(const uint8_t*, size_t, int64_t, int64_t,
                            std::vector<int8_t>&, std::vector<float>&);
void decode_iq3_s_for_npu(const uint8_t*, size_t, int64_t, int64_t,
                          std::vector<int8_t>&, std::vector<float>&);
void decode_iq1_s_for_npu(const uint8_t*, size_t, int64_t, int64_t,
                          std::vector<int8_t>&, std::vector<float>&);
void decode_iq1_m_for_npu(const uint8_t*, size_t, int64_t, int64_t,
                          std::vector<int8_t>&, std::vector<float>&);

// Block dequant (256 values) — exposed for unit tests / embedding dequant.
void dequant_iq2_xxs_block(const uint8_t*, float*);
void dequant_iq2_xs_block(const uint8_t*, float*);
void dequant_iq2_s_block(const uint8_t*, float*);
void dequant_iq3_xxs_block(const uint8_t*, float*);
void dequant_iq3_s_block(const uint8_t*, float*);
void dequant_iq1_s_block(const uint8_t*, float*);
void dequant_iq1_m_block(const uint8_t*, float*);

// Block byte sizes (256 values each).
constexpr size_t IQ2_XXS_BLOCK_BYTES = 66;
constexpr size_t IQ2_XS_BLOCK_BYTES  = 74;
constexpr size_t IQ2_S_BLOCK_BYTES   = 82;
constexpr size_t IQ3_XXS_BLOCK_BYTES = 98;
constexpr size_t IQ3_S_BLOCK_BYTES   = 110;
constexpr size_t IQ1_S_BLOCK_BYTES   = 50;
constexpr size_t IQ1_M_BLOCK_BYTES   = 56;

} // namespace ggnpu

#endif // GGNPU_IQUANTS_H
