#include "tensor.h"
#include "quant/iquants.h"
#include "quant/kquant.h"  // fp16_to_f32, QK_K
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace ggnpu {

namespace {

// ---- Exact ggml grid/codebook tables (verbatim from ggml-common.h) ----
#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "iq_grids.inc"
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END

// ggml block layouts (ggml_half == uint16 fp16). Packed byte-for-byte with the
// GGUF data (static_asserts in ggml-common.h confirm no padding).
typedef struct { uint16_t d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
typedef struct { uint16_t d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs;
typedef struct { uint16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s;
typedef struct { uint16_t d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs;
typedef struct { uint16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[QK_K/64]; } block_iq3_s;
typedef struct { uint16_t d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s;
typedef struct { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; } block_iq1_m;

// IQ1S_DELTA (0.125f) comes from iq_grids.inc, matching ggml's canonical value.

} // namespace

// ================= block dequant (256 values) — verbatim ports =================

void dequant_iq2_xxs_block(const uint8_t* b, float* y) {
    block_iq2_xxs x; std::memcpy(&x, b, sizeof(x));
    const float d = fp16_to_f32(x.d);
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    for (int ib32 = 0; ib32 < (int)(QK_K/32); ++ib32) {
        std::memcpy(aux32, x.qs + 4*ib32, 2*sizeof(uint32_t));
        const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
            const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
            for (int j = 0; j < 8; ++j)
                y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}

void dequant_iq2_xs_block(const uint8_t* b, float* y) {
    block_iq2_xs x; std::memcpy(&x, b, sizeof(x));
    const float d = fp16_to_f32(x.d);
    float db[2];
    for (int ib32 = 0; ib32 < (int)(QK_K/32); ++ib32) {
        db[0] = d * (0.5f + (x.scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (x.scales[ib32] >>  4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (x.qs[4*ib32 + l] & 511));
            const uint8_t signs = ksigns_iq2xs[x.qs[4*ib32 + l] >> 9];
            for (int j = 0; j < 8; ++j)
                y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
    }
}

void dequant_iq2_s_block(const uint8_t* b, float* y) {
    block_iq2_s x; std::memcpy(&x, b, sizeof(x));
    const float d = fp16_to_f32(x.d);
    const uint8_t* qs = x.qs;
    const uint8_t* qh = x.qh;
    const uint8_t* signs = qs + QK_K/8;
    float db[2];
    for (int ib32 = 0; ib32 < (int)(QK_K/32); ++ib32) {
        db[0] = d * (0.5f + (x.scales[ib32] & 0xf)) * 0.25f;
        db[1] = d * (0.5f + (x.scales[ib32] >>  4)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const float dl = db[l/2];
            const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)));
            for (int j = 0; j < 8; ++j)
                y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
            y += 8;
        }
        qs += 4;
        signs += 4;
    }
}

void dequant_iq3_xxs_block(const uint8_t* b, float* y) {
    block_iq3_xxs x; std::memcpy(&x, b, sizeof(x));
    const float d = fp16_to_f32(x.d);
    const uint8_t* qs = x.qs;
    const uint8_t* scales_and_signs = qs + QK_K/4;
    uint32_t aux32;
    for (int ib32 = 0; ib32 < (int)(QK_K/32); ++ib32) {
        std::memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
            const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2*l+0]);
            const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2*l+1]);
            for (int j = 0; j < 4; ++j) {
                y[j+0] = db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                y[j+4] = db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
    }
}

void dequant_iq3_s_block(const uint8_t* b, float* y) {
    block_iq3_s x; std::memcpy(&x, b, sizeof(x));
    const float d = fp16_to_f32(x.d);
    const uint8_t* qs = x.qs;
    const uint8_t* qh = x.qh;
    const uint8_t* signs = x.signs;
    for (int ib32 = 0; ib32 < (int)(QK_K/32); ib32 += 2) {
        const float db1 = d * (1 + 2*(x.scales[ib32/2] & 0xf));
        const float db2 = d * (1 + 2*(x.scales[ib32/2] >>  4));
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
            const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
            for (int j = 0; j < 4; ++j) {
                y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
        signs += 4;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
            const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
            for (int j = 0; j < 4; ++j) {
                y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
            }
            y += 8;
        }
        qh += 2;
        qs += 8;
        signs += 4;
    }
}

void dequant_iq1_s_block(const uint8_t* b, float* y) {
    block_iq1_s x; std::memcpy(&x, b, sizeof(x));
    const float d = fp16_to_f32(x.d);
    const uint8_t* qs = x.qs;
    const uint16_t* qh = x.qh;
    for (int ib = 0; ib < (int)(QK_K/32); ++ib) {
        const float dl = d * (2*((qh[ib] >> 12) & 7) + 1);
        const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 4; ++l) {
            const int8_t* grid = (const int8_t*)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
            for (int j = 0; j < 8; ++j)
                y[j] = dl * (grid[j] + delta);
            y += 8;
        }
        qs += 4;
    }
}

void dequant_iq1_m_block(const uint8_t* b, float* y) {
    block_iq1_m x; std::memcpy(&x, b, sizeof(x));
    const uint16_t* sc = (const uint16_t*)x.scales;
    uint16_t scale_u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = fp16_to_f32(scale_u16);
    const uint8_t* qs = x.qs;
    const uint8_t* qh = x.qh;
    float delta[4];
    uint16_t idx[4];
    for (int ib = 0; ib < (int)(QK_K/32); ++ib) {
        const float dl1 = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1);
        const float dl2 = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1);
        idx[0] = qs[0] | ((qh[0] << 8) & 0x700);
        idx[1] = qs[1] | ((qh[0] << 4) & 0x700);
        idx[2] = qs[2] | ((qh[1] << 8) & 0x700);
        idx[3] = qs[3] | ((qh[1] << 4) & 0x700);
        delta[0] = qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[1] = qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[2] = qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA;
        delta[3] = qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA;
        for (int l = 0; l < 2; ++l) {
            const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) y[j] = dl1 * (grid[j] + delta[l]);
            y += 8;
        }
        for (int l = 2; l < 4; ++l) {
            const int8_t* grid = (const int8_t*)(iq1s_grid + idx[l]);
            for (int j = 0; j < 8; ++j) y[j] = dl2 * (grid[j] + delta[l]);
            y += 8;
        }
        qs += 4;
        qh += 2;
    }
}

// ================= per-row int8 requant (shared) =================

namespace {
using BlockDeqFn = void (*)(const uint8_t*, float*);

void decode_iq_rows(const uint8_t* gguf_data, size_t data_size,
                    int64_t n_rows, int64_t n_cols, size_t block_bytes,
                    BlockDeqFn deq,
                    std::vector<int8_t>& int8_output,
                    std::vector<float>& scales_output) {
    const size_t blocks_per_row = (static_cast<size_t>(n_cols) + QK_K - 1) / QK_K;
    const size_t row_bytes = blocks_per_row * block_bytes;
    if (n_rows <= 0 || n_cols <= 0 ||
        static_cast<size_t>(n_rows) * row_bytes != data_size) {
        // Flat single-row fallback keeps a bad shape from crashing (norm-only use).
        size_t num_blocks = data_size / block_bytes;
        n_cols = static_cast<int64_t>(num_blocks * QK_K);
        n_rows = 1;
    }

    int8_output.assign(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols), 0);
    scales_output.assign(static_cast<size_t>(n_rows), 1.0f);

    std::vector<float> block_f32(QK_K);
    std::vector<float> row_f32(static_cast<size_t>(n_cols));
    for (int64_t r = 0; r < n_rows; r++) {
        const uint8_t* row_data = gguf_data + static_cast<size_t>(r) * row_bytes;
        float max_abs = 0.0f;
        for (size_t bl = 0; bl < blocks_per_row; bl++) {
            deq(row_data + bl * block_bytes, block_f32.data());
            for (size_t j = 0; j < QK_K; j++) {
                size_t k = bl * QK_K + j;
                if (k < static_cast<size_t>(n_cols)) {
                    row_f32[k] = block_f32[j];
                    max_abs = std::max(max_abs, std::fabs(block_f32[j]));
                }
            }
        }
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        float inv = 1.0f / scale;
        scales_output[static_cast<size_t>(r)] = scale;
        for (int64_t k = 0; k < n_cols; k++) {
            float q = std::nearbyint(row_f32[static_cast<size_t>(k)] * inv);
            int8_output[static_cast<size_t>(r) * static_cast<size_t>(n_cols) + static_cast<size_t>(k)] =
                static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
        }
    }
}
} // namespace

void decode_iq2_xxs_for_npu(const uint8_t* d, size_t s, int64_t r, int64_t c,
                            std::vector<int8_t>& o, std::vector<float>& sc) {
    decode_iq_rows(d, s, r, c, IQ2_XXS_BLOCK_BYTES, dequant_iq2_xxs_block, o, sc);
}
void decode_iq2_xs_for_npu(const uint8_t* d, size_t s, int64_t r, int64_t c,
                           std::vector<int8_t>& o, std::vector<float>& sc) {
    decode_iq_rows(d, s, r, c, IQ2_XS_BLOCK_BYTES, dequant_iq2_xs_block, o, sc);
}
void decode_iq2_s_for_npu(const uint8_t* d, size_t s, int64_t r, int64_t c,
                          std::vector<int8_t>& o, std::vector<float>& sc) {
    decode_iq_rows(d, s, r, c, IQ2_S_BLOCK_BYTES, dequant_iq2_s_block, o, sc);
}
void decode_iq3_xxs_for_npu(const uint8_t* d, size_t s, int64_t r, int64_t c,
                            std::vector<int8_t>& o, std::vector<float>& sc) {
    decode_iq_rows(d, s, r, c, IQ3_XXS_BLOCK_BYTES, dequant_iq3_xxs_block, o, sc);
}
void decode_iq3_s_for_npu(const uint8_t* d, size_t s, int64_t r, int64_t c,
                          std::vector<int8_t>& o, std::vector<float>& sc) {
    decode_iq_rows(d, s, r, c, IQ3_S_BLOCK_BYTES, dequant_iq3_s_block, o, sc);
}
void decode_iq1_s_for_npu(const uint8_t* d, size_t s, int64_t r, int64_t c,
                          std::vector<int8_t>& o, std::vector<float>& sc) {
    decode_iq_rows(d, s, r, c, IQ1_S_BLOCK_BYTES, dequant_iq1_s_block, o, sc);
}
void decode_iq1_m_for_npu(const uint8_t* d, size_t s, int64_t r, int64_t c,
                          std::vector<int8_t>& o, std::vector<float>& sc) {
    decode_iq_rows(d, s, r, c, IQ1_M_BLOCK_BYTES, dequant_iq1_m_block, o, sc);
}

} // namespace ggnpu
