#include "quant/quant.h"
#include "quant/iq4_nl.h"
#include "quant/iq4_xs.h"
#include "quant/iquants.h"
#include "bf16.h"
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace ggnpu {

// Centralized dispatch: pick the right decoder based on GgmlType
void decode_for_npu(GgmlType type, const uint8_t* gguf_data, size_t data_size,
                    int64_t n_rows, int64_t n_cols,
                    std::vector<int8_t>& int8_output,
                    std::vector<float>& scales_output) {
    switch (type) {
        case GgmlType::Q4_0:
            decode_q4_0_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::Q4_1:
            decode_q4_1_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::Q4_0_4_4: {
            // ARM-repacked Q4_0: 4 rows are interleaved into a block_q4_0x4 (8 bytes
            // of 4 fp16 row scales + 64 bytes of qs, blck_size_interleave=4), laid
            // out group-major (4 rows) then block-column. De-interleave back to plain
            // row-major Q4_0 bytes, then reuse the Q4_0 decoder. Requires n_rows%4==0
            // (always true for repacked tensors; smaller ones ship as plain Q4_0).
            constexpr int QK = 32;
            constexpr size_t PLAIN_BLK = 18;  // 2 (fp16 d) + 16 (qs)
            constexpr size_t X4_BLK = 72;     // 8 (4x fp16 d) + 64 (qs)
            if (n_rows <= 0 || n_cols <= 0 || (n_rows % 4) != 0) {
                std::cerr << "Warning: Q4_0_4_4 requires n_rows%4==0 (got "
                          << n_rows << "x" << n_cols << "), falling back to F32\n";
                decode_for_npu(GgmlType::F32, gguf_data, data_size, n_rows, n_cols,
                               int8_output, scales_output);
                break;
            }
            const size_t bpr = (static_cast<size_t>(n_cols) + QK - 1) / QK;  // blocks per row
            const size_t groups = static_cast<size_t>(n_rows) / 4;
            std::vector<uint8_t> plain(static_cast<size_t>(n_rows) * bpr * PLAIN_BLK);
            for (size_t g = 0; g < groups; g++) {
                for (size_t b = 0; b < bpr; b++) {
                    const uint8_t* x4 = gguf_data + (g * bpr + b) * X4_BLK;
                    const uint8_t* xd = x4;      // 4 fp16 row scales
                    const uint8_t* xq = x4 + 8;  // 64 interleaved qs bytes
                    for (int r = 0; r < 4; r++) {
                        uint8_t* dst = plain.data() + ((g * 4 + r) * bpr + b) * PLAIN_BLK;
                        std::memcpy(dst, xd + r * 2, 2);  // this row's fp16 scale
                        uint8_t* dqs = dst + 2;           // 16-byte plain qs
                        // reverse of make_block_q4_0x4: chunks with i%4==r belong to row r,
                        // src (plain) offset (i/4)*4, dst (interleaved) offset i*4. The
                        // repack XORs every nibble with 0x8 (mask 0x88888888) to store the
                        // quants unsigned for ARM; XOR back to recover standard Q4_0 nibbles.
                        for (int i = r; i < 16; i += 4) {
                            const uint8_t* s = xq + i * 4;
                            uint8_t* d = dqs + (i / 4) * 4;
                            for (int c = 0; c < 4; c++) d[c] = s[c] ^ 0x88;
                        }
                    }
                }
            }
            decode_q4_0_for_npu(plain.data(), plain.size(), n_rows, n_cols,
                                int8_output, scales_output);
            break;
        }

        case GgmlType::Q8_0:
            decode_q8_0_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::Q2_K:
            decode_q2_k_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::Q3_K:
            decode_q3_k_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::Q4_K:
            decode_q4_k_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::Q5_K:
            decode_q5_k_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ4_NL:
            decode_iq4_nl_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ4_XS:
            decode_iq4_xs_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::Q6_K:
            decode_q6_k_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ2_XXS:
            decode_iq2_xxs_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ2_XS:
            decode_iq2_xs_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ2_S:
            decode_iq2_s_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ3_XXS:
            decode_iq3_xxs_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ3_S:
            decode_iq3_s_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ1_S:
            decode_iq1_s_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::IQ1_M:
            decode_iq1_m_for_npu(gguf_data, data_size, n_rows, n_cols, int8_output, scales_output);
            break;

        case GgmlType::F32:
            // F32 weights: fixed 1/127 scale (single scale/tensor). Used only for
            // the on-the-fly F32 matmul path and norm-style tensors — NOT the
            // per-row kq_path. Clips magnitudes > 1; fine for norms, and the F32
            // matmul weight path re-derives its own scale in the backend.
            {
                size_t elem_count = data_size / ggml_type_size(type);
                int8_output.resize(elem_count);
                scales_output.resize(1);
                scales_output[0] = 1.0f;
                const float* src = reinterpret_cast<const float*>(gguf_data);
                for (size_t i = 0; i < elem_count; i++) {
                    int32_t v = static_cast<int32_t>(src[i] * 127.0f);
                    int8_output[i] = static_cast<int8_t>(std::clamp(v, -128, 127));
                }
            }
            break;

        case GgmlType::F16:
        case GgmlType::BF16: {
            // f16/bf16 weights (2 bytes/value, no blocks). Requantize per row to
            // symmetric int8 + one scale/row, matching the kq_path convention.
            // A prior version routed F16 through the F32 fixed-1/127 case with a
            // single tensor scale — but F16 matmul weights use the per-row kq_path
            // (B laid out [N][K]), so a single scale + non-per-row decode left the
            // backend reading an all-zero B tile → zero logits (e.g. Unsloth
            // UD-Q8_K_XL, which keeps embed/output/some attn tensors in F16).
            const bool is_f16 = (type == GgmlType::F16);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(gguf_data);
            int64_t rows = n_rows, cols = n_cols;
            if (rows <= 0 || cols <= 0 ||
                static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(uint16_t) != data_size) {
                rows = 1;
                cols = static_cast<int64_t>(data_size / sizeof(uint16_t));
            }
            auto to_f32 = [is_f16](uint16_t v) { return is_f16 ? f16_to_f32(v) : bf16_to_f32(v); };
            int8_output.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
            scales_output.assign(static_cast<size_t>(rows), 1.0f);
            for (int64_t r = 0; r < rows; r++) {
                const uint16_t* row = src + static_cast<size_t>(r) * cols;
                float max_abs = 0.0f;
                for (int64_t k = 0; k < cols; k++)
                    max_abs = std::max(max_abs, std::fabs(to_f32(row[k])));
                float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
                float inv = 1.0f / scale;
                scales_output[static_cast<size_t>(r)] = scale;
                for (int64_t k = 0; k < cols; k++) {
                    float q = std::nearbyint(to_f32(row[k]) * inv);
                    int8_output[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(k)] =
                        static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
                }
            }
            break;
        }

        default:
            std::cerr << "Warning: unsupported quant type " << ggml_type_name(type)
                      << " for NPU, falling back to F32\n";
            // Fallback: treat as F32
            decode_for_npu(GgmlType::F32, gguf_data, data_size, n_rows, n_cols,
                           int8_output, scales_output);
            break;
    }
}

} // namespace ggnpu
