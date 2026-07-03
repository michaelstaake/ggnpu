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
        case GgmlType::F16:
            // F32/F16 weights don't need quantization — copy as-is
            {
                size_t elem_count = data_size / ggml_type_size(type);
                int8_output.resize(elem_count);
                scales_output.resize(1);
                scales_output[0] = 1.0f;

                if (type == GgmlType::F32) {
                    const float* src = reinterpret_cast<const float*>(gguf_data);
                    for (size_t i = 0; i < elem_count; i++) {
                        int32_t v = static_cast<int32_t>(src[i] * 127.0f);
                        int8_output[i] = static_cast<int8_t>(std::clamp(v, -128, 127));
                    }
                } else {
                    const uint8_t* raw = gguf_data;
                    for (size_t i = 0; i < elem_count; i++) {
                        uint16_t bits = (static_cast<uint16_t>(raw[i * 2]) << 0) |
                                        (static_cast<uint16_t>(raw[i * 2 + 1]) << 8);
                        uint16_t sign = (bits >> 15) & 1;
                        uint16_t exp = (bits >> 10) & 0x1F;
                        uint16_t frac = bits & 0x3FF;
                        float f;
                        if (exp == 0) {
                            f = (frac == 0) ? 0.0f : (static_cast<float>(frac) / 1024.0f) * powf(2.0f, -14);
                        } else if (exp == 31) {
                            f = (frac == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
                        } else {
                            f = (sign ? -1.0f : 1.0f) * powf(2.0f, static_cast<float>(exp - 15)) * (1.0f + static_cast<float>(frac) / 1024.0f);
                        }
                        int32_t v = static_cast<int32_t>(f * 127.0f);
                        int8_output[i] = static_cast<int8_t>(std::clamp(v, -128, 127));
                    }
                }
            }
            break;

        case GgmlType::BF16: {
            // bf16 weights (2 bytes/value, no blocks). Requantize per row to
            // symmetric int8 + one scale/row, matching the kq_path convention
            // (the F16/F32 case above uses a fixed 1/127 scale, which clips real
            // weight magnitudes > 1 — fine for norms, wrong for matmul weights).
            const uint16_t* src = reinterpret_cast<const uint16_t*>(gguf_data);
            int64_t rows = n_rows, cols = n_cols;
            if (rows <= 0 || cols <= 0 ||
                static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(uint16_t) != data_size) {
                rows = 1;
                cols = static_cast<int64_t>(data_size / sizeof(uint16_t));
            }
            int8_output.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
            scales_output.assign(static_cast<size_t>(rows), 1.0f);
            for (int64_t r = 0; r < rows; r++) {
                const uint16_t* row = src + static_cast<size_t>(r) * cols;
                float max_abs = 0.0f;
                for (int64_t k = 0; k < cols; k++)
                    max_abs = std::max(max_abs, std::fabs(bf16_to_f32(row[k])));
                float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
                float inv = 1.0f / scale;
                scales_output[static_cast<size_t>(r)] = scale;
                for (int64_t k = 0; k < cols; k++) {
                    float q = std::nearbyint(bf16_to_f32(row[k]) * inv);
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
