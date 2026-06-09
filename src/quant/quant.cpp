#include "quant/quant.h"
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

namespace ggnpu {

// Centralized dispatch: pick the right decoder based on GgmlType
void decode_for_npu(GgmlType type, const uint8_t* gguf_data, size_t data_size,
                    std::vector<int8_t>& int8_output,
                    std::vector<float>& scales_output) {
    switch (type) {
        case GgmlType::Q4_0:
            decode_q4_0_for_npu(gguf_data, data_size, int8_output, scales_output);
            break;

        case GgmlType::Q8_0:
            decode_q8_0_for_npu(gguf_data, data_size, int8_output, scales_output);
            break;

        case GgmlType::Q4_K:
            decode_q4_k_for_npu(gguf_data, data_size, int8_output, scales_output);
            break;

        case GgmlType::Q6_K:
            decode_q6_k_for_npu(gguf_data, data_size, int8_output, scales_output);
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

        default:
            std::cerr << "Warning: unsupported quant type " << ggml_type_name(type)
                      << " for NPU, falling back to F32\n";
            // Fallback: treat as F32
            decode_for_npu(GgmlType::F32, gguf_data, data_size, int8_output, scales_output);
            break;
    }
}

} // namespace ggnpu
