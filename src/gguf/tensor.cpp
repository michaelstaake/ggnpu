#include "tensor.h"
#include <cstring>

namespace ggnpu {

namespace {

// Indexed by the ggml type id (matches llama.cpp's ggml_type enum)
constexpr const char* type_names[] = {
    "F32", "F16", "Q4_0", "Q4_1", "UNKNOWN", "UNKNOWN",
    "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K",
    "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS",
    "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS",
    "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16",
};

constexpr size_t block_sizes[] = {
    1, 1, 32, 32, 1, 1,
    32, 32, 32, 32, 256, 256,
    256, 256, 256, 256, 256, 256,
    256, 256, 32, 256, 256, 256,
    1, 1, 1, 1, 1, 256, 1,
};

constexpr size_t type_sizes[] = {
    // Q4_0 block is 18 bytes: fp16 scale (2) + 16 packed 4-bit nibbles (32 vals).
    // (The other legacy low-bit sizes here — Q4_1/Q5_0/Q5_1 — are unused and
    // unverified; only Q4_0 and the K-quants/Q8_0 are exercised.)
    4, 2, 18, 18, 1, 1,
    20, 22, 34, 34, 84, 110,
    144, 176, 210, 292, 66, 74,
    98, 50, 18, 110, 82, 136,
    1, 2, 4, 8, 8, 56, 2
};

} // namespace

size_t ggml_type_size(GgmlType type) {
    int idx = static_cast<int>(type);
    if (idx < 0 || idx >= static_cast<int>(sizeof(type_sizes) / sizeof(type_sizes[0]))) return 4;
    return type_sizes[idx];
}

size_t ggml_blck_size(GgmlType type) {
    int idx = static_cast<int>(type);
    if (idx < 0 || idx >= static_cast<int>(sizeof(block_sizes) / sizeof(block_sizes[0]))) return 1;
    return block_sizes[idx];
}

float ggml_type_sizef(GgmlType type) {
    // Fractional size per element in bytes
    return static_cast<float>(ggml_type_size(type)) / static_cast<float>(ggml_blck_size(type));
}

const char* ggml_type_name(GgmlType type) {
    int idx = static_cast<int>(type);
    if (idx < 0 || idx >= static_cast<int>(sizeof(type_names) / sizeof(type_names[0]))) return "UNKNOWN";
    return type_names[idx];
}

} // namespace ggnpu
