#ifndef GGNPU_TENSOR_H
#define GGNPU_TENSOR_H

#include <vector>
#include <string>
#include <cstdint>
#include <functional>

namespace ggnpu {

enum class GgmlType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
};

size_t ggml_type_size(GgmlType type);
size_t ggml_blck_size(GgmlType type);
float ggml_type_sizef(GgmlType type);
const char* ggml_type_name(GgmlType type);

struct TensorView {
    std::string name;
    std::vector<uint64_t> dims;
    GgmlType type;
    const uint8_t* data = nullptr;
    size_t data_offset = 0;
    size_t stride = 0;
    int32_t n_dims = 0;

    size_t element_count() const {
        size_t count = 1;
        for (auto d : dims) count *= d;
        return count;
    }

    size_t data_size() const {
        return (element_count() / ggml_blck_size(type)) * ggml_type_size(type);
    }
};

struct QuantParams {
    GgmlType src_type;
    GgmlType dst_type;
    int ndim;
};

} // namespace ggnpu

#endif // GGNPU_TENSOR_H
