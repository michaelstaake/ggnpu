#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <sstream>
#include "ggnpu/gguf.h"
#include "ggnpu/tensor.h"

int tests_passed = 0;
int tests_failed = 0;

#define ASSERT_EQ(a, b, msg) do { \
    if (static_cast<int64_t>(a) != static_cast<int64_t>(b)) { \
        std::cerr << "  FAIL: " << msg << " (expected " << static_cast<int64_t>(b) << ", got " << static_cast<int64_t>(a) << ")\n"; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << msg << "\n"; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    if (std::fabs((a) - (b)) > (eps)) { \
        std::cerr << "  FAIL: " << msg << " (expected ~" << (b) << ", got " << (a) << ")\n"; \
        tests_failed++; \
    } else { \
        tests_passed++; \
    } \
} while(0)

void test_ggml_type_functions() {
    std::cout << "  test_ggml_type_functions\n";

    ASSERT_EQ(ggnpu::ggml_type_size(ggnpu::GgmlType::F32), 4, "F32 size");
    ASSERT_EQ(ggnpu::ggml_type_size(ggnpu::GgmlType::F16), 2, "F16 size");
    ASSERT_EQ(ggnpu::ggml_type_size(ggnpu::GgmlType::Q8_0), 34, "Q8_0 size");
    ASSERT_EQ(ggnpu::ggml_type_size(ggnpu::GgmlType::Q4_0), 16, "Q4_0 size");
    ASSERT_EQ(ggnpu::ggml_type_size(ggnpu::GgmlType::Q4_K), 48, "Q4_K size");
    ASSERT_EQ(ggnpu::ggml_type_size(ggnpu::GgmlType::Q6_K), 64, "Q6_K size");

    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::F32), 1, "F32 block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::F16), 1, "F16 block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q8_0), 32, "Q8_0 block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0), 32, "Q4_0 block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_K), 32, "Q4_K block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q6_K), 32, "Q6_K block size");

    const char* f32_name = ggnpu::ggml_type_name(ggnpu::GgmlType::F32);
    ASSERT_TRUE(std::string(f32_name) == "F32", "F32 name");

    const char* q4_name = ggnpu::ggml_type_name(ggnpu::GgmlType::Q4_0);
    ASSERT_TRUE(std::string(q4_name) == "Q4_0", "Q4_0 name");

    const char* q8_name = ggnpu::ggml_type_name(ggnpu::GgmlType::Q8_0);
    ASSERT_TRUE(std::string(q8_name) == "Q8_0", "Q8_0 name");

    float f32_frac = ggnpu::ggml_type_sizef(ggnpu::GgmlType::F32);
    ASSERT_NEAR(f32_frac, 4.0f, 0.001f, "F32 fractional size");

    float q4_frac = ggnpu::ggml_type_sizef(ggnpu::GgmlType::Q4_0);
    ASSERT_TRUE(q4_frac > 0.4f && q4_frac < 0.6f, "Q4_0 fractional size ~0.5");
}

void test_tensor_view() {
    std::cout << "  test_tensor_view\n";

    ggnpu::TensorView tv;
    tv.name = "test_tensor";
    tv.dims = {256, 512};
    tv.type = ggnpu::GgmlType::F32;
    tv.n_dims = 2;

    size_t elem_count = tv.element_count();
    ASSERT_EQ(elem_count, 131072, "F32 element count 256x512");

    ggnpu::TensorView tv_q4;
    tv_q4.dims = {32, 32};
    tv_q4.type = ggnpu::GgmlType::Q4_0;
    tv_q4.n_dims = 2;

    size_t q4_count = tv_q4.element_count();
    ASSERT_EQ(q4_count, 1024, "Q4_0 element count 32x32");

    size_t q4_data_size = tv_q4.data_size();
    ASSERT_EQ(q4_data_size, 512, "Q4_0 data size (1024/32 * 16 = 512)");

    ggnpu::TensorView tv_q8;
    tv_q8.dims = {64, 64};
    tv_q8.type = ggnpu::GgmlType::Q8_0;
    tv_q8.n_dims = 2;

    size_t q8_data_size = tv_q8.data_size();
    ASSERT_EQ(q8_data_size, 4352, "Q8_0 data size (4096/32 * 34 = 4352)");
}

void test_gguf_types() {
    std::cout << "  test_gguf_types\n";

    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgufType::UINT8), 0, "GgufType UINT8");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgufType::INT32), 5, "GgufType INT32");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgufType::FLOAT32), 6, "GgufType FLOAT32");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgufType::STRING), 8, "GgufType STRING");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgufType::ARRAY), 9, "GgufType ARRAY");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgufType::UINT64), 10, "GgufType UINT64");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgufType::FLOAT64), 12, "GgufType FLOAT64");
}

void test_gguf_loader_structure() {
    std::cout << "  test_gguf_loader_structure\n";

    ggnpu::GgufLoader loader;

    ggnpu::GgufHeader header = loader.header();
    ASSERT_EQ(header.magic, 0, "Empty loader has zeroed header");
    ASSERT_EQ(header.version, 0, "Empty loader has zero version");
    ASSERT_EQ(header.tensor_count, 0, "Empty loader has zero tensor count");

    ASSERT_TRUE(loader.tensors().empty(), "Empty loader has no tensors");
    ASSERT_TRUE(loader.kv_pairs().empty(), "Empty loader has no kv pairs");
    ASSERT_TRUE(loader.tensor_data().empty(), "Empty loader has no tensor data");

    std::string arch = loader.architecture();
    ASSERT_TRUE(arch.empty(), "Empty loader has empty architecture");

    int64_t ctx = loader.context_length();
    ASSERT_EQ(ctx, 0, "Empty loader has zero context length");

    uint64_t embd = loader.embedding_length();
    ASSERT_EQ(embd, 0, "Empty loader has zero embedding length");
}

void test_ggml_type_enum_values() {
    std::cout << "  test_ggml_type_enum_values\n";

    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgmlType::F32), 0, "F32 enum value");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgmlType::F16), 1, "F16 enum value");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgmlType::Q4_0), 2, "Q4_0 enum value");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgmlType::Q8_0), 8, "Q8_0 enum value");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgmlType::Q4_K), 10, "Q4_K enum value");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgmlType::Q6_K), 12, "Q6_K enum value");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgmlType::I8), 24, "I8 enum value");
    ASSERT_EQ(static_cast<uint32_t>(ggnpu::GgmlType::BF16), 30, "BF16 enum value");
}

void run_tests() {
    std::cout << "=== GGUF Loader Tests ===\n\n";

    test_ggml_type_functions();
    test_tensor_view();
    test_gguf_types();
    test_gguf_loader_structure();
    test_ggml_type_enum_values();

    std::cout << "\n--- Results ---\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";

    if (tests_failed > 0) {
        std::cout << "\nSOME TESTS FAILED\n";
    } else {
        std::cout << "\nALL TESTS PASSED\n";
    }
}

int main() {
    run_tests();
    return tests_failed > 0 ? 1 : 0;
}
