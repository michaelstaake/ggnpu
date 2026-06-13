#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include "ggnpu/weight_cache.h"
#include "ggnpu/cache.h"
#include "ggnpu/tensor.h"

int tests_passed = 0;
int tests_failed = 0;

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        std::cerr << "  FAIL: " << msg << " (expected " << (b) << ", got " << (a) << ")\n"; \
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

void test_memory_cache_hit() {
    std::cout << "  test_memory_cache_hit\n";
    std::string tmp_dir = "/tmp/ggnpu_test_weight_cache_" + std::to_string(std::time(nullptr));
    ggnpu::CompileCache cache(tmp_dir, true);
    ggnpu::WeightCache weight_cache(cache);

    // Create a simple F32 tensor (4 rows x 8 cols = 128 bytes)
    std::vector<float> data(32, 1.0f);
    const uint8_t* gguf_data = reinterpret_cast<const uint8_t*>(data.data());

    // First call: cache miss, should decode
    const int8_t* result = weight_cache.get_or_decode("test_tensor", gguf_data,
                                                       data.size() * sizeof(float),
                                                       ggnpu::GgmlType::F32, 4, 8);
    ASSERT_TRUE(result != nullptr, "first call returns non-null");

    // Second call: memory cache hit, should return same pointer
    const int8_t* result2 = weight_cache.get_or_decode("test_tensor", gguf_data,
                                                        data.size() * sizeof(float),
                                                        ggnpu::GgmlType::F32, 4, 8);
    ASSERT_TRUE(result == result2, "memory cache hit returns same pointer");
    ASSERT_EQ(weight_cache.cache_size(), 1, "cache size is 1 after two calls");
}

void test_memory_cache_different_tensors() {
    std::cout << "  test_memory_cache_different_tensors\n";
    std::string tmp_dir = "/tmp/ggnpu_test_weight_cache_" + std::to_string(std::time(nullptr));
    ggnpu::CompileCache cache(tmp_dir, true);
    ggnpu::WeightCache weight_cache(cache);

    std::vector<float> data1(32, 1.0f);
    std::vector<float> data2(32, 2.0f);

    const int8_t* r1 = weight_cache.get_or_decode("tensor_a",
                                                   reinterpret_cast<const uint8_t*>(data1.data()),
                                                   data1.size() * sizeof(float),
                                                   ggnpu::GgmlType::F32, 4, 8);
    const int8_t* r2 = weight_cache.get_or_decode("tensor_b",
                                                   reinterpret_cast<const uint8_t*>(data2.data()),
                                                   data2.size() * sizeof(float),
                                                   ggnpu::GgmlType::F32, 4, 8);

    ASSERT_TRUE(r1 != nullptr && r2 != nullptr, "both tensors decoded");
    ASSERT_EQ(weight_cache.cache_size(), 2, "cache size is 2 for two different tensors");
}

void test_clear() {
    std::cout << "  test_clear\n";
    std::string tmp_dir = "/tmp/ggnpu_test_weight_cache_" + std::to_string(std::time(nullptr));
    ggnpu::CompileCache cache(tmp_dir, true);
    ggnpu::WeightCache weight_cache(cache);

    std::vector<float> data(32, 1.0f);
    weight_cache.get_or_decode("tensor_a", reinterpret_cast<const uint8_t*>(data.data()),
                                data.size() * sizeof(float), ggnpu::GgmlType::F32, 4, 8);
    weight_cache.get_or_decode("tensor_b", reinterpret_cast<const uint8_t*>(data.data()),
                                data.size() * sizeof(float), ggnpu::GgmlType::F32, 4, 8);
    ASSERT_EQ(weight_cache.cache_size(), 2, "cache has 2 entries");

    weight_cache.clear();
    ASSERT_EQ(weight_cache.cache_size(), 0, "cache is empty after clear");
}

void test_get_scales() {
    std::cout << "  test_get_scales\n";
    std::string tmp_dir = "/tmp/ggnpu_test_weight_cache_" + std::to_string(std::time(nullptr));
    ggnpu::CompileCache cache(tmp_dir, true);
    ggnpu::WeightCache weight_cache(cache);

    // Scales should be empty for F32 (no quantization)
    const std::vector<float>& scales = weight_cache.get_scales("unknown", ggnpu::GgmlType::F32);
    ASSERT_TRUE(scales.empty(), "scales empty for unknown tensor");

    // Add a tensor and check scales exist
    std::vector<float> data(32, 1.0f);
    weight_cache.get_or_decode("tensor_a", reinterpret_cast<const uint8_t*>(data.data()),
                                data.size() * sizeof(float), ggnpu::GgmlType::F32, 4, 8);
    const std::vector<float>& scales2 = weight_cache.get_scales(
        "tensor_a", ggnpu::GgmlType::F32, data.size() * sizeof(float));
    ASSERT_TRUE(!scales2.empty(), "scales exist for cached tensor");
}

void test_tensor_view_overload() {
    std::cout << "  test_tensor_view_overload\n";
    std::string tmp_dir = "/tmp/ggnpu_test_weight_cache_" + std::to_string(std::time(nullptr));
    ggnpu::CompileCache cache(tmp_dir, true);
    ggnpu::WeightCache weight_cache(cache);

    ggnpu::TensorView tv;
    tv.name = "test_tensor";
    tv.dims = {4, 8};
    tv.type = ggnpu::GgmlType::F32;
    tv.n_dims = 2;
    std::vector<float> data(32, 1.0f);
    tv.data = reinterpret_cast<const uint8_t*>(data.data());

    const int8_t* result = weight_cache.get_or_decode(tv);
    ASSERT_TRUE(result != nullptr, "tensor view overload returns non-null");
}

void test_persistent_cache_roundtrip() {
    std::cout << "  test_persistent_cache_roundtrip\n";
    std::string tmp_dir = "/tmp/ggnpu_test_weight_cache_" + std::to_string(std::time(nullptr));
    ggnpu::CompileCache cache(tmp_dir, true);
    ggnpu::WeightCache weight_cache(cache);

    // Create a Q8_0 tensor (32 rows x 8 cols)
    // Q8_0: each block is 34 bytes for 32 elements
    std::vector<uint8_t> q8_data(32 * 34, 0);
    // Set the scale byte of first block to 1
    q8_data[0] = 1;

    const int8_t* result = weight_cache.get_or_decode("q8_tensor", q8_data.data(),
                                                       q8_data.size(), ggnpu::GgmlType::Q8_0, 32, 8);
    ASSERT_TRUE(result != nullptr, "Q8_0 tensor decoded");

    // Create a new weight cache instance (simulates restart)
    ggnpu::CompileCache cache2(tmp_dir, true);
    ggnpu::WeightCache weight_cache2(cache2);

    const int8_t* result2 = weight_cache2.get_or_decode("q8_tensor", q8_data.data(),
                                                         q8_data.size(), ggnpu::GgmlType::Q8_0, 32, 8);
    ASSERT_TRUE(result2 != nullptr, "Q8_0 tensor loaded from persistent cache");
}

void test_different_types_produce_different_keys() {
    std::cout << "  test_different_types_produce_different_keys\n";
    std::string tmp_dir = "/tmp/ggnpu_test_weight_cache_" + std::to_string(std::time(nullptr));
    ggnpu::CompileCache cache(tmp_dir, true);
    ggnpu::WeightCache weight_cache(cache);

    std::vector<float> data(32, 1.0f);
    const uint8_t* gguf_data = reinterpret_cast<const uint8_t*>(data.data());

    // Same name and size but different types should produce different cache entries
    weight_cache.get_or_decode("same_name", gguf_data, data.size() * sizeof(float),
                                ggnpu::GgmlType::F32, 4, 8);
    weight_cache.get_or_decode("same_name", gguf_data, data.size() * sizeof(float),
                                ggnpu::GgmlType::Q8_0, 4, 8);

    ASSERT_EQ(weight_cache.cache_size(), 2, "different types produce different cache entries");
}

int main() {
    std::cout << "=== Weight Cache Tests ===\n";
    test_memory_cache_hit();
    test_memory_cache_different_tensors();
    test_clear();
    test_get_scales();
    test_tensor_view_overload();
    test_persistent_cache_roundtrip();
    test_different_types_produce_different_keys();
    std::cout << "\nResults: " << tests_passed << " passed, " << tests_failed << " failed\n";
    return tests_failed > 0 ? 1 : 0;
}
