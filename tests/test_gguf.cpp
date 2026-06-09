#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sstream>

#include "gguf.h"
#include "tensor.h"
#include "backend.h"

namespace {

int tests_passed = 0;
int tests_failed = 0;

void assert_true(bool cond, const std::string& msg) {
    if (cond) {
        tests_passed++;
        std::cout << "  PASS: " << msg << "\n";
    } else {
        tests_failed++;
        std::cout << "  FAIL: " << msg << "\n";
    }
}

void test_gguf_type_sizes() {
    std::cout << "\n--- GGUF Type Sizes ---\n";

    assert_true(ggnpu::ggml_blck_size(ggnpu::GgmlType::F32) == 1, "F32 block size = 1");
    assert_true(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0) == 32, "Q4_0 block size = 32");
    assert_true(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q8_0) == 32, "Q8_0 block size = 32");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::F32) == 4, "F32 type size = 4");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::Q4_0) == 16, "Q4_0 type size = 16");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::Q8_0) == 34, "Q8_0 type size = 34");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::Q4_K) == 48, "Q4_K type size = 48");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::Q6_K) == 64, "Q6_K type size = 64");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::BF16) == 2, "BF16 type size = 2");
}

void test_gguf_type_names() {
    std::cout << "\n--- GGUF Type Names ---\n";

    assert_true(std::string(ggnpu::ggml_type_name(ggnpu::GgmlType::F32)) == "F32", "F32 name");
    assert_true(std::string(ggnpu::ggml_type_name(ggnpu::GgmlType::Q4_0)) == "Q4_0", "Q4_0 name");
    assert_true(std::string(ggnpu::ggml_type_name(ggnpu::GgmlType::Q8_0)) == "Q8_0", "Q8_0 name");
    assert_true(std::string(ggnpu::ggml_type_name(ggnpu::GgmlType::Q4_K)) == "Q4_K", "Q4_K name");
    assert_true(std::string(ggnpu::ggml_type_name(ggnpu::GgmlType::Q6_K)) == "Q6_K", "Q6_K name");
}

void test_tensor_view() {
    std::cout << "\n--- Tensor View ---\n";

    ggnpu::TensorView tv;
    tv.name = "test";
    tv.dims = {128, 256};
    tv.type = ggnpu::GgmlType::F32;
    tv.n_dims = 2;

    assert_true(tv.element_count() == 128 * 256, "element_count = 32768");
    assert_true(tv.data_size() == 128 * 256 * 4, "data_size = 131072");

    ggnpu::TensorView tv_q4;
    tv_q4.name = "test_q4";
    tv_q4.dims = {128, 256};
    tv_q4.type = ggnpu::GgmlType::Q4_0;
    tv_q4.n_dims = 2;

    assert_true(tv_q4.element_count() == 128 * 256, "Q4_0 element_count = 32768");
    assert_true(tv_q4.data_size() == (128 * 256 / 32) * 16, "Q4_0 data_size = 16384");
}

void test_gguf_loader_basic() {
    std::cout << "\n--- GGUF Loader Basic ---\n";

    ggnpu::GgufLoader loader;
    assert_true(loader.header().magic == 0, "Empty loader magic = 0");
    assert_true(loader.header().version == 0, "Empty loader version = 0");
    assert_true(loader.header().tensor_count == 0, "Empty loader tensor_count = 0");
}

void test_gguf_epsilon_accessor() {
    std::cout << "\n--- GGUF Epsilon Accessor ---\n";

    ggnpu::GgufLoader loader;
    double default_eps = loader.attention_layer_norm_rms_epsilon();
    assert_true(std::abs(default_eps - 1e-5) < 1e-10,
                "Default epsilon = 1e-5 (got " + std::to_string(default_eps) + ")");
}

void run_tests(const std::string& filter) {
    std::cout << "Running ggnpu tests...\n";

    test_gguf_type_sizes();
    test_gguf_type_names();
    test_tensor_view();
    test_gguf_loader_basic();
    test_gguf_epsilon_accessor();

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << tests_passed << "\n";
    std::cout << "Failed: " << tests_failed << "\n";
    std::cout << "Total:  " << (tests_passed + tests_failed) << "\n";
}

} // namespace

int main(int argc, char* argv[]) {
    std::string filter;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--filter" && i + 1 < argc) {
            filter = argv[++i];
        }
    }

    run_tests(filter);
    return tests_failed > 0 ? 1 : 0;
}
