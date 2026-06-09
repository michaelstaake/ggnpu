#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstring>

#include "tensor.h"
#include "backend.h"
#include "quant/q4_0.h"
#include "quant/q8_0.h"

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

void test_quant_types() {
    std::cout << "\n--- Quantization Types ---\n";

    assert_true(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0) == 32, "Q4_0: 32 values per block");
    assert_true(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q8_0) == 32, "Q8_0: 32 values per block");
    assert_true(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_K) == 32, "Q4_K: 32 values per block");
    assert_true(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q6_K) == 32, "Q6_K: 32 values per block");

    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::Q4_0) == 16, "Q4_0: 16 bytes per block");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::Q8_0) == 34, "Q8_0: 34 bytes per block");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::Q4_K) == 48, "Q4_K: 48 bytes per block");
    assert_true(ggnpu::ggml_type_size(ggnpu::GgmlType::Q6_K) == 64, "Q6_K: 64 bytes per block");
}

void test_rms_norm() {
    std::cout << "\n--- RMS Norm ---\n";

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> output(5);

    ggnpu::RmsNormParams params;
    params.input = input.data();
    params.output = output.data();
    params.size = 5;
    params.eps = 1e-5f;

    auto backend = ggnpu::create_cpu_ref_backend();
    auto status = backend->rms_norm(params);

    assert_true(status == ggnpu::Status::OK, "RMS norm returns OK");

    // Verify output is normalized
    float rms = 0.0f;
    for (int i = 0; i < 5; i++) {
        rms += output[i] * output[i];
    }
    rms = std::sqrt(rms / 5);
    assert_true(std::abs(rms - 1.0f) < 1e-5, "RMS of output ≈ 1.0");
}

void test_softmax() {
    std::cout << "\n--- Softmax ---\n";

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(4);

    ggnpu::SoftmaxParams params;
    params.input = input.data();
    params.output = output.data();
    params.rows = 1;
    params.cols = 4;

    auto backend = ggnpu::create_cpu_ref_backend();
    auto status = backend->softmax(params);

    assert_true(status == ggnpu::Status::OK, "Softmax returns OK");

    // Verify sum = 1
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += output[i];
    }
    assert_true(std::abs(sum - 1.0f) < 1e-5, "Softmax sum ≈ 1.0");

    // Verify all positive
    for (int i = 0; i < 4; i++) {
        assert_true(output[i] > 0, "Softmax output[" + std::to_string(i) + "] > 0");
    }
}

void test_silu() {
    std::cout << "\n--- SiLU ---\n";

    std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> output(5);

    ggnpu::SiluParams params;
    params.input = input.data();
    params.output = output.data();
    params.size = 5;

    auto backend = ggnpu::create_cpu_ref_backend();
    auto status = backend->silu(params);

    assert_true(status == ggnpu::Status::OK, "SiLU returns OK");

    // SiLU(0) = 0
    assert_true(std::abs(output[2]) < 1e-5, "SiLU(0) ≈ 0");

    // SiLU is approximately linear for large positive values
    assert_true(output[4] < input[4], "SiLU(2) < 2");
}

void test_rope() {
    std::cout << "\n--- RoPE ---\n";

    std::vector<float> data = {1.0f, 0.0f, 0.0f, 1.0f}; // [cos(0), sin(0), cos(0), sin(0)]

    ggnpu::RopeParams params;
    params.data = data.data();
    params.n_dims = 4;
    params.offset = 0;
    params.freq_scale = 1.0f;
    params.freq_base = 10000.0f;
    params.rope_dims = 4;

    auto backend = ggnpu::create_cpu_ref_backend();
    auto status = backend->rope(params);

    assert_true(status == ggnpu::Status::OK, "RoPE returns OK");

    // At offset 0, RoPE should be identity
    assert_true(std::abs(data[0] - 1.0f) < 1e-5, "RoPE[0] ≈ 1.0 at offset 0");
    assert_true(std::abs(data[1] - 0.0f) < 1e-5, "RoPE[1] ≈ 0.0 at offset 0");
}

void test_matmul() {
    std::cout << "\n--- MatMul ---\n";

    // 2x3 x 3x2 = 2x2
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {7, 8, 9, 10, 11, 12};
    std::vector<float> C(4, 0);

    ggnpu::MulMatParams params;
    params.A = A.data();
    params.B = B.data();
    params.C = C.data();
    params.M = 2;
    params.N = 2;
    params.K = 3;
    params.lda = 3;
    params.ldb = 2;
    params.ldc = 2;
    params.n_batches = 1;
    params.B_type = ggnpu::GgmlType::F32;

    auto backend = ggnpu::create_cpu_ref_backend();
    auto status = backend->mul_mat_q(params);

    assert_true(status == ggnpu::Status::OK, "MatMul returns OK");

    // Expected: [[58, 64], [139, 154]]
    assert_true(std::abs(C[0] - 58.0f) < 0.1f, "MatMul[0,0] = 58");
    assert_true(std::abs(C[1] - 64.0f) < 0.1f, "MatMul[0,1] = 64");
    assert_true(std::abs(C[2] - 139.0f) < 0.1f, "MatMul[1,0] = 139");
    assert_true(std::abs(C[3] - 154.0f) < 0.1f, "MatMul[1,1] = 154");
}

void test_q8_0_matmul() {
    std::cout << "\n--- Q8_0 MatMul ---\n";

    // Simple 2x4 x 4x2 = 2x2 with Q8_0 quantized weights
    std::vector<int8_t> A = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int8_t> B = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> C(4, 0);

    ggnpu::MulMatParams params;
    params.A = A.data();
    params.B = B.data();
    params.C = C.data();
    params.M = 2;
    params.N = 2;
    params.K = 4;
    params.lda = 4;
    params.ldb = 2;
    params.ldc = 2;
    params.n_batches = 1;
    params.B_type = ggnpu::GgmlType::Q8_0;

    auto backend = ggnpu::create_cpu_ref_backend();
    auto status = backend->mul_mat_q(params);

    assert_true(status == ggnpu::Status::OK, "Q8_0 MatMul returns OK");

    // Expected: [[50, 60], [114, 140]]
    assert_true(std::abs(C[0] - 50.0f) < 0.1f, "Q8_0 MatMul[0,0] = 50");
    assert_true(std::abs(C[1] - 60.0f) < 0.1f, "Q8_0 MatMul[0,1] = 60");
    assert_true(std::abs(C[2] - 114.0f) < 0.1f, "Q8_0 MatMul[1,0] = 114");
    assert_true(std::abs(C[3] - 140.0f) < 0.1f, "Q8_0 MatMul[1,1] = 140");
}

void test_q4_0_decode() {
    std::cout << "\n--- Q4_0 Decode ---\n";

    // Create a simple Q4_0 block
    ggnpu::Q4_0Block block;
    block.d = 1;
    memset(block.signs, 0, 16);
    memset(block.qs, 0, 8);

    std::vector<uint8_t> gguf_data(sizeof(ggnpu::Q4_0Block));
    memcpy(gguf_data.data(), &block, sizeof(ggnpu::Q4_0Block));

    std::vector<int8_t> int8_output;
    std::vector<float> scales_output;

    ggnpu::decode_q4_0_for_npu(gguf_data.data(), gguf_data.size(), int8_output, scales_output);

    assert_true(int8_output.size() == 16, "Q4_0 decode produces 16 values");
    assert_true(scales_output.size() == 1, "Q4_0 decode produces 1 scale");

    // All values should be -8 since qs is all zeros (nibble=0, val=0-8=-8)
    for (int i = 0; i < 16; i++) {
        assert_true(int8_output[i] == -8, "Q4_0 decoded value[" + std::to_string(i) + "] = -8");
    }
}

void run_tests(const std::string& filter) {
    std::cout << "Running ggnpu tests...\n";

    test_quant_types();
    test_rms_norm();
    test_softmax();
    test_silu();
    test_rope();
    test_matmul();
    test_q8_0_matmul();
    test_q4_0_decode();

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
