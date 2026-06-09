#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <memory>
#include <sstream>
#include "ggnpu/model.h"
#include "ggnpu/tensor.h"
#include "ggnpu/backend.h"

namespace {

template<typename T>
typename std::enable_if<std::is_pointer<T>::value, std::string>::type
to_str(const T& v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

template<typename T>
typename std::enable_if<std::is_enum<T>::value, std::string>::type
to_str(const T& v) { return std::to_string(static_cast<int>(v)); }

template<typename T>
typename std::enable_if<!std::is_pointer<T>::value && !std::is_enum<T>::value && !std::is_same<T, std::string>::value, std::string>::type
to_str(const T& v) { return std::to_string(v); }

std::string to_str(const std::string& v) { return "\"" + v + "\""; }

std::string to_str(const char* v) { return std::string(v); }

}

int tests_passed = 0;
int tests_failed = 0;

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        std::cerr << "  FAIL: " << msg << " (expected " << to_str(b) << ", got " << to_str(a) << ")\n"; \
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

void test_llama_hparams_defaults() {
    std::cout << "  test_llama_hparams_defaults\n";

    ggnpu::LlamaHParams hparams;

    ASSERT_EQ(hparams.vocab_size, 32000, "Default vocab size");
    ASSERT_EQ(hparams.context_length, 4096, "Default context length");
    ASSERT_EQ(hparams.embedding_length, 4096, "Default embedding length");
    ASSERT_EQ(hparams.block_count, 32, "Default block count");
    ASSERT_EQ(hparams.feed_forward_length, 11008, "Default FFN length");
    ASSERT_EQ(hparams.attention_head_count, 32, "Default attention heads");
    ASSERT_EQ(hparams.attention_head_count_kv, 8, "Default KV heads");
    ASSERT_NEAR(hparams.attention_layer_norm_rms_epsilon, 5e-5, 1e-10, "Default RMS epsilon is 5e-5");
    ASSERT_EQ(hparams.rope_dimension_count, 128, "Default rope dims");
    ASSERT_NEAR(hparams.rope_freq_scale, 1.0f, 0.001f, "Default rope freq scale");
    ASSERT_EQ(hparams.rope_freq_base, 10000, "Default rope freq base");
}

void test_backend_interface() {
    std::cout << "  test_backend_interface\n";

    auto backend = ggnpu::create_cpu_ref_backend();
    ASSERT_TRUE(backend != nullptr, "CPU ref backend created");
    ASSERT_TRUE(backend->is_available(), "CPU ref backend is available");
    ASSERT_EQ(backend->name(), "cpu_ref", "Backend name is cpu_ref");
    ASSERT_EQ(backend->last_error(), ggnpu::Status::OK, "Last error is OK");
}

void test_rms_norm_cpu_ref() {
    std::cout << "  test_rms_norm_cpu_ref\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    std::vector<float> input = {3.0f, 4.0f};
    std::vector<float> output(2, 0.0f);

    ggnpu::RmsNormParams params;
    params.input = input.data();
    params.output = output.data();
    params.size = 2;
    params.eps = 1e-5f;

    ggnpu::Status st = backend->rms_norm(params);
    ASSERT_EQ(st, ggnpu::Status::OK, "RMSNorm returns OK");

    ASSERT_NEAR(output[0], 0.8485f, 0.001f, "RMSNorm output[0]");
    ASSERT_NEAR(output[1], 1.1314f, 0.001f, "RMSNorm output[1]");
}

void test_silu_cpu_ref() {
    std::cout << "  test_silu_cpu_ref\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    std::vector<float> input = {0.0f, 1.0f, -1.0f};
    std::vector<float> output(3, 0.0f);

    ggnpu::SiluParams params;
    params.input = input.data();
    params.output = output.data();
    params.size = 3;

    ggnpu::Status st = backend->silu(params);
    ASSERT_EQ(st, ggnpu::Status::OK, "Silu returns OK");

    ASSERT_NEAR(output[0], 0.0f, 0.001f, "Silu(0)");
    ASSERT_NEAR(output[1], 0.7311f, 0.001f, "Silu(1)");
    ASSERT_NEAR(output[2], -0.2689f, 0.001f, "Silu(-1)");
}

void test_softmax_cpu_ref() {
    std::cout << "  test_softmax_cpu_ref\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    std::vector<float> input = {1.0f, 2.0f, 3.0f};
    std::vector<float> output(3, 0.0f);

    ggnpu::SoftmaxParams params;
    params.input = input.data();
    params.output = output.data();
    params.rows = 1;
    params.cols = 3;

    ggnpu::Status st = backend->softmax(params);
    ASSERT_EQ(st, ggnpu::Status::OK, "Softmax returns OK");

    float sum = output[0] + output[1] + output[2];
    ASSERT_NEAR(sum, 1.0f, 0.001f, "Softmax sums to 1.0");

    ASSERT_TRUE(output[0] < output[1], "softmax output increasing");
    ASSERT_TRUE(output[1] < output[2], "softmax output increasing");
}

void test_mul_mat_q_f32_cpu_ref() {
    std::cout << "  test_mul_mat_q_f32_cpu_ref\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> C(4, 0.0f);

    ggnpu::MulMatParams params;
    params.A = A.data();
    params.B = B.data();
    params.C = C.data();
    params.M = 2;
    params.N = 2;
    params.K = 2;
    params.lda = 2;
    params.ldb = 2;
    params.ldc = 2;
    params.n_batches = 1;
    params.B_type = ggnpu::GgmlType::F32;

    ggnpu::Status st = backend->mul_mat_q(params);
    ASSERT_EQ(st, ggnpu::Status::OK, "mul_mat_q F32 returns OK");

    ASSERT_NEAR(C[0], 19.0f, 0.01f, "C[0] = 1*5 + 2*7");
    ASSERT_NEAR(C[1], 22.0f, 0.01f, "C[1] = 1*6 + 2*8");
    ASSERT_NEAR(C[2], 43.0f, 0.01f, "C[2] = 3*5 + 4*7");
    ASSERT_NEAR(C[3], 50.0f, 0.01f, "C[3] = 3*6 + 4*8");
}

void test_mul_mat_q_invalid_params() {
    std::cout << "  test_mul_mat_q_invalid_params\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    ggnpu::MulMatParams params;
    params.A = nullptr;
    params.B = nullptr;
    params.C = nullptr;
    params.M = 0;
    params.N = 0;
    params.K = 0;

    ggnpu::Status st = backend->mul_mat_q(params);
    ASSERT_EQ(st, ggnpu::Status::INVALID_PARAM, "mul_mat_q with null params returns INVALID_PARAM");
}

void test_rms_norm_invalid_params() {
    std::cout << "  test_rms_norm_invalid_params\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    ggnpu::RmsNormParams params;
    params.input = nullptr;
    params.output = nullptr;
    params.size = 0;

    ggnpu::Status st = backend->rms_norm(params);
    ASSERT_EQ(st, ggnpu::Status::INVALID_PARAM, "rms_norm with null params returns INVALID_PARAM");
}

void test_flash_attn_cpu_ref() {
    std::cout << "  test_flash_attn_cpu_ref\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    std::vector<float> Q = {1.0f, 0.0f};
    std::vector<float> K = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> V = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> output(2, 0.0f);

    ggnpu::AttnParams params;
    params.Q = Q.data();
    params.K = K.data();
    params.V = V.data();
    params.output = output.data();
    params.batch_size = 1;
    params.n_head = 1;
    params.head_dim = 2;
    params.ctx_len = 2;
    params.freq_factors = nullptr;

    ggnpu::Status st = backend->flash_attn(params);
    ASSERT_EQ(st, ggnpu::Status::OK, "flash_attn returns OK");

    ASSERT_NEAR(output[0], 0.670f, 0.01f, "flash_attn output[0]");
    ASSERT_NEAR(output[1], 0.330f, 0.01f, "flash_attn output[1]");
}

void test_rope_cpu_ref() {
    std::cout << "  test_rope_cpu_ref\n";

    auto backend = ggnpu::create_cpu_ref_backend();

    std::vector<float> data = {3.0f, 4.0f};

    ggnpu::RopeParams params;
    params.data = data.data();
    params.n_dims = 2;
    params.offset = 0;
    params.freq_scale = 1.0f;
    params.freq_base = 10000.0f;
    params.rope_dims = 2;

    ggnpu::Status st = backend->rope(params);
    ASSERT_EQ(st, ggnpu::Status::OK, "rope returns OK");

    ASSERT_NEAR(data[0], 3.0f, 0.001f, "ROPE output[0] at offset 0");
    ASSERT_NEAR(data[1], 4.0f, 0.001f, "ROPE output[1] at offset 0");

    std::vector<float> data2 = {3.0f, 4.0f};
    
    ggnpu::RopeParams params2;
    params2.data = data2.data();
    params2.n_dims = 2;
    params2.offset = 1;
    params2.freq_scale = 1.0f;
    params2.freq_base = 10000.0f;
    params2.rope_dims = 2;
    
    st = backend->rope(params2);
    ASSERT_EQ(st, ggnpu::Status::OK, "rope with offset=1 returns OK");

    ASSERT_NEAR(data2[0], -1.745f, 0.05f, "ROPE output[0] at offset 1");
    ASSERT_NEAR(data2[1], 4.686f, 0.05f, "ROPE output[1] at offset 1");
}

void test_sync() {
    std::cout << "  test_sync\n";

    auto backend = ggnpu::create_cpu_ref_backend();
    backend->sync();
    tests_passed++;
}

void run_tests() {
    std::cout << "=== Architecture / Backend Tests ===\n\n";

    test_llama_hparams_defaults();
    test_backend_interface();
    test_rms_norm_cpu_ref();
    test_silu_cpu_ref();
    test_softmax_cpu_ref();
    test_mul_mat_q_f32_cpu_ref();
    test_mul_mat_q_invalid_params();
    test_rms_norm_invalid_params();
    test_flash_attn_cpu_ref();
    test_rope_cpu_ref();
    test_sync();

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
