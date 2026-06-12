#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include "ggnpu/gguf.h"

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

void test_load_nonexistent_file() {
    std::cout << "  test_load_nonexistent_file\n";
    ggnpu::GgufLoader loader;
    bool result = loader.load("/nonexistent/path/to/model.gguf");
    ASSERT_TRUE(!result, "load returns false for nonexistent file");
}

void test_load_real_model(const std::string& model_path) {
    std::cout << "  test_load_real_model\n";
    ggnpu::GgufLoader loader;
    bool result = loader.load(model_path);
    ASSERT_TRUE(result, "load succeeds for real GGUF file");

    // Verify header
    const auto& header = loader.header();
    ASSERT_EQ(header.magic, 0x46554747, "GGUF magic number (0x46554747)");
    ASSERT_EQ(header.version, 3, "GGUF version 3");
    ASSERT_TRUE(header.tensor_count > 0, "has tensors");
    ASSERT_TRUE(header.kv_count > 0, "has KV pairs");

    // Verify tensor data offset is within file
    ASSERT_TRUE(loader.tensor_data_offset() > 0, "tensor data offset > 0");

    // Verify at least one tensor exists
    const auto& tensors = loader.tensors();
    ASSERT_TRUE(tensors.size() > 0, "tensors vector not empty");

    // Check first tensor has a name
    ASSERT_TRUE(!tensors[0].name.empty(), "first tensor has a name");

    // Verify embedding length (hidden size) is reasonable for Llama
    uint64_t emb = loader.embedding_length();
    ASSERT_TRUE(emb > 0, "embedding length > 0");
    ASSERT_TRUE(emb <= 16384, "embedding length <= 16384 (reasonable upper bound)");

    // Verify attention head counts
    uint64_t heads = loader.attention_head_count();
    uint64_t heads_kv = loader.attention_head_count_kv();
    ASSERT_TRUE(heads > 0, "attention head count > 0");
    ASSERT_TRUE(heads_kv > 0, "KV head count > 0");

    // Verify architecture string
    std::string arch = loader.architecture();
    ASSERT_TRUE(!arch.empty(), "architecture string not empty");
    ASSERT_TRUE(arch == "llama" || arch.find("llama") != std::string::npos,
                "architecture is llama-like: " + arch);

    // Verify context length
    uint64_t ctx = loader.context_length();
    ASSERT_TRUE(ctx > 0, "context length > 0");

    // Verify feed forward length
    uint64_t ff = loader.feed_forward_length();
    ASSERT_TRUE(ff > 0, "feed forward length > 0");

    // Verify block count (number of layers)
    uint64_t blocks = loader.block_count();
    ASSERT_TRUE(blocks > 0, "block count > 0");

    // Verify RMS epsilon
    double eps = loader.attention_layer_norm_rms_epsilon();
    ASSERT_NEAR(eps, 1e-5, 1e-8, "RMS epsilon ~1e-5");

    // Verify rope parameters
    double rope_scale = loader.rope_freq_scale();
    uint64_t rope_base = loader.rope_freq_base();
    ASSERT_TRUE(rope_scale > 0, "rope freq scale > 0");
    ASSERT_TRUE(rope_base > 0, "rope freq base > 0");

    // Verify tensor data is accessible (mmap'd)
    const auto& tdata = loader.tensor_data();
    ASSERT_TRUE(tdata.size() > 0, "tensor data not empty");

    // Check that Q4_K and Q8_0 types exist in the model
    bool has_q4k = false;
    bool has_q80 = false;
    for (const auto& t : tensors) {
        if (t.type == ggnpu::GgmlType::Q4_K) has_q4k = true;
        if (t.type == ggnpu::GgmlType::Q8_0) has_q80 = true;
    }
    ASSERT_TRUE(has_q4k, "model contains Q4_K tensors");
    ASSERT_TRUE(has_q80, "model contains Q8_0 tensors");

    // Verify KV pairs contain tokenizer data
    auto it = loader.kv_pairs().find("tokenizer.ggml.model");
    ASSERT_TRUE(it != loader.kv_pairs().end(), "tokenizer.ggml.model key exists");

    // Verify general.alignment
    uint64_t align = loader.general_alignment();
    ASSERT_TRUE(align > 0, "general alignment > 0");
}

void test_kv_pair_access(const std::string& model_path) {
    std::cout << "  test_kv_pair_access\n";
    ggnpu::GgufLoader loader;
    bool result = loader.load(model_path);
    ASSERT_TRUE(result, "load for KV access");

    // Test get_string
    std::string arch = loader.architecture();
    ASSERT_TRUE(!arch.empty(), "architecture string accessible");

    // Test get_int
    int64_t ctx = static_cast<int64_t>(loader.context_length());
    ASSERT_TRUE(ctx > 0, "context length accessible as int");

    // Test get_float (RMS epsilon)
    double eps = loader.attention_layer_norm_rms_epsilon();
    ASSERT_NEAR(eps, 1e-5, 1e-8, "RMS epsilon accessible as float");

    // Verify tensor data offset matches expectation
    uint64_t tdo = loader.tensor_data_offset();
    ASSERT_TRUE(tdo > 0, "tensor data offset is positive");
}

void test_tensor_info(const std::string& model_path) {
    std::cout << "  test_tensor_info\n";
    ggnpu::GgufLoader loader;
    bool result = loader.load(model_path);
    ASSERT_TRUE(result, "load for tensor info");

    const auto& tensors = loader.tensors();

    // Check that embedding tensor exists (token embedding)
    bool has_embd = false;
    for (const auto& t : tensors) {
        if (t.name.find("token_embd") != std::string::npos) {
            has_embd = true;
            ASSERT_EQ(t.n_dims, 2, "embedding tensor has 2 dims");
            // First dim should be vocab size (~128256 for Llama 3.2 1B)
            ASSERT_TRUE(t.dims[0] > 30000, "vocab size > 30000");
            // Second dim should match embedding length
            ASSERT_EQ(t.dims[1], loader.embedding_length(), "embedding dim matches model hparams");
            break;
        }
    }
    ASSERT_TRUE(has_embd, "token_embd tensor exists");

    // Check that attention weight tensors exist
    bool has_attn_q = false;
    bool has_ffn_gate = false;
    for (const auto& t : tensors) {
        if (t.name.find("attn_q") != std::string::npos) has_attn_q = true;
        if (t.name.find("ffn_gate") != std::string::npos) has_ffn_gate = true;
    }
    ASSERT_TRUE(has_attn_q, "attn_q tensors exist");
    ASSERT_TRUE(has_ffn_gate, "ffn_gate tensors exist");

    // Verify all tensors have positive data_size
    for (const auto& t : tensors) {
        ASSERT_TRUE(t.data_size > 0, "tensor " + t.name + " has positive data_size");
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== GGUF I/O Tests ===\n";

    // Test 1: Nonexistent file (always runs)
    test_load_nonexistent_file();

    if (argc < 2) {
        std::cout << "\nSkipping model-dependent tests (no model path provided)\n";
        std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed\n";
        return 0;
    }

    std::string model_path = argv[1];

    // Check if model file exists
    FILE* f = std::fopen(model_path.c_str(), "rb");
    if (!f) {
        std::cout << "\nModel file not found: " << model_path << "\n";
        std::cout << "Skipping model-dependent tests\n";
        std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed\n";
        return 77; // Skip code (like CTest convention)
    }
    std::fclose(f);

    test_load_real_model(model_path);
    test_kv_pair_access(model_path);
    test_tensor_info(model_path);

    std::cout << "\nResults: " << tests_passed << " passed, " << tests_failed << " failed\n";
    return tests_failed > 0 ? 1 : 0;
}
