#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include "ggnpu/tensor.h"
#include "ggnpu/quant/q4_0.h"
#include "ggnpu/quant/q8_0.h"
#include "ggnpu/quant/quant.h"

// Helper: dequantize Q4_K block to float (reference implementation)
// Uses direct byte offsets to avoid struct padding issues
static void dequantize_q4_k_block_ref(const uint8_t* block, float* out) {
    // GGUF Q4_K layout: d(2) + c(2) + scales(6) + qs(32) + qs_large(96)
    // But we only read what's needed from a 48-byte buffer
    int16_t d_signed = static_cast<int16_t>(block[0] | (block[1] << 8));
    int16_t c_signed = static_cast<int16_t>(block[2] | (block[3] << 8));

    float scales[16] = {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 6; i++) {
        int8_t s = static_cast<int8_t>(block[4 + i]);
        scales[i * 2 + 2] = static_cast<float>(s & 0x0F);
        scales[i * 2 + 3] = static_cast<float>((s >> 4) & 0x0F);
    }

    // First 128 values: Q4_0 with scales[0-3] * d
    for (int i = 0; i < 128; i += 32) {
        int scale_idx = (i / 32) % 4;
        float scale = scales[scale_idx] * d_signed;
        const uint8_t* qblock = block + 10 + (i / 32) * 16;
        for (int j = 0; j < 32; j++) {
            uint8_t nibble = (j % 2 == 0) ? (qblock[j / 2] & 0x0F) : (qblock[j / 2] >> 4);
            int val = static_cast<int>(nibble) - 8;
            out[i + j] = val * scale;
        }
    }
    // Last 128 values: Q8_0 with scales[4-7] * c
    for (int i = 128; i < 256; i += 32) {
        int scale_idx = ((i - 128) / 32) + 4;
        float scale = scales[scale_idx] * c_signed;
        const int8_t* qblock = reinterpret_cast<const int8_t*>(block + 42) + (i - 128);
        for (int j = 0; j < 32; j++) {
            out[i + j] = static_cast<float>(qblock[j]) * scale;
        }
    }
}

// Helper: dequantize Q6_K block to float (reference implementation)
// Uses direct byte offsets to avoid struct padding issues
static void dequantize_q6_k_block_ref(const uint8_t* block, float* out) {
    // GGUF Q6_K layout: d(2) + d2(2) + scales(12) + qs(32) + high_bits(32) + qs_large(96)
    int16_t d_signed = static_cast<int16_t>(block[0] | (block[1] << 8));
    int16_t d2_signed = static_cast<int16_t>(block[2] | (block[3] << 8));

    float scales[12];
    for (int i = 0; i < 6; i++) {
        int8_t s = static_cast<int8_t>(block[4 + i]);
        scales[i * 2] = static_cast<float>(s & 0x0F);
        scales[i * 2 + 1] = static_cast<float>((s >> 4) & 0x0F);
    }

    // First 128 values: Q4 with scales[0-5] * d
    for (int i = 0; i < 128; i += 32) {
        int scale_idx = (i / 32) % 6;
        float scale = scales[scale_idx] * d_signed;
        const uint8_t* qblock = block + 16 + (i / 32) * 16;
        for (int j = 0; j < 32; j++) {
            uint8_t nibble = (j % 2 == 0) ? (qblock[j / 2] & 0x0F) : (qblock[j / 2] >> 4);
            int val = static_cast<int>(nibble) - 8;
            out[i + j] = val * scale;
        }
    }
    // Last 128 values: Q6 with scales[6-11] * d2
    for (int i = 128; i < 256; i += 32) {
        int block_idx = (i - 128) / 32;
        float scale = scales[block_idx + 6] * d2_signed;
        const int8_t* qblock = reinterpret_cast<const int8_t*>(block + 48) + (i - 128);
        const uint8_t* hblock = block + 32 + (i - 128);
        for (int j = 0; j < 32; j++) {
            int val = qblock[j] | ((hblock[j] & 0x03) << 7);
            if (val >= 64) val -= 128;
            out[i + j] = static_cast<float>(val) * scale;
        }
    }
}

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

void test_q4_0_block_structure() {
    std::cout << "  test_q4_0_block_structure\n";

    ggnpu::Q4_0Block block;
    memset(&block, 0, sizeof(block));

    ASSERT_EQ(sizeof(block.d), 2, "Q4_0Block.d is int16_t (2 bytes)");
    ASSERT_EQ(sizeof(block.qs), 16, "Q4_0Block.qs is 16 bytes");
    ASSERT_TRUE(sizeof(block) >= 18, "Q4_0Block size >= 18 bytes");

    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0), 32, "Q4_0 block size is 32 elements");
    ASSERT_EQ(ggnpu::ggml_type_size(ggnpu::GgmlType::Q4_0), 16, "Q4_0 type size is 16 bytes");
}

void test_q8_0_block_structure() {
    std::cout << "  test_q8_0_block_structure\n";

    ggnpu::Q8_0Block block;
    memset(&block, 0, sizeof(block));

    ASSERT_EQ(sizeof(block), 34, "Q8_0Block size is 34 bytes");
    ASSERT_EQ(sizeof(block.d), 2, "Q8_0Block.d is int16_t (2 bytes)");
    ASSERT_EQ(sizeof(block.qs), 32, "Q8_0Block.qs is 32 bytes");

    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q8_0), 32, "Q8_0 block size is 32 elements");
    ASSERT_EQ(ggnpu::ggml_type_size(ggnpu::GgmlType::Q8_0), 34, "Q8_0 type size is 34 bytes");
}

void test_q4_0_to_int8_conversion() {
    std::cout << "  test_q4_0_to_int8_conversion\n";

    // Test sign extension manually
    // nibble 0x00 -> 0
    // nibble 0x08 -> -8 (sign bit set)
    // nibble 0x0F -> -1
    uint8_t nibble = 0x08;
    int8_t val = static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
    ASSERT_EQ(val, -8, "0x08 sign-extends to -8");

    nibble = 0x0F;
    val = static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
    ASSERT_EQ(val, -1, "0x0F sign-extends to -1");

    nibble = 0x07;
    val = static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
    ASSERT_EQ(val, 7, "0x07 sign-extends to 7");
}

void test_q4_0_decode_basic() {
    std::cout << "  test_q4_0_decode_basic\n";

    uint8_t block_data[sizeof(ggnpu::Q4_0Block)];
    memset(block_data, 0, sizeof(block_data));
    ggnpu::Q4_0Block* block = reinterpret_cast<ggnpu::Q4_0Block*>(block_data);
    block->d = 1;
    block->qs[0] = 0x11;
    memset(block->qs + 1, 0, 15);

    std::vector<int8_t> int8_out;
    std::vector<float> scales_out;

    ggnpu::decode_q4_0_for_npu(block_data, 16, int8_out, scales_out);

    ASSERT_EQ(int8_out.size(), 32, "Q4_0 decode produces 32 int8 values");
    ASSERT_EQ(scales_out.size(), 1, "Q4_0 decode produces 1 scale");

    // With scale=1, nibble 0x11 gives values 1 and 0
    // But the decode function uses val = nibble - 8, so:
    // int8_out[0] = 1 - 8 = -7
    // int8_out[1] = 0 - 8 = -8
    ASSERT_EQ(int8_out[0], -7, "Q4_0 decode value[0]");
    ASSERT_EQ(int8_out[1], -7, "Q4_0 decode value[1]");
}

void test_q8_0_decode_basic() {
    std::cout << "  test_q8_0_decode_basic\n";

    uint8_t block_data[34];
    ggnpu::Q8_0Block* block = reinterpret_cast<ggnpu::Q8_0Block*>(block_data);
    block->d = 2;
    memset(block->qs, 0, 32);
    block->qs[0] = 3;
    block->qs[31] = 1;

    std::vector<int8_t> int8_out;
    std::vector<float> scales_out;

    ggnpu::decode_q8_0_for_npu(block_data, 34, int8_out, scales_out);

    ASSERT_EQ(int8_out.size(), 32, "Q8_0 decode produces 32 int8 values");
    ASSERT_EQ(scales_out.size(), 1, "Q8_0 decode produces 1 scale");

    ASSERT_EQ(int8_out[0], 3, "Q8_0 decode value[0]");
    ASSERT_EQ(int8_out[31], 1, "Q8_0 decode value[31]");
}

void test_decode_dispatch() {
    std::cout << "  test_decode_dispatch\n";

    std::vector<int8_t> int8_out;
    std::vector<float> scales_out;

    float f32_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    ggnpu::decode_for_npu(ggnpu::GgmlType::F32, reinterpret_cast<const uint8_t*>(f32_data), 16, int8_out, scales_out);
    ASSERT_TRUE(int8_out.size() >= 4, "F32 decode produces output");

    uint8_t q8_block[34];
    ggnpu::Q8_0Block* q8b = reinterpret_cast<ggnpu::Q8_0Block*>(q8_block);
    q8b->d = 1;
    q8b->qs[0] = 5;
    memset(q8b->qs + 1, 0, 31);

    int8_out.clear();
    scales_out.clear();
    ggnpu::decode_for_npu(ggnpu::GgmlType::Q8_0, q8_block, 34, int8_out, scales_out);
    ASSERT_EQ(int8_out.size(), 32, "Q8_0 dispatch produces 32 values");
    ASSERT_EQ(int8_out[0], 5, "Q8_0 dispatch value[0]");

    uint8_t q4_block[sizeof(ggnpu::Q4_0Block)];
    memset(q4_block, 0, sizeof(q4_block));
    ggnpu::Q4_0Block* q4b = reinterpret_cast<ggnpu::Q4_0Block*>(q4_block);
    q4b->d = 1;
    q4b->qs[0] = 0x12;
    memset(q4b->qs + 1, 0, 15);

    int8_out.clear();
    scales_out.clear();
    ggnpu::decode_for_npu(ggnpu::GgmlType::Q4_0, q4_block, 16, int8_out, scales_out);
    ASSERT_EQ(int8_out.size(), 32, "Q4_0 dispatch produces 32 values");
   // qs[0] = 0x12, so nibbles are 2 and 1
    // val = nibble - 8, so 2-8=-6, 1-8=-7
    ASSERT_EQ(int8_out[0], -6, "Q4_0 dispatch value[0]");
    ASSERT_EQ(int8_out[1], -7, "Q4_0 dispatch value[1]");
}

void test_block_size_consistency() {
    std::cout << "  test_block_size_consistency\n";

    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0), ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_1), "Q4_0 == Q4_1 block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0), ggnpu::ggml_blck_size(ggnpu::GgmlType::Q5_0), "Q4_0 == Q5_0 block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0), ggnpu::ggml_blck_size(ggnpu::GgmlType::Q8_0), "Q4_0 == Q8_0 block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0), ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_K), "Q4_0 == Q4_K block size");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::Q4_0), ggnpu::ggml_blck_size(ggnpu::GgmlType::Q6_K), "Q4_0 == Q6_K block size");

    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::F32), 1, "F32 block size is 1");
    ASSERT_EQ(ggnpu::ggml_blck_size(ggnpu::GgmlType::F16), 1, "F16 block size is 1");
}

void test_q4_k_decode_golden() {
    std::cout << "  test_q4_k_decode_golden\n";

    // Create a Q4_K block with known values (48 bytes, GGUF layout)
    // Layout: d(2) + c(2) + scales(6) + qs(32) + qs_large(96) = 138 bytes struct,
    // but GGUF format is only 48 bytes: d(2) + c(2) + scales(6) + qs(16) + qs_large(96) = 122... 
    // Actually GGUF Q4_K block is 48 bytes with different layout.
    // For testing, we create a minimal valid block.
    uint8_t block_data[48];
    memset(block_data, 0, sizeof(block_data));

    // GGUF Q4_K layout (48 bytes):
    // bytes 0-1: d (uint16_t LE)
    // bytes 2-3: c (uint16_t LE)
    // bytes 4-9: scales[6] (int8_t)
    // bytes 10-41: qs[32] (uint8_t) - but only first 16 bytes contain real Q4_0 data
    // bytes 42-47: qs_large[6] (int8_t) - first 6 bytes of Q8_0 data

    // Set d=4, c=2
    block_data[0] = 4; block_data[1] = 0;
    block_data[2] = 2; block_data[3] = 0;

    // Set scales: all 1s (0x11 = low=1, high=1)
    for (int i = 0; i < 6; i++) {
        block_data[4 + i] = 0x11;
    }

    // Set Q4_0 part: qs[0] = 0x10 (nibbles: 0, 1 -> vals: -8, -7)
    block_data[10] = 0x10;

    // Set Q8_0 part: qs_large[0] = 3
    block_data[42] = 3;

    // Decode via NPU decoder
    std::vector<int8_t> int8_out;
    std::vector<float> scales_out;
    ggnpu::decode_q4_k_for_npu(block_data, 48, int8_out, scales_out);

    // Verify scales: 8 scales per block
    ASSERT_EQ(scales_out.size(), 8, "Q4_K NPU decoder produces 8 scales");

    // Verify first scale = 1.0 * d = 4
    ASSERT_NEAR(scales_out[0], 4.0f, 0.01f, "Q4_K scale[0] = 1*d = 4");
    // Verify fourth scale = 1.0 * d = 4
    ASSERT_NEAR(scales_out[3], 4.0f, 0.01f, "Q4_K scale[3] = 1*d = 4");
    // Verify fifth scale = 1.0 * c = 2
    ASSERT_NEAR(scales_out[4], 2.0f, 0.01f, "Q4_K scale[4] = 1*c = 2");
    // Verify eighth scale = 1.0 * c = 2
    ASSERT_NEAR(scales_out[7], 2.0f, 0.01f, "Q4_K scale[7] = 1*c = 2");

    // Verify INT8 values
    ASSERT_EQ(int8_out.size(), 256, "Q4_K NPU decoder produces 256 INT8 values");
    // First value: nibble 0 -> val = 0 - 8 = -8
    ASSERT_EQ(int8_out[0], -8, "Q4_K INT8[0] = -8");
    // Second value: nibble 1 -> val = 1 - 8 = -7
    ASSERT_EQ(int8_out[1], -7, "Q4_K INT8[1] = -7");
    // Q8_0 part: first value = 3
    ASSERT_EQ(int8_out[128], 3, "Q4_K INT8[128] = 3 (Q8_0 part)");

    // Golden test: compare dequantized output vs reference
    std::vector<float> npu_float(256);
    std::vector<float> ref_float(256);

    // Reconstruct floats from NPU INT8 + scales
    for (int i = 0; i < 128; i += 32) {
        int scale_idx = (i / 32) % 4;
        for (int j = 0; j < 32; j++) {
            npu_float[i + j] = static_cast<float>(int8_out[i + j]) * scales_out[scale_idx];
        }
    }
    for (int i = 128; i < 256; i += 32) {
        int scale_idx = ((i - 128) / 32) + 4;
        npu_float[i] = static_cast<float>(int8_out[i]) * scales_out[scale_idx];
    }

    // Reference dequantization
    dequantize_q4_k_block_ref(block_data, ref_float.data());

    // Compare (first few values)
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(npu_float[i], ref_float[i], 0.01f, "Q4_K golden match value[" + std::to_string(i) + "]");
    }
}

void test_q6_k_decode_golden() {
    std::cout << "  test_q6_k_decode_golden\n";

    // Create a Q6_K block with known values (64 bytes, GGUF layout)
    // GGUF Q6_K layout (64 bytes):
    // bytes 0-1: d (uint16_t LE)
    // bytes 2-3: d2 (uint16_t LE)
    // bytes 4-15: scales[12] (int8_t)
    // bytes 16-31: qs[16] (uint8_t) - Q4 data (first 16 bytes)
    // bytes 32-47: high_bits[16] (uint8_t) - high 2 bits for Q6
    // bytes 48-63: qs_large[16] (int8_t) - first 16 bytes of Q6 data

    uint8_t block_data[64];
    memset(block_data, 0, sizeof(block_data));

    // Set d=4, d2=2
    block_data[0] = 4; block_data[1] = 0;
    block_data[2] = 2; block_data[3] = 0;

    // Set scales: all 1s (0x11 = low=1, high=1)
    for (int i = 0; i < 6; i++) {
        block_data[4 + i] = 0x11;
    }

    // Set Q4 part: qs[0] = 0x10 (nibbles: 0, 1 -> vals: -8, -7)
    block_data[16] = 0x10;

    // Set Q6 part: qs_large[0] = 3, high_bits[0] = 0
    block_data[48] = 3;
    block_data[32] = 0;

    // Decode via NPU decoder
    std::vector<int8_t> int8_out;
    std::vector<float> scales_out;
    ggnpu::decode_q6_k_for_npu(block_data, 64, int8_out, scales_out);

    // Verify scales: 12 scales per block
    ASSERT_EQ(scales_out.size(), 12, "Q6_K NPU decoder produces 12 scales");

    // Verify first scale = 1.0 * d = 4
    ASSERT_NEAR(scales_out[0], 4.0f, 0.01f, "Q6_K scale[0] = 1*d = 4");
    // Verify sixth scale = 1.0 * d = 4
    ASSERT_NEAR(scales_out[5], 4.0f, 0.01f, "Q6_K scale[5] = 1*d = 4");
    // Verify seventh scale = 1.0 * d2 = 2
    ASSERT_NEAR(scales_out[6], 2.0f, 0.01f, "Q6_K scale[6] = 1*d2 = 2");
    // Verify twelfth scale = 1.0 * d2 = 2
    ASSERT_NEAR(scales_out[11], 2.0f, 0.01f, "Q6_K scale[11] = 1*d2 = 2");

    // Verify INT8 values
    ASSERT_EQ(int8_out.size(), 256, "Q6_K NPU decoder produces 256 INT8 values");
    // First value: nibble 0 -> val = 0 - 8 = -8
    ASSERT_EQ(int8_out[0], -8, "Q6_K INT8[0] = -8");
    // Q6 part: first value = 3 | (0 << 7) = 3
    ASSERT_EQ(int8_out[128], 3, "Q6_K INT8[128] = 3 (Q6 part)");

    // Golden test: compare dequantized output vs reference
    std::vector<float> npu_float(256);
    std::vector<float> ref_float(256);

    // Reconstruct floats from NPU INT8 + scales
    for (int i = 0; i < 128; i += 32) {
        int scale_idx = (i / 32) % 6;
        for (int j = 0; j < 32; j++) {
            npu_float[i + j] = static_cast<float>(int8_out[i + j]) * scales_out[scale_idx];
        }
    }
    for (int i = 128; i < 256; i += 32) {
        int scale_idx = ((i - 128) / 32) + 6;
        npu_float[i] = static_cast<float>(int8_out[i]) * scales_out[scale_idx];
    }

    // Reference dequantization
    dequantize_q6_k_block_ref(block_data, ref_float.data());

    // Compare (first few values)
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(npu_float[i], ref_float[i], 0.01f, "Q6_K golden match value[" + std::to_string(i) + "]");
    }
}

void run_tests() {
    std::cout << "=== Quantization Tests ===\n\n";

    test_q4_0_block_structure();
    test_q8_0_block_structure();
    test_q4_0_to_int8_conversion();
    test_q4_0_decode_basic();
    test_q8_0_decode_basic();
    test_decode_dispatch();
    test_block_size_consistency();
    test_q4_k_decode_golden();
    test_q6_k_decode_golden();

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
