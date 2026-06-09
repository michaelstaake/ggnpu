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
    ASSERT_EQ(sizeof(block.qs), 8, "Q4_0Block.qs is 8 bytes");
    ASSERT_TRUE(sizeof(block) >= 10, "Q4_0Block size >= 10 bytes");

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

    uint8_t block_data[16];
    ggnpu::Q4_0Block* block = reinterpret_cast<ggnpu::Q4_0Block*>(block_data);
    block->d = 1;
    block->qs[0] = 0x11;
    memset(block->qs + 1, 0, 7);

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

    uint8_t q4_block[16];
    ggnpu::Q4_0Block* q4b = reinterpret_cast<ggnpu::Q4_0Block*>(q4_block);
    q4b->d = 1;
    q4b->qs[0] = 0x12;
    memset(q4b->qs + 1, 0, 7);

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

void run_tests() {
    std::cout << "=== Quantization Tests ===\n\n";

    test_q4_0_block_structure();
    test_q8_0_block_structure();
    test_q4_0_to_int8_conversion();
    test_q4_0_decode_basic();
    test_q8_0_decode_basic();
    test_decode_dispatch();
    test_block_size_consistency();

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
