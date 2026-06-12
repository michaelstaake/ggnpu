#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include "ggnpu/bf16.h"

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

void test_f32_to_bf16_zero() {
    std::cout << "  test_f32_to_bf16_zero\n";
    uint16_t b = ggnpu::f32_to_bf16(0.0f);
    ASSERT_EQ(b, 0x0000, "f32 zero -> bf16");
}

void test_f32_to_bf16_one() {
    std::cout << "  test_f32_to_bf16_one\n";
    uint16_t b = ggnpu::f32_to_bf16(1.0f);
    ASSERT_EQ(b, 0x3F80, "f32 one -> bf16");
}

void test_f32_to_bf16_negative_one() {
    std::cout << "  test_f32_to_bf16_negative_one\n";
    uint16_t b = ggnpu::f32_to_bf16(-1.0f);
    ASSERT_EQ(b, 0xBF80, "f32 -1 -> bf16");
}

void test_f32_to_bf16_rounding() {
    std::cout << "  test_f32_to_bf16_rounding\n";
    // BF16 has 7 mantissa bits; values beyond that should round
    float f = 1.0f + 1.0f / 256.0f;  // Just above 1.0, within BF16 precision
    uint16_t b = ggnpu::f32_to_bf16(f);
    float back = ggnpu::bf16_to_f32(b);
    ASSERT_NEAR(back, f, 0.001f, "rounding near 1.0");
}

void test_bf16_to_f32_zero() {
    std::cout << "  test_bf16_to_f32_zero\n";
    float f = ggnpu::bf16_to_f32(0x0000);
    ASSERT_EQ(f, 0.0f, "bf16 zero -> f32");
}

void test_bf16_to_f32_one() {
    std::cout << "  test_bf16_to_f32_one\n";
    float f = ggnpu::bf16_to_f32(0x3F80);
    ASSERT_EQ(f, 1.0f, "bf16 one -> f32");
}

void test_roundtrip_normal_values() {
    std::cout << "  test_roundtrip_normal_values\n";
    // Test a range of normal values
    std::vector<float> inputs = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        3.14159f, -3.14159f,
        128.0f, -128.0f,
        0.0078125f,  // 2^-7, near BF16 precision limit
        65504.0f,    // Near max BF16
    };
    for (float f : inputs) {
        float result = ggnpu::bf16_roundtrip_f32(f);
        // BF16 has ~7 digits of precision; check relative error
        if (std::fabs(f) > 1e-6f) {
            float rel_err = std::fabs(f - result) / std::fabs(f);
            ASSERT_TRUE(rel_err < 0.01f, "relative error < 1% for roundtrip of " + std::to_string(f));
        } else {
            ASSERT_NEAR(result, f, 1e-6f, "absolute error near zero for roundtrip");
        }
    }
}

void test_roundtrip_edge_cases() {
    std::cout << "  test_roundtrip_edge_cases\n";
    // Zero
    ASSERT_EQ(ggnpu::bf16_roundtrip_f32(0.0f), 0.0f, "zero roundtrip");
    ASSERT_EQ(ggnpu::bf16_roundtrip_f32(-0.0f), -0.0f, "-zero roundtrip");

    // Very small value (subnormal in BF16)
    float small = 1e-5f;
    float result = ggnpu::bf16_roundtrip_f32(small);
    ASSERT_NEAR(result, small, 1e-7f, "small value roundtrip");

    // Large value
    float large = 1000.0f;
    result = ggnpu::bf16_roundtrip_f32(large);
    float rel_err = std::fabs(large - result) / std::fabs(large);
    ASSERT_TRUE(rel_err < 0.01f, "large value relative error < 1%");
}

void test_convert_f32_to_bf16_vector() {
    std::cout << "  test_convert_f32_to_bf16_vector\n";
    std::vector<float> input = {0.0f, 1.0f, -1.0f, 0.5f, 3.14f};
    auto bf16_buf = ggnpu::convert_f32_to_bf16(input.data(), input.size());

    // Should be 2 bytes per element
    ASSERT_EQ(bf16_buf.size(), input.size() * 2, "bf16 buffer size");

    // Verify first element (0.0 -> 0x0000)
    uint16_t b0 = (static_cast<uint16_t>(bf16_buf[1]) << 8) | bf16_buf[0];
    ASSERT_EQ(b0, 0x0000, "first element is zero");

    // Verify second element (1.0 -> 0x3F80)
    uint16_t b1 = (static_cast<uint16_t>(bf16_buf[3]) << 8) | bf16_buf[2];
    ASSERT_EQ(b1, 0x3F80, "second element is one");
}

void test_convert_bf16_to_f32_vector() {
    std::cout << "  test_convert_bf16_to_f32_vector\n";
    // Create bf16 data: 0.0, 1.0, -1.0
    std::vector<uint8_t> bf16_data = {
        0x00, 0x00,  // 0.0
        0x80, 0x3F,  // 1.0 (little-endian)
        0x80, 0xBF,  // -1.0 (little-endian)
    };

    auto f32_buf = ggnpu::convert_bf16_to_f32(bf16_data.data(), 3);
    ASSERT_EQ(f32_buf.size(), 3, "f32 buffer size");
    ASSERT_NEAR(f32_buf[0], 0.0f, 1e-6f, "zero");
    ASSERT_NEAR(f32_buf[1], 1.0f, 1e-6f, "one");
    ASSERT_NEAR(f32_buf[2], -1.0f, 1e-6f, "negative one");
}

void test_bf16_roundtrip_vector() {
    std::cout << "  test_bf16_roundtrip_vector\n";
    std::vector<float> input = {0.0f, 1.0f, -1.0f, 0.5f, 3.14f, -2.718f};
    std::vector<float> output(input.size());
    ggnpu::bf16_roundtrip_vector(input.data(), output.data(), static_cast<int>(input.size()));

    for (size_t i = 0; i < input.size(); i++) {
        float f = input[i];
        float r = output[i];
        if (std::fabs(f) > 1e-6f) {
            float rel_err = std::fabs(f - r) / std::fabs(f);
            ASSERT_TRUE(rel_err < 0.01f, "relative error < 1% for vector element " + std::to_string(i));
        } else {
            ASSERT_NEAR(r, f, 1e-6f, "absolute error near zero for vector element " + std::to_string(i));
        }
    }
}

void test_f32_to_bf16_preserves_sign() {
    std::cout << "  test_f32_to_bf16_preserves_sign\n";
    uint16_t pos = ggnpu::f32_to_bf16(0.5f);
    uint16_t neg = ggnpu::f32_to_bf16(-0.5f);
    // BF16 sign bit is the MSB (bit 15)
    ASSERT_TRUE((pos & 0x8000) == 0, "positive value has no sign bit");
    ASSERT_TRUE((neg & 0x8000) != 0, "negative value has sign bit");
}

int main() {
    std::cout << "=== BF16 Conversion Tests ===\n";
    test_f32_to_bf16_zero();
    test_f32_to_bf16_one();
    test_f32_to_bf16_negative_one();
    test_f32_to_bf16_rounding();
    test_bf16_to_f32_zero();
    test_bf16_to_f32_one();
    test_roundtrip_normal_values();
    test_roundtrip_edge_cases();
    test_convert_f32_to_bf16_vector();
    test_convert_bf16_to_f32_vector();
    test_bf16_roundtrip_vector();
    test_f32_to_bf16_preserves_sign();
    std::cout << "\nResults: " << tests_passed << " passed, " << tests_failed << " failed\n";
    return tests_failed > 0 ? 1 : 0;
}
