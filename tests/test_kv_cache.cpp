#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include "ggnpu/kv_cache.h"

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

void test_default_constructor() {
    std::cout << "  test_default_constructor\n";
    ggnpu::KVCache cache;
    ASSERT_EQ(cache.n_layers(), 0, "default n_layers");
    ASSERT_EQ(cache.n_ctx(), 0, "default n_ctx");
    ASSERT_EQ(cache.n_head_kv(), 0, "default n_head_kv");
    ASSERT_EQ(cache.head_dim(), 0, "default head_dim");
    ASSERT_EQ(cache.current_position(), 0, "default current_position");
    ASSERT_EQ(cache.capacity(), 0, "default capacity");
}

void test_construction() {
    std::cout << "  test_construction\n";
    ggnpu::KVCache cache(2, 128, 4, 64);
    ASSERT_EQ(cache.n_layers(), 2ULL, "n_layers");
    ASSERT_EQ(cache.n_ctx(), 128ULL, "n_ctx");
    ASSERT_EQ(cache.n_head_kv(), 4ULL, "n_head_kv");
    ASSERT_EQ(cache.head_dim(), 64ULL, "head_dim");
    ASSERT_EQ(cache.current_position(), 0ULL, "initial position");
    // capacity() returns n_ctx_ (context window size)
    ASSERT_EQ(cache.capacity(), 128ULL, "capacity equals n_ctx");
}

void test_resize() {
    std::cout << "  test_resize\n";
    ggnpu::KVCache cache;
    cache.resize(4, 256, 8, 128);
    ASSERT_EQ(cache.n_layers(), 4ULL, "resized n_layers");
    ASSERT_EQ(cache.n_ctx(), 256ULL, "resized n_ctx");
    ASSERT_EQ(cache.current_position(), 0ULL, "position reset on resize");
}

void test_reset() {
    std::cout << "  test_reset\n";
    ggnpu::KVCache cache(1, 64, 2, 32);
    // Write some data
    std::vector<float> keys(32, 1.0f);
    std::vector<float> values(32, 2.0f);
    cache.update(0, 0, keys.data(), values.data(), 32);
    ASSERT_NEAR(cache.key_buffer(0, 0)[0], 1.0f, 1e-6, "key written");
    ASSERT_NEAR(cache.value_buffer(0, 0)[0], 2.0f, 1e-6, "value written");
    // Reset
    cache.reset();
    ASSERT_EQ(cache.current_position(), 0, "position reset");
    ASSERT_NEAR(cache.key_buffer(0, 0)[0], 0.0f, 1e-6, "key zeroed after reset");
    ASSERT_NEAR(cache.value_buffer(0, 0)[0], 0.0f, 1e-6, "value zeroed after reset");
}

void test_update_and_read() {
    std::cout << "  test_update_and_read\n";
    ggnpu::KVCache cache(1, 32, 2, 8);
    // Write keys and values at position 0
    std::vector<float> keys(8, 1.5f);
    std::vector<float> values(8, 2.5f);
    cache.update(0, 0, keys.data(), values.data(), 8);
    // Read back
    const float* k = cache.key_buffer(0, 0);
    const float* v = cache.value_buffer(0, 0);
    ASSERT_TRUE(k != nullptr, "key buffer not null");
    ASSERT_TRUE(v != nullptr, "value buffer not null");
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(k[i], 1.5f, 1e-6, "key value matches");
        ASSERT_NEAR(v[i], 2.5f, 1e-6, "value matches");
    }
}

void test_update_slab() {
    std::cout << "  test_update_slab\n";
    ggnpu::KVCache cache(1, 32, 2, 4);
    // Write a slab of 3 positions (5, 6, 7)
    // Each position has head_kv * dim = 8 floats
    // Position i in the slab gets keys_ptr[i * 8 .. i*8+7]
    std::vector<float> keys(3 * 2 * 4); // 3 positions, head_kv=2, dim=4
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 8; j++) {
            keys[i * 8 + j] = static_cast<float>(i * 10 + j);
        }
    }
    std::vector<float> values(3 * 2 * 4, 99.0f);
    cache.update_slab(0, 5, 8, keys.data(), values.data(), 2, 4);
    // Position 5 (i=0 in slab): gets keys[0..7] = {0,1,2,3,4,5,6,7}
    const float* k5 = cache.key_buffer(0, 5);
    ASSERT_TRUE(k5 != nullptr, "key buffer at pos 5 not null");
    ASSERT_NEAR(k5[0], 0.0f, 1e-6, "slab key[0] at pos 5");
    // Position 6 (i=1 in slab): gets keys[8..15] = {10,11,12,13,14,15,16,17}
    const float* k6 = cache.key_buffer(0, 6);
    ASSERT_TRUE(k6 != nullptr, "key buffer at pos 6 not null");
    ASSERT_NEAR(k6[0], 10.0f, 1e-6, "slab key[0] at pos 6");
    ASSERT_NEAR(k6[7], 17.0f, 1e-6, "slab key[7] at pos 6");
}

void test_position_tracking() {
    std::cout << "  test_position_tracking\n";
    ggnpu::KVCache cache(1, 32, 2, 8);
    ASSERT_EQ(cache.current_position(), 0, "initial position");
    cache.increment_position();
    ASSERT_EQ(cache.current_position(), 1, "after increment");
    cache.increment_position(5);
    ASSERT_EQ(cache.current_position(), 6, "after increment by 5");
    cache.set_position(10);
    ASSERT_EQ(cache.current_position(), 10, "after set_position");
}

void test_bounds_checking() {
    std::cout << "  test_bounds_checking\n";
    ggnpu::KVCache cache(2, 32, 4, 8);
    // Valid access
    ASSERT_TRUE(cache.key_buffer(1, 31) != nullptr, "valid access at layer 1 pos 31");
    // Out of bounds
    ASSERT_TRUE(cache.key_buffer(2, 0) == nullptr, "out of bounds layer");
    ASSERT_TRUE(cache.key_buffer(0, 32) == nullptr, "out of bounds position");
    ASSERT_TRUE(cache.value_buffer(5, 0) == nullptr, "out of bounds layer (value)");
}

void test_multi_layer_access() {
    std::cout << "  test_multi_layer_access\n";
    ggnpu::KVCache cache(3, 16, 2, 4);
    // Write different values to each layer
    for (int layer = 0; layer < 3; layer++) {
        std::vector<float> keys(8, static_cast<float>(layer + 1));
        std::vector<float> values(8, static_cast<float>(layer + 10));
        cache.update(layer, 0, keys.data(), values.data(), 8);
    }
    // Read back and verify each layer has distinct values
    for (int layer = 0; layer < 3; layer++) {
        const float* k = cache.key_buffer(layer, 0);
        const float* v = cache.value_buffer(layer, 0);
        ASSERT_NEAR(k[0], static_cast<float>(layer + 1), 1e-6, "layer key value");
        ASSERT_NEAR(v[0], static_cast<float>(layer + 10), 1e-6, "layer value");
    }
}

void test_const_correctness() {
    std::cout << "  test_const_correctness\n";
    ggnpu::KVCache cache(1, 32, 2, 8);
    std::vector<float> keys(8, 1.0f);
    std::vector<float> values(8, 2.0f);
    cache.update(0, 0, keys.data(), values.data(), 8);
    // Use const reference
    const ggnpu::KVCache& ccache = cache;
    const float* k = ccache.key_buffer(0, 0);
    const float* v = ccache.value_buffer(0, 0);
    ASSERT_TRUE(k != nullptr, "const key buffer not null");
    ASSERT_TRUE(v != nullptr, "const value buffer not null");
    ASSERT_NEAR(k[0], 1.0f, 1e-6, "const key value");
}

int main() {
    std::cout << "=== KV Cache Tests ===\n";
    test_default_constructor();
    test_construction();
    test_resize();
    test_reset();
    test_update_and_read();
    test_update_slab();
    test_position_tracking();
    test_bounds_checking();
    test_multi_layer_access();
    test_const_correctness();
    std::cout << "\nResults: " << tests_passed << " passed, " << tests_failed << " failed\n";
    return tests_failed > 0 ? 1 : 0;
}
