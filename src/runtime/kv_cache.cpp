#include "kv_cache.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace ggnpu {

KVCache::KVCache()
    : n_layers_(0), n_ctx_(0), n_head_kv_(0), head_dim_(0), current_pos_(0) {
}

KVCache::KVCache(uint64_t n_layers, uint64_t n_ctx, uint64_t n_head_kv, uint64_t head_dim)
    : n_layers_(n_layers), n_ctx_(n_ctx), n_head_kv_(n_head_kv), head_dim_(head_dim), current_pos_(0) {
    resize(n_layers, n_ctx, n_head_kv, head_dim);
}

void KVCache::resize(uint64_t n_layers, uint64_t n_ctx, uint64_t n_head_kv, uint64_t head_dim) {
    n_layers_ = n_layers;
    n_ctx_ = n_ctx;
    n_head_kv_ = n_head_kv;
    head_dim_ = head_dim;
    current_pos_ = 0;

    // K and V live in separate vectors (keys_/values_), so each holds exactly
    // one cache's worth — the previous `2 *` here over-allocated both by 2x.
    size_t total = n_layers * n_ctx * n_head_kv * head_dim;
    keys_.resize(total, 0.0f);
    values_.resize(total, 0.0f);
}

void KVCache::reset() {
    std::fill(keys_.begin(), keys_.end(), 0.0f);
    std::fill(values_.begin(), values_.end(), 0.0f);
    current_pos_ = 0;
}

float* KVCache::key_buffer(uint64_t layer, uint64_t pos) {
    if (pos >= n_ctx_ || layer >= n_layers_) return nullptr;
    return keys_.data() + layer * n_ctx_ * n_head_kv_ * head_dim_
         + pos * n_head_kv_ * head_dim_;
}

float* KVCache::value_buffer(uint64_t layer, uint64_t pos) {
    if (pos >= n_ctx_ || layer >= n_layers_) return nullptr;
    return values_.data() + layer * n_ctx_ * n_head_kv_ * head_dim_
         + pos * n_head_kv_ * head_dim_;
}

const float* KVCache::key_buffer(uint64_t layer, uint64_t pos) const {
    if (pos >= n_ctx_ || layer >= n_layers_) return nullptr;
    return keys_.data() + layer * n_ctx_ * n_head_kv_ * head_dim_
         + pos * n_head_kv_ * head_dim_;
}

const float* KVCache::value_buffer(uint64_t layer, uint64_t pos) const {
    if (pos >= n_ctx_ || layer >= n_layers_) return nullptr;
    return values_.data() + layer * n_ctx_ * n_head_kv_ * head_dim_
         + pos * n_head_kv_ * head_dim_;
}

void KVCache::update(uint64_t layer, uint64_t pos, const float* keys_ptr, const float* values_ptr, uint64_t len) {
    if (pos >= n_ctx_ || layer >= n_layers_) return;

    float* k_out = key_buffer(layer, pos);
    float* v_out = value_buffer(layer, pos);

    if (k_out && keys_ptr) {
        std::memcpy(k_out, keys_ptr, len * sizeof(float));
    }
    if (v_out && values_ptr) {
        std::memcpy(v_out, values_ptr, len * sizeof(float));
    }
}

void KVCache::update_slab(uint64_t layer, uint64_t start_pos, uint64_t end_pos,
                          const float* keys_ptr, const float* values_ptr, uint64_t head_kv, uint64_t dim) {
    if (start_pos >= end_pos || start_pos >= n_ctx_ || layer >= n_layers_) return;

    uint64_t count = std::min(end_pos - start_pos, n_ctx_ - start_pos);

    if (keys_ptr) {
        float* k_out = key_buffer(layer, start_pos);
        if (k_out) {
            for (uint64_t i = 0; i < count; i++) {
                std::memcpy(k_out + i * head_kv * dim,
                           keys_ptr + i * head_kv * dim,
                           head_kv * dim * sizeof(float));
            }
        }
    }
   if (values_ptr) {
        float* v_out = value_buffer(layer, start_pos);
        if (v_out) {
            for (uint64_t i = 0; i < count; i++) {
                std::memcpy(v_out + i * head_kv * dim,
                            values_ptr + i * head_kv * dim,
                            head_kv * dim * sizeof(float));
            }
        }
    }

    set_position(end_pos);
}

uint64_t KVCache::current_position() const { return current_pos_; }
void KVCache::set_position(uint64_t pos) { current_pos_ = std::min(pos, n_ctx_); }
void KVCache::increment_position(uint64_t n) { current_pos_ = std::min(current_pos_ + n, n_ctx_); }
uint64_t KVCache::capacity() const { return n_ctx_; }
uint64_t KVCache::n_ctx() const { return n_ctx_; }
uint64_t KVCache::n_layers() const { return n_layers_; }
uint64_t KVCache::n_head_kv() const { return n_head_kv_; }
uint64_t KVCache::head_dim() const { return head_dim_; }

} // namespace ggnpu
