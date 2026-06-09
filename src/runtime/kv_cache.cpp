#include <vector>
#include <cstdint>
#include <cstring>

namespace ggnpu {

// KV Cache management for attention
// Stores key-value pairs for each layer and position

class KVCache {
public:
    KVCache(uint64_t n_layers, uint64_t n_ctx, uint64_t n_head_kv, uint64_t head_dim)
        : n_layers_(n_layers), n_ctx_(n_ctx), n_head_kv_(n_head_kv), head_dim_(head_dim) {
        // Allocate flat buffer: [n_layers][n_ctx][n_head_kv][head_dim]
        size_t total = n_layers * n_ctx * n_head_kv * head_dim * sizeof(float);
        buffer_.resize(total / sizeof(float));
    }

    void reset() {
        std::memset(buffer_.data(), 0, buffer_.size() * sizeof(float));
        current_pos_ = 0;
    }

    float* keys(uint64_t layer, uint64_t pos) {
        if (pos >= n_ctx_) return nullptr;
        return buffer_.data() + layer * n_ctx_ * n_head_kv_ * head_dim_
             + pos * n_head_kv_ * head_dim_;
    }

    float* values(uint64_t layer, uint64_t pos) {
        if (pos >= n_ctx_) return nullptr;
        return buffer_.data() + layer * n_ctx_ * n_head_kv_ * head_dim_
             + n_ctx_ * n_head_kv_ * head_dim_
             + pos * n_head_kv_ * head_dim_;
    }

    const float* keys(uint64_t layer, uint64_t pos) const {
        if (pos >= n_ctx_) return nullptr;
        return buffer_.data() + layer * n_ctx_ * n_head_kv_ * head_dim_
             + pos * n_head_kv_ * head_dim_;
    }

    const float* values(uint64_t layer, uint64_t pos) const {
        if (pos >= n_ctx_) return nullptr;
        return buffer_.data() + layer * n_ctx_ * n_head_kv_ * head_dim_
             + n_ctx_ * n_head_kv_ * head_dim_
             + pos * n_head_kv_ * head_dim_;
    }

    uint64_t current_position() const { return current_pos_; }
    void increment_position(uint64_t n = 1) { current_pos_ += n; }
    uint64_t capacity() const { return n_ctx_; }

private:
    uint64_t n_layers_;
    uint64_t n_ctx_;
    uint64_t n_head_kv_;
    uint64_t head_dim_;
    uint64_t current_pos_ = 0;
    std::vector<float> buffer_;
};

} // namespace ggnpu
