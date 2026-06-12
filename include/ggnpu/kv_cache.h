#ifndef GGNPU_KV_CACHE_H
#define GGNPU_KV_CACHE_H

#include <vector>
#include <cstdint>

namespace ggnpu {

class KVCache {
public:
    KVCache();
    KVCache(uint64_t n_layers, uint64_t n_ctx, uint64_t n_head_kv, uint64_t head_dim);

    void resize(uint64_t n_layers, uint64_t n_ctx, uint64_t n_head_kv, uint64_t head_dim);
    void reset();

    float* key_buffer(uint64_t layer, uint64_t pos);
    float* value_buffer(uint64_t layer, uint64_t pos);
    const float* key_buffer(uint64_t layer, uint64_t pos) const;
    const float* value_buffer(uint64_t layer, uint64_t pos) const;

    void update(uint64_t layer, uint64_t pos, const float* keys_ptr, const float* values_ptr, uint64_t len);
    void update_slab(uint64_t layer, uint64_t start_pos, uint64_t end_pos,
                     const float* keys_ptr, const float* values_ptr,
                     uint64_t head_kv, uint64_t dim);

    uint64_t current_position() const;
    void set_position(uint64_t pos);
    void increment_position(uint64_t n = 1);
    uint64_t capacity() const;
    uint64_t n_ctx() const;
    uint64_t n_layers() const;
    uint64_t n_head_kv() const;
    uint64_t head_dim() const;

private:
    uint64_t n_layers_;
    uint64_t n_ctx_;
    uint64_t n_head_kv_;
    uint64_t head_dim_;
    uint64_t current_pos_;
    std::vector<float> keys_;
    std::vector<float> values_;
};

} // namespace ggnpu

#endif // GGNPU_KV_CACHE_H
