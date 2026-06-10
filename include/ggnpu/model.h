#ifndef GGNPU_MODEL_H
#define GGNPU_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "gguf.h"
#include "tensor.h"
#include "backend.h"
#include "kv_cache.h"

namespace ggnpu {

struct LlamaHParams {
    uint64_t vocab_size = 32000;
    uint64_t context_length = 4096;
    uint64_t embedding_length = 4096;
    uint64_t block_count = 32;
    uint64_t feed_forward_length = 11008;
    uint64_t attention_head_count = 32;
    uint64_t attention_head_count_kv = 8;
    float attention_layer_norm_rms_epsilon = 5e-5f;
    uint64_t rope_dimension_count = 128;
    float rope_freq_scale = 1.0f;
    uint64_t rope_freq_base = 10000;
    uint64_t tensor_data_offset = 0;
};

struct LayerConfig {
    std::vector<TensorView> tensors;
    LlamaHParams hparams;
};

enum class TensorRole {
    UNKNOWN,
    EMBEDDING,
    ATTN_Q,
    ATTN_K,
    ATTN_V,
    ATTN_OUTPUT,
    ATTN_NORM,
    FFN_GATE,
    FFN_UP,
    FFN_DOWN,
    FFN_NORM,
    OUTPUT_NORM,
    OUTPUT,
};

class Model {
public:
    Model();
    ~Model();

    bool load(const std::string& path);
    void unload();

    const GgufLoader& gguf() const { return *gguf_; }
    const LlamaHParams& hparams() const { return hparams_; }
    const std::map<std::string, int>& tensor_map() const { return tensor_map_; }
    const std::vector<TensorView>& tensors() const { return tensors_; }

    std::shared_ptr<Backend> backend() const { return backend_; }
    void set_backend(std::shared_ptr<Backend> backend);

    KVCache& kv_cache() { return *kv_cache_; }
    const KVCache& kv_cache() const { return *kv_cache_; }

    bool is_loaded() const { return loaded_; }
    TensorRole get_tensor_role(const std::string& name) const;
    void set_context_length(uint64_t ctx);

private:
    bool parse_hparams();
    bool build_tensor_map();
    bool prepare_tensors();
    bool init_kv_cache(int64_t ctx_override);

    std::unique_ptr<GgufLoader> gguf_;
    LlamaHParams hparams_;
    std::map<std::string, int> tensor_map_;
    std::map<std::string, TensorRole> tensor_roles_;
    std::vector<TensorView> tensors_;
    std::shared_ptr<Backend> backend_;
    std::unique_ptr<KVCache> kv_cache_;
    bool loaded_ = false;
};

} // namespace ggnpu

#endif // GGNPU_MODEL_H
