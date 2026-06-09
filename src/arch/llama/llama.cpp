#include "model.h"
#include "tensor.h"
#include <map>
#include <algorithm>
#include <iostream>

namespace ggnpu {

namespace {

TensorRole parse_tensor_role(const std::string& name) {
    if (name.find("token_embd") != std::string::npos) return TensorRole::EMBEDDING;
    if (name.find("attn_q") != std::string::npos) return TensorRole::ATTN_Q;
    if (name.find("attn_k") != std::string::npos) return TensorRole::ATTN_K;
    if (name.find("attn_v") != std::string::npos) return TensorRole::ATTN_V;
    if (name.find("attn_output") != std::string::npos) return TensorRole::ATTN_OUTPUT;
    if (name.find("attn_norm") != std::string::npos) return TensorRole::ATTN_NORM;
    if (name.find("ffn_gate") != std::string::npos) return TensorRole::FFN_GATE;
    if (name.find("ffn_up") != std::string::npos) return TensorRole::FFN_UP;
    if (name.find("ffn_down") != std::string::npos) return TensorRole::FFN_DOWN;
    if (name.find("ffn_norm") != std::string::npos) return TensorRole::FFN_NORM;
    if (name.find("output_norm") != std::string::npos) return TensorRole::OUTPUT_NORM;
    if (name.find("output") != std::string::npos) return TensorRole::OUTPUT;
    return TensorRole::UNKNOWN;
}

} // namespace

bool Model::load(const std::string& path) {
    gguf_ = std::make_unique<GgufLoader>();
    if (!gguf_->load(path)) {
        return false;
    }

    if (!parse_hparams()) return false;
    if (!build_tensor_map()) return false;
    if (!prepare_tensors()) return false;
    if (!init_kv_cache(0)) return false;

    loaded_ = true;
    return true;
}

void Model::unload() {
    if (gguf_) gguf_->unload();
    tensors_.clear();
    tensor_map_.clear();
    tensor_roles_.clear();
    kv_cache_.reset();
    loaded_ = false;
}

bool Model::parse_hparams() {
    hparams_.context_length = gguf_->context_length();
    hparams_.embedding_length = gguf_->embedding_length();
    hparams_.block_count = gguf_->block_count();
    hparams_.feed_forward_length = gguf_->feed_forward_length();
    hparams_.attention_head_count = gguf_->attention_head_count();
    hparams_.attention_head_count_kv = gguf_->attention_head_count_kv();
    hparams_.rope_dimension_count = gguf_->rope_dimension_count();
    hparams_.rope_freq_scale = static_cast<float>(gguf_->rope_freq_scale());
    hparams_.rope_freq_base = gguf_->rope_freq_base();
    hparams_.tensor_data_offset = gguf_->tensor_data_offset();

    // Get vocab size from GGUF if available
    auto vocab_it = gguf_->kv_pairs().find("tokenizer.ggml.tokens.length");
    if (vocab_it != gguf_->kv_pairs().end()) {
        hparams_.vocab_size = static_cast<uint64_t>(vocab_it->second.int_value);
    }

    // Default values for missing params
    if (hparams_.attention_layer_norm_rms_epsilon == 0) {
        hparams_.attention_layer_norm_rms_epsilon = 5e-5;
    }
    if (hparams_.rope_freq_base == 0) {
        hparams_.rope_freq_base = 10000;
    }

    return true;
}

bool Model::build_tensor_map() {
    tensor_map_.clear();
    tensor_roles_.clear();

    const auto& gguf_tensors = gguf_->tensors();
    for (int i = 0; i < static_cast<int>(gguf_tensors.size()); i++) {
        const auto& t = gguf_tensors[i];
        tensor_map_[t.name] = i;
        tensor_roles_[t.name] = parse_tensor_role(t.name);
    }

    return !tensor_map_.empty();
}

bool Model::prepare_tensors() {
    tensors_.clear();

    const auto& gguf_tensors = gguf_->tensors();
    const uint8_t* data_base = gguf_->tensor_data().data();

    tensors_.reserve(gguf_tensors.size());

    for (const auto& t : gguf_tensors) {
        TensorView tv;
        tv.name = t.name;
        tv.dims = t.dims;
        tv.type = t.type;
        tv.n_dims = t.n_dims;
        tv.data_offset = t.data_offset;

        // Calculate the actual pointer into the mapped file
        if (data_base && t.data_offset < gguf_->tensor_data().size()) {
            tv.data = data_base + t.data_offset;
        }

        tensors_.push_back(std::move(tv));
    }

    return !tensors_.empty();
}

void Model::set_backend(std::shared_ptr<Backend> backend) {
    backend_ = backend;
}

bool Model::init_kv_cache(int64_t ctx_override) {
    uint64_t ctx = hparams_.context_length;
    if (ctx_override > 0) ctx = static_cast<uint64_t>(ctx_override);
    if (ctx == 0) ctx = 2048;

    uint64_t head_dim = hparams_.rope_dimension_count;
    if (head_dim == 0) {
        head_dim = hparams_.embedding_length / hparams_.attention_head_count;
    }

    kv_cache_ = std::make_unique<KVCache>(
        hparams_.block_count,
        ctx,
        hparams_.attention_head_count_kv,
        head_dim
    );

    return static_cast<bool>(kv_cache_);
}

TensorRole Model::get_tensor_role(const std::string& name) const {
    auto it = tensor_roles_.find(name);
    if (it != tensor_roles_.end()) return it->second;
    return TensorRole::UNKNOWN;
}

void Model::set_context_length(uint64_t ctx) {
    hparams_.context_length = ctx;
    if (loaded_) {
        init_kv_cache(0);
    }
}

Model::Model() {}
Model::~Model() { unload(); }

} // namespace ggnpu
