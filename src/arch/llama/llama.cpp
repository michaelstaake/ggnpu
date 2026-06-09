#include "model.h"
#include "tensor.h"
#include <map>
#include <algorithm>
#include <iostream>

namespace ggnpu {

bool Model::load(const std::string& path) {
    gguf_ = std::make_unique<GgufLoader>();
    if (!gguf_->load(path)) {
        return false;
    }

    if (!parse_hparams()) return false;
    if (!build_tensor_map()) return false;
    if (!prepare_tensors()) return false;

    loaded_ = true;
    return true;
}

void Model::unload() {
    if (gguf_) gguf_->unload();
    tensors_.clear();
    tensor_map_.clear();
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

    // Build tensor map from GGUF tensor list
    const auto& gguf_tensors = gguf_->tensors();
    for (int i = 0; i < static_cast<int>(gguf_tensors.size()); i++) {
        const auto& t = gguf_tensors[i];
        tensor_map_[t.name] = i;
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

Model::Model() {}
Model::~Model() { unload(); }

} // namespace ggnpu
