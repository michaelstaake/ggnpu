#include "model.h"
#include <map>
#include <string>

namespace ggnpu {

// Llama tensor name mapping
// Maps tensor names to their roles in the compute graph

namespace {

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

} // namespace ggnpu
