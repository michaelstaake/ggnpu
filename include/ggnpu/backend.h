#ifndef GGNPU_BACKEND_H
#define GGNPU_BACKEND_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include "tensor.h"

namespace ggnpu {

enum class Status {
    OK = 0,
    ERROR = 1,
    NOT_FOUND = 2,
    INVALID_PARAM = 3,
    NPU_UNAVAILABLE = 4,
    OUT_OF_MEMORY = 5,
};

struct MulMatParams {
    const void* A;
    const void* B;
    void* C;
    int M;
    int N;
    int K;
    int lda;
    int ldb;
    int ldc;
    int32_t n_batches;
    const void* scales = nullptr;
    int n_weight_scales = 0;  // 1 = single scale; N = per output row (B row)
    GgmlType B_type = GgmlType::F32;
};

struct RmsNormParams {
    const float* input;
    float* output;
    int size;
    float eps;
    const float* weight = nullptr;
};

struct RopeParams {
    float* data;
    int n_dims;
    int64_t offset;
    float freq_scale;
    float freq_base;
    int64_t rope_dims;
    // Optional precomputed tables for the NPU path (each [rope_dims/2] floats).
    // When null the backend derives angles from offset/freq_base/freq_scale.
    const float* cos_table = nullptr;
    const float* sin_table = nullptr;
};

// Batched RoPE: apply rotary embedding to all heads of one token at once.
// data layout is [n_heads][n_dims] interleaved (2i, 2i+1 pairs), rotated in place.
// cos/sin tables are shared across heads (depend only on offset and pair index).
struct RopeBatchedParams {
    float* data;
    int n_heads;
    int n_dims;
    int64_t offset;
    float freq_scale;
    float freq_base;
    int64_t rope_dims;
    const float* cos_table = nullptr;  // [rope_dims/2]
    const float* sin_table = nullptr;  // [rope_dims/2]
};

struct SoftmaxParams {
    const float* input;
    float* output;
    int rows;
    int cols;
};

struct SiluParams {
    const float* input;
    float* output;
    int size;
};

struct AttnParams {
    const float* Q;
    const float* K;
    const float* V;
    float* output;
    int batch_size;
    int n_head;
    int head_dim;
    int64_t ctx_len;
    int64_t query_pos = -1;  // causal index; -1 => ctx_len - 1
    const float* freq_factors;
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual Status mul_mat_q(const MulMatParams& params) = 0;
    virtual Status rms_norm(const RmsNormParams& params) = 0;
    virtual Status rope(const RopeParams& params) = 0;

    // Apply RoPE to all heads of a token. Default loops single-head rope();
    // backends may override to batch the work into fewer kernel launches.
    virtual Status rope_batched(const RopeBatchedParams& params) {
        for (int h = 0; h < params.n_heads; h++) {
            RopeParams rp;
            rp.data = params.data + static_cast<size_t>(h) * params.n_dims;
            rp.n_dims = params.n_dims;
            rp.rope_dims = params.rope_dims;
            rp.offset = params.offset;
            rp.freq_scale = params.freq_scale;
            rp.freq_base = params.freq_base;
            rp.cos_table = params.cos_table;
            rp.sin_table = params.sin_table;
            Status s = rope(rp);
            if (s != Status::OK) return s;
        }
        return Status::OK;
    }

    virtual Status softmax(const SoftmaxParams& params) = 0;
    virtual Status silu(const SiluParams& params) = 0;
    virtual Status flash_attn(const AttnParams& params) = 0;
    virtual void sync() = 0;

    virtual bool is_available() const = 0;
    virtual std::string name() const = 0;
    virtual Status last_error() const = 0;
};

std::shared_ptr<Backend> create_cpu_ref_backend();
#ifdef GGNPU_HAS_NPU_BACKEND
std::shared_ptr<Backend> create_amd_xdna_backend(int device_id = 0);
#endif

} // namespace ggnpu

#endif // GGNPU_BACKEND_H
