#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <memory>

#include "gguf.h"
#include "tensor.h"
#include "model.h"
#include "backend.h"
#include "cache.h"
#include "bf16.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "weight_cache.h"
#include "quant/kquant.h"
#include "quant/q4_0.h"
#include "quant/q4_1.h"
#include "quant/q8_0.h"
#include "quant/iq4_nl.h"
#include "quant/iq4_xs.h"
#include "quant/iquants.h"

namespace ggnpu {

namespace {

// Dequantize one row from a mmap'd GGUF weight tensor (accurate; no INT8 cache).
void dequant_tensor_row(const TensorView* tv, int row, float* out, int row_dim) {
    if (!tv || !tv->data || row_dim <= 0) return;

    if (tv->type == GgmlType::F32) {
        const float* src = reinterpret_cast<const float*>(tv->data);
        std::memcpy(out, src + static_cast<size_t>(row) * row_dim, static_cast<size_t>(row_dim) * sizeof(float));
        return;
    }

    if (tv->type == GgmlType::BF16) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(tv->data) + static_cast<size_t>(row) * row_dim;
        for (int k = 0; k < row_dim; k++) out[k] = bf16_to_f32(src[k]);
        return;
    }

    if (tv->type == GgmlType::F16) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(tv->data) + static_cast<size_t>(row) * row_dim;
        for (int k = 0; k < row_dim; k++) out[k] = f16_to_f32(src[k]);
        return;
    }

    if (tv->type == GgmlType::Q4_0) {
        constexpr int QK4_0 = 32;
        constexpr size_t Q4_0_BLOCK_BYTES = 18;
        const size_t blocks_per_row =
            (static_cast<size_t>(row_dim) + QK4_0 - 1) / QK4_0;
        const size_t row_bytes = blocks_per_row * Q4_0_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK4_0);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q4_0_block(row_data + b * Q4_0_BLOCK_BYTES, block_f32.data());
            for (int j = 0; j < QK4_0; j++) {
                size_t k = b * QK4_0 + j;
                if (k < static_cast<size_t>(row_dim)) out[k] = block_f32[j];
            }
        }
        return;
    }

    if (tv->type == GgmlType::Q4_1) {
        constexpr int QK4_1 = 32;
        constexpr size_t Q4_1_BLOCK_BYTES = 20;
        const size_t blocks_per_row =
            (static_cast<size_t>(row_dim) + QK4_1 - 1) / QK4_1;
        const size_t row_bytes = blocks_per_row * Q4_1_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK4_1);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q4_1_block(row_data + b * Q4_1_BLOCK_BYTES, block_f32.data());
            for (int j = 0; j < QK4_1; j++) {
                size_t k = b * QK4_1 + j;
                if (k < static_cast<size_t>(row_dim)) out[k] = block_f32[j];
            }
        }
        return;
    }

    if (tv->type == GgmlType::Q8_0) {
        constexpr int QK8_0 = 32;
        constexpr size_t Q8_0_BLOCK_BYTES = 34;
        const size_t blocks_per_row =
            (static_cast<size_t>(row_dim) + QK8_0 - 1) / QK8_0;
        const size_t row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK8_0);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q8_0_block(row_data + b * Q8_0_BLOCK_BYTES, block_f32.data());
            for (int j = 0; j < QK8_0; j++) {
                size_t k = b * QK8_0 + j;
                if (k < static_cast<size_t>(row_dim)) out[k] = block_f32[j];
            }
        }
        return;
    }

    if (tv->type == GgmlType::IQ4_NL) {
        constexpr int QK4_NL = 32;
        constexpr size_t IQ4_NL_BLOCK_BYTES = 18;
        const size_t blocks_per_row =
            (static_cast<size_t>(row_dim) + QK4_NL - 1) / QK4_NL;
        const size_t row_bytes = blocks_per_row * IQ4_NL_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK4_NL);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_iq4_nl_block(row_data + b * IQ4_NL_BLOCK_BYTES, block_f32.data());
            for (int j = 0; j < QK4_NL; j++) {
                size_t k = b * QK4_NL + j;
                if (k < static_cast<size_t>(row_dim)) out[k] = block_f32[j];
            }
        }
        return;
    }

    const size_t blocks_per_row =
        (static_cast<size_t>(row_dim) + QK_K - 1) / QK_K;

    if (tv->type == GgmlType::IQ4_XS) {
        const size_t row_bytes = blocks_per_row * 136;  // IQ4_XS super-block bytes
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK_K);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_iq4_xs_block(row_data + b * 136, block_f32.data());
            for (size_t j = 0; j < QK_K; j++) {
                size_t k = b * QK_K + j;
                if (k < static_cast<size_t>(row_dim)) out[k] = block_f32[j];
            }
        }
        return;
    }

    // Codebook i-quants: all 256-value super-blocks sharing one dequant signature,
    // so table-dispatch by (block bytes, decode fn) instead of a case each.
    {
        void (*iq_deq)(const uint8_t*, float*) = nullptr;
        size_t iq_bytes = 0;
        switch (tv->type) {
            case GgmlType::IQ2_XXS: iq_deq = dequant_iq2_xxs_block; iq_bytes = IQ2_XXS_BLOCK_BYTES; break;
            case GgmlType::IQ2_XS:  iq_deq = dequant_iq2_xs_block;  iq_bytes = IQ2_XS_BLOCK_BYTES;  break;
            case GgmlType::IQ2_S:   iq_deq = dequant_iq2_s_block;   iq_bytes = IQ2_S_BLOCK_BYTES;   break;
            case GgmlType::IQ3_XXS: iq_deq = dequant_iq3_xxs_block; iq_bytes = IQ3_XXS_BLOCK_BYTES; break;
            case GgmlType::IQ3_S:   iq_deq = dequant_iq3_s_block;   iq_bytes = IQ3_S_BLOCK_BYTES;   break;
            case GgmlType::IQ1_S:   iq_deq = dequant_iq1_s_block;   iq_bytes = IQ1_S_BLOCK_BYTES;   break;
            case GgmlType::IQ1_M:   iq_deq = dequant_iq1_m_block;   iq_bytes = IQ1_M_BLOCK_BYTES;   break;
            default: break;
        }
        if (iq_deq) {
            const size_t row_bytes = blocks_per_row * iq_bytes;
            const uint8_t* row_data =
                static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
            std::vector<float> block_f32(QK_K);
            for (size_t b = 0; b < blocks_per_row; b++) {
                iq_deq(row_data + b * iq_bytes, block_f32.data());
                for (size_t j = 0; j < QK_K; j++) {
                    size_t k = b * QK_K + j;
                    if (k < static_cast<size_t>(row_dim)) out[k] = block_f32[j];
                }
            }
            return;
        }
    }

    if (tv->type == GgmlType::Q2_K) {
        const size_t row_bytes = blocks_per_row * Q2_K_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK_K);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q2_k_block(row_data + b * Q2_K_BLOCK_BYTES, block_f32.data());
            for (size_t j = 0; j < QK_K; j++) {
                size_t k = b * QK_K + j;
                if (k < static_cast<size_t>(row_dim)) out[k] = block_f32[j];
            }
        }
        return;
    }

    if (tv->type == GgmlType::Q3_K) {
        const size_t row_bytes = blocks_per_row * Q3_K_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK_K);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q3_k_block(row_data + b * Q3_K_BLOCK_BYTES, block_f32.data());
            for (size_t j = 0; j < QK_K; j++) {
                size_t k = b * QK_K + j;
                if (k < static_cast<size_t>(row_dim)) out[k] = block_f32[j];
            }
        }
        return;
    }

    if (tv->type == GgmlType::Q6_K) {
        const size_t row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK_K);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q6_k_block(row_data + b * Q6_K_BLOCK_BYTES, block_f32.data());
            for (size_t j = 0; j < QK_K; j++) {
                size_t k = b * QK_K + j;
                if (k < static_cast<size_t>(row_dim)) {
                    out[k] = block_f32[j];
                }
            }
        }
        return;
    }

    if (tv->type == GgmlType::Q4_K) {
        const size_t row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK_K);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q4_k_block(row_data + b * Q4_K_BLOCK_BYTES, block_f32.data());
            for (size_t j = 0; j < QK_K; j++) {
                size_t k = b * QK_K + j;
                if (k < static_cast<size_t>(row_dim)) {
                    out[k] = block_f32[j];
                }
            }
        }
        return;
    }

    if (tv->type == GgmlType::Q5_K) {
        const size_t row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
        const uint8_t* row_data =
            static_cast<const uint8_t*>(tv->data) + static_cast<size_t>(row) * row_bytes;
        std::vector<float> block_f32(QK_K);
        for (size_t b = 0; b < blocks_per_row; b++) {
            dequant_q5_k_block(row_data + b * Q5_K_BLOCK_BYTES, block_f32.data());
            for (size_t j = 0; j < QK_K; j++) {
                size_t k = b * QK_K + j;
                if (k < static_cast<size_t>(row_dim)) {
                    out[k] = block_f32[j];
                }
            }
        }
    }
}

// Forward declaration (definition is later in this file). Needed for NPU code paths
// in compute_logits and various bench/inference helpers when GGNPU_HAS_NPU_BACKEND is enabled.
bool attach_kquant_scales(MulMatParams& p, const TensorView* w, WeightCache& cache);

// Vocab projection: hidden @ weight^T.
// NPU path: decode weight to INT8 via WeightCache, one mul_mat_q call (M=1, N=vocab, K=hidden).
// CPU fallback: per-row dequant + dot product (used when NPU unavailable or unsupported type).
void compute_logits(const float* hidden, const TensorView* weight, float* logits,
                    int vocab_size, int hidden_size, WeightCache& weight_cache,
                    Backend& backend) {
    if (!weight || !hidden || !logits) return;

    bool npu_path = backend.name() == "amd_xdna";

    if (npu_path && (weight->type == GgmlType::Q2_K || weight->type == GgmlType::Q3_K ||
                     weight->type == GgmlType::Q4_K || weight->type == GgmlType::Q6_K ||
                     weight->type == GgmlType::Q4_0 || weight->type == GgmlType::Q4_0_4_4 ||
                     weight->type == GgmlType::Q4_1 ||
                     weight->type == GgmlType::Q8_0 ||
                     weight->type == GgmlType::Q5_K || weight->type == GgmlType::BF16 ||
                     weight->type == GgmlType::F16 ||
                     weight->type == GgmlType::IQ4_NL || weight->type == GgmlType::IQ4_XS ||
                     weight->type == GgmlType::IQ2_XXS || weight->type == GgmlType::IQ2_XS ||
                     weight->type == GgmlType::IQ2_S || weight->type == GgmlType::IQ3_XXS ||
                     weight->type == GgmlType::IQ3_S || weight->type == GgmlType::IQ1_S ||
                     weight->type == GgmlType::IQ1_M)) {
        const int8_t* decoded = weight_cache.get_or_decode(*weight);
        if (!decoded) goto cpu_fallback;

        std::vector<float> C(static_cast<size_t>(vocab_size), 0.0f);
        MulMatParams p;
        p.A = hidden;
        p.B = decoded;
        p.C = C.data();
        p.M = 1;
        p.N = vocab_size;
        p.K = hidden_size;
        p.lda = hidden_size;
        p.ldb = hidden_size;
        p.ldc = vocab_size;
        p.n_batches = 1;
        p.B_type = weight->type;
        if (!attach_kquant_scales(p, weight, weight_cache)) {
            goto cpu_fallback;
        }

        Status st = backend.mul_mat_q(p);
        backend.sync();
        if (st == Status::OK) {
            std::memcpy(logits, C.data(), static_cast<size_t>(vocab_size) * sizeof(float));
            return;
        }
    }

cpu_fallback:
    std::vector<float> row(static_cast<size_t>(hidden_size));
    for (int v = 0; v < vocab_size; v++) {
        dequant_tensor_row(weight, v, row.data(), hidden_size);
        float sum = 0.0f;
        for (int d = 0; d < hidden_size; d++) {
            sum += hidden[d] * row[d];
        }
        logits[v] = sum;
    }
}

// True for the K-quant / packed weight types compute_logits can run on the NPU
// (shared by the single-row and batched vocab projections).
static bool npu_logits_weight_type(GgmlType t) {
    return t == GgmlType::Q2_K || t == GgmlType::Q3_K || t == GgmlType::Q4_K ||
           t == GgmlType::Q6_K || t == GgmlType::Q4_0 || t == GgmlType::Q4_0_4_4 ||
           t == GgmlType::Q4_1 ||
           t == GgmlType::Q8_0 || t == GgmlType::Q5_K || t == GgmlType::BF16 ||
           t == GgmlType::F16 ||
           t == GgmlType::IQ4_NL || t == GgmlType::IQ4_XS || t == GgmlType::IQ2_XXS ||
           t == GgmlType::IQ2_XS || t == GgmlType::IQ2_S || t == GgmlType::IQ3_XXS ||
           t == GgmlType::IQ3_S || t == GgmlType::IQ1_S || t == GgmlType::IQ1_M;
}

// Batched vocab projection: `n_rows` hidden rows @ weight^T -> `n_rows`*vocab
// logits (row-major, ldc=vocab_size). One mul_mat_q at M=n_rows; for n_rows<=16
// this stays on the small-M decode kernel, so verifying a speculative draft
// block of k+1<=16 rows costs one logits launch (same device cost as the M=1
// single-token logits), not n_rows separate launches. CPU fallback dots each row.
void compute_logits_rows(const float* hidden, int n_rows, const TensorView* weight,
                         float* logits, int vocab_size, int hidden_size,
                         WeightCache& weight_cache, Backend& backend) {
    if (!weight || !hidden || !logits || n_rows <= 0) return;

    if (backend.name() == "amd_xdna" && npu_logits_weight_type(weight->type)) {
        const int8_t* decoded = weight_cache.get_or_decode(*weight);
        if (decoded) {
            std::vector<float> C(static_cast<size_t>(n_rows) * vocab_size, 0.0f);
            MulMatParams p;
            p.A = hidden;
            p.B = decoded;
            p.C = C.data();
            p.M = n_rows;
            p.N = vocab_size;
            p.K = hidden_size;
            p.lda = hidden_size;
            p.ldb = hidden_size;
            p.ldc = vocab_size;
            p.n_batches = 1;
            p.B_type = weight->type;
            if (attach_kquant_scales(p, weight, weight_cache)) {
                Status st = backend.mul_mat_q(p);
                backend.sync();
                if (st == Status::OK) {
                    std::memcpy(logits, C.data(),
                                static_cast<size_t>(n_rows) * vocab_size * sizeof(float));
                    return;
                }
            }
        }
    }

    // CPU fallback: dequant each weight row once, dot against every hidden row.
    std::vector<float> wrow(static_cast<size_t>(hidden_size));
    for (int v = 0; v < vocab_size; v++) {
        dequant_tensor_row(weight, v, wrow.data(), hidden_size);
        for (int r = 0; r < n_rows; r++) {
            const float* h = hidden + static_cast<size_t>(r) * hidden_size;
            float sum = 0.0f;
            for (int d = 0; d < hidden_size; d++) sum += h[d] * wrow[d];
            logits[static_cast<size_t>(r) * vocab_size + v] = sum;
        }
    }
}

// Prompt-lookup drafter: find the most recent earlier occurrence of the last
// `ng` tokens of `seq` and return up to `k` tokens that followed it, preferring
// longer n-gram matches. Pure host-side lookup (no draft model); returns empty
// when nothing matches, in which case the caller runs a normal single-token step.
static std::vector<int> lookup_draft(const std::vector<int>& seq,
                                     int ngram_max, int ngram_min, int k) {
    const int n = static_cast<int>(seq.size());
    for (int ng = ngram_max; ng >= ngram_min; ng--) {
        if (n < ng + 1) continue;
        for (int i = n - ng - 1; i >= 0; i--) {  // most-recent match first
            bool match = true;
            for (int j = 0; j < ng; j++) {
                if (seq[i + j] != seq[n - ng + j]) { match = false; break; }
            }
            if (!match) continue;
            std::vector<int> draft;
            for (int j = 0; j < k && (i + ng + j) < n; j++)
                draft.push_back(seq[i + ng + j]);
            if (!draft.empty()) return draft;
        }
    }
    return {};
}

// Legacy name kept for compatibility — forwards to compute_logits with CPU-only path.
void compute_logits_f32(const float* hidden, const TensorView* weight, float* logits,
                        int vocab_size, int hidden_size) {
    if (!weight || !hidden || !logits) return;
    std::vector<float> row(static_cast<size_t>(hidden_size));
    for (int v = 0; v < vocab_size; v++) {
        dequant_tensor_row(weight, v, row.data(), hidden_size);
        float sum = 0.0f;
        for (int d = 0; d < hidden_size; d++) {
            sum += hidden[d] * row[d];
        }
        logits[v] = sum;
    }
}

void print_top_logits(const std::vector<float>& logits, const Tokenizer& tokenizer, int k = 8) {
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + std::min(k, static_cast<int>(order.size())),
                      order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    for (int i = 0; i < k && i < static_cast<int>(order.size()); i++) {
        int id = order[static_cast<size_t>(i)];
        std::cout << "    top" << i << ": id=" << id << " logit=" << logits[static_cast<size_t>(id)]
                  << " text=" << tokenizer.decode(id) << "\n";
    }
}

constexpr const char* VERSION = "0.1.0";

struct InferenceTimings {
    double embed_ms = 0;
    double rms_norm_ms = 0;
    double matmul_ms = 0;
    double rope_ms = 0;
    double kv_expand_ms = 0;
    double flash_attn_ms = 0;
    double residual_ms = 0;
    double silu_ms = 0;
    double logits_ms = 0;
    double sample_ms = 0;
    int token_steps = 0;

    void add(const InferenceTimings& o) {
        embed_ms += o.embed_ms;
        rms_norm_ms += o.rms_norm_ms;
        matmul_ms += o.matmul_ms;
        rope_ms += o.rope_ms;
        kv_expand_ms += o.kv_expand_ms;
        flash_attn_ms += o.flash_attn_ms;
        residual_ms += o.residual_ms;
        silu_ms += o.silu_ms;
        logits_ms += o.logits_ms;
        sample_ms += o.sample_ms;
        token_steps += o.token_steps;
    }

    void print_summary() const {
        const double total = embed_ms + rms_norm_ms + matmul_ms + rope_ms + kv_expand_ms +
                             flash_attn_ms + residual_ms + silu_ms + logits_ms + sample_ms;
        if (total <= 0.0) return;

        auto pct = [&](double ms) {
            return total > 0.0 ? (100.0 * ms / total) : 0.0;
        };
        auto row = [&](const char* name, double ms) {
            if (ms <= 0.0) return;
            std::cout << "    " << name << ": " << ms << " ms (" << pct(ms) << "%)\n";
        };

        std::cout << "\nPer-op timings (" << token_steps << " token steps, "
                  << total << " ms total):\n";
        row("embed", embed_ms);
        row("rms_norm", rms_norm_ms);
        row("matmul", matmul_ms);
        row("rope", rope_ms);
        row("kv_expand", kv_expand_ms);
        row("flash_attn", flash_attn_ms);
        row("residual", residual_ms);
        row("silu", silu_ms);
        row("logits", logits_ms);
        row("sample", sample_ms);
    }
};

class ScopedTimer {
public:
    explicit ScopedTimer(double& accum_ms) : accum_ms_(accum_ms) {
        start_ = std::chrono::steady_clock::now();
    }
    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        accum_ms_ += std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    double& accum_ms_;
    std::chrono::steady_clock::time_point start_;
};

struct CliParams {
    std::string model_path;
    std::string prompt;
    int max_tokens = 128;
    int ctx_size = 0;
    int threads = 4;
    float temp = 0.0f;
    uint64_t seed = 0;
    int npu_device = 0;
    bool no_cache = false;
    std::string cache_dir = "";
    bool verbose = false;
    bool dump_tensors = false;
    bool bench_matmul = false;
    bool bench_layer = false;
    int bench_layer_num = 0;
    bool bench_logits = false;
    bool show_version = false;
    bool quiet = false;
};

void print_help() {
    std::cout << "ggnpu - Run GGUF AI Models on AMD NPUs\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ggnpu [options]                    Text generation\n";
    std::cout << "  ggnpu --dump-tensors               List model tensors\n";
    std::cout << "  ggnpu bench-matmul                 NPU matmul benchmark\n";
    std::cout << "  ggnpu --version                    Show version\n\n";
    std::cout << "Commands:\n";
    std::cout << "  (default)                          Text generation\n";
    std::cout << "  --dump-tensors                     List tensors; no NPU\n";
    std::cout << "  bench-matmul                       NPU matmul benchmark\n";
    std::cout << "  bench-layer [options]              Validate one decoder layer on NPU vs CPU ref\n";
    std::cout << "  bench-logits [options]             Print top logits after prompt (CPU F32)\n";
    std::cout << "  --version                          Version + backend info\n\n";
    std::cout << "Flags:\n";
    std::cout << "  -m, --model <path>                 Path to .gguf (required)\n";
    std::cout << "  -p, --prompt <text>                Input prompt\n";
    std::cout << "  -n, --max-tokens <n>               Max new tokens (default: 128)\n";
    std::cout << "  -c, --ctx-size <n>                 Context window (default: model default)\n";
    std::cout << "  -t, --threads <n>                  CPU threads for I/O (default: 4)\n";
    std::cout << "      --temp <f>                     Temperature (0 = greedy, default: 0)\n";
    std::cout << "      --seed <n>                     RNG seed (0 = random, default: 0)\n";
    std::cout << "      --npu-device <n>               NPU index (default: 0)\n";
    std::cout << "      --no-cache                     Disable caches\n";
    std::cout << "      --cache-dir <path>             Cache directory (default: ~/.cache/ggnpu)\n";
    std::cout << "  -v, --verbose                      Per-op timings\n";
    std::cout << "      --quiet                        Status to stderr; generated text only on stdout\n";
    std::cout << "      --layer <n>                    Layer number for bench-layer (default: 0)\n";
    std::cout << "  -h, --help                         Print help\n";
    std::cout << "      --version                      Print version\n\n";
    std::cout << "Examples:\n";
    std::cout << "  ggnpu -m llama-3.2-1b-Q4_K_M.gguf -p \"The capital of France is\" -n 64\n";
    std::cout << "  ggnpu -m model.gguf -p \"Once upon a time\" -c 4096 -n 256 --temp 0.7 --seed 42\n";
    std::cout << "  ggnpu -m model.gguf --dump-tensors\n";
}

CliParams parse_args(int argc, char* argv[]) {
    CliParams params;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_help();
            exit(0);
        } else if (arg == "--version") {
            params.show_version = true;
        } else if (arg == "--dump-tensors") {
            params.dump_tensors = true;
        } else if (arg == "bench-matmul") {
            params.bench_matmul = true;
        } else if (arg == "bench-layer") {
            params.bench_layer = true;
        } else if (arg == "bench-logits") {
            params.bench_logits = true;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) params.model_path = argv[++i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 < argc) params.prompt = argv[++i];
        } else if (arg == "-n" || arg == "--max-tokens") {
            if (i + 1 < argc) params.max_tokens = std::atoi(argv[++i]);
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (i + 1 < argc) params.ctx_size = std::atoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) params.threads = std::atoi(argv[++i]);
        } else if (arg == "--temp") {
            if (i + 1 < argc) params.temp = std::atof(argv[++i]);
        } else if (arg == "--seed") {
            if (i + 1 < argc) params.seed = std::stoull(argv[++i]);
        } else if (arg == "--npu-device") {
            if (i + 1 < argc) params.npu_device = std::atoi(argv[++i]);
        } else if (arg == "--no-cache") {
            params.no_cache = true;
        } else if (arg == "--cache-dir") {
            if (i + 1 < argc) params.cache_dir = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--quiet") {
            params.quiet = true;
        } else if (arg == "--layer") {
            if (i + 1 < argc) params.bench_layer_num = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help();
            exit(1);
        }
    }

    return params;
}

const char* status_name(Status status) {
    switch (status) {
    case Status::OK:
        return "OK";
    case Status::ERROR:
        return "ERROR";
    case Status::NOT_FOUND:
        return "NOT_FOUND";
    case Status::INVALID_PARAM:
        return "INVALID_PARAM";
    case Status::NPU_UNAVAILABLE:
        return "NPU_UNAVAILABLE";
    case Status::OUT_OF_MEMORY:
        return "OUT_OF_MEMORY";
    default:
        return "UNKNOWN";
    }
}

// BF16 roundtrip functions are in src/utils/bf16.cpp (shared header: ggnpu/bf16.h)
// Used here for bench-layer RMSNorm comparisons matching NPU DMA path.

// Forward declarations — defined later in this translation unit.
struct CliParams;
bool attach_kquant_scales(MulMatParams& p, const TensorView* w, WeightCache& cache);

// Precomputed RoPE embeddings: cos/sin tables indexed by [position][dim/2].
// Angles depend only on position and dimension, not input data — computed once.
struct RopeCache {
    std::vector<float> cos;  // size: max_pos * (n_dims / 2)
    std::vector<float> sin;  // size: max_pos * (n_dims / 2)
    int64_t max_pos = 0;
    int n_dims = 0;
    float rope_freq_scale = 1.0f;
    float rope_freq_base = 10000.0f;
    std::vector<float> freq_factors;

    void build(int64_t max_pos, int n_dims, float freq_scale, float freq_base,
               const std::vector<float>& freq_factors) {
        this->max_pos = max_pos;
        this->n_dims = n_dims;
        this->rope_freq_scale = freq_scale;
        this->rope_freq_base = freq_base;
        this->freq_factors = freq_factors;

        int pairs = n_dims / 2;
        cos.resize(static_cast<size_t>(max_pos) * pairs);
        sin.resize(static_cast<size_t>(max_pos) * pairs);

        for (int64_t pos = 0; pos < max_pos; pos++) {
            for (int i = 0; i < pairs; i++) {
                float ff = (i < static_cast<int>(freq_factors.size()))
                               ? freq_factors[static_cast<size_t>(i)]
                               : 1.0f;
                if (ff == 0.0f) ff = 1.0f;
                float wavelength = std::pow(
                    static_cast<float>(rope_freq_base),
                    2.0f * static_cast<float>(i) / static_cast<float>(n_dims));
                float angle = rope_freq_scale * static_cast<float>(pos) / (wavelength * ff);
                size_t idx = static_cast<size_t>(pos) * pairs + static_cast<size_t>(i);
                cos[idx] = std::cos(angle);
                sin[idx] = std::sin(angle);
            }
        }
    }

    const float* cos_ptr(int64_t pos, int pair_idx) const {
        return cos.data() + static_cast<size_t>(pos) * (n_dims / 2) + pair_idx;
    }
    const float* sin_ptr(int64_t pos, int pair_idx) const {
        return sin.data() + static_cast<size_t>(pos) * (n_dims / 2) + pair_idx;
    }
};

struct CompareResult {
    float max_diff = 0.0f;
    float max_ref = 0.0f;
    int mismatches = 0;
};

CompareResult compare_vectors(const float* ref, const float* actual, int n, float mismatch_thresh) {
    CompareResult r;
    for (int i = 0; i < n; i++) {
        float diff = std::fabs(ref[i] - actual[i]);
        if (diff > r.max_diff) r.max_diff = diff;
        if (std::fabs(ref[i]) > r.max_ref) r.max_ref = std::fabs(ref[i]);
        if (diff > mismatch_thresh) r.mismatches++;
    }
    return r;
}

bool report_compare(const char* label, const CompareResult& r, int n,
                    float rel_thresh, float mismatch_frac, float mismatch_thresh) {
    float rel_error = r.max_ref > 0.0f ? r.max_diff / r.max_ref : r.max_diff;
    std::cout << "  " << label << ":\n";
    std::cout << "    Max absolute diff: " << r.max_diff << "\n";
    if (r.max_ref > 0.0f) std::cout << "    Max ref value: " << r.max_ref << "\n";
    std::cout << "    Relative error: " << rel_error << "\n";
    std::cout << "    Mismatches (>" << mismatch_thresh << "): " << r.mismatches << " / " << n << "\n";
    if (rel_error < rel_thresh && r.mismatches < static_cast<int>(n * mismatch_frac)) {
        std::cout << "    Result: PASS\n\n";
        return true;
    }
    std::cout << "    Result: FAIL\n\n";
    return false;
}

bool attach_kquant_scales(MulMatParams& p, const TensorView* w, WeightCache& cache) {
    if (!w || (w->type != GgmlType::Q2_K && w->type != GgmlType::Q3_K &&
               w->type != GgmlType::Q4_K && w->type != GgmlType::Q6_K &&
               w->type != GgmlType::Q4_0 && w->type != GgmlType::Q4_0_4_4 &&
               w->type != GgmlType::Q4_1 &&
               w->type != GgmlType::Q8_0 &&
               w->type != GgmlType::Q5_K && w->type != GgmlType::BF16 &&
               w->type != GgmlType::F16 &&
               w->type != GgmlType::IQ4_NL && w->type != GgmlType::IQ4_XS &&
               w->type != GgmlType::IQ2_XXS && w->type != GgmlType::IQ2_XS &&
               w->type != GgmlType::IQ2_S && w->type != GgmlType::IQ3_XXS &&
               w->type != GgmlType::IQ3_S && w->type != GgmlType::IQ1_S &&
               w->type != GgmlType::IQ1_M)) {
        return true;
    }
    const auto& scales = cache.get_scales(*w);
    if (scales.empty()) {
        std::cerr << "Error: missing K-quant scales for " << w->name << "\n";
        return false;
    }
    if (static_cast<int>(scales.size()) != 1 && static_cast<int>(scales.size()) != p.N) {
        std::cerr << "Error: scale count " << scales.size() << " != 1 or N=" << p.N
                  << " for " << w->name << "\n";
        return false;
    }
    p.scales = scales.data();
    p.n_weight_scales = static_cast<int>(scales.size());
    return true;
}

// Compare NPU decoded matmul vs CPU on-the-fly dequant matmul.
bool bench_quant_matmul(const char* label, Backend& cpu, Backend& npu, WeightCache& cache,
                        const float* A, const TensorView* weight,
                        int M, int N, int K, int lda, int ldb, int ldc,
                        float rel_thresh = 0.1f, float mismatch_thresh = 2.0f) {
    if (!weight) {
        std::cerr << "Error: weight tensor missing for " << label << "\n";
        return false;
    }

    const int8_t* decoded = cache.get_or_decode(*weight);
    if (!decoded) {
        std::cerr << "Error: failed to decode " << weight->name << "\n";
        return false;
    }

    std::vector<float> cpu_out(static_cast<size_t>(M) * N, 0.0f);
    MulMatParams cpu_p;
    cpu_p.A = A; cpu_p.B = weight->data; cpu_p.C = cpu_out.data();
    cpu_p.M = M; cpu_p.N = N; cpu_p.K = K;
    cpu_p.lda = lda; cpu_p.ldb = ldb; cpu_p.ldc = ldc;
    cpu_p.B_type = weight->type;
    cpu.mul_mat_q(cpu_p);

    std::vector<float> npu_out(static_cast<size_t>(M) * N, 0.0f);
    MulMatParams npu_p;
    npu_p.A = A; npu_p.B = decoded; npu_p.C = npu_out.data();
    npu_p.M = M; npu_p.N = N; npu_p.K = K;
    npu_p.lda = lda; npu_p.ldb = ldb; npu_p.ldc = ldc;
    npu_p.B_type = weight->type;
    if (!attach_kquant_scales(npu_p, weight, cache)) return false;
    Status st = npu.mul_mat_q(npu_p);
    npu.sync();
    if (st != Status::OK) {
        std::cerr << "Error: NPU matmul failed for " << label << ": " << status_name(st) << "\n";
        return false;
    }

    auto cmp = compare_vectors(cpu_out.data(), npu_out.data(), M * N, mismatch_thresh);
    std::cout << "  " << label << " (" << M << " x " << K << " x " << N << "):\n";
    std::cout << "    Type: " << ggml_type_name(weight->type) << "\n";
    return report_compare(label, cmp, M * N, rel_thresh, 0.1f, mismatch_thresh);
}

bool validate_matmul_output(const std::vector<float>& output, int expected_value,
                           float tolerance, std::string* error_message) {
    for (size_t i = 0; i < output.size(); i++) {
        float actual = output[i];
        float expected = static_cast<float>(expected_value);
        if (std::fabs(actual - expected) > tolerance) {
            if (error_message) {
                std::ostringstream oss;
                oss << "output[" << i << "] expected " << expected << ", got " << actual;
                *error_message = oss.str();
            }
            return false;
        }
    }

    return true;
}

int sample_token(const std::vector<float>& logits, float temp, uint64_t& seed) {
    if (temp <= 0.0f) {
        int best = 0;
        float best_val = -INFINITY;
        for (int i = 0; i < static_cast<int>(logits.size()); i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return best;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float max_logit = -INFINITY;
    for (float v : logits) {
        if (v > max_logit) max_logit = v;
    }

    std::vector<float> probs;
    probs.reserve(logits.size());
    float sum_exp = 0.0f;
    for (float v : logits) {
        float exp_val = std::exp((v - max_logit) / temp);
        probs.push_back(exp_val);
        sum_exp += exp_val;
    }

    for (float& p : probs) {
        p /= sum_exp;
    }

    std::discrete_distribution<int> dist2(probs.begin(), probs.end());
    seed = rng();
    return dist2(rng);
}

// Helper: find tensor by name in model
const TensorView* find_tensor(const Model& model, const std::string& name) {
    auto it = model.tensor_map().find(name);
    if (it == model.tensor_map().end()) return nullptr;
    return &model.tensors()[it->second];
}

// Helper: find tensor by pattern (e.g., "blk.{layer}.ffn_gate.weight")
const TensorView* find_tensor_pattern(const Model& model, const std::string& pattern, int layer) {
    std::string expanded = pattern;
    size_t pos = expanded.find("{layer}");
    if (pos != std::string::npos) {
        expanded.replace(pos, 7, std::to_string(layer));
    }
    return find_tensor(model, expanded);
}

// Helper: get raw pointer from TensorView
inline const float* get_float_ptr(const TensorView* tv) {
    if (!tv || !tv->data) return nullptr;
    return reinterpret_cast<const float*>(tv->data);
}

inline float* get_float_ptr_mut(std::vector<float>& buf) {
    return buf.data();
}

// Host RMSNorm over exactly `n` elements (no NPU pad). Used for qwen35's small
// per-head norms (QK-norm over head_dim, ssm gated-norm over head_v_dim) where
// the NPU kernel's pow2 padding would mis-scale a non-256 width.
inline void host_rmsnorm(float* dst, const float* src, int n, const float* w, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += src[i] * src[i];
    const float r = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
    for (int i = 0; i < n; i++) dst[i] = src[i] * r * (w ? w[i] : 1.0f);
}

// Partial NEOX RoPE: rotate only the first `rope_dim` of each `head_dim`-wide
// head (pairs i, i+rope_dim/2), leaving dims [rope_dim, head_dim) unchanged.
// Matches ggml_rope_ext(NEOX) with n_dims=rope_dim (Qwen3.5 rotary_factor 0.25).
inline void rope_neox_partial(float* vec, int n_heads, int head_dim, int rope_dim,
                              int64_t pos, float base) {
    const int half = rope_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* p = vec + static_cast<size_t>(h) * head_dim;
        for (int i = 0; i < half; i++) {
            const float theta =
                static_cast<float>(pos) * std::pow(base, -2.0f * static_cast<float>(i) / static_cast<float>(rope_dim));
            const float c = std::cos(theta), s = std::sin(theta);
            const float x0 = p[i], x1 = p[i + half];
            p[i] = x0 * c - x1 * s;
            p[i + half] = x0 * s + x1 * c;
        }
    }
}

} // namespace

} // namespace ggnpu

int main(int argc, char* argv[]) {
    using namespace ggnpu;

    CliParams params = parse_args(argc, argv);

    if (params.show_version) {
        std::cout << "ggnpu version " << VERSION << "\n";
#ifdef GGNPU_HAS_NPU_BACKEND
        std::cout << "NPU backend: AMD XDNA (XRT)\n";
#else
        std::cout << "NPU backend: disabled (compile with -DGGNPU_NPU_BACKEND=ON)\n";
#endif
#ifdef GGNPU_TEST_CPU
        std::cout << "CPU ref backend: enabled (testing)\n";
#endif
        return 0;
    }

    if (argc == 1) {
        print_help();
        return 0;
    }

    if (params.dump_tensors) {
        if (params.model_path.empty()) {
            std::cerr << "Error: --model is required for --dump-tensors\n";
            return 1;
        }

        GgufLoader loader;
        if (!loader.load(params.model_path)) {
            std::cerr << "Error: failed to load GGUF file: " << params.model_path << "\n";
            return 1;
        }

        std::cout << "GGUF Header:\n";
        std::cout << "  Version: " << loader.header().version << "\n";
        std::cout << "  KV pairs: " << loader.header().kv_count << "\n";
        std::cout << "  Tensors: " << loader.header().tensor_count << "\n\n";

        std::cout << "Metadata:\n";
        std::cout << "  Architecture: " << loader.architecture() << "\n";
        std::cout << "  Context length: " << loader.context_length() << "\n";
        std::cout << "  Embedding length: " << loader.embedding_length() << "\n";
        std::cout << "  Block count: " << loader.block_count() << "\n";
        std::cout << "  Feed forward length: " << loader.feed_forward_length() << "\n";
        std::cout << "  Attention heads: " << loader.attention_head_count() << "\n";
        std::cout << "  Attention heads (KV): " << loader.attention_head_count_kv() << "\n";
        std::cout << "  Rope dimension count: " << loader.rope_dimension_count() << "\n";
        std::cout << "  Rope freq scale: " << loader.rope_freq_scale() << "\n";
        std::cout << "  Rope freq base: " << loader.rope_freq_base() << "\n\n";

        std::cout << "Key-Value Pairs:\n";
        for (const auto& [key, kv] : loader.kv_pairs()) {
            std::cout << "  " << key << " = ";
            switch (kv.value_type) {
                case GgufType::STRING:
                    std::cout << "\"" << kv.string_value << "\"\n";
                    break;
                case GgufType::INT64:
                case GgufType::UINT64:
                    std::cout << kv.int_value << "\n";
                    break;
                case GgufType::FLOAT64:
                    std::cout << kv.float_value << "\n";
                    break;
                case GgufType::INT32:
                case GgufType::UINT32:
                    std::cout << kv.int_value << "\n";
                    break;
                case GgufType::FLOAT32:
                    std::cout << kv.float_value << "\n";
                    break;
                case GgufType::BOOL:
                    std::cout << (kv.int_value ? "true" : "false") << "\n";
                    break;
                case GgufType::ARRAY:
                    std::cout << "[array, " << kv.data.size() << " bytes]\n";
                    break;
                default:
                    std::cout << "[" << kv.data.size() << " bytes]\n";
                    break;
            }
        }

        std::cout << "\nTensors (" << loader.tensors().size() << "):\n";
        for (const auto& t : loader.tensors()) {
            std::cout << "  " << t.name << " dims=[";
            for (size_t i = 0; i < t.dims.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << t.dims[i];
            }
            std::cout << "] type=" << ggml_type_name(t.type) << "\n";
        }

        loader.unload();
        return 0;
    }

    if (params.bench_matmul) {
        std::cout << "GGNPU Matmul Benchmark\n";
        std::cout << "======================\n\n";

        std::shared_ptr<Backend> backend;
#ifdef GGNPU_HAS_NPU_BACKEND
        backend = create_amd_xdna_backend(params.npu_device);
        if (!backend || !backend->is_available()) {
            std::cerr << "Error: NPU backend unavailable. AMD NPU (amdxdna) driver must be loaded.\n";
            std::cerr << "  lsmod | grep amdxdna\n";
            std::cerr << "  ls -la /dev/accel/accel0\n";
            return 1;
        }
#else
        std::cerr << "Error: NPU backend not compiled in. Build with -DGGNPU_NPU_BACKEND=ON\n";
        return 1;
#endif

        std::cout << "Backend: " << backend->name() << "\n\n";

        std::vector<std::tuple<int, int, int>> sizes = {
            {1, 256, 256},       // decode GEMV (exercises small-M kernel)
            {16, 256, 256},      // small-M tile, full M
            {1, 2048, 2048},     // decode attn_q shape (single deep-K span)
            {1, 256, 8192},      // decode ffn_down shape (4 deep-K spans, host accum)
            {256, 256, 256},
            {512, 512, 512},
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {3072, 3072, 3072},
            {4096, 4096, 4096},
            {8192, 8192, 8192},
        };

        for (auto [M, N, K] : sizes) {
            std::vector<float> A(M * K, 1.0f);
            std::vector<float> B(K * N, 1.0f);
            std::vector<float> C(M * N, 0.0f);

            MulMatParams p;
            p.A = A.data();
            p.B = B.data();
            p.C = C.data();
            p.M = M;
            p.N = N;
            p.K = K;
            p.lda = K;
            p.ldb = N;
            p.ldc = N;
            p.n_batches = 1;
            p.B_type = GgmlType::F32;

            Status warmup_status = backend->mul_mat_q(p);
            backend->sync();
            if (warmup_status != Status::OK) {
                std::cerr << "Matmul warmup failed for " << M << "x" << K << " x "
                          << K << "x" << N << ": " << status_name(warmup_status) << "\n";
                return 1;
            }

            std::string validation_error;
            if (!validate_matmul_output(C, K, 1e-3f, &validation_error)) {
                std::cerr << "Matmul validation failed for " << M << "x" << K << " x "
                          << K << "x" << N << ": " << validation_error << "\n";
                return 1;
            }

            auto start = std::chrono::high_resolution_clock::now();
            int iterations = 10;
            for (int i = 0; i < iterations; i++) {
                Status st = backend->mul_mat_q(p);
                backend->sync();
                if (st != Status::OK) {
                    std::cerr << "Matmul benchmark iteration failed for " << M << "x" << K
                              << " x " << K << "x" << N << ": " << status_name(st) << "\n";
                    return 1;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double ops = 2.0 * M * N * K * iterations;
            double tflops = ops / (elapsed_ms * 1e-6) / 1e12;

            std::cout << "  " << M << "x" << K << " x " << K << "x" << N
                      << ": " << tflops << " TFLOPS (" << elapsed_ms / iterations << " ms)\n";
        }

        return 0;
    }

    if (params.model_path.empty()) {
        std::cerr << "Error: --model is required\n\n";
        print_help();
        return 1;
    }

    std::ostream& info = params.quiet ? std::cerr : std::cout;
    info << "GGNPU - GGUF NPU Inference Engine v" << VERSION << "\n\n";

    std::shared_ptr<Backend> backend;
#ifdef GGNPU_HAS_NPU_BACKEND
    backend = create_amd_xdna_backend(params.npu_device);
    if (!backend || !backend->is_available()) {
        std::cerr << "Error: NPU backend unavailable. AMD NPU (amdxdna) driver must be loaded.\n";
        std::cerr << "  lsmod | grep amdxdna\n";
        std::cerr << "  ls -la /dev/accel/accel0\n";
        return 1;
    }
#else
    std::cerr << "Error: NPU backend not compiled in. Build with -DGGNPU_NPU_BACKEND=ON\n";
    return 1;
#endif

    info << "Backend: " << backend->name() << "\n";

    std::string cache_dir = params.cache_dir;
    if (cache_dir.empty()) {
        const char* home = std::getenv("HOME");
        cache_dir = home ? std::string(home) + "/.cache/ggnpu" : "~/.cache/ggnpu";
    }

    info << "Loading model: " << params.model_path << "\n";
    Model model;
    if (!model.load(params.model_path)) {
        std::cerr << "Error: failed to load model\n";
        return 1;
    }
    model.set_backend(backend);

    const auto& hparams = model.hparams();

    // Cap context to avoid multi-GB KV allocation on models with huge metadata ctx
    if (params.ctx_size > 0) {
        model.set_context_length(static_cast<uint64_t>(params.ctx_size));
        info << "  Context size overridden to: " << params.ctx_size << "\n";
        // Reinitialize KV cache to respect overridden context size
        if (!model.reinit_kv_cache(params.ctx_size)) {
            std::cerr << "  Warning: failed to reinitialize KV cache with overridden context size\n";
        }
    } else if (hparams.context_length > 2048) {
        model.set_context_length(2048);
        info << "  Context capped to 2048 (model reports "
                  << hparams.context_length << "; use -c to override)\n";
    }
    info << "  Architecture: " << model.gguf().architecture() << "\n";
    info << "  Context: " << hparams.context_length << "\n";
    info << "  Layers: " << hparams.block_count << "\n";
    info << "  Hidden: " << hparams.embedding_length << "\n";
    info << "  Heads: " << hparams.attention_head_count << " (KV: " << hparams.attention_head_count_kv << ")\n";
    info << "  FFN: " << hparams.feed_forward_length << "\n\n";

    // bench-layer generates its own test input — skip the prompt requirement
    if (params.prompt.empty() && !params.bench_layer && !params.bench_logits) {
        info << "No prompt provided. Use --prompt or -p to specify input.\n";
        model.unload();
        return 0;
    }

    // Load tokenizer
    Tokenizer tokenizer;
    tokenizer.load_from_gguf(model.gguf().kv_pairs());
    info << "Tokenizer: " << tokenizer.vocab_size() << " tokens\n";

    // Tokenize prompt
    std::vector<int> input_tokens = tokenizer.encode(params.prompt, true, false);
    info << "Input tokens: " << input_tokens.size() << "\n";
    if (std::getenv("GGNPU_DEBUG_TOKENS")) {
        info << "  ids:";
        for (int id : input_tokens) info << " " << id;
        info << "\n";
    }
    info << "Prompt: " << params.prompt << "\n\n";

    // Weight cache: decode GGUF quantized weights to INT8 for NPU.
    // Fingerprint the cache by model path + file size so two models that share
    // an architecture (identical tensor names/shapes/quant) never collide on the
    // persistent disk cache and run on each other's weights.
    CompileCache compile_cache(cache_dir, !params.no_cache);
    std::string model_id = params.model_path;
    {
        std::error_code ec;
        auto sz = std::filesystem::file_size(params.model_path, ec);
        if (!ec) model_id += ":" + std::to_string(sz);
    }
    WeightCache weight_cache(compile_cache, model_id);
    info << "Weight cache initialized\n";

    // Setup working buffers
    int hidden_size = static_cast<int>(hparams.embedding_length);
    int ffn_size = static_cast<int>(hparams.feed_forward_length);
    int num_layers = static_cast<int>(hparams.block_count);
    int num_heads = static_cast<int>(hparams.attention_head_count);
    int num_kv_heads = static_cast<int>(hparams.attention_head_count_kv);
    // rope.dimension_count is optional in GGUF (e.g. qwen2 omits it); fall back to
    // the natural head_dim = embedding / n_head. Both equal 64 for Llama 3.2 1B,
    // and head_dim=128 for Qwen2.5 1.5B (rope is applied over the full head).
    int head_dim = static_cast<int>(hparams.rope_dimension_count);
    // Some archs set an explicit head_dim via attention.key_length that differs
    // from embedding/n_head (qwen3: hidden=1024, 16 heads, head_dim=128).
    if (head_dim == 0)
        head_dim = static_cast<int>(model.gguf().get_int(model.gguf().arch() + ".attention.key_length", 0));
    if (head_dim == 0 && num_heads > 0) head_dim = hidden_size / num_heads;
    // Q-side width = n_head * head_dim. This equals hidden_size for llama/qwen2 but
    // is larger for qwen3 (16*128=2048 vs hidden 1024); K/V use n_kv_head*head_dim.
    const int q_dim = num_heads * head_dim;
    int ctx_size = static_cast<int>(model.hparams().context_length);
    float rms_eps = 1e-5f;
    if (hparams.attention_layer_norm_rms_epsilon > 0) {
        rms_eps = static_cast<float>(hparams.attention_layer_norm_rms_epsilon);
    }
    float rope_freq_scale = static_cast<float>(hparams.rope_freq_scale);
    float rope_freq_base = static_cast<float>(hparams.rope_freq_base);

    // Working buffers for activations (resized per iteration for prefill/decode)
    std::vector<float> inp_embd;
    std::vector<float> inp_norm;
    std::vector<float> q_proj;
    std::vector<float> k_proj;
    std::vector<float> v_proj;
    std::vector<float> k_rope;
    std::vector<float> q_rope;
    std::vector<float> attn_output;
    std::vector<float> ffn_gate;
    std::vector<float> ffn_up;
    std::vector<float> ffn_down;
    std::vector<float> ffn_silu;
    const int vocab_size = tokenizer.vocab_size();
    std::vector<float> logits(static_cast<size_t>(vocab_size));

    // Preallocate GQA K/V expand buffers (max size: num_heads * ctx_size * head_dim).
    // Reused every token to avoid per-token allocation.
    const int64_t kv_expand_max = static_cast<int64_t>(num_heads) * ctx_size * head_dim;
    std::vector<float> k_expanded(static_cast<size_t>(kv_expand_max));
    std::vector<float> v_expanded(static_cast<size_t>(kv_expand_max));

    // Llama 3+ ships rope_freqs.weight as per-dimension factors for ggml_rope_ext (not divisors).
    std::vector<float> rope_freq_factors(head_dim / 2, 1.0f);
    const TensorView* rope_freqs_t = find_tensor(model, "rope_freqs.weight");
    if (rope_freqs_t && rope_freqs_t->type == GgmlType::F32) {
        const float* rf = get_float_ptr(rope_freqs_t);
        for (int i = 0; i < head_dim / 2; i++) {
            rope_freq_factors[static_cast<size_t>(i)] = rf[i];
        }
    }

    // Precompute RoPE embeddings for all positions up to context size.
    RopeCache rope_cache;
    rope_cache.build(ctx_size, head_dim, rope_freq_scale, rope_freq_base, rope_freq_factors);

    uint64_t rng_seed = params.seed;
    if (rng_seed == 0) {
        rng_seed = std::chrono::steady_clock::now().time_since_epoch().count();
    }

    int64_t pos = 0;
    int generated = 0;
    int last_sampled = -1;
    size_t prompt_idx = 0;
    int max_layers_override = num_layers;
    if (const char* ml = std::getenv("GGNPU_MAX_LAYERS")) {
        max_layers_override = std::max(0, std::atoi(ml));
        if (max_layers_override > num_layers) max_layers_override = num_layers;
    }

    InferenceTimings total_timings;

    // Llama uses GGML_ROPE_TYPE_NORMAL: rotate adjacent pairs (2i, 2i+1), not NeoX halves.
    // Uses precomputed rope_cache (built above) — no per-call sin/cos.
    auto apply_rope_cpu = [&](float* out, const float* inp, int n_heads, int64_t offset, int n_dims) {
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < n_dims / 2; i++) {
                const float cos_val = *rope_cache.cos_ptr(offset, i);
                const float sin_val = *rope_cache.sin_ptr(offset, i);

                const int i0 = h * n_dims + 2 * i;
                const int i1 = i0 + 1;
                float v0 = inp[i0];
                float v1 = inp[i1];
                out[i0] = v0 * cos_val - v1 * sin_val;
                out[i1] = v0 * sin_val + v1 * cos_val;
            }
        }
    };

    // NPU RoPE: a single backend->rope_batched() launch packs many heads per
    // kernel launch (8 heads / 512-elem kernel for head_dim=64). cos/sin tables
    // from rope_cache are identical across heads at a given offset and already
    // fold in freq_factors. Opt-in via GGNPU_NPU_ROPE=1.
    const bool npu_rope_enabled =
        (backend->name() == "amd_xdna") && std::getenv("GGNPU_NPU_ROPE") != nullptr;
    auto apply_rope_npu = [&](float* out, const float* inp, int n_heads, int64_t offset, int n_dims) -> bool {
        std::memcpy(out, inp, static_cast<size_t>(n_heads) * n_dims * sizeof(float));
        RopeBatchedParams rp;
        rp.data = out;
        rp.n_heads = n_heads;
        rp.n_dims = n_dims;
        rp.rope_dims = n_dims;
        rp.offset = offset;
        rp.freq_scale = rope_freq_scale;
        rp.freq_base = rope_freq_base;
        rp.cos_table = rope_cache.cos_ptr(offset, 0);
        rp.sin_table = rope_cache.sin_ptr(offset, 0);
        return backend->rope_batched(rp) == Status::OK;
    };

    // Dispatch: NPU when enabled (fall back to CPU on any failure), else CPU.
    auto apply_rope = [&](float* out, const float* inp, int n_heads, int64_t offset, int n_dims) {
        if (npu_rope_enabled) {
            if (apply_rope_npu(out, inp, n_heads, offset, n_dims)) return;
            std::cerr << "Warning: NPU RoPE failed; falling back to CPU for this step\n";
        }
        apply_rope_cpu(out, inp, n_heads, offset, n_dims);
    };

    auto rms_norm = [&](float* out, const float* inp, int n_rows, int row_size, float eps,
                        const float* weight) -> bool {
        for (int row = 0; row < n_rows; row++) {
            RmsNormParams rms_params;
            rms_params.input = inp + row * row_size;
            rms_params.output = out + row * row_size;
            rms_params.size = row_size;
            rms_params.eps = eps;
            rms_params.weight = weight;
            if (backend->rms_norm(rms_params) != Status::OK) {
                std::cerr << "Error: rms_norm failed at row " << row << " (N=" << row_size << ")\n";
                return false;
            }
        }
        return true;
    };

    const bool npu_matmul = (backend->name() == "amd_xdna");

    // GGNPU_FP32_SSM: run the whole forward's elementwise math in fp32 on the CPU
    // (matmul + rmsnorm + FFN silu), not just weights. Diagnostic for qwen35 gated
    // delta-net numerical sensitivity: GGNPU_FP32_MATMUL alone leaves rmsnorm/silu
    // on the NPU (int8), whose ~1% residual error is the suspected cause of the
    // recurrent-state collapse by layer ~5. Setting this implies FP32_MATMUL.
    // NOTE: flash_attn (attention layers) still runs on the NPU under this flag.
    const bool fp32_ssm = std::getenv("GGNPU_FP32_SSM") != nullptr;

    auto mul_mat_weight = [&](const float* A, const TensorView* weight, float* C,
                              int M, int N, int K, int lda, int ldb, int ldc,
                              double* matmul_ms = nullptr) -> bool {
        if (!weight) return false;

        std::unique_ptr<ScopedTimer> matmul_timer;
        if (matmul_ms) matmul_timer = std::make_unique<ScopedTimer>(*matmul_ms);

        // Diagnostic: high-precision CPU fp32 reference matmul (dequant weight to
        // f32, exact dot product). Isolates NPU int8 quantization error. M is 1
        // in the qwen35/decode paths. Enable with GGNPU_FP32_MATMUL=1.
        static const bool fp32_matmul = std::getenv("GGNPU_FP32_MATMUL") != nullptr;
        if ((fp32_matmul || fp32_ssm) && M == 1) {
            std::vector<float> wr(static_cast<size_t>(K));
            for (int n = 0; n < N; n++) {
                dequant_tensor_row(weight, n, wr.data(), K);
                float acc = 0.0f;
                for (int k = 0; k < K; k++) acc += A[k] * wr[static_cast<size_t>(k)];
                C[static_cast<size_t>(n)] = acc;
            }
            return true;
        }

        // F32 weight tensors (e.g. qwen35moe's ssm_alpha/ssm_beta projections, which
        // ship unquantized where dense qwen35 uses Q8_0) are not decodable by the NPU
        // int8 path — it overflows to inf. They are always small projections, so
        // compute them exactly on host. Mirrors the F16==BF16 matmul-weight wiring.
        if (weight->type == GgmlType::F32) {
            std::vector<float> wr(static_cast<size_t>(K));
            for (int n = 0; n < N; n++) {
                dequant_tensor_row(weight, n, wr.data(), K);
                for (int m = 0; m < M; m++) {
                    const float* a = A + static_cast<size_t>(m) * lda;
                    float acc = 0.0f;
                    for (int k = 0; k < K; k++) acc += a[k] * wr[static_cast<size_t>(k)];
                    C[static_cast<size_t>(m) * ldc + n] = acc;
                }
            }
            return true;
        }

        MulMatParams mat_params;
        mat_params.A = A;
        mat_params.C = C;
        mat_params.M = M;
        mat_params.N = N;
        mat_params.K = K;
        mat_params.lda = lda;
        mat_params.ldb = ldb;
        mat_params.ldc = ldc;
        mat_params.n_batches = 1;
        mat_params.B_type = weight->type;
        mat_params.scales = nullptr;
        mat_params.n_weight_scales = 0;

        if (npu_matmul) {
            const int8_t* decoded = weight_cache.get_or_decode(*weight);
            if (!decoded) {
                std::cerr << "Error: failed to decode " << weight->name << "\n";
                return false;
            }
            mat_params.B = decoded;
            if (!attach_kquant_scales(mat_params, weight, weight_cache)) return false;
        } else {
            mat_params.B = weight->data;
        }
        return backend->mul_mat_q(mat_params) == Status::OK;
    };

    //====//
    // bench-layer: Validate one decoder layer on NPU vs CPU reference
    //====//
    if (params.bench_layer) {
        info << "GGNPU Layer Benchmark\n";
        info << "=====================\n\n";

        std::shared_ptr<Backend> npu_backend;
#ifdef GGNPU_HAS_NPU_BACKEND
        npu_backend = create_amd_xdna_backend(params.npu_device);
        if (!npu_backend || !npu_backend->is_available()) {
            std::cerr << "Error: NPU backend unavailable for bench-layer\n";
            return 1;
        }
#else
        std::cerr << "Error: NPU backend not compiled in. Build with -DGGNPU_NPU_BACKEND=ON\n";
        return 1;
#endif

        std::shared_ptr<Backend> cpu_backend = create_cpu_ref_backend();

        int layer_idx = params.bench_layer_num;
        if (layer_idx < 0 || layer_idx >= num_layers) {
            std::cerr << "Error: layer " << layer_idx << " out of range [0, "
                      << num_layers - 1 << "]\n";
            return 1;
        }

        info << "Model: " << params.model_path << "\n";
        info << "Layer: " << layer_idx << "\n";
        info << "Hidden: " << hidden_size << "\n";
        info << "FFN size: " << ffn_size << "\n\n";

        // Single test token activations (random but deterministic)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> test_input(static_cast<size_t>(hidden_size));
        for (size_t i = 0; i < test_input.size(); i++) {
            test_input[i] = dist(rng);
        }

        // RMSNorm reference (CPU f32 on bf16-quantized input — matches NPU DMA path)
        std::vector<float> bf16_input(test_input.size());
        bf16_roundtrip_vector(test_input.data(), bf16_input.data(), hidden_size);
        std::vector<float> normed(hidden_size);
        RmsNormParams rms_params;
        rms_params.input = bf16_input.data();
        rms_params.output = normed.data();
        rms_params.size = hidden_size;
        rms_params.eps = rms_eps;
        cpu_backend->rms_norm(rms_params);

        Status st;

        // Test 0a: RMSNorm — NPU vs CPU (Phase 4)
        info << "Testing RMSNorm (Phase 4)...\n";
        {
            std::vector<float> npu_normed(static_cast<size_t>(hidden_size));
            RmsNormParams npu_rms;
            npu_rms.input = test_input.data();
            npu_rms.output = npu_normed.data();
            npu_rms.size = hidden_size;
            npu_rms.eps = rms_eps;
            st = npu_backend->rms_norm(npu_rms);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU rms_norm failed: " << status_name(st) << "\n";
                return 1;
            }
            auto cmp = compare_vectors(normed.data(), npu_normed.data(), hidden_size, 0.01f);
            if (hidden_size != 2048) {
                info << "    Note: Llama hidden=2048 uses rmsnorm_2048_npu6.xclbin; hidden="
                          << hidden_size << " needs a matching shaped kernel\n";
            }
            if (!report_compare("RMSNorm", cmp, hidden_size, 0.01f, 0.05f, 0.01f)) return 1;
        }

        // Test 0a2: RMSNorm at non-pow2 / large hidden sizes (pad-to-pow2 path).
        // 1536 (Qwen/Gemma) pads to the existing 2048 kernel; 3072 (future 3B)
        // needs a 4096 kernel. Synthetic data, independent of the loaded model.
        info << "Testing RMSNorm extra sizes (Phase 6)...\n";
        for (int test_n : {1536, 3072}) {
            std::vector<float> in(static_cast<size_t>(test_n));
            for (int i = 0; i < test_n; i++)
                in[i] = std::sin(0.013f * static_cast<float>(i)) * 1.3f;

            std::vector<float> npu_out(static_cast<size_t>(test_n));
            RmsNormParams np;
            np.input = in.data(); np.output = npu_out.data();
            np.size = test_n; np.eps = rms_eps;
            Status rst = npu_backend->rms_norm(np);
            npu_backend->sync();
            if (rst != Status::OK) {
                info << "  RMSNorm N=" << test_n
                          << ": SKIPPED (status=" << status_name(rst) << ")\n";
                continue;
            }
            // bf16-aware CPU reference (matches NPU's bf16 marshaling).
            std::vector<float> qin(static_cast<size_t>(test_n));
            bf16_roundtrip_vector(in.data(), qin.data(), test_n);
            std::vector<float> cpu_out(static_cast<size_t>(test_n));
            RmsNormParams cp;
            cp.input = qin.data(); cp.output = cpu_out.data();
            cp.size = test_n; cp.eps = rms_eps;
            cpu_backend->rms_norm(cp);
            // Padded-pow2 path carries more bf16 noise than the exact 2048 path
            // (wider reduction + output correction factor), so use the production
            // validator's tolerance (atol 0.02, rtol 0.013) rather than the strict
            // 0.01 used for the native-size test.
            auto cmp = compare_vectors(cpu_out.data(), npu_out.data(), test_n, 0.02f);
            std::string lbl = "RMSNorm N=" + std::to_string(test_n);
            if (!report_compare(lbl.c_str(), cmp, test_n, 0.013f, 0.05f, 0.02f)) return 1;
        }

        // Test 0b–0e: attention matmuls — NPU vs CPU (Phase 4)
        const int kv_dim = num_kv_heads * head_dim;
        info << "Testing attention matmuls (Phase 4)...\n";
        {
            const TensorView* attn_q_w = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", layer_idx);
            const TensorView* attn_k_w = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", layer_idx);
            const TensorView* attn_v_w = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", layer_idx);
            const TensorView* attn_o_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", layer_idx);
            if (!attn_q_w || !attn_k_w || !attn_v_w || !attn_o_w) {
                std::cerr << "Error: attention weights not found for layer " << layer_idx << "\n";
                return 1;
            }
            if (!bench_quant_matmul("attn_q matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), attn_q_w, 1, hidden_size, hidden_size,
                    hidden_size, hidden_size, hidden_size)) return 1;
            if (!bench_quant_matmul("attn_k matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), attn_k_w, 1, kv_dim, hidden_size,
                    hidden_size, kv_dim, kv_dim)) return 1;
            if (!bench_quant_matmul("attn_v matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), attn_v_w, 1, kv_dim, hidden_size,
                    hidden_size, kv_dim, kv_dim)) return 1;
            if (!bench_quant_matmul("attn_output matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), attn_o_w, 1, hidden_size, hidden_size,
                    hidden_size, hidden_size, hidden_size)) return 1;
        }

        // Test 0e2: bf16 matmul — NPU vs CPU f32 (decomposed-attention building block).
        // Includes a QK-shaped case (M=ctx, N=1, K=head_dim) to exercise edge padding.
        info << "Testing bf16 matmul (Phase 6)...\n";
        {
            struct Shape { int M, N, K; const char* label; };
            std::vector<Shape> shapes = {
                {256, 256, 256, "256x256x256"},
                {64, 64, 128,   "64x64x128 (partial tile)"},
                {256, 1, 64,    "QK-shaped 256x1x64"},
            };
            for (auto s : shapes) {
                // Pseudo-random inputs (LCG) — avoid systematic cancellation that
                // would drive the reference near zero and make rel error meaningless.
                std::vector<float> A(static_cast<size_t>(s.M) * s.K);
                std::vector<float> B(static_cast<size_t>(s.K) * s.N);
                uint32_t st_a = 0x12345u, st_b = 0x6789au;
                auto nxt = [](uint32_t& st) {
                    st = st * 1103515245u + 12345u;
                    return (static_cast<float>((st >> 16) & 0x7fff) / 32768.0f - 0.5f);
                };
                for (size_t i = 0; i < A.size(); i++) A[i] = nxt(st_a);
                for (size_t i = 0; i < B.size(); i++) B[i] = nxt(st_b);

                std::vector<float> C_npu(static_cast<size_t>(s.M) * s.N, 0.0f);
                Status mst = npu_backend->matmul_bf16(A.data(), B.data(), C_npu.data(), s.M, s.N, s.K);
                npu_backend->sync();
                if (mst != Status::OK) {
                    info << "  bf16 matmul " << s.label << ": SKIPPED (status="
                              << status_name(mst) << ")\n";
                    continue;
                }
                // CPU reference: bf16-roundtrip A,B to match the NPU compute path.
                std::vector<float> Aq(A.size()), Bq(B.size());
                bf16_roundtrip_vector(A.data(), Aq.data(), static_cast<int>(A.size()));
                bf16_roundtrip_vector(B.data(), Bq.data(), static_cast<int>(B.size()));
                std::vector<float> C_cpu(static_cast<size_t>(s.M) * s.N, 0.0f);
                for (int m = 0; m < s.M; m++)
                    for (int n = 0; n < s.N; n++) {
                        float acc = 0.0f;
                        for (int k = 0; k < s.K; k++)
                            acc += Aq[static_cast<size_t>(m) * s.K + k] * Bq[static_cast<size_t>(k) * s.N + n];
                        C_cpu[static_cast<size_t>(m) * s.N + n] = acc;
                    }
                int n_elem = s.M * s.N;
                auto cmp = compare_vectors(C_cpu.data(), C_npu.data(), n_elem, 0.05f);
                std::string lbl = std::string("bf16 matmul ") + s.label;
                if (!report_compare(lbl.c_str(), cmp, n_elem, 0.03f, 0.05f, 0.05f)) return 1;
            }
        }

        // Test 0f: flash attention — NPU vs CPU (Phase 4, single head, ctx=1)
        info << "Testing flash attention (Phase 4)...\n";
        {
            std::vector<float> Q(static_cast<size_t>(head_dim), 0.1f);
            std::vector<float> K(static_cast<size_t>(head_dim), 0.2f);
            std::vector<float> V(static_cast<size_t>(head_dim), 0.3f);
            std::vector<float> cpu_fa(static_cast<size_t>(head_dim), 0.0f);
            std::vector<float> npu_fa(static_cast<size_t>(head_dim), 0.0f);

            AttnParams ap;
            ap.Q = Q.data(); ap.K = K.data(); ap.V = V.data();
            ap.n_head = 1; ap.head_dim = head_dim; ap.ctx_len = 1;
            ap.batch_size = 1; ap.freq_factors = nullptr;

            ap.output = cpu_fa.data();
            cpu_backend->flash_attn(ap);
            ap.output = npu_fa.data();
            st = npu_backend->flash_attn(ap);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU flash_attn failed: " << status_name(st) << "\n";
                return 1;
            }
            auto cmp = compare_vectors(cpu_fa.data(), npu_fa.data(), head_dim, 0.05f);
            if (!report_compare("flash_attn (1 head, ctx=1)", cmp, head_dim, 0.1f, 0.2f, 0.05f)) return 1;
        }

        // Test 0f2: flash attention — multi-head, ctx>1, causal (exercises full
        // QK -> softmax -> AV). NPU vs host f32; only meaningful when the NPU
        // decomposed path is on (GGNPU_NPU_ATTN), else both run the same host code.
        info << "Testing flash attention (multi-head, ctx>1)...\n";
        {
            const int nh = 4, cl = 40;
            const int64_t qpos = cl - 1;
            std::vector<float> Q(static_cast<size_t>(nh) * head_dim);
            std::vector<float> K(static_cast<size_t>(nh) * cl * head_dim);
            std::vector<float> V(static_cast<size_t>(nh) * cl * head_dim);
            uint32_t st_r = 0xc0ffeeu;
            auto rnd = [&]() {
                st_r = st_r * 1103515245u + 12345u;
                return static_cast<float>((st_r >> 16) & 0x7fff) / 32768.0f - 0.5f;
            };
            for (auto* v : {&Q, &K, &V}) for (auto& x : *v) x = rnd();

            std::vector<float> cpu_fa(static_cast<size_t>(nh) * head_dim, 0.0f);
            std::vector<float> npu_fa(static_cast<size_t>(nh) * head_dim, 0.0f);
            AttnParams ap;
            ap.Q = Q.data(); ap.K = K.data(); ap.V = V.data();
            ap.n_head = nh; ap.head_dim = head_dim; ap.ctx_len = cl;
            ap.query_pos = qpos; ap.batch_size = 1; ap.freq_factors = nullptr;

            ap.output = cpu_fa.data();
            cpu_backend->flash_attn(ap);
            ap.output = npu_fa.data();
            st = npu_backend->flash_attn(ap);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU flash_attn (multi-head) failed: " << status_name(st) << "\n";
                return 1;
            }
            int n_elem = nh * head_dim;
            auto cmp2 = compare_vectors(cpu_fa.data(), npu_fa.data(), n_elem, 0.05f);
            if (!report_compare("flash_attn (4 heads, ctx=40)", cmp2, n_elem, 0.03f, 0.1f, 0.05f)) return 1;
        }

        // Test 0g: RoPE — NPU kernel (Phase 6, pending pre-deinterleaved kernel design).
        info << "Testing RoPE (Phase 6)...\n";
        {
            std::vector<float> rope_in(static_cast<size_t>(head_dim), 0.5f);
            for (int i = 0; i < head_dim; i++) rope_in[i] = static_cast<float>(i) * 0.01f;

            std::vector<float> npu_rope(static_cast<size_t>(head_dim));
            std::memcpy(npu_rope.data(), rope_in.data(), head_dim * sizeof(float));
            RopeParams rp;
            rp.data = npu_rope.data();
            rp.n_dims = head_dim;
            rp.rope_dims = head_dim;
            rp.offset = 0;
            rp.freq_scale = 1.0f;
            rp.freq_base = 10000.0f;
            st = npu_backend->rope(rp);
            npu_backend->sync();
            if (st != Status::OK) {
                info << "  RoPE: SKIPPED (no rope xclbin — Phase 6 pending)\n\n";
            } else {
                // CPU reference for validation when xclbin exists
                std::vector<float> cpu_rope(static_cast<size_t>(head_dim));
                for (int i = 0; i < head_dim / 2; i++) {
                    float cos_val = std::cos(0.0f * 0.01f * (1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim))));
                    float sin_val = std::sin(0.0f * 0.01f * (1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim))));
                    cpu_rope[2 * i]     = rope_in[2 * i] * cos_val - rope_in[2 * i + 1] * sin_val;
                    cpu_rope[2 * i + 1] = rope_in[2 * i] * sin_val + rope_in[2 * i + 1] * cos_val;
                }
                auto cmp = compare_vectors(cpu_rope.data(), npu_rope.data(), head_dim, 0.05f);
                std::string rope_label = "RoPE (head_dim=" + std::to_string(head_dim) + ")";
                if (!report_compare(rope_label.c_str(), cmp, head_dim, 0.1f, 0.2f, 0.05f)) return 1;
            }

            // Test 0g2: batched RoPE — multi-head, nonzero offset (forces >1 launch
            // and exercises the head-packing math, which offset=0 identity hides).
            const int rb_heads = 10;            // > 8 heads/launch -> 2 launches
            const int64_t rb_off = 5;
            std::vector<float> rb_in(static_cast<size_t>(rb_heads) * head_dim);
            for (size_t i = 0; i < rb_in.size(); i++)
                rb_in[i] = std::sin(0.017f * static_cast<float>(i)) * 0.5f;

            std::vector<float> rb_npu = rb_in;
            RopeBatchedParams rbp;
            rbp.data = rb_npu.data();
            rbp.n_heads = rb_heads;
            rbp.n_dims = head_dim;
            rbp.rope_dims = head_dim;
            rbp.offset = rb_off;
            rbp.freq_scale = 1.0f;
            rbp.freq_base = 10000.0f;
            st = npu_backend->rope_batched(rbp);
            npu_backend->sync();
            if (st == Status::OK) {
                std::vector<float> rb_cpu = rb_in;
                for (int h = 0; h < rb_heads; h++) {
                    for (int i = 0; i < head_dim / 2; i++) {
                        float ang = static_cast<float>(rb_off) *
                            (1.0f / std::pow(10000.0f, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim)));
                        float c = std::cos(ang), s = std::sin(ang);
                        float* hd = rb_cpu.data() + static_cast<size_t>(h) * head_dim;
                        float v0 = rb_in[static_cast<size_t>(h) * head_dim + 2 * i];
                        float v1 = rb_in[static_cast<size_t>(h) * head_dim + 2 * i + 1];
                        hd[2 * i]     = v0 * c - v1 * s;
                        hd[2 * i + 1] = v0 * s + v1 * c;
                    }
                }
                int n = rb_heads * head_dim;
                auto cmp2 = compare_vectors(rb_cpu.data(), rb_npu.data(), n, 0.05f);
                if (!report_compare("RoPE batched (10 heads, off=5)", cmp2, n, 0.1f, 0.2f, 0.05f)) return 1;
            }
        }

        // Test 0c: SiLU — NPU vs CPU (Phase 4)
        info << "Testing SiLU (Phase 4)...\n";
        {
            std::vector<float> silu_in(static_cast<size_t>(ffn_size));
            for (int i = 0; i < ffn_size; i++) silu_in[i] = test_input[i % hidden_size];

            std::vector<float> cpu_silu(static_cast<size_t>(ffn_size));
            SiluParams cpu_p;
            cpu_p.input = silu_in.data();
            cpu_p.output = cpu_silu.data();
            cpu_p.size = ffn_size;
            cpu_backend->silu(cpu_p);

            std::vector<float> npu_silu(static_cast<size_t>(ffn_size));
            SiluParams npu_p;
            npu_p.input = silu_in.data();
            npu_p.output = npu_silu.data();
            npu_p.size = ffn_size;
            st = npu_backend->silu(npu_p);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU silu failed: " << status_name(st) << "\n";
                return 1;
            }

            auto cmp = compare_vectors(cpu_silu.data(), npu_silu.data(), ffn_size, 0.05f);
            if (!report_compare("SiLU", cmp, ffn_size, 0.05f, 0.1f, 0.05f)) return 1;
        }

        info << "Testing FFN gate matmul (Phase 3 E2E gate)...\n";
        {
            const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer_idx);
            if (!bench_quant_matmul("FFN gate matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), ffn_gate_w, 1, ffn_size, hidden_size,
                    hidden_size, ffn_size, ffn_size)) return 1;
        }

        // Test 2: FFN up matmul
        {
            const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer_idx);
            if (!bench_quant_matmul("FFN up matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), ffn_up_w, 1, ffn_size, hidden_size,
                    hidden_size, ffn_size, ffn_size)) return 1;
        }

        // Test 3: FFN down matmul (silu(gate) * up) -> hidden_size
        {
            const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer_idx);
            const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer_idx);
            const TensorView* ffn_down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", layer_idx);

            if (!ffn_gate_w || !ffn_up_w || !ffn_down_w) {
                std::cerr << "Error: FFN weights not found for layer " << layer_idx << "\n";
                return 1;
            }

            // Compute gate and up on NPU
            const int8_t* decoded_gate = weight_cache.get_or_decode(*ffn_gate_w);
            const int8_t* decoded_up = weight_cache.get_or_decode(*ffn_up_w);

            std::vector<float> gate_out(static_cast<size_t>(ffn_size), 0.0f);
            std::vector<float> up_out(static_cast<size_t>(ffn_size), 0.0f);

            MulMatParams gate_params;
            gate_params.A = normed.data();
            gate_params.B = decoded_gate;
            gate_params.C = gate_out.data();
            gate_params.M = 1;
            gate_params.N = ffn_size;
            gate_params.K = hidden_size;
            gate_params.lda = hidden_size;
            gate_params.ldb = ffn_size;
            gate_params.ldc = ffn_size;
            gate_params.n_batches = 1;
            gate_params.B_type = ffn_gate_w->type;
            if (!attach_kquant_scales(gate_params, ffn_gate_w, weight_cache)) return 1;
            npu_backend->mul_mat_q(gate_params);

            MulMatParams up_params;
            up_params.A = normed.data();
            up_params.B = decoded_up;
            up_params.C = up_out.data();
            up_params.M = 1;
            up_params.N = ffn_size;
            up_params.K = hidden_size;
            up_params.lda = hidden_size;
            up_params.ldb = ffn_size;
            up_params.ldc = ffn_size;
            up_params.n_batches = 1;
            up_params.B_type = ffn_up_w->type;
            if (!attach_kquant_scales(up_params, ffn_up_w, weight_cache)) return 1;
            npu_backend->mul_mat_q(up_params);
            npu_backend->sync();

            // SwiGLU: silu(gate) * up
            std::vector<float> swiglu(static_cast<size_t>(ffn_size));
            for (int i = 0; i < ffn_size; i++) {
                float gate_silu = gate_out[i] / (1.0f + std::exp(-gate_out[i]));
                swiglu[i] = gate_silu * up_out[i];
            }

            // FFN down matmul on NPU
            const int8_t* decoded_down = weight_cache.get_or_decode(*ffn_down_w);

            std::vector<float> npu_output(static_cast<size_t>(hidden_size), 0.0f);
            MulMatParams down_params;
            down_params.A = swiglu.data();
            down_params.B = decoded_down;
            down_params.C = npu_output.data();
            down_params.M = 1;
            down_params.N = hidden_size;
            down_params.K = ffn_size;
            down_params.lda = ffn_size;
            down_params.ldb = hidden_size;
            down_params.ldc = hidden_size;
            down_params.n_batches = 1;
            down_params.B_type = ffn_down_w->type;
            if (!attach_kquant_scales(down_params, ffn_down_w, weight_cache)) return 1;

            st = npu_backend->mul_mat_q(down_params);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU matmul failed: " << status_name(st) << "\n";
                return 1;
            }

            // CPU reference for FFN down
            std::vector<float> cpu_output(static_cast<size_t>(hidden_size), 0.0f);
            MulMatParams cpu_down_params;
            cpu_down_params.A = swiglu.data();
            cpu_down_params.B = ffn_down_w->data;
            cpu_down_params.C = cpu_output.data();
            cpu_down_params.M = 1;
            cpu_down_params.N = hidden_size;
            cpu_down_params.K = ffn_size;
            cpu_down_params.lda = ffn_size;
            cpu_down_params.ldb = hidden_size;
            cpu_down_params.ldc = hidden_size;
            cpu_down_params.n_batches = 1;
            cpu_down_params.B_type = ffn_down_w->type;
            cpu_backend->mul_mat_q(cpu_down_params);

            float max_diff = 0.0f;
            float max_cpu = 0.0f;
            int mismatches = 0;
            for (int i = 0; i < hidden_size; i++) {
                float diff = std::fabs(cpu_output[i] - npu_output[i]);
                if (diff > max_diff) max_diff = diff;
                if (std::fabs(cpu_output[i]) > max_cpu) max_cpu = std::fabs(cpu_output[i]);
                if (diff > 2.0f) mismatches++;
            }

            float rel_error = max_cpu > 0.0f ? max_diff / max_cpu : max_diff;
            info << "  FFN down matmul (SwiGLU, 1 x " << ffn_size << " x " << hidden_size << "):\n";
            info << "    Type: " << ggml_type_name(ffn_down_w->type) << "\n";
            info << "    Max absolute diff: " << max_diff << "\n";
            info << "    Relative error: " << rel_error << "\n";
            info << "    Mismatches (>2.0): " << mismatches << " / " << hidden_size << "\n";

            if (rel_error < 0.1f && mismatches < hidden_size / 10) {
                info << "    Result: PASS\n\n";
            } else {
                info << "    Result: FAIL\n\n";
                return 1;
            }
        }

        // Test 4: full decoder layer forward — CPU vs NPU (Phase 4)
        info << "Testing full layer forward (Phase 4)...\n";
        {
            auto run_layer = [&](Backend* backend, bool use_npu_weights,
                                 const float* in, float* out) -> bool {
                const TensorView* attn_q_w = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", layer_idx);
                const TensorView* attn_k_w = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", layer_idx);
                const TensorView* attn_v_w = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", layer_idx);
                const TensorView* attn_o_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", layer_idx);
                const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer_idx);
                const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer_idx);
                const TensorView* ffn_down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", layer_idx);
                if (!attn_q_w || !attn_k_w || !attn_v_w || !attn_o_w ||
                    !ffn_gate_w || !ffn_up_w || !ffn_down_w) return false;

                std::vector<float> hidden(static_cast<size_t>(hidden_size));
                std::memcpy(hidden.data(), in, hidden_size * sizeof(float));

                auto do_matmul = [&](const float* A, const TensorView* w, float* C,
                                     int M, int N, int K, int lda, int ldb, int ldc) -> bool {
                    MulMatParams mp;
                    mp.A = A; mp.C = C;
                    mp.M = M; mp.N = N; mp.K = K;
                    mp.lda = lda; mp.ldb = ldb; mp.ldc = ldc;
                    mp.n_batches = 1;
                    if (use_npu_weights) {
                        const int8_t* dec = weight_cache.get_or_decode(*w);
                        if (!dec) return false;
                        mp.B = dec;
                        mp.B_type = w->type;
                        if (!attach_kquant_scales(mp, w, weight_cache)) return false;
                    } else {
                        mp.B = w->data;
                        mp.B_type = w->type;
                    }
                    return backend->mul_mat_q(mp) == Status::OK;
                };

                const TensorView* attn_norm_w =
                    find_tensor_pattern(model, "blk.{layer}.attn_norm.weight", layer_idx);
                const TensorView* ffn_norm_w =
                    find_tensor_pattern(model, "blk.{layer}.ffn_norm.weight", layer_idx);

                // Attention branch
                std::vector<float> norm(static_cast<size_t>(hidden_size));
                RmsNormParams rp;
                rp.input = hidden.data(); rp.output = norm.data();
                rp.size = hidden_size; rp.eps = rms_eps;
                rp.weight = get_float_ptr(attn_norm_w);
                if (backend->rms_norm(rp) != Status::OK) return false;

                std::vector<float> q(hidden_size), k(kv_dim), v(kv_dim);
                std::vector<float> q_r(hidden_size), k_r(kv_dim);
                if (!do_matmul(norm.data(), attn_q_w, q.data(), 1, hidden_size, hidden_size,
                               hidden_size, hidden_size, hidden_size)) return false;
                if (!do_matmul(norm.data(), attn_k_w, k.data(), 1, kv_dim, hidden_size,
                               hidden_size, kv_dim, kv_dim)) return false;
                if (!do_matmul(norm.data(), attn_v_w, v.data(), 1, kv_dim, hidden_size,
                               hidden_size, kv_dim, kv_dim)) return false;
                backend->sync();

                apply_rope(q_r.data(), q.data(), num_heads, 0, head_dim);
                apply_rope(k_r.data(), k.data(), num_kv_heads, 0, head_dim);

                std::vector<float> attn_out(hidden_size, 0.0f);
                for (int h = 0; h < num_heads; h++) {
                    int kv_h = h / (num_heads / num_kv_heads);
                    AttnParams ap;
                    ap.Q = q_r.data() + h * head_dim;
                    ap.K = k_r.data() + kv_h * head_dim;
                    ap.V = v.data() + kv_h * head_dim;
                    ap.output = attn_out.data() + h * head_dim;
                    ap.n_head = 1; ap.head_dim = head_dim; ap.ctx_len = 1;
                    ap.batch_size = 1; ap.freq_factors = nullptr;
                    if (backend->flash_attn(ap) != Status::OK) return false;
                }
                backend->sync();

                std::vector<float> attn_proj(hidden_size, 0.0f);
                if (!do_matmul(attn_out.data(), attn_o_w, attn_proj.data(),
                               1, hidden_size, hidden_size,
                               hidden_size, hidden_size, hidden_size)) return false;
                backend->sync();
                for (int i = 0; i < hidden_size; i++) hidden[i] += attn_proj[i];

                // FFN branch
                rp.input = hidden.data();
                rp.output = norm.data();
                rp.weight = get_float_ptr(ffn_norm_w);
                if (backend->rms_norm(rp) != Status::OK) return false;

                std::vector<float> gate(ffn_size), up(ffn_size), swiglu(ffn_size);
                if (!do_matmul(norm.data(), ffn_gate_w, gate.data(), 1, ffn_size, hidden_size,
                               hidden_size, ffn_size, ffn_size)) return false;
                if (!do_matmul(norm.data(), ffn_up_w, up.data(), 1, ffn_size, hidden_size,
                               hidden_size, ffn_size, ffn_size)) return false;
                backend->sync();

                SiluParams sp;
                sp.input = gate.data(); sp.output = swiglu.data(); sp.size = ffn_size;
                if (backend->silu(sp) != Status::OK) return false;
                for (int i = 0; i < ffn_size; i++) swiglu[i] *= up[i];

                std::vector<float> ffn_out(hidden_size, 0.0f);
                if (!do_matmul(swiglu.data(), ffn_down_w, ffn_out.data(),
                               1, hidden_size, ffn_size,
                               ffn_size, hidden_size, hidden_size)) return false;
                backend->sync();
                for (int i = 0; i < hidden_size; i++) hidden[i] += ffn_out[i];

                std::memcpy(out, hidden.data(), hidden_size * sizeof(float));
                return true;
            };

            std::vector<float> cpu_layer(hidden_size), npu_layer(hidden_size);
            if (!run_layer(cpu_backend.get(), false, test_input.data(), cpu_layer.data())) {
                std::cerr << "Error: CPU layer forward failed\n";
                return 1;
            }
            if (!run_layer(npu_backend.get(), true, test_input.data(), npu_layer.data())) {
                std::cerr << "Error: NPU layer forward failed\n";
                return 1;
            }
            npu_backend->sync();

            auto cmp = compare_vectors(cpu_layer.data(), npu_layer.data(), hidden_size, 2.0f);
            // bf16 rmsnorm + INT8 matmul stack: looser gate vs CPU ref for full layer
            if (!report_compare("full layer forward", cmp, hidden_size, 0.75f, 1.0f, 2.0f)) return 1;
        }

        info << "All layer tests PASSED.\n";
        model.unload();
        return 0;
    }

    const TensorView* tok_embd = find_tensor(model, "token_embd.weight");
    const TensorView* output_w = find_tensor(model, "output.weight");
    const TensorView* logits_w = output_w ? output_w : tok_embd;

    if (!tok_embd) {
        std::cerr << "Error: token_embd.weight not available\n";
        model.unload();
        return 1;
    }

    model.kv_cache().reset();

    // ---- Gemma 4 (gemma4) forward path — Phase G2 skeleton ----
    // Gemma 4 E2B: per-layer head_dim (256 local / 512 global), num_kv_heads=1,
    // KV sharing (layers without attn_k/attn_v reuse an earlier same-type layer),
    // QK-norm, 4 sandwich RMSNorms with the (1+w) weight convention, GeGLU FFN,
    // NeoX RoPE with per-type freq base, sqrt(hidden) embedding scale, and final
    // logit soft-capping. PLE (per-layer embeddings, inp_gate/proj/per_layer_*),
    // post_norm, layer_output_scale and sliding-window masking are deferred to
    // G3/G4 — the core decoder still runs (degraded) without them.
    const bool is_gemma = (model.gguf().arch() == "gemma4");
    struct GemmaLayer { int head_dim; bool is_swa; float rope_base; int ffn; int kv_src; };
    std::vector<GemmaLayer> gplan;
    std::vector<std::vector<float>> gk_store, gv_store;  // per-owning-layer KV: [pos*head_dim]
    float gemma_embd_scale = 1.0f;
    const float gemma_logit_softcap =
        static_cast<float>(model.gguf().get_float("gemma4.final_logit_softcapping", 0.0));
    if (is_gemma) {
        const double rb_global = model.gguf().get_float("gemma4.rope.freq_base", 1e6);
        const double rb_swa = model.gguf().get_float("gemma4.rope.freq_base_swa", 1e4);
        auto ffn_arr = model.gguf().get_int_array(model.gguf().arch_key("feed_forward_length"));
        auto swa_arr = model.gguf().get_int_array(model.gguf().arch_key("attention.sliding_window_pattern"));
        gplan.resize(num_layers);
        int last_local_kv = -1, last_global_kv = -1;
        for (int L = 0; L < num_layers; L++) {
            const TensorView* qw = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", L);
            const TensorView* kw = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", L);
            GemmaLayer gl;
            gl.head_dim = (qw && num_heads > 0) ? static_cast<int>(qw->dims[1] / num_heads) : head_dim;
            gl.is_swa = (L < static_cast<int>(swa_arr.size())) ? (swa_arr[L] != 0) : true;
            gl.rope_base = gl.is_swa ? static_cast<float>(rb_swa) : static_cast<float>(rb_global);
            gl.ffn = (L < static_cast<int>(ffn_arr.size())) ? static_cast<int>(ffn_arr[L]) : ffn_size;
            if (kw) {
                gl.kv_src = L;
                if (gl.is_swa) last_local_kv = L; else last_global_kv = L;
            } else {
                gl.kv_src = gl.is_swa ? last_local_kv : last_global_kv;
            }
            gplan[L] = gl;
        }
        gk_store.resize(num_layers);
        gv_store.resize(num_layers);
        gemma_embd_scale = std::sqrt(static_cast<float>(hidden_size));
        info << "gemma4 forward: " << num_layers << " layers, "
                  << "num_heads=" << num_heads << ", softcap=" << gemma_logit_softcap << "\n";
    }
    // Per-Layer Embeddings (PLE) — Phase G3. Two paths per token combine into a
    // 256-dim auxiliary signal per layer, injected at the end of each block.
    const int gemma_ple_dim =
        static_cast<int>(model.gguf().get_int("gemma4.embedding_length_per_layer_input", 256));
    // Sliding-window size for local (SWA) layers: a query at position p attends
    // only to keys k with p - k < n_swa (STANDARD SWA). 0 disables windowing.
    const int64_t gemma_swa_window =
        is_gemma ? model.gguf().get_int("gemma4.attention.sliding_window", 0) : 0;
    const TensorView* ple_tok_embd = is_gemma ? find_tensor(model, "per_layer_token_embd.weight") : nullptr;
    const TensorView* ple_model_proj = is_gemma ? find_tensor(model, "per_layer_model_proj.weight") : nullptr;
    const TensorView* ple_proj_norm = is_gemma ? find_tensor(model, "per_layer_proj_norm.weight") : nullptr;

    // ---- LFM2 (lfm2) forward path ----
    // LiquidAI LFM2.5: a hybrid that interleaves gated ShortConv blocks with GQA
    // attention blocks; both are followed by a SwiGLU FFN. Per-layer type is read
    // from tensor presence (recurrent/conv layers carry shortconv.* and no attn_*;
    // this matches llama.cpp's is_recr = n_head_kv(il)==0). Norms are plain
    // RMSNorm, RoPE is NeoX, and — an LFM2 quirk — the final norm is stored as
    // `token_embd_norm` while the output projection is tied to the token embedding.
    // qwen3: like qwen2 but with an explicit head_dim (key_length), per-head
    // QK-norm, tied output, and NEOX (split-half) RoPE. Handled inline in the
    // generic transformer path via q_dim, the QK-norm block, and the rope branch.
    const bool is_qwen3 = (model.gguf().arch() == "qwen3");

    const bool is_lfm2 = (model.gguf().arch() == "lfm2");
    const int lfm2_d_conv =
        static_cast<int>(model.gguf().get_int("lfm2.shortconv.l_cache", 3));
    const float lfm2_rope_base =
        static_cast<float>(model.gguf().get_float("lfm2.rope.freq_base", 1e6));
    std::vector<char> lfm2_is_conv(num_layers, 0);              // 1 = ShortConv layer
    std::vector<int>  lfm2_kv_heads(num_layers, 0);             // n_head_kv (attn layers)
    // Per-layer recurrent/attention state, persistent across token steps.
    std::vector<std::vector<float>> lconv_state(num_layers);    // [(d_conv-1)*hidden] per conv layer
    std::vector<std::vector<float>> lk_store(num_layers), lv_store(num_layers);  // [pos*kvh*head_dim] per attn layer
    if (is_lfm2) {
        for (int L = 0; L < num_layers; L++) {
            if (find_tensor_pattern(model, "blk.{layer}.shortconv.in_proj.weight", L)) {
                lfm2_is_conv[L] = 1;
                lconv_state[L].assign(static_cast<size_t>(lfm2_d_conv - 1) * hidden_size, 0.0f);
            } else {
                const TensorView* kw = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", L);
                lfm2_kv_heads[L] = (kw && head_dim > 0) ? static_cast<int>(kw->dims[1] / head_dim) : num_heads;
            }
        }
        info << "lfm2 forward: " << num_layers << " layers, d_conv=" << lfm2_d_conv
                  << ", rope_base=" << lfm2_rope_base << "\n";
    }

    // ---- Qwen3.5 (qwen35) forward path ----
    // Qwen3-Next-style hybrid: a 3:1 interleave of gated-DeltaNet linear-attention
    // (SSM) blocks and gated full-attention blocks. SSM blocks carry ssm_*, fused
    // attn_qkv[hidden->2*key_dim+value_dim] and a separate attn_gate; full-attn
    // blocks (every `full_attention_interval`-th) carry attn_q/k/v/output with the
    // output gate fused into the tail half of attn_q. Both blocks use attn_norm
    // (pre-mixer) + post_attention_norm (pre-FFN) + SwiGLU FFN. Attention head_dim
    // is key_length(256) with partial NEOX RoPE over rope.dimension_count(64).
    // qwen35moe (Qwen3.6-35B-A3B) shares qwen35's Gated-DeltaNet + full-attn mixer
    // verbatim (identical ssm_*/attn_* tensor names and hparams); it only swaps the
    // dense SwiGLU FFN for a 256-expert top-8 MoE plus a sigmoid-gated shared expert.
    // So route it through the same forward, keyed on the arch's own metadata prefix.
    const std::string q35_arch = model.gguf().arch();
    const bool is_qwen35moe = (q35_arch == "qwen35moe");
    const bool is_qwen35 = (q35_arch == "qwen35") || is_qwen35moe;
    const int q35_head_dim = static_cast<int>(model.gguf().get_int(q35_arch + ".attention.key_length", 256));
    const int q35_rope_dim = static_cast<int>(model.gguf().get_int(q35_arch + ".rope.dimension_count", 64));
    const float q35_rope_base = static_cast<float>(model.gguf().get_float(q35_arch + ".rope.freq_base", 1e7));
    const int q35_conv_k   = static_cast<int>(model.gguf().get_int(q35_arch + ".ssm.conv_kernel", 4));
    const int q35_ssm_hd   = static_cast<int>(model.gguf().get_int(q35_arch + ".ssm.state_size", 128));   // head dim (k==v)
    const int q35_value_dim = static_cast<int>(model.gguf().get_int(q35_arch + ".ssm.inner_size", 4096)); // v_heads*hd
    const int q35_k_heads  = static_cast<int>(model.gguf().get_int(q35_arch + ".ssm.group_count", 16));   // key/query heads
    // MoE hparams (qwen35moe only; 0 for the dense qwen35 4B/9B).
    const int q35_n_expert      = static_cast<int>(model.gguf().get_int(q35_arch + ".expert_count", 0));
    const int q35_n_expert_used = static_cast<int>(model.gguf().get_int(q35_arch + ".expert_used_count", 0));
    const int q35_expert_ff     = static_cast<int>(model.gguf().get_int(q35_arch + ".expert_feed_forward_length", 0));
    // Trailing Multi-Token-Prediction (nextn) layers carry blk.N.nextn.* tensors and
    // are NOT part of the autoregressive backbone — running them as decoder layers
    // NaNs out the hidden state. Skip them during standard greedy decode.
    const int q35_nextn = static_cast<int>(model.gguf().get_int(q35_arch + ".nextn_predict_layers", 0));
    const int q35_n_layers = std::min(max_layers_override, num_layers - q35_nextn);
    const int q35_v_heads  = (q35_ssm_hd > 0) ? q35_value_dim / q35_ssm_hd : 32;                     // value heads
    const int q35_key_dim  = q35_k_heads * q35_ssm_hd;                                                // 2048
    const int q35_conv_dim = 2 * q35_key_dim + q35_value_dim;                                         // 8192
    std::vector<char> q35_is_ssm(num_layers, 0);
    std::vector<std::vector<float>> q35_conv_state(num_layers);   // [(conv_k-1)*conv_dim] per SSM layer
    std::vector<std::vector<float>> q35_rec_state(num_layers);    // [v_heads*hd*hd] per SSM layer
    std::vector<std::vector<float>> q35_k_store(num_layers), q35_v_store(num_layers);  // full-attn KV
    if (is_qwen35) {
        for (int L = 0; L < num_layers; L++) {
            if (find_tensor_pattern(model, "blk.{layer}.ssm_conv1d.weight", L)) {
                q35_is_ssm[L] = 1;
                q35_conv_state[L].assign(static_cast<size_t>(q35_conv_k - 1) * q35_conv_dim, 0.0f);
                q35_rec_state[L].assign(static_cast<size_t>(q35_v_heads) * q35_ssm_hd * q35_ssm_hd, 0.0f);
            }
        }
        info << "qwen35 forward: " << num_layers << " layers, head_dim=" << q35_head_dim
                  << ", rope_dim=" << q35_rope_dim << ", v_heads=" << q35_v_heads
                  << ", k_heads=" << q35_k_heads << ", ssm_hd=" << q35_ssm_hd << "\n";
    }

    // Per-op timing accumulator for the gemma4 path (-v/--verbose), mirroring
    // the Llama path's step_timings. Declared here (stable address) rather
    // than freshly per-iteration so the g_rmsnorm/g_rope/g_gelu lambdas below
    // (defined once, outside the decode loop) can capture it by reference;
    // reset at the top of each loop iteration instead of re-declared.
    InferenceTimings step_timings;

    // Gemma4 RMSNorm: raw weight (this arch stores the full scale, no (1+w)
    // offset — verified against the reference graph). Sizes 256/512/1536, all
    // NPU-dispatched via the existing `rms_norm` helper (any N via
    // next_pow2 padding; weight applied host-side inside backend->rms_norm()).
    auto g_rmsnorm = [&](float* out, const float* in, int n, const float* w, float eps) -> bool {
        std::unique_ptr<ScopedTimer> t;
        if (params.verbose) t = std::make_unique<ScopedTimer>(step_timings.rms_norm_ms);
        // Full-fp32 diagnostic: exact host RMSNorm over exactly n elements (no NPU
        // pow2 pad, no int8). See GGNPU_FP32_SSM above.
        if (fp32_ssm) {
            host_rmsnorm(out, in, n, w, eps);
            return true;
        }
        if (!rms_norm(out, in, 1, n, eps, w)) {
            model.unload();
            return false;
        }
        return true;
    };
    // Gemma NeoX RoPE (rotate halves i, i+hd/2), per-layer freq base, over full
    // head_dim. NPU-dispatched via rope_batched() (neox_split_half=true); the
    // underlying kernel is a pure elementwise vector-add — only host
    // (de)interleave indexing differs from Llama's adjacent-pair layout.
    auto g_rope = [&](float* v, int n_heads, int hd, int64_t pos, float base) -> bool {
        std::unique_ptr<ScopedTimer> t;
        if (params.verbose) t = std::make_unique<ScopedTimer>(step_timings.rope_ms);
        RopeBatchedParams rp;
        rp.data = v;
        rp.n_heads = n_heads;
        rp.n_dims = hd;
        rp.rope_dims = hd;
        rp.offset = pos;
        rp.freq_scale = 1.0f;
        rp.freq_base = base;
        rp.neox_split_half = true;
        if (backend->rope_batched(rp) != Status::OK) {
            std::cerr << "Error: gemma RoPE failed\n";
            model.unload();
            return false;
        }
        return true;
    };
    // Gemma GeGLU activation (tanh-approx GELU on the gate). NPU-dispatched
    // via backend->gelu() in place; the caller does the cheap elementwise
    // multiply by the FFN up-projection (or PLE per-layer gate) afterward.
    // Timed into silu_ms (shared "FFN activation" bucket with the Llama path).
    auto g_gelu = [&](float* buf, int n) -> bool {
        std::unique_ptr<ScopedTimer> t;
        if (params.verbose) t = std::make_unique<ScopedTimer>(step_timings.silu_ms);
        GeluParams gp{buf, buf, n};
        if (backend->gelu(gp) != Status::OK) {
            std::cerr << "Error: gemma GELU failed\n";
            model.unload();
            return false;
        }
        return true;
    };

    // Decode-style forward: one token per step (correct KV + causal attention).
    if (!params.bench_logits) {
        info << "Generating: ";
    } else {
        info << "bench-logits: " << input_tokens.size() << " prompt tokens\n";
    }

    // Prompt-lookup speculative decode (opt-in GGNPU_SPEC_DECODE): draft the next
    // few tokens by matching recent n-grams in the sequence, then verify them all
    // in one batched M=(k+1) forward (the same path prefill batching uses). Because
    // the decode matmul kernel is fixed at M=16, verifying up to 15 draft tokens
    // costs the same device time as one M=1 step, so any accepted draft is a near-
    // free extra token. Greedy only (argmax verification == sequential greedy);
    // restricted to the generic transformer path (lfm2/qwen35 are sequential
    // recurrences, gemma4 has a separate forward — both deferred, like prefill batch).
    const bool spec_decode = std::getenv("GGNPU_SPEC_DECODE") != nullptr &&
                             !is_gemma && !is_lfm2 && !is_qwen35 &&
                             params.temp == 0.0f && logits_w != nullptr;
    const int spec_ngram_max = 3;
    const int spec_ngram_min = 1;
    int spec_k = 8;  // draft length; k+1 must stay <= small-M kernel tile (16)
    if (const char* e = std::getenv("GGNPU_SPEC_K")) {
        spec_k = std::max(1, std::min(15, std::atoi(e)));
    }
    // Full token history (prompt + generated) for the n-gram drafter.
    std::vector<int> seq = input_tokens;
    const bool spec_debug = std::getenv("GGNPU_SPEC_DEBUG") != nullptr;
    long spec_steps = 0, spec_proposed = 0, spec_accepted = 0;

    bool first_token = true;
    while (generated < params.max_tokens || params.bench_logits) {
        step_timings = InferenceTimings();
        step_timings.token_steps = 1;

        // Prefill batching: the generic transformer path and gemma4 (not lfm2/
        // qwen35, whose ShortConv/DeltaNet blocks are sequential state recurrences)
        // process all remaining prompt tokens in one M=n_tokens forward pass
        // instead of one M=1 pass per token. This uses the 256-M matmul kernel
        // (full array) for the prompt's Q/K/V/O/FFN (+ gemma PLE) projections — far
        // fewer, fatter launches — cutting time-to-first-token. Decode (post-prompt)
        // stays M=1. Chunked at kPrefillChunk so M stays within one matmul M-tile.
        constexpr int kPrefillChunk = 256;
        const bool batch_prefill = !is_lfm2 && !is_qwen35 &&
                                   std::getenv("GGNPU_NO_PREFILL_BATCH") == nullptr;
        std::vector<int> batch_toks;
        bool do_sample = false;
        int n_draft = 0;  // # of speculative draft tokens appended after last_sampled
        if (prompt_idx < input_tokens.size()) {
            const int remaining = static_cast<int>(input_tokens.size()) - static_cast<int>(prompt_idx);
            const int chunk = batch_prefill ? std::min(remaining, kPrefillChunk) : 1;
            for (int i = 0; i < chunk; i++) batch_toks.push_back(input_tokens[prompt_idx + i]);
            prompt_idx += chunk;
            do_sample = (prompt_idx >= input_tokens.size());
        } else if (params.bench_logits) {
            break;
        } else {
            batch_toks.push_back(last_sampled);
            do_sample = true;
            // Speculative: append an n-gram draft so the block is verified in one
            // batched forward. seq already ends with last_sampled.
            if (spec_decode) {
                std::vector<int> draft = lookup_draft(seq, spec_ngram_max, spec_ngram_min, spec_k);
                for (int dt : draft) batch_toks.push_back(dt);
                n_draft = static_cast<int>(draft.size());
            }
        }
        const int n_tokens = static_cast<int>(batch_toks.size());
        step_timings.token_steps = n_tokens;

        inp_embd.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        inp_norm.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        q_proj.assign(static_cast<size_t>(n_tokens) * q_dim, 0.0f);
        k_proj.assign(static_cast<size_t>(n_tokens) * num_kv_heads * head_dim, 0.0f);
        v_proj.assign(static_cast<size_t>(n_tokens) * num_kv_heads * head_dim, 0.0f);
        k_rope.assign(static_cast<size_t>(n_tokens) * num_kv_heads * head_dim, 0.0f);
        q_rope.assign(static_cast<size_t>(n_tokens) * q_dim, 0.0f);
        attn_output.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        ffn_gate.assign(static_cast<size_t>(n_tokens) * ffn_size, 0.0f);
        ffn_up.assign(static_cast<size_t>(n_tokens) * ffn_size, 0.0f);
        ffn_down.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        ffn_silu.assign(static_cast<size_t>(n_tokens) * ffn_size, 0.0f);

        {
            ScopedTimer t(step_timings.embed_ms);
            // Embed each batched token into its own row (n_tokens==1 for the
            // single-token branches, so this matches the old single-row embed).
            for (int bt = 0; bt < n_tokens; bt++)
                dequant_tensor_row(tok_embd, batch_toks[bt],
                                   inp_embd.data() + static_cast<size_t>(bt) * hidden_size,
                                   hidden_size);
        }

        // ---- Gemma 4 forward ----
        if (is_gemma) {
            // Scale every batched token's embedding row (n_tokens==1 in decode).
            for (int bt = 0; bt < n_tokens; bt++) {
                float* row = inp_embd.data() + static_cast<size_t>(bt) * hidden_size;
                for (int i = 0; i < hidden_size; i++) row[i] *= gemma_embd_scale;
            }

            // Per-Layer Embeddings (PLE, G3): build one ple_dim vector per layer,
            // per batched token.
            //   token-identity: per_layer_token_embd[tok] * sqrt(ple_dim)
            //   context-aware:  RMSNorm(per_layer_model_proj(embed) / sqrt(hidden))
            //   combined:       (token + context) / sqrt(2)
            const int ple_dim = gemma_ple_dim;
            const int ple_total = num_layers * ple_dim;
            // pli is [n_tokens][ple_total]; pgate/inj are per-token working buffers.
            std::vector<float> pli(static_cast<size_t>(n_tokens) * ple_total, 0.0f);
            std::vector<float> pgate, inj;
            {
                // Context-aware projection batched over all rows: [n_tokens][ple_total].
                std::vector<float> ple_ctx(static_cast<size_t>(n_tokens) * ple_total, 0.0f);
                mul_mat_weight(inp_embd.data(), ple_model_proj, ple_ctx.data(),
                               n_tokens, ple_total, hidden_size, hidden_size, ple_total, ple_total,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();
                const float ctx_scale = 1.0f / std::sqrt(static_cast<float>(hidden_size));
                const float tok_scale = std::sqrt(static_cast<float>(ple_dim));
                const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
                const float* plpn = get_float_ptr(ple_proj_norm);
                std::vector<float> ple_tok(static_cast<size_t>(ple_total), 0.0f);
                for (int bt = 0; bt < n_tokens; bt++) {
                    dequant_tensor_row(ple_tok_embd, batch_toks[bt], ple_tok.data(), ple_total);
                    float* ctx_row = ple_ctx.data() + static_cast<size_t>(bt) * ple_total;
                    float* pli_row = pli.data() + static_cast<size_t>(bt) * ple_total;
                    for (int L = 0; L < num_layers; L++) {
                        float* slice = ctx_row + static_cast<size_t>(L) * ple_dim;
                        for (int i = 0; i < ple_dim; i++) slice[i] *= ctx_scale;
                        if (!g_rmsnorm(slice, slice, ple_dim, plpn, rms_eps)) return 1;
                    }
                    for (int i = 0; i < ple_total; i++)
                        pli_row[i] = (ple_tok[i] * tok_scale + ctx_row[i]) * inv_sqrt2;
                }
            }

            // Per-token working buffers (sized per layer; qdim/ffn vary by layer).
            std::vector<float> xn, q, kbuf, vbuf, attn_in, attn_o;
            std::vector<float> gate, up, act, dn, kexp, vexp;

            for (int L = 0; L < max_layers_override; L++) {
                const GemmaLayer& gl = gplan[L];
                const int hd = gl.head_dim;
                const int qdim = num_heads * hd;

                const TensorView* attn_norm_w = find_tensor_pattern(model, "blk.{layer}.attn_norm.weight", L);
                const TensorView* q_w = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", L);
                const TensorView* qn_w = find_tensor_pattern(model, "blk.{layer}.attn_q_norm.weight", L);
                const TensorView* o_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", L);
                const TensorView* postattn_w = find_tensor_pattern(model, "blk.{layer}.post_attention_norm.weight", L);
                const TensorView* ffnnorm_w = find_tensor_pattern(model, "blk.{layer}.ffn_norm.weight", L);
                const TensorView* gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", L);
                const TensorView* up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", L);
                const TensorView* down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", L);
                const TensorView* postffw_w = find_tensor_pattern(model, "blk.{layer}.post_ffw_norm.weight", L);

                // Attention input norm (per row).
                xn.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
                for (int bt = 0; bt < n_tokens; bt++)
                    if (!g_rmsnorm(xn.data() + static_cast<size_t>(bt) * hidden_size,
                                   inp_embd.data() + static_cast<size_t>(bt) * hidden_size,
                                   hidden_size, get_float_ptr(attn_norm_w), rms_eps)) return 1;

                // Q projection (M=n_tokens) -> per-token per-head QK-norm -> RoPE.
                q.assign(static_cast<size_t>(n_tokens) * qdim, 0.0f);
                mul_mat_weight(xn.data(), q_w, q.data(), n_tokens, qdim, hidden_size, hidden_size, qdim, qdim,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();

                // Debug (GGNPU_GEMMA_DEBUG): validate the Q4_0 NPU matmul against a
                // CPU dequant reference on the real layer-0 first-token activations.
                if (L == 0 && pos == 0 && std::getenv("GGNPU_GEMMA_DEBUG")) {
                    std::vector<float> ref(qdim, 0.0f);
                    std::vector<float> wrow(hidden_size);
                    constexpr int QK4_0 = 32; constexpr size_t Q4_0_BLK = 18;
                    const size_t bpr = (hidden_size + QK4_0 - 1) / QK4_0;
                    for (int n = 0; n < qdim; n++) {
                        const uint8_t* rd = static_cast<const uint8_t*>(q_w->data) + static_cast<size_t>(n) * bpr * Q4_0_BLK;
                        std::vector<float> blk(QK4_0);
                        for (size_t b = 0; b < bpr; b++) {
                            dequant_q4_0_block(rd + b * Q4_0_BLK, blk.data());
                            for (int j = 0; j < QK4_0; j++) { size_t k = b*QK4_0+j; if (k < (size_t)hidden_size) wrow[k] = blk[j]; }
                        }
                        double acc = 0; for (int k = 0; k < hidden_size; k++) acc += (double)xn[k] * wrow[k];
                        ref[n] = (float)acc;
                    }
                    double num = 0, den = 0; float amax = 0;
                    for (int n = 0; n < qdim; n++) { double d = q[n]-ref[n]; num += d*d; den += (double)ref[n]*ref[n]; amax = std::max(amax, std::fabs(q[n]-ref[n])); }
                    std::cerr << "[gemma dbg] attn_q L0: rel_l2=" << std::sqrt(num/(den+1e-12))
                              << " max_abs_err=" << amax << " q[0..3]=" << q[0] << "," << q[1] << "," << q[2]
                              << " ref[0..3]=" << ref[0] << "," << ref[1] << "," << ref[2] << "\n";
                    const float* anw = get_float_ptr(attn_norm_w);
                    const TensorView* los = find_tensor_pattern(model, "blk.{layer}.layer_output_scale.weight", L);
                    auto l2v = [](const float* p, int n){ double s=0; for(int i=0;i<n;i++) s+=(double)p[i]*p[i]; return std::sqrt(s); };
                    std::cerr << "[gemma dbg] attn_norm.w[0..3]=" << anw[0] << "," << anw[1] << "," << anw[2]
                              << "  layer_output_scale=" << (los ? get_float_ptr(los)[0] : -999.0f)
                              << "  |embed_scaled|=" << l2v(inp_embd.data(), hidden_size)
                              << "  |xn|=" << l2v(xn.data(), hidden_size)
                              << "  |pli_L0|=" << l2v(pli.data(), ple_dim) << "\n";
                }

                // Gemma4 attention scale is 1.0 (f_attention_scale); the host
                // flash_attn bakes in 1/sqrt(head_dim), so pre-scale Q by sqrt(hd)
                // to cancel it back to an effective scale of 1.0.
                {
                    const float qs = std::sqrt(static_cast<float>(hd));
                    for (int bt = 0; bt < n_tokens; bt++) {
                        float* qrow = q.data() + static_cast<size_t>(bt) * qdim;
                        for (int h = 0; h < num_heads; h++)
                            if (!g_rmsnorm(qrow + h * hd, qrow + h * hd, hd, get_float_ptr(qn_w), rms_eps)) return 1;
                        // Each batched token carries its own absolute position (pos + bt).
                        if (!g_rope(qrow, num_heads, hd, pos + bt, gl.rope_base)) return 1;
                        for (int i = 0; i < qdim; i++) qrow[i] *= qs;
                    }
                }

                // K/V: compute + append if this layer owns them, else reuse the
                // source layer's store. All n_tokens positions are appended (in
                // order) before attention, so each query sees only its own prefix.
                if (gl.kv_src == L) {
                    const TensorView* k_w = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", L);
                    const TensorView* v_w = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", L);
                    const TensorView* kn_w = find_tensor_pattern(model, "blk.{layer}.attn_k_norm.weight", L);
                    kbuf.assign(static_cast<size_t>(n_tokens) * hd, 0.0f);
                    vbuf.assign(static_cast<size_t>(n_tokens) * hd, 0.0f);
                    mul_mat_weight(xn.data(), k_w, kbuf.data(), n_tokens, hd, hidden_size, hidden_size, hd, hd,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    mul_mat_weight(xn.data(), v_w, vbuf.data(), n_tokens, hd, hidden_size, hidden_size, hd, hd,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    backend->sync();
                    for (int bt = 0; bt < n_tokens; bt++) {
                        float* kr = kbuf.data() + static_cast<size_t>(bt) * hd;
                        float* vr = vbuf.data() + static_cast<size_t>(bt) * hd;
                        // K: RMSNorm with attn_k_norm weight, then RoPE (own position).
                        if (!g_rmsnorm(kr, kr, hd, get_float_ptr(kn_w), rms_eps)) return 1;
                        if (!g_rope(kr, 1, hd, pos + bt, gl.rope_base)) return 1;
                        // V: plain RMSNorm (no weight, no RoPE) — Vcur = ggml_rms_norm(Vcur, eps).
                        if (!g_rmsnorm(vr, vr, hd, nullptr, rms_eps)) return 1;
                        gk_store[L].insert(gk_store[L].end(), kr, kr + hd);
                        gv_store[L].insert(gv_store[L].end(), vr, vr + hd);
                    }
                }

                const int src = gl.kv_src;
                const std::vector<float>& ks = gk_store[src];
                const std::vector<float>& vs = gv_store[src];

                // Causal attention: one call per query token over its own (possibly
                // sliding-window) causal prefix. Query at position p=pos+bt attends
                // keys [k0..p]; SWA local layers window k0 = p+1-n_swa, global layers
                // keep k0=0. The KV store now holds this batch's later positions too,
                // but each query slices only up to its own p, so they stay masked.
                attn_in.assign(static_cast<size_t>(n_tokens) * qdim, 0.0f);
                for (int bt = 0; bt < n_tokens; bt++) {
                    const int64_t p = pos + bt;
                    int64_t k0 = 0;
                    if (gl.is_swa && gemma_swa_window > 0)
                        k0 = std::max<int64_t>(0, (p + 1) - gemma_swa_window);
                    const int64_t ctx_len = (p + 1) - k0;
                    // Broadcast the single KV head to all query heads (num_kv_heads=1).
                    kexp.assign(static_cast<size_t>(num_heads) * ctx_len * hd, 0.0f);
                    vexp.assign(static_cast<size_t>(num_heads) * ctx_len * hd, 0.0f);
                    {
                        std::unique_ptr<ScopedTimer> t;
                        if (params.verbose) t = std::make_unique<ScopedTimer>(step_timings.kv_expand_ms);
                        for (int h = 0; h < num_heads; h++) {
                            for (int64_t j = 0; j < ctx_len; j++) {
                                std::memcpy(kexp.data() + (static_cast<size_t>(h) * ctx_len + j) * hd,
                                            ks.data() + static_cast<size_t>(k0 + j) * hd, hd * sizeof(float));
                                std::memcpy(vexp.data() + (static_cast<size_t>(h) * ctx_len + j) * hd,
                                            vs.data() + static_cast<size_t>(k0 + j) * hd, hd * sizeof(float));
                            }
                        }
                    }
                    AttnParams fa;
                    fa.Q = q.data() + static_cast<size_t>(bt) * qdim;
                    fa.K = kexp.data();
                    fa.V = vexp.data();
                    fa.output = attn_in.data() + static_cast<size_t>(bt) * qdim;
                    fa.batch_size = 1;
                    fa.n_head = num_heads;
                    fa.head_dim = hd;
                    fa.ctx_len = ctx_len;
                    fa.query_pos = ctx_len - 1;  // query is the last (newest) key in the slice
                    fa.freq_factors = nullptr;
                    {
                        std::unique_ptr<ScopedTimer> t;
                        if (params.verbose) t = std::make_unique<ScopedTimer>(step_timings.flash_attn_ms);
                        if (backend->flash_attn(fa) != Status::OK) {
                            std::cerr << "Error: gemma flash_attn failed at layer " << L << "\n";
                            model.unload();
                            return 1;
                        }
                    }
                }

                // Output projection (M=n_tokens) -> post-attention norm -> residual.
                attn_o.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
                mul_mat_weight(attn_in.data(), o_w, attn_o.data(), n_tokens, hidden_size, qdim, qdim, hidden_size, hidden_size,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();
                for (int bt = 0; bt < n_tokens; bt++) {
                    float* ao = attn_o.data() + static_cast<size_t>(bt) * hidden_size;
                    if (!g_rmsnorm(ao, ao, hidden_size, get_float_ptr(postattn_w), rms_eps)) return 1;
                    float* h = inp_embd.data() + static_cast<size_t>(bt) * hidden_size;
                    for (int i = 0; i < hidden_size; i++) h[i] += ao[i];
                }

                // FFN: pre-norm -> GeGLU -> down -> post-FFN norm -> residual.
                const int ffn = gl.ffn;
                xn.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
                for (int bt = 0; bt < n_tokens; bt++)
                    if (!g_rmsnorm(xn.data() + static_cast<size_t>(bt) * hidden_size,
                                   inp_embd.data() + static_cast<size_t>(bt) * hidden_size,
                                   hidden_size, get_float_ptr(ffnnorm_w), rms_eps)) return 1;
                gate.assign(static_cast<size_t>(n_tokens) * ffn, 0.0f);
                up.assign(static_cast<size_t>(n_tokens) * ffn, 0.0f);
                mul_mat_weight(xn.data(), gate_w, gate.data(), n_tokens, ffn, hidden_size, hidden_size, ffn, ffn,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                mul_mat_weight(xn.data(), up_w, up.data(), n_tokens, ffn, hidden_size, hidden_size, ffn, ffn,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();
                act.assign(static_cast<size_t>(n_tokens) * ffn, 0.0f);
                for (int bt = 0; bt < n_tokens; bt++) {
                    float* g = gate.data() + static_cast<size_t>(bt) * ffn;
                    float* u = up.data() + static_cast<size_t>(bt) * ffn;
                    float* a = act.data() + static_cast<size_t>(bt) * ffn;
                    if (!g_gelu(g, ffn)) return 1;
                    for (int i = 0; i < ffn; i++) a[i] = g[i] * u[i];
                }
                dn.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
                mul_mat_weight(act.data(), down_w, dn.data(), n_tokens, hidden_size, ffn, ffn, hidden_size, hidden_size,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();
                for (int bt = 0; bt < n_tokens; bt++) {
                    float* d = dn.data() + static_cast<size_t>(bt) * hidden_size;
                    if (!g_rmsnorm(d, d, hidden_size, get_float_ptr(postffw_w), rms_eps)) return 1;
                    float* h = inp_embd.data() + static_cast<size_t>(bt) * hidden_size;
                    for (int i = 0; i < hidden_size; i++) h[i] += d[i];
                }

                // Per-Layer Embedding injection (G3), per token:
                //   inj = post_norm( proj( gelu(inp_gate(h)) * pli[bt][L] ) ); h += inj
                const TensorView* inpgate_w = find_tensor_pattern(model, "blk.{layer}.inp_gate.weight", L);
                const TensorView* proj_w = find_tensor_pattern(model, "blk.{layer}.proj.weight", L);
                const TensorView* postnorm_w = find_tensor_pattern(model, "blk.{layer}.post_norm.weight", L);
                if (inpgate_w && proj_w && postnorm_w) {
                    pgate.assign(static_cast<size_t>(n_tokens) * ple_dim, 0.0f);
                    mul_mat_weight(inp_embd.data(), inpgate_w, pgate.data(),
                                   n_tokens, ple_dim, hidden_size, hidden_size, ple_dim, ple_dim,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    backend->sync();
                    for (int bt = 0; bt < n_tokens; bt++) {
                        float* pg = pgate.data() + static_cast<size_t>(bt) * ple_dim;
                        const float* pli_L = pli.data() + static_cast<size_t>(bt) * ple_total
                                             + static_cast<size_t>(L) * ple_dim;
                        if (!g_gelu(pg, ple_dim)) return 1;
                        for (int i = 0; i < ple_dim; i++) pg[i] = pg[i] * pli_L[i];
                    }
                    inj.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
                    mul_mat_weight(pgate.data(), proj_w, inj.data(),
                                   n_tokens, hidden_size, ple_dim, ple_dim, hidden_size, hidden_size,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    backend->sync();
                    for (int bt = 0; bt < n_tokens; bt++) {
                        float* ij = inj.data() + static_cast<size_t>(bt) * hidden_size;
                        if (!g_rmsnorm(ij, ij, hidden_size, get_float_ptr(postnorm_w), rms_eps)) return 1;
                        float* h = inp_embd.data() + static_cast<size_t>(bt) * hidden_size;
                        for (int i = 0; i < hidden_size; i++) h[i] += ij[i];
                    }
                }

                // Per-layer output scale: multiply the full hidden state (cur *= out_scale).
                // Magnitude is renormalized by the next layer's RMSNorm.
                const TensorView* outscale_w = find_tensor_pattern(model, "blk.{layer}.layer_output_scale.weight", L);
                if (outscale_w) {
                    const float os = get_float_ptr(outscale_w)[0];
                    for (int bt = 0; bt < n_tokens; bt++) {
                        float* h = inp_embd.data() + static_cast<size_t>(bt) * hidden_size;
                        for (int i = 0; i < hidden_size; i++) h[i] *= os;
                    }
                }
            }

            // Final norm into inp_norm (per row; logits read the last token's row).
            const TensorView* out_norm_w = find_tensor(model, "output_norm.weight");
            for (int bt = 0; bt < n_tokens; bt++)
                if (!g_rmsnorm(inp_norm.data() + static_cast<size_t>(bt) * hidden_size,
                               inp_embd.data() + static_cast<size_t>(bt) * hidden_size,
                               hidden_size, get_float_ptr(out_norm_w), rms_eps)) return 1;
        }

        // ---- LFM2 forward ----
        // Single token per step; inp_embd carries the running hidden state and the
        // final normalized state is written into inp_norm for the logits below.
        else if (is_lfm2) {
            std::vector<float> xn(hidden_size);
            std::vector<float> bcx(static_cast<size_t>(3) * hidden_size), yconv(hidden_size);
            std::vector<float> q, kbuf, vbuf, kexp, vexp, attn_in, attn_o(hidden_size);
            std::vector<float> gate(ffn_size), up(ffn_size), act(ffn_size), dn(hidden_size);

            for (int L = 0; L < max_layers_override; L++) {
                const TensorView* attn_norm_w = find_tensor_pattern(model, "blk.{layer}.attn_norm.weight", L);
                const TensorView* ffn_norm_w  = find_tensor_pattern(model, "blk.{layer}.ffn_norm.weight", L);

                // operator_norm: pre-block RMSNorm (plain weight).
                if (!g_rmsnorm(xn.data(), inp_embd.data(), hidden_size, get_float_ptr(attn_norm_w), rms_eps)) return 1;

                if (lfm2_is_conv[L]) {
                    // --- Gated short convolution ---
                    // bcx = in_proj(xn) -> split B, C, x (each `hidden`); bx = B*x;
                    // depthwise causal conv over the last d_conv taps; y = C*conv(bx);
                    // out = out_proj(y).
                    const TensorView* inproj_w  = find_tensor_pattern(model, "blk.{layer}.shortconv.in_proj.weight", L);
                    const TensorView* outproj_w = find_tensor_pattern(model, "blk.{layer}.shortconv.out_proj.weight", L);
                    const TensorView* conv_w    = find_tensor_pattern(model, "blk.{layer}.shortconv.conv.weight", L);
                    std::fill(bcx.begin(), bcx.end(), 0.0f);
                    mul_mat_weight(xn.data(), inproj_w, bcx.data(), 1, 3 * hidden_size, hidden_size,
                                   hidden_size, 3 * hidden_size, 3 * hidden_size,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    backend->sync();
                    const float* b = bcx.data();                    // [0, hidden)
                    const float* c = bcx.data() + hidden_size;      // [hidden, 2*hidden)
                    const float* x = bcx.data() + 2 * hidden_size;  // [2*hidden, 3*hidden)

                    // conv.weight is F32, tap-contiguous {d_conv, hidden}: w[tap,ch] = cw[ch*d_conv + tap].
                    // State holds the previous (d_conv-1) bx vectors, oldest first.
                    const float* cw = get_float_ptr(conv_w);
                    std::vector<float>& st = lconv_state[L];
                    const int dcm1 = lfm2_d_conv - 1;
                    for (int ch = 0; ch < hidden_size; ch++) {
                        const float bx = b[ch] * x[ch];
                        float acc = cw[ch * lfm2_d_conv + dcm1] * bx;  // newest tap = current token
                        for (int k = 0; k < dcm1; k++)
                            acc += cw[ch * lfm2_d_conv + k] * st[static_cast<size_t>(k) * hidden_size + ch];
                        yconv[ch] = c[ch] * acc;
                        // roll the conv state: drop oldest, append current bx
                        for (int k = 0; k < dcm1 - 1; k++)
                            st[static_cast<size_t>(k) * hidden_size + ch] =
                                st[static_cast<size_t>(k + 1) * hidden_size + ch];
                        st[static_cast<size_t>(dcm1 - 1) * hidden_size + ch] = bx;
                    }
                    std::fill(attn_o.begin(), attn_o.end(), 0.0f);
                    mul_mat_weight(yconv.data(), outproj_w, attn_o.data(), 1, hidden_size, hidden_size,
                                   hidden_size, hidden_size, hidden_size,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    backend->sync();
                    for (int i = 0; i < hidden_size; i++) inp_embd[i] += attn_o[i];
                } else {
                    // --- GQA attention ---
                    const int kvh   = lfm2_kv_heads[L];
                    const int qdim  = num_heads * head_dim;
                    const int kvdim = kvh * head_dim;
                    const TensorView* q_w  = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", L);
                    const TensorView* k_w  = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", L);
                    const TensorView* v_w  = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", L);
                    const TensorView* o_w  = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", L);
                    const TensorView* qn_w = find_tensor_pattern(model, "blk.{layer}.attn_q_norm.weight", L);
                    const TensorView* kn_w = find_tensor_pattern(model, "blk.{layer}.attn_k_norm.weight", L);

                    q.assign(qdim, 0.0f);
                    kbuf.assign(kvdim, 0.0f);
                    vbuf.assign(kvdim, 0.0f);
                    mul_mat_weight(xn.data(), q_w, q.data(), 1, qdim, hidden_size, hidden_size, qdim, qdim,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    mul_mat_weight(xn.data(), k_w, kbuf.data(), 1, kvdim, hidden_size, hidden_size, kvdim, kvdim,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    mul_mat_weight(xn.data(), v_w, vbuf.data(), 1, kvdim, hidden_size, hidden_size, kvdim, kvdim,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    backend->sync();

                    // Per-head QK-norm (RMSNorm over head_dim) then NeoX RoPE.
                    for (int h = 0; h < num_heads; h++)
                        if (!g_rmsnorm(q.data() + h * head_dim, q.data() + h * head_dim, head_dim,
                                       get_float_ptr(qn_w), rms_eps)) return 1;
                    for (int h = 0; h < kvh; h++)
                        if (!g_rmsnorm(kbuf.data() + h * head_dim, kbuf.data() + h * head_dim, head_dim,
                                       get_float_ptr(kn_w), rms_eps)) return 1;
                    if (!g_rope(q.data(), num_heads, head_dim, pos, lfm2_rope_base)) return 1;
                    if (!g_rope(kbuf.data(), kvh, head_dim, pos, lfm2_rope_base)) return 1;

                    lk_store[L].insert(lk_store[L].end(), kbuf.begin(), kbuf.end());
                    lv_store[L].insert(lv_store[L].end(), vbuf.begin(), vbuf.end());
                    const int64_t ctx_len = pos + 1;
                    const int group = num_heads / kvh;

                    // Expand K/V from kvh to num_heads (GQA: query head h -> kv head h/group).
                    kexp.assign(static_cast<size_t>(num_heads) * ctx_len * head_dim, 0.0f);
                    vexp.assign(static_cast<size_t>(num_heads) * ctx_len * head_dim, 0.0f);
                    {
                        std::unique_ptr<ScopedTimer> t;
                        if (params.verbose) t = std::make_unique<ScopedTimer>(step_timings.kv_expand_ms);
                        for (int h = 0; h < num_heads; h++) {
                            const int kh = h / group;
                            for (int64_t j = 0; j < ctx_len; j++) {
                                std::memcpy(kexp.data() + (static_cast<size_t>(h) * ctx_len + j) * head_dim,
                                            lk_store[L].data() + (static_cast<size_t>(j) * kvh + kh) * head_dim,
                                            head_dim * sizeof(float));
                                std::memcpy(vexp.data() + (static_cast<size_t>(h) * ctx_len + j) * head_dim,
                                            lv_store[L].data() + (static_cast<size_t>(j) * kvh + kh) * head_dim,
                                            head_dim * sizeof(float));
                            }
                        }
                    }

                    attn_in.assign(qdim, 0.0f);
                    AttnParams fa;
                    fa.Q = q.data();
                    fa.K = kexp.data();
                    fa.V = vexp.data();
                    fa.output = attn_in.data();
                    fa.batch_size = 1;
                    fa.n_head = num_heads;
                    fa.head_dim = head_dim;
                    fa.ctx_len = ctx_len;
                    fa.query_pos = pos;
                    fa.freq_factors = nullptr;
                    {
                        std::unique_ptr<ScopedTimer> t;
                        if (params.verbose) t = std::make_unique<ScopedTimer>(step_timings.flash_attn_ms);
                        if (backend->flash_attn(fa) != Status::OK) {
                            std::cerr << "Error: lfm2 flash_attn failed at layer " << L << "\n";
                            model.unload();
                            return 1;
                        }
                    }

                    std::fill(attn_o.begin(), attn_o.end(), 0.0f);
                    mul_mat_weight(attn_in.data(), o_w, attn_o.data(), 1, hidden_size, qdim,
                                   qdim, hidden_size, hidden_size,
                                   params.verbose ? &step_timings.matmul_ms : nullptr);
                    backend->sync();
                    for (int i = 0; i < hidden_size; i++) inp_embd[i] += attn_o[i];
                }

                // --- SwiGLU FFN (both block types) ---
                if (!g_rmsnorm(xn.data(), inp_embd.data(), hidden_size, get_float_ptr(ffn_norm_w), rms_eps)) return 1;
                const TensorView* gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", L);
                const TensorView* up_w   = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", L);
                const TensorView* down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", L);
                std::fill(gate.begin(), gate.end(), 0.0f);
                std::fill(up.begin(), up.end(), 0.0f);
                mul_mat_weight(xn.data(), gate_w, gate.data(), 1, ffn_size, hidden_size,
                               hidden_size, ffn_size, ffn_size,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                mul_mat_weight(xn.data(), up_w, up.data(), 1, ffn_size, hidden_size,
                               hidden_size, ffn_size, ffn_size,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();
                {
                    std::unique_ptr<ScopedTimer> t;
                    if (params.verbose) t = std::make_unique<ScopedTimer>(step_timings.silu_ms);
                    SiluParams sp; sp.input = gate.data(); sp.output = act.data(); sp.size = ffn_size;
                    if (backend->silu(sp) != Status::OK) {
                        std::cerr << "Error: lfm2 silu failed at layer " << L << "\n";
                        model.unload();
                        return 1;
                    }
                }
                for (int i = 0; i < ffn_size; i++) act[i] *= up[i];
                std::fill(dn.begin(), dn.end(), 0.0f);
                mul_mat_weight(act.data(), down_w, dn.data(), 1, hidden_size, ffn_size,
                               ffn_size, hidden_size, hidden_size,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();
                for (int i = 0; i < hidden_size; i++) inp_embd[i] += dn[i];
            }

            // Final norm (LFM2 stores it as token_embd_norm) into inp_norm.
            const TensorView* out_norm_w = find_tensor(model, "token_embd_norm.weight");
            if (!g_rmsnorm(inp_norm.data(), inp_embd.data(), hidden_size, get_float_ptr(out_norm_w), rms_eps)) return 1;
        }

        // ---- Qwen3.5 (qwen35) forward ----
        else if (is_qwen35) {
            const int Hd    = q35_head_dim;          // attention head dim (256)
            const int nQ    = num_heads;             // 16 query heads
            const int nKV   = num_kv_heads;          // 4 kv heads
            const int qdim  = nQ * Hd;               // 4096
            const int kvdim = nKV * Hd;              // 1024
            const int sk    = q35_ssm_hd;            // 128 SSM head dim
            const int nVH   = q35_v_heads;           // 32 value heads
            const int nKH   = q35_k_heads;           // 16 key heads
            double* mm = params.verbose ? &step_timings.matmul_ms : nullptr;

            std::vector<float> xn(hidden_size);
            // SSM scratch
            std::vector<float> mixed(q35_conv_dim), zgate(q35_value_dim), conv(q35_conv_dim);
            std::vector<float> araw(nVH), braw(nVH), qn(q35_key_dim), kn(q35_key_dim), o_all(q35_value_dim);
            std::vector<float> ssm_o(hidden_size);
            // Full-attn scratch
            std::vector<float> qg(2 * qdim), kbuf(kvdim), vbuf(kvdim), kexp, vexp, attn_in(qdim), attn_o(hidden_size);
            // FFN scratch
            std::vector<float> gate(ffn_size), up(ffn_size), act(ffn_size), dn(hidden_size);

            const bool q35dbg = std::getenv("GGNPU_DBG") && pos == 0;
            auto vnorm = [](const std::vector<float>& v, int n) {
                double s = 0; for (int i = 0; i < n; i++) s += (double)v[i] * v[i]; return std::sqrt(s / n);
            };
            // Compare against llama.cpp eval-callback row-0 values (first prompt token,
            // empty conv/recurrent state ⇒ our pos==0 == llama token-0). Prints [0,1,2 ... n-3,n-2,n-1].
            auto dbg6 = [&](const char* label, const float* v, int n) {
                if (!q35dbg) return;
                double s = 0; for (int i = 0; i < n; i++) s += v[i];
                std::cerr << "  [q35cmp] " << label << " sum=" << s << " = [" << v[0] << ", " << v[1]
                          << ", " << v[2] << ", ..., " << v[n-3] << ", " << v[n-2] << ", " << v[n-1] << "]\n";
            };
            if (q35dbg) std::cerr << "[dbg] embed rms=" << vnorm(inp_embd, hidden_size) << "\n";
            for (int L = 0; L < q35_n_layers; L++) {
                const TensorView* attn_norm_w = find_tensor_pattern(model, "blk.{layer}.attn_norm.weight", L);
                const TensorView* post_norm_w = find_tensor_pattern(model, "blk.{layer}.post_attention_norm.weight", L);
                if (!g_rmsnorm(xn.data(), inp_embd.data(), hidden_size, get_float_ptr(attn_norm_w), rms_eps)) return 1;
                if (q35dbg && L < 5) std::cerr << "[dbg L" << L << (q35_is_ssm[L] ? " SSM" : " ATT")
                    << "] xn_rms=" << vnorm(xn, hidden_size);

                if (q35_is_ssm[L]) {
                    // ===== Gated DeltaNet linear attention =====
                    const TensorView* qkv_w   = find_tensor_pattern(model, "blk.{layer}.attn_qkv.weight", L);
                    const TensorView* gate_w  = find_tensor_pattern(model, "blk.{layer}.attn_gate.weight", L);
                    const TensorView* alpha_w = find_tensor_pattern(model, "blk.{layer}.ssm_alpha.weight", L);
                    const TensorView* beta_w  = find_tensor_pattern(model, "blk.{layer}.ssm_beta.weight", L);
                    const TensorView* conv_w  = find_tensor_pattern(model, "blk.{layer}.ssm_conv1d.weight", L);
                    const TensorView* a_w     = find_tensor_pattern(model, "blk.{layer}.ssm_a", L);
                    const TensorView* dt_w    = find_tensor_pattern(model, "blk.{layer}.ssm_dt.bias", L);
                    const TensorView* snorm_w = find_tensor_pattern(model, "blk.{layer}.ssm_norm.weight", L);
                    const TensorView* out_w   = find_tensor_pattern(model, "blk.{layer}.ssm_out.weight", L);

                    std::fill(mixed.begin(), mixed.end(), 0.0f);
                    std::fill(zgate.begin(), zgate.end(), 0.0f);
                    std::fill(araw.begin(), araw.end(), 0.0f);
                    std::fill(braw.begin(), braw.end(), 0.0f);
                    mul_mat_weight(xn.data(), qkv_w, mixed.data(), 1, q35_conv_dim, hidden_size,
                                   hidden_size, q35_conv_dim, q35_conv_dim, mm);
                    mul_mat_weight(xn.data(), gate_w, zgate.data(), 1, q35_value_dim, hidden_size,
                                   hidden_size, q35_value_dim, q35_value_dim, mm);
                    mul_mat_weight(xn.data(), alpha_w, araw.data(), 1, nVH, hidden_size,
                                   hidden_size, nVH, nVH, mm);
                    mul_mat_weight(xn.data(), beta_w, braw.data(), 1, nVH, hidden_size,
                                   hidden_size, nVH, nVH, mm);
                    backend->sync();

                    // Causal depthwise conv1d over conv_dim channels (kernel conv_k) + SiLU.
                    // conv weight is tap-contiguous {conv_k, conv_dim}: cw[ch*conv_k + tap];
                    // newest tap = current token. State holds the previous conv_k-1 inputs.
                    const float* cw = get_float_ptr(conv_w);
                    std::vector<float>& cst = q35_conv_state[L];
                    const int ckm1 = q35_conv_k - 1;
                    for (int ch = 0; ch < q35_conv_dim; ch++) {
                        float acc = cw[ch * q35_conv_k + ckm1] * mixed[ch];
                        for (int t = 0; t < ckm1; t++)
                            acc += cw[ch * q35_conv_k + t] * cst[static_cast<size_t>(t) * q35_conv_dim + ch];
                        conv[ch] = acc / (1.0f + std::exp(-acc));  // SiLU
                        for (int t = 0; t < ckm1 - 1; t++)
                            cst[static_cast<size_t>(t) * q35_conv_dim + ch] =
                                cst[static_cast<size_t>(t + 1) * q35_conv_dim + ch];
                        if (ckm1 > 0) cst[static_cast<size_t>(ckm1 - 1) * q35_conv_dim + ch] = mixed[ch];
                    }
                    const float* qc = conv.data();                    // q: nKH heads x sk
                    const float* kc = conv.data() + q35_key_dim;      // k: nKH heads x sk
                    const float* vc = conv.data() + 2 * q35_key_dim;  // v: nVH heads x sk
                    if (L == 0) {
                        dbg6("xn", xn.data(), hidden_size);
                        dbg6("mixed(qkv)", mixed.data(), q35_conv_dim);
                        dbg6("conv_q", qc, q35_key_dim);
                        dbg6("conv_v", vc, q35_value_dim);
                        dbg6("araw", araw.data(), nVH);
                        dbg6("braw", braw.data(), nVH);
                        dbg6("zgate", zgate.data(), q35_value_dim);
                    }

                    // L2-normalize q,k per key head (over sk).
                    for (int h = 0; h < nKH; h++) {
                        const float* qh = qc + h * sk;
                        const float* kh = kc + h * sk;
                        float sq = 0.0f, s2 = 0.0f;
                        for (int i = 0; i < sk; i++) { sq += qh[i] * qh[i]; s2 += kh[i] * kh[i]; }
                        // q additionally carries the delta-net 1/sqrt(head_k_dim) scale
                        // (llama.cpp build_delta_net: `q = ggml_scale(q, 1/sqrtf(S_k))`).
                        // It cancels in the per-head gated RMSNorm, but we apply it to
                        // match the reference exactly.
                        const float rq = (1.0f / std::sqrt(sq + 1e-6f)) / std::sqrt((float)sk);
                        const float rk = 1.0f / std::sqrt(s2 + 1e-6f);
                        for (int i = 0; i < sk; i++) { qn[h * sk + i] = qh[i] * rq; kn[h * sk + i] = kh[i] * rk; }
                    }
                    if (L == 0) { dbg6("qn(q_predelta)", qn.data(), q35_key_dim);
                                  dbg6("kn(k_predelta)", kn.data(), q35_key_dim); }

                    // Per value-head gated delta rule recurrence. S[v_head] is [sk_k x sk_v].
                    const float* A_log = get_float_ptr(a_w);
                    const float* dt_b  = get_float_ptr(dt_w);
                    std::vector<float>& S = q35_rec_state[L];
                    for (int h = 0; h < nVH; h++) {
                        // Value head h reads key/query head (h % nKH). llama.cpp expands
                        // q/k from nKH to nVH with ggml_repeat_4d, whose semantics are
                        // modulo tiling (dst head = src head % nKH) — NOT block grouping
                        // (h / vgroup). The two agree only for head 0, so a block mapping
                        // silently corrupts every value head after the first.
                        const int kh = h % nKH;
                        const float* qh = qn.data() + kh * sk;
                        const float* khp = kn.data() + kh * sk;
                        const float* vh = vc + h * sk;
                        const float beta = 1.0f / (1.0f + std::exp(-braw[h]));           // sigmoid
                        const float a = araw[h] + dt_b[h];
                        const float sp = (a > 20.0f) ? a : std::log1p(std::exp(a));      // softplus
                        // ssm_a is stored pre-transformed as -exp(A_log) (llama.cpp applies
                        // no exp/neg): log-decay = softplus(alpha+dt)*ssm_a, decay = exp(that).
                        const float g = std::exp(A_log[h] * sp);                         // decay in (0,1]
                        if (std::getenv("GGNPU_DBG") && L == 0 && h < 3 && pos == 0)
                            std::cerr << "[dbg L0 h" << h << "] ssm_a=" << A_log[h]
                                      << " araw=" << araw[h] << " dt=" << dt_b[h]
                                      << " sp=" << sp << " g=" << g << " beta=" << beta << "\n";
                        float* Sh = S.data() + static_cast<size_t>(h) * sk * sk;

                        // decay S and compute prediction pred[v] = sum_k S[k,v]*k[k].
                        std::vector<float> pred(sk, 0.0f), delta(sk), oh(sk, 0.0f);
                        for (int kk = 0; kk < sk; kk++) {
                            const float kval = khp[kk];
                            float* row = Sh + static_cast<size_t>(kk) * sk;
                            for (int v = 0; v < sk; v++) { row[v] *= g; pred[v] += row[v] * kval; }
                        }
                        for (int v = 0; v < sk; v++) delta[v] = beta * (vh[v] - pred[v]);
                        // write correction (outer product k x delta) and read o[v]=sum_k S[k,v]*q[k].
                        for (int kk = 0; kk < sk; kk++) {
                            const float kval = khp[kk], qval = qh[kk];
                            float* row = Sh + static_cast<size_t>(kk) * sk;
                            for (int v = 0; v < sk; v++) { row[v] += kval * delta[v]; oh[v] += row[v] * qval; }
                        }
                        // Gated RMSNorm (qwen35 build_norm_gated): norm FIRST, then gate.
                        // rmsnorm(o) over sk, * ssm_norm weight, then * silu(z). Verified
                        // against llama.cpp models/qwen35.cpp build_norm_gated (norm then
                        // ggml_mul with silu(gate)) — NOT the HF/Mamba2 gate-before-norm.
                        const float* snw = get_float_ptr(snorm_w);
                        const float* zh = zgate.data() + h * sk;
                        float ss = 0.0f;
                        for (int v = 0; v < sk; v++) ss += oh[v] * oh[v];
                        const float r = 1.0f / std::sqrt(ss / static_cast<float>(sk) + rms_eps);
                        float* od = o_all.data() + h * sk;
                        for (int v = 0; v < sk; v++)
                            od[v] = (oh[v] * r * snw[v]) * (zh[v] / (1.0f + std::exp(-zh[v])));
                    }

                    if (L == 0) dbg6("final_output(o_all)", o_all.data(), q35_value_dim);
                    std::fill(ssm_o.begin(), ssm_o.end(), 0.0f);
                    mul_mat_weight(o_all.data(), out_w, ssm_o.data(), 1, hidden_size, q35_value_dim,
                                   q35_value_dim, hidden_size, hidden_size, mm);
                    backend->sync();
                    if (L == 0) dbg6("linear_attn_out(ssm_o)", ssm_o.data(), hidden_size);
                    if (!std::getenv("GGNPU_Q35_SKIP_SSM"))
                        for (int i = 0; i < hidden_size; i++) inp_embd[i] += ssm_o[i];
                    if (L == 0) dbg6("attn_residual(l0)", inp_embd.data(), hidden_size);
                } else {
                    // ===== Gated full attention =====
                    const TensorView* q_w  = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", L);
                    const TensorView* k_w  = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", L);
                    const TensorView* v_w  = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", L);
                    const TensorView* o_w  = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", L);
                    const TensorView* qn_w = find_tensor_pattern(model, "blk.{layer}.attn_q_norm.weight", L);
                    const TensorView* kn_w = find_tensor_pattern(model, "blk.{layer}.attn_k_norm.weight", L);

                    std::fill(qg.begin(), qg.end(), 0.0f);
                    std::fill(kbuf.begin(), kbuf.end(), 0.0f);
                    std::fill(vbuf.begin(), vbuf.end(), 0.0f);
                    mul_mat_weight(xn.data(), q_w, qg.data(), 1, 2 * qdim, hidden_size,
                                   hidden_size, 2 * qdim, 2 * qdim, mm);
                    mul_mat_weight(xn.data(), k_w, kbuf.data(), 1, kvdim, hidden_size,
                                   hidden_size, kvdim, kvdim, mm);
                    mul_mat_weight(xn.data(), v_w, vbuf.data(), 1, kvdim, hidden_size,
                                   hidden_size, kvdim, kvdim, mm);
                    backend->sync();
                    // attn_q output is per-head interleaved [query(Hd)|gate(Hd)] x nQ heads
                    // (ggml reshapes to [Hd*2, n_head]); gather into contiguous query / gate.
                    std::vector<float> q(qdim), fgate(qdim);
                    for (int h = 0; h < nQ; h++) {
                        std::memcpy(q.data() + h * Hd, qg.data() + static_cast<size_t>(h) * 2 * Hd, Hd * sizeof(float));
                        std::memcpy(fgate.data() + h * Hd, qg.data() + static_cast<size_t>(h) * 2 * Hd + Hd, Hd * sizeof(float));
                    }

                    // Per-head QK-norm (host, exact Hd) then partial NEOX RoPE.
                    for (int h = 0; h < nQ; h++)
                        host_rmsnorm(q.data() + h * Hd, q.data() + h * Hd, Hd, get_float_ptr(qn_w), rms_eps);
                    for (int h = 0; h < nKV; h++)
                        host_rmsnorm(kbuf.data() + h * Hd, kbuf.data() + h * Hd, Hd, get_float_ptr(kn_w), rms_eps);
                    rope_neox_partial(q.data(), nQ, Hd, q35_rope_dim, pos, q35_rope_base);
                    rope_neox_partial(kbuf.data(), nKV, Hd, q35_rope_dim, pos, q35_rope_base);

                    q35_k_store[L].insert(q35_k_store[L].end(), kbuf.begin(), kbuf.end());
                    q35_v_store[L].insert(q35_v_store[L].end(), vbuf.begin(), vbuf.end());
                    const int64_t ctx_len = pos + 1;
                    const int group = nQ / nKV;
                    kexp.assign(static_cast<size_t>(nQ) * ctx_len * Hd, 0.0f);
                    vexp.assign(static_cast<size_t>(nQ) * ctx_len * Hd, 0.0f);
                    for (int h = 0; h < nQ; h++) {
                        const int kh = h / group;
                        for (int64_t j = 0; j < ctx_len; j++) {
                            std::memcpy(kexp.data() + (static_cast<size_t>(h) * ctx_len + j) * Hd,
                                        q35_k_store[L].data() + (static_cast<size_t>(j) * nKV + kh) * Hd,
                                        Hd * sizeof(float));
                            std::memcpy(vexp.data() + (static_cast<size_t>(h) * ctx_len + j) * Hd,
                                        q35_v_store[L].data() + (static_cast<size_t>(j) * nKV + kh) * Hd,
                                        Hd * sizeof(float));
                        }
                    }
                    std::fill(attn_in.begin(), attn_in.end(), 0.0f);
                    AttnParams fa;
                    fa.Q = q.data(); fa.K = kexp.data(); fa.V = vexp.data(); fa.output = attn_in.data();
                    fa.batch_size = 1; fa.n_head = nQ; fa.head_dim = Hd; fa.ctx_len = ctx_len;
                    fa.query_pos = pos; fa.freq_factors = nullptr;
                    if (backend->flash_attn(fa) != Status::OK) {
                        std::cerr << "Error: qwen35 flash_attn failed at layer " << L << "\n";
                        model.unload();
                        return 1;
                    }
                    // Output gate: attn_out *= sigmoid(gate).
                    for (int i = 0; i < qdim; i++) attn_in[i] *= 1.0f / (1.0f + std::exp(-fgate[i]));
                    std::fill(attn_o.begin(), attn_o.end(), 0.0f);
                    mul_mat_weight(attn_in.data(), o_w, attn_o.data(), 1, hidden_size, qdim,
                                   qdim, hidden_size, hidden_size, mm);
                    backend->sync();
                    if (!std::getenv("GGNPU_Q35_SKIP_ATT"))
                        for (int i = 0; i < hidden_size; i++) inp_embd[i] += attn_o[i];
                }

                if (q35dbg && L < 5) std::cerr << " post_mix_rms=" << vnorm(inp_embd, hidden_size);
                // ===== FFN (post_attention_norm as the FFN norm) =====
                if (!g_rmsnorm(xn.data(), inp_embd.data(), hidden_size, get_float_ptr(post_norm_w), rms_eps)) return 1;
                if (is_qwen35moe) {
                    // ---- MoE FFN: router top-k routed experts + sigmoid-gated shared expert ----
                    const int nE  = q35_n_expert;       // 256 total experts
                    const int nEU = q35_n_expert_used;  // 8 routed per token
                    const int eff = q35_expert_ff;      // 512 per-expert hidden
                    // Router logits = ffn_gate_inp (F32 [hidden, nE]) . xn. Tiny (256x2048),
                    // computed host-side.
                    const TensorView* router_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate_inp.weight", L);
                    const float* rw = get_float_ptr(router_w);
                    std::vector<float> rlogits(nE);
                    for (int e = 0; e < nE; e++) {
                        const float* wr = rw + static_cast<size_t>(e) * hidden_size;
                        float acc = 0.0f;
                        for (int i = 0; i < hidden_size; i++) acc += xn[i] * wr[i];
                        rlogits[e] = acc;
                    }
                    // Top-k experts, then softmax over just those (== softmax-over-all then
                    // renormalize the top-k, i.e. Qwen MoE norm_topk_prob=true).
                    std::vector<int> eidx(nE);
                    for (int e = 0; e < nE; e++) eidx[e] = e;
                    std::partial_sort(eidx.begin(), eidx.begin() + nEU, eidx.end(),
                                      [&](int a, int b) { return rlogits[a] > rlogits[b]; });
                    std::vector<float> wsel(nEU);
                    float mx = rlogits[eidx[0]], wsum = 0.0f;
                    for (int j = 0; j < nEU; j++) { wsel[j] = std::exp(rlogits[eidx[j]] - mx); wsum += wsel[j]; }
                    for (int j = 0; j < nEU; j++) wsel[j] /= wsum;

                    // Slice expert e's 2D [K,N] matrix out of a 3D [K,N,nE] expert tensor.
                    // A distinct name gives each expert its own WeightCache entry (only the
                    // ~8 experts touched per token ever get decoded).
                    auto expert_slice = [](const TensorView* base, int e) -> TensorView {
                        TensorView v = *base;
                        v.n_dims = 2;
                        const uint64_t K = base->dims[0], N = base->dims[1];
                        v.dims = {K, N};
                        const size_t per = (static_cast<size_t>(K) * N / ggml_blck_size(base->type))
                                           * ggml_type_size(base->type);
                        v.data = base->data + static_cast<size_t>(e) * per;
                        v.name = base->name + ".e" + std::to_string(e);
                        return v;
                    };
                    const TensorView* gexp = find_tensor_pattern(model, "blk.{layer}.ffn_gate_exps.weight", L);
                    const TensorView* uexp = find_tensor_pattern(model, "blk.{layer}.ffn_up_exps.weight", L);
                    const TensorView* dexp = find_tensor_pattern(model, "blk.{layer}.ffn_down_exps.weight", L);
                    std::vector<float> eg(eff), eu(eff), ea(eff), ed(hidden_size), moe(hidden_size, 0.0f);
                    // SwiGLU per expert: down(silu(gate(x)) * up(x)), weighted by router prob.
                    auto run_expert = [&](const TensorView* g, const TensorView* u,
                                          const TensorView* d, float weight) {
                        std::fill(eg.begin(), eg.end(), 0.0f);
                        std::fill(eu.begin(), eu.end(), 0.0f);
                        mul_mat_weight(xn.data(), g, eg.data(), 1, eff, hidden_size, hidden_size, eff, eff, mm);
                        mul_mat_weight(xn.data(), u, eu.data(), 1, eff, hidden_size, hidden_size, eff, eff, mm);
                        backend->sync();
                        for (int i = 0; i < eff; i++)
                            ea[i] = (eg[i] / (1.0f + std::exp(-eg[i]))) * eu[i];
                        std::fill(ed.begin(), ed.end(), 0.0f);
                        mul_mat_weight(ea.data(), d, ed.data(), 1, hidden_size, eff, eff, hidden_size, hidden_size, mm);
                        backend->sync();
                        for (int i = 0; i < hidden_size; i++) moe[i] += weight * ed[i];
                    };
                    for (int j = 0; j < nEU; j++) {
                        const int e = eidx[j];
                        TensorView gv = expert_slice(gexp, e);
                        TensorView uv = expert_slice(uexp, e);
                        TensorView dv = expert_slice(dexp, e);
                        run_expert(&gv, &uv, &dv, wsel[j]);
                    }
                    // Shared expert (always active), gated by sigmoid(ffn_gate_inp_shexp . x).
                    const TensorView* sg = find_tensor_pattern(model, "blk.{layer}.ffn_gate_shexp.weight", L);
                    const TensorView* su = find_tensor_pattern(model, "blk.{layer}.ffn_up_shexp.weight", L);
                    const TensorView* sd = find_tensor_pattern(model, "blk.{layer}.ffn_down_shexp.weight", L);
                    if (sg && su && sd) {
                        float sgate_val = 1.0f;
                        const TensorView* sgate = find_tensor_pattern(model, "blk.{layer}.ffn_gate_inp_shexp.weight", L);
                        if (sgate) {
                            const float* sw = get_float_ptr(sgate);
                            float acc = 0.0f;
                            for (int i = 0; i < hidden_size; i++) acc += xn[i] * sw[i];
                            sgate_val = 1.0f / (1.0f + std::exp(-acc));
                        }
                        run_expert(sg, su, sd, sgate_val);
                    }
                    if (L == 0) dbg6("moe_ffn_out", moe.data(), hidden_size);
                    for (int i = 0; i < hidden_size; i++) inp_embd[i] += moe[i];
                } else {
                    // ---- Dense SwiGLU FFN (qwen35 4B/9B) ----
                    const TensorView* gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", L);
                    const TensorView* up_w   = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", L);
                    const TensorView* down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", L);
                    std::fill(gate.begin(), gate.end(), 0.0f);
                    std::fill(up.begin(), up.end(), 0.0f);
                    mul_mat_weight(xn.data(), gate_w, gate.data(), 1, ffn_size, hidden_size,
                                   hidden_size, ffn_size, ffn_size, mm);
                    mul_mat_weight(xn.data(), up_w, up.data(), 1, ffn_size, hidden_size,
                                   hidden_size, ffn_size, ffn_size, mm);
                    backend->sync();
                    if (fp32_ssm) {
                        // Exact fp32 SiLU on CPU (full-fp32 diagnostic; see GGNPU_FP32_SSM).
                        for (int i = 0; i < ffn_size; i++)
                            act[i] = gate[i] / (1.0f + std::exp(-gate[i]));
                    } else {
                        SiluParams sp; sp.input = gate.data(); sp.output = act.data(); sp.size = ffn_size;
                        if (backend->silu(sp) != Status::OK) {
                            std::cerr << "Error: qwen35 silu failed at layer " << L << "\n";
                            model.unload();
                            return 1;
                        }
                    }
                    for (int i = 0; i < ffn_size; i++) act[i] *= up[i];
                    std::fill(dn.begin(), dn.end(), 0.0f);
                    mul_mat_weight(act.data(), down_w, dn.data(), 1, hidden_size, ffn_size,
                                   ffn_size, hidden_size, hidden_size, mm);
                    backend->sync();
                    for (int i = 0; i < hidden_size; i++) inp_embd[i] += dn[i];
                }
                if (q35dbg && L < 5) std::cerr << " post_ffn_rms=" << vnorm(inp_embd, hidden_size) << "\n";
                if (q35dbg && L < 6) {
                    double s = 0; for (int i = 0; i < hidden_size; i++) s += inp_embd[i];
                    std::cerr << "  [q35cmp] l_out-" << L << " sum=" << s << "\n";
                }
            }
            // Final output_norm handled by the shared (non-gemma, non-lfm2) path below.
        }

        // Transformer layers (non-gemma, non-lfm2, non-qwen35 path)
        for (int layer = 0; !is_gemma && !is_lfm2 && !is_qwen35 && layer < max_layers_override; layer++) {
            const TensorView* attn_norm_w =
                find_tensor_pattern(model, "blk.{layer}.attn_norm.weight", layer);
            const TensorView* ffn_norm_w =
                find_tensor_pattern(model, "blk.{layer}.ffn_norm.weight", layer);

            // RMSNorm for attention
            {
                ScopedTimer t(step_timings.rms_norm_ms);
                if (!rms_norm(inp_norm.data(), inp_embd.data(), n_tokens, hidden_size, rms_eps,
                              get_float_ptr(attn_norm_w))) {
                    model.unload();
                    return 1;
                }
            }

            // Attention projections
            const TensorView* attn_q = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", layer);
            const TensorView* attn_k = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", layer);
            const TensorView* attn_v = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", layer);

            mul_mat_weight(inp_norm.data(), attn_q, q_proj.data(),
                           n_tokens, q_dim, hidden_size,
                           hidden_size, q_dim, q_dim,
                           params.verbose ? &step_timings.matmul_ms : nullptr);
            mul_mat_weight(inp_norm.data(), attn_k, k_proj.data(),
                           n_tokens, num_kv_heads * head_dim, hidden_size,
                           hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim,
                           params.verbose ? &step_timings.matmul_ms : nullptr);
            mul_mat_weight(inp_norm.data(), attn_v, v_proj.data(),
                           n_tokens, num_kv_heads * head_dim, hidden_size,
                           hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim,
                           params.verbose ? &step_timings.matmul_ms : nullptr);

            backend->sync();

            // QKV bias (Qwen2 etc.): add the per-output-channel bias to each
            // token's projection. Llama has no bias tensors, so this is a no-op
            // there (find returns null). Biases are small F32 vectors.
            auto add_qkv_bias = [&](const char* pattern, float* proj, int dim) {
                const TensorView* b = find_tensor_pattern(model, pattern, layer);
                if (!b || b->type != GgmlType::F32) return;
                const float* bias = get_float_ptr(b);
                for (int i = 0; i < n_tokens; i++)
                    for (int j = 0; j < dim; j++)
                        proj[static_cast<size_t>(i) * dim + j] += bias[j];
            };
            add_qkv_bias("blk.{layer}.attn_q.bias", q_proj.data(), q_dim);
            add_qkv_bias("blk.{layer}.attn_k.bias", k_proj.data(), num_kv_heads * head_dim);
            add_qkv_bias("blk.{layer}.attn_v.bias", v_proj.data(), num_kv_heads * head_dim);

            // Per-head QK-norm (qwen3, and any arch that ships attn_q_norm /
            // attn_k_norm): RMSNorm each head's head_dim slice of Q and K with the
            // shared norm weight, before RoPE. Absent tensors → no-op (llama/qwen2).
            {
                const TensorView* qn_w = find_tensor_pattern(model, "blk.{layer}.attn_q_norm.weight", layer);
                const TensorView* kn_w = find_tensor_pattern(model, "blk.{layer}.attn_k_norm.weight", layer);
                // CPU RMSNorm over exactly head_dim (avoids NPU rms_norm pad-to-pow2
                // scaling issues for the small 128-wide QK-norm).
                auto rmsnorm_head = [&](float* v, const float* w) {
                    double ss = 0.0;
                    for (int i = 0; i < head_dim; i++) ss += static_cast<double>(v[i]) * v[i];
                    float inv = 1.0f / std::sqrt(static_cast<float>(ss / head_dim) + rms_eps);
                    for (int i = 0; i < head_dim; i++) v[i] = v[i] * inv * w[i];
                };
                if (qn_w) {
                    const float* qnw = get_float_ptr(qn_w);
                    for (int tkn = 0; tkn < n_tokens; tkn++)
                        for (int h = 0; h < num_heads; h++)
                            rmsnorm_head(q_proj.data() + (static_cast<size_t>(tkn) * q_dim) + static_cast<size_t>(h) * head_dim, qnw);
                }
                if (kn_w) {
                    const float* knw = get_float_ptr(kn_w);
                    for (int tkn = 0; tkn < n_tokens; tkn++)
                        for (int h = 0; h < num_kv_heads; h++)
                            rmsnorm_head(k_proj.data() + (static_cast<size_t>(tkn) * num_kv_heads * head_dim) + static_cast<size_t>(h) * head_dim, knw);
                }
            }

            {
                ScopedTimer t(step_timings.rope_ms);
                const int kvdim = num_kv_heads * head_dim;
                // Each batched token carries its own absolute position (pos + bt).
                for (int bt = 0; bt < n_tokens; bt++) {
                    const int tpos = pos + bt;
                    float* qr = q_rope.data() + static_cast<size_t>(bt) * q_dim;
                    const float* qp = q_proj.data() + static_cast<size_t>(bt) * q_dim;
                    float* kr = k_rope.data() + static_cast<size_t>(bt) * kvdim;
                    const float* kp = k_proj.data() + static_cast<size_t>(bt) * kvdim;
                    if (is_qwen3) {
                        // qwen3 uses NEOX (split-half) RoPE: pair element i with i+hd/2
                        // per head (vs the interleaved pairs apply_rope uses). Done on
                        // CPU via rope_cache (built with qwen3's freq_base).
                        auto rope_neox = [&](float* out, const float* in, int nh) {
                            const int half = head_dim / 2;
                            for (int h = 0; h < nh; h++) {
                                const float* ih = in + static_cast<size_t>(h) * head_dim;
                                float* oh = out + static_cast<size_t>(h) * head_dim;
                                for (int i = 0; i < half; i++) {
                                    const float c = *rope_cache.cos_ptr(tpos, i);
                                    const float s = *rope_cache.sin_ptr(tpos, i);
                                    const float v0 = ih[i], v1 = ih[i + half];
                                    oh[i]        = v0 * c - v1 * s;
                                    oh[i + half] = v0 * s + v1 * c;
                                }
                            }
                        };
                        rope_neox(qr, qp, num_heads);
                        rope_neox(kr, kp, num_kv_heads);
                    } else {
                        // RoPE: NPU when GGNPU_NPU_ROPE=1 (validated, see bench-layer 0g),
                        // else CPU. apply_rope dispatches and falls back to CPU on failure.
                        apply_rope(qr, qp, num_heads, tpos, head_dim);
                        apply_rope(kr, kp, num_kv_heads, tpos, head_dim);
                    }
                }
            }

            // Write all n_tokens positions [pos, pos+n_tokens) into the KV cache.
            model.kv_cache().update_slab(layer, pos, pos + n_tokens,
                                         k_rope.data(), v_proj.data(),
                                         num_kv_heads, head_dim);

            // Full context after this batch's positions are written. Each query
            // masks future keys via query_pos, so one K/V build serves all queries.
            int64_t ctx_len = pos + n_tokens;
            int64_t num_kv_heads = hparams.attention_head_count_kv;
            int64_t qkv_groups = hparams.attention_head_count / num_kv_heads;

            // Expand K/V from num_kv_heads to num_heads (GQA support).
            // Uses preallocated buffers — resize keeps allocation when capacity suffices.
            const int64_t n_heads = hparams.attention_head_count;
            k_expanded.resize(static_cast<size_t>(n_heads) * ctx_len * head_dim);
            v_expanded.resize(static_cast<size_t>(n_heads) * ctx_len * head_dim);

            {
                ScopedTimer t(step_timings.kv_expand_ms);
                for (int64_t h = 0; h < n_heads; h++) {
                    int64_t kv_h = h / qkv_groups;
                    float* kh = k_expanded.data() + h * ctx_len * head_dim;
                    float* vh = v_expanded.data() + h * ctx_len * head_dim;
                    for (int64_t j = 0; j < ctx_len; j++) {
                        const float* kj = model.kv_cache().key_buffer(layer, j) + kv_h * head_dim;
                        const float* vj = model.kv_cache().value_buffer(layer, j) + kv_h * head_dim;
                        if (kj) std::memcpy(kh + j * head_dim, kj, head_dim * sizeof(float));
                        if (vj) std::memcpy(vh + j * head_dim, vj, head_dim * sizeof(float));
                    }
                }
            }

            std::vector<float> attn_out(static_cast<size_t>(n_tokens) * q_dim, 0.0f);

            {
                ScopedTimer t(step_timings.flash_attn_ms);
                // One attention call per query row. K/V hold the full context
                // (ctx_len); each query at absolute position pos+bt masks keys
                // beyond it via query_pos, giving correct causal attention over
                // its own prefix. For n_tokens==1 (decode) this is one call.
                for (int bt = 0; bt < n_tokens; bt++) {
                    AttnParams fa_params;
                    fa_params.Q = q_rope.data() + static_cast<size_t>(bt) * q_dim;
                    fa_params.K = k_expanded.data();
                    fa_params.V = v_expanded.data();
                    fa_params.output = attn_out.data() + static_cast<size_t>(bt) * q_dim;
                    fa_params.batch_size = 1;
                    fa_params.n_head = num_heads;
                    fa_params.head_dim = head_dim;
                    fa_params.ctx_len = ctx_len;
                    fa_params.query_pos = pos + bt;
                    fa_params.freq_factors = nullptr;
                    if (backend->flash_attn(fa_params) != Status::OK) {
                        std::cerr << "Error: flash_attn failed at layer " << layer << "\n";
                        model.unload();
                        return 1;
                    }
                }
            }

            // Attention output projection
            const TensorView* attn_output_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", layer);
            if (attn_output_w) {
                std::fill(attn_output.begin(), attn_output.end(), 0.0f);
                mul_mat_weight(attn_out.data(), attn_output_w, attn_output.data(),
                               n_tokens, hidden_size, q_dim,
                               q_dim, hidden_size, hidden_size,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();

                {
                    ScopedTimer t(step_timings.residual_ms);
                    for (int t = 0; t < n_tokens; t++) {
                        for (int i = 0; i < hidden_size; i++) {
                            inp_embd[static_cast<size_t>(t) * hidden_size + i] +=
                                attn_output[static_cast<size_t>(t) * hidden_size + i];
                        }
                    }
                }
            }
            // NOTE: residual add on CPU — acceptable for MVP (see IMPLEMENTATION.md §7.1)

            // FFN path
            {
                ScopedTimer t(step_timings.rms_norm_ms);
                if (!rms_norm(inp_norm.data(), inp_embd.data(), n_tokens, hidden_size, rms_eps,
                              get_float_ptr(ffn_norm_w))) {
                    model.unload();
                    return 1;
                }
            }

            const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer);
            const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer);
            const TensorView* ffn_down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", layer);

            mul_mat_weight(inp_norm.data(), ffn_gate_w, ffn_gate.data(),
                           n_tokens, ffn_size, hidden_size,
                           hidden_size, ffn_size, ffn_size,
                           params.verbose ? &step_timings.matmul_ms : nullptr);
            mul_mat_weight(inp_norm.data(), ffn_up_w, ffn_up.data(),
                           n_tokens, ffn_size, hidden_size,
                           hidden_size, ffn_size, ffn_size,
                           params.verbose ? &step_timings.matmul_ms : nullptr);

            backend->sync();

            // SwiGLU: silu(gate) * up per token row
            for (int t = 0; t < n_tokens; t++) {
                {
                    ScopedTimer st(step_timings.silu_ms);
                    SiluParams silu_params;
                    silu_params.input = ffn_gate.data() + t * ffn_size;
                    silu_params.output = ffn_silu.data() + t * ffn_size;
                    silu_params.size = ffn_size;
                    if (backend->silu(silu_params) != Status::OK) {
                        std::cerr << "Error: silu failed at layer " << layer << "\n";
                        model.unload();
                        return 1;
                    }
                }

                for (int i = 0; i < ffn_size; i++) {
                    ffn_silu[static_cast<size_t>(t) * ffn_size + i] *=
                        ffn_up[static_cast<size_t>(t) * ffn_size + i];
                }
            }

            // FFN down projection
            if (ffn_down_w) {
                std::fill(ffn_down.begin(), ffn_down.end(), 0.0f);
                mul_mat_weight(ffn_silu.data(), ffn_down_w, ffn_down.data(),
                               n_tokens, hidden_size, ffn_size,
                               ffn_size, hidden_size, hidden_size,
                               params.verbose ? &step_timings.matmul_ms : nullptr);
                backend->sync();

                {
                    ScopedTimer rt(step_timings.residual_ms);
                    for (int t = 0; t < n_tokens; t++) {
                        for (int i = 0; i < hidden_size; i++) {
                            inp_embd[static_cast<size_t>(t) * hidden_size + i] +=
                                ffn_down[static_cast<size_t>(t) * hidden_size + i];
                        }
                    }
                }
            }
        }

        // Final normalization (output_norm before logits) — non-gemma, non-lfm2
        // path (those branches already wrote inp_norm with their own final norm).
        if (!is_gemma && !is_lfm2) {
            const TensorView* output_norm_w = find_tensor(model, "output_norm.weight");
            ScopedTimer t(step_timings.rms_norm_ms);
            if (!rms_norm(inp_norm.data(), inp_embd.data(), n_tokens, hidden_size, rms_eps,
                          get_float_ptr(output_norm_w))) {
                model.unload();
                return 1;
            }
        }

        pos += n_tokens;

        if (!do_sample) {
            backend->sync();
            if (params.verbose) {
                total_timings.add(step_timings);
            }
            continue;
        }

        // Speculative verification: the block was [last_sampled, draft_1..draft_k]
        // (k=n_draft). Score all n_tokens rows at once (one batched logits launch),
        // then greedily accept the longest draft prefix whose argmax matches. Commit
        // matched+1 tokens (accepted drafts + one bonus/correction). `pos += n_tokens`
        // above over-counted by the rejected drafts, so roll it back to
        // old_pos + 1 + matched (= committed length). KV for rejected positions is
        // overwritten next step (attention never reads past the new query positions).
        if (n_draft > 0) {
            std::vector<float> spec_logits(static_cast<size_t>(n_tokens) * vocab_size, 0.0f);
            {
                ScopedTimer t(step_timings.logits_ms);
                compute_logits_rows(inp_norm.data(), n_tokens, logits_w, spec_logits.data(),
                                    vocab_size, hidden_size, weight_cache, *backend);
            }
            backend->sync();

            auto row_argmax = [&](int r) -> int {
                const float* row = spec_logits.data() + static_cast<size_t>(r) * vocab_size;
                int best = 0;
                float bv = row[0];
                for (int v = 1; v < vocab_size; v++)
                    if (row[v] > bv) { bv = row[v]; best = v; }
                return best;
            };

            // cand[r] = greedy token predicted from row r (fed token at pos+r).
            // Accept draft batch_toks[matched+1] while it equals cand[matched].
            int matched = 0;
            std::vector<int> committed;
            committed.reserve(static_cast<size_t>(n_draft) + 1);
            while (true) {
                int c = row_argmax(matched);
                committed.push_back(c);  // new token at old_pos + matched + 1
                if (matched < n_draft && c == batch_toks[matched + 1]) {
                    matched++;           // draft accepted; verify the next one
                } else {
                    break;               // mismatch or drafts exhausted -> bonus token
                }
            }

            pos -= (n_draft - matched);  // discard rejected draft positions

            spec_steps++;
            spec_proposed += n_draft;
            spec_accepted += matched;

            bool stop = false;
            for (int c : committed) {
                std::cout << tokenizer.decode(c);
                std::cout.flush();
                first_token = false;
                seq.push_back(c);
                last_sampled = c;
                generated++;
                if (c == tokenizer.eos_token_id()) { stop = true; break; }
                if (pos >= ctx_size) { stop = true; break; }
                if (generated >= params.max_tokens) { stop = true; break; }
            }

            if (params.verbose) {
                total_timings.add(step_timings);
            }
            if (stop) break;
            continue;
        }

        // Logits: only the last token's row feeds sampling (batched prefill
        // computed n_tokens rows in inp_norm; we sample the next token from the
        // final prompt position). NPU INT8 mul_mat_q (cached decode) with CPU fallback.
        if (logits_w) {
            ScopedTimer t(step_timings.logits_ms);
            const float* last_row = inp_norm.data() +
                static_cast<size_t>(n_tokens - 1) * hidden_size;
            compute_logits(last_row, logits_w, logits.data(),
                           vocab_size, hidden_size, weight_cache, *backend);
        }

        backend->sync();

        // Gemma 4 final logit soft-capping: cap * tanh(logit / cap).
        if (is_gemma && gemma_logit_softcap > 0.0f) {
            for (float& l : logits)
                l = gemma_logit_softcap * std::tanh(l / gemma_logit_softcap);
        }

        if (params.bench_logits) {
            if (params.verbose) {
                total_timings.add(step_timings);
            }
            info << "Top logits after prompt (pos=" << pos
                      << ", layers=" << max_layers_override << "):\n";
            for (int paris_id : {12366, 60704}) {
                if (paris_id < vocab_size) {
                    info << "    candidate id=" << paris_id << " logit="
                              << logits[static_cast<size_t>(paris_id)]
                              << " text=" << tokenizer.decode(paris_id) << "\n";
                }
            }
            print_top_logits(logits, tokenizer, 10);
            model.unload();
            return 0;
        }

        int next_token = 0;
        {
            ScopedTimer t(step_timings.sample_ms);
            next_token = sample_token(logits, params.temp, rng_seed);
        }
        std::string decoded = tokenizer.decode(next_token);

        if (first_token) {
            std::cout << decoded;
            first_token = false;
        } else {
            std::cout << decoded;
        }
        std::cout.flush();

        last_sampled = next_token;
        seq.push_back(next_token);
        generated++;

        if (next_token == tokenizer.eos_token_id()) break;
        if (pos >= ctx_size) break;

        if (params.verbose) {
            total_timings.add(step_timings);
        }
    }

    info << "\n\nGenerated " << generated << " tokens.\n";
    if (spec_debug && spec_steps > 0) {
        info << "Speculative: " << spec_steps << " steps, "
             << spec_accepted << "/" << spec_proposed << " drafts accepted ("
             << (100.0 * spec_accepted / std::max(1L, spec_proposed)) << "%), "
             << (static_cast<double>(generated) / spec_steps)
             << " tokens/step (vs 1.0 baseline)\n";
    }

    if (params.verbose) {
        total_timings.print_summary();
    }

    model.unload();
    return 0;
}
