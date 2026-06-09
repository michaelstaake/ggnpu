#include "backend.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

namespace ggnpu {

// CPU reference backend for testing
// Used when GGNPU_TEST_CPU=1 or for validation

class CpuRefBackend : public Backend {
public:
    CpuRefBackend() : last_status_(Status::OK) {}

    Status mul_mat_q(const MulMatParams& params) override {
        // Simplified matmul: just copy for now
        // Full implementation would decode quantized weights and compute
        if (!params.A || !params.B || !params.C) return Status::INVALID_PARAM;

        int M = params.M;
        int N = params.N;
        int K = params.K;

        // For F32 x F32 -> F32
        if (params.scales) {
            // Quantized matmul with scales
            const float* scales = static_cast<const float*>(params.scales);
            const int8_t* A_int8 = static_cast<const int8_t*>(params.A);
            const int8_t* B_int8 = static_cast<const int8_t*>(params.B);
            float* C_float = static_cast<float*>(params.C);

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += static_cast<float>(A_int8[m * K + k]) *
                               static_cast<float>(B_int8[k * N + n]);
                    }
                    C_float[m * N + n] = sum * scales[m];
                }
            }
        } else {
            // Simple F32 matmul
            const float* A = static_cast<const float*>(params.A);
            const float* B = static_cast<const float*>(params.B);
            float* C = static_cast<float*>(params.C);

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[m * K + k] * B[k * N + n];
                    }
                    C[m * N + n] = sum;
                }
            }
        }

        return Status::OK;
    }

    Status rms_norm(const RmsNormParams& params) override {
        if (!params.input || !params.output) return Status::INVALID_PARAM;

        float variance = 0.0f;
        for (int i = 0; i < params.size; i++) {
            variance += params.input[i] * params.input[i];
        }
        variance /= params.size;
        variance += params.eps;
        float inv_std = 1.0f / std::sqrt(variance);

        for (int i = 0; i < params.size; i++) {
            params.output[i] = params.input[i] * inv_std;
        }

        return Status::OK;
    }

    Status rope(const RopeParams& params) override {
        if (!params.data) return Status::INVALID_PARAM;

        for (int64_t i = 0; i < params.rope_dims; i += 2) {
            float ratio = 1.0f / std::pow(10000.0f, static_cast<float>(i) / params.n_dims);
            float val = params.offset * ratio * params.freq_scale;
            float cos_val = std::cos(val);
            float sin_val = std::sin(val);

            float v0 = params.data[i];
            float v1 = params.data[i + 1];
            params.data[i] = v0 * cos_val - v1 * sin_val;
            params.data[i + 1] = v0 * sin_val + v1 * cos_val;
        }

        return Status::OK;
    }

    Status softmax(const SoftmaxParams& params) override {
        if (!params.input || !params.output) return Status::INVALID_PARAM;

        for (int r = 0; r < params.rows; r++) {
            float max_val = -INFINITY;
            for (int c = 0; c < params.cols; c++) {
                float val = params.input[r * params.cols + c];
                if (val > max_val) max_val = val;
            }

            float sum = 0.0f;
            for (int c = 0; c < params.cols; c++) {
                params.output[r * params.cols + c] = std::exp(params.input[r * params.cols + c] - max_val);
                sum += params.output[r * params.cols + c];
            }

            for (int c = 0; c < params.cols; c++) {
                params.output[r * params.cols + c] /= sum;
            }
        }

        return Status::OK;
    }

    Status silu(const SiluParams& params) override {
        if (!params.input || !params.output) return Status::INVALID_PARAM;

        for (int i = 0; i < params.size; i++) {
            float x = params.input[i];
            params.output[i] = x / (1.0f + std::exp(-x));
        }

        return Status::OK;
    }

    Status flash_attn(const AttnParams& params) override {
        // Simplified attention: QK^T / sqrt(d) + V
        if (!params.Q || !params.K || !params.V || !params.output) return Status::INVALID_PARAM;

        int bs = params.batch_size;
        int nh = params.n_head;
        int hd = params.head_dim;
        int64_t cl = params.ctx_len;

        for (int b = 0; b < bs; b++) {
            for (int h = 0; h < nh; h++) {
                // Q: [bs, nh, 1, hd]
                // K: [bs, nh, cl, hd]
                // V: [bs, nh, cl, hd]
                // Output: [bs, nh, 1, hd]

                float scale = 1.0f / std::sqrt(static_cast<float>(hd));
                float* out = params.output + b * nh * hd + h * hd;

                // Compute attention weights (simplified: uniform)
                for (int d = 0; d < hd; d++) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < cl; ki++) {
                        float qk = 0.0f;
                        const float* q = params.Q + b * nh * hd + h * hd + d;
                        const float* k_ptr = params.K + b * nh * cl * hd + h * cl * hd + ki * hd + d;
                        qk = static_cast<float>(*q) * static_cast<float>(*k_ptr) * scale;
                        float attn = std::exp(qk);
                        sum += attn * params.V[b * nh * cl * hd + h * cl * hd + ki * hd + d];
                    }
                    out[d] = sum;
                }
            }
        }

        return Status::OK;
    }

    void sync() override {}

    bool is_available() const override { return true; }
    std::string name() const override { return "cpu_ref"; }
    Status last_error() const override { return last_status_; }

private:
    Status last_status_;
};

std::shared_ptr<Backend> create_cpu_ref_backend() {
    return std::make_shared<CpuRefBackend>();
}

} // namespace ggnpu
