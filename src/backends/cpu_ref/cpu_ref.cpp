#include "backend.h"
#include "tensor.h"
#include "quant/q4_0.h"
#include "quant/q8_0.h"
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

    static int8_t q4_0_to_int8(uint8_t nibble) {
        return static_cast<int8_t>((nibble & 0x08) ? (nibble | 0xF0) : nibble);
    }

    Status mul_mat_q(const MulMatParams& params) override {
        if (!params.A || !params.B || !params.C) return Status::INVALID_PARAM;

        int M = params.M;
        int N = params.N;
        int K = params.K;
        float* C = static_cast<float*>(params.C);

        memset(C, 0, M * N * sizeof(float));

        switch (params.B_type) {
        case GgmlType::Q8_0: {
            const float* A_f32 = static_cast<const float*>(params.A);
            const int8_t* B_q = static_cast<const int8_t*>(params.B);
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A_f32[m * K + k] *
                               static_cast<float>(B_q[k * N + n]);
                    }
                    C[m * N + n] = sum;
                }
            }
            break;
        }

        case GgmlType::Q4_0: {
            const float* A_f32 = static_cast<const float*>(params.A);
            const uint8_t* B_q = static_cast<const uint8_t*>(params.B);
            size_t B_block_size = ggml_type_size(GgmlType::Q4_0);
            size_t num_blocks = K / ggml_blck_size(GgmlType::Q4_0);

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (size_t b = 0; b < num_blocks; b++) {
                        Q4_0Block block;
                        memcpy(&block, B_q + b * B_block_size, sizeof(Q4_0Block));
                        int16_t d_raw = block.d;
                        float d = static_cast<float>(static_cast<int16_t>(d_raw));

                        for (int k_in_block = 0; k_in_block < 16; k_in_block++) {
                            int k = static_cast<int>(b * 16 + k_in_block);
                            if (k >= K) break;

                            uint8_t nibble;
                            if (k_in_block % 2 == 0) {
                                nibble = block.qs[k_in_block / 2] & 0x0F;
                            } else {
                                nibble = (block.qs[k_in_block / 2] >> 4) & 0x0F;
                            }
                            int8_t bv = q4_0_to_int8(nibble);
                            sum += A_f32[m * K + k] * static_cast<float>(bv) * d;
                        }
                    }
                    C[m * N + n] = sum;
                }
            }
            break;
        }

        case GgmlType::F32:
        default: {
            const float* A = static_cast<const float*>(params.A);
            const float* B = static_cast<const float*>(params.B);
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[m * K + k] * B[k * N + n];
                    }
                    C[m * N + n] = sum;
                }
            }
            break;
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
