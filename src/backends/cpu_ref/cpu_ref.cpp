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

        case GgmlType::Q4_K: {
            // Q4_K: 48 bytes per block, 256 values per block
            // Mixed quantization: first 128 values Q4_0, last 128 values Q8_0
            const float* A_f32 = static_cast<const float*>(params.A);
            const uint8_t* B_q = static_cast<const uint8_t*>(params.B);
            constexpr size_t Q4_K_BLOCK_SIZE = 48;
            constexpr int Q4_K_VALUES_PER_BLOCK = 256;
            size_t num_blocks = K / Q4_K_VALUES_PER_BLOCK;

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (size_t b = 0; b < num_blocks; b++) {
                        const uint8_t* block = B_q + b * Q4_K_BLOCK_SIZE;

                        // Read d and c (16-bit little-endian, multiplied by 4)
                        int16_t d_raw = static_cast<int16_t>(block[0] | (block[1] << 8));
                        int16_t c_raw = static_cast<int16_t>(block[2] | (block[3] << 8));
                        float d = static_cast<float>(d_raw) * 0.25f;
                        float c = static_cast<float>(c_raw) * 0.25f;

                        // Decode 6-byte scale array into 8 scale values
                        float scales[8];
                        scales[0] = d;
                        scales[1] = c;
                        for (int i = 0; i < 6; i++) {
                            int8_t s = static_cast<int8_t>(block[4 + i]);
                            scales[i * 2 + 2] = static_cast<float>(s & 0x0F) * d;
                            scales[i * 2 + 3] = static_cast<float>((s >> 4) & 0x0F) * c;
                        }

                        // First 128 values: Q4_0 with per-group scales (groups of 32)
                        for (int i = 0; i < 128; i += 32) {
                            int scale_idx = (i / 32) % 4;
                            float scale = scales[scale_idx];
                            const uint8_t* qblock = block + 10 + (i / 32) * 16;

                            for (int j = 0; j < 32; j++) {
                                uint8_t nibble = (j % 2 == 0) ? (qblock[j / 2] & 0x0F) : (qblock[j / 2] >> 4);
                                int8_t val = static_cast<int8_t>(nibble - 8);
                                int k = static_cast<int>(b * Q4_K_VALUES_PER_BLOCK + i + j);
                                if (k < K) {
                                    sum += A_f32[m * K + k] * static_cast<float>(val) * scale;
                                }
                            }
                        }

                        // Last 128 values: Q8_0 with per-group scales (groups of 32)
                        for (int i = 128; i < 256; i += 32) {
                            int scale_idx = ((i - 128) / 32) + 4;
                            float scale = scales[scale_idx];
                            const uint8_t* qblock = block + 42 + (i - 128);

                            for (int j = 0; j < 32; j++) {
                                int k = static_cast<int>(b * Q4_K_VALUES_PER_BLOCK + i + j);
                                if (k < K) {
                                    sum += A_f32[m * K + k] * static_cast<float>(static_cast<int8_t>(qblock[j])) * scale;
                                }
                            }
                        }
                    }
                    C[m * N + n] = sum;
                }
            }
            break;
        }

        case GgmlType::Q6_K: {
            // Q6_K: 64 bytes per block, 256 values per block
            // Mixed quantization: first 128 values Q4, last 128 values Q6
            const float* A_f32 = static_cast<const float*>(params.A);
            const uint8_t* B_q = static_cast<const uint8_t*>(params.B);
            constexpr size_t Q6_K_BLOCK_SIZE = 64;
            constexpr int Q6_K_VALUES_PER_BLOCK = 256;
            size_t num_blocks = K / Q6_K_VALUES_PER_BLOCK;

            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float sum = 0.0f;
                    for (size_t b = 0; b < num_blocks; b++) {
                        const uint8_t* block = B_q + b * Q6_K_BLOCK_SIZE;

                        // Read d and d2 (16-bit little-endian, multiplied by 4)
                        int16_t d_raw = static_cast<int16_t>(block[0] | (block[1] << 8));
                        int16_t d2_raw = static_cast<int16_t>(block[2] | (block[3] << 8));
                        float d = static_cast<float>(d_raw) * 0.25f;
                        float d2 = static_cast<float>(d2_raw) * 0.25f;

                        // Decode 6-byte scale array into 12 scale values (each byte → 2 scales)
                        float scales[12];
                        for (int i = 0; i < 6; i++) {
                            int8_t s = static_cast<int8_t>(block[4 + i]);
                            scales[i * 2] = static_cast<float>(s & 0x0F);
                            scales[i * 2 + 1] = static_cast<float>((s >> 4) & 0x0F);
                        }
                        // First 6 scales * d, last 6 scales * d2
                        for (int i = 0; i < 6; i++) {
                            scales[i] *= d;
                            scales[i + 6] *= d2;
                        }

                        // First 128 values: Q4 with per-group scales (groups of 32, 6 groups)
                        for (int i = 0; i < 128; i += 32) {
                            int scale_idx = (i / 32) % 6;
                            float scale = scales[scale_idx];
                            const uint8_t* qblock = block + 16 + (i / 32) * 16;

                            for (int j = 0; j < 32; j++) {
                                uint8_t nibble = (j % 2 == 0) ? (qblock[j / 2] & 0x0F) : (qblock[j / 2] >> 4);
                                int8_t val = static_cast<int8_t>(nibble - 8);
                                int k = static_cast<int>(b * Q6_K_VALUES_PER_BLOCK + i + j);
                                if (k < K) {
                                    sum += A_f32[m * K + k] * static_cast<float>(val) * scale;
                                }
                            }
                        }

                        // Last 128 values: Q6 with per-group scales (groups of 32, 4 groups)
                        for (int i = 128; i < 256; i += 32) {
                            int block_idx = (i - 128) / 32;
                            float scale = scales[block_idx + 6];
                            const uint8_t* qblock = block + 48 + (i - 128);
                            const uint8_t* hblock = block + 32 + (i - 128);

                            for (int j = 0; j < 32; j++) {
                                int val = static_cast<int>(static_cast<int8_t>(qblock[j])) |
                                          ((hblock[j] & 0x03) << 7);
                                if (val >= 64) val -= 128; // Convert to signed
                                int k = static_cast<int>(b * Q6_K_VALUES_PER_BLOCK + i + j);
                                if (k < K) {
                                    sum += A_f32[m * K + k] * static_cast<float>(val) * scale;
                                }
                            }
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
        // Proper decomposed flash attention v1:
        //   attn = softmax(Q @ K^T / sqrt(d)) @ V
        // Q: [n_head, head_dim] (single query token)
        // K: [n_head, ctx_len, head_dim]
        // V: [n_head, ctx_len, head_dim]
        // Output: [n_head, head_dim]
        if (!params.Q || !params.K || !params.V || !params.output) return Status::INVALID_PARAM;

        int nh = params.n_head;
        int hd = params.head_dim;
        int64_t cl = params.ctx_len;

        float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        for (int h = 0; h < nh; h++) {
            const float* Qh = params.Q + h * hd;
            const float* Kh = params.K + h * cl * hd;
            const float* Vh = params.V + h * cl * hd;
            float* outh = params.output + h * hd;

            // Compute attention weights: scores[j] = Q @ K[j] * scale
            std::vector<float> scores(cl, 0.0f);
            for (int64_t j = 0; j < cl; j++) {
                float sum = 0.0f;
                for (int d = 0; d < hd; d++) {
                    sum += Qh[d] * Kh[j * hd + d];
                }
                scores[j] = sum * scale;
            }

            // Softmax over scores
            float max_val = -INFINITY;
            for (int64_t j = 0; j < cl; j++) {
                if (scores[j] > max_val) max_val = scores[j];
            }
            float sum = 0.0f;
            std::vector<float> weights(cl);
            for (int64_t j = 0; j < cl; j++) {
                weights[j] = std::exp(scores[j] - max_val);
                sum += weights[j];
            }
            for (int64_t j = 0; j < cl; j++) {
                weights[j] /= sum;
            }

            // Weighted V sum: out = sum_j weights[j] * V[j]
            std::fill(outh, outh + hd, 0.0f);
            for (int64_t j = 0; j < cl; j++) {
                for (int d = 0; d < hd; d++) {
                    outh[d] += weights[j] * Vh[j * hd + d];
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
