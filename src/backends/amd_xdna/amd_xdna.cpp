#include "backend.h"
#include "tensor.h"
#include "cache.h"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_hw_context.h>
#include <cstring>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <mutex>
#include <functional>
#include <unordered_map>

namespace ggnpu {

namespace fs = std::filesystem;

namespace {

constexpr xrt::memory_group kDefaultMemGroup = 0;
constexpr const char* kTritonXdnaKernelName = "MLIR_AIE";
constexpr uint32_t kNpuOpcode = 3;
constexpr int kMatmulTile = 256;  // prebuilt matmul xclbin is fixed 256x256x256 int8
// Prebuilt rmsnorm xclbin is 32x256 bf16 (per-row norm over 256). Llama hidden=2048
// needs a dedicated M=1,N=2048 kernel; until then use CPU for other sizes.
constexpr int kRmsnormKernelCols = 256;
constexpr int kSiluKernelSize = 8192;  // prebuilt silu xclbin default N

//====//
// BF16 conversion utilities
// The rmsnorm/softmax/silu kernels are BF16, but the backend sends F32.
// These functions convert between f32 (host) and bf16 (device).
//====//

// Convert a single f32 value to bf16 (round to nearest, ties to even)
static inline uint16_t f32_to_bf16(float f) {
    uint32_t* bits = reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (*bits >> 16) & 0x8000;
    uint32_t exp = (*bits >> 16) & 0x7f80;
    uint32_t frac = (*bits >> 16) & 0x7fff;

    // Handle NaNs
    if (exp == 0x7f80) {
        return sign | 0x7e00 | (frac ? 0x0200 : 0);
    }

    // Handle overflow/underflow to inf/zeros
    if (exp < 0x3880) {  // subnormal or zero -> 0.0
        return sign;
    }
    if (exp > 0x5080) {  // overflow -> inf
        return sign | 0x7f80;
    }

    // Round: take the top 16 bits and round the 17th bit
    uint32_t rounded = (frac & 0x4000) ? ((frac >> 16) + 1) : (frac >> 16);
    return sign | (exp - 0x3800) | rounded;
}

// Convert a single bf16 value to f32
static inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float* result = reinterpret_cast<float*>(&bits);
    return *result;
}

// Convert f32 vector to bf16 vector (in-place, writes to separate buffer)
static std::vector<uint8_t> convert_f32_to_bf16(const void* f32_data, size_t count) {
    const float* f32 = static_cast<const float*>(f32_data);
    std::vector<uint8_t> bf16_buf(count * 2);  // 2 bytes per bf16
    for (size_t i = 0; i < count; i++) {
        uint16_t* bf16_ptr = reinterpret_cast<uint16_t*>(bf16_buf.data() + i * 2);
        *bf16_ptr = f32_to_bf16(f32[i]);
    }
    return bf16_buf;
}

// Convert bf16 vector to f32 vector
static std::vector<float> convert_bf16_to_f32(const void* bf16_data, size_t count) {
    const uint16_t* bf16 = static_cast<const uint16_t*>(bf16_data);
    std::vector<float> f32_buf(count);
    for (size_t i = 0; i < count; i++) {
        f32_buf[i] = bf16_to_f32(bf16[i]);
    }
    return f32_buf;
}

struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const noexcept {
        return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
};

struct TupleHash {
    size_t operator()(const std::tuple<int, int, int64_t>& t) const noexcept {
        size_t h = std::hash<int>{}(std::get<0>(t));
        h ^= std::hash<int>{}(std::get<1>(t)) << 1;
        h ^= std::hash<int64_t>{}(std::get<2>(t)) << 2;
        return h;
    }
};

// NPUs (amdxdna) do not support the legacy device::load_xclbin path
// ("load_axlf: Operation not supported"); register + hw_context is required.
xrt::hw_context register_xclbin_from_data(xrt::device& dev, const std::vector<uint8_t>& data) {
    std::vector<char> copy(data.begin(), data.end());
    xrt::xclbin xbin(copy);
    xrt::uuid uuid = dev.register_xclbin(xbin);
    return xrt::hw_context(dev, uuid);
}

} // namespace

// Forward declarations from kernels.cpp
namespace detail {
    std::vector<uint8_t> load_xclbin_file(const std::string& path);
    std::string find_prebuilt_xclbin(const std::string& xclbin_name, const std::string& cache_dir);
    std::string find_prebuilt_sequence(const std::string& xclbin_name, const std::string& cache_dir);
    std::vector<uint32_t> load_sequence_file(const std::string& path);
    std::vector<uint8_t> jit_compile_matmul(int M, int N, int K, int profile);
    std::vector<uint8_t> jit_compile_rmsnorm(int N, int profile);
    std::vector<uint8_t> jit_compile_rope(int n_dims, int profile);
    std::vector<uint8_t> jit_compile_softmax(int cols, int rows, int profile);
    std::vector<uint8_t> jit_compile_silu(int size, int profile);
    std::vector<uint8_t> jit_compile_flash_attn(int n_head, int head_dim, int64_t ctx_len, int profile);
    std::string make_cache_key(const std::string& op, int M, int N, int K, const std::string& profile);
    bool jit_compilation_available();
}

// Internal buffer wrapper for XRT buffer objects
class XrtBuffer {
public:
    XrtBuffer() = default;
    XrtBuffer(xrt::device& dev, size_t size, xrt::bo::flags flags)
        : bo_(dev, size, flags, kDefaultMemGroup) {}

    void* map() {
        if (!is_valid()) return nullptr;
        return bo_.map<void*>();
    }

    void sync(xclBOSyncDirection dir) {
        if (!is_valid()) return;
        bo_.sync(dir);
    }

    xrt::bo& handle() { return bo_; }
    const xrt::bo& handle() const { return bo_; }
    bool is_valid() const { return static_cast<bool>(bo_); }
    size_t size() const { return bo_.size(); }

private:
    xrt::bo bo_;
};

// Buffer manager for allocating and tracking XRT buffers
class BufferMgr {
public:
    explicit BufferMgr(xrt::device& dev) : device_(dev) {}

    std::shared_ptr<XrtBuffer> alloc(size_t size, bool host_backed = false) {
        xrt::bo::flags flags = host_backed ? xrt::bo::flags::host_only : xrt::bo::flags::normal;
        auto buf = std::make_shared<XrtBuffer>(device_, size, flags);
        buffers_.push_back(buf);
        return buf;
    }

    void copy_to(XrtBuffer& buf, const void* src, size_t count) {
        if (!buf.is_valid()) return;
        void* mapped = buf.map();
        if (mapped) {
            std::memcpy(mapped, src, count);
        }
        buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void copy_from(XrtBuffer& buf, void* dst, size_t count) {
        if (!buf.is_valid()) return;
        buf.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        void* mapped = buf.map();
        if (mapped) {
            std::memcpy(dst, mapped, count);
        }
    }

private:
    xrt::device& device_;
    std::vector<std::shared_ptr<XrtBuffer>> buffers_;
};

// Cached kernel for matmul: holds xrt::run for a specific shape
struct CachedMatmulKernel {
    xrt::run run;
    xrt::bo bo_instr;
    size_t instr_words = 0;
    int M, N, K;
    GgmlType B_type;
};

// Cached kernel for RMSNorm (BF16)
struct CachedRmsNormKernel {
    xrt::run run;
    xrt::kernel krnl;
    xrt::bo bo_instr;
    size_t instr_words = 0;
    int N;
};

// Cached kernel for RoPE
struct CachedRopeKernel {
    xrt::run run;
    int n_dims;
};

// Cached kernel for Softmax (BF16)
struct CachedSoftmaxKernel {
    xrt::run run;
    xrt::kernel krnl;
    xrt::bo bo_instr;
    size_t instr_words = 0;
    int rows, cols;
};

// Cached kernel for SiLU (BF16)
struct CachedSiluKernel {
    xrt::run run;
    xrt::kernel krnl;
    xrt::bo bo_instr;
    size_t instr_words = 0;
    int size;
};

static void rms_norm_cpu_ref(const RmsNormParams& params) {
    float variance = 0.0f;
    for (int i = 0; i < params.size; i++) variance += params.input[i] * params.input[i];
    variance /= params.size;
    variance += params.eps;
    float inv_std = 1.0f / std::sqrt(variance);
    for (int i = 0; i < params.size; i++) {
        float w = params.weight ? params.weight[i] : 1.0f;
        params.output[i] = params.input[i] * inv_std * w;
    }
}

static void silu_cpu_ref(const SiluParams& params) {
    for (int i = 0; i < params.size; i++) {
        float x = params.input[i];
        params.output[i] = x / (1.0f + std::exp(-x));
    }
}

// CPU reference flash attention (used when NPU kernel is unavailable)
static void flash_attn_cpu_ref(const AttnParams& params) {
    int nh = params.n_head;
    int hd = params.head_dim;
    int64_t cl = params.ctx_len;
    int64_t qpos = params.query_pos >= 0 ? params.query_pos : cl - 1;
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int h = 0; h < nh; h++) {
        const float* Qh = params.Q + h * hd;
        const float* Kh = params.K + h * cl * hd;
        const float* Vh = params.V + h * cl * hd;
        float* outh = params.output + h * hd;

        std::vector<float> scores(cl, -INFINITY);
        for (int64_t j = 0; j <= qpos && j < cl; j++) {
            float sum = 0.0f;
            for (int d = 0; d < hd; d++) sum += Qh[d] * Kh[j * hd + d];
            scores[j] = sum * scale;
        }

        float max_val = -INFINITY;
        for (int64_t j = 0; j < cl; j++) if (scores[j] > max_val) max_val = scores[j];

        float sum = 0.0f;
        std::vector<float> weights(cl);
        for (int64_t j = 0; j < cl; j++) {
            weights[j] = std::exp(scores[j] - max_val);
            sum += weights[j];
        }
        for (int64_t j = 0; j < cl; j++) weights[j] /= sum;

        std::fill(outh, outh + hd, 0.0f);
        for (int64_t j = 0; j < cl; j++)
            for (int d = 0; d < hd; d++)
                outh[d] += weights[j] * Vh[j * hd + d];
    }
}

// Cached kernel for FlashAttention
struct CachedFlashAttnKernel {
    xrt::run run;
    int n_head;
    int head_dim;
    int64_t ctx_len;
};

// AMD XDNA NPU backend
class AmdXdnaBackend : public Backend {
public:
    AmdXdnaBackend(int device_id, const std::string& cache_dir = "")
        : last_status_(Status::OK) {

        // Determine cache directory
        if (cache_dir.empty()) {
            const char* home = std::getenv("HOME");
            cache_dir_ = home ? std::string(home) + "/.cache/ggnpu" : "~/.cache/ggnpu";
        } else {
            cache_dir_ = cache_dir;
        }

        // Initialize XRT device
        try {
            device_ = std::make_shared<xrt::device>(device_id);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to open XRT device " << device_id << ": " << e.what() << "\n";
            std::cerr << "Is the AMD NPU driver (amdxdna) loaded?\n";
            std::cerr << "  lsmod | grep amdxdna\n";
            std::cerr << "Is /dev/accel/accel0 accessible?\n";
            std::cerr << "  ls -la /dev/accel/\n";
            last_status_ = Status::NPU_UNAVAILABLE;
            return;
        }

        // Detect NPU profile
        npu_profile_ = detect_npu_profile(device_id);
        if (npu_profile_ == 4) profile_str_ = "npu4";
        else if (npu_profile_ == 5) profile_str_ = "npu5";
        else profile_str_ = "npu6";

        std::cout << "NPU device opened: profile=" << profile_str_ << "\n";

        // Initialize buffer manager
        buf_mgr_ = std::make_shared<BufferMgr>(*device_);

        // Initialize compile cache
        cache_ = std::make_unique<CompileCache>(cache_dir_);

        // Load prebuilt xclbins (each op has its own xclbin — do not short-circuit)
        bool any_kernel_loaded = load_matmul_xclbin();
        any_kernel_loaded = load_rmsnorm_xclbin() || any_kernel_loaded;
        any_kernel_loaded = load_softmax_xclbin() || any_kernel_loaded;
        any_kernel_loaded = load_silu_xclbin() || any_kernel_loaded;
        if (!any_kernel_loaded) {
            std::cerr << "Warning: no NPU kernels available. Prebuilt xclbins not found.\n";
            std::cerr << "  Run with --dump-tensors to test GGUF parsing without NPU.\n";
            std::cerr << "  Prebuilt xclbins are needed, or install Triton-XDNA for JIT compilation.\n";
            std::cerr << "  Run: bash scripts/setup-triton-env.sh && ./scripts/build-kernels.sh npu6\n";
        }
    }

    ~AmdXdnaBackend() override = default;

    Status mul_mat_q(const MulMatParams& params) override {
        if (!params.A || !params.B || !params.C) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        int M = params.M;
        int N = params.N;
        int K = params.K;

        // The prebuilt Triton-XDNA matmul kernel is fixed-shape INT8 in /
        // INT32 out at kMatmulTile^3. Larger problems are tiled on the host
        // (Phase 2 smoke path; not yet performance-optimized).
        std::string cache_key = std::string("matmul_") + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = matmul_kernels_.find(cache_key);
        if (it == matmul_kernels_.end()) {
            if (!ensure_matmul_kernel(kMatmulTile, kMatmulTile, kMatmulTile, params.B_type, cache_key)) {
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            it = matmul_kernels_.find(cache_key);
        }

        auto& kernel = it->second;

        if (!kernel.bo_instr || kernel.instr_words == 0) {
            std::cerr << "Error: matmul instruction sequence (matmul_" << profile_str_
                      << "_sequence.bin) missing from xclbin cache\n";
            last_status_ = Status::NPU_UNAVAILABLE;
            return last_status_;
        }

        constexpr int T = kMatmulTile;
        size_t tile_bytes_in = static_cast<size_t>(T) * T;                  // int8
        size_t tile_bytes_out = static_cast<size_t>(T) * T * sizeof(int32_t);

        if (!buf_a_ || buf_a_->size() < tile_bytes_in) buf_a_ = buf_mgr_->alloc(tile_bytes_in, true);
        if (!buf_b_ || buf_b_->size() < tile_bytes_in) buf_b_ = buf_mgr_->alloc(tile_bytes_in, true);
        if (!buf_c_ || buf_c_->size() < tile_bytes_out) buf_c_ = buf_mgr_->alloc(tile_bytes_out, true);

        const float* A = static_cast<const float*>(params.A);
        float* C = static_cast<float*>(params.C);
        std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);

        std::vector<int8_t> a_tile(static_cast<size_t>(T) * T);
        std::vector<int8_t> b_tile(static_cast<size_t>(T) * T);
        std::vector<int32_t> c_tile(static_cast<size_t>(T) * T);

        auto to_i8 = [](float v) -> int8_t {
            float r = std::nearbyint(v);
            if (r > 127.0f) r = 127.0f;
            if (r < -128.0f) r = -128.0f;
            return static_cast<int8_t>(r);
        };

        // K-quants are decoded host-side to int8 with one per-tensor weight
        // scale. Activations get a per-call dynamic scale; the product of
        // both scales converts the raw INT32 accumulators back to float.
        const bool kq_path = (params.B_type == GgmlType::Q4_K ||
                              params.B_type == GgmlType::Q6_K) && params.scales;
        float a_scale = 1.0f;
        if (kq_path) {
            float a_max = 0.0f;
            for (int i = 0; i < M; i++)
                for (int k = 0; k < K; k++)
                    a_max = std::max(a_max, std::fabs(A[i * params.lda + k]));
            a_scale = a_max > 0.0f ? a_max / 127.0f : 1.0f;
        }
        const float inv_a = 1.0f / a_scale;
        const float w_scale = kq_path ? static_cast<const float*>(params.scales)[0] : 1.0f;
        const float out_scale = a_scale * w_scale;

        for (int m0 = 0; m0 < M; m0 += T) {
            int mc = std::min(T, M - m0);
            for (int n0 = 0; n0 < N; n0 += T) {
                int nc = std::min(T, N - n0);
                for (int k0 = 0; k0 < K; k0 += T) {
                    int kc = std::min(T, K - k0);

                    std::fill(a_tile.begin(), a_tile.end(), 0);
                    for (int i = 0; i < mc; i++)
                        for (int k = 0; k < kc; k++)
                            a_tile[i * T + k] = to_i8(A[(m0 + i) * params.lda + (k0 + k)] * inv_a);

                    std::fill(b_tile.begin(), b_tile.end(), 0);
                    if (params.B_type == GgmlType::I8 ||
                        params.B_type == GgmlType::Q4_0 ||
                        params.B_type == GgmlType::Q4_K ||
                        params.B_type == GgmlType::Q6_K ||
                        params.B_type == GgmlType::Q8_0) {
                        // Already-decoded INT8 weights (from weight cache or raw I8).
                        // Quantized types here mean "decode already done on host".
                        const int8_t* B = static_cast<const int8_t*>(params.B);
                        if (kq_path) {
                            // Decoded buffer keeps GGUF row-major order:
                            // N rows of K values -> B[n * K + k]
                            for (int k = 0; k < kc; k++)
                                for (int j = 0; j < nc; j++)
                                    b_tile[k * T + j] = B[static_cast<size_t>(n0 + j) * K + (k0 + k)];
                        } else {
                            for (int k = 0; k < kc; k++)
                                for (int j = 0; j < nc; j++)
                                    b_tile[k * T + j] = B[(k0 + k) * params.ldb + (n0 + j)];
                        }
                    } else if (params.B_type == GgmlType::F32) {
                        const float* B = static_cast<const float*>(params.B);
                        for (int k = 0; k < kc; k++)
                            for (int j = 0; j < nc; j++)
                                b_tile[k * T + j] = to_i8(B[(k0 + k) * params.ldb + (n0 + j)]);
                    } else {
                        std::cerr << "Error: matmul B_type not supported on NPU path yet\n";
                        last_status_ = Status::INVALID_PARAM;
                        return last_status_;
                    }

                    buf_mgr_->copy_to(*buf_a_, a_tile.data(), tile_bytes_in);
                    buf_mgr_->copy_to(*buf_b_, b_tile.data(), tile_bytes_in);

                    try {
                        kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                                   buf_a_->handle(), buf_b_->handle(), buf_c_->handle());
                        kernel.run.wait();
                    } catch (const std::exception& e) {
                        std::cerr << "Error: matmul kernel execution failed: " << e.what() << "\n";
                        last_status_ = Status::ERROR;
                        return last_status_;
                    }

                    buf_mgr_->copy_from(*buf_c_, c_tile.data(), tile_bytes_out);

                    for (int i = 0; i < mc; i++)
                        for (int j = 0; j < nc; j++)
                            C[(m0 + i) * params.ldc + (n0 + j)] += static_cast<float>(c_tile[i * T + j]) * out_scale;
                }
            }
        }

        return Status::OK;
    }

    Status rms_norm(const RmsNormParams& params) override {
        if (!params.input || !params.output || params.size <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        int N = params.size;
        if (N != kRmsnormKernelCols || params.weight) {
            rms_norm_cpu_ref(params);
            return Status::OK;
        }

        std::string cache_key = "rmsnorm_" + std::to_string(N) + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rmsnorm_kernels_.find(N);
        if (it == rmsnorm_kernels_.end()) {
            if (!ensure_rmsnorm_kernel(N, cache_key)) {
                static bool warned = false;
                if (!warned) {
                    std::cerr << "Warning: rmsnorm NPU kernel unavailable; using CPU fallback\n";
                    warned = true;
                }
                rms_norm_cpu_ref(params);
                return Status::OK;
            }
            it = rmsnorm_kernels_.find(N);
        }

        auto& kernel = it->second;

        // RMSNorm kernels are BF16 — convert f32 input → bf16 for DMA
        size_t bf16_bytes = N * sizeof(uint16_t);
        std::vector<uint8_t> bf16_input;
        xrt::bo buf_bf16_in;

        if (kernel.bo_instr && kernel.instr_words > 0) {
            // BF16 kernel with instruction sequence: convert f32→bf16, pass opcode+instr
            bf16_input = convert_f32_to_bf16(params.input, N);

            buf_bf16_in = xrt::bo(*device_, bf16_bytes,
                                  XCL_BO_FLAGS_CACHEABLE, kernel.krnl.group_id(0));
            void* mapped = buf_bf16_in.map<void*>();
            std::memcpy(mapped, bf16_input.data(), bf16_bytes);
            buf_bf16_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Output buffer (BF16)
            xrt::bo buf_bf16_out(*device_, bf16_bytes,
                                 XCL_BO_FLAGS_CACHEABLE, kernel.krnl.group_id(0));

            try {
                kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                           buf_bf16_in, buf_bf16_out);
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Warning: rmsnorm kernel execution failed (" << e.what()
                          << "); using CPU fallback\n";
                rms_norm_cpu_ref(params);
                return Status::OK;
            }

            // Convert bf16 output → f32
            std::vector<uint8_t> bf16_out_data(bf16_bytes);
            buf_bf16_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            void* mapped_out = buf_bf16_out.map<void*>();
            std::memcpy(bf16_out_data.data(), mapped_out, bf16_bytes);
            auto f32_result = convert_bf16_to_f32(bf16_out_data.data(), N);
            std::memcpy(params.output, f32_result.data(), N * sizeof(float));

        } else {
            // Fallback: F32 kernel without instruction sequence (legacy path)
            size_t size_bytes = N * sizeof(float);
            if (!buf_rmsnorm_in_ || buf_rmsnorm_in_->size() < size_bytes) {
                buf_rmsnorm_in_ = buf_mgr_->alloc(size_bytes, true);
            }
            if (!buf_rmsnorm_out_ || buf_rmsnorm_out_->size() < size_bytes) {
                buf_rmsnorm_out_ = buf_mgr_->alloc(size_bytes, true);
            }

            buf_mgr_->copy_to(*buf_rmsnorm_in_, params.input, size_bytes);

            try {
                kernel.run(buf_rmsnorm_in_->handle(), buf_rmsnorm_out_->handle(),
                           static_cast<uint32_t>(N));
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Warning: rmsnorm kernel execution failed (" << e.what()
                          << "); using CPU fallback\n";
                rms_norm_cpu_ref(params);
                return Status::OK;
            }

            buf_mgr_->copy_from(*buf_rmsnorm_out_, params.output, size_bytes);
        }

        return Status::OK;
    }

    Status rope(const RopeParams& params) override {
        // RoPE is relatively cheap (O(n_dims) per head) compared to matmul.
        // Keep on CPU for now; the NPU kernel infrastructure is in place for Phase 6+.
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
        if (!params.input || !params.output || params.rows <= 0 || params.cols <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        int rows = params.rows;
        int cols = params.cols;
        std::string cache_key = "softmax_" + std::to_string(rows) + "x" + std::to_string(cols) + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto key = std::make_pair(rows, cols);
        auto it = softmax_kernels_.find(key);
        if (it == softmax_kernels_.end()) {
            if (!ensure_softmax_kernel(rows, cols, cache_key)) {
                std::cerr << "Error: softmax NPU kernel unavailable for " << rows << "x" << cols << "\n";
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            it = softmax_kernels_.find(key);
        }

        auto& kernel = it->second;

        // Softmax kernels are BF16 — convert f32 input → bf16 for DMA
        size_t total_elements = rows * cols;
        size_t bf16_bytes = total_elements * sizeof(uint16_t);
        std::vector<uint8_t> bf16_input;
        xrt::bo buf_bf16_in;

        if (kernel.bo_instr && kernel.instr_words > 0) {
            // BF16 kernel with instruction sequence: convert f32→bf16, pass opcode+instr
            bf16_input = convert_f32_to_bf16(params.input, total_elements);

            buf_bf16_in = xrt::bo(*device_, bf16_bytes,
                                  XCL_BO_FLAGS_CACHEABLE, kernel.krnl.group_id(0));
            void* mapped = buf_bf16_in.map<void*>();
            std::memcpy(mapped, bf16_input.data(), bf16_bytes);
            buf_bf16_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            // Output buffer (BF16)
            xrt::bo buf_bf16_out(*device_, bf16_bytes,
                                 XCL_BO_FLAGS_CACHEABLE, kernel.krnl.group_id(0));

            try {
                kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                           buf_bf16_in, buf_bf16_out);
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Error: softmax kernel execution failed: " << e.what() << "\n";
                last_status_ = Status::ERROR;
                return last_status_;
            }

            // Convert bf16 output → f32
            std::vector<uint8_t> bf16_out_data(bf16_bytes);
            buf_bf16_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            void* mapped_out = buf_bf16_out.map<void*>();
            std::memcpy(bf16_out_data.data(), mapped_out, bf16_bytes);
            auto f32_result = convert_bf16_to_f32(bf16_out_data.data(), total_elements);
            std::memcpy(params.output, f32_result.data(), total_elements * sizeof(float));

        } else {
            // Fallback: F32 kernel without instruction sequence (legacy path)
            size_t size_bytes = rows * cols * sizeof(float);
            if (!buf_softmax_in_ || buf_softmax_in_->size() < size_bytes) {
                buf_softmax_in_ = buf_mgr_->alloc(size_bytes, true);
            }
            if (!buf_softmax_out_ || buf_softmax_out_->size() < size_bytes) {
                buf_softmax_out_ = buf_mgr_->alloc(size_bytes, true);
            }

            buf_mgr_->copy_to(*buf_softmax_in_, params.input, size_bytes);

            try {
                kernel.run(buf_softmax_in_->handle(), buf_softmax_out_->handle(),
                           static_cast<uint32_t>(rows), static_cast<uint32_t>(cols));
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Error: softmax kernel execution failed: " << e.what() << "\n";
                last_status_ = Status::ERROR;
                return last_status_;
            }

            buf_mgr_->copy_from(*buf_softmax_out_, params.output, size_bytes);
        }

        return Status::OK;
    }

    Status silu(const SiluParams& params) override {
        if (!params.input || !params.output || params.size <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        // SiLU NPU launch is not yet stable on amdxdna (opcode-3 arity mismatch).
        (void)params.size;
        silu_cpu_ref(params);
        return Status::OK;
    }

    Status flash_attn(const AttnParams& params) override {
        if (!params.Q || !params.K || !params.V || !params.output || params.n_head <= 0 || params.head_dim <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        int n_head = params.n_head;
        int head_dim = params.head_dim;
        int64_t ctx_len = params.ctx_len;
        std::string cache_key = "flash_attn_" + std::to_string(n_head) + "x" + std::to_string(head_dim) + "x" + std::to_string(ctx_len) + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto key = std::make_tuple(n_head, head_dim, ctx_len);
        auto it = flash_attn_kernels_.find(key);
        if (it == flash_attn_kernels_.end()) {
            if (!ensure_flash_attn_kernel(n_head, head_dim, ctx_len, cache_key)) {
                static bool warned = false;
                if (!warned) {
                    std::cerr << "Warning: flash_attn NPU kernel unavailable; using CPU fallback\n";
                    warned = true;
                }
                flash_attn_cpu_ref(params);
                return Status::OK;
            }
            it = flash_attn_kernels_.find(key);
        }

        auto& kernel = it->second;

        size_t size_q = n_head * head_dim * sizeof(float);
        size_t size_k = ctx_len * head_dim * sizeof(float);
        size_t size_v = ctx_len * head_dim * sizeof(float);
        size_t size_out = n_head * head_dim * sizeof(float);

        if (!buf_fa_q_ || buf_fa_q_->size() < size_q) {
            buf_fa_q_ = buf_mgr_->alloc(size_q, true);
        }
        if (!buf_fa_k_ || buf_fa_k_->size() < size_k) {
            buf_fa_k_ = buf_mgr_->alloc(size_k, true);
        }
        if (!buf_fa_v_ || buf_fa_v_->size() < size_v) {
            buf_fa_v_ = buf_mgr_->alloc(size_v, true);
        }
        if (!buf_fa_out_ || buf_fa_out_->size() < size_out) {
            buf_fa_out_ = buf_mgr_->alloc(size_out, true);
        }

        buf_mgr_->copy_to(*buf_fa_q_, params.Q, size_q);
        buf_mgr_->copy_to(*buf_fa_k_, params.K, size_k);
        buf_mgr_->copy_to(*buf_fa_v_, params.V, size_v);

        try {
            kernel.run(buf_fa_q_->handle(), buf_fa_k_->handle(), buf_fa_v_->handle(),
                       buf_fa_out_->handle(),
                       static_cast<uint32_t>(n_head),
                       static_cast<uint32_t>(head_dim),
                       static_cast<uint32_t>(ctx_len));
            kernel.run.wait();
        } catch (const std::exception& e) {
            std::cerr << "Error: flash_attn kernel execution failed: " << e.what() << "\n";
            last_status_ = Status::ERROR;
            return last_status_;
        }

        buf_mgr_->copy_from(*buf_fa_out_, params.output, size_out);

        return Status::OK;
    }

    void sync() override {
        std::lock_guard<std::mutex> lock(mutex_);
        // xrt::run::wait() is called implicitly by the run() call
    }

    bool is_available() const override {
        return last_status_ != Status::NPU_UNAVAILABLE && device_ != nullptr;
    }

    std::string name() const override {
        return "amd_xdna";
    }

    Status last_error() const override {
        return last_status_;
    }

private:
    // Detect NPU profile from PCI info
    int detect_npu_profile(int device_id) {
        try {
            std::string dev_str = std::to_string(device_id);
            std::string sysfs = "/sys/class/xilinx_dma/dma" + dev_str + "/device";

            if (fs::exists(sysfs)) {
                fs::path vendor_path = fs::path(sysfs) / "vendor";
                fs::path device_path = fs::path(sysfs) / "device";

                if (fs::exists(vendor_path) && fs::exists(device_path)) {
                    std::ifstream vfile(vendor_path), dfile(device_path);
                    std::string vendor, dev;
                    if (vfile >> vendor && dfile >> dev) {
                        if (vendor == "0x1022" && dev == "0x17f0") return 6;  // Krackan
                        if (vendor == "0x1022" && (dev == "0x17d0" || dev == "0x17e0")) return 4; // Strix Point
                    }
                }
            }
        } catch (...) {}
        return 6; // Default to npu6 (Krackan)
    }

    // Load or compile the matmul xclbin for the target NPU profile
    bool load_matmul_xclbin() {
        std::string xclbin_name = "matmul_" + profile_str_ + ".xclbin";
        std::string xclbin_path = detail::find_prebuilt_xclbin(xclbin_name, cache_dir_);

        if (xclbin_path.empty() && cache_->has_xclbin(xclbin_name)) {
            xclbin_path = cache_->get_xclbin_path(xclbin_name);
        }

        if (xclbin_path.empty()) {
            // Try JIT compilation with a common shape
            if (detail::jit_compilation_available()) {
                std::vector<uint8_t> xclbin_data = detail::jit_compile_matmul(256, 256, 256, npu_profile_);
                if (!xclbin_data.empty()) {
                    std::string cache_key = detail::make_cache_key("matmul", 256, 256, 256, profile_str_);
                    cache_->store_xclbin(cache_key, xclbin_data);
                    xclbin_path = cache_->get_xclbin_path(cache_key);
                }
            }
        }

        if (xclbin_path.empty()) {
            std::cerr << "Warning: no matmul xclbin found for profile " << profile_str_ << "\n";
            std::cerr << "  Place prebuilt xclbin at: " << xclbin_path << "\n";
            std::cerr << "  Build kernels: bash scripts/setup-triton-env.sh && ./scripts/build-kernels.sh npu6 matmul\n";
            return false;
        }

        try {
            xclbin_data_ = detail::load_xclbin_file(xclbin_path);
            if (xclbin_data_.empty()) {
                std::cerr << "Error: failed to read xclbin: " << xclbin_path << "\n";
                return false;
            }

            hw_ctx_matmul_ = register_xclbin_from_data(*device_, xclbin_data_);
            std::cout << "Loaded xclbin: " << xclbin_path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load xclbin: " << e.what() << "\n";
            return false;
        }
    }

    // Ensure a matmul kernel exists for the given dimensions
    bool ensure_matmul_kernel(int M, int N, int K, GgmlType B_type, const std::string& cache_key) {
        // Check if we have a cached xclbin for this shape
        if (cache_->has_xclbin(cache_key)) {
            std::string cached_path = cache_->get_xclbin_path(cache_key);
            return load_matmul_kernel_for_shape(cached_path, M, N, K, B_type, cache_key);
        }

        // Try to create kernel from the base loaded xclbin (works for any shape if kernel is dimension-agnostic)
        if (hw_ctx_matmul_) {
            return create_matmul_kernel_from_loaded_xclbin(M, N, K, B_type, cache_key);
        }

        // Try JIT compilation
        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_matmul(M, N, K, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_matmul_kernel_for_shape(cache_->get_xclbin_path(cache_key), M, N, K, B_type, cache_key);
            }
        }

        std::cerr << "Error: no xclbin available for matmul " << M << "x" << N << "x" << K << "\n";
        return false;
    }

    bool load_matmul_kernel_for_shape(const std::string& path, int M, int N, int K, GgmlType B_type, const std::string& cache_key) {
        try {
            auto data = detail::load_xclbin_file(path);
            if (data.empty()) return false;

            xrt::hw_context ctx = register_xclbin_from_data(*device_, data);
            matmul_shape_ctxs_.push_back(ctx);

            xrt::kernel krnl(ctx, kTritonXdnaKernelName);
            xrt::run run(krnl);
            CachedMatmulKernel cached{run, {}, 0, M, N, K, B_type};
            std::string seq_path = detail::find_prebuilt_sequence(fs::path(path).filename().string(), cache_dir_);
            if (!seq_path.empty()) {
                auto words = detail::load_sequence_file(seq_path);
                if (!words.empty()) {
                    cached.bo_instr = xrt::bo(*device_, words.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
                    void* mapped = cached.bo_instr.map<void*>();
                    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
                    cached.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    cached.instr_words = words.size();
                }
            }
            matmul_kernels_[cache_key] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load matmul kernel from cache: " << e.what() << "\n";
            return false;
        }
    }

    bool create_matmul_kernel_from_loaded_xclbin(int M, int N, int K, GgmlType B_type, const std::string& cache_key) {
        try {
            xrt::kernel krnl(hw_ctx_matmul_, kTritonXdnaKernelName);
            xrt::run run(krnl);
            CachedMatmulKernel cached{run, {}, 0, M, N, K, B_type};
            std::string seq_path = detail::find_prebuilt_sequence("matmul_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                auto words = detail::load_sequence_file(seq_path);
                if (!words.empty()) {
                    cached.bo_instr = xrt::bo(*device_, words.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
                    void* mapped = cached.bo_instr.map<void*>();
                    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
                    cached.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    cached.instr_words = words.size();
                }
            }
            matmul_kernels_[cache_key] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to create matmul kernel: " << e.what() << "\n";
            return false;
        }
    }

    // Load or compile the rmsnorm xclbin
    bool load_rmsnorm_xclbin() {
        std::string xclbin_name = "rmsnorm_" + profile_str_ + ".xclbin";
        std::string xclbin_path = detail::find_prebuilt_xclbin(xclbin_name, cache_dir_);

        if (xclbin_path.empty() && cache_->has_xclbin(xclbin_name)) {
            xclbin_path = cache_->get_xclbin_path(xclbin_name);
        }

        if (xclbin_path.empty()) {
            if (detail::jit_compilation_available()) {
                std::vector<uint8_t> xclbin_data = detail::jit_compile_rmsnorm(4096, npu_profile_);
                if (!xclbin_data.empty()) {
                    std::string cache_key = detail::make_cache_key("rmsnorm", 4096, 0, 0, profile_str_);
                    cache_->store_xclbin(cache_key, xclbin_data);
                    xclbin_path = cache_->get_xclbin_path(cache_key);
                }
            }
        }

        if (xclbin_path.empty()) {
            std::cerr << "Warning: no rmsnorm xclbin found for profile " << profile_str_ << "\n";
            return false;
        }

        try {
            xclbin_data_rmsnorm_ = detail::load_xclbin_file(xclbin_path);
            if (xclbin_data_rmsnorm_.empty()) {
                std::cerr << "Error: failed to read xclbin: " << xclbin_path << "\n";
                return false;
            }

            hw_ctx_rmsnorm_ = register_xclbin_from_data(*device_, xclbin_data_rmsnorm_);
            std::cout << "Loaded rmsnorm xclbin: " << xclbin_path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load rmsnorm xclbin: " << e.what() << "\n";
            return false;
        }
    }

    bool ensure_rmsnorm_kernel(int N, const std::string& cache_key) {
        if (cache_->has_xclbin(cache_key)) {
            std::string cached_path = cache_->get_xclbin_path(cache_key);
            return load_rmsnorm_kernel_for_shape(cached_path, N, cache_key);
        }

        if (hw_ctx_rmsnorm_) {
            return create_rmsnorm_kernel_from_loaded_xclbin(N, cache_key);
        }

        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_rmsnorm(N, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_rmsnorm_kernel_for_shape(cache_->get_xclbin_path(cache_key), N, cache_key);
            }
        }

        std::cerr << "Error: no xclbin available for rmsnorm N=" << N << "\n";
        return false;
    }

    bool load_rmsnorm_kernel_for_shape(const std::string& path, int N, const std::string& cache_key) {
        try {
            auto data = detail::load_xclbin_file(path);
            if (data.empty()) return false;

            xrt::hw_context ctx = register_xclbin_from_data(*device_, data);
            matmul_shape_ctxs_.push_back(ctx);

            xrt::kernel krnl(ctx, kTritonXdnaKernelName);
            xrt::run run(krnl);

            CachedRmsNormKernel cached{run, krnl, {}, 0, N};
            // Load instruction sequence for BF16 kernel (opcode-3 convention)
            std::string seq_path = detail::find_prebuilt_sequence("rmsnorm_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                auto words = detail::load_sequence_file(seq_path);
                if (!words.empty()) {
                    cached.bo_instr = xrt::bo(*device_, words.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
                    void* mapped = cached.bo_instr.map<void*>();
                    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
                    cached.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    cached.instr_words = words.size();
                }
            }
            rmsnorm_kernels_[N] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load rmsnorm kernel: " << e.what() << "\n";
            return false;
        }
    }

    bool create_rmsnorm_kernel_from_loaded_xclbin(int N, const std::string& cache_key) {
        try {
            xrt::kernel krnl(hw_ctx_rmsnorm_, kTritonXdnaKernelName);
            xrt::run run(krnl);

            CachedRmsNormKernel cached{run, krnl, {}, 0, N};
            // Load instruction sequence for BF16 kernel (opcode-3 convention)
            std::string seq_path = detail::find_prebuilt_sequence("rmsnorm_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                auto words = detail::load_sequence_file(seq_path);
                if (!words.empty()) {
                    cached.bo_instr = xrt::bo(*device_, words.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
                    void* mapped = cached.bo_instr.map<void*>();
                    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
                    cached.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    cached.instr_words = words.size();
                }
            }
            rmsnorm_kernels_[N] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to create rmsnorm kernel: " << e.what() << "\n";
            return false;
        }
    }

    // Load or compile the softmax xclbin
    bool load_softmax_xclbin() {
        std::string xclbin_name = "softmax_" + profile_str_ + ".xclbin";
        std::string xclbin_path = detail::find_prebuilt_xclbin(xclbin_name, cache_dir_);

        if (xclbin_path.empty() && cache_->has_xclbin(xclbin_name)) {
            xclbin_path = cache_->get_xclbin_path(xclbin_name);
        }

        if (xclbin_path.empty()) {
            if (detail::jit_compilation_available()) {
                std::vector<uint8_t> xclbin_data = detail::jit_compile_softmax(1024, 1, npu_profile_);
                if (!xclbin_data.empty()) {
                    std::string cache_key = detail::make_cache_key("softmax", 1, 1024, 0, profile_str_);
                    cache_->store_xclbin(cache_key, xclbin_data);
                    xclbin_path = cache_->get_xclbin_path(cache_key);
                }
            }
        }

        if (xclbin_path.empty()) {
            std::cerr << "Warning: no softmax xclbin found for profile " << profile_str_ << "\n";
            return false;
        }

        try {
            xclbin_data_softmax_ = detail::load_xclbin_file(xclbin_path);
            if (xclbin_data_softmax_.empty()) {
                std::cerr << "Error: failed to read xclbin: " << xclbin_path << "\n";
                return false;
            }

            hw_ctx_softmax_ = register_xclbin_from_data(*device_, xclbin_data_softmax_);
            std::cout << "Loaded softmax xclbin: " << xclbin_path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load softmax xclbin: " << e.what() << "\n";
            return false;
        }
    }

    bool ensure_softmax_kernel(int rows, int cols, const std::string& cache_key) {
        if (cache_->has_xclbin(cache_key)) {
            std::string cached_path = cache_->get_xclbin_path(cache_key);
            return load_softmax_kernel_for_shape(cached_path, rows, cols, cache_key);
        }

        if (hw_ctx_softmax_) {
            return create_softmax_kernel_from_loaded_xclbin(rows, cols, cache_key);
        }

        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_softmax(cols, rows, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_softmax_kernel_for_shape(cache_->get_xclbin_path(cache_key), rows, cols, cache_key);
            }
        }

        std::cerr << "Error: no xclbin available for softmax " << rows << "x" << cols << "\n";
        return false;
    }

    bool load_softmax_kernel_for_shape(const std::string& path, int rows, int cols, const std::string& cache_key) {
        try {
            auto data = detail::load_xclbin_file(path);
            if (data.empty()) return false;

            xrt::hw_context ctx = register_xclbin_from_data(*device_, data);
            matmul_shape_ctxs_.push_back(ctx);

            xrt::kernel krnl(ctx, kTritonXdnaKernelName);
            xrt::run run(krnl);

            CachedSoftmaxKernel cached{run, krnl, {}, 0, rows, cols};
            // Load instruction sequence for BF16 kernel (opcode-3 convention)
            std::string seq_path = detail::find_prebuilt_sequence("softmax_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                auto words = detail::load_sequence_file(seq_path);
                if (!words.empty()) {
                    cached.bo_instr = xrt::bo(*device_, words.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
                    void* mapped = cached.bo_instr.map<void*>();
                    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
                    cached.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    cached.instr_words = words.size();
                }
            }
            softmax_kernels_[std::make_pair(rows, cols)] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load softmax kernel: " << e.what() << "\n";
            return false;
        }
    }

    bool create_softmax_kernel_from_loaded_xclbin(int rows, int cols, const std::string& cache_key) {
        try {
            xrt::kernel krnl(hw_ctx_softmax_, kTritonXdnaKernelName);
            xrt::run run(krnl);

            CachedSoftmaxKernel cached{run, krnl, {}, 0, rows, cols};
            // Load instruction sequence for BF16 kernel (opcode-3 convention)
            std::string seq_path = detail::find_prebuilt_sequence("softmax_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                auto words = detail::load_sequence_file(seq_path);
                if (!words.empty()) {
                    cached.bo_instr = xrt::bo(*device_, words.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
                    void* mapped = cached.bo_instr.map<void*>();
                    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
                    cached.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    cached.instr_words = words.size();
                }
            }
            softmax_kernels_[std::make_pair(rows, cols)] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to create softmax kernel: " << e.what() << "\n";
            return false;
        }
    }

    // Load or compile the silu xclbin
    bool load_silu_xclbin() {
        std::string xclbin_name = "silu_" + profile_str_ + ".xclbin";
        std::string xclbin_path = detail::find_prebuilt_xclbin(xclbin_name, cache_dir_);

        if (xclbin_path.empty() && cache_->has_xclbin(xclbin_name)) {
            xclbin_path = cache_->get_xclbin_path(xclbin_name);
        }

        if (xclbin_path.empty()) {
            if (detail::jit_compilation_available()) {
                std::vector<uint8_t> xclbin_data = detail::jit_compile_silu(8192, npu_profile_);
                if (!xclbin_data.empty()) {
                    std::string cache_key = detail::make_cache_key("silu", 8192, 0, 0, profile_str_);
                    cache_->store_xclbin(cache_key, xclbin_data);
                    xclbin_path = cache_->get_xclbin_path(cache_key);
                }
            }
        }

        if (xclbin_path.empty()) {
            std::cerr << "Warning: no silu xclbin found for profile " << profile_str_ << "\n";
            return false;
        }

        try {
            xclbin_data_silu_ = detail::load_xclbin_file(xclbin_path);
            if (xclbin_data_silu_.empty()) {
                std::cerr << "Error: failed to read xclbin: " << xclbin_path << "\n";
                return false;
            }

            hw_ctx_silu_ = register_xclbin_from_data(*device_, xclbin_data_silu_);
            std::cout << "Loaded silu xclbin: " << xclbin_path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load silu xclbin: " << e.what() << "\n";
            return false;
        }
    }

    bool ensure_silu_kernel(int size, const std::string& cache_key) {
        if (cache_->has_xclbin(cache_key)) {
            std::string cached_path = cache_->get_xclbin_path(cache_key);
            return load_silu_kernel_for_shape(cached_path, size, cache_key);
        }

        if (hw_ctx_silu_) {
            return create_silu_kernel_from_loaded_xclbin(size, cache_key);
        }

        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_silu(size, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_silu_kernel_for_shape(cache_->get_xclbin_path(cache_key), size, cache_key);
            }
        }

        std::cerr << "Error: no xclbin available for silu size=" << size << "\n";
        return false;
    }

    bool load_silu_kernel_for_shape(const std::string& path, int size, const std::string& cache_key) {
        try {
            auto data = detail::load_xclbin_file(path);
            if (data.empty()) return false;

            xrt::hw_context ctx = register_xclbin_from_data(*device_, data);
            matmul_shape_ctxs_.push_back(ctx);

            xrt::kernel krnl(ctx, kTritonXdnaKernelName);
            xrt::run run(krnl);

            CachedSiluKernel cached{run, krnl, {}, 0, size};
            // Load instruction sequence for BF16 kernel (opcode-3 convention)
            std::string seq_path = detail::find_prebuilt_sequence("silu_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                auto words = detail::load_sequence_file(seq_path);
                if (!words.empty()) {
                    cached.bo_instr = xrt::bo(*device_, words.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
                    void* mapped = cached.bo_instr.map<void*>();
                    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
                    cached.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    cached.instr_words = words.size();
                }
            }
            silu_kernels_[size] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load silu kernel: " << e.what() << "\n";
            return false;
        }
    }

    bool create_silu_kernel_from_loaded_xclbin(int size, const std::string& cache_key) {
        try {
            xrt::kernel krnl(hw_ctx_silu_, kTritonXdnaKernelName);
            xrt::run run(krnl);

            CachedSiluKernel cached{run, krnl, {}, 0, size};
            // Load instruction sequence for BF16 kernel (opcode-3 convention)
            std::string seq_path = detail::find_prebuilt_sequence("silu_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                auto words = detail::load_sequence_file(seq_path);
                if (!words.empty()) {
                    cached.bo_instr = xrt::bo(*device_, words.size() * sizeof(uint32_t),
                                              XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
                    void* mapped = cached.bo_instr.map<void*>();
                    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
                    cached.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                    cached.instr_words = words.size();
                }
            }
            silu_kernels_[size] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to create silu kernel: " << e.what() << "\n";
            return false;
        }
    }

    // Load or compile the flash_attn xclbin
    bool load_flash_attn_xclbin() {
        std::string xclbin_name = "flash_attn_" + profile_str_ + ".xclbin";
        std::string xclbin_path = detail::find_prebuilt_xclbin(xclbin_name, cache_dir_);

        if (xclbin_path.empty() && cache_->has_xclbin(xclbin_name)) {
            xclbin_path = cache_->get_xclbin_path(xclbin_name);
        }

        if (xclbin_path.empty()) {
            if (detail::jit_compilation_available()) {
                std::vector<uint8_t> xclbin_data = detail::jit_compile_flash_attn(8, 128, 2048, npu_profile_);
                if (!xclbin_data.empty()) {
                    std::string cache_key = detail::make_cache_key("flash_attn", 8, 128, 2048, profile_str_);
                    cache_->store_xclbin(cache_key, xclbin_data);
                    xclbin_path = cache_->get_xclbin_path(cache_key);
                }
            }
        }

        if (xclbin_path.empty()) {
            std::cerr << "Warning: no flash_attn xclbin found for profile " << profile_str_ << "\n";
            return false;
        }

        try {
            xclbin_data_fa_ = detail::load_xclbin_file(xclbin_path);
            if (xclbin_data_fa_.empty()) {
                std::cerr << "Error: failed to read xclbin: " << xclbin_path << "\n";
                return false;
            }

            hw_ctx_fa_ = register_xclbin_from_data(*device_, xclbin_data_fa_);
            std::cout << "Loaded flash_attn xclbin: " << xclbin_path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load flash_attn xclbin: " << e.what() << "\n";
            return false;
        }
    }

    bool ensure_flash_attn_kernel(int n_head, int head_dim, int64_t ctx_len, const std::string& cache_key) {
        if (cache_->has_xclbin(cache_key)) {
            std::string cached_path = cache_->get_xclbin_path(cache_key);
            return load_flash_attn_kernel_for_shape(cached_path, n_head, head_dim, ctx_len, cache_key);
        }

        if (hw_ctx_fa_) {
            return create_flash_attn_kernel_from_loaded_xclbin(n_head, head_dim, ctx_len, cache_key);
        }

        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_flash_attn(n_head, head_dim, ctx_len, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_flash_attn_kernel_for_shape(cache_->get_xclbin_path(cache_key), n_head, head_dim, ctx_len, cache_key);
            }
        }

        return false;
    }

    bool load_flash_attn_kernel_for_shape(const std::string& path, int n_head, int head_dim, int64_t ctx_len, const std::string& cache_key) {
        try {
            auto data = detail::load_xclbin_file(path);
            if (data.empty()) return false;

            xrt::hw_context ctx = register_xclbin_from_data(*device_, data);
            matmul_shape_ctxs_.push_back(ctx);

            xrt::kernel krnl(ctx, kTritonXdnaKernelName);
            xrt::run run(krnl);

            flash_attn_kernels_[std::make_tuple(n_head, head_dim, ctx_len)] = CachedFlashAttnKernel{run, n_head, head_dim, ctx_len};
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load flash_attn kernel: " << e.what() << "\n";
            return false;
        }
    }

    bool create_flash_attn_kernel_from_loaded_xclbin(int n_head, int head_dim, int64_t ctx_len, const std::string& cache_key) {
        try {
            xrt::kernel krnl(hw_ctx_fa_, kTritonXdnaKernelName);
            xrt::run run(krnl);

            flash_attn_kernels_[std::make_tuple(n_head, head_dim, ctx_len)] = CachedFlashAttnKernel{run, n_head, head_dim, ctx_len};
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to create flash_attn kernel: " << e.what() << "\n";
            return false;
        }
    }

    std::shared_ptr<xrt::device> device_;
    std::shared_ptr<BufferMgr> buf_mgr_;
    std::unique_ptr<CompileCache> cache_;

    std::vector<uint8_t> xclbin_data_;
    xrt::hw_context hw_ctx_matmul_;
    std::vector<xrt::hw_context> matmul_shape_ctxs_;  // keeps per-shape contexts alive

    std::vector<uint8_t> xclbin_data_rmsnorm_;
    xrt::hw_context hw_ctx_rmsnorm_;

    std::vector<uint8_t> xclbin_data_softmax_;
    xrt::hw_context hw_ctx_softmax_;

    std::vector<uint8_t> xclbin_data_silu_;
    xrt::hw_context hw_ctx_silu_;

    std::vector<uint8_t> xclbin_data_fa_;
    xrt::hw_context hw_ctx_fa_;

    std::shared_ptr<XrtBuffer> buf_a_;
    std::shared_ptr<XrtBuffer> buf_b_;
    std::shared_ptr<XrtBuffer> buf_c_;

    std::shared_ptr<XrtBuffer> buf_rmsnorm_in_;
    std::shared_ptr<XrtBuffer> buf_rmsnorm_out_;

    std::shared_ptr<XrtBuffer> buf_softmax_in_;
    std::shared_ptr<XrtBuffer> buf_softmax_out_;

    std::shared_ptr<XrtBuffer> buf_silu_in_;
    std::shared_ptr<XrtBuffer> buf_silu_bf16_in_;
    std::shared_ptr<XrtBuffer> buf_silu_bf16_out_;
    std::shared_ptr<XrtBuffer> buf_silu_out_;

    std::shared_ptr<XrtBuffer> buf_fa_q_;
    std::shared_ptr<XrtBuffer> buf_fa_k_;
    std::shared_ptr<XrtBuffer> buf_fa_v_;
    std::shared_ptr<XrtBuffer> buf_fa_out_;

    std::unordered_map<std::string, CachedMatmulKernel> matmul_kernels_;
    std::unordered_map<int, CachedRmsNormKernel> rmsnorm_kernels_;
    std::unordered_map<std::pair<int, int>, CachedSoftmaxKernel, PairHash> softmax_kernels_;
    std::unordered_map<int, CachedSiluKernel> silu_kernels_;
    std::unordered_map<std::tuple<int, int, int64_t>, CachedFlashAttnKernel, TupleHash> flash_attn_kernels_;

    int npu_profile_ = 6;
    std::string profile_str_ = "npu6";
    std::string cache_dir_;

    Status last_status_;
    mutable std::mutex mutex_;
};

std::shared_ptr<Backend> create_amd_xdna_backend(int device_id) {
    try {
        return std::make_shared<AmdXdnaBackend>(device_id);
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to create AMD XDNA backend: " << e.what() << "\n";
        return nullptr;
    }
}

} // namespace ggnpu
