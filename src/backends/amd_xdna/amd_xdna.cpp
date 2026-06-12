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
// Prebuilt rmsnorm xclbin is M=1,N=2048 bf16 (Llama 3.2 hidden). Legacy 32x256
// xclbins still work for N=256 bench sizes; rebuild with compile_kernels.py for 2048.
constexpr int kRmsnormKernelCols = 256;
constexpr int kRmsnormKernelHidden = 2048;
constexpr int kRmsnormHiddenPadRows = 2;  // M=2,N=2048 (row 0 duplicated; M=1 won't compile)
constexpr int kSiluKernelSize = 8192;  // prebuilt silu xclbin default N
constexpr int kSoftmaxKernelRows = 256;  // prebuilt softmax xclbin is 256x256 bf16
constexpr int kSoftmaxKernelCols = 256;
constexpr int kFlashAttnHeads = 8;      // prebuilt flash_attn: 8 heads, 128 head_dim, 2048 ctx
constexpr int kFlashAttnHeadDim = 128;
constexpr int64_t kFlashAttnCtxLen = 2048;

//====//
// BF16 conversion utilities
// The rmsnorm/softmax/silu kernels are BF16, but the backend sends F32.
// These functions convert between f32 (host) and bf16 (device).
//====//

// Convert a single f32 value to bf16 (round to nearest)
static inline uint16_t f32_to_bf16(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

// Convert a single bf16 value to f32
static inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
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

namespace {

xrt::bo make_data_bo(xrt::device& dev, size_t nbytes) {
    return xrt::bo(dev, nbytes, XCL_BO_FLAGS_CACHEABLE, kDefaultMemGroup);
}

bool load_instr_bo(xrt::device& dev, xrt::kernel& krnl, xrt::bo& bo_instr,
                   size_t& instr_words, const std::string& seq_path) {
    auto words = detail::load_sequence_file(seq_path);
    if (words.empty()) return false;
    bo_instr = xrt::bo(dev, words.size() * sizeof(uint32_t),
                       XCL_BO_FLAGS_CACHEABLE, krnl.group_id(1));
    void* mapped = bo_instr.map<void*>();
    std::memcpy(mapped, words.data(), words.size() * sizeof(uint32_t));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    instr_words = words.size();
    return true;
}

} // namespace

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

struct BTileKey {
    const int8_t* B = nullptr;
    int n0 = 0;
    int k0 = 0;
    int nc = 0;
    int kc = 0;
    int K = 0;

    bool operator==(const BTileKey& o) const {
        return B == o.B && n0 == o.n0 && k0 == o.k0 && nc == o.nc && kc == o.kc && K == o.K;
    }
};

struct BTileKeyHash {
    size_t operator()(const BTileKey& k) const noexcept {
        size_t h = std::hash<const void*>{}(k.B);
        h ^= std::hash<int>{}(k.n0) << 1;
        h ^= std::hash<int>{}(k.k0) << 2;
        h ^= std::hash<int>{}(k.nc) << 3;
        h ^= std::hash<int>{}(k.kc) << 4;
        h ^= std::hash<int>{}(k.K) << 5;
        return h;
    }
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

// Sanity-check NPU rmsnorm_2048 output on first call (mis-built xclbins return zeros).
static bool rmsnorm_npu_matches_cpu(const float* input, const float* npu_out, int N, float eps) {
    std::vector<float> cpu_out(static_cast<size_t>(N));
    RmsNormParams rp{input, cpu_out.data(), N, eps, nullptr};
    rms_norm_cpu_ref(rp);

    float max_diff = 0.0f;
    float max_ref = 0.0f;
    for (int i = 0; i < N; i++) {
        max_diff = std::max(max_diff, std::fabs(cpu_out[static_cast<size_t>(i)] - npu_out[i]));
        max_ref = std::max(max_ref, std::fabs(cpu_out[static_cast<size_t>(i)]));
    }
    if (max_ref < 1e-6f) return true;
    return (max_diff / max_ref) < 0.05f;
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

        // Preload Llama-shaped flash attention if present (optional).
        load_flash_attn_xclbin();
    }

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

        const int ping = matmul_buf_ping_;
        matmul_buf_ping_ = 1 - matmul_buf_ping_;
        if (!buf_a_[ping] || buf_a_[ping]->size() < tile_bytes_in) {
            buf_a_[ping] = buf_mgr_->alloc(tile_bytes_in, true);
        }
        if (!buf_b_[ping] || buf_b_[ping]->size() < tile_bytes_in) {
            buf_b_[ping] = buf_mgr_->alloc(tile_bytes_in, true);
        }
        if (!buf_c_[ping] || buf_c_[ping]->size() < tile_bytes_out) {
            buf_c_[ping] = buf_mgr_->alloc(tile_bytes_out, true);
        }
        auto& buf_a = buf_a_[ping];
        auto& buf_b = buf_b_[ping];
        auto& buf_c = buf_c_[ping];

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
        const float* wscales = kq_path ? static_cast<const float*>(params.scales) : nullptr;
        const bool per_row_w = kq_path && params.n_weight_scales > 1;

        std::vector<float> a_scales(static_cast<size_t>(M), 1.0f);
        if (kq_path) {
            for (int i = 0; i < M; i++) {
                float a_max = 0.0f;
                for (int k = 0; k < K; k++)
                    a_max = std::max(a_max, std::fabs(A[i * params.lda + k]));
                a_scales[static_cast<size_t>(i)] = a_max > 0.0f ? a_max / 127.0f : 1.0f;
            }
        }

        auto weight_scale = [&](int n_idx) -> float {
            if (!kq_path || !wscales) return 1.0f;
            if (per_row_w && n_idx < params.n_weight_scales) return wscales[n_idx];
            return wscales[0];
        };

        for (int m0 = 0; m0 < M; m0 += T) {
            int mc = std::min(T, M - m0);
            for (int n0 = 0; n0 < N; n0 += T) {
                int nc = std::min(T, N - n0);
                for (int k0 = 0; k0 < K; k0 += T) {
                    int kc = std::min(T, K - k0);

                    std::fill(a_tile.begin(), a_tile.end(), 0);
                    for (int i = 0; i < mc; i++) {
                        const float inv_a = 1.0f / a_scales[static_cast<size_t>(m0 + i)];
                        for (int k = 0; k < kc; k++)
                            a_tile[i * T + k] =
                                to_i8(A[(m0 + i) * params.lda + (k0 + k)] * inv_a);
                    }

                    const int8_t* B_int8 = nullptr;
                    if (params.B_type == GgmlType::I8 ||
                        params.B_type == GgmlType::Q4_0 ||
                        params.B_type == GgmlType::Q4_K ||
                        params.B_type == GgmlType::Q6_K ||
                        params.B_type == GgmlType::Q8_0) {
                        B_int8 = static_cast<const int8_t*>(params.B);
                    }

                    BTileKey bkey{B_int8, n0, k0, nc, kc, K};
                    auto b_cached = b_tile_cache_.find(bkey);
                    if (b_cached == b_tile_cache_.end()) {
                        std::fill(b_tile.begin(), b_tile.end(), 0);
                        if (B_int8) {
                            if (kq_path) {
                                for (int k = 0; k < kc; k++)
                                    for (int j = 0; j < nc; j++)
                                        b_tile[k * T + j] =
                                            B_int8[static_cast<size_t>(n0 + j) * K + (k0 + k)];
                            } else {
                                for (int k = 0; k < kc; k++)
                                    for (int j = 0; j < nc; j++)
                                        b_tile[k * T + j] =
                                            B_int8[(k0 + k) * params.ldb + (n0 + j)];
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
                        b_cached = b_tile_cache_.emplace(bkey, b_tile).first;
                    }

                    buf_mgr_->copy_to(*buf_a, a_tile.data(), tile_bytes_in);
                    // Host b_tile_cache_ avoids re-packing; always DMA each tile because
                    // buf_b ping-pong buffers are shared across all keys in one matmul.
                    buf_mgr_->copy_to(*buf_b, b_cached->second.data(), tile_bytes_in);

                    try {
                        kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                                   buf_a->handle(), buf_b->handle(), buf_c->handle());
                        kernel.run.wait();
                    } catch (const std::exception& e) {
                        std::cerr << "Error: matmul kernel execution failed: " << e.what() << "\n";
                        last_status_ = Status::ERROR;
                        return last_status_;
                    }

                    buf_mgr_->copy_from(*buf_c, c_tile.data(), tile_bytes_out);

                    for (int i = 0; i < mc; i++) {
                        const float a_scale = a_scales[static_cast<size_t>(m0 + i)];
                        for (int j = 0; j < nc; j++) {
                            const float out_scale = a_scale * weight_scale(n0 + j);
                            C[(m0 + i) * params.ldc + (n0 + j)] +=
                                static_cast<float>(c_tile[i * T + j]) * out_scale;
                        }
                    }
                }
            }
        }

        return Status::OK;
    }

    Status rms_norm(const RmsNormParams& params) override {
        // RMSNorm on NPU: prebuilt M=1,N=2048 (Llama hidden) or JIT for other sizes.
        // Learned weights are applied on CPU after the unweighted NPU norm (O(N) multiply).

        if (!params.input || !params.output || params.size <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        int N = params.size;
        const float* weight = params.weight;
        std::vector<float> unweighted_out;
        RmsNormParams npu_params = params;
        if (weight) {
            unweighted_out.resize(static_cast<size_t>(N));
            npu_params.weight = nullptr;
            npu_params.output = unweighted_out.data();
        }

        std::string cache_key = "rmsnorm_" + std::to_string(N) + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rmsnorm_kernels_.find(N);
        if (it == rmsnorm_kernels_.end()) {
            if (!ensure_rmsnorm_kernel(N, cache_key)) {
                std::cerr << "Error: no NPU rmsnorm kernel for N=" << N << "\n";
                if (N == kRmsnormKernelHidden) {
                    std::cerr << "  Build: ./scripts/build-kernels.sh npu6 rmsnorm_2048\n";
                    std::cerr << "  Expect: " << cache_dir_ << "/xclbin/rmsnorm_2048_"
                              << profile_str_ << ".xclbin\n";
                }
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            it = rmsnorm_kernels_.find(N);
        }

        auto& kernel = it->second;
        const float* input_ptr = npu_params.input;
        float* output_ptr = npu_params.output;

        // RMSNorm kernels are BF16 — convert f32 input → bf16 for DMA.
        // Llama hidden=2048 uses M=2,N=2048 bf16 kernel; duplicate row 0 into both rows.
        const int npu_rows = (N == kRmsnormKernelHidden) ? kRmsnormHiddenPadRows : 1;
        const size_t row_bf16_bytes = static_cast<size_t>(N) * sizeof(uint16_t);
        const size_t bf16_bytes = static_cast<size_t>(npu_rows) * row_bf16_bytes;
        std::vector<uint8_t> bf16_input;
        xrt::bo buf_bf16_in;

        if (kernel.bo_instr && kernel.instr_words > 0) {
            std::vector<uint8_t> row_bf16 = convert_f32_to_bf16(input_ptr, N);
            bf16_input.resize(bf16_bytes);
            for (int r = 0; r < npu_rows; r++) {
                std::memcpy(bf16_input.data() + static_cast<size_t>(r) * row_bf16_bytes,
                            row_bf16.data(), row_bf16_bytes);
            }

            buf_bf16_in = make_data_bo(*device_, bf16_bytes);
            void* mapped = buf_bf16_in.map<void*>();
            std::memcpy(mapped, bf16_input.data(), bf16_bytes);
            buf_bf16_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            xrt::bo buf_bf16_out = make_data_bo(*device_, bf16_bytes);

            try {
                kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                           buf_bf16_in, buf_bf16_out);
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Error: rmsnorm kernel execution failed: " << e.what() << "\n";
                last_status_ = Status::ERROR;
                return last_status_;
            }

            std::vector<uint8_t> bf16_out_data(bf16_bytes);
            buf_bf16_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            void* mapped_out = buf_bf16_out.map<void*>();
            std::memcpy(bf16_out_data.data(), mapped_out, bf16_bytes);
            auto f32_result = convert_bf16_to_f32(bf16_out_data.data(), N);
            std::memcpy(output_ptr, f32_result.data(), N * sizeof(float));

            if (N == kRmsnormKernelHidden && !rmsnorm_hidden_npu_checked_) {
                rmsnorm_hidden_npu_checked_ = true;
                if (!rmsnorm_npu_matches_cpu(input_ptr, output_ptr, N, npu_params.eps)) {
                    std::cerr << "Error: rmsnorm_2048 NPU kernel produced incorrect output\n";
                    std::cerr << "  The xclbin may be mis-built. Rebuild:\n";
                    std::cerr << "    ./scripts/build-kernels.sh npu6 rmsnorm_2048\n";
                    last_status_ = Status::ERROR;
                    return last_status_;
                }
            }

        } else {
            if (N == kRmsnormKernelHidden) {
                std::cerr << "Error: rmsnorm_2048 xclbin missing BF16 instruction sequence\n";
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            // Legacy F32 kernel without instruction sequence (N=256 bench only)
            size_t size_bytes = N * sizeof(float);
            if (!buf_rmsnorm_in_ || buf_rmsnorm_in_->size() < size_bytes) {
                buf_rmsnorm_in_ = buf_mgr_->alloc(size_bytes, true);
            }
            if (!buf_rmsnorm_out_ || buf_rmsnorm_out_->size() < size_bytes) {
                buf_rmsnorm_out_ = buf_mgr_->alloc(size_bytes, true);
            }

            buf_mgr_->copy_to(*buf_rmsnorm_in_, input_ptr, size_bytes);

            try {
                kernel.run(buf_rmsnorm_in_->handle(), buf_rmsnorm_out_->handle(),
                           static_cast<uint32_t>(N));
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Error: rmsnorm kernel execution failed: " << e.what() << "\n";
                last_status_ = Status::ERROR;
                return last_status_;
            }

            buf_mgr_->copy_from(*buf_rmsnorm_out_, output_ptr, size_bytes);
        }

        if (weight) {
            for (int i = 0; i < N; i++) {
                params.output[i] = unweighted_out[static_cast<size_t>(i)] * weight[i];
            }
        }

        return Status::OK;
    }

    Status rope(const RopeParams& params) override {
        // RoPE is relatively cheap (O(n_dims) per head) compared to matmul.
        // Use CPU for now; the NPU kernel infrastructure is in place for Phase 6+.
        // The NPU kernel path is available via jit_compile_rope() for future use.
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

            buf_bf16_in = make_data_bo(*device_, bf16_bytes);
            void* mapped = buf_bf16_in.map<void*>();
            std::memcpy(mapped, bf16_input.data(), bf16_bytes);
            buf_bf16_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            xrt::bo buf_bf16_out = make_data_bo(*device_, bf16_bytes);

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

        int N = params.size;
        
        // Prebuilt SiLU xclbin is fixed-size 8192 (Llama 1B/3B FFN).
        if (N != kSiluKernelSize) {
            std::cerr << "Error: SiLU NPU kernel is N=" << kSiluKernelSize
                      << " but model requested N=" << N << "\n";
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        std::string cache_key = "silu_" + std::to_string(N) + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = silu_kernels_.find(N);
        if (it == silu_kernels_.end()) {
            if (!ensure_silu_kernel(N, cache_key)) {
                std::cerr << "Error: SiLU NPU kernel unavailable for N=" << N << "\n";
                std::cerr << "  Build: ./scripts/build-kernels.sh npu6 silu\n";
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            it = silu_kernels_.find(N);
        }

        auto& kernel = it->second;

        size_t bf16_bytes = N * sizeof(uint16_t);
        std::vector<uint8_t> bf16_input;

        if (kernel.bo_instr && kernel.instr_words > 0) {
            bf16_input = convert_f32_to_bf16(params.input, N);

            if (!buf_silu_bf16_in_ || buf_silu_bf16_in_->size() < bf16_bytes) {
                buf_silu_bf16_in_ = buf_mgr_->alloc(bf16_bytes, true);
            }
            if (!buf_silu_bf16_out_ || buf_silu_bf16_out_->size() < bf16_bytes) {
                buf_silu_bf16_out_ = buf_mgr_->alloc(bf16_bytes, true);
            }

            buf_mgr_->copy_to(*buf_silu_bf16_in_, bf16_input.data(), bf16_bytes);

            try {
                kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                           buf_silu_bf16_in_->handle(), buf_silu_bf16_out_->handle());
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Error: SiLU kernel execution failed: " << e.what() << "\n";
                last_status_ = Status::ERROR;
                return last_status_;
            }

            std::vector<uint8_t> bf16_out_data(bf16_bytes);
            buf_mgr_->copy_from(*buf_silu_bf16_out_, bf16_out_data.data(), bf16_bytes);
            auto f32_result = convert_bf16_to_f32(bf16_out_data.data(), N);
            std::memcpy(params.output, f32_result.data(), N * sizeof(float));

        } else {
            std::cerr << "Error: SiLU xclbin missing BF16 instruction sequence\n";
            last_status_ = Status::NPU_UNAVAILABLE;
            return last_status_;
        }

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
                std::cerr << "Error: no NPU flash_attn kernel for "
                          << n_head << "x" << head_dim << "x" << ctx_len << "\n";
                std::cerr << "  Build: ./scripts/build-kernels.sh npu6 flash_attn_32x64x2048\n";
                std::cerr << "  Expect: " << cache_dir_ << "/xclbin/flash_attn_"
                          << n_head << "x" << head_dim << "x" << ctx_len << "_"
                          << profile_str_ << ".xclbin\n";
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            it = flash_attn_kernels_.find(key);
        }

        auto& kernel = it->second;

        size_t size_q = static_cast<size_t>(n_head) * head_dim * sizeof(float);
        size_t size_k = static_cast<size_t>(n_head) * ctx_len * head_dim * sizeof(float);
        size_t size_v = static_cast<size_t>(n_head) * ctx_len * head_dim * sizeof(float);
        size_t size_out = static_cast<size_t>(n_head) * head_dim * sizeof(float);

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
                std::vector<uint8_t> xclbin_data =
                    detail::jit_compile_rmsnorm(kRmsnormKernelHidden, npu_profile_);
                if (!xclbin_data.empty()) {
                    std::string cache_key = "rmsnorm_" + std::to_string(kRmsnormKernelHidden) +
                                            "_" + profile_str_;
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

        // Shape-specific cache (e.g. rmsnorm_2048_npu6.xclbin from JIT or manual install).
        std::string shaped_name = "rmsnorm_" + std::to_string(N) + "_" + profile_str_ + ".xclbin";
        std::string shaped_path = detail::find_prebuilt_xclbin(shaped_name, cache_dir_);
        if (!shaped_path.empty()) {
            return load_rmsnorm_kernel_for_shape(shaped_path, N, cache_key);
        }

        // Legacy prebuilt rmsnorm_npu6.xclbin is 32x256 — only safe for N=256.
        if (N == kRmsnormKernelCols && hw_ctx_rmsnorm_) {
            return create_rmsnorm_kernel_from_loaded_xclbin(N, cache_key);
        }

        // N=2048 (and other sizes): JIT-compile M=1,N=<size> kernel.
        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_rmsnorm(N, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_rmsnorm_kernel_for_shape(cache_->get_xclbin_path(cache_key), N, cache_key);
            }
        }

        // Do not use generic rmsnorm_npu6.xclbin for N=2048 unless it was rebuilt as M=1,N=2048
        // (install as rmsnorm_2048_npu6.xclbin). Legacy 32x256 prebuilt gives wrong results at N=2048.

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
            // Use the xclbin filename from the path to find matching shape-specific sequence file
            std::string seq_path = detail::find_prebuilt_sequence(fs::path(path).filename().string(), cache_dir_);
            if (!seq_path.empty()) {
                load_instr_bo(*device_, krnl, cached.bo_instr, cached.instr_words, seq_path);
            }
            if (N == kRmsnormKernelHidden &&
                (!cached.bo_instr || cached.instr_words == 0)) {
                std::cerr << "Error: rmsnorm_2048 xclbin missing instruction sequence\n";
                return false;
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
            std::string seq_path = detail::find_prebuilt_sequence("rmsnorm_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                load_instr_bo(*device_, krnl, cached.bo_instr, cached.instr_words, seq_path);
            }
            rmsnorm_kernels_[N] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to create rmsnorm kernel from loaded xclbin: " << e.what() << "\n";
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

        // Only reuse prebuilt kernel for matching shape (256x256)
        if (rows == kSoftmaxKernelRows && cols == kSoftmaxKernelCols && hw_ctx_softmax_) {
            return create_softmax_kernel_from_loaded_xclbin(rows, cols, cache_key);
        }

        // For other shapes, try JIT compilation (required for size != 256x256)
        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_softmax(cols, rows, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_softmax_kernel_for_shape(cache_->get_xclbin_path(cache_key), rows, cols, cache_key);
            }
        }

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
            std::cerr << "Warning: failed to create softmax kernel from loaded xclbin: " << e.what() << "\n";
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

        // Only reuse prebuilt kernel for matching shape (size=8192)
        if (size == kSiluKernelSize && hw_ctx_silu_) {
            return create_silu_kernel_from_loaded_xclbin(size, cache_key);
        }

        // For other sizes, try JIT compilation (required for size != 8192)
        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_silu(size, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_silu_kernel_for_shape(cache_->get_xclbin_path(cache_key), size, cache_key);
            }
        }

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
                load_instr_bo(*device_, krnl, cached.bo_instr, cached.instr_words, seq_path);
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
            std::string seq_path = detail::find_prebuilt_sequence("silu_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                load_instr_bo(*device_, krnl, cached.bo_instr, cached.instr_words, seq_path);
            }
            silu_kernels_[size] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to create silu kernel from loaded xclbin: " << e.what() << "\n";
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

        std::string shaped_name = "flash_attn_" + std::to_string(n_head) + "x" +
                                  std::to_string(head_dim) + "x" + std::to_string(ctx_len) + "_" +
                                  profile_str_ + ".xclbin";
        std::string shaped_path = detail::find_prebuilt_xclbin(shaped_name, cache_dir_);
        if (!shaped_path.empty()) {
            return load_flash_attn_kernel_for_shape(shaped_path, n_head, head_dim, ctx_len, cache_key);
        }

        // Reuse generic prebuilt kernel for matching shape (8 heads, 128 head_dim, 2048 ctx).
        if (n_head == kFlashAttnHeads && head_dim == kFlashAttnHeadDim && ctx_len == kFlashAttnCtxLen &&
            hw_ctx_fa_) {
            return create_flash_attn_kernel_from_loaded_xclbin(n_head, head_dim, ctx_len, cache_key);
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
            flash_attn_kernels_[std::make_tuple(n_head, head_dim, ctx_len)] =
                CachedFlashAttnKernel{run, n_head, head_dim, ctx_len};
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to create flash_attn kernel from loaded xclbin: " << e.what() << "\n";
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

    std::shared_ptr<XrtBuffer> buf_a_[2];
    std::shared_ptr<XrtBuffer> buf_b_[2];
    std::shared_ptr<XrtBuffer> buf_c_[2];
    int matmul_buf_ping_ = 0;
    std::unordered_map<BTileKey, std::vector<int8_t>, BTileKeyHash> b_tile_cache_;

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
    bool rmsnorm_hidden_npu_checked_ = false;
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
