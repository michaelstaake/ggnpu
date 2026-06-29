#include "backend.h"
#include "tensor.h"
#include "cache.h"
#include "bf16.h"
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
#include <unordered_set>
#include <chrono>

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
constexpr int kRopePairs = 32;          // prebuilt rope xclbin: n_pairs=32 (head_dim=64, Llama 1B/3B)
constexpr int kFlashAttnHeads = 8;      // prebuilt flash_attn: 8 heads, 128 head_dim, 2048 ctx
constexpr int kFlashAttnHeadDim = 128;
constexpr int64_t kFlashAttnCtxLen = 2048;

// BF16 conversion utilities are in src/utils/bf16.cpp (shared header: ggnpu/bf16.h)
// The rmsnorm/softmax/silu kernels are BF16, but the backend sends F32.
// These functions convert between f32 (host) and bf16 (device).

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

// Cached kernel for matmul: one xrt::kernel; pipeline slots hold per-tile runs.
struct CachedMatmulKernel {
    xrt::kernel krnl;
    xrt::bo bo_instr;
    size_t instr_words = 0;
    int M, N, K;
    GgmlType B_type;
};

// Cached BF16 matmul kernel (fixed 256x256x256 tile; host tiles larger problems).
struct CachedMatmulBf16Kernel {
    xrt::run run;
    xrt::kernel krnl;
    xrt::bo bo_instr;
    size_t instr_words = 0;
    bool loaded = false;
};

// Per-tile pipeline slot for overlapped matmul submission.
struct MatmulPipelineSlot {
    std::shared_ptr<XrtBuffer> buf_a;
    std::shared_ptr<XrtBuffer> buf_b;
    std::shared_ptr<XrtBuffer> buf_c;
    xrt::run run;
    bool run_initialized = false;
};

// Cached kernel for RMSNorm (BF16)
struct CachedRmsNormKernel {
    xrt::run run;
    xrt::kernel krnl;
    xrt::bo bo_instr;
    size_t instr_words = 0;
    int N;
};

// Cached kernel for RoPE (BF16)
struct CachedRopeKernel {
    xrt::run run;
    xrt::kernel krnl;
    xrt::bo bo_instr;
    size_t instr_words = 0;
    int n_dims;
    int N;
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

// Global B-tile cache: persists across mul_mat_q() calls.
// Keyed by (weight_pointer, n0, k0, nc, kc, K) so the same weight tile
// is only packed once and reused across Q/K/V/O projections.
static std::unordered_map<BTileKey, std::vector<int8_t>, BTileKeyHash>& get_global_b_tile_cache() {
    static std::unordered_map<BTileKey, std::vector<int8_t>, BTileKeyHash> cache;
    return cache;
}

// Smallest power of 2 >= n (n >= 1). RMSNorm kernels need BLOCK_N = power of 2
// (Triton tl.arange constraint); non-pow2 hidden sizes pad up to this.
static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

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

static std::vector<float> roundtrip_bf16_f32(const float* input, int N) {
    auto bf16 = convert_f32_to_bf16(input, static_cast<size_t>(N));
    return convert_bf16_to_f32(bf16.data(), static_cast<size_t>(N));
}

// Validation-only reference (not a production fallback). Compare against bf16-quantized
// input/output to match the NPU DMA path (bf16 kernel + host f32↔bf16 marshaling).
static bool rmsnorm_npu_matches_cpu(const float* input, const float* npu_out, int N, float eps) {
    auto qinput = roundtrip_bf16_f32(input, N);
    std::vector<float> cpu_out(static_cast<size_t>(N));
    RmsNormParams rp{qinput.data(), cpu_out.data(), N, eps, nullptr};
    rms_norm_cpu_ref(rp);

    // npu_out is already f32 expanded from the bf16 DMA buffer — do not roundtrip again.
    constexpr float kRtol = 0.01f;
    constexpr float kAtol = 0.02f;  // bf16 output quant on large activations (~18 peak)
    float max_diff = 0.0f;
    float max_ref = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        float ref = cpu_out[static_cast<size_t>(i)];
        float diff = std::fabs(ref - npu_out[i]);
        max_diff = std::max(max_diff, diff);
        max_ref = std::max(max_ref, std::fabs(ref));
        float tol = kAtol + kRtol * std::fabs(ref);
        if (diff > tol) mismatches++;
    }
    if (max_ref < 1e-6f) return true;
    // Global gate: typical <0.5% on random data; allow rare outliers on large activations.
    if (max_diff / max_ref >= 0.012f) return false;
    return mismatches <= std::max(1, N / 512);
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
        load_rope_xclbin();  // optional — rope is Phase 6; no warning if absent
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

        // Batch size: number of tiles to submit before waiting.
        // Reduces synchronization overhead from N tiles → N/batch_size waits.
        // Configurable via GGNPU_MATMUL_BATCH_SIZE env var (default 8).
        int batch_size = 8;
        const char* batch_env = std::getenv("GGNPU_MATMUL_BATCH_SIZE");
        if (batch_env && *batch_env) {
            batch_size = std::atoi(batch_env);
            if (batch_size <= 0) batch_size = 1;
            if (batch_size > 64) batch_size = 64;  // cap to avoid excessive memory
        }

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
        for (int i = 0; i < M; i++) {
            float a_max = 0.0f;
            for (int k = 0; k < K; k++)
                a_max = std::max(a_max, std::fabs(A[i * params.lda + k]));
            a_scales[static_cast<size_t>(i)] = a_max > 0.0f ? a_max / 127.0f : 1.0f;
        }

        float b_scale = 1.0f;
        if (!kq_path && params.B_type == GgmlType::F32) {
            const float* B = static_cast<const float*>(params.B);
            float b_max = 0.0f;
            for (int k = 0; k < K; k++)
                for (int j = 0; j < N; j++)
                    b_max = std::max(b_max, std::fabs(B[k * params.ldb + j]));
            b_scale = b_max > 0.0f ? b_max / 127.0f : 1.0f;
        }

        auto weight_scale = [&](int n_idx) -> float {
            if (!kq_path || !wscales) return 1.0f;
            if (per_row_w && n_idx < params.n_weight_scales) return wscales[n_idx];
            return wscales[0];
        };

        //====//
        // Batch tile execution: pack + DMA + submit all tiles, then wait once
        // per batch. Each slot has its own xrt::run and A/B/C buffers so the NPU
        // can overlap kernel execution while the host prepares later tiles.
        //====//

        // Timing accumulators for profiling (only used when GGNPU_MATMUL_TIMING=1)
        double total_dma_a_ms = 0, total_dma_b_ms = 0, total_kernel_ms = 0,
               total_dma_c_ms = 0, total_pack_ms = 0, total_submit_ms = 0, total_wait_ms = 0;
        int tile_count = 0;
        bool do_timing = (std::getenv("GGNPU_MATMUL_TIMING") != nullptr);

        const int8_t* B_int8_base = nullptr;
        if (params.B_type == GgmlType::I8 || params.B_type == GgmlType::Q4_0 ||
            params.B_type == GgmlType::Q4_K || params.B_type == GgmlType::Q6_K ||
            params.B_type == GgmlType::Q8_0) {
            B_int8_base = static_cast<const int8_t*>(params.B);
        }

        auto resolve_b_tile = [&](const BTileKey& key) -> const std::vector<int8_t>& {
            auto& global_cache = get_global_b_tile_cache();
            auto b_cached = global_cache.find(key);
            if (b_cached != global_cache.end()) return b_cached->second;

            std::fill(b_tile.begin(), b_tile.end(), 0);
            if (B_int8_base) {
                if (kq_path) {
                    for (int k = 0; k < key.kc; k++)
                        for (int j = 0; j < key.nc; j++)
                            b_tile[k * T + j] =
                                B_int8_base[static_cast<size_t>(key.n0 + j) * K + (key.k0 + k)];
                } else {
                    for (int k = 0; k < key.kc; k++)
                        for (int j = 0; j < key.nc; j++)
                            b_tile[k * T + j] =
                                B_int8_base[(key.k0 + k) * params.ldb + (key.n0 + j)];
                }
            } else if (params.B_type == GgmlType::F32) {
                const float* B = static_cast<const float*>(params.B);
                const float inv_b = 1.0f / b_scale;
                for (int k = 0; k < key.kc; k++)
                    for (int j = 0; j < key.nc; j++)
                        b_tile[k * T + j] =
                            to_i8(B[(key.k0 + k) * params.ldb + (key.n0 + j)] * inv_b);
            } else {
                static const std::vector<int8_t> kEmptyBTile;
                return kEmptyBTile;
            }
            auto [it, _] = global_cache.emplace(key, b_tile);
            return it->second;
        };

        struct TileWork {
            int m0, mc, n0, nc, k0, kc;
            BTileKey bkey;
            bool is_kq_path;
        };

        std::vector<TileWork> batch;
        batch.reserve(static_cast<size_t>(batch_size));

        auto flush_batch = [&]() -> Status {
            if (batch.empty()) return Status::OK;

            const int n = static_cast<int>(batch.size());
            ensure_matmul_pipeline_slots(kernel.krnl, n, tile_bytes_in, tile_bytes_out);

            auto batch_submit_start = do_timing ? std::chrono::high_resolution_clock::now()
                                                : std::chrono::high_resolution_clock::time_point{};

            // Phase 1: pack, DMA, and submit all tiles (no waits).
            for (int i = 0; i < n; i++) {
                const auto& work = batch[static_cast<size_t>(i)];
                auto& slot = matmul_slots_[static_cast<size_t>(i)];

                auto t_pack_start = do_timing ? std::chrono::high_resolution_clock::now()
                                              : std::chrono::high_resolution_clock::time_point{};
                std::fill(a_tile.begin(), a_tile.end(), 0);
                for (int row = 0; row < work.mc; row++) {
                    const float inv_a = 1.0f / a_scales[static_cast<size_t>(work.m0 + row)];
                    for (int k = 0; k < work.kc; k++)
                        a_tile[row * T + k] =
                            to_i8(A[(work.m0 + row) * params.lda + (work.k0 + k)] * inv_a);
                }
                if (do_timing) {
                    total_pack_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_pack_start).count();
                }

                if (!B_int8_base && params.B_type != GgmlType::F32) {
                    std::cerr << "Error: matmul B_type not supported on NPU path yet\n";
                    last_status_ = Status::INVALID_PARAM;
                    return Status::INVALID_PARAM;
                }
                const std::vector<int8_t>& b_data = resolve_b_tile(work.bkey);
                if (b_data.empty() && params.B_type != GgmlType::F32) {
                    last_status_ = Status::INVALID_PARAM;
                    return Status::INVALID_PARAM;
                }

                auto t_dma_a_start = do_timing ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};
                // DMA only the real M rows of A. Rows [mc, T) in the device buffer
                // hold stale data and produce output rows we never read back, so
                // skipping them is safe and saves up to T/mc (256x for decode).
                buf_mgr_->copy_to(*slot.buf_a, a_tile.data(),
                                  static_cast<size_t>(work.mc) * T);
                if (do_timing) {
                    total_dma_a_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_dma_a_start).count();
                }

                auto t_dma_b_start = do_timing ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};
                buf_mgr_->copy_to(*slot.buf_b, b_data.data(), tile_bytes_in);
                if (do_timing) {
                    total_dma_b_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_dma_b_start).count();
                }

                try {
                    slot.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                             slot.buf_a->handle(), slot.buf_b->handle(), slot.buf_c->handle());
                } catch (const std::exception& e) {
                    std::cerr << "Error: matmul kernel execution failed: " << e.what() << "\n";
                    last_status_ = Status::ERROR;
                    return Status::ERROR;
                }
            }

            if (do_timing) {
                total_submit_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - batch_submit_start).count();
            }

            auto batch_wait_start = do_timing ? std::chrono::high_resolution_clock::now()
                                              : std::chrono::high_resolution_clock::time_point{};

            // Phase 2: wait, read back, and accumulate (kernels may overlap on device).
            for (int i = 0; i < n; i++) {
                const auto& work = batch[static_cast<size_t>(i)];
                auto& slot = matmul_slots_[static_cast<size_t>(i)];

                auto t_kernel_start = do_timing ? std::chrono::high_resolution_clock::now()
                                                : std::chrono::high_resolution_clock::time_point{};
                slot.run.wait();
                if (do_timing) {
                    total_kernel_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_kernel_start).count();
                }

                auto t_dma_c_start = do_timing ? std::chrono::high_resolution_clock::now()
                                               : std::chrono::high_resolution_clock::time_point{};
                // Read back only the real M rows of C (rows [mc, T) are ignored).
                buf_mgr_->copy_from(*slot.buf_c, c_tile.data(),
                                    static_cast<size_t>(work.mc) * T * sizeof(int32_t));
                if (do_timing) {
                    total_dma_c_ms += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_dma_c_start).count();
                }

                for (int row = 0; row < work.mc; row++) {
                    const float a_scale = a_scales[static_cast<size_t>(work.m0 + row)];
                    for (int j = 0; j < work.nc; j++) {
                        float out_scale = a_scale * weight_scale(work.n0 + j);
                        if (!work.is_kq_path && params.B_type == GgmlType::F32) {
                            out_scale *= b_scale;
                        }
                        C[(work.m0 + row) * params.ldc + (work.n0 + j)] +=
                            static_cast<float>(c_tile[row * T + j]) * out_scale;
                    }
                }

                tile_count++;
            }

            if (do_timing) {
                total_wait_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - batch_wait_start).count();
                std::cerr << "batch " << n << " tiles: submit=" << std::chrono::duration<double, std::milli>(
                    batch_wait_start - batch_submit_start).count()
                          << "ms wait+accum=" << std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - batch_wait_start).count() << "ms\n";
            }

            batch.clear();
            return Status::OK;
        };

        for (int m0 = 0; m0 < M; m0 += T) {
            int mc = std::min(T, M - m0);
            for (int n0 = 0; n0 < N; n0 += T) {
                int nc = std::min(T, N - n0);
                for (int k0 = 0; k0 < K; k0 += T) {
                    int kc = std::min(T, K - k0);

                    TileWork work{m0, mc, n0, nc, k0, kc, {B_int8_base, n0, k0, nc, kc, K}, kq_path};
                    batch.push_back(work);

                    // Flush batch when full or at last tile
                    if (static_cast<int>(batch.size()) >= batch_size || 
                        (m0 + mc >= M && n0 + nc >= N && k0 + kc >= K)) {
                        Status st = flush_batch();
                        if (st != Status::OK) return st;
                    }
                }
            }
        }

        // Print summary timing when GGNPU_MATMUL_TIMING=1
        if (do_timing && tile_count > 0) {
            double total_ms = total_dma_a_ms + total_dma_b_ms + total_kernel_ms +
                              total_dma_c_ms + total_pack_ms;
            std::cerr << "\n=== mul_mat_q timing summary (" << tile_count << " tiles) ===\n";
            std::cerr << "  pack:        " << (total_pack_ms / tile_count) << " ms/tile\n";
            std::cerr << "  dmaA:        " << (total_dma_a_ms / tile_count) << " ms/tile\n";
            std::cerr << "  dmaB:        " << (total_dma_b_ms / tile_count) << " ms/tile\n";
            std::cerr << "  kernel wait: " << (total_kernel_ms / tile_count) << " ms/tile\n";
            std::cerr << "  dmaC:        " << (total_dma_c_ms / tile_count) << " ms/tile\n";
            std::cerr << "  per-tile:    " << (total_ms / tile_count) << " ms/tile\n";
            std::cerr << "  batch submit wall: " << total_submit_ms << " ms total\n";
            std::cerr << "  batch wait wall:   " << total_wait_ms << " ms total\n";
            std::cerr << "========================================\n\n";
        }

        return Status::OK;
    }

    Status rms_norm(const RmsNormParams& params) override {
        // RMSNorm on NPU: rmsnorm_2048_npu6.xclbin (M=2,N=2048 bf16, BLOCK_N=256 tiles).
        // First call for N=2048 validates against bf16-roundtrip f32 reference (<1% rel).
        // Learned weights (γ) are applied on the host after the unweighted norm.

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

        // The BF16 kernel's BLOCK_N must be a power of 2 (Triton tl.arange). For
        // non-pow2 hidden sizes (e.g. 1536) pad up to N_pad = next_pow2(N) and use
        // the N_pad kernel. The kernel divides the sum-of-squares by N_pad, so its
        // rstd is too large by sqrt(N_pad/N); correct it with a CONSTANT output
        // factor c = sqrt(N/N_pad). Zero padding contributes nothing to the sum.
        // The input is NOT scaled, so it stays on the same bf16 grid as the
        // reference (full accuracy); the only approximation is that eps is
        // effectively scaled by N_pad/N (~1e-6 relative, negligible). c == 1 for
        // pow2 N, making this an exact no-op for hidden=2048.
        const int N_pad = next_pow2(N);
        const float rms_out_corr = std::sqrt(static_cast<float>(N) / static_cast<float>(N_pad));

        std::string cache_key = "rmsnorm_" + std::to_string(N_pad) + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rmsnorm_kernels_.find(N_pad);
        if (it == rmsnorm_kernels_.end()) {
            if (!ensure_rmsnorm_kernel(N_pad, cache_key)) {
                std::cerr << "Error: no NPU rmsnorm kernel for N=" << N
                          << " (padded to " << N_pad << ")\n";
                if (detail::jit_compilation_available()) {
                    std::cerr << "  JIT compilation failed — check Triton-XDNA environment.\n";
                } else {
                    std::cerr << "  Install Triton-XDNA for JIT: pip install triton-xdna\n";
                    std::cerr << "  Or build prebuilt: ./scripts/build-kernels.sh npu6 rmsnorm_"
                              << N_pad << "\n";
                }
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            it = rmsnorm_kernels_.find(N_pad);
        }

        auto& kernel = it->second;
        const float* input_ptr = npu_params.input;
        float* output_ptr = npu_params.output;

        // RMSNorm kernels are BF16 — convert f32 input → bf16 for DMA. The pow2
        // kernels are compiled M=2 (M=1 won't compile); duplicate row 0 into both.
        const int npu_rows = (N_pad >= 512) ? kRmsnormHiddenPadRows : 1;
        const size_t row_bf16_bytes = static_cast<size_t>(N_pad) * sizeof(uint16_t);
        const size_t bf16_bytes = static_cast<size_t>(npu_rows) * row_bf16_bytes;
        std::vector<uint8_t> bf16_input;

        if (kernel.bo_instr && kernel.instr_words > 0) {
            // Zero-padded f32 row of N_pad cols (input unscaled), then bf16.
            std::vector<float> padded_row(static_cast<size_t>(N_pad), 0.0f);
            for (int i = 0; i < N; i++) padded_row[i] = input_ptr[i];
            std::vector<uint8_t> row_bf16 = convert_f32_to_bf16(padded_row.data(), N_pad);
            bf16_input.resize(bf16_bytes);
            for (int r = 0; r < npu_rows; r++) {
                std::memcpy(bf16_input.data() + static_cast<size_t>(r) * row_bf16_bytes,
                            row_bf16.data(), row_bf16_bytes);
            }

            if (!buf_rmsnorm_bf16_in_ || buf_rmsnorm_bf16_in_->size() < bf16_bytes) {
                buf_rmsnorm_bf16_in_ = buf_mgr_->alloc(bf16_bytes, true);
            }
            if (!buf_rmsnorm_bf16_out_ || buf_rmsnorm_bf16_out_->size() < bf16_bytes) {
                buf_rmsnorm_bf16_out_ = buf_mgr_->alloc(bf16_bytes, true);
            }

            buf_mgr_->copy_to(*buf_rmsnorm_bf16_in_, bf16_input.data(), bf16_bytes);

            try {
                kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                           buf_rmsnorm_bf16_in_->handle(), buf_rmsnorm_bf16_out_->handle());
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Error: rmsnorm kernel execution failed: " << e.what() << "\n";
                last_status_ = Status::ERROR;
                return last_status_;
            }

            std::vector<uint8_t> bf16_out_data(bf16_bytes);
            buf_mgr_->copy_from(*buf_rmsnorm_bf16_out_, bf16_out_data.data(), bf16_bytes);
            auto f32_result = convert_bf16_to_f32(bf16_out_data.data(), N_pad);
            for (int i = 0; i < N; i++) output_ptr[i] = f32_result[i] * rms_out_corr;

            if (rmsnorm_validated_.find(N) == rmsnorm_validated_.end()) {
                rmsnorm_validated_.insert(N);
                if (!rmsnorm_npu_matches_cpu(input_ptr, output_ptr, N, npu_params.eps)) {
                    std::cerr << "Error: rmsnorm NPU kernel failed bf16-aware validation for N="
                              << N << " (padded to " << N_pad << ")\n";
                    std::cerr << "  Rebuild: ./scripts/build-kernels.sh npu6 rmsnorm_" << N_pad << "\n";
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

    // BF16 GEMM C[M,N] = A[M,K] @ B[K,N] via the fixed 256^3 matmul_bf16 kernel.
    // Host tiles M/N/K into 256-blocks and zero-pads the edges; K-blocks are
    // accumulated in f32 on the host. Building block for decomposed NPU attention.
    Status matmul_bf16(const float* A, const float* B, float* C,
                       int M, int N, int K) override {
        if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!ensure_matmul_bf16_kernel()) {
            std::cerr << "Error: no NPU matmul_bf16 kernel.\n"
                      << "  Build: GGNPU_EXPERIMENTAL=1 ./scripts/build-kernels.sh npu6 matmul_bf16\n";
            last_status_ = Status::NPU_UNAVAILABLE;
            return last_status_;
        }
        auto& kernel = matmul_bf16_kernel_;

        constexpr int T = kMatmulTile;            // 256
        const size_t in_bytes  = static_cast<size_t>(T) * T * sizeof(uint16_t);
        const size_t out_bytes = static_cast<size_t>(T) * T * sizeof(float);
        if (!buf_mmbf16_a_ || buf_mmbf16_a_->size() < in_bytes)
            buf_mmbf16_a_ = buf_mgr_->alloc(in_bytes, true);
        if (!buf_mmbf16_b_ || buf_mmbf16_b_->size() < in_bytes)
            buf_mmbf16_b_ = buf_mgr_->alloc(in_bytes, true);
        if (!buf_mmbf16_c_ || buf_mmbf16_c_->size() < out_bytes)
            buf_mmbf16_c_ = buf_mgr_->alloc(out_bytes, true);

        const int Mt = (M + T - 1) / T;
        const int Nt = (N + T - 1) / T;
        const int Kt = (K + T - 1) / T;

        // Optional phase profiling (GGNPU_NPU_ATTN_TIMING=1): accumulate across
        // all calls, print a summary every 256 launches.
        static const bool do_timing = std::getenv("GGNPU_NPU_ATTN_TIMING") != nullptr;
        static long g_launches = 0;
        static double g_pack = 0, g_dma_in = 0, g_run = 0, g_dma_out = 0;
        using clk = std::chrono::high_resolution_clock;
        auto ms_since = [](clk::time_point t) {
            return std::chrono::duration<double, std::milli>(clk::now() - t).count();
        };

        // Persistent staging (zeroed once on alloc, padding stays zero across
        // calls): only the real sub-region of each tile is converted/written, so
        // packing cost scales with the real data (ctx*head_dim), not the 256^2 tile.
        const size_t tile_elems = static_cast<size_t>(T) * T;
        if (mmbf16_stage_a_.size() != tile_elems) {
            mmbf16_stage_a_.assign(tile_elems, 0);  // bf16 stored as uint16_t
            mmbf16_stage_b_.assign(tile_elems, 0);
            mmbf16_cacc_.assign(tile_elems, 0.0f);
            mmbf16_craw_.assign(out_bytes, 0);
            // Staging is all-zero; no prior real region to clear.
            mmbf16_pa_rows_ = mmbf16_pa_cols_ = mmbf16_pb_rows_ = mmbf16_pb_cols_ = 0;
        }
        uint16_t* stage_a = mmbf16_stage_a_.data();
        uint16_t* stage_b = mmbf16_stage_b_.data();
        std::vector<float>& c_acc = mmbf16_cacc_;
        std::vector<uint8_t>& c_raw = mmbf16_craw_;
        // prev_* (persistent members) track the last-written real extent of each
        // stage so we only re-zero the padding border that shrank — across calls.
        int& prev_arows = mmbf16_pa_rows_; int& prev_acols = mmbf16_pa_cols_;
        int& prev_brows = mmbf16_pb_rows_; int& prev_bcols = mmbf16_pb_cols_;

        for (int mi = 0; mi < Mt; mi++) {
            const int rm = std::min(T, M - mi * T);   // real A rows
            for (int ni = 0; ni < Nt; ni++) {
                const int nc = std::min(T, N - ni * T);  // real B/out cols
                const bool accumulate = (Kt > 1);
                if (accumulate)
                    for (int r = 0; r < rm; r++)
                        std::fill_n(&c_acc[static_cast<size_t>(r) * T], nc, 0.0f);
                for (int ki = 0; ki < Kt; ki++) {
                    const int kc = std::min(T, K - ki * T);  // real A cols / B rows
                    auto t0 = do_timing ? clk::now() : clk::time_point{};

                    // Pack A real region [rm,kc] -> stage_a (bf16), padding stays 0.
                    clear_stage_border(stage_a, prev_arows, prev_acols, rm, kc, T);
                    for (int r = 0; r < rm; r++) {
                        const float* arow = A + static_cast<size_t>(mi * T + r) * K + ki * T;
                        auto row = convert_f32_to_bf16(arow, static_cast<size_t>(kc));
                        std::memcpy(&stage_a[static_cast<size_t>(r) * T], row.data(), kc * sizeof(uint16_t));
                    }
                    prev_arows = rm; prev_acols = kc;
                    // Pack B real region [kc,nc] -> stage_b.
                    clear_stage_border(stage_b, prev_brows, prev_bcols, kc, nc, T);
                    for (int r = 0; r < kc; r++) {
                        const float* brow = B + static_cast<size_t>(ki * T + r) * N + ni * T;
                        auto row = convert_f32_to_bf16(brow, static_cast<size_t>(nc));
                        std::memcpy(&stage_b[static_cast<size_t>(r) * T], row.data(), nc * sizeof(uint16_t));
                    }
                    prev_brows = kc; prev_bcols = nc;
                    if (do_timing) { g_pack += ms_since(t0); t0 = clk::now(); }

                    buf_mgr_->copy_to(*buf_mmbf16_a_, stage_a, in_bytes);
                    buf_mgr_->copy_to(*buf_mmbf16_b_, stage_b, in_bytes);
                    if (do_timing) { g_dma_in += ms_since(t0); t0 = clk::now(); }

                    try {
                        kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                                   buf_mmbf16_a_->handle(), buf_mmbf16_b_->handle(),
                                   buf_mmbf16_c_->handle());
                        kernel.run.wait();
                    } catch (const std::exception& e) {
                        std::cerr << "Error: matmul_bf16 kernel execution failed: " << e.what() << "\n";
                        last_status_ = Status::ERROR;
                        return last_status_;
                    }
                    if (do_timing) { g_run += ms_since(t0); t0 = clk::now(); }

                    buf_mgr_->copy_from(*buf_mmbf16_c_, c_raw.data(), out_bytes);
                    if (accumulate) {
                        const float* c_f32 = reinterpret_cast<const float*>(c_raw.data());
                        for (int r = 0; r < rm; r++)
                            for (int c = 0; c < nc; c++)
                                c_acc[static_cast<size_t>(r) * T + c] += c_f32[static_cast<size_t>(r) * T + c];
                    }
                    if (do_timing) {
                        g_dma_out += ms_since(t0);
                        if (++g_launches % 256 == 0)
                            std::cerr << "[mmbf16] launches=" << g_launches
                                      << " pack=" << g_pack << "ms dma_in=" << g_dma_in
                                      << "ms run+wait=" << g_run << "ms dma_out=" << g_dma_out << "ms\n";
                    }
                }
                // Write the in-bounds portion of this output tile. Single K-tile:
                // read straight from the device buffer; else from the accumulator.
                const float* src = accumulate
                    ? c_acc.data()
                    : reinterpret_cast<const float*>(c_raw.data());
                for (int r = 0; r < rm; r++)
                    std::memcpy(C + static_cast<size_t>(mi * T + r) * N + ni * T,
                                &src[static_cast<size_t>(r) * T], nc * sizeof(float));
            }
        }

        last_status_ = Status::OK;
        return last_status_;
    }

    // Zero the padding border of a [T,T] bf16 stage so that the region outside the
    // current real [rows,cols] is 0, given the previous real extent. Only the rows
    // and columns that shrank need clearing (the real region is overwritten anyway).
    static void clear_stage_border(uint16_t* stage, int prev_rows, int prev_cols,
                                   int rows, int cols, int T) {
        // Columns [cols, prev_cols) of the rows we will write [0, rows).
        if (prev_cols > cols)
            for (int r = 0; r < rows; r++)
                std::fill_n(&stage[static_cast<size_t>(r) * T + cols], prev_cols - cols, uint16_t{0});
        // Whole rows [rows, prev_rows) (their full real-col extent from last time).
        for (int r = rows; r < prev_rows; r++)
            std::fill_n(&stage[static_cast<size_t>(r) * T], std::max(cols, prev_cols), uint16_t{0});
    }

    Status rope(const RopeParams& params) override {
        // RoPE on NPU: binary vector-add kernel called TWICE per head.
        // Host precomputes the element-wise products; NPU adds them:
        //   even call: in=[x_e*cos | -x_o*sin] → out_e = x_e*cos - x_o*sin
        //   odd  call: in=[x_e*sin |  x_o*cos] → out_o = x_e*sin + x_o*cos
        //
        // 2-buffer protocol: in_ptr=[t1|t2] (2*n_pairs BF16), out_ptr=n_pairs BF16.
        // Kernel computes out[i] = t1[i] + t2[i]. Pure binary add → standard
        // @pad_and_promote_binary_bf16 transform → single AIR herd → xclbin.
        if (!params.data || params.rope_dims <= 0 || params.n_dims <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        int n_dims  = static_cast<int>(params.rope_dims);
        int n_pairs = n_dims / 2;
        std::string cache_key = "rope_" + std::to_string(n_dims) + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rope_kernels_.find(n_dims);
        if (it == rope_kernels_.end()) {
            if (!ensure_rope_kernel(n_dims, n_dims, cache_key)) {
                std::cerr << "Error: no NPU RoPE kernel for dims=" << n_dims << " N=" << n_dims << "\n";
                std::cerr << "  Install Triton-XDNA for JIT: pip install triton-xdna\n";
                std::cerr << "  Or build prebuilt: ./scripts/build-kernels.sh npu6 rope\n";
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            it = rope_kernels_.find(n_dims);
        }

        auto& kernel = it->second;

        // Three distinct buffers: t1, t2 inputs + out, each pad_len BF16. Matches
        // the kernel's three pointer args (in1_ptr, in2_ptr, out_ptr) → three shim
        // DMA flows, like the proven matmul (A, B, C) pattern.
        //
        // The kernel is compiled for a PADDED length so flatten_tile_forall yields
        // a multi-core herd of standard 256-element tiles (see rope_aie2p.mlir).
        // This must match rope_pad in compile_kernels.py. Only the first n_pairs
        // elements are meaningful; the rest are zero-padded.
        int pad_len = std::max(512, ((n_pairs + 255) / 256) * 256 * 2);
        size_t buf_bytes = static_cast<size_t>(pad_len) * sizeof(uint16_t);

        if (!buf_rope_t1_       || buf_rope_t1_->size()       < buf_bytes)
            buf_rope_t1_       = buf_mgr_->alloc(buf_bytes, true);
        if (!buf_rope_t2_       || buf_rope_t2_->size()       < buf_bytes)
            buf_rope_t2_       = buf_mgr_->alloc(buf_bytes, true);
        if (!buf_rope_bf16_out_ || buf_rope_bf16_out_->size() < buf_bytes)
            buf_rope_bf16_out_ = buf_mgr_->alloc(buf_bytes, true);

        // Deinterleave x: x_e = data[2i], x_o = data[2i+1]
        std::vector<float> x_e(n_pairs), x_o(n_pairs);
        std::vector<float> cos_vals(n_pairs), sin_vals(n_pairs);
        for (int i = 0; i < n_pairs; i++) {
            x_e[i] = params.data[2 * i];
            x_o[i] = params.data[2 * i + 1];
        }

        if (params.cos_table && params.sin_table) {
            std::memcpy(cos_vals.data(), params.cos_table, n_pairs * sizeof(float));
            std::memcpy(sin_vals.data(), params.sin_table, n_pairs * sizeof(float));
        } else {
            for (int i = 0; i < n_pairs; i++) {
                float ratio = 1.0f / std::pow(params.freq_base,
                                  (2.0f * i) / static_cast<float>(n_dims));
                float angle = static_cast<float>(params.offset) * ratio * params.freq_scale;
                cos_vals[i] = std::cos(angle);
                sin_vals[i] = std::sin(angle);
            }
        }

        // Run kernel with separate t1, t2 buffers (zero-padded to pad_len); read
        // back only the first n_pairs meaningful outputs.
        auto run_half = [&](std::vector<float>& t1, std::vector<float>& t2, float* dst) -> bool {
            std::vector<float> t1_pad(pad_len, 0.0f), t2_pad(pad_len, 0.0f);
            std::memcpy(t1_pad.data(), t1.data(), n_pairs * sizeof(float));
            std::memcpy(t2_pad.data(), t2.data(), n_pairs * sizeof(float));
            auto bf16_t1 = convert_f32_to_bf16(t1_pad.data(), pad_len);
            auto bf16_t2 = convert_f32_to_bf16(t2_pad.data(), pad_len);
            buf_mgr_->copy_to(*buf_rope_t1_, bf16_t1.data(), buf_bytes);
            buf_mgr_->copy_to(*buf_rope_t2_, bf16_t2.data(), buf_bytes);
            try {
                kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                           buf_rope_t1_->handle(), buf_rope_t2_->handle(),
                           buf_rope_bf16_out_->handle());
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Error: RoPE kernel execution failed: " << e.what() << "\n";
                return false;
            }
            std::vector<uint8_t> bf16_out(buf_bytes);
            buf_mgr_->copy_from(*buf_rope_bf16_out_, bf16_out.data(), buf_bytes);
            auto f32_out = convert_bf16_to_f32(bf16_out.data(), pad_len);
            std::memcpy(dst, f32_out.data(), n_pairs * sizeof(float));
            return true;
        };

        // even: t1 = x_e*cos, t2 = -x_o*sin → out_e = t1+t2 = x_e*cos - x_o*sin
        std::vector<float> t1_e(n_pairs), t2_e(n_pairs);
        for (int i = 0; i < n_pairs; i++) {
            t1_e[i] =  x_e[i] * cos_vals[i];
            t2_e[i] = -x_o[i] * sin_vals[i];
        }
        std::vector<float> out_e(n_pairs), out_o(n_pairs);
        if (!run_half(t1_e, t2_e, out_e.data())) {
            last_status_ = Status::ERROR;
            return last_status_;
        }

        // odd: t1 = x_e*sin, t2 = x_o*cos → out_o = t1+t2 = x_e*sin + x_o*cos
        std::vector<float> t1_o(n_pairs), t2_o(n_pairs);
        for (int i = 0; i < n_pairs; i++) {
            t1_o[i] = x_e[i] * sin_vals[i];
            t2_o[i] = x_o[i] * cos_vals[i];
        }
        if (!run_half(t1_o, t2_o, out_o.data())) {
            last_status_ = Status::ERROR;
            return last_status_;
        }

        // Reinterleave into params.data
        for (int i = 0; i < n_pairs; i++) {
            params.data[2 * i]     = out_e[i];
            params.data[2 * i + 1] = out_o[i];
        }

        return Status::OK;
    }

    // Batched RoPE: pack multiple heads' even+odd halves into each kernel launch.
    // The kernel is a pure vector-add of pad_len elements; one launch handles
    // G = pad_len / (2*n_pairs) heads (8 for n_pairs=32 -> 512-elem kernel), so
    // 32 q-heads need 4 launches instead of 64 single-head/half launches.
    Status rope_batched(const RopeBatchedParams& params) override {
        if (!params.data || params.rope_dims <= 0 || params.n_dims <= 0 ||
            params.n_heads <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        int n_dims  = static_cast<int>(params.rope_dims);
        int n_pairs = n_dims / 2;
        std::string cache_key = "rope_" + std::to_string(n_dims) + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rope_kernels_.find(n_dims);
        if (it == rope_kernels_.end()) {
            if (!ensure_rope_kernel(n_dims, n_dims, cache_key)) {
                last_status_ = Status::NPU_UNAVAILABLE;
                return last_status_;
            }
            it = rope_kernels_.find(n_dims);
        }
        auto& kernel = it->second;

        int pad_len = std::max(512, ((n_pairs + 255) / 256) * 256 * 2);
        size_t buf_bytes = static_cast<size_t>(pad_len) * sizeof(uint16_t);
        if (!buf_rope_t1_       || buf_rope_t1_->size()       < buf_bytes)
            buf_rope_t1_       = buf_mgr_->alloc(buf_bytes, true);
        if (!buf_rope_t2_       || buf_rope_t2_->size()       < buf_bytes)
            buf_rope_t2_       = buf_mgr_->alloc(buf_bytes, true);
        if (!buf_rope_bf16_out_ || buf_rope_bf16_out_->size() < buf_bytes)
            buf_rope_bf16_out_ = buf_mgr_->alloc(buf_bytes, true);

        // cos/sin shared across heads (depend only on offset and pair index).
        std::vector<float> cos_vals(n_pairs), sin_vals(n_pairs);
        if (params.cos_table && params.sin_table) {
            std::memcpy(cos_vals.data(), params.cos_table, n_pairs * sizeof(float));
            std::memcpy(sin_vals.data(), params.sin_table, n_pairs * sizeof(float));
        } else {
            for (int i = 0; i < n_pairs; i++) {
                float ratio = 1.0f / std::pow(params.freq_base,
                                  (2.0f * i) / static_cast<float>(n_dims));
                float angle = static_cast<float>(params.offset) * ratio * params.freq_scale;
                cos_vals[i] = std::cos(angle);
                sin_vals[i] = std::sin(angle);
            }
        }

        const int heads_per_launch = pad_len / (2 * n_pairs);  // >= 1

        std::vector<float> t1_pad(pad_len), t2_pad(pad_len);
        std::vector<uint8_t> bf16_out(buf_bytes);

        for (int h0 = 0; h0 < params.n_heads; h0 += heads_per_launch) {
            int g_cur = std::min(heads_per_launch, params.n_heads - h0);
            std::fill(t1_pad.begin(), t1_pad.end(), 0.0f);
            std::fill(t2_pad.begin(), t2_pad.end(), 0.0f);

            // Pack: [even (g_cur*n_pairs)] then [odd (g_cur*n_pairs)].
            //   t1 = [x_e*cos | x_e*sin],  t2 = [-x_o*sin | x_o*cos]
            //   out = t1+t2 = [out_e | out_o]
            for (int g = 0; g < g_cur; g++) {
                const float* hd = params.data + static_cast<size_t>(h0 + g) * n_dims;
                int even = g * n_pairs;
                int odd  = g_cur * n_pairs + g * n_pairs;
                for (int i = 0; i < n_pairs; i++) {
                    float xe = hd[2 * i];
                    float xo = hd[2 * i + 1];
                    t1_pad[even + i] =  xe * cos_vals[i];
                    t2_pad[even + i] = -xo * sin_vals[i];
                    t1_pad[odd  + i] =  xe * sin_vals[i];
                    t2_pad[odd  + i] =  xo * cos_vals[i];
                }
            }

            auto bf16_t1 = convert_f32_to_bf16(t1_pad.data(), pad_len);
            auto bf16_t2 = convert_f32_to_bf16(t2_pad.data(), pad_len);
            buf_mgr_->copy_to(*buf_rope_t1_, bf16_t1.data(), buf_bytes);
            buf_mgr_->copy_to(*buf_rope_t2_, bf16_t2.data(), buf_bytes);
            try {
                kernel.run(kNpuOpcode, kernel.bo_instr, kernel.instr_words,
                           buf_rope_t1_->handle(), buf_rope_t2_->handle(),
                           buf_rope_bf16_out_->handle());
                kernel.run.wait();
            } catch (const std::exception& e) {
                std::cerr << "Error: batched RoPE kernel execution failed: " << e.what() << "\n";
                last_status_ = Status::ERROR;
                return last_status_;
            }
            buf_mgr_->copy_from(*buf_rope_bf16_out_, bf16_out.data(), buf_bytes);
            auto out = convert_bf16_to_f32(bf16_out.data(), pad_len);

            // Unpack + reinterleave back into params.data.
            for (int g = 0; g < g_cur; g++) {
                float* hd = params.data + static_cast<size_t>(h0 + g) * n_dims;
                int even = g * n_pairs;
                int odd  = g_cur * n_pairs + g * n_pairs;
                for (int i = 0; i < n_pairs; i++) {
                    hd[2 * i]     = out[even + i];
                    hd[2 * i + 1] = out[odd  + i];
                }
            }
        }

        last_status_ = Status::OK;
        return last_status_;
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
            // Legacy NPU F32 kernel without instruction sequence (N=256 bench only)
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

    // Fused flash_attn Triton kernel has no working mlir transform yet (scf.for +
    // online softmax). Host f32 decomposed attention matches cpu_ref; INT8 NPU matmul
    // QK/AV loses too much precision for coherent 16-layer inference.
    // Decomposed attention on the NPU: QK and AV as bf16 GEMMs (matmul_bf16),
    // softmax + causal mask on the host between them. Opt-in (GGNPU_NPU_ATTN);
    // falls back to flash_attn_decomposed on any failure. Per head:
    //   scores[cl] = Kh[cl,hd] @ Qh[hd,1]   (QK)
    //   softmax(scale*scores, causal mask)   (host)
    //   out[hd]    = w[1,cl] @ Vh[cl,hd]      (AV)
    Status flash_attn_npu(const AttnParams& params) {
        const int nh = params.n_head;
        const int hd = params.head_dim;
        const int64_t cl = params.ctx_len;
        const int64_t qpos = params.query_pos >= 0 ? params.query_pos : cl - 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        std::vector<float> scores(static_cast<size_t>(cl));
        for (int h = 0; h < nh; h++) {
            const float* Qh = params.Q + static_cast<int64_t>(h) * hd;
            const float* Kh = params.K + static_cast<int64_t>(h) * cl * hd;
            const float* Vh = params.V + static_cast<int64_t>(h) * cl * hd;
            float* outh = params.output + static_cast<int64_t>(h) * hd;

            // QK: scores[cl,1] = Kh[cl,hd] @ Qh[hd,1]
            if (matmul_bf16(Kh, Qh, scores.data(), static_cast<int>(cl), 1, hd) != Status::OK)
                return Status::ERROR;

            // scale + causal mask + softmax (host, f32)
            float sm_max = -INFINITY;
            for (int64_t j = 0; j < cl; j++) {
                if (j <= qpos) scores[static_cast<size_t>(j)] *= scale;
                else           scores[static_cast<size_t>(j)] = -INFINITY;
                sm_max = std::max(sm_max, scores[static_cast<size_t>(j)]);
            }
            float sm_sum = 0.0f;
            for (int64_t j = 0; j < cl; j++) {
                float e = std::exp(scores[static_cast<size_t>(j)] - sm_max);
                scores[static_cast<size_t>(j)] = e;
                sm_sum += e;
            }
            if (sm_sum > 0.0f)
                for (int64_t j = 0; j < cl; j++) scores[static_cast<size_t>(j)] /= sm_sum;

            // AV: out[1,hd] = weights[1,cl] @ Vh[cl,hd]
            if (matmul_bf16(scores.data(), Vh, outh, 1, hd, static_cast<int>(cl)) != Status::OK)
                return Status::ERROR;
        }
        return Status::OK;
    }

    Status flash_attn_decomposed(const AttnParams& params) {
        static bool noted = false;
        if (!noted) {
            std::cerr << "Note: flash_attn uses host f32 (fused NPU kernel not buildable; "
                      << "set GGNPU_EXPERIMENTAL=1 to attempt compile)\n";
            noted = true;
        }

        int nh = params.n_head;
        int hd = params.head_dim;
        int64_t cl = params.ctx_len;
        int64_t qpos = params.query_pos >= 0 ? params.query_pos : cl - 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        for (int h = 0; h < nh; h++) {
            const float* Qh = params.Q + h * hd;
            const float* Kh = params.K + static_cast<int64_t>(h) * cl * hd;
            const float* Vh = params.V + static_cast<int64_t>(h) * cl * hd;
            float* outh = params.output + h * hd;

            std::vector<float> scores(static_cast<size_t>(cl), -INFINITY);
            for (int64_t j = 0; j <= qpos && j < cl; j++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++)
                    dot += Qh[d] * Kh[j * hd + d];
                scores[static_cast<size_t>(j)] = dot * scale;
            }

            float sm_max = -INFINITY;
            for (int64_t j = 0; j < cl; j++)
                sm_max = std::max(sm_max, scores[static_cast<size_t>(j)]);
            float sm_sum = 0.0f;
            std::vector<float> weights(static_cast<size_t>(cl));
            for (int64_t j = 0; j < cl; j++) {
                weights[static_cast<size_t>(j)] = std::exp(scores[static_cast<size_t>(j)] - sm_max);
                sm_sum += weights[static_cast<size_t>(j)];
            }
            if (sm_sum > 0.0f) {
                for (int64_t j = 0; j < cl; j++)
                    weights[static_cast<size_t>(j)] /= sm_sum;
            }

            std::fill(outh, outh + hd, 0.0f);
            for (int64_t j = 0; j < cl; j++) {
                for (int d = 0; d < hd; d++)
                    outh[d] += weights[static_cast<size_t>(j)] * Vh[j * hd + d];
            }
        }
        return Status::OK;
    }

    Status flash_attn(const AttnParams& params) override {
        if (!params.Q || !params.K || !params.V || !params.output || params.n_head <= 0 || params.head_dim <= 0) {
            last_status_ = Status::INVALID_PARAM;
            return last_status_;
        }

        // Fused NPU flash_attn only when an explicit shaped xclbin is present (experimental).
        // Default build does not ship one — flash_attn_aie2p.mlir cannot lower the Triton kernel.
        const char* force_fused = std::getenv("GGNPU_FLASH_ATTN_FUSED");
        if (force_fused && force_fused[0] != '0') {
            int n_head = params.n_head;
            int head_dim = params.head_dim;
            int64_t ctx_len = params.ctx_len;
            std::string cache_key = "flash_attn_" + std::to_string(n_head) + "x" +
                                    std::to_string(head_dim) + "x" + std::to_string(ctx_len) + "_" +
                                    profile_str_;

            std::lock_guard<std::mutex> lock(mutex_);
            auto key = std::make_tuple(n_head, head_dim, ctx_len);
            auto it = flash_attn_kernels_.find(key);
            if (it == flash_attn_kernels_.end()) {
                if (!ensure_flash_attn_kernel(n_head, head_dim, ctx_len, cache_key)) {
                    std::cerr << "Error: GGNPU_FLASH_ATTN_FUSED set but no flash_attn xclbin for shape "
                              << n_head << "x" << head_dim << "x" << ctx_len << "\n";
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

        // Opt-in decomposed NPU attention (QK/AV bf16 GEMMs + host softmax).
        // Falls back to host f32 on any failure.
        const char* npu_attn = std::getenv("GGNPU_NPU_ATTN");
        if (npu_attn && npu_attn[0] != '0') {
            Status s = flash_attn_npu(params);
            if (s == Status::OK) { last_status_ = s; return s; }
            std::cerr << "Warning: NPU attention failed; falling back to host f32\n";
        }

        last_status_ = flash_attn_decomposed(params);
        return last_status_;
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
    void ensure_matmul_pipeline_slots(const xrt::kernel& krnl, int slot_count,
                                      size_t tile_bytes_in, size_t tile_bytes_out) {
        if (slot_count <= 0) return;
        const size_t need = static_cast<size_t>(slot_count);
        if (matmul_slots_.size() < need) {
            const size_t old = matmul_slots_.size();
            matmul_slots_.resize(need);
            for (size_t i = old; i < need; ++i) {
                matmul_slots_[i].run = xrt::run(krnl);
                matmul_slots_[i].run_initialized = true;
            }
        }
        for (int i = 0; i < slot_count; ++i) {
            auto& slot = matmul_slots_[static_cast<size_t>(i)];
            if (!slot.run_initialized) {
                slot.run = xrt::run(krnl);
                slot.run_initialized = true;
            }
            if (!slot.buf_a || slot.buf_a->size() < tile_bytes_in)
                slot.buf_a = buf_mgr_->alloc(tile_bytes_in, true);
            if (!slot.buf_b || slot.buf_b->size() < tile_bytes_in)
                slot.buf_b = buf_mgr_->alloc(tile_bytes_in, true);
            if (!slot.buf_c || slot.buf_c->size() < tile_bytes_out)
                slot.buf_c = buf_mgr_->alloc(tile_bytes_out, true);
        }
    }

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
            matmul_hw_ready_ = true;
            std::cout << "Loaded xclbin: " << xclbin_path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load xclbin: " << e.what() << "\n";
            return false;
        }
    }

    // Ensure a matmul kernel exists for the given dimensions
    bool ensure_matmul_kernel(int M, int N, int K, GgmlType B_type, const std::string& cache_key) {
        // Reuse the xclbin registered at backend init. Re-registering from cache can
        // produce an invalid kernel handle (segfault in xrt::kernel::group_id).
        if (matmul_hw_ready_) {
            return create_matmul_kernel_from_loaded_xclbin(M, N, K, B_type, cache_key);
        }

        // Fallback: load from per-shape cached xclbin (JIT path)
        if (cache_->has_xclbin(cache_key)) {
            std::string cached_path = cache_->get_xclbin_path(cache_key);
            return load_matmul_kernel_for_shape(cached_path, M, N, K, B_type, cache_key);
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
            CachedMatmulKernel cached{std::move(krnl), {}, 0, M, N, K, B_type};
            std::string seq_path = detail::find_prebuilt_sequence(fs::path(path).filename().string(), cache_dir_);
            if (!seq_path.empty()) {
                load_instr_bo(*device_, cached.krnl, cached.bo_instr, cached.instr_words, seq_path);
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
            CachedMatmulKernel cached{std::move(krnl), {}, 0, M, N, K, B_type};
            std::string seq_path = detail::find_prebuilt_sequence("matmul_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                load_instr_bo(*device_, cached.krnl, cached.bo_instr, cached.instr_words, seq_path);
            }
            matmul_kernels_[cache_key] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to create matmul kernel: " << e.what() << "\n";
            return false;
        }
    }

    // Lazily load the fixed 256^3 bf16 matmul xclbin (matmul_bf16_<profile>.xclbin).
    bool ensure_matmul_bf16_kernel() {
        if (matmul_bf16_kernel_.loaded) return true;
        std::string name = "matmul_bf16_" + profile_str_ + ".xclbin";
        std::string path = detail::find_prebuilt_xclbin(name, cache_dir_);
        if (path.empty() && cache_->has_xclbin(name)) path = cache_->get_xclbin_path(name);
        if (path.empty()) return false;
        try {
            auto data = detail::load_xclbin_file(path);
            if (data.empty()) return false;
            xrt::hw_context ctx = register_xclbin_from_data(*device_, data);
            matmul_shape_ctxs_.push_back(ctx);  // keep context alive
            xrt::kernel krnl(ctx, kTritonXdnaKernelName);
            xrt::run run(krnl);
            CachedMatmulBf16Kernel cached{run, krnl, {}, 0, true};
            std::string seq = detail::find_prebuilt_sequence(name, cache_dir_);
            if (!seq.empty()) {
                load_instr_bo(*device_, krnl, cached.bo_instr, cached.instr_words, seq);
            }
            if (!cached.bo_instr || cached.instr_words == 0) {
                std::cerr << "Error: matmul_bf16 xclbin missing instruction sequence\n";
                return false;
            }
            matmul_bf16_kernel_ = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load matmul_bf16 kernel: " << e.what() << "\n";
            return false;
        }
    }

    // Load or compile the rmsnorm xclbin (prefer Llama-shaped rmsnorm_2048 when present).
    bool load_rmsnorm_xclbin() {
        std::string shaped_name = "rmsnorm_2048_" + profile_str_ + ".xclbin";
        std::string xclbin_name = shaped_name;
        std::string xclbin_path = detail::find_prebuilt_xclbin(shaped_name, cache_dir_);

        if (xclbin_path.empty()) {
            xclbin_name = "rmsnorm_" + profile_str_ + ".xclbin";
            xclbin_path = detail::find_prebuilt_xclbin(xclbin_name, cache_dir_);
            if (xclbin_path.empty() && cache_->has_xclbin(xclbin_name)) {
                xclbin_path = cache_->get_xclbin_path(xclbin_name);
            }
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
                    xclbin_name = cache_key + ".xclbin";
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
            rmsnorm_xclbin_name_ = fs::path(xclbin_path).filename().string();
            std::cout << "Loaded rmsnorm xclbin: " << xclbin_path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load rmsnorm xclbin: " << e.what() << "\n";
            return false;
        }
    }

    bool ensure_rmsnorm_kernel(int N, const std::string& cache_key) {
        // Reuse preloaded hw_ctx when the loaded xclbin matches (same pattern as SiLU).
        if (hw_ctx_rmsnorm_) {
            if (N == kRmsnormKernelHidden &&
                rmsnorm_xclbin_name_.find("rmsnorm_2048") != std::string::npos) {
                return create_rmsnorm_kernel_from_loaded_xclbin(N, cache_key);
            }
            if (N == kRmsnormKernelCols &&
                rmsnorm_xclbin_name_.find("rmsnorm_2048") == std::string::npos) {
                return create_rmsnorm_kernel_from_loaded_xclbin(N, cache_key);
            }
        }

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
            std::string seq_xclbin = rmsnorm_xclbin_name_.empty()
                ? "rmsnorm_" + profile_str_ + ".xclbin"
                : rmsnorm_xclbin_name_;
            std::string seq_path = detail::find_prebuilt_sequence(seq_xclbin, cache_dir_);
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

    // Load or compile the RoPE kernel
    bool ensure_rope_kernel(int n_dims, int N, const std::string& cache_key) {
        if (cache_->has_xclbin(cache_key)) {
            return load_rope_kernel_for_shape(cache_->get_xclbin_path(cache_key), n_dims, N, cache_key);
        }

        // Reuse preloaded xclbin for the default shape (n_pairs=kRopePairs → n_dims=64)
        if (n_dims == kRopePairs * 2 && hw_ctx_rope_) {
            return create_rope_kernel_from_loaded_xclbin(n_dims, cache_key);
        }

        // Try JIT compilation for non-standard shapes
        if (detail::jit_compilation_available()) {
            std::vector<uint8_t> xclbin_data = detail::jit_compile_rope(N, npu_profile_);
            if (!xclbin_data.empty()) {
                cache_->store_xclbin(cache_key, xclbin_data);
                return load_rope_kernel_for_shape(cache_->get_xclbin_path(cache_key), n_dims, N, cache_key);
            }
        }

        return false;
    }

    bool load_rope_kernel_for_shape(const std::string& path, int n_dims, int N, const std::string& cache_key) {
        try {
            auto data = detail::load_xclbin_file(path);
            if (data.empty()) return false;

            xrt::hw_context ctx = register_xclbin_from_data(*device_, data);
            matmul_shape_ctxs_.push_back(ctx);

            xrt::kernel krnl(ctx, kTritonXdnaKernelName);
            xrt::run run(krnl);

            CachedRopeKernel cached{run, krnl, {}, 0, n_dims, N};
            std::string seq_path = detail::find_prebuilt_sequence("rope_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                load_instr_bo(*device_, krnl, cached.bo_instr, cached.instr_words, seq_path);
            }
            rope_kernels_[n_dims] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load rope kernel: " << e.what() << "\n";
            return false;
        }
    }

    bool create_rope_kernel_from_loaded_xclbin(int n_dims, const std::string& cache_key) {
        try {
            xrt::kernel krnl(hw_ctx_rope_, kTritonXdnaKernelName);
            xrt::run run(krnl);

            CachedRopeKernel cached{run, krnl, {}, 0, n_dims, n_dims};
            std::string seq_path = detail::find_prebuilt_sequence("rope_" + profile_str_ + ".xclbin", cache_dir_);
            if (!seq_path.empty()) {
                load_instr_bo(*device_, krnl, cached.bo_instr, cached.instr_words, seq_path);
            }
            rope_kernels_[n_dims] = std::move(cached);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to create rope kernel from loaded xclbin: " << e.what() << "\n";
            return false;
        }
    }

    bool load_rope_xclbin() {
        std::string xclbin_name = "rope_" + profile_str_ + ".xclbin";
        std::string xclbin_path = detail::find_prebuilt_xclbin(xclbin_name, cache_dir_);

        if (xclbin_path.empty() && cache_->has_xclbin(xclbin_name)) {
            xclbin_path = cache_->get_xclbin_path(xclbin_name);
        }

        if (xclbin_path.empty()) return false;

        try {
            xclbin_data_rope_ = detail::load_xclbin_file(xclbin_path);
            if (xclbin_data_rope_.empty()) return false;

            hw_ctx_rope_ = register_xclbin_from_data(*device_, xclbin_data_rope_);
            std::cout << "Loaded rope xclbin: " << xclbin_path << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to load rope xclbin: " << e.what() << "\n";
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
    bool matmul_hw_ready_ = false;
    std::vector<xrt::hw_context> matmul_shape_ctxs_;  // keeps per-shape contexts alive

    // BF16 matmul (decomposed NPU attention building block)
    CachedMatmulBf16Kernel matmul_bf16_kernel_;
    std::shared_ptr<XrtBuffer> buf_mmbf16_a_;
    std::shared_ptr<XrtBuffer> buf_mmbf16_b_;
    std::shared_ptr<XrtBuffer> buf_mmbf16_c_;
    // Persistent host staging for matmul_bf16 packing (padding kept zero across
    // calls; prev_* track the last real extent written into each stage).
    std::vector<uint16_t> mmbf16_stage_a_;
    std::vector<uint16_t> mmbf16_stage_b_;
    std::vector<float> mmbf16_cacc_;
    std::vector<uint8_t> mmbf16_craw_;
    int mmbf16_pa_rows_ = 0, mmbf16_pa_cols_ = 0;
    int mmbf16_pb_rows_ = 0, mmbf16_pb_cols_ = 0;

    std::vector<uint8_t> xclbin_data_rmsnorm_;
    xrt::hw_context hw_ctx_rmsnorm_;
    std::string rmsnorm_xclbin_name_;

    std::vector<uint8_t> xclbin_data_softmax_;
    xrt::hw_context hw_ctx_softmax_;

    std::vector<uint8_t> xclbin_data_silu_;
    xrt::hw_context hw_ctx_silu_;

    std::vector<uint8_t> xclbin_data_rope_;
    xrt::hw_context hw_ctx_rope_;
    std::string rope_xclbin_name_;

    std::vector<uint8_t> xclbin_data_fa_;
    xrt::hw_context hw_ctx_fa_;

    std::vector<MatmulPipelineSlot> matmul_slots_;

    std::shared_ptr<XrtBuffer> buf_rmsnorm_in_;
    std::shared_ptr<XrtBuffer> buf_rmsnorm_out_;
    std::shared_ptr<XrtBuffer> buf_rmsnorm_bf16_in_;
    std::shared_ptr<XrtBuffer> buf_rmsnorm_bf16_out_;

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

    std::shared_ptr<XrtBuffer> buf_rope_t1_;        // t1 input  (n_pairs BF16)
    std::shared_ptr<XrtBuffer> buf_rope_t2_;        // t2 input  (n_pairs BF16)
    std::shared_ptr<XrtBuffer> buf_rope_bf16_out_;  // out       (n_pairs BF16)

    std::unordered_map<std::string, CachedMatmulKernel> matmul_kernels_;
    std::unordered_map<int, CachedRmsNormKernel> rmsnorm_kernels_;
    std::unordered_map<std::pair<int, int>, CachedSoftmaxKernel, PairHash> softmax_kernels_;
    std::unordered_map<int, CachedSiluKernel> silu_kernels_;
    std::unordered_map<std::tuple<int, int, int64_t>, CachedFlashAttnKernel, TupleHash> flash_attn_kernels_;
    std::unordered_map<int, CachedRopeKernel> rope_kernels_;

    int npu_profile_ = 6;
    std::string profile_str_ = "npu6";
    std::string cache_dir_;

    Status last_status_;
    mutable std::mutex mutex_;
    std::unordered_set<int> rmsnorm_validated_;  // hidden sizes validated vs CPU ref
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
