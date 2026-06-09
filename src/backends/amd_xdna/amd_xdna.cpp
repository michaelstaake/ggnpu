#include "backend.h"
#include "tensor.h"
#include "cache.h"
#include <xrt.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <xrt/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <mutex>

namespace ggnpu {

namespace fs = std::filesystem;

// Forward declarations from kernels.cpp
namespace detail {
    std::vector<uint8_t> load_xclbin_file(const std::string& path);
    std::string find_prebuilt_xclbin(const std::string& xclbin_name, const std::string& cache_dir);
    std::vector<uint8_t> jit_compile_matmul(int M, int N, int K, int profile);
    std::string make_cache_key(const std::string& op, int M, int N, int K, const std::string& profile);
}

// Internal buffer wrapper
class XrtBuffer {
public:
    XrtBuffer() = default;
    XrtBuffer(xrt::device& dev, size_t size, xrt::bo::flags flags)
        : bo_(dev, size, dev.get_device_membase(), flags) {}

    void* map() {
        if (!is_valid()) return nullptr;
        return bo_.map<void*>();
    }

    void sync(xrt::bo::direction dir) {
        if (!is_valid()) return;
        bo_.sync(dir);
    }

    xrt::bo& handle() { return bo_; }
    const xrt::bo& handle() const { return bo_; }
    bool is_valid() const { return bo_.is_valid(); }
    size_t size() const { return bo_.size(); }

private:
    xrt::bo bo_;
};

// Buffer manager
class BufferMgr {
public:
    explicit BufferMgr(xrt::device& dev) : device_(dev) {}

    std::shared_ptr<XrtBuffer> alloc(size_t size, bool host_backed = false) {
        xrt::bo::flags flags = host_backed ? xrt::bo::flags::host : xrt::bo::flags::flags_type::default_flags;
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
        buf.sync(xrt::bo::direction::HTOD);
    }

    void copy_from(XrtBuffer& buf, void* dst, size_t count) {
        if (!buf.is_valid()) return;
        buf.sync(xrt::bo::direction::DHTO);
        void* mapped = buf.map();
        if (mapped) {
            std::memcpy(dst, mapped, count);
        }
    }

private:
    xrt::device& device_;
    std::vector<std::shared_ptr<XrtBuffer>> buffers_;
};

// Cached kernel for a specific (op, M, N, K) shape
struct CachedKernel {
    xrt::ip_handle ip_handle;
    xrt::run run;
    int M, N, K;
    GgmlType B_type;
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

        // Load or compile matmul xclbin
        load_matmul_xclbin();
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

        // Find or create cached kernel for this shape
        auto key = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
        std::string cache_key = "matmul_" + key + "_" + profile_str_;

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = matmul_kernels_.find(cache_key);
        if (it == matmul_kernels_.end()) {
            // Need to compile/load this kernel shape
            if (!ensure_matmul_kernel(M, N, K, params.B_type, cache_key)) {
                last_status_ = Status::ERROR;
                return last_status_;
            }
            it = matmul_kernels_.find(cache_key);
        }

        auto& kernel = it->second;

        // Allocate or reuse buffers
        size_t size_a = M * K * sizeof(float);
        size_t size_b = K * N * ggml_type_size(params.B_type);
        size_t size_c = M * N * sizeof(float);

        if (!buf_a_ || buf_a_->size() < size_a) {
            buf_a_ = buf_mgr_->alloc(size_a, true);
        }
        if (!buf_b_ || buf_b_->size() < size_b) {
            buf_b_ = buf_mgr_->alloc(size_b, true);
        }
        if (!buf_c_ || buf_c_->size() < size_c) {
            buf_c_ = buf_mgr_->alloc(size_c, true);
        }

        // Copy inputs to device
        buf_mgr_->copy_to(*buf_a_, params.A, size_a);
        buf_mgr_->copy_to(*buf_b_, params.B, size_b);

        // Set kernel arguments
        // Kernel expects: A (buf), B (buf), C (buf), M, N, K
        try {
            kernel.run(buf_a_->handle(), buf_b_->handle(), buf_c_->handle(),
                       static_cast<uint32_t>(M), static_cast<uint32_t>(N), static_cast<uint32_t>(K));
        } catch (const std::exception& e) {
            std::cerr << "Error: kernel execution failed: " << e.what() << "\n";
            last_status_ = Status::ERROR;
            return last_status_;
        }

        return Status::OK;
    }

    Status rms_norm(const RmsNormParams& params) override {
        // Stub: compute on host for now, NPU kernel to be added in Phase 4
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
        // Stub: compute on host for now, NPU kernel to be added in Phase 4
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
        // Stub: compute on host for now, NPU kernel to be added in Phase 4
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
        // Stub: compute on host for now, NPU kernel to be added in Phase 4
        for (int i = 0; i < params.size; i++) {
            float x = params.input[i];
            params.output[i] = x / (1.0f + std::exp(-x));
        }
        return Status::OK;
    }

    Status flash_attn(const AttnParams& params) override {
        // Stub: not yet implemented for NPU
        last_status_ = Status::NOT_FOUND;
        return last_status_;
    }

    void sync() override {
        std::lock_guard<std::mutex> lock(mutex_);
        // xrt::run::wait() is called implicitly by the run() call
        // Additional sync can be added here if needed
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
            // Try JIT compilation
            std::vector<uint8_t> xclbin_data = detail::jit_compile_matmul(256, 256, 256, npu_profile_);
            if (!xclbin_data.empty()) {
                std::string cache_key = detail::make_cache_key("matmul", 256, 256, 256, profile_str_);
                cache_->store_xclbin(cache_key, xclbin_data);
                xclbin_path = cache_->get_xclbin_path(cache_key);
            }
        }

        if (xclbin_path.empty()) {
            std::cerr << "Warning: no matmul xclbin found for profile " << profile_str_ << "\n";
            std::cerr << "  Place prebuilt xclbin at: " << xclbin_name << "\n";
            std::cerr << "  Or build with mlir-aie to generate one.\n";
            return false;
        }

        try {
            xclbin_data_ = detail::load_xclbin_file(xclbin_path);
            if (xclbin_data_.empty()) {
                std::cerr << "Error: failed to read xclbin: " << xclbin_path << "\n";
                return false;
            }

            xclbin_uuid_ = device_->load_xclbin(xclbin_data_);
            std::cout << "Loaded xclbin: " << xclbin_path << " UUID=" << xclbin_uuid_ << "\n";
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
            return load_kernel_for_shape(cached_path, M, N, K, B_type, cache_key);
        }

        // Try to find a generic xclbin that works for multiple shapes
        // The matmul kernel should be dimension-agnostic (programmed at runtime)
        if (matmul_kernels_.empty() && !xclbin_uuid_.is_none()) {
            return create_kernel_from_loaded_xclbin(M, N, K, B_type, cache_key);
        }

        std::cerr << "Error: no xclbin available for matmul " << M << "x" << N << "x" << K << "\n";
        return false;
    }

    bool load_kernel_for_shape(const std::string& path, int M, int N, int K, GgmlType B_type, const std::string& cache_key) {
        try {
            auto data = detail::load_xclbin_file(path);
            if (data.empty()) return false;

            xrt::device tmp_device(*device_);
            xrt::uuid tmp_uuid = tmp_device.load_xclbin(data);

            xrt::ip_handle ip(tmp_device, tmp_uuid);
            xrt::run run(ip, "matmul");

            matmul_kernels_[cache_key] = {ip, run, M, N, K, B_type};
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load kernel from cache: " << e.what() << "\n";
            return false;
        }
    }

    bool create_kernel_from_loaded_xclbin(int M, int N, int K, GgmlType B_type, const std::string& cache_key) {
        try {
            xrt::ip_handle ip(*device_, xclbin_uuid_);
            xrt::run run(ip, "matmul");

            matmul_kernels_[cache_key] = {ip, run, M, N, K, B_type};
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to create kernel: " << e.what() << "\n";
            return false;
        }
    }

    std::shared_ptr<xrt::device> device_;
    std::shared_ptr<BufferMgr> buf_mgr_;
    std::unique_ptr<CompileCache> cache_;

    std::vector<uint8_t> xclbin_data_;
    xrt::uuid xclbin_uuid_;

    std::shared_ptr<XrtBuffer> buf_a_;
    std::shared_ptr<XrtBuffer> buf_b_;
    std::shared_ptr<XrtBuffer> buf_c_;

    std::unordered_map<std::string, CachedKernel> matmul_kernels_;

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
