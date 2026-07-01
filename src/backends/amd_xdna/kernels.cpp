#include "backend.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <array>

namespace ggnpu {

namespace fs = std::filesystem;

namespace detail {

//====//
// Utility: check if a command exists
//====//
bool command_exists(const std::string& cmd) {
    return std::system(("which " + cmd + " > /dev/null 2>&1").c_str()) == 0;
}

//====//
// Utility: read environment variable
//====//
std::string get_env(const std::string& key, const std::string& default_val = "") {
    const char* val = std::getenv(key.c_str());
    return val ? std::string(val) : default_val;
}

//====//
// Utility: execute command and capture output
//====//
std::string exec_command(const std::string& cmd) {
    std::array<char, 4096> buffer;
    std::string output;
    std::string full_cmd = cmd + " 2>&1";

    FILE* pipe = popen(full_cmd.c_str(), "r");
    if (!pipe) return "";

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        output += buffer.data();
    }

    pclose(pipe);
    return output;
}

//====//
// Build cache key from op and dimensions
//====//
std::string make_cache_key(const std::string& op, int M, int N, int K, const std::string& profile) {
    return op + "_" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + "_" + profile;
}

//====//
// Load xclbin from file
//====//
std::vector<uint8_t> load_xclbin_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
        return buffer;
    }
    return {};
}

//====//
// Try to find prebuilt xclbin in common locations
//====//
std::string find_prebuilt_sequence(const std::string& xclbin_name, const std::string& cache_dir) {
    std::string base = xclbin_name;
    if (base.size() > 7 && base.substr(base.size() - 7) == ".xclbin") {
        base = base.substr(0, base.size() - 7);
    }
    std::vector<std::string> candidates = {
        base + "_sequence.bin",
        base + ".sequence.bin",
    };
    for (const auto& name : candidates) {
        fs::path p = fs::path(cache_dir) / "xclbin" / name;
        if (fs::exists(p)) return p.string();
    }
    return "";
}

std::vector<uint32_t> load_sequence_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    std::streamsize size = file.tellg();
    if (size <= 0 || (size % 4) != 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> words(static_cast<size_t>(size / 4));
    file.read(reinterpret_cast<char*>(words.data()), size);
    return words;
}

std::string find_prebuilt_xclbin(const std::string& xclbin_name, const std::string& cache_dir) {
    // Check cache/xclbin directory first
    fs::path cache_path = fs::path(cache_dir) / "xclbin" / xclbin_name;
    if (fs::exists(cache_path)) {
        return cache_path.string();
    }

    // Check standard installation paths
    std::vector<std::string> search_paths = {
        "/opt/ggnpu/xclbin/",
        "/usr/share/ggnpu/xclbin/",
        "./xclbin/",
    };

    for (const auto& path : search_paths) {
        fs::path p = fs::path(path) / xclbin_name;
        if (fs::exists(p)) {
            return p.string();
        }
    }

    return "";
}

//====//
// Find the Triton-XDNA compile script
//====//
std::string find_triton_compile_script() {
    // Check common locations
    std::vector<std::string> candidates = {
        "kernels/triton/compile_kernels.py",
        "../kernels/triton/compile_kernels.py",
        get_env("GGNPU_REPO_ROOT", "") + "/kernels/triton/compile_kernels.py",
    };

    for (const auto& candidate : candidates) {
        if (candidate.empty()) continue;
        if (fs::exists(candidate)) {
            return candidate;
        }
    }

    return "";
}

//====//
// Find python3
//====//
std::string find_python() {
    if (command_exists("python3")) return "python3";
    if (command_exists("python")) return "python";
    return "";
}

//====//
// Check if Triton-XDNA is available
//====//
bool triton_xdna_available() {
    std::string python = find_python();
    if (python.empty()) return false;

    std::string cmd = python + " -c \"from triton.backends.amd_triton_npu.driver import NPUDriver\" >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

//====//
// Get Triton-XDNA availability status message
//====//
std::string get_jit_status_message() {
    if (triton_xdna_available()) {
        return "JIT compilation available via Triton-XDNA";
    }
    return "Triton-XDNA not found. Install with: pip install triton-xdna";
}

//====//
// Check if JIT compilation is available
//====//
bool jit_compilation_available() {
    return triton_xdna_available();
}

//====//
// JIT compile any kernel using Triton-XDNA
// Compiles Triton Python kernels to .xclbin via Triton-XDNA compiler
//====//
std::vector<uint8_t> jit_compile_kernel(const std::string& op_name,
                                         const std::string& profile_str,
                                         int npu_profile) {
    std::string python = find_python();
    if (python.empty()) {
        std::cerr << "Error: python3 not found\n";
        return {};
    }

    if (!triton_xdna_available()) {
        std::cerr << "Info: Triton-XDNA not available. Prebuilt xclbins needed.\n";
        std::cerr << "  Install: pip install triton-xdna\n";
        return {};
    }

    std::string script = find_triton_compile_script();
    if (script.empty()) {
        std::cerr << "Warning: could not find Triton compile script\n";
        std::cerr << "  Place kernels/triton/compile_kernels.py in the repo\n";
        return {};
    }

    // Build cache directory
    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path xclbin_dir = fs::path(cache_dir) / "xclbin";
    fs::create_directories(xclbin_dir);

    fs::path xclbin_file = xclbin_dir / (op_name + "_" + profile_str + ".xclbin");

    // If xclbin already exists, use it
    if (fs::exists(xclbin_file)) {
        std::vector<uint8_t> xclbin_data = load_xclbin_file(xclbin_file.string());
        if (!xclbin_data.empty()) {
            return xclbin_data;
        }
    }

    // Build compilation command
    std::string cmd = python + " \"" + script + "\"";
    cmd += " --op " + op_name;
    cmd += " --profile " + profile_str;
    cmd += " --output-dir " + xclbin_dir.string();

    // Add kernel-specific parameters
    if (op_name == "matmul") {
        // Parameters will be passed via environment or a config file
        // For now, use default dimensions
        cmd += " --M 256 --N 256 --K 256";
    } else if (op_name == "rmsnorm") {
        cmd += " --M 32 --N 256";
    } else if (op_name == "silu" || op_name == "rope") {
        cmd += " --N 2048";
    } else if (op_name == "softmax") {
        cmd += " --rows 1 --cols 1024";
    } else if (op_name == "flash_attn" || op_name == "rope") {
        // Experimental kernels are skipped by compile_kernels.py unless GGNPU_EXPERIMENTAL=1.
        std::cerr << "Info: " << op_name << " is experimental; no JIT (set GGNPU_EXPERIMENTAL=1 to try)\n";
        return {};
    }

    // Execute compilation
    std::cerr << "JIT: compiling " << op_name << " for " << profile_str << " (Triton-XDNA)\n";
    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Error: Triton-XDNA compilation failed\n";
        return {};
    }

    // Read compiled xclbin
    if (!fs::exists(xclbin_file)) {
        std::cerr << "Error: xclbin not produced by compiler\n";
        return {};
    }

    std::vector<uint8_t> xclbin_data = load_xclbin_file(xclbin_file.string());
    if (xclbin_data.empty()) {
        std::cerr << "Error: xclbin is empty\n";
        return {};
    }

    std::cerr << "JIT: compiled " << xclbin_data.size() << " bytes\n";

    return xclbin_data;
}

//====//
// JIT compile matmul kernel
//====//
std::vector<uint8_t> jit_compile_matmul(int M, int N, int K, int npu_profile) {
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    return jit_compile_kernel("matmul", profile_str, npu_profile);
}

//====//
// JIT compile rmsnorm kernel
//====//
std::vector<uint8_t> jit_compile_rmsnorm(int N, int npu_profile) {
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    std::string python = find_python();
    if (python.empty()) return {};

    std::string script = find_triton_compile_script();
    if (script.empty()) return {};

    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path xclbin_dir = fs::path(cache_dir) / "xclbin";
    fs::create_directories(xclbin_dir);

    std::string cache_key = "rmsnorm_" + std::to_string(N) + "_" + profile_str;
    fs::path xclbin_file = xclbin_dir / (cache_key + ".xclbin");
    if (fs::exists(xclbin_file)) {
        auto data = load_xclbin_file(xclbin_file.string());
        if (!data.empty()) return data;
    }

    int M = (N == 256) ? 32 : 2;
    std::string cmd = python + " \"" + script + "\"";
    cmd += " --op rmsnorm --profile " + profile_str;
    cmd += " --output-dir " + xclbin_dir.string();
    cmd += " --M " + std::to_string(M) + " --N " + std::to_string(N);

    std::cerr << "JIT: compiling rmsnorm M=" << M << " N=" << N << " for " << profile_str
              << " (Triton-XDNA)\n";
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "Error: Triton-XDNA rmsnorm compilation failed\n";
        return {};
    }

    fs::path default_xclbin = xclbin_dir / ("rmsnorm_" + profile_str + ".xclbin");
    if (!fs::exists(xclbin_file) && fs::exists(default_xclbin)) {
        fs::copy_file(default_xclbin, xclbin_file, fs::copy_options::overwrite_existing);
        fs::path default_seq = xclbin_dir / ("rmsnorm_" + profile_str + "_sequence.bin");
        fs::path shaped_seq = xclbin_dir / (cache_key + "_sequence.bin");
        if (fs::exists(default_seq) && !fs::exists(shaped_seq)) {
            fs::copy_file(default_seq, shaped_seq, fs::copy_options::overwrite_existing);
        }
    }

    if (!fs::exists(xclbin_file)) {
        std::cerr << "Error: rmsnorm xclbin not produced for N=" << N << "\n";
        return {};
    }

    return load_xclbin_file(xclbin_file.string());
}

//====//
// JIT compile rope kernel
//====//
std::vector<uint8_t> jit_compile_rope(int n_dims, int npu_profile) {
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    return jit_compile_kernel("rope", profile_str, npu_profile);
}

//====//
// JIT compile softmax kernel
//====//
std::vector<uint8_t> jit_compile_softmax(int cols, int rows, int npu_profile) {
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    return jit_compile_kernel("softmax", profile_str, npu_profile);
}

//====//
// JIT compile silu kernel
//====//
std::vector<uint8_t> jit_compile_silu(int size, int npu_profile) {
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    return jit_compile_kernel("silu", profile_str, npu_profile);
}

//====//
// JIT compile gelu kernel
//====//
std::vector<uint8_t> jit_compile_gelu(int size, int npu_profile) {
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    return jit_compile_kernel("gelu", profile_str, npu_profile);
}

//====//
// JIT compile flash_attn kernel
//====//
std::vector<uint8_t> jit_compile_flash_attn(int n_head, int head_dim, int64_t ctx_len, int npu_profile) {
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    return jit_compile_kernel("flash_attn", profile_str, npu_profile);
}

} // namespace detail

} // namespace ggnpu
