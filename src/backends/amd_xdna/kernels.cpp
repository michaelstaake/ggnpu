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
#include <regex>

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
// Read a file into a string
//====//
std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";

    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

//====//
// Write a string to a file
//====//
bool write_file(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << content;
    return true;
}

//====//
// Replace all occurrences of a substring
//====//
std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    if (from.empty()) return str;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

//====//
// Find the repository root directory (parent of kernels/amd/)
//====//
std::string find_repo_root() {
    // Check common locations relative to the executable
    std::vector<std::string> candidates = {
        ".",
        "..",
        "../..",
        get_env("GGNPU_REPO_ROOT", ""),
    };

    for (const auto& candidate : candidates) {
        if (candidate.empty()) continue;
        fs::path kernels_path = fs::path(candidate) / "kernels" / "amd";
        if (fs::exists(kernels_path)) {
            return fs::canonical(candidate).string();
        }
    }

    // Fallback: try to find from /proc/self/exe
    char exe_path[4096];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len != -1) {
        exe_path[len] = '\0';
        fs::path exe_dir = fs::path(exe_path).parent_path();
        for (int depth = 0; depth < 5; depth++) {
            fs::path kernels_path = exe_dir / "kernels" / "amd";
            if (fs::exists(kernels_path)) {
                return fs::canonical(exe_dir).string();
            }
            exe_dir = exe_dir.parent_path();
        }
    }

    return "";
}

//====//
// Copy MLIR template and substitute dimensions
// For matmul: replaces __M__, __N__, __K__ placeholders
// For other ops: replaces __SIZE__ placeholder
//====//
std::string load_mlir_template(const std::string& op_name, const std::string& repo_root) {
    std::string mlir_path;

    if (op_name == "matmul") {
        mlir_path = fs::path(repo_root) / "kernels" / "amd" / "matmul_i8" / "matmul.mlir";
    } else if (op_name == "rmsnorm") {
        mlir_path = fs::path(repo_root) / "kernels" / "amd" / "rmsnorm" / "rmsnorm.mlir";
    } else if (op_name == "rope") {
        mlir_path = fs::path(repo_root) / "kernels" / "amd" / "rope" / "rope.mlir";
    } else if (op_name == "softmax") {
        mlir_path = fs::path(repo_root) / "kernels" / "amd" / "softmax" / "softmax.mlir";
    } else if (op_name == "silu") {
        mlir_path = fs::path(repo_root) / "kernels" / "amd" / "silu" / "silu.mlir";
    } else if (op_name == "flash_attn") {
        mlir_path = fs::path(repo_root) / "kernels" / "amd" / "fused_attn" / "flash_attn.mlir";
    } else {
        return "";
    }

    std::string content = read_file(mlir_path);
    if (content.empty()) {
        std::cerr << "Warning: could not read MLIR template: " << mlir_path << "\n";
        return "";
    }

    return content;
}

//====//
// Find aiecc.py path
//====//
std::string find_aiecc() {
    std::string aie_home = get_env("AIE_HOME");
    if (!aie_home.empty()) {
        fs::path aiecc_path = fs::path(aie_home) / "bin" / "aiecc.py";
        if (fs::exists(aiecc_path)) return aiecc_path.string();
    }

    if (command_exists("aiecc.py")) return "aiecc.py";

    return "";
}

//====//
// Check if JIT compilation is available
//====//
bool jit_compilation_available() {
    return !find_aiecc().empty();
}

//====//
// Get JIT compilation status message
//====//
std::string get_jit_status_message() {
    if (jit_compilation_available()) {
        return "JIT compilation available via mlir-aie";
    }

    std::string aie_home = get_env("AIE_HOME");
    if (!aie_home.empty()) {
        return "AIE_HOME set but aiecc.py not found at: " + aie_home + "/bin/aiecc.py";
    }

    return "mlir-aie not found. Set AIE_HOME or install mlir-aie to enable JIT compilation";
}

//====//
// JIT compile any kernel using mlir-aie
// Generic function that:
//   1. Loads the MLIR template from kernels/amd/
//   2. Copies it to a temp file
//   3. Calls aiecc.py to compile
//   4. Returns the xclbin data
//====//
std::vector<uint8_t> jit_compile_kernel(const std::string& op_name,
                                         const std::string& profile_str,
                                         int npu_profile) {
    std::string aiecc = find_aiecc();
    if (aiecc.empty()) {
        std::cerr << "Info: mlir-aie (aiecc.py) not found. Skipping JIT compilation.\n";
        std::cerr << "  Set AIE_HOME to the mlir-aie build directory.\n";
        std::cerr << "  Prebuilt xclbins will be needed.\n";
        return {};
    }

    std::string repo_root = find_repo_root();
    if (repo_root.empty()) {
        std::cerr << "Warning: could not find repository root for JIT compilation.\n";
        std::cerr << "  Set GGNPU_REPO_ROOT or place kernels/amd/ relative to the binary.\n";
        return {};
    }

    // Load MLIR template
    std::string mlir_source = load_mlir_template(op_name, repo_root);
    if (mlir_source.empty()) {
        std::cerr << "Error: could not load MLIR template for " << op_name << "\n";
        return {};
    }

    // Create temp directory for compilation
    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path tmp_dir = fs::path(cache_dir) / "xclbin_tmp";
    fs::create_directories(tmp_dir);

    fs::path mlir_file = tmp_dir / (op_name + "_" + profile_str + ".mlir");
    fs::path xclbin_file = tmp_dir / (op_name + "_" + profile_str + ".xclbin");

    // Write MLIR to temp file
    if (!write_file(mlir_file.string(), mlir_source)) {
        std::cerr << "Error: failed to write MLIR temp file: " << mlir_file << "\n";
        return {};
    }

    // Build compilation command
    std::string cmd = "\"" + aiecc + "\"";
    cmd += " --target=aie2p";
    cmd += " --npu-profile=" + std::to_string(npu_profile);

    std::string aie_home = get_env("AIE_HOME");
    if (!aie_home.empty()) {
        cmd += " -I\"" + aie_home + "/include\"";
    }

    std::string peano_home = get_env("PEANO_HOME");
    if (!peano_home.empty()) {
        cmd += " -I\"" + peano_home + "/include\"";
        cmd += " -L\"" + peano_home + "/lib\"";
    }

    cmd += " \"" + mlir_file.string() + "\"";
    cmd += " -o \"" + xclbin_file.string() + "\"";

    // Execute compilation
    std::cerr << "JIT: compiling " << op_name << " for " << profile_str << "\n";
    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Error: JIT compilation failed (exit code " << result << ")\n";
        fs::remove(mlir_file);
        return {};
    }

    // Read compiled xclbin
    if (!fs::exists(xclbin_file)) {
        std::cerr << "Error: xclbin not produced by compiler\n";
        fs::remove(mlir_file);
        return {};
    }

    std::vector<uint8_t> xclbin_data = load_xclbin_file(xclbin_file.string());
    if (xclbin_data.empty()) {
        std::cerr << "Error: xclbin is empty\n";
        fs::remove(mlir_file);
        fs::remove(xclbin_file);
        return {};
    }

    std::cerr << "JIT: compiled " << xclbin_data.size() << " bytes\n";

    // Clean up temp files
    fs::remove(mlir_file);
    fs::remove(xclbin_file);

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

    return jit_compile_kernel("rmsnorm", profile_str, npu_profile);
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
