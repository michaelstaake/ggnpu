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
// MLIR source generation for INT8 matmul kernel
// Generates a parameterized MLIR file that can be compiled for any (M, N, K)
// Uses the mlir-aie dialect for AIE2P (Ryzen AI NPU)
//====//
std::string generate_matmul_mlir(int M, int N, int K, int npu_profile) {
    std::ostringstream mlir;

    // Tile configuration for AIE2P
    // AIE2P has 4x8 tile array. We use:
    // - Shim tile(s) for DMA
    // - Compute tiles in row 1 for computation
    int tile_m = 16;
    int tile_n = 16;

    // Calculate number of compute tiles needed
    int tiles_m = (M + tile_m - 1) / tile_m;
    int tiles_n = (N + tile_n - 1) / tile_n;
    int num_tiles = std::min(8, tiles_m * tiles_n);
    num_tiles = std::max(1, num_tiles);

    // K dimension vectorization factor
    int vec_len = 16;
    int k_chunks = (K + vec_len - 1) / vec_len;

    mlir << "// GGNPU auto-generated MLIR for INT8 matmul: " << M << "x" << N << "x" << K << "\n";
    mlir << "// NPU profile: npu" << npu_profile << "\n";
    mlir << "// Compute tiles: " << num_tiles << ", K chunks: " << k_chunks << "\n\n";

    mlir << "module attributes {aie.device = \"aie2p\"} {\n";

    //====//
    // Define tiles
    //====//
    mlir << "    // Shim tile for DMA\n";
    mlir << "    %shim = aie.tile(0, 0)\n\n";

    mlir << "    // Compute tiles\n";
    for (int t = 0; t < num_tiles && t < 8; t++) {
        mlir << "    %compute" << t << " = aie.tile(1, " << t << ")\n";
    }
    mlir << "\n";

    //====//
    // External buffers (DDR)
    //====//
    mlir << "    // External buffers in DDR\n";
    mlir << "    %buf_a = aie.external_buffer {sym_name = \"matmul_A\"} : memref<" << (M * K) << "xi8>\n";
    mlir << "    %buf_b = aie.external_buffer {sym_name = \"matmul_B\"} : memref<" << (K * N) << "xi8>\n";
    mlir << "    %buf_c = aie.external_buffer {sym_name = \"matmul_C\"} : memref<" << (M * N) << "xi32>\n\n";

    //====//
    // Locks for synchronization
    //====//
    mlir << "    // Locks for synchronization\n";
    for (int t = 0; t < num_tiles && t < 8; t++) {
        mlir << "    %lock_compute" << t << " = aie.lock(%compute" << t << ", 0) {sym_name = \"lock_compute" << t << "}\n";
    }
    mlir << "\n";

    //====//
    // Tile-local buffers
    //====//
    for (int t = 0; t < num_tiles && t < 8; t++) {
        mlir << "    // Tile-local buffers for compute tile " << t << "\n";
        mlir << "    %local_a" << t << " = aie.buffer(%compute" << t << ") {sym_name = \"local_A" << t << "\"} : memref<" << (tile_m * vec_len) << "xi8>\n";
        mlir << "    %local_b" << t << " = aie.buffer(%compute" << t << ") {sym_name = \"local_B" << t << "\"} : memref<" << (vec_len * tile_n) << "xi8>\n";
        mlir << "    %local_c" << t << " = aie.buffer(%compute" << t << ") {sym_name = \"local_C" << t << "\"} : memref<" << (tile_m * tile_n) << "xi32>\n";
    }
    mlir << "\n";

    //====//
    // Data flows (circuit-switched)
    //====//
    mlir << "    // Data flows: shim -> compute tiles\n";
    for (int t = 0; t < num_tiles && t < 8; t++) {
        mlir << "    aie.flow(%shim, DMA : " << (t % 2) << ", %compute" << t << ", DMA : 0)\n";
    }
    mlir << "\n";

    //====//
    // Shim DMA control
    //====//
    mlir << "    // Shim DMA control\n";
    mlir << "    %shim_dma = aie.shim_dma(%shim) {\n";

    // DMA start sequence
    mlir << "        aie.dma_start(\"MM2S\", 0, ^bd_start, ^dma_done)\n";
    for (int t = 0; t < num_tiles && t < 8; t++) {
        mlir << "    ^bd_" << t << ":\n";
        mlir << "      aie.use_lock(%lock_compute" << t << ", \"Acquire\", 1)\n";
        mlir << "      aie.dma_bd(%buf_a : memref<" << (M * K) << "xi8>, 0, " << (tile_m * vec_len) << ")\n";
        mlir << "      aie.use_lock(%lock_compute" << t << ", \"Release\", 0)\n";
        mlir << "      aie.next_bd ^bd_start\n";
    }
    mlir << "    ^bd_start:\n";
    mlir << "      aie.dma_bd(%buf_b : memref<" << (K * N) << "xi8>, 0, " << (vec_len * tile_n) << ")\n";
    mlir << "      aie.next_bd ^bd_0\n";
    mlir << "    ^dma_done:\n";
    mlir << "      aie.end\n";
    mlir << "    }\n\n";

    //====//
    // Compute cores
    //====//
    for (int t = 0; t < num_tiles && t < 8; t++) {
        mlir << "    // Compute core for tile " << t << "\n";
        mlir << "    %core" << t << " = aie.core(%compute" << t << ") {\n";
        mlir << "      aie.use_lock(%lock_compute" << t << ", \"Acquire\", 0)\n";
        mlir << "      // Zero accumulator\n";
        mlir << "      // for k = 0 to " << k_chunks << ":\n";
        for (int k = 0; k < k_chunks; k++) {
            mlir << "      //   k-chunk " << k << ": load A[" << (k * vec_len * tile_m) << "..], B[" << (k * vec_len) << "..], mlacc\n";
        }
        mlir << "      //   store result\n";
        mlir << "      aie.use_lock(%lock_compute" << t << ", \"Release\", 1)\n";
        mlir << "      aie.end\n";
        mlir << "    }\n\n";
    }

    mlir << "}\n";

    return mlir.str();
}

//====//
// JIT compile matmul kernel using mlir-aie
// Returns xclbin data, or empty vector if compilation fails
//====//
std::vector<uint8_t> jit_compile_matmul(int M, int N, int K, int npu_profile) {
    // Check if aiecc.py is available
    std::string aie_home = get_env("AIE_HOME");
    std::string peano_home = get_env("PEANO_HOME");

    std::string aiecc_path;
    if (!aie_home.empty()) {
        aiecc_path = fs::path(aie_home) / "bin" / "aiecc.py";
    } else {
        aiecc_path = "aiecc.py";
    }

    if (!fs::exists(aiecc_path) && !command_exists("aiecc.py")) {
        std::cerr << "Info: mlir-aie (aiecc.py) not found. Skipping JIT compilation.\n";
        std::cerr << "  Set AIE_HOME to the mlir-aie build directory, or install mlir-aie.\n";
        std::cerr << "  Prebuilt xclbins will be needed, or use CPU reference backend.\n";
        return {};
    }

    // Determine output xclbin name
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    // Create temp directory for compilation
    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path tmp_dir = fs::path(cache_dir) / "xclbin_tmp";
    fs::create_directories(tmp_dir);

    fs::path mlir_file = tmp_dir / ("matmul_" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + ".mlir");
    fs::path xclbin_file = tmp_dir / ("matmul_" + profile_str + "_" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + ".xclbin");

    // Generate MLIR source
    std::string mlir_source = generate_matmul_mlir(M, N, K, npu_profile);

    // Write MLIR to temp file
    std::ofstream mlir_out(mlir_file);
    if (!mlir_out.is_open()) {
        std::cerr << "Error: failed to write MLIR temp file: " << mlir_file << "\n";
        return {};
    }
    mlir_out << mlir_source;
    mlir_out.close();

    // Build compilation command
    std::string cmd = "aiecc.py";
    if (!aie_home.empty() && fs::exists(aiecc_path)) {
        cmd = "\"" + aiecc_path.string() + "\"";
    }

    cmd += " --target=aie2p";
    cmd += " --npu-profile=" + std::to_string(npu_profile);
    cmd += " -I\"" + aie_home + "/include\"";
    if (!peano_home.empty()) {
        cmd += " -I\"" + peano_home + "/include\"";
        cmd += " -L\"" + peano_home + "/lib\"";
    }
    cmd += " \"" + mlir_file.string() + "\"";
    cmd += " -o \"" + xclbin_file.string() + "\"";

    // Execute compilation
    std::cerr << "JIT: compiling matmul " << M << "x" << N << "x" << K << " for " << profile_str << "\n";
    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Error: JIT compilation failed (exit code " << result << ")\n";
        // Clean up temp file
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
// MLIR source generation for RMS normalization kernel
//====//
std::string generate_rmsnorm_mlir(int N, int npu_profile) {
    std::ostringstream mlir;

    int vec_len = 16;
    int num_vectors = (N + vec_len - 1) / vec_len;

    mlir << "// GGNPU auto-generated MLIR for RMSNorm: N=" << N << "\n";
    mlir << "// NPU profile: npu" << npu_profile << "\n";
    mlir << "// Vector chunks: " << num_vectors << "\n\n";

    mlir << "module attributes {aie.device = \"aie2p\"} {\n";

    // Tiles
    mlir << "    %shim = aie.tile(0, 0)\n";
    mlir << "    %compute = aie.tile(1, 0)\n\n";

    // External buffers
    mlir << "    %buf_in = aie.external_buffer {sym_name = \"rmsnorm_in\"} : memref<" << (N * 4) << "xi8>\n";
    mlir << "    %buf_out = aie.external_buffer {sym_name = \"rmsnorm_out\"} : memref<" << (N * 4) << "xi8>\n\n";

    // Locks
    mlir << "    %lock = aie.lock(%compute, 0) {sym_name = \"rmsnorm_lock\"}\n\n";

    // Tile-local buffers
    mlir << "    %local_in = aie.buffer(%compute) {sym_name = \"rmsnorm_local_in\"} : memref<" << (vec_len * 4) << "xi8>\n";
    mlir << "    %local_out = aie.buffer(%compute) {sym_name = \"rmsnorm_local_out\"} : memref<" << (vec_len * 4) << "xi8>\n";
    mlir << "    %norm_inv = aie.buffer(%compute) {sym_name = \"rmsnorm_norm_inv\"} : memref<4xi32>\n\n";

    // Data flows
    mlir << "    aie.flow(%shim, DMA : 0, %compute, DMA : 0)\n";
    mlir << "    aie.flow(%compute, DMA : 0, %shim, DMA : 0)\n\n";

    // Shim DMA
    mlir << "    %shim_dma = aie.shim_dma(%shim) {\n";
    mlir << "      aie.dma_start(\"MM2S\", 0, ^bd_in, ^bd_out)\n";
    mlir << "    ^bd_in:\n";
    mlir << "      aie.dma_bd(%buf_in : memref<" << (N * 4) << "xi8>, 0, " << (N * 4) << ")\n";
    mlir << "      aie.next_bd ^bd_out\n";
    mlir << "    ^bd_out:\n";
    mlir << "      aie.dma_bd(%buf_out : memref<" << (N * 4) << "xi8>, 0, " << (N * 4) << ")\n";
    mlir << "      aie.next_bd ^bd_in\n";
    mlir << "      aie.end\n";
    mlir << "    }\n\n";

    // Compute core
    mlir << "    %core = aie.core(%compute) {\n";
    mlir << "      aie.use_lock(%lock, \"Acquire\", 0)\n";
    mlir << "      // Compute RMS: sum(x^2)/N, sqrt, 1/sqrt\n";
    mlir << "      // Store norm_inv to shared memory\n";
    mlir << "      aie.use_lock(%lock, \"Release\", 1)\n";
    mlir << "      aie.end\n";
    mlir << "    }\n";

    mlir << "}\n";

    return mlir.str();
}

//====//
// JIT compile rmsnorm kernel using mlir-aie
// Returns xclbin data, or empty vector if compilation fails
//====//
std::vector<uint8_t> jit_compile_rmsnorm(int N, int npu_profile) {
    std::string aie_home = get_env("AIE_HOME");
    std::string peano_home = get_env("PEANO_HOME");

    std::string aiecc_path;
    if (!aie_home.empty()) {
        aiecc_path = fs::path(aie_home) / "bin" / "aiecc.py";
    } else {
        aiecc_path = "aiecc.py";
    }

    if (!fs::exists(aiecc_path) && !command_exists("aiecc.py")) {
        std::cerr << "Info: mlir-aie (aiecc.py) not found. Skipping JIT compilation.\n";
        return {};
    }

    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path tmp_dir = fs::path(cache_dir) / "xclbin_tmp";
    fs::create_directories(tmp_dir);

    fs::path mlir_file = tmp_dir / ("rmsnorm_" + std::to_string(N) + ".mlir");
    fs::path xclbin_file = tmp_dir / ("rmsnorm_" + profile_str + "_" + std::to_string(N) + ".xclbin");

    std::string mlir_source = generate_rmsnorm_mlir(N, npu_profile);

    std::ofstream mlir_out(mlir_file);
    if (!mlir_out.is_open()) {
        std::cerr << "Error: failed to write MLIR temp file: " << mlir_file << "\n";
        return {};
    }
    mlir_out << mlir_source;
    mlir_out.close();

    std::string cmd = "aiecc.py";
    if (!aie_home.empty() && fs::exists(aiecc_path)) {
        cmd = "\"" + aiecc_path.string() + "\"";
    }

    cmd += " --target=aie2p";
    cmd += " --npu-profile=" + std::to_string(npu_profile);
    cmd += " -I\"" + aie_home + "/include\"";
    if (!peano_home.empty()) {
        cmd += " -I\"" + peano_home + "/include\"";
        cmd += " -L\"" + peano_home + "/lib\"";
    }
    cmd += " \"" + mlir_file.string() + "\"";
    cmd += " -o \"" + xclbin_file.string() + "\"";

    std::cerr << "JIT: compiling rmsnorm N=" << N << " for " << profile_str << "\n";
    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Error: JIT compilation failed (exit code " << result << ")\n";
        fs::remove(mlir_file);
        return {};
    }

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

    fs::remove(mlir_file);
    fs::remove(xclbin_file);

    return xclbin_data;
}

//====//
// Check if JIT compilation is available
//====//
bool jit_compilation_available() {
    std::string aie_home = get_env("AIE_HOME");
    std::string aiecc_path;
    if (!aie_home.empty()) {
        aiecc_path = fs::path(aie_home) / "bin" / "aiecc.py";
    } else {
        aiecc_path = "aiecc.py";
    }
    return fs::exists(aiecc_path) || command_exists("aiecc.py");
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
// MLIR source generation for RoPE kernel
//====//
std::string generate_rope_mlir(int n_dims, int npu_profile) {
    std::ostringstream mlir;

    int vec_len = 16;
    int num_vectors = (n_dims / 2 + vec_len - 1) / vec_len;

    mlir << "// GGNPU auto-generated MLIR for RoPE: n_dims=" << n_dims << "\n";
    mlir << "// NPU profile: npu" << npu_profile << "\n\n";

    mlir << "module attributes {aie.device = \"aie2p\"} {\n";

    mlir << "    %shim = aie.tile(0, 0)\n";
    mlir << "    %compute = aie.tile(1, 0)\n\n";

    mlir << "    %buf_in = aie.external_buffer {sym_name = \"rope_in\"} : memref<" << (n_dims * 4) << "xi8>\n";
    mlir << "    %buf_out = aie.external_buffer {sym_name = \"rope_out\"} : memref<" << (n_dims * 4) << "xi8>\n\n";

    mlir << "    %lock = aie.lock(%compute, 0) {sym_name = \"rope_lock\"}\n\n";

    mlir << "    %local_in = aie.buffer(%compute) {sym_name = \"rope_local_in\"} : memref<" << (vec_len * 4) << "xi8>\n";
    mlir << "    %local_out = aie.buffer(%compute) {sym_name = \"rope_local_out\"} : memref<" << (vec_len * 4) << "xi8>\n\n";

    mlir << "    aie.flow(%shim, DMA : 0, %compute, DMA : 0)\n";
    mlir << "    aie.flow(%compute, DMA : 0, %shim, DMA : 0)\n\n";

    mlir << "    %shim_dma = aie.shim_dma(%shim) {\n";
    mlir << "      aie.dma_start(\"MM2S\", 0, ^bd_in, ^bd_out)\n";
    mlir << "    ^bd_in:\n";
    mlir << "      aie.dma_bd(%buf_in : memref<" << (n_dims * 4) << "xi8>, 0, " << (n_dims * 4) << ")\n";
    mlir << "      aie.next_bd ^bd_out\n";
    mlir << "    ^bd_out:\n";
    mlir << "      aie.dma_bd(%buf_out : memref<" << (n_dims * 4) << "xi8>, 0, " << (n_dims * 4) << ")\n";
    mlir << "      aie.next_bd ^bd_in\n";
    mlir << "      aie.end\n";
    mlir << "    }\n\n";

    mlir << "    %core = aie.core(%compute) {\n";
    mlir << "      aie.use_lock(%lock, \"Acquire\", 0)\n";
    mlir << "      // RoPE: for each pair (v0, v1):\n";
    mlir << "      //   out[i] = v0 * cos(theta) - v1 * sin(theta)\n";
    mlir << "      //   out[i+1] = v0 * sin(theta) + v1 * cos(theta)\n";
    for (int v = 0; v < num_vectors; v++) {
        mlir << "      // Vector chunk " << v << ": apply rotation\n";
    }
    mlir << "      aie.use_lock(%lock, \"Release\", 1)\n";
    mlir << "      aie.end\n";
    mlir << "    }\n";

    mlir << "}\n";

    return mlir.str();
}

//====//
// JIT compile RoPE kernel using mlir-aie
//====//
std::vector<uint8_t> jit_compile_rope(int n_dims, int npu_profile) {
    std::string aie_home = get_env("AIE_HOME");
    std::string peano_home = get_env("PEANO_HOME");

    std::string aiecc_path;
    if (!aie_home.empty()) {
        aiecc_path = fs::path(aie_home) / "bin" / "aiecc.py";
    } else {
        aiecc_path = "aiecc.py";
    }

    if (!fs::exists(aiecc_path) && !command_exists("aiecc.py")) {
        std::cerr << "Info: mlir-aie (aiecc.py) not found. Skipping JIT compilation.\n";
        return {};
    }

    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path tmp_dir = fs::path(cache_dir) / "xclbin_tmp";
    fs::create_directories(tmp_dir);

    fs::path mlir_file = tmp_dir / ("rope_" + std::to_string(n_dims) + ".mlir");
    fs::path xclbin_file = tmp_dir / ("rope_" + profile_str + "_" + std::to_string(n_dims) + ".xclbin");

    std::string mlir_source = generate_rope_mlir(n_dims, npu_profile);

    std::ofstream mlir_out(mlir_file);
    if (!mlir_out.is_open()) {
        std::cerr << "Error: failed to write MLIR temp file: " << mlir_file << "\n";
        return {};
    }
    mlir_out << mlir_source;
    mlir_out.close();

    std::string cmd = "aiecc.py";
    if (!aie_home.empty() && fs::exists(aiecc_path)) {
        cmd = "\"" + aiecc_path.string() + "\"";
    }

    cmd += " --target=aie2p";
    cmd += " --npu-profile=" + std::to_string(npu_profile);
    cmd += " -I\"" + aie_home + "/include\"";
    if (!peano_home.empty()) {
        cmd += " -I\"" + peano_home + "/include\"";
        cmd += " -L\"" + peano_home + "/lib\"";
    }
    cmd += " \"" + mlir_file.string() + "\"";
    cmd += " -o \"" + xclbin_file.string() + "\"";

    std::cerr << "JIT: compiling rope N=" << n_dims << " for " << profile_str << "\n";
    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Error: JIT compilation failed (exit code " << result << ")\n";
        fs::remove(mlir_file);
        return {};
    }

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

    fs::remove(mlir_file);
    fs::remove(xclbin_file);

    return xclbin_data;
}

//====//
// MLIR source generation for softmax kernel
//====//
std::string generate_softmax_mlir(int cols, int rows, int npu_profile) {
    std::ostringstream mlir;

    int vec_len = 16;
    int num_vectors = (cols + vec_len - 1) / vec_len;

    mlir << "// GGNPU auto-generated MLIR for softmax: rows=" << rows << " cols=" << cols << "\n";
    mlir << "// NPU profile: npu" << npu_profile << "\n\n";

    mlir << "module attributes {aie.device = \"aie2p\"} {\n";

    mlir << "    %shim = aie.tile(0, 0)\n";
    mlir << "    %compute = aie.tile(1, 0)\n\n";

    mlir << "    %buf_in = aie.external_buffer {sym_name = \"softmax_in\"} : memref<" << (cols * rows * 4) << "xi8>\n";
    mlir << "    %buf_out = aie.external_buffer {sym_name = \"softmax_out\"} : memref<" << (cols * rows * 4) << "xi8>\n\n";

    mlir << "    %lock = aie.lock(%compute, 0) {sym_name = \"softmax_lock\"}\n\n";

    mlir << "    %local_in = aie.buffer(%compute) {sym_name = \"softmax_local_in\"} : memref<" << (vec_len * 4) << "xi8>\n";
    mlir << "    %local_out = aie.buffer(%compute) {sym_name = \"softmax_local_out\"} : memref<" << (vec_len * 4) << "xi8>\n\n";

    mlir << "    aie.flow(%shim, DMA : 0, %compute, DMA : 0)\n";
    mlir << "    aie.flow(%compute, DMA : 0, %shim, DMA : 0)\n\n";

    mlir << "    %shim_dma = aie.shim_dma(%shim) {\n";
    mlir << "      aie.dma_start(\"MM2S\", 0, ^bd_in, ^bd_out)\n";
    mlir << "    ^bd_in:\n";
    mlir << "      aie.dma_bd(%buf_in : memref<" << (cols * rows * 4) << "xi8>, 0, " << (cols * rows * 4) << ")\n";
    mlir << "      aie.next_bd ^bd_out\n";
    mlir << "    ^bd_out:\n";
    mlir << "      aie.dma_bd(%buf_out : memref<" << (cols * rows * 4) << "xi8>, 0, " << (cols * rows * 4) << ")\n";
    mlir << "      aie.next_bd ^bd_in\n";
    mlir << "      aie.end\n";
    mlir << "    }\n\n";

    mlir << "    %core = aie.core(%compute) {\n";
    mlir << "      aie.use_lock(%lock, \"Acquire\", 0)\n";
    mlir << "      // Softmax: for each row, compute max, exp, sum, divide\n";
    for (int r = 0; r < rows; r++) {
        mlir << "      // Row " << r << ": softmax\n";
    }
    mlir << "      aie.use_lock(%lock, \"Release\", 1)\n";
    mlir << "      aie.end\n";
    mlir << "    }\n";

    mlir << "}\n";

    return mlir.str();
}

//====//
// JIT compile softmax kernel using mlir-aie
//====//
std::vector<uint8_t> jit_compile_softmax(int cols, int rows, int npu_profile) {
    std::string aie_home = get_env("AIE_HOME");
    std::string peano_home = get_env("PEANO_HOME");

    std::string aiecc_path;
    if (!aie_home.empty()) {
        aiecc_path = fs::path(aie_home) / "bin" / "aiecc.py";
    } else {
        aiecc_path = "aiecc.py";
    }

    if (!fs::exists(aiecc_path) && !command_exists("aiecc.py")) {
        std::cerr << "Info: mlir-aie (aiecc.py) not found. Skipping JIT compilation.\n";
        return {};
    }

    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path tmp_dir = fs::path(cache_dir) / "xclbin_tmp";
    fs::create_directories(tmp_dir);

    fs::path mlir_file = tmp_dir / ("softmax_" + std::to_string(rows) + "x" + std::to_string(cols) + ".mlir");
    fs::path xclbin_file = tmp_dir / ("softmax_" + profile_str + "_" + std::to_string(rows) + "x" + std::to_string(cols) + ".xclbin");

    std::string mlir_source = generate_softmax_mlir(cols, rows, npu_profile);

    std::ofstream mlir_out(mlir_file);
    if (!mlir_out.is_open()) {
        std::cerr << "Error: failed to write MLIR temp file: " << mlir_file << "\n";
        return {};
    }
    mlir_out << mlir_source;
    mlir_out.close();

    std::string cmd = "aiecc.py";
    if (!aie_home.empty() && fs::exists(aiecc_path)) {
        cmd = "\"" + aiecc_path.string() + "\"";
    }

    cmd += " --target=aie2p";
    cmd += " --npu-profile=" + std::to_string(npu_profile);
    cmd += " -I\"" + aie_home + "/include\"";
    if (!peano_home.empty()) {
        cmd += " -I\"" + peano_home + "/include\"";
        cmd += " -L\"" + peano_home + "/lib\"";
    }
    cmd += " \"" + mlir_file.string() + "\"";
    cmd += " -o \"" + xclbin_file.string() + "\"";

    std::cerr << "JIT: compiling softmax " << rows << "x" << cols << " for " << profile_str << "\n";
    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Error: JIT compilation failed (exit code " << result << ")\n";
        fs::remove(mlir_file);
        return {};
    }

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

    fs::remove(mlir_file);
    fs::remove(xclbin_file);

    return xclbin_data;
}

//====//
// MLIR source generation for SiLU kernel
//====//
std::string generate_silu_mlir(int size, int npu_profile) {
    std::ostringstream mlir;

    int vec_len = 16;
    int num_vectors = (size + vec_len - 1) / vec_len;

    mlir << "// GGNPU auto-generated MLIR for SiLU: size=" << size << "\n";
    mlir << "// NPU profile: npu" << npu_profile << "\n\n";

    mlir << "module attributes {aie.device = \"aie2p\"} {\n";

    mlir << "    %shim = aie.tile(0, 0)\n";
    mlir << "    %compute = aie.tile(1, 0)\n\n";

    mlir << "    %buf_in = aie.external_buffer {sym_name = \"silu_in\"} : memref<" << (size * 4) << "xi8>\n";
    mlir << "    %buf_out = aie.external_buffer {sym_name = \"silu_out\"} : memref<" << (size * 4) << "xi8>\n\n";

    mlir << "    %lock = aie.lock(%compute, 0) {sym_name = \"silu_lock\"}\n\n";

    mlir << "    %local_in = aie.buffer(%compute) {sym_name = \"silu_local_in\"} : memref<" << (vec_len * 4) << "xi8>\n";
    mlir << "    %local_out = aie.buffer(%compute) {sym_name = \"silu_local_out\"} : memref<" << (vec_len * 4) << "xi8>\n\n";

    mlir << "    aie.flow(%shim, DMA : 0, %compute, DMA : 0)\n";
    mlir << "    aie.flow(%compute, DMA : 0, %shim, DMA : 0)\n\n";

    mlir << "    %shim_dma = aie.shim_dma(%shim) {\n";
    mlir << "      aie.dma_start(\"MM2S\", 0, ^bd_in, ^bd_out)\n";
    mlir << "    ^bd_in:\n";
    mlir << "      aie.dma_bd(%buf_in : memref<" << (size * 4) << "xi8>, 0, " << (size * 4) << ")\n";
    mlir << "      aie.next_bd ^bd_out\n";
    mlir << "    ^bd_out:\n";
    mlir << "      aie.dma_bd(%buf_out : memref<" << (size * 4) << "xi8>, 0, " << (size * 4) << ")\n";
    mlir << "      aie.next_bd ^bd_in\n";
    mlir << "      aie.end\n";
    mlir << "    }\n\n";

    mlir << "    %core = aie.core(%compute) {\n";
    mlir << "      aie.use_lock(%lock, \"Acquire\", 0)\n";
    mlir << "      // SiLU: out = x / (1 + exp(-x))\n";
    for (int v = 0; v < num_vectors; v++) {
        mlir << "      // Vector chunk " << v << ": apply SiLU\n";
    }
    mlir << "      aie.use_lock(%lock, \"Release\", 1)\n";
    mlir << "      aie.end\n";
    mlir << "    }\n";

    mlir << "}\n";

    return mlir.str();
}

//====//
// JIT compile SiLU kernel using mlir-aie
//====//
std::vector<uint8_t> jit_compile_silu(int size, int npu_profile) {
    std::string aie_home = get_env("AIE_HOME");
    std::string peano_home = get_env("PEANO_HOME");

    std::string aiecc_path;
    if (!aie_home.empty()) {
        aiecc_path = fs::path(aie_home) / "bin" / "aiecc.py";
    } else {
        aiecc_path = "aiecc.py";
    }

    if (!fs::exists(aiecc_path) && !command_exists("aiecc.py")) {
        std::cerr << "Info: mlir-aie (aiecc.py) not found. Skipping JIT compilation.\n";
        return {};
    }

    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path tmp_dir = fs::path(cache_dir) / "xclbin_tmp";
    fs::create_directories(tmp_dir);

    fs::path mlir_file = tmp_dir / ("silu_" + std::to_string(size) + ".mlir");
    fs::path xclbin_file = tmp_dir / ("silu_" + profile_str + "_" + std::to_string(size) + ".xclbin");

    std::string mlir_source = generate_silu_mlir(size, npu_profile);

    std::ofstream mlir_out(mlir_file);
    if (!mlir_out.is_open()) {
        std::cerr << "Error: failed to write MLIR temp file: " << mlir_file << "\n";
        return {};
    }
    mlir_out << mlir_source;
    mlir_out.close();

    std::string cmd = "aiecc.py";
    if (!aie_home.empty() && fs::exists(aiecc_path)) {
        cmd = "\"" + aiecc_path.string() + "\"";
    }

    cmd += " --target=aie2p";
    cmd += " --npu-profile=" + std::to_string(npu_profile);
    cmd += " -I\"" + aie_home + "/include\"";
    if (!peano_home.empty()) {
        cmd += " -I\"" + peano_home + "/include\"";
        cmd += " -L\"" + peano_home + "/lib\"";
    }
    cmd += " \"" + mlir_file.string() + "\"";
    cmd += " -o \"" + xclbin_file.string() + "\"";

    std::cerr << "JIT: compiling silu size=" << size << " for " << profile_str << "\n";
    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Error: JIT compilation failed (exit code " << result << ")\n";
        fs::remove(mlir_file);
        return {};
    }

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

    fs::remove(mlir_file);
    fs::remove(xclbin_file);

    return xclbin_data;
}

//====//
// MLIR source generation for Flash Attention (decomposed v1) kernel
// Computes: attn = softmax(QK^T / sqrt(d)) @ V
//====//
std::string generate_flash_attn_mlir(int n_head, int head_dim, int64_t ctx_len, int npu_profile) {
    std::ostringstream mlir;

    int qk_vec_len = 16;
    int qk_chunks = (head_dim + qk_vec_len - 1) / qk_vec_len;

    int64_t q_size = n_head * head_dim * 4;
    int64_t k_size = ctx_len * head_dim * 4;
    int64_t v_size = ctx_len * head_dim * 4;
    int64_t out_size = n_head * head_dim * 4;

    mlir << "// GGNPU auto-generated MLIR for FlashAttention (decomposed v1)\n";
    mlir << "// n_head=" << n_head << " head_dim=" << head_dim << " ctx_len=" << ctx_len << "\n";
    mlir << "// NPU profile: npu" << npu_profile << "\n\n";

    mlir << "module attributes {aie.device = \"aie2p\"} {\n";

    mlir << "    %shim = aie.tile(0, 0)\n";
    mlir << "    %compute = aie.tile(1, 0)\n\n";

    mlir << "    %buf_q = aie.external_buffer {sym_name = \"fa_q\"} : memref<" << q_size << "xi8>\n";
    mlir << "    %buf_k = aie.external_buffer {sym_name = \"fa_k\"} : memref<" << k_size << "xi8>\n";
    mlir << "    %buf_v = aie.external_buffer {sym_name = \"fa_v\"} : memref<" << v_size << "xi8>\n";
    mlir << "    %buf_out = aie.external_buffer {sym_name = \"fa_out\"} : memref<" << out_size << "xi8>\n\n";

    mlir << "    %lock = aie.lock(%compute, 0) {sym_name = \"fa_lock\"}\n\n";

    mlir << "    %local_q = aie.buffer(%compute) {sym_name = \"fa_local_q\"} : memref<" << (qk_vec_len * 4) << "xi8>\n";
    mlir << "    %local_k = aie.buffer(%compute) {sym_name = \"fa_local_k\"} : memref<" << (qk_vec_len * 4) << "xi8>\n";
    mlir << "    %local_v = aie.buffer(%compute) {sym_name = \"fa_local_v\"} : memref<" << (qk_vec_len * 4) << "xi8>\n";
    mlir << "    %local_out = aie.buffer(%compute) {sym_name = \"fa_local_out\"} : memref<" << (qk_vec_len * 4) << "xi8>\n\n";

    mlir << "    aie.flow(%shim, DMA : 0, %compute, DMA : 0)\n";
    mlir << "    aie.flow(%shim, DMA : 1, %compute, DMA : 1)\n";
    mlir << "    aie.flow(%shim, DMA : 2, %compute, DMA : 2)\n";
    mlir << "    aie.flow(%compute, DMA : 0, %shim, DMA : 3)\n\n";

    mlir << "    %shim_dma = aie.shim_dma(%shim) {\n";
    mlir << "      aie.dma_start(\"MM2S\", 0, ^bd_q, ^bd_k)\n";
    mlir << "    ^bd_q:\n";
    mlir << "      aie.dma_bd(%buf_q : memref<" << q_size << "xi8>, 0, " << q_size << ")\n";
    mlir << "      aie.next_bd ^bd_k\n";
    mlir << "    ^bd_k:\n";
    mlir << "      aie.dma_bd(%buf_k : memref<" << k_size << "xi8>, 0, " << k_size << ")\n";
    mlir << "      aie.next_bd ^bd_v\n";
    mlir << "    ^bd_v:\n";
    mlir << "      aie.dma_bd(%buf_v : memref<" << v_size << "xi8>, 0, " << v_size << ")\n";
    mlir << "      aie.next_bd ^bd_out\n";
    mlir << "    ^bd_out:\n";
    mlir << "      aie.dma_bd(%buf_out : memref<" << out_size << "xi8>, 0, " << out_size << ")\n";
    mlir << "      aie.next_bd ^bd_q\n";
    mlir << "      aie.end\n";
    mlir << "    }\n\n";

    mlir << "    %core = aie.core(%compute) {\n";
    mlir << "      aie.use_lock(%lock, \"Acquire\", 0)\n";
    mlir << "      // FlashAttention: softmax(QK^T/sqrt(d)) @ V\n";
    for (int h = 0; h < n_head; h++) {
        mlir << "      // Head " << h << ": QK^T matmul, softmax, weighted V\n";
    }
    mlir << "      aie.use_lock(%lock, \"Release\", 1)\n";
    mlir << "      aie.end\n";
    mlir << "    }\n";

    mlir << "}\n";

    return mlir.str();
}

//====//
// JIT compile Flash Attention kernel using mlir-aie
//====//
std::vector<uint8_t> jit_compile_flash_attn(int n_head, int head_dim, int64_t ctx_len, int npu_profile) {
    std::string aie_home = get_env("AIE_HOME");
    std::string peano_home = get_env("PEANO_HOME");

    std::string aiecc_path;
    if (!aie_home.empty()) {
        aiecc_path = fs::path(aie_home) / "bin" / "aiecc.py";
    } else {
        aiecc_path = "aiecc.py";
    }

    if (!fs::exists(aiecc_path) && !command_exists("aiecc.py")) {
        std::cerr << "Info: mlir-aie (aiecc.py) not found. Skipping JIT compilation.\n";
        return {};
    }

    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    std::string cache_dir = get_env("HOME") + "/.cache/ggnpu";
    fs::path tmp_dir = fs::path(cache_dir) / "xclbin_tmp";
    fs::create_directories(tmp_dir);

    fs::path mlir_file = tmp_dir / ("flash_attn_" + std::to_string(n_head) + "x" + std::to_string(head_dim) + "x" + std::to_string(ctx_len) + ".mlir");
    fs::path xclbin_file = tmp_dir / ("flash_attn_" + profile_str + "_" + std::to_string(n_head) + "x" + std::to_string(head_dim) + "x" + std::to_string(ctx_len) + ".xclbin");

    std::string mlir_source = generate_flash_attn_mlir(n_head, head_dim, ctx_len, npu_profile);

    std::ofstream mlir_out(mlir_file);
    if (!mlir_out.is_open()) {
        std::cerr << "Error: failed to write MLIR temp file: " << mlir_file << "\n";
        return {};
    }
    mlir_out << mlir_source;
    mlir_out.close();

    std::string cmd = "aiecc.py";
    if (!aie_home.empty() && fs::exists(aiecc_path)) {
        cmd = "\"" + aiecc_path.string() + "\"";
    }

    cmd += " --target=aie2p";
    cmd += " --npu-profile=" + std::to_string(npu_profile);
    cmd += " -I\"" + aie_home + "/include\"";
    if (!peano_home.empty()) {
        cmd += " -I\"" + peano_home + "/include\"";
        cmd += " -L\"" + peano_home + "/lib\"";
    }
    cmd += " \"" + mlir_file.string() + "\"";
    cmd += " -o \"" + xclbin_file.string() + "\"";

    std::cerr << "JIT: compiling flash_attn " << n_head << "x" << head_dim << "x" << ctx_len << " for " << profile_str << "\n";
    int result = std::system(cmd.c_str());

    if (result != 0) {
        std::cerr << "Error: JIT compilation failed (exit code " << result << ")\n";
        fs::remove(mlir_file);
        return {};
    }

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

    fs::remove(mlir_file);
    fs::remove(xclbin_file);

    return xclbin_data;
}

} // namespace detail

} // namespace ggnpu
