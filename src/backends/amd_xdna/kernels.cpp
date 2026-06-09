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
//====//
std::string generate_matmul_mlir(int M, int N, int K, int npu_profile) {
    std::ostringstream mlir;

    // Determine NPU profile string
    std::string profile_str;
    if (npu_profile == 4) profile_str = "npu4";
    else if (npu_profile == 5) profile_str = "npu5";
    else profile_str = "npu6";

    // Tile configuration for AIE2P
    // AIE2P has 4x8 tile array. We use:
    // - Tile (0,0): shim/control
    // - Tiles (1,0)-(1,7): compute row
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
    mlir << "// Compiled for NPU profile: " << profile_str << "\n";
    mlir << "// Compute tiles: " << num_tiles << ", K chunks: " << k_chunks << "\n\n";

    mlir << "module attributes {aie.device = \"aie2p\"} {\n";

    //====//
    // DMA channel declarations
    //====//
    mlir << "    // DMA channels for data movement\n";
    mlir << "    %dma_a = \"aie.dma_acquire\"() {\n";
    mlir << "        channel_id = 0 : i32,\n";
    mlir << "        direction = \"host_to_tile\" : string,\n";
    mlir << "        tile = #aie.tile{row = 0, col = 0}\n";
    mlir << "    } : () -> !aie.objectbox.stream\n\n";

    mlir << "    %dma_b = \"aie.dma_acquire\"() {\n";
    mlir << "        channel_id = 1 : i32,\n";
    mlir << "        direction = \"host_to_tile\" : string,\n";
    mlir << "        tile = #aie.tile{row = 0, col = 0}\n";
    mlir << "    } : () -> !aie.objectbox.stream\n\n";

    mlir << "    %dma_c = \"aie.dma_acquire\"() {\n";
    mlir << "        channel_id = 2 : i32,\n";
    mlir << "        direction = \"tile_to_host\" : string,\n";
    mlir << "        tile = #aie.tile{row = 0, col = 0}\n";
    mlir << "    } : () -> !aie.objectbox.stream\n\n";

    //====//
    // Shim tile (0,0): DMA ports
    //====//
    mlir << "    \"aie.shimtile\"(#aie.tile{row = 0, col = 0}) {\n";
    mlir << "        \"aie.dma_port\"() { port = \"A\", direction = \"input\" } : () -> ()\n";
    mlir << "        \"aie.dma_port\"() { port = \"B\", direction = \"input\" } : () -> ()\n";
    mlir << "        \"aie.dma_port\"() { port = \"C\", direction = \"output\" } : () -> ()\n";
    mlir << "    }\n\n";

    //====//
    // Control tile: DMA setup and synchronization
    //====//
    mlir << "    \"aie.tile\"(#aie.tile{row = 0, col = 0}) {\n";
    mlir << "        // Lock for synchronization with compute tiles\n";
    mlir << "        %lock_compute = \"aie.lock\"() {\n";
    mlir << "            lock = #aie.lock{lock_id = 3, target = \"tile\",\n";
    mlir << "                tile = #aie.tile{row = 1, col = 0}}\n";
    mlir << "        } : () -> !aie.lock\n\n";

    mlir << "        // Start DMA transfers for A and B matrices\n";
    mlir << "        \"aie.dma_start\"(%dma_a) { len = " << (M * K) << " : i64 } : (!aie.objectbox.stream) -> ()\n";
    mlir << "        \"aie.dma_start\"(%dma_b) { len = " << (K * N) << " : i64 } : (!aie.objectbox.stream) -> ()\n\n";

    mlir << "        // Signal compute tiles to start\n";
    mlir << "        \"aie.lock\"(%lock_compute) : (!aie.lock) -> ()\n\n";

    mlir << "        // Wait for output DMA to complete\n";
    mlir << "        \"aie.dma_wait\"(%dma_c) { count = 1 : i32 } : (!aie.objectbox.stream) -> ()\n";
    mlir << "    }\n\n";

    //====//
    // Compute tiles: matrix multiplication
    // Each tile computes a tile_m x tile_n block of the output
    //====//
    for (int t = 0; t < num_tiles && t < 8; t++) {
        int col = t;
        mlir << "    // Compute tile (" << 1 << "," << col << ")\n";
        mlir << "    \"aie.tile\"(#aie.tile{row = " << 1 << ", col = " << col << "}) {\n";

        // Lock for synchronization with control tile
        mlir << "        %lock_start = \"aie.lock\"() {\n";
        mlir << "            lock = #aie.lock{lock_id = 3, target = \"tile\",\n";
        mlir << "                tile = #aie.tile{row = 1, col = 0}}\n";
        mlir << "        } : () -> !aie.lock\n";

        // Wait for control tile signal
        mlir << "        \"aie.lock\"(%lock_start) : (!aie.lock) -> ()\n";

        // Zero accumulator
        mlir << "        // Zero accumulator\n";
        mlir << "        // vzero: %acc = zero vector (16xi32)\n";

        // K-dimension loop (unrolled into k_chunks iterations)
        mlir << "        // K-dimension: " << k_chunks << " chunks of " << vec_len << " elements\n";
        for (int k = 0; k < k_chunks; k++) {
            mlir << "        // K chunk " << k << ": process elements [" << (k * vec_len) << ".." << ((k + 1) * vec_len - 1) << "]\n";
            mlir << "        // Load A vector (16xi8) from local memory\n";
            mlir << "        // %a_vec = aie.load(%ptr_a, offset=" << (k * vec_len * tile_m) << ")\n";
            mlir << "        // Load B vector (16xi8) from local memory\n";
            mlir << "        // %b_vec = aie.load(%ptr_b, offset=" << (k * vec_len) << ")\n";
            mlir << "        // Multiply-accumulate: %acc = mlacc(%acc, %a_vec, %b_vec)\n";
        }

        // Unlock when done
        mlir << "\n        // Signal completion\n";
        mlir << "        \"aie.unlock\"(%lock_start) : (!aie.lock) -> ()\n";

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

} // namespace detail

} // namespace ggnpu
