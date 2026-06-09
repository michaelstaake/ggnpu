#include "cache.h"
#include <fstream>
#include <filesystem>

namespace ggnpu {

CompileCache::CompileCache(const std::string& cache_dir, bool enabled)
    : cache_dir_(cache_dir), enabled_(enabled) {
    if (enabled_) {
        std::filesystem::create_directories(cache_dir_);
        std::filesystem::create_directories(cache_dir_ + "/xclbin");
        std::filesystem::create_directories(cache_dir_ + "/weights");
    }
}

bool CompileCache::get_weights(const std::string& key, std::vector<int8_t>& out_int8,
                               std::vector<float>& out_scales) {
    if (!enabled_) return false;

    std::string weight_path = cache_dir_ + "/weights/" + key;
    std::string meta_path = weight_path + ".meta";

    std::ifstream meta(meta_path, std::ios::binary);
    if (!meta.is_open()) return false;

    uint64_t int8_size, scales_size;
    meta.read(reinterpret_cast<char*>(&int8_size), 8);
    meta.read(reinterpret_cast<char*>(&scales_size), 8);

    out_int8.resize(int8_size);
    out_scales.resize(scales_size / sizeof(float));

    std::ifstream weights(weight_path, std::ios::binary);
    if (!weights.is_open()) return false;

    weights.read(reinterpret_cast<char*>(out_int8.data()), int8_size);
    weights.read(reinterpret_cast<char*>(out_scales.data()), scales_size);

    return true;
}

void CompileCache::store_weights(const std::string& key, const std::vector<int8_t>& int8,
                                 const std::vector<float>& scales) {
    if (!enabled_) return;

    std::string weight_path = cache_dir_ + "/weights/" + key;
    std::string meta_path = weight_path + ".meta";

    std::ofstream meta(meta_path, std::ios::binary);
    uint64_t int8_size = int8.size();
    uint64_t scales_size = scales.size() * sizeof(float);
    meta.write(reinterpret_cast<char*>(&int8_size), 8);
    meta.write(reinterpret_cast<char*>(&scales_size), 8);

    std::ofstream weights(weight_path, std::ios::binary);
    weights.write(reinterpret_cast<const char*>(int8.data()), int8_size);
    weights.write(reinterpret_cast<const char*>(scales.data()), scales_size);
}

std::string CompileCache::get_xclbin_path(const std::string& key) {
    if (!enabled_) return "";
    return cache_dir_ + "/xclbin/" + key + ".xclbin";
}

void CompileCache::store_xclbin(const std::string& key, const std::vector<uint8_t>& data) {
    if (!enabled_) return;
    std::string path = get_xclbin_path(key);

    std::ofstream f(path, std::ios::binary);
    if (f.is_open()) {
        f.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
}

bool CompileCache::has_xclbin(const std::string& key) {
    if (!enabled_) return false;
    return std::filesystem::exists(get_xclbin_path(key));
}

} // namespace ggnpu
