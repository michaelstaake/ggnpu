#ifndef GGNPU_CACHE_H
#define GGNPU_CACHE_H

#include <string>
#include <vector>
#include <cstdint>

namespace ggnpu {

class CompileCache {
public:
    CompileCache(const std::string& cache_dir, bool enabled = true);
    ~CompileCache() = default;

    bool is_enabled() const { return enabled_; }

    bool get_weights(const std::string& key, std::vector<int8_t>& out_int8,
                     std::vector<float>& out_scales);

    void store_weights(const std::string& key, const std::vector<int8_t>& int8,
                       const std::vector<float>& scales);

    std::string get_xclbin_path(const std::string& key);
    void store_xclbin(const std::string& key, const std::vector<uint8_t>& data);
    bool has_xclbin(const std::string& key);

private:
    std::string cache_dir_;
    bool enabled_;
};

} // namespace ggnpu

#endif // GGNPU_CACHE_H
