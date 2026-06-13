#ifndef GGNPU_WEIGHT_CACHE_H
#define GGNPU_WEIGHT_CACHE_H

#include "tensor.h"
#include "quant/quant.h"
#include "cache.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <cstdint>
#include <functional>
#include <iostream>

namespace ggnpu {

// Cached decoded weight: INT8 values + scales for NPU consumption
struct DecodedWeight {
    std::vector<int8_t> int8_data;
    std::vector<float> scales;
    size_t data_size = 0;
    GgmlType type = GgmlType::F32;
};

// WeightCache: bridges GGUF tensors → decoded INT8 weights → NPU buffers
//
// Usage:
//   1. Create WeightCache with a CompileCache instance
//   2. Call get_or_decode(tensor_name, gguf_data, data_size, ggml_type)
//   3. Returns pointer to int8_data ready for NPU consumption
//
// Cache key: (model_path_hash, tensor_name)
// On cache miss: dispatches to correct decode function, stores result
class WeightCache {
public:
    explicit WeightCache(CompileCache& compile_cache)
        : cache_(compile_cache) {}

    // Get decoded INT8 weights for a tensor.
    // On cache miss: decodes from GGUF format and stores.
    // Returns pointer to int8_data, or nullptr on failure.
    const int8_t* get_or_decode(const std::string& tensor_name,
                                const uint8_t* gguf_data,
                                size_t data_size,
                                GgmlType type,
                                int64_t n_rows = 0,
                                int64_t n_cols = 0) {
        std::string key = make_key(tensor_name, type, data_size);

        std::lock_guard<std::mutex> lock(mutex_);

        auto mem_it = memory_cache_.find(key);
        if (mem_it != memory_cache_.end()) {
            return mem_it->second.int8_data.data();
        }

        // Check persistent cache
        std::vector<int8_t> cached_int8;
        std::vector<float> cached_scales;
        if (cache_.get_weights(key, cached_int8, cached_scales) &&
            static_cast<size_t>(cached_int8.size()) > 0) {
            memory_cache_[key] = {std::move(cached_int8), std::move(cached_scales), data_size, type};
            return memory_cache_[key].int8_data.data();
        }

        // Cache miss: decode from GGUF format
        std::vector<int8_t> int8_output;
        std::vector<float> scales_output;

        try {
            decode_for_npu(type, gguf_data, data_size, n_rows, n_cols,
                           int8_output, scales_output);
        } catch (const std::exception& e) {
            std::cerr << "Error: failed to decode tensor " << tensor_name
                      << " (" << ggml_type_name(type) << "): " << e.what() << "\n";
            return nullptr;
        }

        // Store in memory cache
        DecodedWeight dw;
        dw.int8_data = std::move(int8_output);
        dw.scales = std::move(scales_output);
        dw.data_size = data_size;
        dw.type = type;
        memory_cache_[key] = std::move(dw);

        // Store in persistent cache (non-blocking, fire-and-forget)
        cache_.store_weights(key, memory_cache_[key].int8_data, memory_cache_[key].scales);

        return memory_cache_[key].int8_data.data();
    }

    // Get the scales for a tensor (needed for some NPU kernels)
    const int8_t* get_or_decode(const TensorView& tv) {
        int64_t n_rows = 0;
        int64_t n_cols = 0;
        if (tv.n_dims >= 2) {
            n_cols = static_cast<int64_t>(tv.dims[0]);
            n_rows = static_cast<int64_t>(tv.dims[1]);
        } else if (tv.n_dims == 1) {
            n_cols = static_cast<int64_t>(tv.dims[0]);
            n_rows = 1;
        }
        return get_or_decode(tv.name, tv.data, tv.data_size(), tv.type, n_rows, n_cols);
    }

    const std::vector<float>& get_scales(const std::string& tensor_name, GgmlType type,
                                         size_t data_size = 0) {
        std::string key = make_key(tensor_name, type, data_size);
        auto it = memory_cache_.find(key);
        if (it != memory_cache_.end()) {
            return it->second.scales;
        }
        static std::vector<float> empty;
        return empty;
    }

    const std::vector<float>& get_scales(const TensorView& tv) {
        return get_scales(tv.name, tv.type, tv.data_size());
    }

    // Clear all cached weights
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        memory_cache_.clear();
    }

    // Get cache hit statistics
    size_t cache_size() const {
        return memory_cache_.size();
    }

private:
    std::string make_key(const std::string& tensor_name, GgmlType type, size_t data_size = 0) {
        std::hash<std::string> hasher;
        size_t hash = hasher(tensor_name);
        const char* ver = (type == GgmlType::Q4_K || type == GgmlType::Q6_K) ? "w3_" : "w_";
        return std::string(ver) + std::to_string(hash) + "_t" + std::to_string(static_cast<int>(type))
               + "_" + std::to_string(data_size);
    }

    CompileCache& cache_;
    std::map<std::string, DecodedWeight> memory_cache_;
    mutable std::mutex mutex_;
};

} // namespace ggnpu

#endif // GGNPU_WEIGHT_CACHE_H
