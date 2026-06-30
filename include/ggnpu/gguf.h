#ifndef GGNPU_GGUF_H
#define GGNPU_GGUF_H

#include <string>
#include <vector>
#include <map>
#include <span>
#include <cstdint>
#include <memory>
#include "tensor.h"

namespace ggnpu {

enum class GgufType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

struct GgufHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
};

struct GgufKV {
    std::string key;
    GgufType value_type = GgufType::UINT8;
    std::vector<uint8_t> data;
    std::string string_value;
    int64_t int_value = 0;
    double float_value = 0.0;
};

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    GgmlType type = GgmlType::F32;
    int32_t n_dims = 0;
    size_t data_offset = 0;
    size_t data_size = 0;
    size_t name_offset = 0;
};

class GgufLoader {
public:
    GgufLoader();
    ~GgufLoader();

    bool load(const std::string& path);
    void unload();

    const GgufHeader& header() const { return header_; }
    const std::map<std::string, GgufKV>& kv_pairs() const { return kv_pairs_; }
    const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }
    // Zero-copy view over the tensor-data region of the mmap'd file. Valid
    // until unload(); never owns or copies the ~GB of weights.
    std::span<const uint8_t> tensor_data() const {
        return {tensor_data_ptr_, tensor_data_size_};
    }

    std::string arch() const;
    std::string architecture() const;
    uint64_t context_length() const;
    uint64_t embedding_length() const;
    uint64_t block_count() const;
    uint64_t feed_forward_length() const;
    uint64_t attention_head_count() const;
    uint64_t attention_head_count_kv() const;
    double attention_layer_norm_rms_epsilon() const;
    uint64_t rope_dimension_count() const;
    double rope_freq_scale() const;
    uint64_t rope_freq_base() const;
    uint64_t tensor_data_offset() const;
    uint64_t general_alignment() const;

    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    int64_t get_int(const std::string& key, int64_t default_val = 0) const;
    double get_float(const std::string& key, double default_val = 0.0) const;

    // Build an architecture-prefixed metadata key, e.g. arch_key("block_count")
    // -> "qwen2.block_count". GGUF namespaces hparams under general.architecture,
    // so accessors resolve per-arch instead of being hardcoded to "llama.".
    std::string arch_key(const std::string& suffix) const;

private:
    bool parse_header();
    bool parse_kv();
    bool parse_tensors();
    bool map_tensor_data();

    // Buffered metadata reader. The GGUF header/KV/tensor-info section is parsed
    // with thousands of tiny fields (e.g. a 128k-entry tokenizer array), so one
    // ::read syscall per field cost ~16 s on load. These batch fd_ into a
    // userspace buffer; read_pos_ tracks the logical offset for header_end.
    ssize_t buffered_read(void* dst, size_t n);
    bool read_string_buffered(std::string& out);

    GgufHeader header_;
    std::map<std::string, GgufKV> kv_pairs_;
    std::vector<GgufTensorInfo> tensors_;
    const uint8_t* tensor_data_ptr_ = nullptr;  // view into mapped_data_, no copy
    size_t tensor_data_size_ = 0;
    uint8_t* mapped_data_ = nullptr;
    int fd_ = -1;
    size_t file_size_ = 0;
    size_t header_end_offset_ = 0; // file offset right after tensor metadata
    std::string path_;

    std::vector<uint8_t> read_buf_;   // userspace read buffer for metadata
    size_t read_buf_pos_ = 0;         // consumed bytes within read_buf_
    size_t read_buf_len_ = 0;         // valid bytes in read_buf_
    size_t read_pos_ = 0;             // logical file offset of next byte
};

} // namespace ggnpu

#endif // GGNPU_GGUF_H
