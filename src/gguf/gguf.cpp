#include "gguf.h"
#include <fstream>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace ggnpu {

namespace {

constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
constexpr uint32_t GGUF_VERSION = 3;

uint64_t read_u64_le(const uint8_t* p) {
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

int64_t read_i64_le(const uint8_t* p) {
    return static_cast<int64_t>(read_u64_le(p));
}

uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

float read_f32_le(const uint8_t* p) {
    uint32_t v = read_u32_le(p);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

double read_f64_le(const uint8_t* p) {
    uint64_t v = read_u64_le(p);
    double d;
    std::memcpy(&d, &v, 8);
    return d;
}

void write_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 16) & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 24) & 0xff));
}

void write_u64_le(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    }
}

void append_gguf_string(std::vector<uint8_t>& buf, const std::string& s) {
    write_u64_le(buf, s.size());
    buf.insert(buf.end(), s.begin(), s.end());
}

} // namespace

GgufLoader::GgufLoader() : fd_(-1) {
    std::memset(&header_, 0, sizeof(header_));
}

GgufLoader::~GgufLoader() { unload(); }

ssize_t GgufLoader::buffered_read(void* dst, size_t n) {
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t got = 0;
    constexpr size_t kChunk = 1u << 20; // 1 MB
    while (got < n) {
        if (read_buf_pos_ >= read_buf_len_) {
            if (read_buf_.size() < kChunk) read_buf_.resize(kChunk);
            ssize_t r = ::read(fd_, read_buf_.data(), read_buf_.size());
            if (r <= 0) break;
            read_buf_len_ = static_cast<size_t>(r);
            read_buf_pos_ = 0;
        }
        size_t take = std::min(read_buf_len_ - read_buf_pos_, n - got);
        std::memcpy(out + got, read_buf_.data() + read_buf_pos_, take);
        read_buf_pos_ += take;
        got += take;
    }
    read_pos_ += got;
    return static_cast<ssize_t>(got);
}

bool GgufLoader::read_string_buffered(std::string& out) {
    uint8_t len_buf[8];
    if (buffered_read(len_buf, 8) != 8) return false;
    uint64_t str_len = read_u64_le(len_buf);
    if (str_len > 64 * 1024 * 1024) return false;
    out.resize(str_len);
    if (str_len > 0) {
        if (buffered_read(&out[0], str_len) != static_cast<ssize_t>(str_len)) return false;
    }
    return true;
}

bool GgufLoader::load(const std::string& path) {
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) < 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    file_size_ = static_cast<size_t>(st.st_size);

    if (!parse_header()) return false;
    if (!parse_kv()) return false;
    if (!parse_tensors()) return false;
    if (!map_tensor_data()) return false;

    return true;
}

void GgufLoader::unload() {
    if (mapped_data_) {
        munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    tensor_data_ptr_ = nullptr;
    tensor_data_size_ = 0;
    kv_pairs_.clear();
    tensors_.clear();
    std::memset(&header_, 0, sizeof(header_));
}

bool GgufLoader::parse_header() {
    uint8_t buf[24];
    ssize_t nread = buffered_read(buf, 24);
    if (nread != 24) return false;

    uint32_t magic;
    std::memcpy(&magic, buf, 4);
    if (magic != GGUF_MAGIC) return false;
    header_.magic = magic;

    uint32_t version;
    std::memcpy(&version, buf + 4, 4);
    header_.version = version;
    if (version != GGUF_VERSION) return false;

    std::memcpy(&header_.tensor_count, buf + 8, 8);
    std::memcpy(&header_.kv_count, buf + 16, 8);

    return true;
}

bool GgufLoader::parse_kv() {
    for (uint64_t i = 0; i < header_.kv_count; i++) {
        // Read key length
        uint8_t key_len_buf[8];
        if (buffered_read(key_len_buf, 8) != 8) return false;
        uint64_t key_len = read_u64_le(key_len_buf);
        if (key_len > 1024) return false;

        // Read key
        std::string key(key_len, '\0');
        if (buffered_read(&key[0], key_len) != static_cast<ssize_t>(key_len)) return false;

        // Read value type
        uint8_t type_buf[4];
        if (buffered_read(type_buf, 4) != 4) return false;
        uint32_t type_val = read_u32_le(type_buf);
        GgufType value_type = static_cast<GgufType>(type_val);

        GgufKV kv;
        kv.key = key;
        kv.value_type = value_type;

        switch (value_type) {
            case GgufType::STRING: {
                uint8_t str_key_len_buf[8];
                if (buffered_read(str_key_len_buf, 8) != 8) return false;
                uint64_t str_len = read_u64_le(str_key_len_buf);
                if (str_len > 64 * 1024 * 1024) return false;
                kv.string_value.resize(str_len);
                if (buffered_read(&kv.string_value[0], str_len) != static_cast<ssize_t>(str_len)) return false;
                break;
            }
            case GgufType::ARRAY: {
                // ARRAY has no data_len field - directly read array_type and array_length
                uint8_t arr_type_buf[4];
                if (buffered_read(arr_type_buf, 4) != 4) return false;
                uint32_t arr_type_val = read_u32_le(arr_type_buf);
                GgufType arr_type = static_cast<GgufType>(arr_type_val);

                uint8_t arr_count_buf[8];
                if (buffered_read(arr_count_buf, 8) != 8) return false;
                uint64_t arr_count = read_u64_le(arr_count_buf);

                size_t elem_size = 4;
                switch (arr_type) {
                    case GgufType::UINT8:  elem_size = 1; break;
                    case GgufType::INT8:   elem_size = 1; break;
                    case GgufType::UINT16: elem_size = 2; break;
                    case GgufType::INT16:  elem_size = 2; break;
                    case GgufType::UINT32: elem_size = 4; break;
                    case GgufType::INT32:  elem_size = 4; break;
                    case GgufType::FLOAT32: elem_size = 4; break;
                    case GgufType::BOOL:   elem_size = 1; break;
                    case GgufType::UINT64: elem_size = 8; break;
                    case GgufType::INT64:  elem_size = 8; break;
                    case GgufType::FLOAT64: elem_size = 8; break;
                    default: elem_size = 4; break;
                }

                if (arr_type == GgufType::STRING) {
                    write_u32_le(kv.data, static_cast<uint32_t>(arr_type));
                    write_u64_le(kv.data, arr_count);
                    for (uint64_t j = 0; j < arr_count; j++) {
                        std::string elem;
                        if (!read_string_buffered(elem)) return false;
                        append_gguf_string(kv.data, elem);
                    }
                } else {
                    uint64_t total_bytes = arr_count * elem_size;
                    kv.data.resize(total_bytes);
                    if (total_bytes > 0) {
                        if (buffered_read(kv.data.data(), total_bytes) != static_cast<ssize_t>(total_bytes)) return false;
                    }
                }
                break;
            }
            default: {
                // For scalar types: val_type is followed directly by the value
                // No data_length field for scalar types in this GGUF version
                switch (value_type) {
                    case GgufType::UINT8:
                    case GgufType::INT8:
                    case GgufType::BOOL:
                        {
                            uint8_t v;
                            if (buffered_read(&v, 1) != 1) return false;
                            kv.int_value = v;
                        }
                        break;
                    case GgufType::UINT16:
                    case GgufType::INT16:
                        {
                            uint8_t buf[2];
                            if (buffered_read(buf, 2) != 2) return false;
                            kv.int_value = static_cast<int64_t>(buf[0] | (buf[1] << 8));
                        }
                        break;
                    case GgufType::UINT32:
                    case GgufType::INT32:
                    case GgufType::FLOAT32:
                        {
                            uint8_t buf[4];
                            if (buffered_read(buf, 4) != 4) return false;
                            if (value_type == GgufType::FLOAT32) {
                                kv.float_value = static_cast<double>(read_f32_le(buf));
                            } else {
                                kv.int_value = static_cast<int64_t>(static_cast<int32_t>(read_u32_le(buf)));
                            }
                        }
                        break;
                    case GgufType::UINT64:
                    case GgufType::INT64:
                    case GgufType::FLOAT64:
                        {
                            uint8_t buf[8];
                            if (buffered_read(buf, 8) != 8) return false;
                            if (value_type == GgufType::FLOAT64) {
                                kv.float_value = read_f64_le(buf);
                            } else {
                                kv.int_value = read_i64_le(buf);
                            }
                        }
                        break;
                    default:
                        return false;
                }
                break;
            }
        }

        kv_pairs_[key] = std::move(kv);
    }

    return true;
}

bool GgufLoader::parse_tensors() {
    tensors_.clear();
    tensors_.reserve(header_.tensor_count);

    for (uint64_t i = 0; i < header_.tensor_count; i++) {
        GgufTensorInfo info;

        // Read name length
        uint8_t name_key_len_buf[8];
        if (buffered_read(name_key_len_buf, 8) != 8) {
            return false;
        }
        uint64_t name_len = read_u64_le(name_key_len_buf);
        if (name_len > 1024) return false;

        // Read name
        std::string name(name_len, '\0');
        if (buffered_read(&name[0], name_len) != static_cast<ssize_t>(name_len)) return false;
        info.name = name;

        // Read number of dimensions
        uint8_t ndim_buf[4];
        if (buffered_read(ndim_buf, 4) != 4) return false;
        uint32_t ndim = read_u32_le(ndim_buf);
        info.n_dims = static_cast<int32_t>(ndim);

        // Read dimensions
        info.dims.resize(ndim);
        for (uint32_t j = 0; j < ndim; j++) {
            uint8_t dim_buf[8];
            if (buffered_read(dim_buf, 8) != 8) return false;
            info.dims[j] = read_u64_le(dim_buf);
        }

        // Read type
        uint8_t type_buf[4];
        if (buffered_read(type_buf, 4) != 4) return false;
        uint32_t type_val = read_u32_le(type_buf);
        info.type = static_cast<GgmlType>(type_val);

        // Read data offset (8 bytes)
        uint8_t data_off_buf[8];
        if (buffered_read(data_off_buf, 8) != 8) return false;
        info.data_offset = read_u64_le(data_off_buf);

        // Calculate tensor data size
        size_t elem_count = 1;
        for (auto d : info.dims) elem_count *= d;
        info.data_size = (elem_count / ggml_blck_size(info.type)) * ggml_type_size(info.type);

        tensors_.push_back(std::move(info));
    }

    // Logical bytes consumed by buffered_read == file offset where tensor data
    // begins (the fd's real position is ahead by the buffered remainder).
    header_end_offset_ = read_pos_;

    return true;
}

bool GgufLoader::map_tensor_data() {
    // Memory map the entire file
    mapped_data_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (mapped_data_ == MAP_FAILED) {
        mapped_data_ = nullptr;
        return false;
    }

    // Get tensor data offset from metadata
    uint64_t data_offset = tensor_data_offset();
    if (data_offset == 0 || data_offset >= file_size_) {
        // Data section starts right after the tensor metadata, padded to
        // general.alignment (default 32)
        uint64_t align = 32;
        auto it = kv_pairs_.find("general.alignment");
        if (it != kv_pairs_.end() && it->second.int_value > 0) align = static_cast<uint64_t>(it->second.int_value);
        data_offset = (header_end_offset_ + align - 1) / align * align;
    }

    // Tensor offsets in the header are relative to the data section start
    // (already alignment-padded by the writer) — keep them relative, since
    // consumers index from tensor_data().data().
    //
    // Point a view directly into the mmap rather than copying the ~GB of
    // tensor data onto the heap. The previous std::vector::assign here forced
    // every page through a memcpy on load (~16 s for Llama 1B), defeating the
    // whole purpose of the mmap. Pages now fault in lazily on first access.
    tensor_data_ptr_ = mapped_data_ + static_cast<size_t>(data_offset);
    tensor_data_size_ = file_size_ - static_cast<size_t>(data_offset);

    return true;
}

std::string GgufLoader::arch() const {
    return get_string("general.architecture", get_string("general.name", ""));
}

std::string GgufLoader::architecture() const {
    return arch();
}

uint64_t GgufLoader::context_length() const {
    return static_cast<uint64_t>(get_int("llama.context_length", 0));
}

uint64_t GgufLoader::embedding_length() const {
    return static_cast<uint64_t>(get_int("llama.embedding_length", 0));
}

uint64_t GgufLoader::block_count() const {
    return static_cast<uint64_t>(get_int("llama.block_count", 0));
}

uint64_t GgufLoader::feed_forward_length() const {
    return static_cast<uint64_t>(get_int("llama.feed_forward_length", 0));
}

uint64_t GgufLoader::attention_head_count() const {
    return static_cast<uint64_t>(get_int("llama.attention.head_count", 0));
}

uint64_t GgufLoader::attention_head_count_kv() const {
    return static_cast<uint64_t>(get_int("llama.attention.head_count_kv", 0));
}

double GgufLoader::attention_layer_norm_rms_epsilon() const {
    return get_float("llama.attention.layer_norm_rms_epsilon", 1e-5);
}

uint64_t GgufLoader::rope_dimension_count() const {
    return static_cast<uint64_t>(get_int("llama.rope.dimension_count", 0));
}

double GgufLoader::rope_freq_scale() const {
    return get_float("llama.rope.freq_scale", 1.0);
}

uint64_t GgufLoader::rope_freq_base() const {
    return static_cast<uint64_t>(get_float("llama.rope.freq_base", 10000.0));
}

uint64_t GgufLoader::tensor_data_offset() const {
    auto it = kv_pairs_.find("general.data_offset");
    if (it != kv_pairs_.end() && it->second.data.size() >= 8) {
        return static_cast<uint64_t>(read_u64_le(it->second.data.data()));
    }
    if (header_end_offset_ == 0) return 0;
    uint64_t align = general_alignment();
    return (header_end_offset_ + align - 1) / align * align;
}

uint64_t GgufLoader::general_alignment() const {
    return static_cast<uint64_t>(get_float("general.alignment", 32.0));
}

std::string GgufLoader::get_string(const std::string& key, const std::string& default_val) const {
    auto it = kv_pairs_.find(key);
    if (it != kv_pairs_.end()) {
        return it->second.string_value;
    }
    return default_val;
}

int64_t GgufLoader::get_int(const std::string& key, int64_t default_val) const {
    auto it = kv_pairs_.find(key);
    if (it != kv_pairs_.end()) {
        if (it->second.value_type == GgufType::INT64 && it->second.data.size() >= 8) {
            return read_i64_le(it->second.data.data());
        }
        if (it->second.value_type == GgufType::UINT64 && it->second.data.size() >= 8) {
            return static_cast<int64_t>(read_u64_le(it->second.data.data()));
        }
        if (it->second.value_type == GgufType::FLOAT64 && it->second.data.size() >= 8) {
            return static_cast<int64_t>(read_f64_le(it->second.data.data()));
        }
        // Fall back to int_value if already parsed
        if (it->second.value_type != GgufType::STRING &&
            it->second.value_type != GgufType::ARRAY) {
            return it->second.int_value;
        }
    }
    return default_val;
}

double GgufLoader::get_float(const std::string& key, double default_val) const {
    auto it = kv_pairs_.find(key);
    if (it != kv_pairs_.end()) {
        if (it->second.value_type == GgufType::FLOAT64 && it->second.data.size() >= 8) {
            return read_f64_le(it->second.data.data());
        }
        if (it->second.value_type == GgufType::FLOAT32 && it->second.data.size() >= 4) {
            return static_cast<double>(read_f32_le(it->second.data.data()));
        }
        // Fall back to float_value if already parsed
        if (it->second.value_type != GgufType::STRING &&
            it->second.value_type != GgufType::ARRAY) {
            return it->second.float_value;
        }
    }
    return default_val;
}

} // namespace ggnpu
