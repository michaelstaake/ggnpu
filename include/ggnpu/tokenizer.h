#ifndef GGNPU_TOKENIZER_H
#define GGNPU_TOKENIZER_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <cstdint>
#include "gguf.h"

namespace ggnpu {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer() = default;

    bool load_from_gguf(const std::map<std::string, GgufKV>& kv_pairs);

    std::vector<int> encode(const std::string& text, bool add_bos = true, bool add_eos = true) const;
    std::string decode(int token_id) const;
    std::string decode(const std::vector<int>& tokens) const;

    int bos_token_id() const { return bos_token_id_; }
    int eos_token_id() const { return eos_token_id_; }
    int vocab_size() const { return static_cast<int>(vocab_.size()); }

private:
    std::vector<std::pair<std::string, std::string>> bpe_merge_rules_;
    std::map<std::string, int> vocab_;
    std::map<int, std::string> reverse_vocab_;
    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
    bool add_bos_ = true;
    bool add_eos_ = true;

    std::vector<int> bpe_tokenize(const std::string& text) const;
    std::vector<std::string> split_to_bytes(const std::string& text) const;
    std::vector<std::pair<std::string, std::string>> parse_merges(const std::vector<uint8_t>& data, size_t data_len) const;
    std::string bytes_to_unicode_str() const;
};

} // namespace ggnpu

#endif // GGNPU_TOKENIZER_H
