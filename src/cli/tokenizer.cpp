#include "tokenizer.h"
#include <sstream>
#include <algorithm>
#include <set>
#include <cmath>
#include <cstring>

namespace ggnpu {

namespace {

std::string bytes_to_unicode() {
    std::string result;
    std::set<uint8_t> bytes_set;
    for (uint8_t b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || b > 161) {
            bytes_set.insert(b);
        }
    }
    for (uint8_t b = 0; b < 256; b++) {
        if (bytes_set.find(b) == bytes_set.end()) {
            result += static_cast<char>(b);
        }
    }
    return result;
}

} // namespace

Tokenizer::Tokenizer() {}

bool Tokenizer::load_from_gguf(const std::map<std::string, GgufKV>& kv_pairs) {
    vocab_.clear();
    bpe_merge_rules_.clear();

    auto get_string = [&](const std::string& key, const std::string& def = "") -> std::string {
        auto it = kv_pairs.find(key);
        if (it != kv_pairs.end()) return it->second.string_value;
        return def;
    };

    auto get_int = [&](const std::string& key, int64_t def = 0) -> int64_t {
        auto it = kv_pairs.find(key);
        if (it != kv_pairs.end()) return it->second.int_value;
        return def;
    };

    bos_token_id_ = static_cast<int>(get_int("tokenizer.ggml.bos_token_id", -1));
    eos_token_id_ = static_cast<int>(get_int("tokenizer.ggml.eos_token_id", -1));
    add_bos_ = get_int("tokenizer.ggml.add_bos_token", 1) != 0;
    add_eos_ = get_int("tokenizer.ggml.add_eos_token", 1) != 0;

    auto tokens_it = kv_pairs.find("tokenizer.ggml.tokens");
    if (tokens_it != kv_pairs.end() && tokens_it->second.data.size() >= 8) {
        uint32_t arr_type;
        std::memcpy(&arr_type, tokens_it->second.data.data(), 4);
        uint64_t arr_len;
        std::memcpy(&arr_len, tokens_it->second.data.data() + 4, 8);

        size_t offset = 12;
        for (uint64_t i = 0; i < arr_len && offset + 8 <= tokens_it->second.data.size(); i++) {
            uint64_t str_len;
            std::memcpy(&str_len, tokens_it->second.data.data() + offset, 8);
            offset += 8;

            if (offset + str_len > tokens_it->second.data.size()) break;

            std::string token(reinterpret_cast<const char*>(tokens_it->second.data.data() + offset), str_len);
            offset += str_len;

            int tid = static_cast<int>(vocab_.size());
            vocab_[token] = tid;
            reverse_vocab_[tid] = token;
        }
    }

    auto merges_it = kv_pairs.find("tokenizer.ggml.merges");
    if (merges_it != kv_pairs.end() && !merges_it->second.data.empty()) {
        bpe_merge_rules_ = parse_merges(merges_it->second.data, merges_it->second.data.size());
    }

    return !vocab_.empty();
}

std::vector<std::pair<std::string, std::string>> Tokenizer::parse_merges(
    const std::vector<uint8_t>& data, size_t data_len) const {

    std::vector<std::pair<std::string, std::string>> rules;
    if (data_len < 8) return rules;

    uint64_t arr_len;
    std::memcpy(&arr_len, data.data(), 8);

    size_t offset = 12;
    for (uint64_t i = 0; i < arr_len && offset + 8 <= data_len; i++) {
        uint64_t str_len;
        std::memcpy(&str_len, data.data() + offset, 8);
        offset += 8;

        if (offset + str_len > data_len) break;
        std::string s1(reinterpret_cast<const char*>(data.data() + offset), str_len);
        offset += str_len;

        if (offset + 8 > data_len) break;
        std::memcpy(&str_len, data.data() + offset, 8);
        offset += 8;

        if (offset + str_len > data_len) break;
        std::string s2(reinterpret_cast<const char*>(data.data() + offset), str_len);
        offset += str_len;

        rules.emplace_back(s1, s2);
    }

    return rules;
}

std::vector<std::string> Tokenizer::split_to_bytes(const std::string& text) const {
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) {
        tokens.push_back(std::string(1, static_cast<char>(c)));
    }
    return tokens;
}

std::vector<int> Tokenizer::bpe_tokenize(const std::string& text) const {
    if (text.empty()) return {};

    std::vector<std::string> tokens = split_to_bytes(text);

    if (bpe_merge_rules_.empty()) {
        std::vector<int> result;
        result.reserve(tokens.size());
        for (auto& t : tokens) {
            auto it = vocab_.find(t);
            if (it != vocab_.end()) {
                result.push_back(it->second);
            }
        }
        return result;
    }

    for (auto& [w1, w2] : bpe_merge_rules_) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 0; i + 1 < tokens.size(); i++) {
                if (tokens[i] == w1 && tokens[i + 1] == w2) {
                    std::string merged = w1 + w2;
                    if (vocab_.find(merged) != vocab_.end()) {
                        tokens[i] = merged;
                        tokens.erase(tokens.begin() + static_cast<int>(i + 1));
                        changed = true;
                    }
                }
            }
        }
    }

    std::vector<int> result;
    result.reserve(tokens.size());
    for (auto& t : tokens) {
        auto it = vocab_.find(t);
        if (it != vocab_.end()) {
            result.push_back(it->second);
        }
    }

    return result;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<int> tokens = bpe_tokenize(text);

    if (add_bos && bos_token_id_ >= 0) {
        tokens.insert(tokens.begin(), bos_token_id_);
    }
    if (add_eos && eos_token_id_ >= 0) {
        tokens.push_back(eos_token_id_);
    }

    return tokens;
}

std::string Tokenizer::decode(int token_id) const {
    auto it = reverse_vocab_.find(token_id);
    if (it != reverse_vocab_.end()) return it->second;
    return "[UNK:" + std::to_string(token_id) + "]";
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    for (int tid : tokens) {
        result += decode(tid);
    }
    return result;
}

} // namespace ggnpu
