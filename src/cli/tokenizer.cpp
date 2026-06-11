#include "tokenizer.h"
#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <queue>
#include <utility>

namespace ggnpu {

namespace {

uint64_t read_u64_le(const uint8_t* p) {
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

std::vector<std::string> parse_string_array(const std::vector<uint8_t>& data) {
    std::vector<std::string> out;
    if (data.size() < 12) return out;

    uint64_t arr_len = read_u64_le(data.data() + 4);
    size_t offset = 12;
    for (uint64_t i = 0; i < arr_len && offset + 8 <= data.size(); i++) {
        uint64_t str_len = read_u64_le(data.data() + offset);
        offset += 8;
        if (offset + str_len > data.size()) break;
        out.emplace_back(reinterpret_cast<const char*>(data.data() + offset), str_len);
        offset += str_len;
    }
    return out;
}

struct BpeSymbol {
    int prev = -1;
    int next = -1;
    const char* text = nullptr;
    size_t n = 0;
};

struct BpeBigram {
    int left = -1;
    int right = -1;
    std::string text;
    int rank = -1;
    size_t size = 0;

    bool operator>(const BpeBigram& other) const {
        return rank > other.rank || (rank == other.rank && left > other.left);
    }
};

} // namespace

Tokenizer::Tokenizer() {}

bool Tokenizer::load_from_gguf(const std::map<std::string, GgufKV>& kv_pairs) {
    vocab_.clear();
    reverse_vocab_.clear();
    bpe_ranks_.clear();
    pre_type_ = PreType::Default;

    auto get_int = [&](const std::string& key, int64_t def = 0) -> int64_t {
        auto it = kv_pairs.find(key);
        if (it != kv_pairs.end()) return it->second.int_value;
        return def;
    };

    bos_token_id_ = static_cast<int>(get_int("tokenizer.ggml.bos_token_id", -1));
    eos_token_id_ = static_cast<int>(get_int("tokenizer.ggml.eos_token_id", -1));
    add_bos_ = get_int("tokenizer.ggml.add_bos_token", 1) != 0;
    add_eos_ = get_int("tokenizer.ggml.add_eos_token", 1) != 0;

    auto pre_it = kv_pairs.find("tokenizer.ggml.pre");
    if (pre_it != kv_pairs.end()) {
        const std::string& pre = pre_it->second.string_value;
        if (pre == "llama-bpe" || pre == "llama3" || pre == "llama-v3") {
            pre_type_ = PreType::Llama3;
        } else if (pre == "gpt-2" || pre == "default") {
            pre_type_ = PreType::Gpt2;
        }
    }

    auto tokens_it = kv_pairs.find("tokenizer.ggml.tokens");
    if (tokens_it != kv_pairs.end()) {
        for (const auto& token : parse_string_array(tokens_it->second.data)) {
            int tid = static_cast<int>(vocab_.size());
            vocab_[token] = tid;
            reverse_vocab_[tid] = token;
        }
    }

    auto merges_it = kv_pairs.find("tokenizer.ggml.merges");
    if (merges_it != kv_pairs.end()) {
        auto merges = parse_string_array(merges_it->second.data);
        for (size_t i = 0; i < merges.size(); i++) {
            const std::string& merge = merges[i];
            size_t space = merge.find(' ');
            if (space == std::string::npos) continue;
            std::string left = merge.substr(0, space);
            std::string right = merge.substr(space + 1);
            bpe_ranks_[{left, right}] = static_cast<int>(i);
        }
    }

    return !vocab_.empty();
}

int Tokenizer::find_bpe_rank(const std::string& left, const std::string& right) const {
    auto it = bpe_ranks_.find({left, right});
    if (it == bpe_ranks_.end()) return -1;
    return it->second;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    std::vector<std::string> regex_exprs;
    switch (pre_type_) {
        case PreType::Llama3:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case PreType::Gpt2:
            regex_exprs = {
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            };
            break;
        default:
            return {text};
    }
    return unicode_regex_split(text, regex_exprs, true);
}

std::vector<int> Tokenizer::bpe_tokenize_word(const std::string& word) const {
    std::vector<int> output;
    if (word.empty()) return output;

    std::vector<BpeSymbol> symbols;
    std::priority_queue<BpeBigram, std::vector<BpeBigram>, std::greater<BpeBigram>> work_queue;

    auto add_new_bigram = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        std::string left_token(symbols[left].text, symbols[left].n);
        std::string right_token(symbols[right].text, symbols[right].n);
        int rank = find_bpe_rank(left_token, right_token);
        if (rank < 0) return;
        work_queue.push({left, right, left_token + right_token, rank,
                         left_token.size() + right_token.size()});
    };

    int index = 0;
    size_t offset = 0;
    while (offset < word.size()) {
        BpeSymbol sym;
        size_t char_len = std::min(word.size() - offset, unicode_len_utf8(word[offset]));
        sym.text = word.c_str() + offset;
        sym.n = char_len;
        offset += sym.n;
        sym.prev = index - 1;
        sym.next = (offset == word.size()) ? -1 : index + 1;
        index++;
        symbols.push_back(sym);
    }

    for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
        add_new_bigram(i - 1, i);
    }

    while (!work_queue.empty()) {
        BpeBigram bigram = work_queue.top();
        work_queue.pop();

        auto& left_symbol = symbols[bigram.left];
        auto& right_symbol = symbols[bigram.right];
        if (left_symbol.n == 0 || right_symbol.n == 0) continue;

        std::string left_token(left_symbol.text, left_symbol.n);
        std::string right_token(right_symbol.text, right_symbol.n);
        if (left_token + right_token != bigram.text) continue;

        left_symbol.n += right_symbol.n;
        right_symbol.n = 0;
        left_symbol.next = right_symbol.next;
        if (right_symbol.next >= 0) {
            symbols[right_symbol.next].prev = bigram.left;
        }

        add_new_bigram(left_symbol.prev, bigram.left);
        add_new_bigram(bigram.left, left_symbol.next);
    }

    for (int i = 0; i != -1; i = symbols[i].next) {
        if (symbols[i].n == 0) continue;
        std::string str(symbols[i].text, symbols[i].n);
        auto it = vocab_.find(str);
        if (it != vocab_.end()) {
            output.push_back(it->second);
        } else {
            for (char c : str) {
                auto bit = vocab_.find(std::string(1, c));
                if (bit != vocab_.end()) output.push_back(bit->second);
            }
        }
    }

    return output;
}

std::vector<int> Tokenizer::bpe_tokenize(const std::string& text) const {
    std::vector<int> result;
    if (text.empty()) return result;

    for (const auto& word : pretokenize(text)) {
        auto word_tokens = bpe_tokenize_word(word);
        result.insert(result.end(), word_tokens.begin(), word_tokens.end());
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

std::string Tokenizer::decode_piece(const std::string& piece) const {
    std::string result;
    size_t offset = 0;
    while (offset < piece.size()) {
        bool matched = false;
        for (size_t len = std::min<size_t>(4, piece.size() - offset); len > 0; len--) {
            std::string sub = piece.substr(offset, len);
            try {
                result += static_cast<char>(unicode_utf8_to_byte(sub));
                offset += len;
                matched = true;
                break;
            } catch (const std::exception&) {
                continue;
            }
        }
        if (!matched) {
            result += piece[offset++];
        }
    }
    return result;
}

std::string Tokenizer::decode(int token_id) const {
    auto it = reverse_vocab_.find(token_id);
    if (it != reverse_vocab_.end()) return decode_piece(it->second);
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
