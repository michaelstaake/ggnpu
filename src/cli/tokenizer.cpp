#include "tokenizer.h"
#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <queue>
#include <utility>

namespace ggnpu {

namespace {

// SentencePiece meta-space (U+2581), substituted for ' ' in unigram vocabs.
constexpr char kSpmSpace[] = "\xe2\x96\x81";

// GGUF token_type values (tokenizer.ggml.token_type).
enum TokenType { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3,
                 TT_USER_DEFINED = 4, TT_UNUSED = 5, TT_BYTE = 6 };

uint64_t read_u64_le(const uint8_t* p) {
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
           ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
           ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
           ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

// Decode a numeric metadata ARRAY straight from its GgufKV (tokenizer only
// receives the kv map, not a GgufLoader). Elements are raw little-endian.
std::vector<float> parse_f32_array(const GgufKV& kv) {
    std::vector<float> out;
    if (kv.array_type != GgufType::FLOAT32) return out;
    if (kv.data.size() < kv.array_length * 4) return out;
    out.resize(kv.array_length);
    for (uint64_t i = 0; i < kv.array_length; i++) {
        uint32_t v;
        std::memcpy(&v, kv.data.data() + i * 4, 4);
        std::memcpy(&out[i], &v, 4);
    }
    return out;
}

std::vector<int32_t> parse_i32_array(const GgufKV& kv) {
    std::vector<int32_t> out;
    if (kv.array_type != GgufType::INT32 && kv.array_type != GgufType::UINT32) return out;
    if (kv.data.size() < kv.array_length * 4) return out;
    out.resize(kv.array_length);
    for (uint64_t i = 0; i < kv.array_length; i++) {
        std::memcpy(&out[i], kv.data.data() + i * 4, 4);
    }
    return out;
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

// SentencePiece unigram merge candidate. Merged by *highest* score first;
// ties broken by lowest left index (matches llama.cpp llm_tokenizer_spm).
struct SpmBigram {
    int left = -1;
    int right = -1;
    std::string text;
    float score = 0.0f;
    size_t size = 0;

    // Higher priority = popped first from a std::priority_queue (max-heap).
    bool operator<(const SpmBigram& o) const {
        return score < o.score || (score == o.score && left > o.left);
    }
};

} // namespace

Tokenizer::Tokenizer() {}

bool Tokenizer::load_from_gguf(const std::map<std::string, GgufKV>& kv_pairs) {
    vocab_.clear();
    reverse_vocab_.clear();
    bpe_ranks_.clear();
    token_scores_.clear();
    token_types_.clear();
    pre_type_ = PreType::Default;
    model_type_ = ModelType::Bpe;

    auto get_int = [&](const std::string& key, int64_t def = 0) -> int64_t {
        auto it = kv_pairs.find(key);
        if (it != kv_pairs.end()) return it->second.int_value;
        return def;
    };

    bos_token_id_ = static_cast<int>(get_int("tokenizer.ggml.bos_token_id", -1));
    eos_token_id_ = static_cast<int>(get_int("tokenizer.ggml.eos_token_id", -1));
    unk_token_id_ = static_cast<int>(get_int("tokenizer.ggml.unknown_token_id", -1));
    add_bos_ = get_int("tokenizer.ggml.add_bos_token", 1) != 0;
    add_eos_ = get_int("tokenizer.ggml.add_eos_token", 1) != 0;
    // SentencePiece default is to prefix a space; gemma sets this false.
    add_space_prefix_ = get_int("tokenizer.ggml.add_space_prefix", 1) != 0;

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
        auto tokens = parse_string_array(tokens_it->second.data);
        for (size_t i = 0; i < tokens.size(); i++) {
            int tid = static_cast<int>(i);  // id = array position (handles dup strings)
            vocab_[tokens[i]] = tid;
            reverse_vocab_[tid] = tokens[i];
        }
    }

    auto scores_it = kv_pairs.find("tokenizer.ggml.scores");
    if (scores_it != kv_pairs.end()) token_scores_ = parse_f32_array(scores_it->second);
    auto ttype_it = kv_pairs.find("tokenizer.ggml.token_type");
    if (ttype_it != kv_pairs.end()) token_types_ = parse_i32_array(ttype_it->second);

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

    // Pick the tokenization algorithm. `tokenizer.ggml.model` is authoritative:
    // "gpt2" is byte-level BPE (Llama 3 / Qwen2). SentencePiece vocabs (gemma,
    // llama-spm, t5) instead ship per-token scores — note gemma also carries a
    // merges list, so merge presence alone is NOT a BPE signal. Rule: gpt2 ⇒ BPE;
    // otherwise unigram when scores are present; else fall back to BPE.
    std::string model_str;
    auto model_it = kv_pairs.find("tokenizer.ggml.model");
    if (model_it != kv_pairs.end()) model_str = model_it->second.string_value;
    if (model_str != "gpt2" && !token_scores_.empty()) {
        model_type_ = ModelType::Unigram;
    } else {
        model_type_ = ModelType::Bpe;
    }

    return !vocab_.empty();
}

int Tokenizer::piece_to_id(const std::string& piece) const {
    auto it = vocab_.find(piece);
    return it == vocab_.end() ? -1 : it->second;
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

std::vector<int> Tokenizer::spm_tokenize(const std::string& text) const {
    std::vector<int> output;
    if (text.empty()) return output;

    // SentencePiece normalization: substitute meta-space for ' '. Optionally
    // prefix a meta-space to the whole string (gemma disables this).
    std::string norm;
    if (add_space_prefix_) norm += kSpmSpace;
    for (char c : text) {
        if (c == ' ') norm += kSpmSpace;
        else norm += c;
    }

    std::vector<BpeSymbol> symbols;
    std::priority_queue<SpmBigram> work_queue;

    auto add_new_bigram = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        std::string left_token(symbols[left].text, symbols[left].n);
        std::string right_token(symbols[right].text, symbols[right].n);
        std::string merged = left_token + right_token;
        int id = piece_to_id(merged);
        if (id < 0 || id >= static_cast<int>(token_scores_.size())) return;
        work_queue.push({left, right, merged, token_scores_[id], merged.size()});
    };

    int index = 0;
    size_t offset = 0;
    while (offset < norm.size()) {
        BpeSymbol sym;
        size_t char_len = std::min(norm.size() - offset, unicode_len_utf8(norm[offset]));
        sym.text = norm.c_str() + offset;
        sym.n = char_len;
        offset += sym.n;
        sym.prev = index - 1;
        sym.next = (offset == norm.size()) ? -1 : index + 1;
        index++;
        symbols.push_back(sym);
    }

    for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
        add_new_bigram(i - 1, i);
    }

    while (!work_queue.empty()) {
        SpmBigram bigram = work_queue.top();
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

    for (int i = 0; i != -1 && i < static_cast<int>(symbols.size()); i = symbols[i].next) {
        if (symbols[i].n == 0) continue;
        std::string str(symbols[i].text, symbols[i].n);
        int id = piece_to_id(str);
        if (id >= 0) {
            output.push_back(id);
        } else {
            // Byte fallback: emit one <0xXX> token per raw byte.
            for (unsigned char b : str) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
                int bid = piece_to_id(buf);
                output.push_back(bid >= 0 ? bid : unk_token_id_);
            }
        }
    }

    return output;
}

std::string Tokenizer::spm_decode_piece(int token_id) const {
    auto it = reverse_vocab_.find(token_id);
    if (it == reverse_vocab_.end()) return "";
    const std::string& piece = it->second;

    // Byte tokens ("<0xXX>") decode to their raw byte.
    if (token_id >= 0 && token_id < static_cast<int>(token_types_.size()) &&
        token_types_[token_id] == TT_BYTE) {
        unsigned int b = 0;
        if (std::sscanf(piece.c_str(), "<0x%02X>", &b) == 1) {
            return std::string(1, static_cast<char>(b));
        }
    }

    // Normal tokens: literal UTF-8 with meta-space restored to ' '.
    std::string result;
    result.reserve(piece.size());
    for (size_t i = 0; i < piece.size();) {
        if (i + 3 <= piece.size() && std::memcmp(piece.data() + i, kSpmSpace, 3) == 0) {
            result += ' ';
            i += 3;
        } else {
            result += piece[i];
            i += 1;
        }
    }
    return result;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<int> tokens = is_unigram() ? spm_tokenize(text) : bpe_tokenize(text);

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
    if (is_unigram()) return spm_decode_piece(token_id);
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
