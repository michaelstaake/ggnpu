#ifndef GGNPU_TOKENIZER_H
#define GGNPU_TOKENIZER_H

#include <string>
#include <vector>
#include <map>
#include <utility>
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
    bool is_unigram() const { return model_type_ == ModelType::Unigram; }

private:
    enum class PreType { Default, Gpt2, Llama3, Tekken };
    // BPE (Llama/Qwen merge-rank) vs Unigram (SentencePiece score-max, e.g. gemma).
    enum class ModelType { Bpe, Unigram };

    using BpePair = std::pair<std::string, std::string>;

    std::map<BpePair, int> bpe_ranks_;
    std::map<std::string, int> vocab_;
    std::map<int, std::string> reverse_vocab_;
    PreType pre_type_ = PreType::Default;
    ModelType model_type_ = ModelType::Bpe;
    // Unigram (SentencePiece) state, indexed by token id.
    std::vector<float> token_scores_;
    std::vector<int32_t> token_types_;
    bool add_space_prefix_ = true;
    int unk_token_id_ = -1;
    int bos_token_id_ = -1;
    int eos_token_id_ = -1;
    bool add_bos_ = true;
    bool add_eos_ = true;

    int find_bpe_rank(const std::string& left, const std::string& right) const;
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<int> bpe_tokenize(const std::string& text) const;
    std::vector<int> bpe_tokenize_word(const std::string& word) const;
    std::string decode_piece(const std::string& piece) const;

    // Unigram / SentencePiece path.
    int piece_to_id(const std::string& piece) const;
    std::vector<int> spm_tokenize(const std::string& text) const;
    std::string spm_decode_piece(int token_id) const;
};

} // namespace ggnpu

#endif // GGNPU_TOKENIZER_H
