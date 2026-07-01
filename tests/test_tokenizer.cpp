#include "tokenizer.h"
#include "gguf.h"
#include <iostream>
#include <cstdlib>

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; std::exit(1); } \
} while (0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        std::cerr << "FAIL: " << msg << " (got " << (a) << ", expected " << (b) << ")\n"; \
        std::exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_tokenizer <model.gguf>\n";
        return 1;
    }

    ggnpu::GgufLoader loader;
    ASSERT_TRUE(loader.load(argv[1]), "load GGUF");

    ggnpu::Tokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(loader.kv_pairs()), "load tokenizer");

    // Golden encodings verified against a reference SentencePiece/BPE tokenizer.
    auto check_golden = [&](const char* label, const std::string& text,
                            const std::vector<int>& expected) {
        auto got = tokenizer.encode(text, false, false);
        if (got.size() != expected.size()) {
            std::cerr << "FAIL: " << label << " token count (got " << got.size()
                      << ", expected " << expected.size() << ")\n  got:";
            for (int t : got) std::cerr << " " << t;
            std::cerr << "\n";
            std::exit(1);
        }
        for (size_t i = 0; i < expected.size(); i++) {
            if (got[i] != expected[i]) {
                std::cerr << "FAIL: " << label << " token[" << i << "] got " << got[i]
                          << " expected " << expected[i]
                          << " ('" << tokenizer.decode(got[i]) << "')\n";
                std::exit(1);
            }
        }
    };

    // Round-trip: encode then decode reconstructs the input text.
    auto check_roundtrip = [&](const std::string& text) {
        auto got = tokenizer.encode(text, false, false);
        std::string back = tokenizer.decode(got);
        if (back != text) {
            std::cerr << "FAIL: round-trip mismatch\n  in:  '" << text
                      << "'\n  out: '" << back << "'\n";
            std::exit(1);
        }
    };

    if (tokenizer.is_unigram()) {
        // Gemma 3n (gemma4) — SentencePiece unigram, add_space_prefix=false.
        ASSERT_EQ(tokenizer.vocab_size(), 262144, "Gemma vocab size");
        ASSERT_EQ(tokenizer.bos_token_id(), 2, "Gemma BOS id");
        check_golden("gemma france", "The capital of France is",
                     {818, 5279, 529, 7001, 563});
        check_golden("gemma hello", "Hello, world!",
                     {9259, 236764, 1902, 236888});
        check_golden("gemma code", "def f(x): return x*2",
                     {2063, 517, 236769, 236781, 1473, 994, 1123, 236829, 236778});
        check_roundtrip("The capital of France is");
        check_roundtrip("Hello, world!");
        check_roundtrip("def f(x): return x*2");
        // BOS prepend still works via encode flags.
        auto with_bos = tokenizer.encode("Hello, world!", true, false);
        ASSERT_EQ(with_bos.front(), 2, "Gemma BOS prepended");

        // Metadata array accessors (G2+ per-layer FFN + SWA/global pattern).
        auto ffn = loader.get_int_array(loader.arch_key("feed_forward_length"));
        ASSERT_EQ(ffn.size(), 35u, "gemma per-layer FFN count");
        ASSERT_EQ(ffn[0], 6144, "gemma ffn[0]");
        ASSERT_EQ(ffn[34], 12288, "gemma ffn[34]");
        auto swa = loader.get_int_array(loader.arch_key("attention.sliding_window_pattern"));
        ASSERT_EQ(swa.size(), 35u, "gemma SWA pattern count");
        ASSERT_EQ(swa[0], 1, "gemma SWA[0] local");
        ASSERT_EQ(swa[4], 0, "gemma SWA[4] global");
        // Float array accessor on the scores table.
        auto scores = loader.get_float_array("tokenizer.ggml.scores");
        ASSERT_EQ(scores.size(), 262144u, "gemma scores count");

        std::cout << "All tokenizer tests passed (gemma unigram).\n";
        return 0;
    }

    // BPE models (Llama 3 / Qwen2, tokenizer.ggml.model == "gpt2").
    if (tokenizer.vocab_size() == 128256) {
        // Llama 3.2 reference tokenization (matches llama.cpp).
        auto tokens = tokenizer.encode("Hello", true, false);
        ASSERT_EQ(tokens.size(), 2u, "Hello token count");
        ASSERT_EQ(tokens[0], 128000, "BOS token");
        ASSERT_EQ(tokens[1], 9906, "Hello token");

        std::string decoded = tokenizer.decode(tokens);
        ASSERT_TRUE(decoded.find("Hello") != std::string::npos, "decode contains Hello");

        check_golden("llama france", "The capital of France is",
                     {791, 6864, 315, 9822, 374});
    } else {
        // Other BPE arch (e.g. Qwen2): smoke-check that encode yields tokens.
        // (Strict goldens are Llama-specific; BPE decode fidelity is exercised
        // by real inference, not asserted here.)
        auto got = tokenizer.encode("The capital of France is", false, false);
        ASSERT_TRUE(!got.empty(), "BPE encode non-empty");
    }

    std::cout << "All tokenizer tests passed (bpe).\n";
    return 0;
}
