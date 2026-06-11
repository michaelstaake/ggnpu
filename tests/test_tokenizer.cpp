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
    ASSERT_EQ(tokenizer.vocab_size(), 128256, "Llama 3.2 vocab size");

    auto tokens = tokenizer.encode("Hello", true, false);
    ASSERT_EQ(tokens.size(), 2u, "Hello token count");
    ASSERT_EQ(tokens[0], 128000, "BOS token");
    ASSERT_EQ(tokens[1], 9906, "Hello token");

    std::string decoded = tokenizer.decode(tokens);
    ASSERT_TRUE(decoded.find("Hello") != std::string::npos, "decode contains Hello");

    std::cout << "All tokenizer tests passed.\n";
    return 0;
}
