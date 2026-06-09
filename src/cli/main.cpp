#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <filesystem>

#include "gguf.h"
#include "tensor.h"
#include "model.h"
#include "backend.h"
#include "graph.h"
#include "cache.h"

namespace ggnpu {

namespace {

constexpr const char* VERSION = "0.1.0";

struct CliParams {
    std::string model_path;
    std::string prompt;
    int max_tokens = 128;
    int ctx_size = 0;
    int threads = 4;
    float temp = 0.0f;
    uint64_t seed = 0;
    int npu_device = 0;
    bool no_cache = false;
    std::string cache_dir = "";
    bool verbose = false;
    bool dump_tensors = false;
    bool bench_matmul = false;
    bool show_version = false;
};

void print_help() {
    std::cout << "ggnpu - Run GGUF AI Models on AMD and Intel NPUs\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ggnpu [options]                    Text generation\n";
    std::cout << "  ggnpu --dump-tensors               List model tensors\n";
    std::cout << "  ggnpu bench-matmul                 NPU matmul benchmark\n";
    std::cout << "  ggnpu --version                    Show version\n\n";
    std::cout << "Commands:\n";
    std::cout << "  (default)                          Text generation\n";
    std::cout << "  --dump-tensors                     List tensors; no NPU\n";
    std::cout << "  bench-matmul                       NPU matmul benchmark\n";
    std::cout << "  --version                          Version + backend info\n\n";
    std::cout << "Flags:\n";
    std::cout << "  -m, --model <path>                 Path to .gguf (required)\n";
    std::cout << "  -p, --prompt <text>                Input prompt\n";
    std::cout << "  -n, --max-tokens <n>               Max new tokens (default: 128)\n";
    std::cout << "  -c, --ctx-size <n>                 Context window (default: model default)\n";
    std::cout << "  -t, --threads <n>                  CPU threads for I/O (default: 4)\n";
    std::cout << "      --temp <f>                     Temperature (0 = greedy, default: 0)\n";
    std::cout << "      --seed <n>                     RNG seed (0 = random, default: 0)\n";
    std::cout << "      --npu-device <n>               NPU index (default: 0)\n";
    std::cout << "      --no-cache                     Disable caches\n";
    std::cout << "      --cache-dir <path>             Cache directory (default: ~/.cache/ggnpu)\n";
    std::cout << "  -v, --verbose                      Per-op timings\n";
    std::cout << "  -h, --help                         Print help\n";
    std::cout << "      --version                      Print version\n\n";
    std::cout << "Examples:\n";
    std::cout << "  ggnpu -m llama-3.2-1b-Q4_K_M.gguf -p \"The capital of France is\" -n 64\n";
    std::cout << "  ggnpu -m model.gguf -p \"Once upon a time\" -c 4096 -n 256 --temp 0.7 --seed 42\n";
    std::cout << "  ggnpu -m model.gguf --dump-tensors\n";
}

CliParams parse_args(int argc, char* argv[]) {
    CliParams params;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_help();
            exit(0);
        } else if (arg == "--version") {
            params.show_version = true;
        } else if (arg == "--dump-tensors") {
            params.dump_tensors = true;
        } else if (arg == "bench-matmul") {
            params.bench_matmul = true;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) params.model_path = argv[++i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 < argc) params.prompt = argv[++i];
        } else if (arg == "-n" || arg == "--max-tokens") {
            if (i + 1 < argc) params.max_tokens = std::atoi(argv[++i]);
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (i + 1 < argc) params.ctx_size = std::atoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) params.threads = std::atoi(argv[++i]);
        } else if (arg == "--temp") {
            if (i + 1 < argc) params.temp = std::atof(argv[++i]);
        } else if (arg == "--seed") {
            if (i + 1 < argc) params.seed = std::stoull(argv[++i]);
        } else if (arg == "--npu-device") {
            if (i + 1 < argc) params.npu_device = std::atoi(argv[++i]);
        } else if (arg == "--no-cache") {
            params.no_cache = true;
        } else if (arg == "--cache-dir") {
            if (i + 1 < argc) params.cache_dir = argv[++i];
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help();
            exit(1);
        }
    }

    return params;
}

// Simple tokenizer (byte-pair encoding stub)
// Full implementation would parse tokenizer.ggml.model or tokenizer.json

struct SimpleTokenizer {
    std::vector<int> encode(const std::string& text) {
        // Simplified: character-level tokenization
        // Production would use actual tokenizer from GGUF
        std::vector<int> tokens;
        std::string current;

        for (char c : text) {
            if (c == ' ') {
                if (!current.empty()) {
                    // Simple hash-based token ID
                    tokens.push_back(hash_string(current));
                    current.clear();
                }
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            tokens.push_back(hash_string(current));
        }

        return tokens;
    }

    std::string decode(int token_id) {
        // Simplified: reverse of encode
        // Production would use actual tokenizer
        return "token_" + std::to_string(token_id);
    }

private:
    int hash_string(const std::string& s) {
        int hash = 5381;
        for (char c : s) {
            hash = ((hash << 5) + hash) + c;
        }
        return std::abs(hash) % 50000;
    }
};

// Simple sampler
int sample_token(const std::vector<float>& logits, float temp, uint64_t& seed) {
    if (temp <= 0.0f) {
        // Greedy sampling
        int best = 0;
        float best_val = -INFINITY;
        for (int i = 0; i < static_cast<int>(logits.size()); i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best = i;
            }
        }
        return best;
    }

    // Temperature sampling
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Apply temperature
    float max_logit = -INFINITY;
    for (float v : logits) {
        if (v > max_logit) max_logit = v;
    }

    std::vector<float> probs;
    float sum_exp = 0.0f;
    for (float v : logits) {
        float exp_val = std::exp((v - max_logit) / temp);
        probs.push_back(exp_val);
        sum_exp += exp_val;
    }

    for (float& p : probs) {
        p /= sum_exp;
    }

    std::discrete_distribution<int> dist2(probs.begin(), probs.end());
    seed = rng(); // advance seed
    return dist2(rng);
}

} // namespace

} // namespace ggnpu

int main(int argc, char* argv[]) {
    using namespace ggnpu;

    CliParams params = parse_args(argc, argv);

    // Version
    if (params.show_version) {
        std::cout << "ggnpu version " << VERSION << "\n";
#ifdef GGNPU_HAS_NPU_BACKEND
        std::cout << "NPU backend: AMD XDNA (XRT)\n";
#else
        std::cout << "NPU backend: disabled (compile with -DGGNPU_NPU_BACKEND=ON)\n";
#endif
#ifdef GGNPU_TEST_CPU
        std::cout << "CPU ref backend: enabled (testing)\n";
#endif
        return 0;
    }

    // Help
    if (argc == 1) {
        print_help();
        return 0;
    }

    // Dump tensors
    if (params.dump_tensors) {
        if (params.model_path.empty()) {
            std::cerr << "Error: --model is required for --dump-tensors\n";
            return 1;
        }

        GgufLoader loader;
        if (!loader.load(params.model_path)) {
            std::cerr << "Error: failed to load GGUF file: " << params.model_path << "\n";
            return 1;
        }

        std::cout << "GGUF Header:\n";
        std::cout << "  Version: " << loader.header().version << "\n";
        std::cout << "  KV pairs: " << loader.header().kv_count << "\n";
        std::cout << "  Tensors: " << loader.header().tensor_count << "\n\n";

        std::cout << "Metadata:\n";
        std::cout << "  Architecture: " << loader.architecture() << "\n";
        std::cout << "  Context length: " << loader.context_length() << "\n";
        std::cout << "  Embedding length: " << loader.embedding_length() << "\n";
        std::cout << "  Block count: " << loader.block_count() << "\n";
        std::cout << "  Feed forward length: " << loader.feed_forward_length() << "\n";
        std::cout << "  Attention heads: " << loader.attention_head_count() << "\n";
        std::cout << "  Attention heads (KV): " << loader.attention_head_count_kv() << "\n";
        std::cout << "  Rope dimension count: " << loader.rope_dimension_count() << "\n";
        std::cout << "  Rope freq scale: " << loader.rope_freq_scale() << "\n";
        std::cout << "  Rope freq base: " << loader.rope_freq_base() << "\n\n";

        std::cout << "Key-Value Pairs:\n";
        for (const auto& [key, kv] : loader.kv_pairs()) {
            std::cout << "  " << key << " = ";
            switch (kv.value_type) {
                case GgufType::STRING:
                    std::cout << "\"" << kv.string_value << "\"\n";
                    break;
                case GgufType::INT64:
                case GgufType::UINT64:
                    std::cout << kv.int_value << "\n";
                    break;
                case GgufType::FLOAT64:
                    std::cout << kv.float_value << "\n";
                    break;
                case GgufType::INT32:
                case GgufType::UINT32:
                    std::cout << kv.int_value << "\n";
                    break;
                case GgufType::FLOAT32:
                    std::cout << kv.float_value << "\n";
                    break;
                case GgufType::BOOL:
                    std::cout << (kv.int_value ? "true" : "false") << "\n";
                    break;
                case GgufType::ARRAY:
                    std::cout << "[array, " << kv.data.size() << " bytes]\n";
                    break;
                default:
                    std::cout << "[" << kv.data.size() << " bytes]\n";
                    break;
            }
        }

        std::cout << "\nTensors (" << loader.tensors().size() << "):\n";
        for (const auto& t : loader.tensors()) {
            std::cout << "  " << t.name << " dims=[";
            for (size_t i = 0; i < t.dims.size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << t.dims[i];
            }
            std::cout << "] type=" << ggml_type_name(t.type) << "\n";
        }

        loader.unload();
        return 0;
    }

    // Bench matmul
    if (params.bench_matmul) {
        std::cout << "GGNPU Matmul Benchmark\n";
        std::cout << "======================\n\n";

        // Select backend
        std::shared_ptr<Backend> backend;
#ifdef GGNPU_HAS_NPU_BACKEND
        backend = create_amd_xdna_backend(params.npu_device);
        if (!backend || !backend->is_available()) {
            std::cerr << "Warning: NPU backend unavailable, using CPU reference\n";
            backend = create_cpu_ref_backend();
        }
#else
        backend = create_cpu_ref_backend();
#endif

        std::cout << "Backend: " << backend->name() << "\n\n";

        // Test various matrix sizes
        std::vector<std::tuple<int, int, int>> sizes = {
            {256, 256, 256},
            {512, 512, 512},
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {3072, 3072, 3072},
            {4096, 4096, 4096},
            {8192, 8192, 8192},
        };

        for (auto [M, N, K] : sizes) {
            // Allocate buffers
            std::vector<float> A(M * K, 1.0f);
            std::vector<float> B(K * N, 1.0f);
            std::vector<float> C(M * N, 0.0f);

            MulMatParams p;
            p.A = A.data();
            p.B = B.data();
            p.C = C.data();
            p.M = M;
            p.N = N;
            p.K = K;
            p.lda = K;
            p.ldb = N;
            p.ldc = N;
            p.n_batches = 1;

            // Warmup
            backend->mul_mat_q(p);
            backend->sync();

            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();
            int iterations = 10;
            for (int i = 0; i < iterations; i++) {
                backend->mul_mat_q(p);
                backend->sync();
            }
            auto end = std::chrono::high_resolution_clock::now();

            double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            double ops = 2.0 * M * N * K * iterations;
            double tflops = ops / (elapsed_ms * 1e-6) / 1e12;

            std::cout << "  " << M << "x" << K << " x " << K << "x" << N
                      << ": " << tflops << " TFLOPS (" << elapsed_ms / iterations << " ms)\n";
        }

        return 0;
    }

    // Text generation
    if (params.model_path.empty()) {
        std::cerr << "Error: --model is required\n\n";
        print_help();
        return 1;
    }

    std::cout << "GGNPU - GGUF NPU Inference Engine v" << VERSION << "\n\n";

    // Select backend
    std::shared_ptr<Backend> backend;
#ifdef GGNPU_HAS_NPU_BACKEND
    backend = create_amd_xdna_backend(params.npu_device);
    if (!backend || !backend->is_available()) {
        std::cerr << "Warning: NPU backend unavailable, using CPU reference for testing\n";
        backend = create_cpu_ref_backend();
    }
#else
    backend = create_cpu_ref_backend();
#endif

    std::cout << "Backend: " << backend->name() << "\n";

    // Setup cache
    std::string cache_dir = params.cache_dir;
    if (cache_dir.empty()) {
        const char* home = std::getenv("HOME");
        cache_dir = home ? std::string(home) + "/.cache/ggnpu" : "~/.cache/ggnpu";
    }

    CompileCache cache(cache_dir, !params.no_cache);

    // Load model
    std::cout << "Loading model: " << params.model_path << "\n";
    Model model;
    if (!model.load(params.model_path)) {
        std::cerr << "Error: failed to load model\n";
        return 1;
    }
    model.set_backend(backend);

    const auto& hparams = model.hparams();
    std::cout << "  Architecture: " << model.gguf().architecture() << "\n";
    std::cout << "  Context: " << hparams.context_length << "\n";
    std::cout << "  Layers: " << hparams.block_count << "\n";
    std::cout << "  Hidden: " << hparams.embedding_length << "\n";
    std::cout << "  Heads: " << hparams.attention_head_count << " (KV: " << hparams.attention_head_count_kv << ")\n";
    std::cout << "  FFN: " << hparams.feed_forward_length << "\n\n";

    if (params.prompt.empty()) {
        std::cout << "No prompt provided. Use --prompt or -p to specify input.\n";
        model.unload();
        return 0;
    }

    // Tokenize
    SimpleTokenizer tokenizer;
    std::vector<int> input_tokens = tokenizer.encode(params.prompt);
    std::cout << "Input tokens: " << input_tokens.size() << "\n";
    std::cout << "Prompt: " << params.prompt << "\n\n";

    // Generate
    std::cout << "Generating: ";

    uint64_t rng_seed = params.seed;
    if (rng_seed == 0) {
        rng_seed = std::chrono::steady_clock::now().time_since_epoch().count();
    }

    int generated = 0;
    for (int i = 0; i < params.max_tokens; i++) {
        // Simplified generation loop
        // In production, this would run the full compute graph

        // Sample next token
        std::vector<float> logits(100);
        for (auto& l : logits) {
            l = static_cast<float>(std::rand()) / RAND_MAX;
        }

        int next_token = sample_token(logits, params.temp, rng_seed);
        std::string decoded = tokenizer.decode(next_token);

        std::cout << decoded << " ";
        generated++;

        // Stop on special tokens (simplified)
        if (next_token == 0 || next_token == 1) break;
    }

    std::cout << "\n\nGenerated " << generated << " tokens.\n";

    // Cleanup
    model.unload();
    return 0;
}
