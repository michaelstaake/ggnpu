#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <sstream>

#include "gguf.h"
#include "tensor.h"
#include "model.h"
#include "backend.h"
#include "graph.h"
#include "cache.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "weight_cache.h"

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

int sample_token(const std::vector<float>& logits, float temp, uint64_t& seed) {
    if (temp <= 0.0f) {
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

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float max_logit = -INFINITY;
    for (float v : logits) {
        if (v > max_logit) max_logit = v;
    }

    std::vector<float> probs;
    probs.reserve(logits.size());
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
    seed = rng();
    return dist2(rng);
}

// Helper: find tensor by name in model
const TensorView* find_tensor(const Model& model, const std::string& name) {
    auto it = model.tensor_map().find(name);
    if (it == model.tensor_map().end()) return nullptr;
    return &model.tensors()[it->second];
}

// Helper: find tensor by pattern (e.g., "blk.{layer}.ffn_gate.weight")
const TensorView* find_tensor_pattern(const Model& model, const std::string& pattern, int layer) {
    std::string expanded = pattern;
    size_t pos = expanded.find("{layer}");
    if (pos != std::string::npos) {
        expanded.replace(pos, 7, std::to_string(layer));
    }
    return find_tensor(model, expanded);
}

// Helper: get raw pointer from TensorView
inline const float* get_float_ptr(const TensorView* tv) {
    if (!tv || !tv->data) return nullptr;
    return reinterpret_cast<const float*>(tv->data);
}

inline float* get_float_ptr_mut(std::vector<float>& buf) {
    return buf.data();
}

// Build and execute a compute graph for one transformer layer
Status execute_layer_graph(ComputeGraph& graph, std::shared_ptr<Backend> backend,
                           const Model& model, WeightCache& weight_cache,
                           const std::vector<float>& inp_norm,
                           std::vector<float>& q_proj, std::vector<float>& k_proj,
                           std::vector<float>& v_proj, std::vector<float>& attn_output,
                           std::vector<float>& ffn_gate, std::vector<float>& ffn_up,
                           std::vector<float>& ffn_down, std::vector<float>& ffn_silu,
                           int layer, int n_tokens, int hidden_size, int ffn_size,
                           int num_kv_heads, int head_dim, int /*num_heads*/,
                           float /*rms_eps*/) {
    graph.reset();
    graph.set_backend(backend);

    auto add_f32_matmul = [&](const std::string& name, const float* A, const int8_t* B,
                               float* C, int M, int N, int K, GgmlType B_type) {
        auto matmul_node = graph.add_node(OpType::MUL_MAT_Q, name);
        matmul_node->cpu_buffer = const_cast<float*>(A);
        matmul_node->M = M;
        matmul_node->N = N;
        matmul_node->K = K;
        matmul_node->lda = K;
        matmul_node->ldb = N;
        matmul_node->ldc = N;
        matmul_node->n_batches = 1;
        matmul_node->B_type = B_type;

        auto input_node = graph.add_node(OpType::VIEW, name + "_input");
        input_node->cpu_buffer = const_cast<float*>(A);
        input_node->size = M * K;

        auto weight_node = graph.add_node(OpType::COPY, name + "_weight");
        weight_node->cpu_buffer = const_cast<int8_t*>(B);
        weight_node->size = K * N;

        auto out_node = graph.add_node(OpType::COPY, name + "_out");
        out_node->cpu_buffer = C;
        out_node->size = M * N;

        graph.connect(input_node, matmul_node);
        graph.connect(weight_node, matmul_node);
        graph.connect(matmul_node, out_node);

        return out_node;
    };

    auto add_rmsnorm = [&](const std::string& name, const float* input, float* output,
                            int size, float eps) {
        auto rms_node = graph.add_node(OpType::RMS_NORM, name);
        rms_node->cpu_buffer = output;
        rms_node->size = size;
        rms_node->eps = eps;

        auto input_node = graph.add_node(OpType::VIEW, name + "_input");
        input_node->cpu_buffer = const_cast<float*>(input);
        input_node->size = size;

        graph.connect(input_node, rms_node);
        return rms_node;
    };

    auto add_silu = [&](const std::string& name, const float* input, float* output, int size) {
        auto silu_node = graph.add_node(OpType::SILU, name);
        silu_node->cpu_buffer = output;
        silu_node->size = size;

        auto input_node = graph.add_node(OpType::VIEW, name + "_input");
        input_node->cpu_buffer = const_cast<float*>(input);
        input_node->size = size;

        graph.connect(input_node, silu_node);
        return silu_node;
    };

    auto add_add_residual = [&](const std::string& name, const float* a, const float* b,
                                 float* out, int size) {
        auto add_node = graph.add_node(OpType::ADD, name);
        add_node->cpu_buffer = out;
        add_node->size = size;

        auto a_node = graph.add_node(OpType::VIEW, name + "_a");
        a_node->cpu_buffer = const_cast<float*>(a);
        a_node->size = size;

        auto b_node = graph.add_node(OpType::VIEW, name + "_b");
        b_node->cpu_buffer = const_cast<float*>(b);
        b_node->size = size;

        graph.connect(a_node, add_node);
        graph.connect(b_node, add_node);
        return add_node;
    };

    // Attention path
    const TensorView* attn_q = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", layer);
    const TensorView* attn_k = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", layer);
    const TensorView* attn_v = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", layer);

    if (attn_q) {
        const int8_t* decoded_q = weight_cache.get_or_decode(attn_q->name,
            attn_q->data, attn_q->data_size(), attn_q->type);
        if (decoded_q) {
            add_f32_matmul("attn_q", inp_norm.data(), decoded_q, q_proj.data(),
                          n_tokens, hidden_size, hidden_size, attn_q->type);
        }
    }

    if (attn_k) {
        const int8_t* decoded_k = weight_cache.get_or_decode(attn_k->name,
            attn_k->data, attn_k->data_size(), attn_k->type);
        if (decoded_k) {
            add_f32_matmul("attn_k", inp_norm.data(), decoded_k, k_proj.data(),
                          n_tokens, num_kv_heads * head_dim, hidden_size, attn_k->type);
        }
    }

    if (attn_v) {
        const int8_t* decoded_v = weight_cache.get_or_decode(attn_v->name,
            attn_v->data, attn_v->data_size(), attn_v->type);
        if (decoded_v) {
            add_f32_matmul("attn_v", inp_norm.data(), decoded_v, v_proj.data(),
                          n_tokens, num_kv_heads * head_dim, hidden_size, attn_v->type);
        }
    }

    // Attention output projection
    const TensorView* attn_output_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", layer);
    if (attn_output_w) {
        const int8_t* decoded_attn_out = weight_cache.get_or_decode(attn_output_w->name,
            attn_output_w->data, attn_output_w->data_size(), attn_output_w->type);
        if (decoded_attn_out) {
            add_f32_matmul("attn_out_proj", attn_output.data(), decoded_attn_out,
                          attn_output.data(), n_tokens, hidden_size, hidden_size,
                          attn_output_w->type);
        }
    }

    // FFN path
    const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer);
    const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer);
    const TensorView* ffn_down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", layer);

    if (ffn_gate_w) {
        const int8_t* decoded_gate = weight_cache.get_or_decode(ffn_gate_w->name,
            ffn_gate_w->data, ffn_gate_w->data_size(), ffn_gate_w->type);
        if (decoded_gate) {
            add_f32_matmul("ffn_gate", inp_norm.data(), decoded_gate, ffn_gate.data(),
                          n_tokens, ffn_size, hidden_size, ffn_gate_w->type);
        }
    }

    if (ffn_up_w) {
        const int8_t* decoded_up = weight_cache.get_or_decode(ffn_up_w->name,
            ffn_up_w->data, ffn_up_w->data_size(), ffn_up_w->type);
        if (decoded_up) {
            add_f32_matmul("ffn_up", inp_norm.data(), decoded_up, ffn_up.data(),
                          n_tokens, ffn_size, hidden_size, ffn_up_w->type);
        }
    }

    if (ffn_down_w) {
        const int8_t* decoded_down = weight_cache.get_or_decode(ffn_down_w->name,
            ffn_down_w->data, ffn_down_w->data_size(), ffn_down_w->type);
        if (decoded_down) {
            add_silu("ffn_silu", ffn_up.data(), ffn_silu.data(), ffn_size);
            add_f32_matmul("ffn_down", ffn_silu.data(), decoded_down, ffn_down.data(),
                          n_tokens, hidden_size, ffn_size, ffn_down_w->type);
        }
    }

    Status st = graph.compile();
    if (st != Status::OK) return st;
    return graph.execute();
}

} // namespace

} // namespace ggnpu

int main(int argc, char* argv[]) {
    using namespace ggnpu;

    CliParams params = parse_args(argc, argv);

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

    if (argc == 1) {
        print_help();
        return 0;
    }

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

    if (params.bench_matmul) {
        std::cout << "GGNPU Matmul Benchmark\n";
        std::cout << "======================\n\n";

        std::shared_ptr<Backend> backend;
#ifdef GGNPU_HAS_NPU_BACKEND
        backend = create_amd_xdna_backend(params.npu_device);
        if (!backend || !backend->is_available()) {
#ifdef GGNPU_TEST_CPU
            std::cerr << "Warning: NPU backend unavailable, using CPU reference for testing\n";
            backend = create_cpu_ref_backend();
#else
            std::cerr << "Error: NPU backend unavailable. AMD NPU (amdxdna) driver must be loaded.\n";
            std::cerr << "  lsmod | grep amdxdna\n";
            std::cerr << "  ls -la /dev/accel/accel0\n";
            return 1;
#endif
        }
#else
#ifdef GGNPU_TEST_CPU
        backend = create_cpu_ref_backend();
#else
        std::cerr << "Error: NPU backend not compiled in. Build with -DGGNPU_NPU_BACKEND=ON\n";
        return 1;
#endif
#endif

        std::cout << "Backend: " << backend->name() << "\n\n";

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
            p.B_type = GgmlType::F32;

            backend->mul_mat_q(p);
            backend->sync();

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

    if (params.model_path.empty()) {
        std::cerr << "Error: --model is required\n\n";
        print_help();
        return 1;
    }

    std::cout << "GGNPU - GGUF NPU Inference Engine v" << VERSION << "\n\n";

    std::shared_ptr<Backend> backend;
#ifdef GGNPU_HAS_NPU_BACKEND
    backend = create_amd_xdna_backend(params.npu_device);
    if (!backend || !backend->is_available()) {
#ifdef GGNPU_TEST_CPU
        std::cerr << "Warning: NPU backend unavailable, using CPU reference for testing\n";
        backend = create_cpu_ref_backend();
#else
        std::cerr << "Error: NPU backend unavailable. AMD NPU (amdxdna) driver must be loaded.\n";
        std::cerr << "  lsmod | grep amdxdna\n";
        std::cerr << "  ls -la /dev/accel/accel0\n";
        return 1;
#endif
    }
#else
#ifdef GGNPU_TEST_CPU
    backend = create_cpu_ref_backend();
#else
    std::cerr << "Error: NPU backend not compiled in. Build with -DGGNPU_NPU_BACKEND=ON\n";
    return 1;
#endif
#endif

    std::cout << "Backend: " << backend->name() << "\n";

    std::string cache_dir = params.cache_dir;
    if (cache_dir.empty()) {
        const char* home = std::getenv("HOME");
        cache_dir = home ? std::string(home) + "/.cache/ggnpu" : "~/.cache/ggnpu";
    }

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

    // Load tokenizer
    Tokenizer tokenizer;
    tokenizer.load_from_gguf(model.gguf().kv_pairs());
    std::cout << "Tokenizer: " << tokenizer.vocab_size() << " tokens\n";

    // Tokenize prompt
    std::vector<int> input_tokens = tokenizer.encode(params.prompt, true, false);
    std::cout << "Input tokens: " << input_tokens.size() << "\n";
    std::cout << "Prompt: " << params.prompt << "\n\n";

    // Weight cache: decode GGUF quantized weights to INT8 for NPU
    CompileCache compile_cache(cache_dir, !params.no_cache);
    WeightCache weight_cache(compile_cache);
    std::cout << "Weight cache initialized\n";

    if (params.ctx_size > 0) {
        model.set_context_length(static_cast<uint64_t>(params.ctx_size));
    }

    // Setup working buffers
    int hidden_size = static_cast<int>(hparams.embedding_length);
    int ffn_size = static_cast<int>(hparams.feed_forward_length);
    int num_layers = static_cast<int>(hparams.block_count);
    int num_heads = static_cast<int>(hparams.attention_head_count);
    int num_kv_heads = static_cast<int>(hparams.attention_head_count_kv);
    int head_dim = static_cast<int>(hparams.rope_dimension_count);
    int ctx_size = params.ctx_size > 0 ? params.ctx_size : static_cast<int>(hparams.context_length);
    float rms_eps = 1e-5f;
    if (hparams.attention_layer_norm_rms_epsilon > 0) {
        rms_eps = static_cast<float>(hparams.attention_layer_norm_rms_epsilon);
    }
    float rope_freq_scale = static_cast<float>(hparams.rope_freq_scale);
    float rope_freq_base = static_cast<float>(hparams.rope_freq_base);

    // Working buffers for activations
    std::vector<float> inp_embd(hidden_size);
    std::vector<float> inp_norm(hidden_size);
    std::vector<float> q_proj(hidden_size);
    std::vector<float> k_proj(num_kv_heads * head_dim);
    std::vector<float> v_proj(num_kv_heads * head_dim);
    std::vector<float> k_rope(num_kv_heads * head_dim);
    std::vector<float> q_rope(num_heads * head_dim);
    std::vector<float> attn_output(hidden_size);
    std::vector<float> ffn_gate(ffn_size);
    std::vector<float> ffn_up(ffn_size);
    std::vector<float> ffn_down(hidden_size);
    std::vector<float> ffn_silu(ffn_size);
    std::vector<float> logits(static_cast<size_t>(hparams.vocab_size));

    // RoPE frequency buffer
    std::vector<float> rope_freqs(head_dim / 2);
    for (int i = 0; i < head_dim / 2; i++) {
        rope_freqs[i] = 1.0f / std::pow(rope_freq_base, static_cast<float>(i * 2) / head_dim) / rope_freq_scale;
    }

    uint64_t rng_seed = params.seed;
    if (rng_seed == 0) {
        rng_seed = std::chrono::steady_clock::now().time_since_epoch().count();
    }

    int64_t pos = 0;
    int generated = 0;
    std::vector<int> current_tokens = input_tokens;

    auto apply_rope = [&](float* out, const float* inp, int n_heads, int64_t offset,
                          int n_dims, const std::vector<float>& freqs) {
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < n_dims / 2; i++) {
                float freq = freqs[i];
                float val = static_cast<float>(offset) * freq;
                float cos_val = std::cos(val);
                float sin_val = std::sin(val);

                float v0 = inp[h * n_dims + i];
                float v1 = inp[h * n_dims + i + n_dims / 2];
                out[h * n_dims + i] = v0 * cos_val - v1 * sin_val;
                out[h * n_dims + i + n_dims / 2] = v0 * sin_val + v1 * cos_val;
            }
        }
    };

    auto rms_norm = [&](float* out, const float* inp, int size, float eps) {
        RmsNormParams rms_params;
        rms_params.input = inp;
        rms_params.output = out;
        rms_params.size = size;
        rms_params.eps = eps;
        backend->rms_norm(rms_params);
    };

    // Dequantize token embeddings once before the generation loop
    const TensorView* tok_embd = find_tensor(model, "token_embd.weight");
    std::vector<float> tok_embd_f32;
    const float* embd_data = nullptr;
    if (tok_embd) {
        if (tok_embd->type == GgmlType::F32) {
            embd_data = get_float_ptr(tok_embd);
        } else {
            const int8_t* decoded = weight_cache.get_or_decode(tok_embd->name,
                tok_embd->data, tok_embd->data_size(), tok_embd->type);
            if (decoded) {
                int64_t embd_dim = static_cast<int64_t>(hparams.embedding_length);
                int64_t vocab_size = static_cast<int64_t>(hparams.vocab_size);
                tok_embd_f32.resize(static_cast<size_t>(vocab_size * embd_dim));
                for (int64_t i = 0; i < vocab_size; i++) {
                    for (int64_t d = 0; d < embd_dim; d++) {
                        tok_embd_f32[static_cast<size_t>(i * embd_dim + d)] =
                            static_cast<float>(decoded[i * embd_dim + d]);
                    }
                }
                embd_data = tok_embd_f32.data();
                std::cout << "  Dequantized token_embd.weight ("
                    << ggml_type_name(tok_embd->type) << " -> F32)\n";
            }
        }
    }

    // Main generation loop
    std::cout << "Generating: ";

    bool first_token = true;
    while (generated < params.max_tokens) {
        int n_tokens = static_cast<int>(current_tokens.size());
        if (n_tokens == 0) break;

        // Embedding lookup - get embeddings for each token
        if (!embd_data) {
            std::cerr << "Error: token_embd.weight not available\n";
            break;
        }
        int64_t embd_dim = static_cast<int64_t>(hparams.embedding_length);

        std::fill(inp_embd.begin(), inp_embd.end(), 0.0f);
        for (int t = 0; t < n_tokens; t++) {
            int tok_id = current_tokens[t];
            for (int d = 0; d < hidden_size; d++) {
                inp_embd[d] += embd_data[tok_id * embd_dim + d];
            }
        }

        // Transformer layers
        for (int layer = 0; layer < num_layers; layer++) {
            // RMSNorm for attention
            rms_norm(inp_norm.data(), inp_embd.data(), hidden_size, rms_eps);

            // Attention projections
            const TensorView* attn_q = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", layer);
            const TensorView* attn_k = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", layer);
            const TensorView* attn_v = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", layer);

            if (attn_q) {
                const int8_t* decoded_q = weight_cache.get_or_decode(attn_q->name,
                    attn_q->data, attn_q->data_size(), attn_q->type);
                if (decoded_q) {
                    MulMatParams q_params;
                    q_params.A = inp_norm.data();
                    q_params.B = decoded_q;
                    q_params.C = q_proj.data();
                    q_params.M = n_tokens;
                    q_params.N = hidden_size;
                    q_params.K = hidden_size;
                    q_params.lda = hidden_size;
                    q_params.ldb = hidden_size;
                    q_params.ldc = hidden_size;
                    q_params.n_batches = 1;
                    q_params.B_type = attn_q->type;
                    backend->mul_mat_q(q_params);
                }
            }

            if (attn_k) {
                const int8_t* decoded_k = weight_cache.get_or_decode(attn_k->name,
                    attn_k->data, attn_k->data_size(), attn_k->type);
                if (decoded_k) {
                    MulMatParams k_params;
                    k_params.A = inp_norm.data();
                    k_params.B = decoded_k;
                    k_params.C = k_proj.data();
                    k_params.M = n_tokens;
                    k_params.N = num_kv_heads * head_dim;
                    k_params.K = hidden_size;
                    k_params.lda = hidden_size;
                    k_params.ldb = hidden_size;
                    k_params.ldc = num_kv_heads * head_dim;
                    k_params.n_batches = 1;
                    k_params.B_type = attn_k->type;
                    backend->mul_mat_q(k_params);
                }
            }

            if (attn_v) {
                const int8_t* decoded_v = weight_cache.get_or_decode(attn_v->name,
                    attn_v->data, attn_v->data_size(), attn_v->type);
                if (decoded_v) {
                    MulMatParams v_params;
                    v_params.A = inp_norm.data();
                    v_params.B = decoded_v;
                    v_params.C = v_proj.data();
                    v_params.M = n_tokens;
                    v_params.N = num_kv_heads * head_dim;
                    v_params.K = hidden_size;
                    v_params.lda = hidden_size;
                    v_params.ldb = hidden_size;
                    v_params.ldc = num_kv_heads * head_dim;
                    v_params.n_batches = 1;
                    v_params.B_type = attn_v->type;
                    backend->mul_mat_q(v_params);
                }
            }

            backend->sync();

            // Apply RoPE to Q and K
            apply_rope(q_rope.data(), q_proj.data(), num_heads, pos,
                      head_dim, rope_freqs);
            apply_rope(k_rope.data(), k_proj.data(), num_kv_heads, pos,
                      head_dim, rope_freqs);

            // Store KV in cache
            model.kv_cache().update_slab(layer, pos, pos + n_tokens,
                                         k_rope.data(), v_proj.data(),
                                         num_kv_heads, head_dim);

            // Attention: compute softmax(QK^T / sqrt(d)) @ V via NPU
            int64_t ctx_len = pos + n_tokens;
            int64_t num_kv_heads = hparams.attention_head_count_kv;
            int64_t qkv_groups = hparams.attention_head_count / num_kv_heads;

            // Expand K/V from num_kv_heads to num_heads (GQA support)
            // Each KV head is repeated qkv_groups times for the corresponding query heads
            std::vector<float> k_expanded(static_cast<size_t>(hparams.attention_head_count * ctx_len * head_dim));
            std::vector<float> v_expanded(static_cast<size_t>(hparams.attention_head_count * ctx_len * head_dim));

            for (int64_t h = 0; h < static_cast<int64_t>(hparams.attention_head_count); h++) {
                int64_t kv_h = h / qkv_groups;
                float* kh = k_expanded.data() + h * ctx_len * head_dim;
                float* vh = v_expanded.data() + h * ctx_len * head_dim;
                for (int64_t j = 0; j < ctx_len; j++) {
                    const float* kj = model.kv_cache().key_buffer(layer, j) + kv_h * head_dim;
                    const float* vj = model.kv_cache().value_buffer(layer, j) + kv_h * head_dim;
                    if (kj) std::memcpy(kh + j * head_dim, kj, head_dim * sizeof(float));
                    if (vj) std::memcpy(vh + j * head_dim, vj, head_dim * sizeof(float));
                }
            }

            std::vector<float> attn_out(static_cast<size_t>(n_tokens) * head_dim, 0.0f);

            for (int64_t i = 0; i < n_tokens; i++) {
                AttnParams fa_params;
                fa_params.Q = q_rope.data() + static_cast<int>(i) * head_dim;
                fa_params.K = k_expanded.data();
                fa_params.V = v_expanded.data();
                fa_params.output = attn_out.data() + static_cast<int>(i) * head_dim;
                fa_params.batch_size = 1;
                fa_params.n_head = static_cast<int>(hparams.attention_head_count);
                fa_params.head_dim = head_dim;
                fa_params.ctx_len = ctx_len;
                fa_params.freq_factors = nullptr;

                backend->flash_attn(fa_params);
            }

            // Attention output projection
            const TensorView* attn_output_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", layer);
            if (attn_output_w) {
                std::fill(attn_output.begin(), attn_output.end(), 0.0f);
                const int8_t* decoded_attn_out = weight_cache.get_or_decode(attn_output_w->name,
                    attn_output_w->data, attn_output_w->data_size(), attn_output_w->type);
                if (decoded_attn_out) {
                    MulMatParams out_params;
                    out_params.A = attn_out.data();
                    out_params.B = decoded_attn_out;
                    out_params.C = attn_output.data();
                    out_params.M = n_tokens;
                    out_params.N = hidden_size;
                    out_params.K = hidden_size;
                    out_params.lda = head_dim;
                    out_params.ldb = hidden_size;
                    out_params.ldc = hidden_size;
                    out_params.n_batches = 1;
                    out_params.B_type = attn_output_w->type;
                    backend->mul_mat_q(out_params);
                    backend->sync();

                    // Add residual
                    for (int i = 0; i < hidden_size; i++) {
                        inp_embd[i] += attn_output[i];
                    }
                }
            }

            // FFN path
            rms_norm(inp_norm.data(), inp_embd.data(), hidden_size, rms_eps);

            const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer);
            const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer);
            const TensorView* ffn_down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", layer);

            if (ffn_gate_w) {
                const int8_t* decoded_gate = weight_cache.get_or_decode(ffn_gate_w->name,
                    ffn_gate_w->data, ffn_gate_w->data_size(), ffn_gate_w->type);
                if (decoded_gate) {
                    MulMatParams gate_params;
                    gate_params.A = inp_norm.data();
                    gate_params.B = decoded_gate;
                    gate_params.C = ffn_gate.data();
                    gate_params.M = n_tokens;
                    gate_params.N = ffn_size;
                    gate_params.K = hidden_size;
                    gate_params.lda = hidden_size;
                    gate_params.ldb = ffn_size;
                    gate_params.ldc = ffn_size;
                    gate_params.n_batches = 1;
                    gate_params.B_type = ffn_gate_w->type;
                    backend->mul_mat_q(gate_params);
                }
            }

            if (ffn_up_w) {
                const int8_t* decoded_up = weight_cache.get_or_decode(ffn_up_w->name,
                    ffn_up_w->data, ffn_up_w->data_size(), ffn_up_w->type);
                if (decoded_up) {
                    MulMatParams up_params;
                    up_params.A = inp_norm.data();
                    up_params.B = decoded_up;
                    up_params.C = ffn_up.data();
                    up_params.M = n_tokens;
                    up_params.N = ffn_size;
                    up_params.K = hidden_size;
                    up_params.lda = hidden_size;
                    up_params.ldb = ffn_size;
                    up_params.ldc = ffn_size;
                    up_params.n_batches = 1;
                    up_params.B_type = ffn_up_w->type;
                    backend->mul_mat_q(up_params);
                }
            }

            backend->sync();

            // SwiGLU: gate * silu(up)
            SiluParams silu_params;
            silu_params.input = ffn_up.data();
            silu_params.output = ffn_silu.data();
            silu_params.size = ffn_size;
            backend->silu(silu_params);

            for (int i = 0; i < ffn_size; i++) {
                ffn_silu[i] *= ffn_gate[i];
            }

            // FFN down projection
            if (ffn_down_w) {
                std::fill(ffn_down.begin(), ffn_down.end(), 0.0f);
                const int8_t* decoded_down = weight_cache.get_or_decode(ffn_down_w->name,
                    ffn_down_w->data, ffn_down_w->data_size(), ffn_down_w->type);
                if (decoded_down) {
                    MulMatParams down_params;
                    down_params.A = ffn_silu.data();
                    down_params.B = decoded_down;
                    down_params.C = ffn_down.data();
                    down_params.M = n_tokens;
                    down_params.N = hidden_size;
                    down_params.K = ffn_size;
                    down_params.lda = ffn_size;
                    down_params.ldb = hidden_size;
                    down_params.ldc = hidden_size;
                    down_params.n_batches = 1;
                    down_params.B_type = ffn_down_w->type;
                    backend->mul_mat_q(down_params);
                    backend->sync();

                    // Add residual
                    for (int i = 0; i < hidden_size; i++) {
                        inp_embd[i] += ffn_down[i];
                    }
                }
            }
        }

        // Final normalization
        rms_norm(inp_norm.data(), inp_embd.data(), hidden_size, rms_eps);

        // Logits projection: inp_norm @ tok_embd.T (or output.weight if present)
        std::fill(logits.begin(), logits.end(), 0.0f);
        const TensorView* tok_embd_out = find_tensor(model, "output.weight");
        const float* embd_ptr = embd_data;
        const float* logits_ptr = tok_embd_out ? get_float_ptr(tok_embd_out) : embd_ptr;

        if (logits_ptr && embd_ptr) {
            int vocab_size = static_cast<int>(hparams.vocab_size);
            for (int t = 0; t < n_tokens; t++) {
                for (int v = 0; v < vocab_size; v++) {
                    float sum = 0.0f;
                    for (int d = 0; d < hidden_size; d++) {
                        sum += inp_norm[t * hidden_size + d] * logits_ptr[v * hidden_size + d];
                    }
                    logits[v] += sum;
                }
            }
        }

        backend->sync();

        // Sample next token from last position
        int next_token = sample_token(logits, params.temp, rng_seed);
        std::string decoded = tokenizer.decode(next_token);

        if (first_token) {
            std::cout << decoded;
            first_token = false;
        } else {
            std::cout << decoded;
        }
        std::cout.flush();

        current_tokens.clear();
        current_tokens.push_back(next_token);
        pos += n_tokens;
        generated++;

        if (next_token == tokenizer.eos_token_id() || next_token == 0) break;
        if (pos >= ctx_size) break;
    }

    std::cout << "\n\nGenerated " << generated << " tokens.\n";

    model.unload();
    return 0;
}
