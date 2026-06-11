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
    bool bench_layer = false;
    int bench_layer_num = 0;
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
    std::cout << "  bench-layer [options]              Validate one decoder layer on NPU vs CPU ref\n";
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
    std::cout << "      --layer <n>                    Layer number for bench-layer (default: 0)\n";
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
        } else if (arg == "bench-layer") {
            params.bench_layer = true;
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
        } else if (arg == "--layer") {
            if (i + 1 < argc) params.bench_layer_num = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_help();
            exit(1);
        }
    }

    return params;
}

const char* status_name(Status status) {
    switch (status) {
    case Status::OK:
        return "OK";
    case Status::ERROR:
        return "ERROR";
    case Status::NOT_FOUND:
        return "NOT_FOUND";
    case Status::INVALID_PARAM:
        return "INVALID_PARAM";
    case Status::NPU_UNAVAILABLE:
        return "NPU_UNAVAILABLE";
    case Status::OUT_OF_MEMORY:
        return "OUT_OF_MEMORY";
    default:
        return "UNKNOWN";
    }
}

struct CompareResult {
    float max_diff = 0.0f;
    float max_ref = 0.0f;
    int mismatches = 0;
};

CompareResult compare_vectors(const float* ref, const float* actual, int n, float mismatch_thresh) {
    CompareResult r;
    for (int i = 0; i < n; i++) {
        float diff = std::fabs(ref[i] - actual[i]);
        if (diff > r.max_diff) r.max_diff = diff;
        if (std::fabs(ref[i]) > r.max_ref) r.max_ref = std::fabs(ref[i]);
        if (diff > mismatch_thresh) r.mismatches++;
    }
    return r;
}

bool report_compare(const char* label, const CompareResult& r, int n,
                    float rel_thresh, float mismatch_frac, float mismatch_thresh) {
    float rel_error = r.max_ref > 0.0f ? r.max_diff / r.max_ref : r.max_diff;
    std::cout << "  " << label << ":\n";
    std::cout << "    Max absolute diff: " << r.max_diff << "\n";
    if (r.max_ref > 0.0f) std::cout << "    Max ref value: " << r.max_ref << "\n";
    std::cout << "    Relative error: " << rel_error << "\n";
    std::cout << "    Mismatches (>" << mismatch_thresh << "): " << r.mismatches << " / " << n << "\n";
    if (rel_error < rel_thresh && r.mismatches < static_cast<int>(n * mismatch_frac)) {
        std::cout << "    Result: PASS\n\n";
        return true;
    }
    std::cout << "    Result: FAIL\n\n";
    return false;
}

void attach_kquant_scales(MulMatParams& p, const TensorView* w, WeightCache& cache) {
    if (w && (w->type == GgmlType::Q4_K || w->type == GgmlType::Q6_K)) {
        const auto& scales = cache.get_scales(*w);
        if (!scales.empty()) {
            p.scales = scales.data();
            p.n_weight_scales = static_cast<int>(scales.size());
        }
    }
}

// Compare NPU decoded matmul vs CPU on-the-fly dequant matmul.
bool bench_quant_matmul(const char* label, Backend& cpu, Backend& npu, WeightCache& cache,
                        const float* A, const TensorView* weight,
                        int M, int N, int K, int lda, int ldb, int ldc,
                        float rel_thresh = 0.1f, float mismatch_thresh = 2.0f) {
    if (!weight) {
        std::cerr << "Error: weight tensor missing for " << label << "\n";
        return false;
    }

    const int8_t* decoded = cache.get_or_decode(*weight);
    if (!decoded) {
        std::cerr << "Error: failed to decode " << weight->name << "\n";
        return false;
    }

    std::vector<float> cpu_out(static_cast<size_t>(M) * N, 0.0f);
    MulMatParams cpu_p;
    cpu_p.A = A; cpu_p.B = weight->data; cpu_p.C = cpu_out.data();
    cpu_p.M = M; cpu_p.N = N; cpu_p.K = K;
    cpu_p.lda = lda; cpu_p.ldb = ldb; cpu_p.ldc = ldc;
    cpu_p.B_type = weight->type;
    cpu.mul_mat_q(cpu_p);

    std::vector<float> npu_out(static_cast<size_t>(M) * N, 0.0f);
    MulMatParams npu_p;
    npu_p.A = A; npu_p.B = decoded; npu_p.C = npu_out.data();
    npu_p.M = M; npu_p.N = N; npu_p.K = K;
    npu_p.lda = lda; npu_p.ldb = ldb; npu_p.ldc = ldc;
    npu_p.B_type = weight->type;
    attach_kquant_scales(npu_p, weight, cache);
    Status st = npu.mul_mat_q(npu_p);
    npu.sync();
    if (st != Status::OK) {
        std::cerr << "Error: NPU matmul failed for " << label << ": " << status_name(st) << "\n";
        return false;
    }

    auto cmp = compare_vectors(cpu_out.data(), npu_out.data(), M * N, mismatch_thresh);
    std::cout << "  " << label << " (" << M << " x " << K << " x " << N << "):\n";
    std::cout << "    Type: " << ggml_type_name(weight->type) << "\n";
    return report_compare(label, cmp, M * N, rel_thresh, 0.1f, mismatch_thresh);
}

bool validate_matmul_output(const std::vector<float>& output, int expected_value,
                           float tolerance, std::string* error_message) {
    for (size_t i = 0; i < output.size(); i++) {
        float actual = output[i];
        float expected = static_cast<float>(expected_value);
        if (std::fabs(actual - expected) > tolerance) {
            if (error_message) {
                std::ostringstream oss;
                oss << "output[" << i << "] expected " << expected << ", got " << actual;
                *error_message = oss.str();
            }
            return false;
        }
    }

    return true;
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

            Status warmup_status = backend->mul_mat_q(p);
            backend->sync();
            if (warmup_status != Status::OK) {
                std::cerr << "Matmul warmup failed for " << M << "x" << K << " x "
                          << K << "x" << N << ": " << status_name(warmup_status) << "\n";
                return 1;
            }

            std::string validation_error;
            if (!validate_matmul_output(C, K, 1e-3f, &validation_error)) {
                std::cerr << "Matmul validation failed for " << M << "x" << K << " x "
                          << K << "x" << N << ": " << validation_error << "\n";
                return 1;
            }

            auto start = std::chrono::high_resolution_clock::now();
            int iterations = 10;
            for (int i = 0; i < iterations; i++) {
                Status st = backend->mul_mat_q(p);
                backend->sync();
                if (st != Status::OK) {
                    std::cerr << "Matmul benchmark iteration failed for " << M << "x" << K
                              << " x " << K << "x" << N << ": " << status_name(st) << "\n";
                    return 1;
                }
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

    // Cap context to avoid multi-GB KV allocation on models with huge metadata ctx
    if (params.ctx_size > 0) {
        model.set_context_length(static_cast<uint64_t>(params.ctx_size));
        std::cout << "  Context size overridden to: " << params.ctx_size << "\n";
        // Reinitialize KV cache to respect overridden context size
        if (!model.reinit_kv_cache(params.ctx_size)) {
            std::cerr << "  Warning: failed to reinitialize KV cache with overridden context size\n";
        }
    } else if (hparams.context_length > 2048) {
        model.set_context_length(2048);
        std::cout << "  Context capped to 2048 (model reports "
                  << hparams.context_length << "; use -c to override)\n";
    }
    std::cout << "  Architecture: " << model.gguf().architecture() << "\n";
    std::cout << "  Context: " << hparams.context_length << "\n";
    std::cout << "  Layers: " << hparams.block_count << "\n";
    std::cout << "  Hidden: " << hparams.embedding_length << "\n";
    std::cout << "  Heads: " << hparams.attention_head_count << " (KV: " << hparams.attention_head_count_kv << ")\n";
    std::cout << "  FFN: " << hparams.feed_forward_length << "\n\n";

    // bench-layer generates its own test input — skip the prompt requirement
    if (params.prompt.empty() && !params.bench_layer) {
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

    // Setup working buffers
    int hidden_size = static_cast<int>(hparams.embedding_length);
    int ffn_size = static_cast<int>(hparams.feed_forward_length);
    int num_layers = static_cast<int>(hparams.block_count);
    int num_heads = static_cast<int>(hparams.attention_head_count);
    int num_kv_heads = static_cast<int>(hparams.attention_head_count_kv);
    int head_dim = static_cast<int>(hparams.rope_dimension_count);
    int ctx_size = static_cast<int>(model.hparams().context_length);
    float rms_eps = 1e-5f;
    if (hparams.attention_layer_norm_rms_epsilon > 0) {
        rms_eps = static_cast<float>(hparams.attention_layer_norm_rms_epsilon);
    }
    float rope_freq_scale = static_cast<float>(hparams.rope_freq_scale);
    float rope_freq_base = static_cast<float>(hparams.rope_freq_base);

    // Working buffers for activations (resized per iteration for prefill/decode)
    std::vector<float> inp_embd;
    std::vector<float> inp_norm;
    std::vector<float> q_proj;
    std::vector<float> k_proj;
    std::vector<float> v_proj;
    std::vector<float> k_rope;
    std::vector<float> q_rope;
    std::vector<float> attn_output;
    std::vector<float> ffn_gate;
    std::vector<float> ffn_up;
    std::vector<float> ffn_down;
    std::vector<float> ffn_silu;
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

    auto rms_norm = [&](float* out, const float* inp, int n_rows, int row_size, float eps,
                        const float* weight) {
        for (int row = 0; row < n_rows; row++) {
            RmsNormParams rms_params;
            rms_params.input = inp + row * row_size;
            rms_params.output = out + row * row_size;
            rms_params.size = row_size;
            rms_params.eps = eps;
            rms_params.weight = weight;
            backend->rms_norm(rms_params);
        }
    };

    auto mul_mat_weight = [&](const float* A, const TensorView* weight, float* C,
                              int M, int N, int K, int lda, int ldb, int ldc) {
        if (!weight) return;
        const int8_t* decoded = weight_cache.get_or_decode(*weight);
        if (!decoded) return;

        MulMatParams mat_params;
        mat_params.A = A;
        mat_params.B = decoded;
        mat_params.C = C;
        mat_params.M = M;
        mat_params.N = N;
        mat_params.K = K;
        mat_params.lda = lda;
        mat_params.ldb = ldb;
        mat_params.ldc = ldc;
        mat_params.n_batches = 1;
        mat_params.B_type = weight->type;
        attach_kquant_scales(mat_params, weight, weight_cache);
        backend->mul_mat_q(mat_params);
    };

    //====//
    // bench-layer: Validate one decoder layer on NPU vs CPU reference
    //====//
    if (params.bench_layer) {
        std::cout << "GGNPU Layer Benchmark\n";
        std::cout << "=====================\n\n";

        std::shared_ptr<Backend> npu_backend;
#ifdef GGNPU_HAS_NPU_BACKEND
        npu_backend = create_amd_xdna_backend(params.npu_device);
        if (!npu_backend || !npu_backend->is_available()) {
            std::cerr << "Error: NPU backend unavailable for bench-layer\n";
            return 1;
        }
#else
        std::cerr << "Error: NPU backend not compiled in. Build with -DGGNPU_NPU_BACKEND=ON\n";
        return 1;
#endif

        std::shared_ptr<Backend> cpu_backend = create_cpu_ref_backend();

        int layer_idx = params.bench_layer_num;
        if (layer_idx < 0 || layer_idx >= num_layers) {
            std::cerr << "Error: layer " << layer_idx << " out of range [0, "
                      << num_layers - 1 << "]\n";
            return 1;
        }

        std::cout << "Model: " << params.model_path << "\n";
        std::cout << "Layer: " << layer_idx << "\n";
        std::cout << "Hidden: " << hidden_size << "\n";
        std::cout << "FFN size: " << ffn_size << "\n\n";

        // Single test token activations (random but deterministic)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> test_input(static_cast<size_t>(hidden_size));
        for (size_t i = 0; i < test_input.size(); i++) {
            test_input[i] = dist(rng);
        }

        // RMSNorm reference (CPU)
        std::vector<float> normed(hidden_size);
        RmsNormParams rms_params;
        rms_params.input = test_input.data();
        rms_params.output = normed.data();
        rms_params.size = hidden_size;
        rms_params.eps = rms_eps;
        cpu_backend->rms_norm(rms_params);

        Status st;

        // Test 0a: RMSNorm — NPU vs CPU (Phase 4)
        std::cout << "Testing RMSNorm (Phase 4)...\n";
        {
            std::vector<float> npu_normed(static_cast<size_t>(hidden_size));
            RmsNormParams npu_rms;
            npu_rms.input = test_input.data();
            npu_rms.output = npu_normed.data();
            npu_rms.size = hidden_size;
            npu_rms.eps = rms_eps;
            st = npu_backend->rms_norm(npu_rms);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU rms_norm failed: " << status_name(st) << "\n";
                return 1;
            }
            auto cmp = compare_vectors(normed.data(), npu_normed.data(), hidden_size, 0.01f);
            if (hidden_size != 256) {
                std::cout << "    Note: prebuilt rmsnorm xclbin is 32x256; hidden="
                          << hidden_size << " uses CPU fallback on NPU path\n";
            }
            if (!report_compare("RMSNorm", cmp, hidden_size, 0.05f, 0.1f, 0.01f)) return 1;
        }

        // Test 0b–0e: attention matmuls — NPU vs CPU (Phase 4)
        const int kv_dim = num_kv_heads * head_dim;
        std::cout << "Testing attention matmuls (Phase 4)...\n";
        {
            const TensorView* attn_q_w = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", layer_idx);
            const TensorView* attn_k_w = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", layer_idx);
            const TensorView* attn_v_w = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", layer_idx);
            const TensorView* attn_o_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", layer_idx);
            if (!attn_q_w || !attn_k_w || !attn_v_w || !attn_o_w) {
                std::cerr << "Error: attention weights not found for layer " << layer_idx << "\n";
                return 1;
            }
            if (!bench_quant_matmul("attn_q matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), attn_q_w, 1, hidden_size, hidden_size,
                    hidden_size, hidden_size, hidden_size)) return 1;
            if (!bench_quant_matmul("attn_k matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), attn_k_w, 1, kv_dim, hidden_size,
                    hidden_size, kv_dim, kv_dim)) return 1;
            if (!bench_quant_matmul("attn_v matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), attn_v_w, 1, kv_dim, hidden_size,
                    hidden_size, kv_dim, kv_dim)) return 1;
            if (!bench_quant_matmul("attn_output matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), attn_o_w, 1, hidden_size, hidden_size,
                    hidden_size, hidden_size, hidden_size)) return 1;
        }

        // Test 0f: flash attention — NPU vs CPU (Phase 4, single head, ctx=1)
        std::cout << "Testing flash attention (Phase 4)...\n";
        {
            std::vector<float> Q(static_cast<size_t>(head_dim), 0.1f);
            std::vector<float> K(static_cast<size_t>(head_dim), 0.2f);
            std::vector<float> V(static_cast<size_t>(head_dim), 0.3f);
            std::vector<float> cpu_fa(static_cast<size_t>(head_dim), 0.0f);
            std::vector<float> npu_fa(static_cast<size_t>(head_dim), 0.0f);

            AttnParams ap;
            ap.Q = Q.data(); ap.K = K.data(); ap.V = V.data();
            ap.n_head = 1; ap.head_dim = head_dim; ap.ctx_len = 1;
            ap.batch_size = 1; ap.freq_factors = nullptr;

            ap.output = cpu_fa.data();
            cpu_backend->flash_attn(ap);
            ap.output = npu_fa.data();
            st = npu_backend->flash_attn(ap);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU flash_attn failed: " << status_name(st) << "\n";
                return 1;
            }
            auto cmp = compare_vectors(cpu_fa.data(), npu_fa.data(), head_dim, 0.05f);
            if (!report_compare("flash_attn (1 head, ctx=1)", cmp, head_dim, 0.1f, 0.2f, 0.05f)) return 1;
        }

        // Test 0c: SiLU — NPU vs CPU (Phase 4)
        std::cout << "Testing SiLU (Phase 4)...\n";
        {
            std::vector<float> silu_in(static_cast<size_t>(ffn_size));
            for (int i = 0; i < ffn_size; i++) silu_in[i] = test_input[i % hidden_size];

            std::vector<float> cpu_silu(static_cast<size_t>(ffn_size));
            SiluParams cpu_p;
            cpu_p.input = silu_in.data();
            cpu_p.output = cpu_silu.data();
            cpu_p.size = ffn_size;
            cpu_backend->silu(cpu_p);

            std::vector<float> npu_silu(static_cast<size_t>(ffn_size));
            SiluParams npu_p;
            npu_p.input = silu_in.data();
            npu_p.output = npu_silu.data();
            npu_p.size = ffn_size;
            st = npu_backend->silu(npu_p);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU silu failed: " << status_name(st) << "\n";
                return 1;
            }

            auto cmp = compare_vectors(cpu_silu.data(), npu_silu.data(), ffn_size, 0.05f);
            if (!report_compare("SiLU", cmp, ffn_size, 0.05f, 0.1f, 0.05f)) return 1;
        }

        std::cout << "Testing FFN gate matmul (Phase 3 E2E gate)...\n";
        {
            const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer_idx);
            if (!bench_quant_matmul("FFN gate matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), ffn_gate_w, 1, ffn_size, hidden_size,
                    hidden_size, ffn_size, ffn_size)) return 1;
        }

        // Test 2: FFN up matmul
        {
            const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer_idx);
            if (!bench_quant_matmul("FFN up matmul", *cpu_backend, *npu_backend, weight_cache,
                    normed.data(), ffn_up_w, 1, ffn_size, hidden_size,
                    hidden_size, ffn_size, ffn_size)) return 1;
        }

        // Test 3: FFN down matmul (silu(gate) * up) -> hidden_size
        {
            const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer_idx);
            const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer_idx);
            const TensorView* ffn_down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", layer_idx);

            if (!ffn_gate_w || !ffn_up_w || !ffn_down_w) {
                std::cerr << "Error: FFN weights not found for layer " << layer_idx << "\n";
                return 1;
            }

            // Compute gate and up on NPU
            const int8_t* decoded_gate = weight_cache.get_or_decode(*ffn_gate_w);
            const int8_t* decoded_up = weight_cache.get_or_decode(*ffn_up_w);

            std::vector<float> gate_out(static_cast<size_t>(ffn_size), 0.0f);
            std::vector<float> up_out(static_cast<size_t>(ffn_size), 0.0f);

            MulMatParams gate_params;
            gate_params.A = normed.data();
            gate_params.B = decoded_gate;
            gate_params.C = gate_out.data();
            gate_params.M = 1;
            gate_params.N = ffn_size;
            gate_params.K = hidden_size;
            gate_params.lda = hidden_size;
            gate_params.ldb = ffn_size;
            gate_params.ldc = ffn_size;
            gate_params.n_batches = 1;
            gate_params.B_type = ffn_gate_w->type;
            attach_kquant_scales(gate_params, ffn_gate_w, weight_cache);
            npu_backend->mul_mat_q(gate_params);

            MulMatParams up_params;
            up_params.A = normed.data();
            up_params.B = decoded_up;
            up_params.C = up_out.data();
            up_params.M = 1;
            up_params.N = ffn_size;
            up_params.K = hidden_size;
            up_params.lda = hidden_size;
            up_params.ldb = ffn_size;
            up_params.ldc = ffn_size;
            up_params.n_batches = 1;
            up_params.B_type = ffn_up_w->type;
            attach_kquant_scales(up_params, ffn_up_w, weight_cache);
            npu_backend->mul_mat_q(up_params);
            npu_backend->sync();

            // SwiGLU: silu(gate) * up
            std::vector<float> swiglu(static_cast<size_t>(ffn_size));
            for (int i = 0; i < ffn_size; i++) {
                float gate_silu = gate_out[i] / (1.0f + std::exp(-gate_out[i]));
                swiglu[i] = gate_silu * up_out[i];
            }

            // FFN down matmul on NPU
            const int8_t* decoded_down = weight_cache.get_or_decode(*ffn_down_w);

            std::vector<float> npu_output(static_cast<size_t>(hidden_size), 0.0f);
            MulMatParams down_params;
            down_params.A = swiglu.data();
            down_params.B = decoded_down;
            down_params.C = npu_output.data();
            down_params.M = 1;
            down_params.N = hidden_size;
            down_params.K = ffn_size;
            down_params.lda = ffn_size;
            down_params.ldb = hidden_size;
            down_params.ldc = hidden_size;
            down_params.n_batches = 1;
            down_params.B_type = ffn_down_w->type;
            attach_kquant_scales(down_params, ffn_down_w, weight_cache);

            st = npu_backend->mul_mat_q(down_params);
            npu_backend->sync();
            if (st != Status::OK) {
                std::cerr << "Error: NPU matmul failed: " << status_name(st) << "\n";
                return 1;
            }

            // CPU reference for FFN down
            std::vector<float> cpu_output(static_cast<size_t>(hidden_size), 0.0f);
            MulMatParams cpu_down_params;
            cpu_down_params.A = swiglu.data();
            cpu_down_params.B = ffn_down_w->data;
            cpu_down_params.C = cpu_output.data();
            cpu_down_params.M = 1;
            cpu_down_params.N = hidden_size;
            cpu_down_params.K = ffn_size;
            cpu_down_params.lda = ffn_size;
            cpu_down_params.ldb = hidden_size;
            cpu_down_params.ldc = hidden_size;
            cpu_down_params.n_batches = 1;
            cpu_down_params.B_type = ffn_down_w->type;
            cpu_backend->mul_mat_q(cpu_down_params);

            float max_diff = 0.0f;
            float max_cpu = 0.0f;
            int mismatches = 0;
            for (int i = 0; i < hidden_size; i++) {
                float diff = std::fabs(cpu_output[i] - npu_output[i]);
                if (diff > max_diff) max_diff = diff;
                if (std::fabs(cpu_output[i]) > max_cpu) max_cpu = std::fabs(cpu_output[i]);
                if (diff > 2.0f) mismatches++;
            }

            float rel_error = max_cpu > 0.0f ? max_diff / max_cpu : max_diff;
            std::cout << "  FFN down matmul (SwiGLU, 1 x " << ffn_size << " x " << hidden_size << "):\n";
            std::cout << "    Type: " << ggml_type_name(ffn_down_w->type) << "\n";
            std::cout << "    Max absolute diff: " << max_diff << "\n";
            std::cout << "    Relative error: " << rel_error << "\n";
            std::cout << "    Mismatches (>2.0): " << mismatches << " / " << hidden_size << "\n";

            if (rel_error < 0.1f && mismatches < hidden_size / 10) {
                std::cout << "    Result: PASS\n\n";
            } else {
                std::cout << "    Result: FAIL\n\n";
                return 1;
            }
        }

        // Test 4: full decoder layer forward — CPU vs NPU (Phase 4)
        std::cout << "Testing full layer forward (Phase 4)...\n";
        {
            auto run_layer = [&](Backend* backend, bool use_npu_weights,
                                 const float* in, float* out) -> bool {
                const TensorView* attn_q_w = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", layer_idx);
                const TensorView* attn_k_w = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", layer_idx);
                const TensorView* attn_v_w = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", layer_idx);
                const TensorView* attn_o_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", layer_idx);
                const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer_idx);
                const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer_idx);
                const TensorView* ffn_down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", layer_idx);
                if (!attn_q_w || !attn_k_w || !attn_v_w || !attn_o_w ||
                    !ffn_gate_w || !ffn_up_w || !ffn_down_w) return false;

                std::vector<float> hidden(static_cast<size_t>(hidden_size));
                std::memcpy(hidden.data(), in, hidden_size * sizeof(float));

                auto do_matmul = [&](const float* A, const TensorView* w, float* C,
                                     int M, int N, int K, int lda, int ldb, int ldc) -> bool {
                    MulMatParams mp;
                    mp.A = A; mp.C = C;
                    mp.M = M; mp.N = N; mp.K = K;
                    mp.lda = lda; mp.ldb = ldb; mp.ldc = ldc;
                    mp.n_batches = 1;
                    if (use_npu_weights) {
                        const int8_t* dec = weight_cache.get_or_decode(*w);
                        if (!dec) return false;
                        mp.B = dec;
                        mp.B_type = w->type;
                        attach_kquant_scales(mp, w, weight_cache);
                    } else {
                        mp.B = w->data;
                        mp.B_type = w->type;
                    }
                    return backend->mul_mat_q(mp) == Status::OK;
                };

                const TensorView* attn_norm_w =
                    find_tensor_pattern(model, "blk.{layer}.attn_norm.weight", layer_idx);
                const TensorView* ffn_norm_w =
                    find_tensor_pattern(model, "blk.{layer}.ffn_norm.weight", layer_idx);

                // Attention branch
                std::vector<float> norm(static_cast<size_t>(hidden_size));
                RmsNormParams rp;
                rp.input = hidden.data(); rp.output = norm.data();
                rp.size = hidden_size; rp.eps = rms_eps;
                rp.weight = get_float_ptr(attn_norm_w);
                if (backend->rms_norm(rp) != Status::OK) return false;

                std::vector<float> q(hidden_size), k(kv_dim), v(kv_dim);
                std::vector<float> q_r(hidden_size), k_r(kv_dim);
                if (!do_matmul(norm.data(), attn_q_w, q.data(), 1, hidden_size, hidden_size,
                               hidden_size, hidden_size, hidden_size)) return false;
                if (!do_matmul(norm.data(), attn_k_w, k.data(), 1, kv_dim, hidden_size,
                               hidden_size, kv_dim, kv_dim)) return false;
                if (!do_matmul(norm.data(), attn_v_w, v.data(), 1, kv_dim, hidden_size,
                               hidden_size, kv_dim, kv_dim)) return false;
                backend->sync();

                apply_rope(q_r.data(), q.data(), num_heads, 0, head_dim, rope_freqs);
                apply_rope(k_r.data(), k.data(), num_kv_heads, 0, head_dim, rope_freqs);

                std::vector<float> attn_out(hidden_size, 0.0f);
                for (int h = 0; h < num_heads; h++) {
                    int kv_h = h / (num_heads / num_kv_heads);
                    AttnParams ap;
                    ap.Q = q_r.data() + h * head_dim;
                    ap.K = k_r.data() + kv_h * head_dim;
                    ap.V = v.data() + kv_h * head_dim;
                    ap.output = attn_out.data() + h * head_dim;
                    ap.n_head = 1; ap.head_dim = head_dim; ap.ctx_len = 1;
                    ap.batch_size = 1; ap.freq_factors = nullptr;
                    if (backend->flash_attn(ap) != Status::OK) return false;
                }
                backend->sync();

                std::vector<float> attn_proj(hidden_size, 0.0f);
                if (!do_matmul(attn_out.data(), attn_o_w, attn_proj.data(),
                               1, hidden_size, hidden_size,
                               hidden_size, hidden_size, hidden_size)) return false;
                backend->sync();
                for (int i = 0; i < hidden_size; i++) hidden[i] += attn_proj[i];

                // FFN branch
                rp.input = hidden.data();
                rp.output = norm.data();
                rp.weight = get_float_ptr(ffn_norm_w);
                if (backend->rms_norm(rp) != Status::OK) return false;

                std::vector<float> gate(ffn_size), up(ffn_size), swiglu(ffn_size);
                if (!do_matmul(norm.data(), ffn_gate_w, gate.data(), 1, ffn_size, hidden_size,
                               hidden_size, ffn_size, ffn_size)) return false;
                if (!do_matmul(norm.data(), ffn_up_w, up.data(), 1, ffn_size, hidden_size,
                               hidden_size, ffn_size, ffn_size)) return false;
                backend->sync();

                SiluParams sp;
                sp.input = gate.data(); sp.output = swiglu.data(); sp.size = ffn_size;
                if (backend->silu(sp) != Status::OK) return false;
                for (int i = 0; i < ffn_size; i++) swiglu[i] *= up[i];

                std::vector<float> ffn_out(hidden_size, 0.0f);
                if (!do_matmul(swiglu.data(), ffn_down_w, ffn_out.data(),
                               1, hidden_size, ffn_size,
                               ffn_size, hidden_size, hidden_size)) return false;
                backend->sync();
                for (int i = 0; i < hidden_size; i++) hidden[i] += ffn_out[i];

                std::memcpy(out, hidden.data(), hidden_size * sizeof(float));
                return true;
            };

            std::vector<float> cpu_layer(hidden_size), npu_layer(hidden_size);
            if (!run_layer(cpu_backend.get(), false, test_input.data(), cpu_layer.data())) {
                std::cerr << "Error: CPU layer forward failed\n";
                return 1;
            }
            if (!run_layer(npu_backend.get(), true, test_input.data(), npu_layer.data())) {
                std::cerr << "Error: NPU layer forward failed\n";
                return 1;
            }
            npu_backend->sync();

            auto cmp = compare_vectors(cpu_layer.data(), npu_layer.data(), hidden_size, 2.0f);
            if (!report_compare("full layer forward", cmp, hidden_size, 0.15f, 0.2f, 2.0f)) return 1;
        }

        std::cout << "All layer tests PASSED.\n";
        model.unload();
        return 0;
    }

    // Dequantize token embeddings once before the generation loop
    const TensorView* tok_embd = find_tensor(model, "token_embd.weight");
    std::vector<float> tok_embd_f32;
    std::vector<float> output_w_f32;
    const float* embd_data = nullptr;
    if (tok_embd) {
        if (tok_embd->type == GgmlType::F32) {
            embd_data = get_float_ptr(tok_embd);
        } else {
            const int8_t* decoded = weight_cache.get_or_decode(*tok_embd);
            if (decoded) {
                int64_t embd_dim = static_cast<int64_t>(hparams.embedding_length);
                int64_t vocab_size = static_cast<int64_t>(hparams.vocab_size);
                const auto& embd_scales = weight_cache.get_scales(*tok_embd);
                const bool per_row = embd_scales.size() > 1;
                tok_embd_f32.resize(static_cast<size_t>(vocab_size * embd_dim));
                for (int64_t i = 0; i < vocab_size; i++) {
                    float row_scale = embd_scales.empty() ? 1.0f
                        : (per_row ? embd_scales[static_cast<size_t>(i)] : embd_scales[0]);
                    for (int64_t d = 0; d < embd_dim; d++) {
                        tok_embd_f32[static_cast<size_t>(i * embd_dim + d)] =
                            static_cast<float>(decoded[static_cast<size_t>(i * embd_dim + d)]) *
                            row_scale;
                    }
                }
                embd_data = tok_embd_f32.data();
                std::cout << "  Dequantized token_embd.weight ("
                    << ggml_type_name(tok_embd->type) << " -> F32)\n";
            }
        }
    }

    const TensorView* output_w = find_tensor(model, "output.weight");
    const float* output_w_ptr = embd_data;
    if (output_w) {
        if (output_w->type == GgmlType::F32) {
            output_w_ptr = get_float_ptr(output_w);
        } else {
            const int8_t* decoded = weight_cache.get_or_decode(*output_w);
            if (decoded) {
                int64_t embd_dim = static_cast<int64_t>(hparams.embedding_length);
                int64_t vocab_size = static_cast<int64_t>(hparams.vocab_size);
                const auto& out_scales = weight_cache.get_scales(*output_w);
                const bool per_row = out_scales.size() > 1;
                output_w_f32.resize(static_cast<size_t>(vocab_size * embd_dim));
                for (int64_t v = 0; v < vocab_size; v++) {
                    float row_scale = out_scales.empty() ? 1.0f
                        : (per_row ? out_scales[static_cast<size_t>(v)] : out_scales[0]);
                    for (int64_t d = 0; d < embd_dim; d++) {
                        output_w_f32[static_cast<size_t>(v * embd_dim + d)] =
                            static_cast<float>(decoded[static_cast<size_t>(v * embd_dim + d)]) *
                            row_scale;
                    }
                }
                output_w_ptr = output_w_f32.data();
                std::cout << "  Dequantized output.weight ("
                    << ggml_type_name(output_w->type) << " -> F32)\n";
            }
        }
    }

    // Main generation loop
    std::cout << "Generating: ";

    bool first_token = true;
    while (generated < params.max_tokens) {
        int n_tokens = static_cast<int>(current_tokens.size());
        if (n_tokens == 0) break;

        // Embedding lookup - one hidden vector per token
        if (!embd_data) {
            std::cerr << "Error: token_embd.weight not available\n";
            break;
        }
        int64_t embd_dim = static_cast<int64_t>(hparams.embedding_length);

        inp_embd.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        inp_norm.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        q_proj.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        k_proj.assign(static_cast<size_t>(n_tokens) * num_kv_heads * head_dim, 0.0f);
        v_proj.assign(static_cast<size_t>(n_tokens) * num_kv_heads * head_dim, 0.0f);
        k_rope.assign(static_cast<size_t>(n_tokens) * num_kv_heads * head_dim, 0.0f);
        q_rope.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        attn_output.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        ffn_gate.assign(static_cast<size_t>(n_tokens) * ffn_size, 0.0f);
        ffn_up.assign(static_cast<size_t>(n_tokens) * ffn_size, 0.0f);
        ffn_down.assign(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);
        ffn_silu.assign(static_cast<size_t>(n_tokens) * ffn_size, 0.0f);

        for (int t = 0; t < n_tokens; t++) {
            int tok_id = current_tokens[t];
            for (int d = 0; d < hidden_size; d++) {
                inp_embd[static_cast<size_t>(t) * hidden_size + d] =
                    embd_data[tok_id * embd_dim + d];
            }
        }

        // Transformer layers
        for (int layer = 0; layer < num_layers; layer++) {
            const TensorView* attn_norm_w =
                find_tensor_pattern(model, "blk.{layer}.attn_norm.weight", layer);
            const TensorView* ffn_norm_w =
                find_tensor_pattern(model, "blk.{layer}.ffn_norm.weight", layer);

            // RMSNorm for attention
            rms_norm(inp_norm.data(), inp_embd.data(), n_tokens, hidden_size, rms_eps,
                     get_float_ptr(attn_norm_w));

            // Attention projections
            const TensorView* attn_q = find_tensor_pattern(model, "blk.{layer}.attn_q.weight", layer);
            const TensorView* attn_k = find_tensor_pattern(model, "blk.{layer}.attn_k.weight", layer);
            const TensorView* attn_v = find_tensor_pattern(model, "blk.{layer}.attn_v.weight", layer);

            mul_mat_weight(inp_norm.data(), attn_q, q_proj.data(),
                           n_tokens, hidden_size, hidden_size,
                           hidden_size, hidden_size, hidden_size);
            mul_mat_weight(inp_norm.data(), attn_k, k_proj.data(),
                           n_tokens, num_kv_heads * head_dim, hidden_size,
                           hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim);
            mul_mat_weight(inp_norm.data(), attn_v, v_proj.data(),
                           n_tokens, num_kv_heads * head_dim, hidden_size,
                           hidden_size, num_kv_heads * head_dim, num_kv_heads * head_dim);

            backend->sync();

            // Apply RoPE to Q and K (per token row)
            for (int t = 0; t < n_tokens; t++) {
                int64_t token_pos = pos + t;
                apply_rope(q_rope.data() + t * hidden_size,
                           q_proj.data() + t * hidden_size,
                           num_heads, token_pos, head_dim, rope_freqs);
                apply_rope(k_rope.data() + t * num_kv_heads * head_dim,
                           k_proj.data() + t * num_kv_heads * head_dim,
                           num_kv_heads, token_pos, head_dim, rope_freqs);
            }

            // Store KV in cache (per token row)
            for (int t = 0; t < n_tokens; t++) {
                model.kv_cache().update_slab(layer, pos + t, pos + t + 1,
                                             k_rope.data() + t * num_kv_heads * head_dim,
                                             v_proj.data() + t * num_kv_heads * head_dim,
                                             num_kv_heads, head_dim);
            }

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

            std::vector<float> attn_out(static_cast<size_t>(n_tokens) * hidden_size, 0.0f);

            for (int t = 0; t < n_tokens; t++) {
                for (int h = 0; h < num_heads; h++) {
                    AttnParams fa_params;
                    fa_params.Q = q_rope.data() + t * hidden_size + h * head_dim;
                    fa_params.K = k_expanded.data() + h * ctx_len * head_dim;
                    fa_params.V = v_expanded.data() + h * ctx_len * head_dim;
                    fa_params.output = attn_out.data() + t * hidden_size + h * head_dim;
                    fa_params.batch_size = 1;
                    fa_params.n_head = 1;
                    fa_params.head_dim = head_dim;
                    fa_params.ctx_len = ctx_len;
                    fa_params.query_pos = pos + t;
                    fa_params.freq_factors = nullptr;
                    backend->flash_attn(fa_params);
                }
            }

            // Attention output projection
            const TensorView* attn_output_w = find_tensor_pattern(model, "blk.{layer}.attn_output.weight", layer);
            if (attn_output_w) {
                std::fill(attn_output.begin(), attn_output.end(), 0.0f);
                mul_mat_weight(attn_out.data(), attn_output_w, attn_output.data(),
                               n_tokens, hidden_size, hidden_size,
                               hidden_size, hidden_size, hidden_size);
                backend->sync();

                for (int t = 0; t < n_tokens; t++) {
                    for (int i = 0; i < hidden_size; i++) {
                        inp_embd[static_cast<size_t>(t) * hidden_size + i] +=
                            attn_output[static_cast<size_t>(t) * hidden_size + i];
                    }
                }
            }
            // NOTE: residual add on CPU — acceptable for MVP (see IMPLEMENTATION.md §7.1)

            // FFN path
            rms_norm(inp_norm.data(), inp_embd.data(), n_tokens, hidden_size, rms_eps,
                     get_float_ptr(ffn_norm_w));

            const TensorView* ffn_gate_w = find_tensor_pattern(model, "blk.{layer}.ffn_gate.weight", layer);
            const TensorView* ffn_up_w = find_tensor_pattern(model, "blk.{layer}.ffn_up.weight", layer);
            const TensorView* ffn_down_w = find_tensor_pattern(model, "blk.{layer}.ffn_down.weight", layer);

            mul_mat_weight(inp_norm.data(), ffn_gate_w, ffn_gate.data(),
                           n_tokens, ffn_size, hidden_size,
                           hidden_size, ffn_size, ffn_size);
            mul_mat_weight(inp_norm.data(), ffn_up_w, ffn_up.data(),
                           n_tokens, ffn_size, hidden_size,
                           hidden_size, ffn_size, ffn_size);

            backend->sync();

            // SwiGLU: silu(gate) * up per token row
            for (int t = 0; t < n_tokens; t++) {
                SiluParams silu_params;
                silu_params.input = ffn_gate.data() + t * ffn_size;
                silu_params.output = ffn_silu.data() + t * ffn_size;
                silu_params.size = ffn_size;
                backend->silu(silu_params);

                for (int i = 0; i < ffn_size; i++) {
                    ffn_silu[static_cast<size_t>(t) * ffn_size + i] *=
                        ffn_up[static_cast<size_t>(t) * ffn_size + i];
                }
            }

            // FFN down projection
            if (ffn_down_w) {
                std::fill(ffn_down.begin(), ffn_down.end(), 0.0f);
                mul_mat_weight(ffn_silu.data(), ffn_down_w, ffn_down.data(),
                               n_tokens, hidden_size, ffn_size,
                               ffn_size, hidden_size, hidden_size);
                backend->sync();

                for (int t = 0; t < n_tokens; t++) {
                    for (int i = 0; i < hidden_size; i++) {
                        inp_embd[static_cast<size_t>(t) * hidden_size + i] +=
                            ffn_down[static_cast<size_t>(t) * hidden_size + i];
                    }
                }
            }
        }

        // Final normalization (output_norm before logits)
        const TensorView* output_norm_w = find_tensor(model, "output_norm.weight");
        rms_norm(inp_norm.data(), inp_embd.data(), n_tokens, hidden_size, rms_eps,
                 get_float_ptr(output_norm_w));

        // Logits projection from the last token position
        // NOTE: CPU fallback — not yet routed through mul_mat_q (see IMPLEMENTATION.md §7.1)
        std::fill(logits.begin(), logits.end(), 0.0f);
        const float* logits_ptr = output_w_ptr;

        if (logits_ptr) {
            int vocab_size = static_cast<int>(hparams.vocab_size);
            const float* last_hidden = inp_norm.data() + static_cast<size_t>(n_tokens - 1) * hidden_size;
            for (int v = 0; v < vocab_size; v++) {
                float sum = 0.0f;
                for (int d = 0; d < hidden_size; d++) {
                    sum += last_hidden[d] * logits_ptr[v * hidden_size + d];
                }
                logits[v] = sum;
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
