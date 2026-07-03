// Cross-check i-quant decoders against a trusted Q8_0 decode of the SAME tensor
// from the SAME model (different quant file). Correct decoders should reconstruct
// nearly-identical float weights, so per-row cosine similarity should be ~0.99+.
#include "gguf.h"
#include "quant/quant.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

using namespace ggnpu;

// Reconstruct per-row float weights from a decode_for_npu result (int8 + per-row scale).
static std::vector<float> reconstruct(const GgufLoader& g, const GgufTensorInfo& t) {
    auto span = g.tensor_data();
    const uint8_t* base = span.data() + t.data_offset;
    int64_t n_cols = static_cast<int64_t>(t.dims[0]);
    int64_t n_rows = t.n_dims > 1 ? static_cast<int64_t>(t.dims[1]) : 1;
    std::vector<int8_t> q; std::vector<float> sc;
    decode_for_npu(t.type, base, t.data_size, n_rows, n_cols, q, sc);
    std::vector<float> out(q.size());
    bool per_row = sc.size() == static_cast<size_t>(n_rows);
    for (int64_t r = 0; r < n_rows; r++) {
        float s = per_row ? sc[r] : sc[0];
        for (int64_t c = 0; c < n_cols; c++)
            out[r*n_cols + c] = q[r*n_cols + c] * s;
    }
    return out;
}

static const GgufTensorInfo* find(const GgufLoader& g, const std::string& n) {
    for (auto& t : g.tensors()) if (t.name == n) return &t;
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <iq.gguf> <ref.gguf> <tensor>\n", argv[0]); return 2; }
    GgufLoader iq, ref;
    if (!iq.load(argv[1])) { printf("load fail %s\n", argv[1]); return 1; }
    if (!ref.load(argv[2])) { printf("load fail %s\n", argv[2]); return 1; }
    std::string tname = argv[3];
    const GgufTensorInfo *ti = find(iq, tname), *tr = find(ref, tname);
    if (!ti || !tr) { printf("tensor not found\n"); return 1; }
    printf("tensor %s  iq_type=%s ref_type=%s dims=[%llu,%llu]\n", tname.c_str(),
           ggml_type_name(ti->type), ggml_type_name(tr->type),
           (unsigned long long)ti->dims[0], (unsigned long long)(ti->n_dims>1?ti->dims[1]:1));

    auto a = reconstruct(iq, *ti);
    auto b = reconstruct(ref, *tr);
    if (a.size() != b.size()) { printf("size mismatch %zu vs %zu\n", a.size(), b.size()); return 1; }

    // Global cosine similarity + relative RMS error.
    double dot=0, na=0, nb=0, se=0, sb=0;
    for (size_t i=0;i<a.size();i++){ dot+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i];
        double e=(double)a[i]-b[i]; se+=e*e; sb+=(double)b[i]*b[i]; }
    double cos = dot/(std::sqrt(na)*std::sqrt(nb)+1e-12);
    double rrms = std::sqrt(se/(sb+1e-12));
    printf("  cosine=%.5f  rel_rms_err=%.4f  (n=%zu)\n", cos, rrms, a.size());
    printf("  %s\n", cos>0.9 ? "PLAUSIBLE (decoder likely correct)" : "SUSPECT (decoder likely wrong)");
    return 0;
}
