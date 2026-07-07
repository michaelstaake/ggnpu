# Dev Benchmark Log

Running record of performance/correctness test runs, so we remember what was
measured, on what commit, and what the result was. **Append a new entry every
time we run a benchmark or timed test** — newest at the top. Do not edit or
delete old entries; they are the historical record.

Each entry records: commit, date, the exact command/data, and the result
(timings + correctness). Capture per-op timings with `-v` when relevant.

Hardware unless noted: AMD Ryzen AI 7 350 (Krackan / XDNA2 / `npu6`),
`/dev/accel/accel0`, Ubuntu.

---

## 2026-07-06 — Widen-N decode matmul (kWideN=512): ~10% off decode matmul

Decode is device-bound on INT8 matmul kernel execution (confirmed after the
A-pack hoist removed the 39% host pack, which had been hidden behind kernel
execution — per-tile total barely moved). At M<=16 the AIE array's M lanes are
mostly idle, so the array is partly fill/drain-latency-bound; a wider N tile per
launch rides that latency for near-free extra output and halves the launch count.

Built `matmul_small_m_deepk_wide` (M=16, N=512, K=2048) via Triton-XDNA (same
transform as small_m/deep-K; toolchain needs build-kernels.sh's LD_LIBRARY_PATH
for the in-repo boost-lib — `xclbinutil` links libboost_filesystem.so.1.90.0).
Added host routing: N-tile width is now a runtime value (kWideN when N%512==0 on
the deep-K path, else 256). Default-on; disable with GGNPU_NO_WIDEN_N.

**Result (Llama-1B Q4_K_M, `-p "The capital of France is" -n 32 -c 512 --temp 0`):**

| Path | decode matmul | total (37 steps) | Δ |
|------|---------------|------------------|---|
| GGNPU_NO_WIDEN_N (N=256) | 8167 ms | 10984 ms | baseline |
| default (N=512 widen) | 7327 ms | 10121 ms | **matmul −10.3%, total −7.9% (~+8.5% tok/s)** |

**N sweep:** N=256 → 8072 ms, **N=512 → 7305 ms (best)**, N=1024 → 8363 ms
(regresses — array becomes N-throughput-bound + larger DMA, and fewer matmuls
are N%1024==0). So 512 is the sweet spot. Logits (vocab N=128256, not %512)
stays on the N=256 path.

Correctness unchanged across every arch (all still greedy-correct → "Paris"):
Llama-1B, Qwen3.5-4B/Qwen3-0.6B, Gemma-4-E2B (K=1536), LFM2.5-230M, Ministral-3-3B.

## 2026-07-06 — qwen35 (Qwen3.5 Gated DeltaNet) now correct on Q6_K (head-mapping fix)

`qwen35` hybrid SSM+attention was the last `models/` architecture producing
garbage. Root cause was **not** the long-suspected "numerical sensitivity" — it
was a value-head→key-head mapping bug in the delta-net recurrence
(`src/cli/main.cpp`, `is_qwen35` block): the per-value-head loop used
`kh = h / vgroup` (block grouping) but llama.cpp expands q/k from `nKH`=16 to
`nVH`=32 with `ggml_repeat_4d`, which is **modulo tiling** → value head `h`
reads key head `h % nKH`. The two agree only for head 0, so layer-0 head-0
tensors matched the reference while every later value head was silently wrong.
One-line fix: `const int kh = h % nKH;`. Diagnosed by per-tensor comparison
against `llama-eval-callback` and `llama-simple` on the newly-added Q6_K files.

**Correctness (greedy, `--temp 0`)** — matches llama.cpp token-for-token:
```
./build-npu/ggnpu -m models/Qwen3.5-4B-Q6_K.gguf -p "The capital of France is" -n 20 --temp 0
  → " Paris.\nA. True\nB. False\n\n<think>..."   (llama-simple ref: "Paris. A. True B. False Answer: A")
./build-npu/ggnpu -m models/Qwen3.5-9B-Q6_K.gguf -p "The capital of France is" -n 15 --temp 0
  → " Paris.\n\nWhat is the capital of France?"
```
Also correct: "The quick brown fox"→" jumps over the lazy dog.", "Q: What is
2+2? A:"→" 4.". Note qwen35 is a *thinking* model — compare with `llama-simple`
(raw completion), not `llama-cli` (forces chat template + `<think>`).

**Quant sweep** — all six `qwen35` GGUFs in `models/` now generate correct
greedy output (`-p "The capital of France is" -n 15 -c 512 --temp 0`):

| Model | Quant | Output (first tokens) | Result |
|-------|-------|-----------------------|--------|
| Qwen3.5-4B | Q6_K   | " Paris. A. True B. False" | ✅ |
| Qwen3.5-4B | Q4_K_M | " Paris. A. True B. False" | ✅ |
| Qwen3.5-9B | Q6_K   | " Paris." | ✅ |
| Qwen3.5-9B | Q3_K_M | " Paris. This is a well-known fact, and it is the most populous" | ✅ |
| Qwen3.5-9B | UD-IQ2_XXS | " Paris. A. True B. False" | ✅ (2-bit i-quant) |
| ornith-9b  | IQ2_M  | " the city of Paris. A. 100 B." | ✅ (2-bit i-quant; MTP blk.32 skipped) |

The head-mapping fix is quant-independent, so the same fix carries every
k-quant and i-quant. i-quant decoders (IQ2_XXS/IQ2_M) were already in place
from the 2026-07-03 quant-coverage work; the arch bug was the only thing
standing between them and coherent output.

## 2026-06-30 — Qwen2.5-Coder-1.5B runs end-to-end on the NPU (2nd architecture)

Second model architecture (`qwen2`) brought up on the NPU. Multi-arch is
metadata-driven: `GgufLoader` now resolves hparams under `general.architecture`
(`arch_key()`) instead of hardcoded `llama.*`. Qwen2 specifics handled generically
— QKV bias add after projections (no-op for Llama), `rope.dimension_count`
fallback to `embedding/n_head` (=128), RMSNorm pad-to-pow2 (1536→2048), SiLU
host-tiling through the 8192 kernel (FFN=8960). RMSNorm validation gate widened
1.2%→2.5% (Qwen2 activations peak ~1.68% vs Llama ~1.15%; still catches the old
~8% cast bug). On commit after `eded2b3` (uncommitted).

**Correctness**
```
./build-npu/ggnpu -m models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  -p "def fibonacci(n):" -c 2048 -n 40
```
→ coherent, correct Python:
```
    if n == 0:  return 0
    elif n == 1: return 1
    else: return fibonacci(n-1) + fibonacci(n-2)
```
(28 layers, hidden 1536, 12 heads / 2 KV, head_dim 128, FFN 8960, vocab 151936,
RoPE θ=1e6.)

**No Llama regression:** `compare_logits.py --check` → top-1 ` Paris`;
`test_e2e_logits` PASS; France prompt coherent.

---

## 2026-06-30 — resident weight BOs (decode now device-bound), ~5%

After deep-K, profiling (`GGNPU_MATMUL_TIMING=1`, logits call, 501 tiles) showed
the per-launch **weight DMA** (`dmaB`) had become 64% of decode matmul — it used
to be hidden behind the 256³ kernel, but deep-K shrank the device kernel so it was
exposed. INT8 weights never change across tokens, so `mul_mat_q` now packs each
deep-K weight tile into a **resident device BO** once and binds it directly on
every later launch (no per-token host→device copy). Gated to the deep-K decode
path (`use_deepk && int8`); prefill keeps per-call staging. Opt out
`GGNPU_NO_RESIDENT_W=1`. On commit after `7156ce4` (uncommitted).

**Memory-neutral:** the resident BOs replace the pageable host B-tile cache
(no longer populated on the int8 path), so max RSS is unchanged: **2738 MB**
(resident ON) vs **2737 MB** (OFF).

**Per-tile (logits, 501 tiles)**

| component | resident OFF | resident ON |
|-----------|------|------|
| dmaB (weight) | 31.5 µs | **0.2 µs** |
| pack (A) | 12.7 µs | 12.4 µs |
| kernel wait | 2.4 µs | 26.4 µs |
| submit wall | 23.5 ms | 7.7 ms |
| wait wall | 2.2 ms | 14.2 ms |

The freed weight-DMA time was **overlapping device execution** (the NPU runs
batch kernels serially while the host packs/DMAs the next tile), so removing it
mostly exposed the true device wait rather than shrinking the wall. Net per-tile
47.4 → 39.8 µs. **Decode is now device-bound:** the deep-K tile takes ~26 µs of
real device execution (the earlier "4 µs kernel-wait" was an artifact of the
device finishing during the long weight-DMA submit). Host pack already overlaps
device, so further host-side work won't move the wall — the next lever is
device-side (widen N per launch to amortize the remaining ~9× per-launch device
overhead, or more cores).

**Perf A/B — `-v`, 21 token steps, France prompt n=16** (vs deep-K, `7156ce4`)

| op | deep-K (`GGNPU_NO_RESIDENT_W=1`) | + resident W | Δ |
|----|------|------|------|
| matmul | 6206.1 ms | 5859.8 ms | −5.6% |
| logits | 1366.1 ms | 1297.2 ms | −5.0% |
| **total** | 7980.5 ms | 7581.8 ms | **−5.0%** (2.63 → 2.77 tok/s) |

Correctness: `compare_logits.py --check` → top-1 ` Paris`; France n=16 coherent.

---

## 2026-06-30 — deep-K decode matmul (in-kernel K=2048 reduction), ~2× decode

New `matmul_small_m_deepk` xclbin (M=16, N=256, **K=2048**) folds the whole K
reduction into one kernel launch instead of issuing K/256 separate 256³ tile
launches accumulated on the host. `mul_mat_q` routes small-M matmuls whose K is a
multiple of `kDeepK` (2048) to it (opt out `GGNPU_NO_DEEPK=1`); K=8192 (ffn_down)
runs as 4 deep-K spans accumulated on the host. Same INT8 datapath/transform as
small-M — only the K extent per launch grows (the transform's PHASE 4 K-reduction
loop just iterates 32× instead of 4×, accumulating in the L1 packed-C buffer). On
commit after `53542da` (uncommitted working tree).

**Why it works:** the prior profile showed decode matmul was bounded by
fixed per-launch device overhead (~13 µs/tile, ~35× the actual MACs), not the
MACs or DMA. A K=2048 matmul paid that overhead 8× (8 K-tiles). Deep-K pays it
once, so kernel-wait per output block drops ~8×.

**Correctness (all on hardware)**
- `bench-matmul`: `1×2048×2048` (single deep-K span) and `1×8192×256` (4 spans,
  host accum) both PASS (A=B=1 → C=K). 256³…4096³ paths unchanged.
- `test_e2e_logits` PASS (25 s); `compare_logits.py --check` → top-1 ` Paris`
  (id 12366).
- France prompt n=16 → ` Paris, and it is the most visited city in the world…` ✅
  (greedy output diverges from the K=256 path after a near-tie; top-1 logit is
  identical — deep-K does one int32 accumulation + one f32 scale vs 8 host f32
  partial sums, so it is at least as accurate).

**Perf A/B — same build, `-v`, 21 token steps, France prompt n=16**

```
./build-npu/ggnpu -v -m models/llama-3.2-1b-q4_k_m.gguf \
  -p "The capital of France is" -c 2048 -n 16            # deep-K ON (default)
GGNPU_NO_DEEPK=1 ./build-npu/ggnpu -v ... -n 16          # small-M K=256 baseline
```

| op | small-M K=256 | deep-K K=2048 | Δ |
|----|------|------|------|
| matmul | 12211.8 ms | 6206.1 ms | **−49.2%** |
| logits | 2581.4 ms | 1366.1 ms | −47.1% |
| **total** | 15255.0 ms | 7980.5 ms | **−47.7%** |
| tok/s | ~1.38 | **~2.63** | **+91%** |

Cumulative vs the original 256³ path (18902 ms, 2026-06-30 small-M entry):
**2.37× faster decode** (256³ → small-M → deep-K). Matmul is still the top cost
(78%) + logits (17%); the next lever would be a deep-K kernel that also widens N
or reduces the 512 KB/tile weight DMA (now ~19% of per-launch time).

---

## 2026-06-30 — small-M matmul kernel validated (branch `matmul-small-m-wip`)

Decode-optimized `matmul_small_m` xclbin (M=16, N=K=256) now correct on hardware
and wired into `mul_mat_q` (routes M ≤ 16; opt out `GGNPU_NO_SMALL_M=1`). See
IMPLEMENTATION.md §7.1 for the AIR herd-consistency fix.

**Correctness**
- `bench-matmul` M ∈ {1, 16, ...}: PASS (A=B=1 → C=K).
- `bench-layer` layer 0: all matmuls PASS; attn_q rel err 0.0166 = identical to
  256³ baseline (small-M output is bit-equivalent).
- `ctest` (8/8) PASS incl. `test_e2e_logits` (top-1 12366, ` Paris`).
- France prompt n=16 → `Paris. It is the most visited city in the world. ...` ✅

**Perf A/B — same build, `-v`, 21 token steps**

```
./build-npu/ggnpu -v -m models/llama-3.2-1b-q4_k_m.gguf \
  -p "The capital of France is" -c 2048 -n 16            # small-M ON (default)
GGNPU_NO_SMALL_M=1 ./build-npu/ggnpu -v ... -n 16        # 256³ baseline
```

| op | 256³ baseline | small-M | Δ |
|----|------|------|------|
| matmul | 15354.4 ms | 12304.9 ms | **−19.9%** |
| logits | 3134.6 ms | 2585.9 ms | −17.5% |
| **total** | 18901.9 ms | 15297.8 ms | **−19.1%** |
| tok/s | ~1.11 | **~1.37** | **+23%** |

The win is bounded (<16×) by the 8-core (2×4) herd vs the 256³'s 16 cores and
unchanged host pack/DMA; only device row-compute shrinks (8 M-rows/core vs 64).
A 2×8 (16-core) herd does not place on npu2 (4 compute rows max).

---

## 2026-06-29 — commit `8959e21`

**Run A — wall-clock generation timing**

```
./build-npu/ggnpu -m models/llama-3.2-1b-q4_k_m.gguf \
  -p "The capital of France is" -c 2048 -n 32
```

- Model: llama-3.2-1b-q4_k_m.gguf (16 layers, hidden 2048, 32 heads / 8 KV, FFN 8192)
- Output: `Paris. It is the most visited city in the world. ...` ✅ correct
- Wall clock: **32.61 s** for 32 generated tokens → **≈1.0 tok/s**
- Max RSS: 2.73 GB
- flash_attn on host f32 (default); RoPE on host (default)

**Run B — per-op breakdown (`-v`, n=8 → 13 token steps)**

```
./build-npu/ggnpu -v -m models/llama-3.2-1b-q4_k_m.gguf \
  -p "The capital of France is" -c 2048 -n 8
```

Total 13 token steps: **13230.2 ms** (≈1.0 tok/s).

| Op | Time | % |
|----|------|---|
| **matmul** | 10903 ms | **82.41%** |
| **logits** | 2079.6 ms | **15.72%** |
| rms_norm | 129.5 ms | 0.98% |
| silu | 87.9 ms | 0.66% |
| flash_attn (host f32) | 18.1 ms | 0.14% |
| residual | 3.66 ms | 0.03% |
| sample | 3.52 ms | 0.03% |
| kv_expand | 2.49 ms | 0.02% |
| rope | 2.17 ms | 0.02% |
| embed | 0.24 ms | 0.00% |

**Run C — matmul internal timing (`GGNPU_MATMUL_TIMING=1`, n=4)**

Per-tile breakdown for the 256³ INT8 tile (logits call, 4008 tiles):

| Component | ms/tile | share |
|-----------|---------|-------|
| **kernel wait** | 0.0207 | **66%** |
| dmaB | 0.0049 | 15% |
| dmaC | 0.0028 | 9% |
| pack | 0.0024 | 8% |
| dmaA | 0.0007 | 2% |
| **per-tile total** | **0.0314** | |

Batching (size 24) already hides submit overhead (submit ≈0.22 ms / 24-tile
batch). The cost is **on-device kernel execution, not driver round-trips**.
~18,800 of these 256³ tiles run per generated token (16 layers + logits).
Effective rate ≈ 0.8 TMAC/s — a tiny fraction of npu6's ~50 TOPS, and during
decode (M=1) **255/256 of each tile's row-compute is wasted** (kernel is fixed
256³; only 1 output row is needed).

**Takeaway:** throughput is entirely **matmul-bound** (82%) plus the vocab
logits projection (16%, itself a matmul). Attention/norms/activations are
noise. Making this usable = making NPU matmul faster (and/or fewer
host↔device round-trips per matmul). Flash-attn being on the host CPU is
**not** the bottleneck.
