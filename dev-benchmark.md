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
