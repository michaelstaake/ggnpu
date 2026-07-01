# GGNPU — AI Implementation Handoff

This document is the complete specification for building **ggnpu**: a custom llama.cpp-like runtime that runs standard GGUF models on AMD NPUs. Give this file (and the repo) to an AI coding agent and work through the phases in order.

**Repository:** https://github.com/michaelstaake/ggnpu  
**License:** GPL-3.0  
**Current state:** **Phases 1–5 MVP passed** on hardware; Phase 6 in progress. NPU path: INT8 matmul (incl. logits via mul_mat_q), RMSNorm (any hidden via pad-to-pow2), SiLU N=8192, RoPE (batched kernel, opt-in `GGNPU_NPU_ROPE=1`). Flash attention uses **host f32** (fused NPU kernel blocked on Triton-XDNA multi-reduction lowering). E2E regression passes: `ctest -R test_e2e_logits`, `python3 scripts/compare_logits.py --check` (France prompt → Paris). Next focus: fused flash_attn on NPU, RoPE perf tuning, matmul perf. See **§7.1**.

**Verdict:** The supported path is host-native: install host prerequisites, build `ggnpu` locally, and provide local `.xclbin` + `*_sequence.bin` kernels under `~/.cache/ggnpu/xclbin/`.

**Benchmark log:** Every timed/correctness run is recorded in [dev-benchmark.md](dev-benchmark.md) (commit, date, command, result). **Append a new entry whenever you run a benchmark or timed test** — it is the historical performance record. Latest: Llama 3.2 1B Q4_K_M ≈ **1.37 tok/s** with the validated decode-optimized small-M matmul kernel (≈1.0 tok/s before; ~20% faster matmul, see §7.1). Runtime is still **~80% matmul + 17% logits** — throughput is matmul-bound, not attention-bound.

---

## 1. Mission

Build a standalone C++20 inference binary named `ggnpu` that:

1. Loads **standard HuggingFace GGUF** files with **no user-side conversion** (no ONNX export, no AMD Quark, no re-quantization step).
2. Runs **all transformer tensor math** on the **AMD NPU** — matmul, attention, norms, activations, RoPE.
3. Uses the **host CPU only** for control plane: CLI, GGUF parsing, mmap, tokenization, sampling, logging.
4. Targets **Ubuntu 26.04 LTS** (Linux **7.0**) and **AMD Ryzen AI 7 350** (Krackan / XDNA2 / `npu6`).
5. Builds and runs **directly on the host** with `/dev/accel/accel0`, firmware, XRT, and local `.xclbin` kernels.
6. Ships simple user docs: `README`, `docs/host-setup-guide.md`, `docs/usage.md`, and `ggnpu --help`.

### 1.1 Deployment model (native host)

| Layer | Where it runs |
|-------|----------------|
| `amdxdna` driver, `/dev/accel/accel0` | **Host** (kernel) |
| NPU firmware | **Host** filesystem |
| XRT runtime + headers | **Host** |
| `ggnpu` binary | **Host build output** |
| `.xclbin` kernels | **Host cache** (`~/.cache/ggnpu/xclbin`) |
| Inference CLI | **`./build-npu/ggnpu ...`** |

**Users must** install XRT and build `ggnpu` on the host.

**Kernel compilation** remains optional for users who already have prebuilt `.xclbin` artifacts, but native host deployment is the default path.

See **§9** and `docs/host-setup-guide.md`.

---

## 2. Hard constraints

### Forbidden

| Item | Reason |
|------|--------|
| FastFlowLM | User requirement |
| llama.cpp as a base/fork | Build from scratch |
| VitisAI Execution Provider | Proprietary ONNX stack |
| ONNX Runtime for inference | Same |
| GPU (`/dev/dri`, ROCm, Vulkan) for inference | User requirement |
| CPU tensor math in **production** builds | User requirement |
| User preprocessing (ONNX, Quark, custom model format) | User requirement |

### Allowed

| Item | Role |
|------|------|
| Linux `amdxdna` kernel driver + firmware | Hardware access via `/dev/accel/accel0` |
| **XRT** (`libxrt`, NPU plugin) | Runtime: open device, allocate buffers, load kernels, submit work |
| **Triton-XDNA** | Build-time: pip install; compile Triton kernels → `.xclbin` via triton-shared + MLIR-AIR/AIE |
| `cpu_ref` backend | **Unit tests only** (`GGNPU_TEST_CPU=1`), never production fallback |

### Production rule

If NPU initialization fails, `ggnpu` must **exit with an error**. No silent fallback to CPU or GPU.

Ops marked **NPU** in §2.1 must run on the NPU or **fail with an error**. Ops marked **CPU** are intentional host tensor math (flash_attn today). RoPE is the one explicit exception: it has both paths and an intentional CPU fallback (NPU is opt-in via `GGNPU_NPU_ROPE=1`). Otherwise no silent fallback from a failed NPU op to CPU for the same op.

### 2.1 CPU vs NPU operation map

**Keep this table updated** when wiring a new kernel or moving work off the host. Target: Llama 3.2 1B Q4_K_M on Krackan (`npu6`). Status values: **NPU** (tensor math on device — errors if kernel missing or wrong), **CPU** (intentional host tensor math — not on NPU yet), **Host** (control plane, I/O, cheap post-processing).

**No CPU fallback:** ops marked **NPU** must run on the NPU or fail with an error. Host work after an NPU kernel (γ multiply, f32↔bf16 conversion, tiling) is not a fallback.

| Operation | Where | Status | Notes |
|-----------|-------|--------|-------|
| GGUF load / mmap | Host | Host | `gguf.cpp`, `model.cpp` |
| Tokenization / BPE | Host | Host | `tokenizer.cpp` |
| Sampling (argmax / temp) | Host | Host | `main.cpp` |
| Weight cache decode (Q4_K/Q6_K → INT8) | Host | Host | `weight_cache`; feeds NPU matmul |
| Per-token embedding dequant | Host | Host | `dequant_tensor_row()` in `main.cpp` |
| Activation quantize (f32 → INT8 tiles) | Host | Host | `amd_xdna.cpp` matmul path |
| **INT8 matmul** (Q/K/V/O, FFN gate/up/down) | NPU | **NPU** | `matmul_npu6.xclbin` (256³ tiles); host tiling/quant; no CPU fallback |
| **RMSNorm** (any hidden) | NPU | **NPU** | pow2 kernels (`rmsnorm_2048`/`rmsnorm_4096`); non-pow2 N pads to next pow2 + constant output correction in `amd_xdna.cpp` (1536→2048: rel 0.0072). f32 reductions in `rmsnorm_aie2p.mlir`; per-N bf16-aware validation — no CPU fallback |
| RMSNorm learned weights (`γ`) | Host | Host | O(N) multiply after unweighted NPU norm in `amd_xdna.cpp` |
| **RoPE** (Q, K) | NPU | **NPU** (opt-in) | `rope_npu6.xclbin` binary-add kernel; `rope_batched()` packs ~8 heads/launch. Wired via `apply_rope()` in `main.cpp` when `GGNPU_NPU_ROPE=1` (default CPU; opt-in pending perf tuning), CPU fallback on failure |
| KV cache write | Host | Host | `kv_cache.cpp` |
| GQA KV expand (repeat KV heads) | Host | Host | memcpy loops in `main.cpp` |
| **Flash attention** (32×64, ctx≤2048) | Host | **CPU** (NPU opt-in) | Default host f32 decomposed path (`flash_attn_decomposed`); matches `cpu_ref`. `GGNPU_NPU_ATTN=1` runs QK/AV as bf16 GEMMs on the NPU + host softmax (`flash_attn_npu`, validated but slow — opt-in). Fused NPU blocked (§7.1) |
| **SiLU** (FFN gate, N=8192) | NPU | **NPU** | `silu_npu6.xclbin`; errors if xclbin missing — no CPU fallback |
| **Softmax** | NPU | **NPU** | `softmax_npu6.xclbin`; wired in backend, not used in decoder loop today |
| SwiGLU `silu(gate) * up` multiply | Host | Host | elementwise after NPU SiLU |
| Residual adds (attn + FFN) | Host | Host | cheap; acceptable for MVP |
| **Logits** (output projection) | Host/NPU | **NPU** | `compute_logits()` — INT8 mul_mat_q via WeightCache; CPU fallback for F32 weights |
| `bench-layer` / `cpu_ref` reference | Host | Host | tests only; not production inference |

**xclbin artifacts (npu6):**

| Kernel | Artifact | Llama 1B shape |
|--------|----------|----------------|
| matmul | `matmul_npu6.xclbin` | 256³ INT8 tiles |
| rmsnorm | `rmsnorm_2048_npu6.xclbin`, `rmsnorm_4096_npu6.xclbin` | M=2, N=pow2 bf16; non-pow2 hidden pads up at runtime (rebuild after mlir changes; see below) |
| silu | `silu_npu6.xclbin` | N=8192 bf16 |
| softmax | `softmax_npu6.xclbin` | 256×256 (not used in decoder loop today) |
| flash_attn | — (experimental) | `GGNPU_EXPERIMENTAL=1 ./scripts/build-kernels.sh npu6 flash_attn_32x64x2048` — placeholder transform; not used in default inference |
| rope | `rope_npu6.xclbin` | binary-add kernel (n_pairs=32, pad 512); `rope_batched()` packs heads. Opt-in via `GGNPU_NPU_ROPE=1` |

Build: `./scripts/build-kernels.sh npu6 [kernel_name]`

After `rmsnorm_aie2p.mlir` changes, force rebuild:

```bash
rm ~/.cache/ggnpu/xclbin/rmsnorm_2048_npu6*
./scripts/build-kernels.sh npu6 rmsnorm_2048
```

### AIE kernel design guardrails

The AMD XDNA 2 architecture has no safety net — no OS thread scheduler, no hardware cache handlers, no hazard checking. These four guardrails prevent AI agents (and humans) from writing code that compiles but crashes the NPU or runs slower than CPU execution.

#### 1. Memory-First Design

Memory bandwidth, not compute, is the killer bottleneck on XDNA 2. All memory operations must be strictly block-based and explicitly partitioned to match the Strix Point 4 MB shared L2 memory tile array.

- **Tensor execution loop:** Structure so that while Compute Tile Row 1 processes Layer N, the XRT DMA engine streams weights for Layer N+1 from host DDR5 into L2 tiles.
- **Never** let compute tiles stall waiting for host memory fetches.
- **Never** write continuous, un-chunked data reads. All DMA transfers must be block-aligned.
- **L2 awareness:** Each tile row has limited local memory. Tile large matmuls to fit within L2; stream remaining data from DDR.

#### 2. Strict Vectorization and Type Constraints

The AIE cores have 512-bit vector registers. Standard C math operators (`+`, `*`) will not auto-vectorize efficiently into VLIW.

- **Write core kernels using explicit AMD AIE API vector intrinsics**, not scalar loops.
- **Native math types:** Restrict kernel functions strictly to `INT8`, `BF16`, or `FP8`. No scalar `float` or `double` in hot paths.
- **No scalar fallback:** If a computation doesn't map to a vector intrinsic, restructure the algorithm — don't drop to scalar C.

#### 3. Prohibit Branch Logic in Kernels

VLIW processors inside XDNA compute tiles have a flat instruction pipeline with zero hardware-level hazard checking. Branching breaks instruction pipelining.

- **Kernels must be fully deterministic and completely unrolled** where possible.
- **Zero conditional branch logic (`if/else`, `switch`, `while` with non-constant bounds)** inside hot execution loops.
- Use predication, lookup tables, or arithmetic tricks instead of branches.
- Loop bounds must be compile-time constants or kernel parameters passed as immediate values.

#### 4. Two-Layer Output Architecture

LLMs confuse what code goes where when dealing with XRT. Clearly separate:

| Layer | Where it runs | What it does |
|-------|--------------|--------------|
| **Control Code** | NPU internal microcontroller (core 0) | Data movement graph, DMA setup, kernel launch via IRON API / `XAie_TxnOpcode` sequence |
| **Kernel Execution Code** | Spatial compute tile rows (Triton-XDNA compiled) | Vectorized INT8/BF16 math kernels, no branches, fully unrolled loops |

The control code **never** contains tensor math. The kernel code **never** handles DMA setup or kernel launch. This separation is enforced by the triton-shared + MLIR-AIR/AIE compilation pipeline.

### "No conversion" clarified

Users run:

```bash
ggnpu -m Llama-3.2-3B-Q4_K_M.gguf -p "Hello"
```

**Allowed at runtime (transparent to user):**

- Parse GGUF, mmap weights
- JIT-compile NPU kernels per shape; cache under `~/.cache/ggnpu/`
- On first use per layer: decode GGUF quant blocks → INT8 NPU weight buffers (cached)

**Not allowed:** asking users to export ONNX, run AMD tooling, or store a second model file.

---

## 3. Target hardware

| Item | Value |
|------|-------|
| CPU | AMD Ryzen AI 7 350 (Krackan Point) |
| NPU arch | XDNA2 (AIE2P), 4×8 tile array, ~50 TOPS, 4096 KB L2, 16 HW contexts |
| PCI ID | `1022:17f0` rev `0x20` → profile `npu6`, vbnv `RyzenAI-npu6` |
| OS | Ubuntu 26.04 LTS, Linux 7.0 |
| Device node | `/dev/accel/accel0` (group `render`) — **not** `/dev/dri` or `/dev/kfd` |
| Kernel | `CONFIG_DRM_ACCEL_AMDXDNA`, `CONFIG_AMD_IOMMU=y`, IOMMU enabled (`amd_iommu=on`) |
| Host limit | `ulimit -l unlimited` (memlock for buffer pinning) |
| Firmware | `/usr/lib/firmware/amdnpu/` (from linux-firmware) |

Also detect at runtime: Strix Point (`npu4`/`npu5`) for broader testing; select correct xclbin profile via PCI ID or `xrt-smi examine`.

---

## 4. Dependency roles (XRT, Triton-XDNA)

These are **not** inference frameworks. ggnpu builds the GGUF runtime; these handle NPU kernel compile and dispatch.

```
GGUF file
  → ggnpu (parse, graph, quant, schedule)     ← YOU BUILD THIS
  → XRT (load xclbin, DMA buffers, run)       ← runtime, every inference
  → amdxdna kernel (/dev/accel/accel0)

NPU kernels (.xclbin):
  @triton.jit → triton-shared (Linalg) → MLIR Transform dialect → MLIR-AIR / MLIR-AIE → xclbin
                                                ↑ build time / first-run cache miss only
```

| Tool | When it runs | What it does |
|------|--------------|--------------|
| **XRT** | Every inference | Open NPU, pinned `xrt::bo` buffers, load `.xclbin`, `xrt::run`, sync |
| **Triton-XDNA** | Build / cache miss | `pip install triton-xdna`; compile Triton kernels → `.xclbin` via triton-shared + MLIR-AIR/AIE |

**Why XRT not raw ioctls:** Raw `DRM_IOCTL_AMDXDNA_*` requires reimplementing BO lifecycle, PASID/IOMMU SVA, ERT mailbox, firmware coupling. XRT is the official shim (like libdrm for GPU).

**Native deployment:** Production delivery now assumes a host-native install. XRT runs on the host, `ggnpu` is built locally, and `.xclbin` kernels live in `~/.cache/ggnpu/xclbin/`. See `docs/host-setup-guide.md`.

---

## 5. Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Host CPU (control only)                                  │
│  CLI → GGUF parser → tokenizer → sampler → scheduler   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│ ggnpu core                                               │
│  arch plugin (llama) → compute graph → KV cache          │
│  quant metadata → compile cache                          │
└──────────────────────────┬──────────────────────────────┘
                           │ Backend interface
┌──────────────────────────▼──────────────────────────────┐
│ AmdXdnaBackend (XRT)                                     │
│  mul_mat_q, rms_norm, rope, softmax, silu, flash_attn   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│ AMD NPU (amdxdna) — AIE2P tiles + 4 MB L2                │
└─────────────────────────────────────────────────────────┘
```

### Layer 1 — GGUF I/O (`src/gguf/`)

Implement per [GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). Do **not** vendor llama.cpp.

- Parse header v3, KV metadata, tensor index
- `mmap()` tensor blob; honor `general.alignment` (default 32)
- `TensorView { name, dims, ggml_type, const void* data }`
- Tokenizer from `tokenizer.ggml.*` or `tokenizer.huggingface.json`
- Read `general.architecture`, block count, context length, RoPE, GQA head counts

### Layer 2 — Architecture plugin (`src/arch/llama/`)

| Priority | Arch | Quants | Status |
|----------|------|--------|--------|
| P0 | `llama` (3.2 1B/3B) | `Q4_K_M`, `Q8_0` | **Validated** (1B) |
| P1 | `qwen2` (2.5 1.5B) | `Q4_K_M` | **Validated** — coherent codegen on NPU |
| P2 | `mistral`, `gemma2` | extend tensor maps | not yet |

Map tensor names → roles (`attn_q`, `ffn_gate`, etc.). Handle GQA. Respect `llama.tensor_data_layout` permutes.

**Multi-arch support is metadata-driven, not per-arch plugins.** GGUF namespaces
hparams under `general.architecture`, so `GgufLoader::arch_key()` resolves
`<arch>.<suffix>` (was hardcoded `llama.*`). What Qwen2 needed beyond Llama, all
generic: (1) **QKV bias** — Qwen2 ships `attn_{q,k,v}.bias`; the forward pass adds
them after the projections (no-op when absent, e.g. Llama). (2) **`rope.dimension_count`
fallback** — optional in GGUF (Qwen2 omits it); fall back to `embedding/n_head`
(=128 for Qwen2.5 1.5B). (3) **non-Llama sizes** — RMSNorm pads any hidden to the
next pow2 (1536→2048) and SiLU host-tiles any FFN width through the fixed 8192
kernel (Qwen2 FFN=8960), so no per-model kernel builds. (4) RoPE θ=1e6 and
`rms_eps=1e-6` come straight from metadata. The bf16 RMSNorm validation gate was a
gross-error detector tuned to Llama (~1.15% peak); widened to 2.5% to admit Qwen2's
activation distribution (~1.68%) while still catching the old ~8% cast bug.

### Layer 3 — Quantization (`src/quant/`)

| Phase | `ggml_type` values |
|-------|-------------------|
| P0 | `F32`, `F16`, `Q8_0`, `Q4_0` |
| P1 | `Q4_K`, `Q6_K` (required for `Q4_K_M` — mixed per-tensor types) — **decoders implemented** |
| P2 | `IQ4_NL`, others |

**NPU weight path:** GGUF weights stay mmap'd. On first use per tensor:

1. Decode quant blocks on CPU (host only)
2. Build INT8 + scales buffer for NPU
3. Cache keyed by `(model_path, mtime, tensor_name)`
4. Reuse for all subsequent tokens

Activations: quantize per forward on CPU into pinned buffers → DMA → NPU.

### Layer 4 — Compute graph (`src/graph/`)

Ops: `MUL_MAT_Q`, `RMS_NORM`, `ROPE`, `SOFTMAX`, `ADD`, `MUL`, `SILU`, `FLASH_ATTN` (v2).

- **Prefill graph:** `n_tokens > 1`
- **Decode graph:** `n_tokens == 1`, KV index = current position
- **KV cache:** `[n_layer][n_ctx][n_head_kv][head_dim]`, pre-allocated
- **Known gap:** `init_kv_cache()` currently uses GGUF `context_length`, not CLI `-c`. Must cap at `-c` (MVP: 2048) to avoid multi-GB allocation on models with 128k metadata ctx — see §7.1.

### Layer 5 — Backend interface (`include/ggnpu/backend.h`)

```cpp
struct Backend {
  virtual Status mul_mat_q(MulMatParams) = 0;
  virtual Status rms_norm(RmsNormParams) = 0;
  virtual Status rope(RopeParams) = 0;
  virtual Status softmax(SoftmaxParams) = 0;
  virtual Status silu(SiluParams) = 0;
  virtual Status flash_attn(AttnParams) = 0;  // v2
  virtual void sync() = 0;
};
```

Production: `AmdXdnaBackend`. Tests only: `cpu_ref`.

### Layer 6 — AMD XDNA backend (`src/backends/amd_xdna/`)

**Architecture split (Guardrail #4):**

The NPU backend consists of two code paths that must never mix:

- **Control plane** (`amd_xdna.cpp`): XRT device management, buffer allocation, xclbin loading, kernel launch. Runs on host CPU, talks to NPU microcontroller via IRON API / `XAie_TxnOpcode`.
- **Kernel plane** (`kernels/triton/`): Triton-XDNA Python kernels. Runs on spatial compute tiles. Vectorized intrinsics only, no branches, no scalar math.

**Bring-up:**

1. `xrt::device` on `/dev/accel/accel0`
2. Detect `npu6` (Krackan) → `aie2p` kernel profile
3. HW context; unlimited memlock; `xrt::bo` for weights/activations
- [x] Load `.xclbin` from `~/.cache/ggnpu/xclbin/` or JIT-compile

**Known gap:** SiLU NPU kernel is fixed at N=8192 (Llama 1B FFN). Other sizes need JIT compile — hard error if missing.

**Known gap:** Flash attention is host f32 decomposed (accurate; INT8 NPU matmul QK/AV loses logits coherence). Fused NPU kernel blocked on mlir transform (`flash_attn_aie2p.mlir` is a placeholder).

**RoPE:** NPU kernel (`rope_npu6.xclbin`) is wired via `rope_batched()` (packs ~8 heads/launch). Defaults to CPU; enable the NPU path with `GGNPU_NPU_ROPE=1` (opt-in pending perf tuning vs the per-token launch overhead), with CPU fallback on failure.

**RMSNorm:** any hidden size works — non-pow2 N (e.g. 1536) pads to the next power of 2 at runtime and applies a constant output correction. Only pow2 kernels need building (`rmsnorm_2048`, `rmsnorm_4096`); a missing pow2 kernel is a hard error (no CPU fallback).

**Kernel compile cache key:** `(op, M, N, K, dtype, npu_profile)`

**Llama 3B typical matmul shapes:**

| Layer | K / N |
|-------|-------|
| Q/K/V/O proj | 3072, 2048 |
| FFN gate/up | 8192 |
| FFN down | 8192 → 3072 |

Use the Triton-XDNA compilation flow:

```
Triton kernel (@triton.jit)
  -> triton-shared (Linalg)
    -> MLIR Transform dialect (tiling, bufferization, vectorization)
      -> MLIR-AIR / MLIR-AIE
        -> XRT binary (aie.xclbin)
```

Install with: `pip install triton-xdna`

**NPU op rollout order:**

1. `MUL_MAT_Q` (~70–80% of time)
2. `RMS_NORM`
3. `ROPE`
4. `SOFTMAX` + attention matmul (decomposed v1)
5. `SILU` + SwiGLU mul
6. `FLASH_ATTN` fused (v2 perf)

### Layer 7 — CLI (`src/cli/main.cpp`)

See Section 8 for full flag reference.

---

## 6. Repository layout

Create this structure:

```
ggnpu/
├── .gitignore                     # ignores models/ (local test GGUFs)
├── CMakeLists.txt
├── README.md
├── IMPLEMENTATION.md              # this file
├── models/                        # local test models (gitignored)
├── docs/
│   ├── usage.md                   # CLI flags (≤80 lines)
│   ├── architecture.md
│   ├── amd-krackan.md
│   └── host-setup-guide.md
   ├── cmake/                         # FindXRT
├── third_party/                   # minimal only (xxhash, etc.)
├── include/ggnpu/
│   ├── gguf.h, tensor.h, graph.h, backend.h, model.h
├── src/
│   ├── gguf/
│   ├── quant/
│   ├── arch/llama/
│   ├── graph/
│   ├── runtime/
│   ├── cache/
│   ├── backends/
│   │   ├── cpu_ref/               # tests only
│   │   └── amd_xdna/
│   └── cli/main.cpp
├── kernels/triton/
│   └── compile_kernels.py
├── tests/
└── scripts/
    ├── setup-host.sh
    ├── build-kernels.sh
    └── verify-npu.sh
```

---

## 7. Implementation phases

Work through these in order. Do not skip ahead.

### Phase status summary

| Phase | Goal | Status |
|-------|------|--------|
| 0 Scaffold | CMake, scripts, docs | Mostly done (native host flow documented; no CI) |
| 1 GGUF loader | Parse, mmap, dump | **Done** |
| 2 NPU matmul smoke | `bench-matmul` on hardware | **Done** — validated output, host-tiled INT8 256³ kernel |
| 3 Q4_K weight path | Decode + one E2E matmul | **Done** — `bench-layer` FFN gate/up/down PASS vs CPU ref |
| 4 Full decoder layer | One layer vs CPU ref | **Done** — NPU matmuls + RMSNorm N=2048 + SiLU; flash_attn on host f32 |
| 5 Inference MVP | Coherent text, Llama 1B ctx 2048 | **Done** — France prompt → Paris; `bench-logits` + `test_e2e_logits` regression |
| 6 Production | Native deployment, 3B, L2 tiling | **Partial** — native setup documented; Llama 1B E2E validated on NPU; RMSNorm generalized to any hidden (1536/3072 via pad-to-pow2; 4096 kernel triggers AIR L2 split); batched RoPE on NPU (opt-in); decode-optimized small-M matmul validated (~20% faster matmul, §7.1). Remaining: fused flash_attn, RoPE perf, 3B model validation |

### Phase 0 — Scaffold

- [x] CMake project, C++20
- [ ] CI (compile on x86; NPU tests `MANUAL`)
- [x] `scripts/setup-host.sh` — host prerequisite checks and next steps
- [x] `scripts/verify-npu.sh` — hardware, driver, accel0, XRT, cache checks
- [x] `docs/architecture.md`, `docs/amd-krackan.md`, `docs/host-setup-guide.md`

**Done when:** native configure/build succeeds; host passes `verify-npu.sh`; `./build-npu/ggnpu bench-matmul` reaches backend init on hardware.

### Phase 1 — GGUF loader

- [x] Parser, mmap, metadata API
- [x] Quant layouts: F16, Q8_0, Q4_0
- [x] Llama hparams + tensor name map
- [x] Unit tests on real GGUF files

**Done when:** `ggnpu -m model.gguf --dump-tensors` prints correct inventory. **Gate passed** on `models/llama-3.2-1b-q4_k_m.gguf`.

### Phase 2 — NPU matmul smoke — **DONE**

- [x] XRT device init on Krackan (`xrt::register_xclbin` + `xrt::hw_context`; `load_xclbin` is unsupported on amdxdna)
- [x] INT8 matmul xclbin (fixed 256×256×256, INT8→INT32) + instruction sequence, host-tiled to any shape
- [x] Compile cache populated (`~/.cache/ggnpu/xclbin/`: matmul, rmsnorm, softmax, silu + `*_sequence.bin`)
- [x] Copy matmul output from device back to host in `src/backends/amd_xdna/amd_xdna.cpp`
- [x] `xrt::run::wait()` after each launch (runs are async)
- [x] `ggnpu bench-matmul` validates and benches on hardware (~1 ms per 256³ tile)

### Phase 3 — Q4_K weight path — **DONE**

- [x] Q4_K + Q6_K block decoders (real ggml layouts: 144 B / 210 B super-blocks, fp16 super-scales)
- [x] Transparent INT8 NPU weight cache (disk under `~/.cache/ggnpu/weights/`)
- [x] One `ffn_gate` matmul end-to-end from GGUF on NPU — `bench-layer` passes all three FFN matmuls (gate/up Q4_K, down Q6_K) at <0.09 relative error vs CPU reference

**Done when:** output within tolerance vs CPU reference dequant matmul. **Gate passed** on `models/llama-3.2-1b-q4_k_m.gguf` layer 0.

**Note:** decoded-weight cache entries are not versioned. After decoder changes, clear `~/.cache/ggnpu/weights/` or stale int8 weights will be reused.

### Phase 4 — Full decoder layer — **DONE**

- [x] `bench-layer` validates all attention matmuls + full layer forward on NPU vs CPU ref
- [x] bf16 f32↔bf16 marshaling for rmsnorm/softmax/silu DMA paths
- [x] Hard error when NPU elementwise kernels unavailable or produce wrong output (no CPU fallback)
- [x] Load all prebuilt xclbins at backend init (matmul, rmsnorm, softmax, silu)
- [x] All matmuls in one layer on NPU (gate/up/down + attn_q/k/v/output benched)
- [x] Full single-layer forward CPU vs NPU (`bench-layer` test 4)
- [x] RMSNorm on NPU, any hidden size — **works** (`rmsnorm_aie2p.mlir` keeps reductions in f32; non-pow2 N pads to next pow2 + constant correction; rebuild xclbin after mlir changes)
- [x] SiLU N=8192 on NPU — **works**
- [x] RoPE on NPU (batched kernel, opt-in `GGNPU_NPU_ROPE=1`) + CPU path (correct Llama 3 NORMAL pairing + `rope_freqs` freq factors)
- [x] KV cache write + causal attention E2E in inference loop

**Done when:** one-layer forward matches CPU reference. **Gate passed** via `bench-layer`.

### Phase 5 — Inference MVP — **DONE**

- [x] Prefill + decode loops — sequential per-token forward with KV cache (decode-style prefill)
- [x] Tokenizer + greedy sampling — `test_tokenizer` verifies `Hello` → `[128000, 9906]` and France prompt token ids
- [x] Coherent text generation — RoPE NORMAL adjacent pairs, `rope_freqs` as freq factors, per-row Q6_K embeddings, F32 logits projection
- [x] Target: Llama 3.2 1B Q4_K_M, ctx 2048 — greedy top-1 after France prompt is ` Paris` (id 12366); matches llama.cpp logits within ~0.01
- [x] Fix KV cache to respect `-c` / cap default ctx to 2048
- [x] Dequant `token_embd.weight` for Q4_K / Q6_K (per-token row dequant for embeddings)
- [x] `bench-logits` CLI + `scripts/compare_logits.py` + `ctest -R test_e2e_logits` regression

**Done when:**

```bash
ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 8
# → Paris, and it is the most visited
ctest -R test_e2e_logits   # top-1 id 12366 (skips if model missing)
```

Coherent text runs with matmul + RMSNorm + SiLU on NPU and flash_attn on host f32 (§2.1). Fused NPU attention is future work (`GGNPU_EXPERIMENTAL=1`).

### Phase 6 — Production

- [ ] Llama 3.2 3B; L2-aware tiling
- [ ] Ship validated prebuilt xclbins or document local kernel build clearly
- [ ] Clear errors: missing firmware, IOMMU off, memlock
- [ ] `docs/usage.md`, README quick start, `ggnpu --help` in sync

**Done when:** New user runs inference using only `README.md` + `docs/host-setup-guide.md` on a native host install.

### 7.1 Readiness snapshot

Assessment of whether the project can run a model on the NPU **today**.

#### Ready (host-native)

| Layer | Status | Notes |
|-------|--------|-------|
| NPU hardware | Present | PCI `1022:17f0` (Strix/Krackan) |
| Kernel driver | Loaded | `amdxdna` module (host) |
| Device node | Present | `/dev/accel/accel0` on host |
| Firmware | Present | `/usr/lib/firmware/amdnpu` |
| Native configure/build | Works | `cmake -S . -B build-npu ...` |
| GGUF loading | Works | `--dump-tensors` |
| Test models | On disk | See §17 |

**Host still needs:** XRT, `libxrt-dev`, and a native `ggnpu` build. `pip install triton-xdna` is only needed to build kernels locally.

#### Kernel status

| Check | Status | Notes |
|-------|--------|-------|
| `.xclbin` in `~/.cache/ggnpu/xclbin` | **Present** | matmul, rmsnorm (2048/4096), softmax, silu, rope (npu6) + `*_sequence.bin` each |
| `bench-matmul` E2E | **Passes** | Validated correct output on hardware |
| rmsnorm/softmax/silu/rope on NPU | **Works** | bf16 kernels + host f32↔bf16 marshaling in `amd_xdna.cpp` |
| flash_attn xclbin | Not in default build | host f32 inference; experimental fused build needs `GGNPU_EXPERIMENTAL=1` |
| Full inference E2E | **Validated** | France prompt → Paris on `build-npu/ggnpu`; `test_e2e_logits`, `compare_logits.py --check` |
| NPU utilization | **Moderate** | Matmul + RMSNorm + SiLU on NPU (RoPE opt-in); flash_attn, logits, residuals on host |

Production commands use the **native host build** (§9). NPU ops (matmul, rmsnorm, silu) **fail with an error** if the xclbin is missing, mis-built, or fails first-call validation — no silent CPU fallback for those ops. Flash attention is **intentionally** host f32 until a fused NPU kernel ships.

#### Known code gaps (post–Phase 5 / toward Phase 6)

| Gap | File | Impact |
|-----|------|--------|
| Fused flash attention on NPU | `flash_attn_aie2p.mlir`, `compile_kernels.py` | **Blocked**: the 4-reduction kernel compiles to AIR with **no `air.herd`** (compute stuck at L2), so `airrt-to-npu` fails to legalize `airrt.dma_memcpy_nd`/`airrt.segment_load`. Recommended fix: decompose into 3 single-reduction NPU kernels (QK matvec → softmax → AV matvec) with host-side ctx masking. Workaround: host f32 decomposed path. See **Known limitation** below. |
| Flash attention fused opt-in | `amd_xdna.cpp` | `GGNPU_FLASH_ATTN_FUSED=1` + experimental xclbin; default is host f32 |
| RMSNorm xclbin staleness | `build-kernels.sh`, `rmsnorm_aie2p.mlir` | Old `rmsnorm_2048` xclbins (~8% error) must be rebuilt after mlir fix |
| SiLU FFN=8192 on NPU | `amd_xdna.cpp` | **Works** when `silu_npu6.xclbin` present |
| Matmul perf: host-side tiling | `amd_xdna.cpp` | Host weight-tile cache; per-tile DMA. Decode uses small-M (M=16) + **deep-K (in-kernel K=2048 reduction, ~2× decode)**; device weight persist deferred |
| RoPE | `main.cpp`, `amd_xdna.cpp` | NPU batched kernel (`rope_batched`, opt-in `GGNPU_NPU_ROPE=1`) + CPU default; correct Llama 3 math |
| Logits projection | `main.cpp` | NPU INT8 mul_mat_q with WeightCache decode (Q4_K/Q6_K); CPU fallback for F32 weights |
| Residual adds on CPU | `main.cpp` | Cheap; acceptable for MVP |
| `execute_layer_graph()` unused | `main.cpp` | Already removed |

#### Fixed — Phase 5 inference quality

| Fix | File |
|-----|------|
| RoPE **GGML_ROPE_TYPE_NORMAL** adjacent pairs `(2i, 2i+1)` — was NeoX half-split (wrong logits) | `src/cli/main.cpp` |
| Llama 3 `rope_freqs` used as **freq_factors**, not angle divisors | `src/cli/main.cpp` |
| Q6_K **per-token row** embedding dequant (bulk INT8 cache produced garbage activations) | `src/cli/main.cpp` |
| Decode-style prefill: one token per forward, KV reset at start | `src/cli/main.cpp` |
| **NPU logits** via INT8 mul_mat_q + WeightCache decode (Q4_K/Q6_K); CPU fallback | `src/cli/main.cpp` |
| `bench-logits` CLI + `scripts/compare_logits.py` + `scripts/test-e2e-logits.sh` (`ctest -R test_e2e_logits`) | `src/cli/main.cpp`, `scripts/`, `CMakeLists.txt` |
| Weight cache key includes `data_size`; clear `~/.cache/ggnpu/weights/` after decoder changes | `include/ggnpu/weight_cache.h` |
| `attach_kquant_scales` validates scale count | `src/cli/main.cpp` |
| France prompt golden tokens in `test_tokenizer` | `tests/test_tokenizer.cpp` |

#### Recently fixed (startup / load perf)

Model load + startup was **~16 s** before first token (dominated everything;
`bench-layer`/inference profiling showed matmul as 85% of *compute*, but wall
clock was mostly load). Three independent issues, all host-side:

| Fix | File | Win |
|-----|------|-----|
| **KV cache allocated at full model context.** `init_kv_cache(0)` during `load()` used the model's reported context (Llama 3.2 = **131072**) → ~8 GB per buffer; combined with the bug below this tried to allocate **17 GB** on a 16 GB host and swap-thrashed for ~13 s, only to be discarded when the CLI recaps to 2048. `load()` now allocates a capped default (`kDefaultCtxCap=2048`); `-c` still raises it via `reinit_kv_cache()`. | `src/arch/llama/llama.cpp` | **13.4 s → 85 ms** |
| **KV cache `2×` double-count.** `keys_`/`values_` are *separate* vectors but `total` carried a `2 *` factor, so each was sized 2× too large. | `src/runtime/kv_cache.cpp` | halves KV memory |
| **GGUF metadata parsed one `::read` syscall per field.** The 128k-entry tokenizer array drove ~**818k** read syscalls. Now batched through a 1 MB userspace buffer (`buffered_read`); `header_end_offset_` tracked logically instead of via `lseek`. | `src/gguf/gguf.cpp`, `include/ggnpu/gguf.h` | KV/parse ~0.2 s |
| **GGUF tensor data was memcpy'd off the mmap.** `map_tensor_data()` did `tensor_data_.assign(mmap…)`, copying the ~0.7 GB weight region onto the heap on every load. `tensor_data()` now returns a `std::span` view into the mmap (zero-copy; pages fault lazily). | `src/gguf/gguf.cpp`, `include/ggnpu/gguf.h` | no ~0.7 GB copy |
| **Clean error for non-llama metadata.** Non-`llama.*` GGUFs (e.g. qwen2 uses `qwen2.*` keys) read head_count=0 and **SIGFPE'd** on `embedding/heads`; now errors cleanly. (Full qwen support still needs arch-prefix-aware metadata reading.) | `src/arch/llama/llama.cpp` | no core dump |

**Result:** Llama 3.2 1B end-to-end wall **~31 s → ~14 s** (n≈5); time-to-first
output dominated by the first-token weight decode + B-tile pack (~6 s), not load.
Steady-state decode ~0.84 s/token. `test_e2e_logits` still passes (top-1
unchanged). Remaining runtime is the INT8 matmul path (per-tile driver
round-trips at the fixed 256³ kernel granularity — see attention note re: no
in-repo bigger-tile generator).

#### Matmul driver round-trips are already hidden — it's device-kernel-bound

Investigated whether the per-tile host↔driver round-trips (~14.8k tiles/token,
each pack A + DMA A/B + start + wait + DMA C) are the matmul floor. **They are
not** — they're already overlapped by the batched submission pipeline, so
removing them yields ~0 net. Two attempts, both measured on hardware:

- **`xrt::runlist` (one execute/wait ioctl per batch instead of one start/wait
  per tile).** *Regressed badly:* `execute()` cost ~4.4 ms **per call** (it
  rebuilds the command chain on every reset+re-add) and serialized kernels
  (kernel-wait 8.5 → 80 µs/tile). runlist is for build-once/run-many, not the
  rapid-reconfigure pattern here. Reverted.
- **Persistent device weight BOs** (pack each weight tile once, bind the resident
  BO on later tokens — kills the redundant per-tile weight memcpy+sync). Worked
  and dropped dmaB 9 → 0.1 µs/tile, but steady-state total only 70.8 → 69.6
  µs/tile (~2%): the freed time reappeared in the wait phase because the weight
  DMA was already hidden behind device-kernel execution. Not worth ~1 GB of
  pinned BOs + 19k allocations. Reverted.

**Conclusion:** steady-state matmul (~70 µs/tile, batch 24) is bounded by the
256³ INT8 **device kernel**, not host/driver overhead. The real lever was a
**small-M kernel**: decode is M=1, but the 256³ kernel computes 256 M-rows, so
255/256 of each tile's row-compute is wasted. This is now **done** — see the
validated small-M kernel below (~20% faster matmul, ~1.0 → 1.37 tok/s). Also
kept the free win: default `GGNPU_MATMUL_BATCH_SIZE` 8 → 24 (~8%).

#### Small-M matmul kernel — **VALIDATED** (hardware-correct + ~20% decode win, 2026-06-30)

A decode-optimized INT8 matmul tile (`matmul_small_m` in `compile_kernels.py`,
transform `matmul_small_m_aie2p.mlir`) shrinks the M tile to **16** (N/K stay
256). Decode is M=1, so the 256³ kernel computes 256 M-rows and wastes 255/256
of each tile; the small-M kernel computes only 16. `mul_mat_q` routes M ≤ 16
calls to it automatically when `matmul_small_m_<profile>.xclbin` is present
(opt out with `GGNPU_NO_SMALL_M=1`). Same INT8 datapath, packing strides, and
B-tile cache as the 256³ path — only the M-tile granularity, kernel handle, and
pipeline slots differ.

**Result (Llama 3.2 1B Q4_K_M, 21 token steps, same build A/B):**

| op | 256³ (`GGNPU_NO_SMALL_M=1`) | small-M | Δ |
|----|------|------|------|
| matmul | 15354 ms | 12305 ms | **−19.9%** |
| logits | 3135 ms | 2586 ms | −17.5% |
| **total** | 18902 ms | 15298 ms | **−19.1%** (1.11 → 1.37 tok/s) |

Output is bit-for-bit equivalent to the 256³ path: `bench-layer` attn_q rel
error is 0.0166 with both kernels; `test_e2e_logits` top-1 unchanged (12366,
` Paris`). The win is bounded below 16× because the small-M kernel uses 8 cores
(2×4) vs the 256³'s 16, and host pack/DMA plus fixed per-core kernel overhead
are unchanged; only the device row-compute shrinks.

**How the transform was fixed (the hard part).** The original probe changed only
PHASE 5's compute herd tile `[8,8,0] → [1,8,0]` (M-packed=2 → 2 M-groups, so A
varies across the herd and promotes to L1 instead of being hoisted as
loop-invariant — the `airrt-to-npu` L3 error). That compiled but was **numerically
wrong**: only the first of 4 N-groups computed (output cols 0–63 correct, 64–255
zero). Root cause: the packed-C buffer is promoted **per-core to L1** (PHASE 3),
so the prologue-fill, compute, and epilogue-unpack herds must all partition it
with the **same grid**. The 256³ kernel has all three at 4×4; the bare
`[1,8,0]` change left the compute herd at (2,4) but the prologue
(`interchange [1,0,2,3]` + `[8,8]`) and epilogue (`[64,64]`) at (4,1) — so the
epilogue read the wrong cores' L1 tiles. Fix: bring all three herds to the
**(M=2, N=4)** grid:
- PHASE 5 compute: `[1,8,0]` (M/1=2 groups, N/8=4 groups).
- PHASE 6 prologue fill: keep `interchange [1,0,2,3]`, tile `[8,8] → [1,8]`.
- PHASE 6 epilogue unpack: tile `[64,64] → [8,64]` (M/8=2, N/64=4).

A 2×8 (16-core) variant does **not** place on npu2 (only 4 compute rows; a
size-8 herd dim overflows), and a larger-M kernel gives the same per-core work
with more waste — so (2,4) at M=16 is the sweet spot.

- **Tooling:** direct `compile_kernels.py` runs need
  `LD_LIBRARY_PATH=third_party/boost-lib/usr/lib/x86_64-linux-gnu` for
  `xclbinutil` (otherwise missing `libboost_filesystem.so.1.90.0`).
  `build-kernels.sh` already sets this, and now builds `matmul_small_m`.
- **Validation hook:** `bench-matmul` includes M ∈ {1, 16} shapes (A=B=1 → C=K
  known-answer) that exercise the small-M kernel directly.

#### Deep-K decode matmul — **VALIDATED** (~2× decode, in-kernel K reduction, 2026-06-30)

After small-M, profiling showed decode matmul was bounded by **fixed per-launch
device overhead** (~13 µs/tile, ~35× the actual MACs), not the MACs or DMA. A
K=2048 matmul tiled K at 256 paid that overhead **8×** (8 separate launches +
host f32 accumulation of the 8 partial Cs). `matmul_small_m_deepk`
(`compile_kernels.py` + the **same** `matmul_small_m_aie2p.mlir` transform) is the
small-M kernel compiled at **K=2048**: the Triton `tl.dot` is unchanged and the
transform's PHASE 4 K-reduction loop (tile factor 8 packed = 64 raw K) simply
iterates 32× instead of 4×, accumulating across all of K in the L1 packed-C
buffer. One launch now replaces 8. No transform edits were needed — only the
launch `BLOCK_SIZE_K`.

`mul_mat_q` routes a small-M (M ≤ 16) matmul to the deep-K kernel when `K %
kDeepK == 0` (`kDeepK = 2048`); opt out with `GGNPU_NO_DEEPK=1`. K=8192
(ffn_down) runs as 4 deep-K spans accumulated on the host. The host path
generalizes the K tile to `T_k` (256 or 2048): A is `[T_m × T_k]`, B is
`[T_k × 256]`, C is `[T_m × 256]` int32; B-tile cache, packing strides, and
batched pipeline are otherwise identical. The per-tile B weight DMA grows to
512 KB but there are 8× fewer tiles, so total cached weight bytes and DMA are
unchanged.

**Result (Llama 3.2 1B Q4_K_M, 21 token steps, same build A/B):**

| op | small-M K=256 (`GGNPU_NO_DEEPK=1`) | deep-K | Δ |
|----|------|------|------|
| matmul | 12211.8 ms | 6206.1 ms | **−49.2%** |
| logits | 2581.4 ms | 1366.1 ms | −47.1% |
| **total** | 15255.0 ms | 7980.5 ms | **−47.7%** (1.38 → 2.63 tok/s) |

Correct on hardware: `bench-matmul` `1×2048×2048` (single span) and `1×8192×256`
(4 spans, host accum) PASS (A=B=1 → C=K); `test_e2e_logits` PASS;
`compare_logits.py --check` → top-1 ` Paris`. The int8 values and int32 partial
sums are bit-identical to the K=256 path; deep-K just replaces 8 host f32
partial-sum adds with one f32 scale, so it is at least as accurate (greedy output
can still diverge after a near-tie). Cumulative vs the original 256³ path:
**2.37× faster decode**.

The next matmul lever is widening N per launch (amortize the same overhead across
N too) and/or cutting the now-exposed 512 KB/tile weight DMA (~19% of per-launch
time), e.g. resident weight BOs — re-evaluate now that device compute shrank
again. Both are kernel-shape work (the matmul transform generator isn't in-repo).

#### Resident weight BOs — **DONE** (decode now device-bound, ~5%, 2026-06-30)

With deep-K shrinking the device kernel, the per-launch **weight DMA** became 64%
of decode matmul (it was hidden behind the 256³ kernel before). INT8 weights are
constant across tokens, so `mul_mat_q` packs each deep-K weight tile into a
**resident device BO** once (`resident_b_bos_`, keyed like the host B-tile cache)
and binds it directly on every later launch — no per-token host→device copy.
Gated to the deep-K decode path (`use_deepk && B_int8_base`); prefill keeps the
per-call staging copy. Opt out `GGNPU_NO_RESIDENT_W=1`.

**Memory-neutral** (the resident BOs replace the pageable B-tile cache, which is
no longer populated on the int8 path): max RSS 2738 MB on vs 2737 MB off. Gives
~5% (matmul 6206 → 5860 ms, 2.63 → 2.77 tok/s); top-1 ` Paris` unchanged.

**Decode is now device-bound.** With resident BOs on, profiling shows the freed
weight-DMA time was overlapping device execution (the NPU runs batched kernels
serially while the host packs the next tile), so the deep-K tile's true device
cost (~26 µs) is now the floor — further host-side work (pack-once, etc.) can't
move the wall. The remaining lever is **device-side**: the ~26 µs tile is still
~9× the raw MAC time, so widening N per launch (amortize per-launch device
overhead across more output cols) or more cores would help — but that is
transform/kernel-shape work (the matmul transform generator isn't in-repo, so the
N tiling in `matmul_small_m_aie2p.mlir` would be hand-edited, like small-M).

#### Recently fixed (kernels / layer bench)

| Fix | File |
|-----|------|
| **RMSNorm N=2048 accuracy:** `rmsnorm_aie2p.mlir` no longer casts `vector.multi_reduction` to bf16 (was ~8% error on hardware); typical <0.5% rel, max ~1.15% on large activations | `kernels/triton/transforms/rmsnorm_aie2p.mlir` |
| **RMSNorm N=2048 on NPU:** removed host f32 fallback; first-call bf16-aware validation | `src/backends/amd_xdna/amd_xdna.cpp` |
| **Flash attention:** default to host f32 decomposed; fused NPU opt-in via `GGNPU_FLASH_ATTN_FUSED=1`; removed from default `build-kernels.sh` | `amd_xdna.cpp`, `build-kernels.sh`, `flash_attn_aie2p.mlir` |
| **bench-layer RMSNorm:** CPU ref uses bf16-roundtrip input; 1% rel threshold | `src/cli/main.cpp` |
| Removed misleading "SiLU uses CPU fallback" comment from `bench-layer` test (NPU path is actually exercised) | `src/cli/main.cpp` |
| Added clarifying comments for CPU fallbacks: logits projection, residual adds | `src/cli/main.cpp` |
| **bf16 marshaling gate:** `create_rmsnorm/softmax/silu_kernel_from_loaded_xclbin` now properly load instruction sequences and create kernels (were returning false) | `src/backends/amd_xdna/amd_xdna.cpp` |
| **RMSNorm shape:** prebuilt kernel now accepted for N=2048 (Llama hidden) in addition to N=256 | `src/backends/amd_xdna/amd_xdna.cpp` |
| **FlashAttention kernel setup:** `create_flash_attn_kernel_from_loaded_xclbin` now creates kernel from loaded xclbin (was returning false) | `src/backends/amd_xdna/amd_xdna.cpp` |
| **SiLU arbitrary sizes:** JIT compilation path for sizes != 8192; hard error if JIT unavailable | `src/backends/amd_xdna/amd_xdna.cpp` |
| **KV cache override:** `reinit_kv_cache()` public method added; CLI `-c/--ctx-size` now reinitializes KV cache | `model.h`, `llama.cpp`, `main.cpp` |
| **RMSNorm arbitrary sizes:** JIT compilation path for sizes != 256/2048; hard error if JIT unavailable | `src/backends/amd_xdna/amd_xdna.cpp` |

#### Known limitation: fused flash attention does not map to an AIE herd

**Issue (verified 2026-06):** the fused flash-attention kernel compiles through
the custom transform (`flash_attn_aie2p.mlir`) and produces valid AIR, but
**that AIR contains no `air.herd`** — all four reductions and both matmuls stay
at the `air.segment` (L2) level and are never tiled onto AIE cores. The build
then fails in the final `airrt-to-npu` lowering:
```
error: failed to legalize operation 'airrt.dma_memcpy_nd' ... explicitly marked illegal
error: failed to legalize operation 'airrt.segment_load' ... explicitly marked illegal
```
(The older "requires exactly one producer_op handle (got N)" error is stale — it
came from `split_handle` expecting 2 reduces when softmax decomposition adds
`max` + `sum` = 4. The current transform gets past that; the herd-mapping is the
real wall.)

**Root cause:** the transform tiles a single `forall [1]` over head_dim and tries
to fuse all four reduction chains (QK sum, softmax max, softmax sum, AV sum) into
it. The fusion does not produce an AIE-core herd for a 4-reduction dependency
graph, so the work is left on the memtile. Confirmed by `grep air.herd
asm_air_output.mlir` → 0 (working rmsnorm/rope kernels → 1). This is a transform
/ AIR mapping limitation for multi-reduction kernels, not a single missing pass.

**Workaround (production):** host f32 decomposed flash attention
(`flash_attn_decomposed()` in `amd_xdna.cpp`). Matches CPU reference; no xclbin.

**Recommended path forward — decompose, don't fuse.** Each attention sub-step is
a *single*-reduction op that the proven rmsnorm-style transform already maps to a
herd, so run them as three NPU kernels orchestrated by the host:
1. **QK matvec** `scores[j] = Σ_d Q[d]·K[j,d]` — per-row reduce over head_dim
   (identical shape to rmsnorm's reduction).
2. **softmax** — existing `softmax_npu6.xclbin`.
3. **AV matvec** `out[d] = Σ_j w[j]·V[j,d]` — per-row reduce over ctx (the
   transpose+reduce already present in the AIR).

Variable ctx is handled **on the host between fixed-size kernels** (no in-kernel
masking → no dynamic DMA): pad K/V to a fixed `N_pad`; padded QK scores are 0;
the host sets scores[j≥ctx] = -inf before softmax; padded softmax weights ≈ 0 so
padded AV terms vanish. Use bf16 (not the INT8 matmul kernel, which loses
attention-logit coherence). This is a sizable feature (2 new matvec kernels +
transforms + host orchestration + per-head loop), tracked as future work.

**Progress (WIP).** Two approaches were explored:

1. *rmsnorm-derived QK transform* (`attn_qk` + `attn_qk_aie2p.mlir`): promotes
   both reduction inputs (Q broadcast + K), reaching AIE herd placement, but hits
   a structural wall — rmsnorm produces a 2-D elementwise output while QK is a
   **matvec with a 1-D output**, so the 2-D pipeline (output-generic navigation,
   `[0,16]` vectorization) doesn't fit. Abandoned in favor of (2).

2. **bf16 matmul datapath — WORKS, hardware-validated.** `matmul_bf16`
   (`compile_kernels.py` + `matmul_bf16_aie2p.mlir`) is the int8 matmul transform
   with the `vector.contract` casts retargeted (`i16 → bf16` inputs, `i32 → f32`
   accumulator) — the int8 path already widens to 16-bit i16, and bf16 is 16-bit,
   so the `pack=[8,8,8]` tiling is unchanged. It **compiles end-to-end** to a
   67 KB xclbin (AIR: bf16 contracts, f32 accumulation, 3 herds) and **runs
   correctly on the NPU**: a 256³ GEMM vs a torch reference gives rel err 0.0099,
   0/65536 mismatches (`scripts/validate-bf16-matmul.py`). This is the foundation
   for decomposed NPU attention: QK = K[ctx,d] @ Q[d,1] and AV = P[1,ctx] @
   V[ctx,d] as bf16 GEMMs (far better than int8 for attention-logit coherence).
   Build with `GGNPU_EXPERIMENTAL=1 ./scripts/build-kernels.sh npu6 matmul_bf16`.

**Done — decomposed NPU attention works end-to-end (opt-in `GGNPU_NPU_ATTN=1`).**
- `Backend::matmul_bf16(A, B, C, M, N, K)` (`amd_xdna.cpp`): f32 host buffers,
  bf16 NPU compute; host tiles M/N/K into 256-blocks, zero-pads edges,
  accumulates K-blocks in f32.
- `flash_attn_npu()`: per head, QK = `Kh[cl,hd] @ Qh[hd,1]` and AV =
  `w[1,cl] @ Vh[cl,hd]` run via `matmul_bf16`; scale + causal mask + softmax on
  the host between them. Dispatched from `flash_attn()` behind `GGNPU_NPU_ATTN`,
  CPU fallback on failure (same pattern as RoPE).
- `bench-layer` validates on hardware: bf16 matmul 256³/partial/QK-shape (rel
  0.006–0.007); flash_attn 1-head ctx=1 (0.0026) and **4-head ctx=40 causal**
  (0.0062, full QK→softmax→AV). End-to-end (reduced layers) matches the host
  baseline.

**Perf (why it stays opt-in).** Profiling (`GGNPU_NPU_ATTN_TIMING=1`) drove a
host-side fix: `matmul_bf16` now packs only the real `[rows,cols]` sub-region to
bf16 (persistent staging, padding kept zero across calls) instead of zeroing +
converting the full 256² tile every launch. Per 256 launches the host overhead
dropped ~207 → ~44 ms (pack 124 → 1.5 ms, 80×). Full-model NPU attention went
from *not finishing in 300 s* to **~34.5 s for 6 tokens** (coherent output).

After the host fix, **NPU attention is at near-parity with the host path**: full
model, 6 tokens, the same prompt → 32.7 s (host) vs 34.7 s (NPU attention), a
~6 % difference. So attention is *not* the runtime bottleneck — model load plus
the int8 projection/FFN matmuls dominate. The remaining `run+wait` waste (the
fixed 256³ kernel doing 16 M MACs when the degenerate dim needs a sliver) is only
~2 s, **not worth** the higher-risk attention-shaped-xclbin work (small N/M L1
tile; the matmul transform generator isn't in-repo, so `matmul_bf16_aie2p.mlir`'s
`l1=64` tiling would need hand-editing). NPU attention stays opt-in (bf16 vs the
host's exact f32); host f32 `flash_attn_decomposed()` remains the default. If
perf is pursued further, target the int8 matmul path, not attention.

> Running a kernel directly on the NPU from Python (no C++) needs three env
> fixes the build path doesn't set: `LD_PRELOAD=libuuid.so.1` (launcher needs
> `uuid_unparse_lower`), the bundled boost lib on `LD_LIBRARY_PATH` (xclbinutil),
> and the system XRT runtime (`libxrt_core.so.2` etc.) reachable under
> `$XILINX_XRT/lib`. `scripts/validate-bf16-matmul.py` wraps all three.

#### Recently fixed

| Fix | File |
|-----|------|
| **Flash attention kernel:** rewrote from scf.for loop-carried online-softmax to elementwise + reductions (q_bcast * k_row → tl.sum, softmax decomposition, weights * v_row → tl.sum). Transform script follows rmsnorm/softmax patterns. Blocked on Triton-XDNA multi-reduction lowering (§7.1 Known limitation). | `compile_kernels.py`, `flash_attn_aie2p.mlir` |
| **Flash attention:** default to host f32 decomposed; fused NPU opt-in via `GGNPU_FLASH_ATTN_FUSED=1`; removed from default `build-kernels.sh` | `amd_xdna.cpp`, `build-kernels.sh`, `flash_attn_aie2p.mlir` |
| **RoPE precomputation:** cos/sin tables built once at startup (indexed by [position][dim/2]); apply_rope does table lookups instead of per-call sin/cos | `src/cli/main.cpp` |
| **KV expand buffer preallocation:** k_expanded/v_expanded allocated once to max size; resize reuses capacity — eliminates two vector allocations per token | `src/cli/main.cpp` |
| **Dead code removal:** removed unused execute_tile lambda (duplicate of flush_batch) from mul_mat_q path in amd_xdna.cpp | `amd_xdna.cpp` |


|-----|------|
| GGUF STRING arrays (`tokenizer.ggml.tokens`, `merges`) were skipped during load → 0 vocab; now serialized into `GgufKV::data` | `src/gguf/gguf.cpp` |
| Llama-bpe tokenizer: unicode pretokenize + ranked BPE merges + GPT-2 byte decode | `src/cli/tokenizer.cpp`, `src/cli/unicode*.cpp` |
| RMSNorm learned weights (`attn_norm`, `ffn_norm`, `output_norm`) wired into inference | `backend.h`, `main.cpp`, `cpu_ref.cpp`, `amd_xdna.cpp` |
| SwiGLU fixed to `silu(gate) * up` (was `gate * silu(up)`) | `src/cli/main.cpp` |
| Causal masking in CPU flash_attn (`query_pos`) | `backend.h`, `cpu_ref.cpp`, `amd_xdna.cpp`, `main.cpp` |
| `register_xclbin` + `xrt::hw_context` instead of unsupported `load_xclbin` | `src/backends/amd_xdna/amd_xdna.cpp` |
| `xrt::run::wait()` after kernel launch (runs were fire-and-forget, output stayed zero) | `src/backends/amd_xdna/amd_xdna.cpp` |
| Matmul: f32↔int8 conversion + host tiling onto fixed 256³ INT8 kernel | `src/backends/amd_xdna/amd_xdna.cpp` |
| Instruction sequence (`*_sequence.bin`) loaded into cacheable BO, passed via opcode-3 convention | `src/backends/amd_xdna/{amd_xdna,kernels}.cpp` |

#### Fixed — Phase 3 gate

| Fix | File |
|-----|------|
| Q4_K/Q6_K decoders rewritten to real ggml layouts (144 B / 210 B super-blocks, fp16 d/dmin, 6-bit / int8 scales); previous layouts were fictional | `src/quant/{q4_k,q6_k}.cpp`, `include/ggnpu/quant/kquant.h` |
| `GgmlType` enum aligned with ggml type ids (Q4_K=12, Q6_K=14; was 10/12, so Q4_K tensors decoded as Q6_K) + corrected type/block size tables | `include/ggnpu/tensor.h`, `src/gguf/tensor.cpp` |
| GGUF tensor offsets: use header offsets relative to aligned data section instead of recomputing cumulatively (every tensor pointer was wrong) | `src/gguf/gguf.cpp` |
| INT8 matmul scaling: per-tensor weight scale (from decoder) × per-call dynamic activation scale applied to outputs; decoded weights indexed row-major N×K | `src/backends/amd_xdna/amd_xdna.cpp` |
| CPU reference Q4_K/Q6_K matmul uses real dequant with per-row block offsets | `src/backends/cpu_ref/cpu_ref.cpp` |
| `bench-layer` passes scales for gate/up in SwiGLU test; `token_embd` dequant applies per-tensor scale | `src/cli/main.cpp` |

#### Previously fixed

| Fix | File |
|-----|------|
| `current_tokens` seeded from prompt | `main.cpp` |
| KV cache respects `-c`; default ctx capped to 2048 | `llama.cpp`, `main.cpp` |
| Q4_K `token_embd` dequant | `main.cpp` |
| Per-token prefill buffers (no embedding sum) | `main.cpp` |
| Attention output matmul dims + per-head flash_attn | `main.cpp` |
| Q4_K scales passed to `mul_mat_q` | `main.cpp` |
| CMake finds Ubuntu XRT via `FindXRT.cmake` | `CMakeLists.txt` |

**KV RAM formula:** `2 × n_layers × n_ctx × n_head_kv × head_dim × 4` bytes.

For Llama 3.2 1B at metadata ctx 131072: `2 × 16 × 131072 × 8 × 64 × 4 ≈ 8.6 GB`. At MVP ctx 2048: **~128 MB**.

#### What works today

```bash
# Host check
bash scripts/verify-npu.sh

# Native build + validated NPU matmul (Phase 2 gate — PASSES)
cmake -S . -B build-npu -DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=OFF -DGGNPU_BUILD_TESTS=ON
cmake --build build-npu -j1
./build-npu/ggnpu bench-matmul
# → all sizes validate; ~1 ms per 256³ tile (8192³ takes minutes due to host tiling)

# Phase 3 gate — PASSES: FFN gate/up/down matmuls from GGUF on NPU vs CPU ref
./build-npu/ggnpu bench-layer -m models/llama-3.2-1b-q4_k_m.gguf --layer 0
# → RMSNorm, attn_q/k/v/output, flash_attn, SiLU, FFN matmuls, full layer forward

# KV cache respects -c flag (Phase 4 fix)
./build-npu/ggnpu -m models/llama-3.2-1b-q4_k_m.gguf --dump-tensors -c 2048
# → KV cache allocated at 2048 tokens, not model metadata ctx

# Unit tests (CPU reference backend)
cd build && ctest

# E2E logits regression (requires models/llama-3.2-1b-q4_k_m.gguf)
ctest -R test_e2e_logits
python3 scripts/compare_logits.py --check          # top-1 Paris
python3 scripts/compare_logits.py --check --ref  # + logit vs llama-cpp-python

# Phase 5 gate — PASSES: coherent generation
./build-npu/ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 8
./build-npu/ggnpu bench-logits -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048
```

#### What does not work today

- **Fused NPU flash attention** — blocked: 4-reduction kernel maps to AIR with no `air.herd`, so `airrt-to-npu` rejects it (§7.1 Known limitation). Recommended fix: decompose into 3 single-reduction NPU kernels (QK → softmax → AV). Host f32 decomposed path is production.
- **NPU RoPE** — gather/index math not lowered by Triton-XDNA transforms yet.
- **High NPU utilization** — flash_attn, RoPE, residuals still on host; matmul (incl. logits INT8) and norms/activations on NPU.
- **Stale rmsnorm xclbins** — pre-fix `rmsnorm_2048_npu6.xclbin` artifacts give ~8% error; delete and rebuild (see §2.1).
- **No CI** — run `ctest` and `compare_logits.py --check` locally after changes.

#### Path forward (next work, in order)

1. **Fused flash attention on NPU** — blocked on upstream Triton-XDNA fix for multi-reduction kernels. Workaround: host f32 decomposed path is production-quality. See §7.1 Known limitation for kernel restructuring options.
2. **RoPE on NPU** — working transform for gather/pair-shuffle loads.
3. **Matmul perf:** batched tiles (done), small-M decode kernel (done), **deep-K in-kernel reduction (done, ~2× decode)**, **resident weight BOs (done, ~5%; decode now device-bound)**. Remaining is device-side: widen N per launch (amortize per-launch device overhead across N) or more cores — transform/kernel-shape work.
4. **Phase 6:** 3B model, L2-aware tiling, production docs/errors, optional prebuilt xclbin distribution.

### 7.2 Memory constraints (16 GB RAM dev machine)

Dev laptop has ~14 GiB RAM + 4 GiB swap. Cursor IDE can consume several GB; combined with an oversized KV cache this is the highest OOM risk.

| Activity | RAM impact | Recommendation |
|----------|------------|----------------|
| Native `ggnpu` build | ~2–4 GB at `-j1`; **much higher** at `-j2+` | **Always** `cmake --build … -j1` (§16) |
| Local Triton-XDNA kernel build | **Low** (`pip install triton-xdna`) | No heavy build step required |
| Load Llama 1B Q4_K_M (mmap) | ~770 MB virtual | Fine |
| KV at **131k ctx** (current code) | **~8.6 GB** | **Will OOM** with IDE open |
| KV at **2048 ctx** (after fix) | **~128 MB** | Comfortable |
| Weight decode disk cache | ~1–2 GB first run | Disk, not RAM |
| NPU pinned buffers | Tens–hundreds of MB | Requires memlock unlimited |

**RAM hygiene before inference:**

- Run `ggnpu` from a plain terminal outside the IDE when memory is tight
- Do not run `GGNPU_BUILD_KERNELS=ON` on 16 GB RAM (but Triton-XDNA `pip install` is lightweight)
- Start with `models/llama-3.2-1b-q4_k_m.gguf` only; avoid 3B+ until KV is fixed
- Set `ulimit -l unlimited` in the shell session

---

### 7.3 Gemma 3n (`gemma4`) support roadmap

`models/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` reports `general.architecture =
gemma4` and is **Gemma 3n E2B** (MatFormer family). This is the next model-breadth
target. Unlike Qwen2 (a small metadata delta over Llama — QKV bias + rope fallback,
§ multi-arch), Gemma 3n is one of the densest open architectures and is a
**multi-session effort**. This section is the plan of record; update the checkboxes
as phases land.

#### Feature inventory (verified via `--dump-tensors`)

| Feature | Evidence | Runtime delta |
|---------|----------|---------------|
| **SentencePiece *unigram* tokenizer** | `tokenizer.ggml.model=gemma4`, `scores` + `token_type` arrays, 262144 vocab | Current tokenizer (`src/cli/tokenizer.cpp`) is **BPE-only** (llama3/gpt2 merge-rank). Unigram is a different algorithm (score-maximizing segmentation) — a real addition, not config. |
| **Per-Layer Embeddings (PLE)** | `per_layer_token_embd` [8960×262144], `per_layer_model_proj`, `per_layer_proj_norm`, per-block `inp_gate` [1536,256] / `proj` [256,1536] | A second embedding pathway: look up a 256-dim per-layer vector per token and inject it into every block via gating. New forward substructure. |
| **QK-norm** | `blk.N.attn_q_norm` / `attn_k_norm` (F32) | Per-head RMSNorm applied to Q and K before attention. |
| **Sandwich norms (5/block)** | `attn_norm`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm`, `post_norm` | Gemma pre+post norm pattern; RMSNorm uses `1+w` weight convention and √hidden embedding scaling. |
| **Heterogeneous attention (SWA vs global)** | `sliding_window=512`, `sliding_window_pattern` (35-elem array), `key_length=512` / `key_length_swa=256`, `rope.freq_base=1e6` / `rope.freq_base_swa=1e4`; `blk.0.attn_q_norm=[256]` vs `blk.34.attn_q_norm=[512]` | Per-layer: local layers use head_dim 256 + windowed mask + rope base 1e4; global layers use head_dim 512 + full mask + rope base 1e6. head_dim is decoupled from embedding (1536/8=192≠256). |
| **Shared KV across layers** | `shared_kv_layers=20`; e.g. `blk.20` has **no** `attn_k`/`attn_v`/`attn_k_norm` | Some layers reuse a preceding layer's KV projections instead of computing their own. Needs a layer→KV-source mapping. |
| **MatFormer variable FFN widths** | `feed_forward_length` is an **array** (blk.0=6144, blk.20/34=12288) | FFN width varies per layer; loader must read the array, forward must not assume a constant. |
| **Per-layer output scale** | `blk.N.layer_output_scale` [1] (F32) | Scalar multiply on each block's output. |
| **Final logit soft-capping** | `final_logit_softcapping=30` | `logits = 30 * tanh(logits/30)` before sampling. |

Other metadata: 35 layers, hidden 1536, 8 attn / 1 KV head, `rms_eps=1e-6`,
`embedding_length_per_layer_input=256`, ctx 131072 (use `-c 2048`), non-tied
(`output.weight` present via `output_norm` + main embed — confirm at load).

#### Phased plan (correctness-first, host-hybrid — same path Llama/Qwen2 took: matmul/norm/silu on NPU, orchestration on host)

Each phase is independently verifiable; do not start a phase before the prior one's milestone passes.

- [x] **Phase G1 — Load + metadata + tokenizer.** *(done 2026-07-01)*
  - Array-valued metadata parsing: `GgufKV` now records `array_type`/`array_length`; `GgufLoader::get_int_array` / `get_float_array` decode numeric arrays (`src/gguf/`). Verified on gemma4 `feed_forward_length` (35× i32: 6144×15 then 12288×20) and `attention.sliding_window_pattern` (35× bool: every 5th layer = global).
  - SentencePiece-**unigram** tokenizer in `src/cli/tokenizer.cpp`: score-max merge (llama.cpp `llm_tokenizer_spm` algorithm) over `tokens`+`scores`, meta-space (U+2581) normalization with `add_space_prefix=false`, `<0xXX>` byte fallback, unigram decode path. Auto-selected via `tokenizer.ggml.model` (`gpt2` ⇒ BPE, else scores ⇒ unigram) — note gemma ships **both** merges and scores, so merge-presence is not a BPE signal.
  - **Milestone met:** `--dump-tensors` exits clean; `test_tokenizer` (dispatches on tokenizer type) passes gemma goldens `"The capital of France is"→[818,5279,529,7001,563]`, `"Hello, world!"`, a code snippet, plus round-trip; registered as ctest `test_tokenizer_gemma`. Full ctest suite green (8/8). No Llama/Qwen regression.
- [ ] **Phase G2 — Forward skeleton (no PLE, all-global attention).**
  - Embedding √hidden scale, `1+w` RMSNorm, 5 sandwich norms, QK-norm, GQA (8:1), head_dim from `key_length`.
  - Treat every layer as global (ignore SWA, ignore shared-KV — give each layer its own KV) to validate plumbing first.
  - **Milestone:** runs end-to-end, output is roughly on-topic (not yet reference-correct).
- [ ] **Phase G3 — PLE pathway.**
  - Second embedding lookup + `per_layer_model_proj`/`per_layer_proj_norm` + per-block `inp_gate`/`proj` injection.
  - **Milestone:** output quality jumps toward reference; `compare_logits.py` gap narrows sharply.
- [ ] **Phase G4 — Heterogeneous attention.**
  - Apply `sliding_window_pattern`: windowed mask + dual rope base + per-layer head_dim (256 local / 512 global); implement shared-KV layer→source mapping (`shared_kv_layers`).
  - **Milestone:** logits within tolerance of reference across a multi-token prompt.
- [ ] **Phase G5 — Finishing + validation.**
  - `layer_output_scale`, final logit soft-cap, then full E2E vs llama.cpp reference logits.
  - NPU kernel size checks: head_dim 512, FFN 12288, 256-dim per-head QK-norm — confirm existing host-tiled matmul/rmsnorm absorb these or add tiling (cf. [[npu-rmsnorm-pad-pow2]], SiLU host-tiling).

#### Risks / open questions

- **Unigram tokenizer** is the biggest non-forward piece; get a golden reference early (llama.cpp `--verbose-prompt` or HF tokenizer) to avoid debugging tokenization and model math simultaneously.
- **Shared-KV semantics** (`shared_kv_layers=20`): confirm exactly which layers share and from where by checking which blocks omit `attn_k`/`attn_v` (blk.20+ observed). Get the mapping right or attention silently corrupts.
- **head_dim 512 global layers** may stress NPU attention/matmul kernels sized around ≤128 head_dim; the decomposed host-f32 attention path is the safe fallback while validating.
- Memory: 2.6 GB Q4 file + PLE table; fine on the dev box only with `-c 2048` (131k ctx KV would OOM).

---

## 8. CLI reference

Implement all flags below. `ggnpu --help` must match `docs/usage.md`.

### Commands

| Command | Purpose |
|---------|---------|
| `ggnpu` (default) | Text generation |
| `ggnpu --dump-tensors` | List tensors; no NPU |
| `ggnpu bench-matmul` | NPU matmul benchmark |
| `ggnpu --version` | Version + XRT version |

### Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model` | `-m` | required | Path to `.gguf` |
| `--prompt` | `-p` | `""` | Input prompt |
| `--max-tokens` | `-n` | `128` | Max new tokens |
| `--ctx-size` | `-c` | model default | Context window |
| `--threads` | `-t` | `4` | CPU threads for I/O only |
| `--temp` | | `0.0` | Temperature (`0` = greedy) |
| `--seed` | | `0` | RNG seed (`0` = random) |
| `--npu-device` | | `0` | NPU index |
| `--no-cache` | | off | Disable caches |
| `--cache-dir` | | `~/.cache/ggnpu` | Cache directory |
| `--verbose` | `-v` | off | Per-op timings |
| `--help` | `-h` | | Print help |
| `--version` | | | Print version |

### Examples (put in `docs/usage.md`)

```bash
# Basic
ggnpu -m llama-3.2-1b-Q4_K_M.gguf -p "The capital of France is" -n 64

# Sampling
ggnpu -m model.gguf -p "Once upon a time" -c 4096 -n 256 --temp 0.7 --seed 42

# Inspect model
ggnpu -m model.gguf --dump-tensors
```

**Doc rules:** `docs/usage.md` ≤ 80 lines; every new flag → code + `--help` + docs + test.

---

## 9. Host prerequisites (native deployment)

`ggnpu` is now intended to build and run on the host.

### Required on host

| Item | Purpose |
|------|---------|
| Linux + `amdxdna` | NPU kernel driver |
| `/dev/accel/accel0` | NPU device node |
| `/usr/lib/firmware/amdnpu` | Firmware path |
| `libxrt2`, `libxrt-npu2`, `libxrt-dev` | Runtime + headers |
| CMake + C++ toolchain | Build `ggnpu` |
| User in `render` group | Open accel device |
| `amd_iommu=on` | Boot parameter |
| BIOS NPU/IPU enabled | Hardware |

### One-time host setup

```bash
# Driver (if not already present on your distro kernel)
sudo apt install amdxdna-dkms   # or use distro kernel with amdxdna built-in

# XRT + build tools
sudo apt install libxrt2 libxrt-npu2 libxrt-dev build-essential cmake git

# accel0 access
sudo usermod -aG render $USER
# Log out and back in

# Verify hardware
lspci -vd 1022:17f0
lsmod | grep amdxdna
ls -la /dev/accel/accel0
bash scripts/verify-npu.sh
```

### Build `ggnpu`

```bash
mkdir -p build-npu && cd build-npu
cmake .. -DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=OFF -DGGNPU_BUILD_TESTS=ON
cmake --build . -j1
```

### Kernel artifacts

Place prebuilt `.xclbin` files in `~/.cache/ggnpu/xclbin/`, or build them locally with Triton-XDNA:

```bash
bash scripts/setup-triton-env.sh
source ~/triton-env/bin/activate
./scripts/build-kernels.sh npu6 matmul
```

Kernel authoring uses Triton-XDNA (`kernels/triton/compile_kernels.py`). IRON/`XAie_TxnOpcode` references in this document describe the XRT control plane inside compiled xclbins, not a separate ggnpu build dependency.

---

## 10. Native run flow

**All user-facing commands now run on the host.** See `docs/host-setup-guide.md` for the full guide.

### Quick start

```bash
cmake -S . -B build-npu -DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=OFF -DGGNPU_BUILD_TESTS=ON
cmake --build build-npu -j1
bash scripts/setup-triton-env.sh && source ~/triton-env/bin/activate
./scripts/build-kernels.sh npu6 matmul

./build-npu/ggnpu bench-matmul
./build-npu/ggnpu \
  -m models/llama-3.2-1b-q4_k_m.gguf -p "Hello" -c 2048 -n 32
```

**Requirements:** XRT installed on the host, `render` group membership, memlock configured, and local `.xclbin` kernels available.

---

## 11. Testing

| Level | What | Where |
|-------|------|-------|
| Unit | GGUF parse, quant decode | CPU, CI |
| Unit | Graph, KV indexing | CPU, CI |
| Integration | Each NPU op vs CPU ref | Krackan hardware |
| Golden | 1-layer L∞ diff < ε | Hardware |
| E2E | 16-token generation stable | Hardware |
| Perf | tok/s prefill/decode | Hardware |

Correctness: independent CPU reference math (do not link llama.cpp). Optional cross-check script vs `llama-cli`.

---

## 12. Performance expectations

Do not promise CUDA-class speed in v1.

| Milestone | Expectation |
|-----------|-------------|
| Phase 2 matmul | Correctness + NPU utilization |
| Phase 5 MVP | Single-digit to low tens tok/s for 1B |
| Phase 6+ | Improve via fused attention + L2 tiling |

---

## 13. Risks

| Risk | Mitigation |
|------|------------|
| Q4_K mixed tensor types | Q4_K + Q6_K decoders done (real ggml layouts); embedding dequant in CLI done |
| First-run JIT latency | Prebuilt xclbins + persistent cache |
| XRT/driver/firmware skew | Pin versions; `verify-npu.sh` checks |
| 4 MB L2 limit | Tile matmuls; stream from DDR |
| Triton-XDNA complexity | Start from upstream examples + [IRON tutorial PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-ipdps-2025.pdf) |
| Triton-XDNA / xclbin build OOM | `pip install triton-xdna` is lightweight; no heavy build step |
| KV cache over-allocation | `init_kv_cache()` uses full GGUF `context_length`; must respect `-c` before inference on 16 GB hosts — **fixed** via `reinit_kv_cache()` |
| GGUF models with 128k+ ctx metadata | Cap KV at CLI `-c` or sensible MVP default (2048) regardless of metadata |
| IDE memory pressure (Cursor) | Run heavy builds and inference outside IDE; **always** `cmake --build … -j1` |

---

## 14. Key references

| Resource | URL |
|----------|-----|
| GGUF spec | https://github.com/ggml-org/ggml/blob/master/docs/gguf.md |
| K-quants | https://github.com/iuliaturc/gguf-docs/blob/main/k-quants.md |
| Kernel amdxdna docs | https://docs.kernel.org/accel/amdxdna/amdnpu.html |
| xdna-driver | https://github.com/amd/xdna-driver |
| XRT native APIs | https://xilinx.github.io/XRT/2024.2/html/xrt_native_apis.html |
| Triton-XDNA (AIE2P transforms) | https://github.com/amd/Triton-XDNA |
| MLIR-AIR / MLIR-AIE | https://github.com/Xilinx/mlir-aie |
| OllamaAMDNPU (matmul reference) | https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU |

---

## 15. Definition of done (v1)

On **Ubuntu 24.04 or 26.04** + **Ryzen AI 7 350**, via the native host flow:

1. User installs the NPU driver, XRT, and build prerequisites on host (§9).
2. User builds `ggnpu` locally.
3. User downloads a stock HuggingFace GGUF into `models/`.
4. User provides or builds required `.xclbin` kernels locally.
5. User runs `./build-npu/ggnpu -m models/model.gguf -p "..."` with no preprocessing.
6. `README`, `docs/host-setup-guide.md`, and `ggnpu --help` explain how to run it.

---

## 16. Agent instructions

1. Read this file completely before writing code.
2. Execute phases 0→6 in order; complete each "Done when" gate before proceeding.
3. Match existing repo conventions; keep diffs focused.
4. Never add forbidden dependencies.
5. **Never implement CPU fallback for NPU ops.** CPU tensor math is allowed only when:
   - The op is intentionally **Host** or **CPU** in §2.1 (control plane, or not wired to NPU yet).
   - `cpu_ref` is used as a **reference / source of truth** in tests (`GGNPU_TEST_CPU=1`), not in production inference.
   - The work genuinely belongs on CPU (GGUF decode, tokenization, sampling, cheap post-processing after an NPU kernel).
   
   Ops marked **NPU** in §2.1 must run on the NPU or **fail with an error**. If an xclbin is missing, a kernel launch fails, validation fails, or output is wrong, propagate the error and stop — **do not** catch the failure and run the same tensor op on CPU. Fixing the NPU path is the only acceptable fix; a CPU workaround is forbidden even as a temporary stub.
6. **Use `-j1` for all builds** on the 16 GB dev machine. Always run `cmake --build <dir> -j1` (or `make -j1`). Do not use `-j2`, `-j$(nproc)`, or unconstrained Ninja parallelism — parallel C++ compilation plus the IDE is a common OOM trigger (see §7.2).
7. Add tests for every parser/quant/graph component; NPU tests marked `MANUAL`.
8. Update `docs/usage.md` and `--help` whenever CLI flags change.
9. Commit logically per phase; write clear commit messages.

### Triton kernel rules (enforce on all code generation)

When generating NPU kernel code (`kernels/triton/`), **always** apply the four guardrails from Section 2:

- **Memory-first:** Block-based DMA, no un-chunked reads, overlap compute with streaming.
- **Vector intrinsics only:** No scalar `+`/`*` in kernels. Use AIE vector ops. Types: INT8, BF16, FP8.
- **No branches:** Zero `if/else`/`switch`/`while` in hot loops. Predication or lookup tables only.
- **Two layers:** Control code (IRON API, DMA setup) never contains tensor math. Kernel code (Triton-XDNA compiled) never handles DMA or launch.

**Start here:** Phases 1–5 MVP passed on hardware; Phase 6 in progress. NPU: matmul (incl. logits INT8), RMSNorm (any hidden, pad-to-pow2) + SiLU + RoPE (batched, opt-in `GGNPU_NPU_ROPE=1`). Host: flash_attn. Regression: `ctest -R test_e2e_logits`, `python3 scripts/compare_logits.py --check`. Next work, in order:

1. **NPU attention** — decomposed path **works, opt-in** (`GGNPU_NPU_ATTN=1`): QK/AV as bf16 GEMMs (`matmul_bf16`) + host softmax (`flash_attn_npu`). Validated, but slow due to 256³ tile padding on attention's degenerate dims — next is perf (QK/AV-shaped xclbins, head batching) before it can replace the host f32 default. Fused single-kernel attention remains blocked (§7.1 Known limitation, no `air.herd`).
2. **RoPE on NPU perf** — wired and validated (`rope_batched`, ~8 heads/launch); opt-in until per-token launch overhead is tuned to beat CPU.
3. **Matmul perf:** batched tiles + small-M + **deep-K (done, ~2× decode, `matmul_small_m_deepk`)** + **resident weight BOs (done, ~5%)**. Decode is now device-bound; remaining lever is device-side (widen N per launch / more cores), hand-edited transform work.
4. **Phase 6:** 3B, L2 tiling, production polish; keep `README.md`, `docs/host-setup-guide.md`, and §9–§10 aligned.
5. **Hygiene:** rebuild `rmsnorm_2048` after mlir changes; clear `~/.cache/ggnpu/weights/` after Q4_K/Q6_K decoder changes.

Hardware-facing conventions an agent must know (all in `src/backends/amd_xdna/`):

- xclbins load via `device.register_xclbin()` + `xrt::hw_context`; `device.load_xclbin()` fails on amdxdna.
- Kernel name in every Triton-XDNA xclbin is `MLIR_AIE`; launch convention is `run(opcode=3, bo_instr, n_instr_words, bo0, bo1, bo2)` where `bo_instr` holds the `*_sequence.bin` words in a `XCL_BO_FLAGS_CACHEABLE` buffer at `group_id(1)`.
- Runs are async: always `run.wait()` before reading output.
- Each xclbin is compiled fixed-shape (see `kernels/triton/compile_kernels.py`): matmul 256³ int8→int32, rmsnorm M=2×N=2048 bf16, softmax 256×256 bf16, silu N=8192 bf16. flash_attn builds are experimental (`GGNPU_EXPERIMENTAL=1`).

See §7.1 for full blocker list and §7.2 for RAM constraints.

---

## 17. Local test models

Put GGUF files in `models/` for development and E2E tests. This directory is **not committed**.

**`.gitignore`** (repo root):

```gitignore
# Local GGUF models for testing (not committed)
models/
```

### Models on dev machine

| File | Approx. size | Notes |
|------|--------------|-------|
| `models/llama-3.2-1b-q4_k_m.gguf` | ~770 MB | **MVP target**; ctx metadata 131072 — use `-c 2048` after KV fix |
| `models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | ~1.1 GB | P1 arch; use after Llama 1B works |
| `models/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | ~2.6 GB | **Gemma 3n E2B** (`gemma4`); not yet supported — see §7.3 roadmap. Use `-c 2048`. |

Llama 3.2 1B metadata (verified via `--dump-tensors`): 16 layers, 2048 hidden, 32 attn / 8 KV heads, 64 rope dim, `Q4_K_M` (file_type 15).

Example commands:

```bash
# Inspect (works today)
ggnpu -m models/llama-3.2-1b-q4_k_m.gguf --dump-tensors

# MVP (Phase 5 — works today)
ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 64
ggnpu bench-logits -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048
```
