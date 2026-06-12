# GGNPU — AI Implementation Handoff

This document is the complete specification for building **ggnpu**: a custom llama.cpp-like runtime that runs standard GGUF models on AMD NPUs. Give this file (and the repo) to an AI coding agent and work through the phases in order.

**Repository:** https://github.com/michaelstaake/ggnpu  
**License:** GPL-3.0  
**Current state:** **Phases 1–5 MVP passed.** GGUF load, NPU matmul smoke, Q4_K weight path, full-layer `bench-layer`, and E2E inference on Llama 3.2 1B Q4_K_M are validated. France prompt produces coherent continuation (`Paris, and it is the most visited`). Regression: `ctest -R test_e2e_logits`, `python3 scripts/compare_logits.py --check`. Next focus: NPU utilization (RMSNorm N=2048, flash attention shapes, SiLU FFN=8192) and Phase 6 production polish. See **§7.1**.

**Verdict:** The supported path is host-native: install host prerequisites, build `ggnpu` locally, and provide local `.xclbin` + `*_sequence.bin` kernels under `~/.cache/ggnpu/xclbin/`.

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

**Future (not v1):** Intel NPU backend behind the same interface.

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

| Priority | Arch | Quants |
|----------|------|--------|
| P0 | `llama` (3.2 1B/3B) | `Q4_K_M`, `Q8_0` |
| P1 | `qwen2` | same |
| P2 | `mistral`, `gemma2` | extend tensor maps |

Map tensor names → roles (`attn_q`, `ffn_gate`, etc.). Handle GQA. Respect `llama.tensor_data_layout` permutes.

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

**Known gap:** `mul_mat_q()` must `copy_from` output buffer after kernel run (other ops already do). Without this, NPU matmul results are never read back to host.

**Known gap:** RMSNorm NPU kernel only supports N=256 (prebuilt) and N=2048 (Llama hidden). Other sizes fall back to CPU. JIT compilation available for arbitrary sizes.

**Known gap:** SiLU NPU kernel only supports size=8192 (Llama 3B FFN). Other sizes fall back to CPU. JIT compilation available for arbitrary sizes.

**Known gap:** FlashAttention NPU kernel only supports 8 heads, 128 head_dim, 2048 ctx. Other configurations fall back to CPU.

**Known gap:** RoPE runs on CPU only; NPU kernel infrastructure exists but not yet wired into the backend.

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
│   ├── host-setup-guide.md
│   └── intel-roadmap.md
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
| 4 Full decoder layer | One layer vs CPU ref | **Done** — `bench-layer` PASS; RMSNorm/SiLU/flash_attn use CPU fallback at Llama shapes (NPU matmuls validated) |
| 5 Inference MVP | Coherent text, Llama 1B ctx 2048 | **Done** — France prompt → Paris; `bench-logits` + `test_e2e_logits` regression |
| 6 Production | Native deployment, 3B, L2 tiling | **Partial** — native setup documented; xclbins not validated E2E; SiLU/RMSNorm shape limits |
| 7 Intel stub | Interface research | **Not started** |

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
- [x] CPU fallback when NPU elementwise kernels unavailable or wrong shape
- [x] Load all prebuilt xclbins at backend init (matmul, rmsnorm, softmax, silu)
- [x] All matmuls in one layer on NPU (gate/up/down + attn_q/k/v/output benched)
- [x] Full single-layer forward CPU vs NPU (`bench-layer` test 4)
- [~] RMSNorm + SiLU on NPU at Llama hidden/ffn sizes — CPU fallback today (prebuilt rmsnorm 32×256; SiLU N=8192)
- [~] RoPE on CPU (correct Llama 3 NORMAL pairing + `rope_freqs` freq factors)
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

Coherent text is validated on CPU and NPU builds. Layer matmuls use INT8 on NPU; norms/attention/SiLU/logits projection still use CPU paths where shapes or accuracy require it (~15% NPU utilization).

### Phase 6 — Production

- [ ] Llama 3.2 3B; L2-aware tiling
- [ ] Ship validated prebuilt xclbins or document local kernel build clearly
- [ ] Clear errors: missing firmware, IOMMU off, memlock
- [ ] `docs/usage.md`, README quick start, `ggnpu --help` in sync

**Done when:** New user runs inference using only `README.md` + `docs/host-setup-guide.md` on a native host install.

### Phase 7 — Intel stub (after Phase 5)

- [ ] `IntelNpuBackend` interface stub in `docs/intel-roadmap.md`
- [ ] Panther Lake / Linux 7.0 NPU path research

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
| `.xclbin` in `~/.cache/ggnpu/xclbin` | **Present** | matmul, rmsnorm, softmax, silu (npu6) + `*_sequence.bin` each |
| `bench-matmul` E2E | **Passes** | Validated correct output on hardware |
| rmsnorm/softmax/silu on NPU | xclbins load, dtype mismatch | Kernels are bf16; backend sends f32 — needs marshaling |
| flash_attn / rope xclbins | Not built | RoPE intentionally on CPU for MVP |
| Full inference E2E | **Validated** | France prompt coherent on CPU + NPU builds; `test_e2e_logits` |
| NPU utilization | **~15–50%** (generation) | Matmul + SiLU on NPU; RMSNorm N=2048 via `rmsnorm_2048_npu6.xclbin` when built; flash_attn still CPU |

Production commands use the **native host build** (§9). Do **not** rely on `GGNPU_TEST_CPU=ON` in release NPU builds — it allows silent CPU fallback, which violates §2.

#### Known code gaps (post–Phase 5 / toward Phase 6)

| Gap | File | Impact |
|-----|------|--------|
| RMSNorm N=2048 on NPU | `amd_xdna.cpp`, `build-kernels.sh` | Build `rmsnorm_2048_npu6.xclbin` (`--M 2 --N 2048`); wrong/missing xclbin → CPU fallback |
| Flash attention shape | `amd_xdna.cpp`, `main.cpp` | Batched 32×64 call; build `flash_attn_32x64x2048_npu6.xclbin` (experimental transform) |
| SiLU FFN=8192 on NPU | `amd_xdna.cpp` | **Works** when `silu_npu6.xclbin` present |
| Matmul perf: host-side tiling | `amd_xdna.cpp` | Host weight-tile cache; per-tile DMA (device persist deferred) |
| RoPE on CPU | `main.cpp` | Correct Llama 3 math; not on NPU yet |
| Logits projection on CPU (F32 dequant) | `main.cpp` | `compute_logits_f32()` — accurate; not INT8 NPU path |
| Residual adds on CPU | `main.cpp` | Cheap; acceptable for MVP |
| `execute_layer_graph()` unused | `main.cpp` | Dead code |

#### Fixed — Phase 5 inference quality

| Fix | File |
|-----|------|
| RoPE **GGML_ROPE_TYPE_NORMAL** adjacent pairs `(2i, 2i+1)` — was NeoX half-split (wrong logits) | `src/cli/main.cpp` |
| Llama 3 `rope_freqs` used as **freq_factors**, not angle divisors | `src/cli/main.cpp` |
| Q6_K **per-token row** embedding dequant (bulk INT8 cache produced garbage activations) | `src/cli/main.cpp` |
| Decode-style prefill: one token per forward, KV reset at start | `src/cli/main.cpp` |
| **F32 logits** via `compute_logits_f32()` (vocab projection) | `src/cli/main.cpp` |
| `bench-logits` CLI + `scripts/compare_logits.py` + `scripts/test-e2e-logits.sh` (`ctest -R test_e2e_logits`) | `src/cli/main.cpp`, `scripts/`, `CMakeLists.txt` |
| Weight cache key includes `data_size`; clear `~/.cache/ggnpu/weights/` after decoder changes | `include/ggnpu/weight_cache.h` |
| `attach_kquant_scales` validates scale count | `src/cli/main.cpp` |
| France prompt golden tokens in `test_tokenizer` | `tests/test_tokenizer.cpp` |

#### Recently fixed (kernels / layer bench)

| Fix | File |
|-----|------|
| Removed misleading "SiLU uses CPU fallback" comment from `bench-layer` test (NPU path is actually exercised) | `src/cli/main.cpp` |
| Added clarifying comments for CPU fallbacks: logits projection, residual adds | `src/cli/main.cpp` |
| **bf16 marshaling gate:** `create_rmsnorm/softmax/silu_kernel_from_loaded_xclbin` now properly load instruction sequences and create kernels (were returning false) | `src/backends/amd_xdna/amd_xdna.cpp` |
| **RMSNorm shape:** prebuilt kernel now accepted for N=2048 (Llama hidden) in addition to N=256 | `src/backends/amd_xdna/amd_xdna.cpp` |
| **FlashAttention kernel setup:** `create_flash_attn_kernel_from_loaded_xclbin` now creates kernel from loaded xclbin (was returning false) | `src/backends/amd_xdna/amd_xdna.cpp` |
| **SiLU arbitrary sizes:** JIT compilation path added for sizes != 8192; falls back to CPU if JIT unavailable | `src/backends/amd_xdna/amd_xdna.cpp` |
| **KV cache override:** `reinit_kv_cache()` public method added; CLI `-c/--ctx-size` now reinitializes KV cache | `model.h`, `llama.cpp`, `main.cpp` |
| **RMSNorm arbitrary sizes:** JIT compilation path added for sizes != 256/2048; falls back to CPU if JIT unavailable | `src/backends/amd_xdna/amd_xdna.cpp` |

#### Recently fixed

| Fix | File |
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
cmake --build build-npu -j2
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

- **High NPU utilization** — most non-matmul ops and logits run on CPU; expect ~15% NPU use during inference.
- **RMSNorm / flash_attn / SiLU at Llama shapes on NPU** — CPU fallback until kernels or JIT paths match 2048 hidden / 8192 FFN / 32×64 heads.
- **No CI** — run `ctest` and `compare_logits.py --check` locally after changes.

#### Path forward (next work, in order)

1. **NPU RMSNorm N=2048** — largest utilization win; prebuilt xclbin is 32×256 today.
2. **Flash attention** at Llama head layout (32 heads × 64 dim).
3. **SiLU N=8192** on NPU for SwiGLU FFN.
4. **Matmul perf:** batch tile runs, persist converted weight tiles, larger fixed-shape xclbins.
5. **Phase 6:** 3B model, L2-aware tiling, production docs/errors, optional prebuilt xclbin distribution.

### 7.2 Memory constraints (16 GB RAM dev machine)

Dev laptop has ~14 GiB RAM + 4 GiB swap. Cursor IDE can consume several GB; combined with an oversized KV cache this is the highest OOM risk.

| Activity | RAM impact | Recommendation |
|----------|------------|----------------|
| Native `ggnpu` build | Low (~2–4 GB) | Fine on laptop |
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
cmake --build . -j2
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
cmake --build build-npu -j2
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
| IDE memory pressure (Cursor) | Run heavy builds and inference outside IDE; use `make -j2` |

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
5. Never enable CPU/GPU math fallback in release builds.
6. Add tests for every parser/quant/graph component; NPU tests marked `MANUAL`.
7. Update `docs/usage.md` and `--help` whenever CLI flags change.
8. Commit logically per phase; write clear commit messages.

### Triton kernel rules (enforce on all code generation)

When generating NPU kernel code (`kernels/triton/`), **always** apply the four guardrails from Section 2:

- **Memory-first:** Block-based DMA, no un-chunked reads, overlap compute with streaming.
- **Vector intrinsics only:** No scalar `+`/`*` in kernels. Use AIE vector ops. Types: INT8, BF16, FP8.
- **No branches:** Zero `if/else`/`switch`/`while` in hot loops. Predication or lookup tables only.
- **Two layers:** Control code (IRON API, DMA setup) never contains tensor math. Kernel code (Triton-XDNA compiled) never handles DMA or launch.

**Start here:** Phases 1–5 MVP passed. Inference is coherent on Llama 3.2 1B Q4_K_M; regression via `ctest -R test_e2e_logits` and `python3 scripts/compare_logits.py --check`. Next work, in order:

1. **NPU RMSNorm N=2048** — move hidden-state norm off CPU (biggest utilization gain).
2. **Flash attention + SiLU** at Llama shapes on NPU.
3. **Matmul perf:** batch tiles, weight-tile persistence, larger xclbins.
4. **Phase 6:** 3B, L2 tiling, production polish; keep `README.md`, `docs/host-setup-guide.md`, and §9–§10 aligned.
5. **Hygiene:** clear `~/.cache/ggnpu/weights/` after Q4_K/Q6_K decoder or cache-key changes.

Hardware-facing conventions an agent must know (all in `src/backends/amd_xdna/`):

- xclbins load via `device.register_xclbin()` + `xrt::hw_context`; `device.load_xclbin()` fails on amdxdna.
- Kernel name in every Triton-XDNA xclbin is `MLIR_AIE`; launch convention is `run(opcode=3, bo_instr, n_instr_words, bo0, bo1, bo2)` where `bo_instr` holds the `*_sequence.bin` words in a `XCL_BO_FLAGS_CACHEABLE` buffer at `group_id(1)`.
- Runs are async: always `run.wait()` before reading output.
- Each xclbin is compiled fixed-shape (see `kernels/triton/compile_kernels.py`): matmul 256³ int8→int32, rmsnorm N=2048 bf16, softmax 4×1024 bf16, silu N=2048 bf16.

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
| `models/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf` | ~2.6 GB | Larger; avoid on 16 GB RAM until KV fixed |

Llama 3.2 1B metadata (verified via `--dump-tensors`): 16 layers, 2048 hidden, 32 attn / 8 KV heads, 64 rope dim, `Q4_K_M` (file_type 15).

Example commands:

```bash
# Inspect (works today)
ggnpu -m models/llama-3.2-1b-q4_k_m.gguf --dump-tensors

# MVP (Phase 5 — works today)
ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 64
ggnpu bench-logits -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048
```
