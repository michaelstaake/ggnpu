# GGNPU — AI Implementation Handoff

This document is the complete specification for building **ggnpu**: a custom llama.cpp-like runtime that runs standard GGUF models on AMD NPUs. Give this file (and the repo) to an AI coding agent and work through the phases in order.

**Repository:** https://github.com/michaelstaake/ggnpu  
**License:** GPL-3.0  
**Current state (2025-06-09):** **Not ready to run a model on the NPU.** Phase 1 is complete (GGUF load/dump). Docker runtime image builds on Ubuntu 26.04 and opens the Krackan NPU from inside a container. **Production delivery is Docker-only** — users do not install XRT, mlir-aie, or `ggnpu` on the host. Phase 2 is blocked on xclbin kernels (builder image + `ggnpu-cache` volume). See **§7.1**.

**Verdict:** Docker path is the supported product surface; Phase 2 (`bench-matmul` + xclbins via `docker/Dockerfile.builder`) is the next gate.

---

## 1. Mission

Build a standalone C++20 inference binary named `ggnpu` that:

1. Loads **standard HuggingFace GGUF** files with **no user-side conversion** (no ONNX export, no AMD Quark, no re-quantization step).
2. Runs **all transformer tensor math** on the **AMD NPU** — matmul, attention, norms, activations, RoPE.
3. Uses the **host CPU only** for control plane: CLI, GGUF parsing, mmap, tokenization, sampling, logging.
4. Targets **Ubuntu 26.04 LTS** (Linux **7.0**) and **AMD Ryzen AI 7 350** (Krackan / XDNA2 / `npu6`).
5. Ships and runs **only inside Docker** (`docker/Dockerfile`, `docker compose`). Host passes through `/dev/accel/accel0` and firmware; all toolchain and runtime deps live in the image.
6. Ships simple user docs: `README`, `docs/docker.md`, `docs/usage.md`, and `ggnpu --help`.

### 1.1 Deployment model (Docker-only)

| Layer | Where it runs |
|-------|----------------|
| `amdxdna` driver, `/dev/accel/accel0` | **Host** (kernel) |
| NPU firmware | **Host** path mounted read-only into container |
| Docker Engine | **Host** (native Linux; not Docker Desktop VM) |
| XRT, mlir-aie, Peano, `ggnpu` binary, `.xclbin` kernels | **Inside Docker images** |
| Inference CLI | **`docker compose run ggnpu …`** |

**Users must not** install `libxrt2`, `libxrt-dev`, mlir-aie, or build `ggnpu` on the host for production use.

**Contributors** may still build natively for unit tests (`ctest`, CPU ref) and debugging; that path is unsupported for end users.

**Images:**

| Image | Dockerfile | Role |
|-------|------------|------|
| `ggnpu:latest` | `docker/Dockerfile` | Runtime: XRT + NPU-enabled `ggnpu` |
| `ggnpu-builder:latest` | `docker/Dockerfile.builder` | One-shot: mlir-aie + xclbin compile → `ggnpu-cache` volume |

See **§10** and `docs/docker.md`.

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
| **mlir-aie** | Build-time: compile spatial NPU programs → `.xclbin` |
| **Peano** (`aie2p-none-unknown-elf`) | Build-time: compile C++ tile code inside mlir-aie pipeline |
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
| **Kernel Execution Code** | Spatial compute tile rows (Peano ELFs) | Vectorized INT8/BF16 math kernels, no branches, fully unrolled loops |

The control code **never** contains tensor math. The kernel code **never** handles DMA setup or kernel launch. This separation is enforced by the IRON/mlir-aie compilation pipeline.

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

## 4. Dependency roles (XRT, mlir-aie, Peano)

These are **not** inference frameworks. ggnpu builds the GGUF runtime; these handle NPU kernel compile and dispatch.

```
GGUF file
  → ggnpu (parse, graph, quant, schedule)     ← YOU BUILD THIS
  → XRT (load xclbin, DMA buffers, run)       ← runtime, every inference
  → amdxdna kernel (/dev/accel/accel0)

NPU kernels (.xclbin):
  C++/IRON source → Peano (tile ELF) → mlir-aie (overlay) → xclbin
                    ↑ build time / first-run cache miss only
```

| Tool | When it runs | What it does |
|------|--------------|--------------|
| **XRT** | Every inference | Open NPU, pinned `xrt::bo` buffers, load `.xclbin`, `xrt::run`, sync |
| **mlir-aie** | Build / cache miss | Map ops onto AIE tiles, produce `.xclbin` + ctrlcode for DMA |
| **Peano** | Build / cache miss | Compile vectorized INT8/BF16 C++ for individual AIE cores |

**Why XRT not raw ioctls:** Raw `DRM_IOCTL_AMDXDNA_*` requires reimplementing BO lifecycle, PASID/IOMMU SVA, ERT mailbox, firmware coupling. XRT is the official shim (like libdrm for GPU).

**Docker:** Production delivery only. mlir-aie/Peano stay in `docker/Dockerfile.builder`; runtime image `docker/Dockerfile` ships `ggnpu` + XRT. xclbins live in the `ggnpu-cache` Docker volume (populated by the builder service). See `docs/docker.md`.

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
- **Kernel plane** (`kernels/amd/`): Peano-compiled C++ tile code. Runs on spatial compute tiles. Vectorized intrinsics only, no branches, no scalar math.

**Bring-up:**

1. `xrt::device` on `/dev/accel/accel0`
2. Detect `npu6` (Krackan) → `aie2p` kernel profile
3. HW context; unlimited memlock; `xrt::bo` for weights/activations
4. Load `.xclbin` from `~/.cache/ggnpu/xclbin/` or JIT-compile (cache empty on dev machine as of 2025-06-09)

**Known gap:** `mul_mat_q()` must `copy_from` output buffer after kernel run (other ops already do). Without this, NPU matmul results are never read back to host.

**Kernel compile cache key:** `(op, M, N, K, dtype, npu_profile)`

**Llama 3B typical matmul shapes:**

| Layer | K / N |
|-------|-------|
| Q/K/V/O proj | 3072, 2048 |
| FFN gate/up | 8192 |
| FFN down | 8192 → 3072 |

Start from mlir-aie `programming_examples/matrix_multiplication`. Use `transform_aie2p.mlir` for Krackan ([Triton-XDNA](https://github.com/amd/Triton-XDNA)).

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
│   ├── docker.md
│   └── intel-roadmap.md
├── cmake/                         # FindXRT, FindPeano
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
├── kernels/amd/
│   ├── matmul_i8/
│   ├── rmsnorm/
│   ├── rope/
│   ├── softmax/
│   └── fused_attn/
├── tests/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
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
| 0 Scaffold | CMake, scripts, docs | Mostly done (Docker runtime image builds; no CI) |
| 1 GGUF loader | Parse, mmap, dump | **Done** |
| 2 NPU matmul smoke | `bench-matmul` on hardware | **Not done** — needs XRT, xclbins (host setup) |
| 3 Q4_K weight path | Decode + one E2E matmul | Decoders + disk cache done; NPU E2E not validated |
| 4 Full decoder layer | All ops on NPU | **Not done** |
| 5 Inference MVP | Coherent text, Llama 1B ctx 2048 | **Not done** — KV `-c`/cap fix done, attention dims fixed, not validated E2E |
| 6 Production | Docker, 3B, L2 tiling | **Partial** — Docker runtime + builder images exist; xclbins not validated E2E |
| 7 Intel stub | Interface research | **Not started** |

### Phase 0 — Scaffold

- [x] CMake project, C++20
- [ ] CI (compile on x86; NPU tests `MANUAL`)
- [x] `scripts/setup-host.sh` — host checks for Docker deployments (no XRT/mlir-aie install)
- [x] `scripts/verify-npu.sh` — hardware, driver, accel0, optional Docker-oriented checks
- [x] `docs/architecture.md`, `docs/amd-krackan.md`, `docs/docker.md`
- [x] `docker/Dockerfile` — runtime image (Ubuntu 26.04, NPU-enabled `ggnpu`)
- [x] `docker/Dockerfile.builder` — mlir-aie + kernel build
- [x] `docker/docker-compose.yml` — `ggnpu` + `builder` profile

**Done when:** `docker compose build ggnpu` succeeds; host passes `verify-npu.sh` hardware checks; NPU opens inside container (`bench-matmul` reaches backend init).

### Phase 1 — GGUF loader

- [x] Parser, mmap, metadata API
- [x] Quant layouts: F16, Q8_0, Q4_0
- [x] Llama hparams + tensor name map
- [x] Unit tests on real GGUF files

**Done when:** `ggnpu -m model.gguf --dump-tensors` prints correct inventory. **Gate passed** on `models/llama-3.2-1b-q4_k_m.gguf`.

### Phase 2 — NPU matmul smoke

- [ ] XRT device init on Krackan
- [ ] Build INT8 matmul xclbin (e.g. 1×3072 × 3072×3072)
- [ ] Compile cache (`~/.cache/ggnpu/xclbin/` — currently empty)
- [x] Copy matmul output from device back to host in `src/backends/amd_xdna/amd_xdna.cpp`
- [x] Make `ggnpu bench-matmul` fail fast on backend errors and incorrect output

**Done when:** `ggnpu bench-matmul` runs on NPU with measurable throughput.

### Phase 3 — Q4_K weight path

- [x] Q4_K + Q6_K block decoders
- [x] Transparent INT8 NPU weight cache (disk under `~/.cache/ggnpu/weights/`)
- [ ] One `ffn_gate` matmul end-to-end from GGUF on NPU

**Done when:** output within tolerance vs CPU reference dequant matmul.

### Phase 4 — Full decoder layer

- [ ] All matmuls in one layer on NPU
- [ ] RMSNorm, RoPE, attention (decomposed), SiLU on NPU (RoPE currently CPU fallback in NPU backend)
- [ ] KV cache write

**Done when:** one-layer forward matches CPU reference.

### Phase 5 — Inference MVP

- [x] Prefill + decode loops (per-token buffers; not validated E2E)
- [x] Tokenizer + greedy sampling
- [ ] Target: Llama 3.2 1B Q4_K_M, ctx 2048 — needs NPU xclbins + E2E test
- [x] Fix KV cache to respect `-c` / cap default ctx to 2048
- [x] Dequant `token_embd.weight` for Q4_K models

**Done when:**

```bash
ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048
```

produces coherent text with all math on NPU.

### Phase 6 — Production

- [ ] Llama 3.2 3B; L2-aware tiling
- [x] Docker image + compose (`docker/Dockerfile`, `docker/Dockerfile.builder`, `docker-compose.yml`)
- [ ] Ship validated prebuilt xclbins (builder populates `ggnpu-cache` volume; MLIR kernels still maturing)
- [ ] Clear errors: missing firmware, IOMMU off, memlock
- [ ] `docs/usage.md`, README quick start, `ggnpu --help` in sync

**Done when:** New user runs inference using only `README.md` + `docs/docker.md` (no host package installs beyond Docker + driver).

### Phase 7 — Intel stub (after Phase 5)

- [ ] `IntelNpuBackend` interface stub in `docs/intel-roadmap.md`
- [ ] Panther Lake / Linux 7.0 NPU path research

### 7.1 Readiness snapshot (dev laptop, 2025-06-09)

Assessment of whether the project can run a model on the NPU **today**.

#### Ready (host + Docker)

| Layer | Status | Notes |
|-------|--------|-------|
| NPU hardware | Present | PCI `1022:17f0` (Strix/Krackan) |
| Kernel driver | Loaded | `amdxdna` module (host) |
| Device node | Present | `/dev/accel/accel0` passed into container |
| Firmware | Present | Bind-mount `/usr/lib/firmware/amdnpu` |
| Docker runtime image | Builds | `docker/Dockerfile` on Ubuntu 26.04 |
| NPU from container | Works | `NPU device opened: profile=npu6` in `docker run … bench-matmul` |
| GGUF loading | Works | `--dump-tensors` (native dev or future container cmd) |
| Test models | On disk | See §17 |

**Host does not need:** XRT, `libxrt-dev`, mlir-aie, Peano, or a native `ggnpu` binary.

#### Blocked (kernels)

| Check | Status | Required |
|-------|--------|----------|
| `.xclbin` in `ggnpu-cache` volume | **Empty** | Run `docker compose --profile build run --rm builder` |
| `bench-matmul` E2E | **Fails** | `matmul_npu6.xclbin` in cache volume |
| Full inference on NPU | **Not validated** | Phase 2–5 |

Production commands use **Docker only** (§10). Do **not** rely on `GGNPU_TEST_CPU=ON` — it allows silent CPU fallback, which violates §2.

#### Known code gaps (remaining before Phase 5)

| Gap | File | Impact |
|-----|------|--------|
| RoPE on CPU | `src/backends/amd_xdna/amd_xdna.cpp`, `main.cpp` | Explicit CPU fallback; OK for early MVP, not “all math on NPU” |
| Logits projection on CPU | `src/cli/main.cpp` | Dot product not routed through `mul_mat_q` |
| Residual adds on CPU | `src/cli/main.cpp` | Cheap; acceptable for MVP |
| MLIR kernels skeletal | `kernels/amd/*` | Must compile with mlir-aie; not validated on hardware |
| `execute_layer_graph()` unused | `src/cli/main.cpp` | Dead code; main calls backend directly |

#### Recently fixed (2025-06-09)

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
# Host check (no XRT/mlir-aie on host)
bash scripts/verify-npu.sh

# Docker runtime + NPU device open
docker compose -f docker/docker-compose.yml build ggnpu
docker compose -f docker/docker-compose.yml run --rm ggnpu bench-matmul
# → opens NPU; fails until xclbins exist in ggnpu-cache volume

# Contributor-only: unit tests on host CPU
cd build && ctest
```

#### What does not work today

```bash
# Native host inference — not supported for users
./build/ggnpu bench-matmul
./build/ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "Hello" -c 2048

# Docker inference — blocked until builder populates xclbins
docker compose -f docker/docker-compose.yml run --rm ggnpu \
  -m /models/llama-3.2-1b-q4_k_m.gguf -p "Hello" -c 2048
```

#### Minimum path to first NPU activity (Docker)

1. Host: install Docker only; ensure `amdxdna`, `/dev/accel/accel0`, firmware — see §9
2. `bash scripts/verify-npu.sh` (hardware section; ignore host XRT/mlir-aie failures)
3. `docker compose -f docker/docker-compose.yml build ggnpu`
4. `docker compose -f docker/docker-compose.yml build builder`
5. `docker compose -f docker/docker-compose.yml --profile build run --rm builder` (populates `ggnpu-cache` volume)
6. `docker compose -f docker/docker-compose.yml run --rm ggnpu bench-matmul`
7. Inference via `docker compose run --rm ggnpu -m /models/… -p "…" -c 2048`

### 7.2 Memory constraints (16 GB RAM dev machine)

Dev laptop has ~14 GiB RAM + 4 GiB swap. Cursor IDE can consume several GB; combined with an oversized KV cache this is the highest OOM risk.

| Activity | RAM impact | Recommendation |
|----------|------------|----------------|
| Docker build `ggnpu` image | Low (~2–4 GB) | Fine on laptop |
| Docker `builder` (mlir-aie + xclbins) | **Very high** (16–32 GB typical) | Runs inside container; use a machine with enough RAM |
| Load Llama 1B Q4_K_M (mmap) | ~770 MB virtual | Fine |
| KV at **131k ctx** (current code) | **~8.6 GB** | **Will OOM** with IDE open |
| KV at **2048 ctx** (after fix) | **~128 MB** | Comfortable |
| Weight decode disk cache | ~1–2 GB first run | Disk, not RAM |
| NPU pinned buffers | Tens–hundreds of MB | Requires memlock unlimited |

**RAM hygiene before inference:**

- Run `ggnpu` from a plain terminal outside the IDE when memory is tight
- Do not run `GGNPU_BUILD_KERNELS=ON` or mlir-aie builds on 16 GB RAM
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

## 9. Host prerequisites (Docker deployments)

**ggnpu is Docker-only for users.** The host installs **only** what is needed to run containers with NPU passthrough. Do **not** install XRT, mlir-aie, or build `ggnpu` on the host.

### Required on host

| Item | Purpose |
|------|---------|
| Linux + `amdxdna` | NPU kernel driver |
| `/dev/accel/accel0` | Passed into container via `--device` |
| `/usr/lib/firmware/amdnpu` | Bind-mounted read-only into container |
| Docker Engine (native Linux) | Run `ggnpu` and `ggnpu-builder` images |
| User in `render` group | Open accel device from container (`RENDER_GID` in compose) |
| `amd_iommu=on` | Boot parameter |
| BIOS NPU/IPU enabled | Hardware |

### One-time host setup

```bash
# Driver (if not already present on your distro kernel)
sudo apt install amdxdna-dkms   # or use distro kernel with amdxdna built-in

# Docker
sudo apt install docker.io docker-compose-v2
sudo usermod -aG docker $USER

# accel0 access
sudo usermod -aG render $USER
# Log out and back in

# Verify hardware (does not require XRT on host)
lspci -vd 1022:17f0
lsmod | grep amdxdna
ls -la /dev/accel/accel0
bash scripts/verify-npu.sh
```

Copy `docker/.env.example` → `docker/.env` and set `RENDER_GID` to `$(getent group render | cut -d: -f3)`.

### Not required on host

- `libxrt2`, `libxrt-npu2`, `libxrt-dev`
- mlir-aie, Peano, `aiecc.py`
- `cmake` / native `ggnpu` build
- `~/.cache/ggnpu` on host (use Docker volume `ggnpu-cache` instead)

### Contributor native build (optional, unsupported for users)

For `ctest` and debugging only:

```bash
mkdir build && cd build
cmake .. -DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=ON
make -j2 && ctest
```

Native NPU inference on the host is **not** a supported product path.

---

## 10. Docker (production)

**All user-facing commands run inside Docker.** See `docs/docker.md` for the full guide.

### Quick start

```bash
cp docker/.env.example docker/.env

docker compose -f docker/docker-compose.yml build ggnpu
docker compose -f docker/docker-compose.yml build builder
docker compose -f docker/docker-compose.yml --profile build run --rm builder

docker compose -f docker/docker-compose.yml run --rm ggnpu bench-matmul
docker compose -f docker/docker-compose.yml run --rm ggnpu \
  -m /models/llama-3.2-1b-q4_k_m.gguf -p "Hello" -c 2048 -n 32
```

### Images

| Image | Dockerfile | Contents |
|-------|------------|----------|
| `ggnpu:latest` | `docker/Dockerfile` | Ubuntu 26.04, XRT runtime, NPU-enabled `ggnpu` |
| `ggnpu-builder:latest` | `docker/Dockerfile.builder` | mlir-aie toolchain + kernel compile |

### Volumes

| Volume | Purpose |
|--------|---------|
| `ggnpu-cache` | Shared xclbin cache (`/root/.cache/ggnpu/xclbin` in container) |
| `../models` | GGUF models (read-only bind mount) |

### Manual `docker run`

```bash
RENDER_GID="$(getent group render | cut -d: -f3)"

docker run --rm \
  --device=/dev/accel/accel0 \
  --group-add "${RENDER_GID}" \
  --ulimit memlock=-1:-1 \
  -v /usr/lib/firmware/amdnpu:/usr/lib/firmware/amdnpu:ro \
  -v ggnpu-cache:/root/.cache/ggnpu \
  -v "$(pwd)/models:/models:ro" \
  ggnpu:latest bench-matmul
```

**Requirements:** native `docker.io` on Linux (not Docker Desktop VM). Host provides driver + firmware only.

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
| Q4_K mixed tensor types | Q4_K + Q6_K decoders done; embedding dequant still needed in CLI |
| First-run JIT latency | Prebuilt xclbins + persistent cache |
| XRT/driver/firmware skew | Pin versions; `verify-npu.sh` checks |
| 4 MB L2 limit | Tile matmuls; stream from DDR |
| mlir-aie complexity | Start from upstream matmul example + [IRON tutorial PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-ipdps-2025.pdf) |
| mlir-aie / xclbin build OOM | Do not build on 16 GB RAM; ship prebuilt xclbins or use remote builder |
| KV cache over-allocation | `init_kv_cache()` uses full GGUF `context_length`; must respect `-c` before inference on 16 GB hosts |
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
| mlir-aie | https://github.com/Xilinx/mlir-aie |
| mlir-aie Ryzen build guide | https://github.com/Xilinx/mlir-aie/blob/main/docs/Building.md |
| Triton-XDNA (AIE2P transforms) | https://github.com/amd/Triton-XDNA |
| OllamaAMDNPU (matmul reference) | https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU |

---

## 15. Definition of done (v1)

On **Ubuntu 26.04** + **Ryzen AI 7 350**, **via Docker only**:

1. User installs Docker and NPU driver on host (§9); no XRT/mlir-aie on host.
2. User builds/pulls `ggnpu:latest` and runs the builder once for xclbins.
3. User downloads a stock HuggingFace GGUF into `models/`.
4. User runs `docker compose … run --rm ggnpu -m /models/model.gguf -p "…"` with no preprocessing.
5. Model generates coherent text with transformer math on the NPU.
6. `README`, `docs/docker.md`, and `ggnpu --help` explain how to run it.

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

### AIE kernel rules (enforce on all code generation)

When generating NPU kernel code (`kernels/amd/`), **always** apply the four guardrails from Section 2:

- **Memory-first:** Block-based DMA, no un-chunked reads, overlap compute with streaming.
- **Vector intrinsics only:** No scalar `+`/`*` in kernels. Use AIE vector ops. Types: INT8, BF16, FP8.
- **No branches:** Zero `if/else`/`switch`/`while` in hot loops. Predication or lookup tables only.
- **Two layers:** Control code (IRON API, DMA setup) never contains tensor math. Kernel code (Peano ELF) never handles DMA or launch.

**Start here (2025-06-09):** Phase 1 gate passed. Docker runtime opens NPU from container. Next work, in order:

1. **Docker builder (Phase 2):** `docker compose --profile build run --rm builder` → populate `ggnpu-cache` with `matmul_npu6.xclbin`.
2. **Phase 2 smoke:** `docker compose run --rm ggnpu bench-matmul` must pass with correct output.
3. **Phase 3 E2E:** One `ffn_gate` matmul from GGUF on NPU vs CPU ref (in container).
4. **Phases 4–5:** Full layer + inference MVP via `docker compose run --rm ggnpu -m /models/llama-3.2-1b-q4_k_m.gguf …`.
5. **Docs:** Keep `README.md`, `docs/docker.md`, and §9–§10 aligned — **Docker-only** for users; native build is contributor-only.

See §7.1 for full blocker list and §7.2 for RAM constraints.

---

## 17. Local test models

Put GGUF files in `models/` for development and E2E tests. This directory is **not committed**.

**`.gitignore`** (repo root):

```gitignore
# Local GGUF models for testing (not committed)
models/
```

### Models on dev machine (2025-06-09)

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

# MVP target (after Phase 5 fixes)
ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 64
```
