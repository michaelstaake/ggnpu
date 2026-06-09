# GGNPU — AI Implementation Handoff

This document is the complete specification for building **ggnpu**: a custom llama.cpp-like runtime that runs standard GGUF models on AMD NPUs. Give this file (and the repo) to an AI coding agent and work through the phases in order.

**Repository:** https://github.com/michaelstaake/ggnpu  
**License:** GPL-3.0  
**Current state:** Phases 0–3 partially implemented. GGUF parser, quantization decoders, CPU reference backend, compute graph, KV cache, tokenizer, CLI, and XRT NPU backend skeleton exist. Full inference loop works on CPU ref; NPU ops need kernel integration. See `git log` for commit history.

---

## 1. Mission

Build a standalone C++20 inference binary named `ggnpu` that:

1. Loads **standard HuggingFace GGUF** files with **no user-side conversion** (no ONNX export, no AMD Quark, no re-quantization step).
2. Runs **all transformer tensor math** on the **AMD NPU** — matmul, attention, norms, activations, RoPE.
3. Uses the **host CPU only** for control plane: CLI, GGUF parsing, mmap, tokenization, sampling, logging.
4. Targets **Ubuntu 26.04 LTS** (Linux **7.0**) and **AMD Ryzen AI 7 350** (Krackan / XDNA2 / `npu6`).
5. Works **natively** or inside **Docker** with `/dev/accel/accel0` passthrough.
6. Ships simple user docs: `README`, `docs/usage.md`, and `ggnpu --help`.

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

**Docker:** Ship prebuilt `.xclbin` in the image. mlir-aie/Peano stay in a **builder** image, not the runtime image.

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
| P1 | `Q4_K`, `Q6_K` (required for `Q4_K_M` — mixed per-tensor types) |
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
4. Load `.xclbin` from `~/.cache/ggnpu/xclbin/` or JIT-compile

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

### Phase 0 — Scaffold

- [ ] CMake project, C++20, CI (compile on x86; NPU tests `MANUAL`)
- [ ] `scripts/setup-host.sh` — XRT PPA, memlock limits, `render` group
- [ ] `scripts/verify-npu.sh` — `lspci`, `lsmod amdxdna`, `/dev/accel/accel0`, `xrt-smi examine`
- [ ] `docs/architecture.md`, `docs/amd-krackan.md`

**Done when:** binary links against XRT; `verify-npu.sh` documents host prerequisites.

### Phase 1 — GGUF loader

- [ ] Parser, mmap, metadata API
- [ ] Quant layouts: F16, Q8_0, Q4_0
- [ ] Llama hparams + tensor name map
- [ ] Unit tests on real GGUF files

**Done when:** `ggnpu -m model.gguf --dump-tensors` prints correct inventory.

### Phase 2 — NPU matmul smoke

- [ ] XRT device init on Krackan
- [ ] Build INT8 matmul xclbin (e.g. 1×3072 × 3072×3072)
- [ ] Compile cache

**Done when:** `ggnpu bench-matmul` runs on NPU with measurable throughput.

### Phase 3 — Q4_K weight path

- [ ] Q4_K + Q6_K block decoders
- [ ] Transparent INT8 NPU weight cache
- [ ] One `ffn_gate` matmul end-to-end from GGUF

**Done when:** output within tolerance vs CPU reference dequant matmul.

### Phase 4 — Full decoder layer

- [ ] All matmuls in one layer on NPU
- [ ] RMSNorm, RoPE, attention (decomposed), SiLU on NPU
- [ ] KV cache write

**Done when:** one-layer forward matches CPU reference.

### Phase 5 — Inference MVP

- [ ] Prefill + decode loops
- [ ] Tokenizer + greedy sampling
- [ ] Target: Llama 3.2 1B Q4_K_M, ctx 2048

**Done when:**

```bash
ggnpu -m llama-3.2-1b-Q4_K_M.gguf -p "The capital of France is"
```

produces coherent text with all math on NPU.

### Phase 6 — Production

- [ ] Llama 3.2 3B; L2-aware tiling
- [ ] Docker image + compose
- [ ] Clear errors: missing firmware, IOMMU off, memlock
- [ ] `docs/usage.md`, README quick start, `ggnpu --help` in sync

**Done when:** Docker run works; new user can infer using only README + `docs/usage.md`.

### Phase 7 — Intel stub (after Phase 5)

- [ ] `IntelNpuBackend` interface stub in `docs/intel-roadmap.md`
- [ ] Panther Lake / Linux 7.0 NPU path research

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

## 9. Host setup

```bash
# Verify hardware
lspci -vd 1022:17f0
lsmod | grep amdxdna
ls -la /dev/accel/

# Install XRT (Ubuntu)
sudo add-apt-repository ppa:amd-team/xrt
sudo apt install libxrt2 libxrt-npu2 amdxdna-dkms

# Memlock (required)
echo '* soft memlock unlimited' | sudo tee /etc/security/limits.d/99-amdxdna.conf
echo '* hard memlock unlimited' | sudo tee -a /etc/security/limits.d/99-amdxdna.conf

# User group
sudo usermod -aG render $USER

# Verify
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
./scripts/verify-npu.sh
```

**BIOS:** Enable NPU/IPU (Advanced → CPU Configuration → IPU).  
**Boot:** `amd_iommu=on` (never `amd_iommu=off`).

---

## 10. Docker

**Host:** `amdxdna` loaded, IOMMU on, native `docker.io` (not Docker Desktop VM).

```bash
docker run --rm \
  --device=/dev/accel/accel0 \
  --group-add render \
  --ulimit memlock=-1:-1 \
  -v /usr/lib/firmware/amdnpu:/usr/lib/firmware/amdnpu:ro \
  -v $HOME/.cache/ggnpu:/root/.cache/ggnpu \
  -v /path/to/models:/models:ro \
  ggnpu:latest -m /models/llama-3.2-1b-Q4_K_M.gguf -p "Hello"
```

**Image:** `ubuntu:26.04` base; ship `ggnpu` binary + prebuilt xclbins; mount firmware from host; pin XRT version to match host driver.

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
| Q4_K mixed tensor types | Implement Q4_K + Q6_K in Phase 3 |
| First-run JIT latency | Prebuilt xclbins + persistent cache |
| XRT/driver/firmware skew | Pin versions; `verify-npu.sh` checks |
| 4 MB L2 limit | Tile matmuls; stream from DDR |
| mlir-aie complexity | Start from upstream matmul example + [IRON tutorial PDF](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen/ai/iron-for-ryzen-ai-tutorial-ipdps-2025.pdf) |

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

On **Ubuntu 26.04** + **Ryzen AI 7 350**, natively or in Docker:

1. User downloads a stock HuggingFace GGUF (e.g. Llama 3.2 1B Q4_K_M).
2. User runs `ggnpu -m model.gguf -p "..."` with no preprocessing.
3. Model generates coherent text.
4. All transformer math runs on the NPU.
5. `README`, `docs/usage.md`, and `ggnpu --help` explain how to run it.

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

When generating NPU kernel code (`kernels/amd/`), **always** apply the four guardrails from Section 2b:

- **Memory-first:** Block-based DMA, no un-chunked reads, overlap compute with streaming.
- **Vector intrinsics only:** No scalar `+`/`*` in kernels. Use AIE vector ops. Types: INT8, BF16, FP8.
- **No branches:** Zero `if/else`/`switch`/`while` in hot loops. Predication or lookup tables only.
- **Two layers:** Control code (IRON API, DMA setup) never contains tensor math. Kernel code (Peano ELF) never handles DMA or launch.

**Start here:** Phase 0 — scaffold CMake, `scripts/verify-npu.sh`, and `docs/amd-krackan.md`.

---

## 17. Local test models

Put GGUF files in `models/` for development and E2E tests. This directory is **not committed**.

**`.gitignore`** (repo root):

```gitignore
# Local GGUF models for testing (not committed)
models/
```

Example test paths used in docs and scripts:

```bash
ggnpu -m models/llama-3.2-1b-Q4_K_M.gguf -p "Hello" -n 32
ggnpu -m models/llama-3.2-1b-Q4_K_M.gguf --dump-tensors
```
