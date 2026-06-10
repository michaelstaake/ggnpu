# ggnpu

Run GGUF models on AMD NPUs (Krackan / XDNA2).

## Native host setup

`ggnpu` is now intended to be built and run directly on the host.

### Host prerequisites

| Required on host | Notes |
|------------------|-------|
| Linux with `amdxdna` loaded | Ryzen AI / XDNA-capable host |
| `/dev/accel/accel0` | NPU device node |
| `/usr/lib/firmware/amdnpu` | Firmware directory |
| XRT runtime and headers | `libxrt2`, `libxrt-npu2`, `libxrt-dev` |
| CMake and C++ toolchain | `cmake`, `g++`, `make` or Ninja |
| User in host `render` group | Required to open the accel device |

BIOS: NPU/IPU enabled. Boot: `amd_iommu=on`.

### Quick start

```bash
git clone https://github.com/michaelstaake/ggnpu.git
cd ggnpu

# Verify host prerequisites
bash scripts/setup-host.sh
bash scripts/verify-npu.sh

# Build ggnpu with the NPU backend
cmake -S . -B build-npu \
  -DGGNPU_NPU_BACKEND=ON \
  -DGGNPU_TEST_CPU=OFF \
  -DGGNPU_BUILD_TESTS=ON
cmake --build build-npu -j2

# Smoke test
./build-npu/ggnpu bench-matmul

# Optional: build xclbin kernels locally if ~/.cache/ggnpu/xclbin is empty
# export AIE_HOME=/path/to/mlir-aie
# export PEANO_HOME=/path/to/peano
# ./scripts/build-kernels.sh npu6 matmul

# Inference (put GGUF files in models/)
./build-npu/ggnpu \
  -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 32
```

Full host setup details: [docs/host-setup-guide.md](docs/host-setup-guide.md)

### Verify host

```bash
bash scripts/setup-host.sh
bash scripts/verify-npu.sh
```

These scripts check hardware, driver, permissions, XRT, and the local xclbin cache.

---

**Implementation spec:** [IMPLEMENTATION.md](IMPLEMENTATION.md) — complete handoff for contributors and AI agents.

**Kernel builds:** Local kernel compilation still requires `mlir-aie` and Peano. If you already have prebuilt `.xclbin` files, place them under `~/.cache/ggnpu/xclbin/`.
