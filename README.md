# ggnpu

Run GGUF models on AMD NPUs (Krackan / XDNA2).

## Native host setup

`ggnpu` is intended to be built and run directly on the host. Docker is not used.

### Host prerequisites

| Required on host | Notes |
|------------------|-------|
| Linux with `amdxdna` loaded | Ryzen AI / XDNA-capable host |
| `/dev/accel/accel0` | NPU device node |
| `/usr/lib/firmware/amdnpu` | Firmware directory |
| XRT runtime and headers | `libxrt2`, `libxrt-npu2`, `libxrt-dev` |
| CMake and C++ toolchain | `cmake`, `g++`, `make` or Ninja |
| Python 3.10+ | For kernel compilation via Triton-XDNA |
| User in host `render` group | Required to open the accel device |

BIOS: NPU/IPU enabled. Boot: `amd_iommu=on`.

### Full installation

```bash
# 1. Core build tools
sudo apt update
sudo apt install build-essential cmake git clang lld ninja-build python3-pip python3-venv

# 2. XRT runtime + NPU driver (from AMD PPA)
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update
sudo apt install libxrt2 libxrt-npu2 libxrt-dev amdxdna-dkms

# 3. Memlock limits (required for NPU pinned DMA buffers)
echo '* soft memlock unlimited' | sudo tee /etc/security/limits.d/99-amdxdna.conf
echo '* hard memlock unlimited' | sudo tee -a /etc/security/limits.d/99-amdxdna.conf

# 4. Add user to render group
sudo usermod -aG render $USER

# 5. Log out and back in (or reboot) for group/limits to take effect

# 6. Verify everything
bash scripts/setup-host.sh
bash scripts/verify-npu.sh
```

### Quick start

```bash
git clone https://github.com/michaelstaake/ggnpu.git
cd ggnpu

# Build ggnpu with the NPU backend
cmake -S . -B build-npu \
  -DGGNPU_NPU_BACKEND=ON \
  -DGGNPU_TEST_CPU=OFF \
  -DGGNPU_BUILD_TESTS=ON
cmake --build build-npu -j2

# Smoke test
./build-npu/ggnpu bench-matmul

# Inference (put GGUF files in models/)
./build-npu/ggnpu \
  -m models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 32
```

### Build NPU kernels (Triton-XDNA)

`bench-matmul` and inference need `.xclbin` kernel files in `~/.cache/ggnpu/xclbin/`.
No NPU hardware is needed to build kernels — the compilation runs entirely on CPU.
You can build on a separate machine (no NPU required) and copy the `.xclbin` files to the NPU machine.

**Install Triton-XDNA** (one-time, replaces mlir-aie + Peano):

Triton-XDNA is distributed via GitHub releases, not PyPI. Install it using:

```bash
# On the NPU host (system-wide):
pip install triton-xdna --break-system-packages \
  --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti

# Or in a venv (for kernel building on a separate machine):
python3 -m venv ~/triton-env
source ~/triton-env/bin/activate
pip install triton-xdna \
  --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels \
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 \
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly \
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
```

This installs the complete compiler stack (Triton + mlir-air + mlir-aie + llvm-aie) as Python wheels.

**Set up Triton-XDNA (one-time, for kernel builds):**

```bash
bash scripts/setup-triton-env.sh          # creates ~/triton-env (or .venv-triton)
source ~/triton-env/bin/activate        # or: source .venv-triton/bin/activate
```

**Build ggnpu kernels:**

```bash
# Build all kernels for npu6 (Krackan)
./scripts/build-kernels.sh npu6

# Build only specific kernel
./scripts/build-kernels.sh npu6 matmul

# Build for multiple profiles
./scripts/build-kernels.sh npu4 npu5 npu6
```

Output goes to `~/.cache/ggnpu/xclbin/`.

**Building on a non-NPU machine:** Steps work on any Ubuntu 24.04+ machine with ~16 GB RAM (32 GB recommended). No NPU, no `amdxdna` driver, no firmware needed. After building, copy the xclbins to the NPU machine:
```bash
scp -r ~/.cache/ggnpu/xclbin/* user@npu-machine:~/.cache/ggnpu/xclbin/
```

### Verify host

```bash
bash scripts/setup-host.sh
bash scripts/verify-npu.sh
```

These scripts check hardware, driver, permissions, XRT, and the local xclbin cache.

---

**Implementation spec:** [IMPLEMENTATION.md](IMPLEMENTATION.md) — complete handoff for contributors and AI agents.

**Full host setup guide:** [docs/host-setup-guide.md](docs/host-setup-guide.md)

**Kernel builds:** Local kernel compilation requires `Triton-XDNA`. If you already have prebuilt `.xclbin` files, place them under `~/.cache/ggnpu/xclbin/`.
