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

### Build NPU kernels (mlir-aie + Peano)

`bench-matmul` and inference need `.xclbin` kernel files in `~/.cache/ggnpu/xclbin/`.
If you already have prebuilt kernels, copy them there.
Otherwise, build them from source:

```bash
# 1. Clone mlir-aie
git clone https://github.com/Xilinx/mlir-aie.git ~/mlir-aie
cd ~/mlir-aie
git submodule update --init --recursive

# 2. Build using pre-built wheels (lightweight, works on 16 GB RAM)
python3 -m venv ironenv
source ironenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r python/requirements.txt
python3 -m pip install -r python/requirements_dev.txt
bash ./utils/build-mlir-aie-from-wheels.sh
source utils/env_setup.sh install

# 3. Clone and build Peano (AIE tile code compiler)
git clone https://github.com/Xilinx/llvm-aie.git ~/llvm-aie
cd ~/llvm-aie
# Follow llvm-aie build instructions in its README

# 4. Build ggnpu kernels
export AIE_HOME=~/mlir-aie/build
export PEANO_HOME=~/llvm-aie/build
cd ~/Documents/GitHub/ggnpu
./scripts/build-kernels.sh npu6 matmul
```

The build script compiles all kernel types: `matmul`, `rmsnorm`, `rope`, `softmax`, `silu`, `flash_attn`.
For 16 GB RAM machines, build kernels one at a time (e.g. `matmul` first):

```bash
./scripts/build-kernels.sh npu6 matmul
./scripts/build-kernels.sh npu6 rmsnorm
# ... etc
```

Output goes to `~/.cache/ggnpu/xclbin/`.

### Verify host

```bash
bash scripts/setup-host.sh
bash scripts/verify-npu.sh
```

These scripts check hardware, driver, permissions, XRT, and the local xclbin cache.

---

**Implementation spec:** [IMPLEMENTATION.md](IMPLEMENTATION.md) — complete handoff for contributors and AI agents.

**Full host setup guide:** [docs/host-setup-guide.md](docs/host-setup-guide.md)

**Kernel builds:** Local kernel compilation requires `mlir-aie` and Peano. If you already have prebuilt `.xclbin` files, place them under `~/.cache/ggnpu/xclbin/`.
