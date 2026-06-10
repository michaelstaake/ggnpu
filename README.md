# ggnpu

Run GGUF models on AMD NPUs (Krackan / XDNA2).

## Native host setup

`ggnpu` is intended to be built and run directly on the host.

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
No NPU hardware is needed to build kernels — the compilation runs entirely on CPU.
You can build on a separate machine (no NPU required) and copy the `.xclbin` files to the NPU machine.

```bash
# 1. Clone mlir-aie
git clone https://github.com/Xilinx/mlir-aie.git ~/mlir-aie
cd ~/mlir-aie
git submodule update --init --recursive

# 2. Create and activate venv (critical — pip will fail outside venv)
python3 -m venv ironenv
source ironenv/bin/activate

# 3. Install Python deps
pip install --upgrade pip
pip install -r python/requirements.txt
pip install -r python/requirements_dev.txt
pip install nanobind

# 4. Build mlir-aie using pre-built wheels (lightweight)
bash ./utils/build-mlir-aie-from-wheels.sh
source utils/env_setup.sh install

# 5. Clone and build Peano (AIE tile code compiler)
git clone https://github.com/Xilinx/llvm-aie.git ~/llvm-aie
cd ~/llvm-aie
mkdir build && cd build
cmake -GNinja ..
ninja

# 6. Set environment variables
export AIE_HOME=~/mlir-aie/build
export PEANO_HOME=~/llvm-aie/build

# 7. Build ggnpu kernels (start with matmul — the critical kernel)
cd ~/Documents/GitHub/ggnpu
./scripts/build-kernels.sh npu6 matmul

# Build additional kernels as needed
./scripts/build-kernels.sh npu6 rmsnorm
./scripts/build-kernels.sh npu6 rope
./scripts/build-kernels.sh npu6 softmax
./scripts/build-kernels.sh npu6 silu
./scripts/build-kernels.sh npu6 flash_attn
```

Output goes to `~/.cache/ggnpu/xclbin/`.

**Building on a non-NPU machine:** Steps 1-7 work on any Ubuntu 24.04+ machine with ~16 GB RAM (32 GB recommended). No NPU, no `amdxdna` driver, no firmware needed. After building, copy the xclbins to the NPU machine:
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

**Kernel builds:** Local kernel compilation requires `mlir-aie` and Peano. If you already have prebuilt `.xclbin` files, place them under `~/.cache/ggnpu/xclbin/`.
