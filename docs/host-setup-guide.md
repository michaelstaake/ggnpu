# Host Setup Guide for GGNPU

This guide walks through the host setup required to build and run `ggnpu` directly on an AMD XDNA NPU system.

## Prerequisites

- Ubuntu 24.04 or 26.04
- AMD Ryzen AI processor (Strix Point / Krackan)
- `lspci -vd 1022:17f0` shows the NPU device
- `lsmod | grep amdxdna` shows the kernel driver loaded

## Step 1: Install build and runtime dependencies

```bash
# Core build tools
sudo apt update
sudo apt install build-essential cmake git clang lld ninja-build python3-pip python3-venv libssl-dev

# Add AMD XRT PPA (if available)
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update

# Install XRT packages
sudo apt install libxrt2 libxrt-npu2 libxrt-dev amdxdna-dkms

# If PPA is not available, download from AMD's XRT releases:
# https://github.com/Xilinx/XRT/releases
# Then:
# sudo dpkg -i libxrt2*.deb libxrt-npu2*.deb libxrt-dev*.deb
```

Verify installation:
```bash
ls -la /opt/xilinx/xrt/setup.sh
dpkg -l | grep xrt
```

## Step 2: Set Memlock Limits

The NPU requires unlimited memlock for pinned DMA buffers.

```bash
# Create limits configuration
echo '* soft memlock unlimited' | sudo tee /etc/security/limits.d/99-amdxdna.conf
echo '* hard memlock unlimited' | sudo tee -a /etc/security/limits.d/99-amdxdna.conf

# Verify (requires re-login)
ulimit -l
# Should output: unlimited
```

## Step 3: Add User to Render Group

The NPU device node requires `render` group membership.

```bash
sudo usermod -aG render $USER

# Verify (requires re-login)
groups | grep render
# Should include: render
```

## Step 4: Enable IOMMU

IOMMU must be enabled in the kernel for NPU buffer management.

```bash
# Check current setting
dmesg | grep -i iommu
# Should show: AMD-IOMMU enabled

# If not enabled, add to GRUB:
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/&amd_iommu=on /' /etc/default/grub
sudo update-grub

# Reboot required
sudo reboot
```

## Step 5: Re-login

After group and limits changes, you must log out and back in (or reboot).

```bash
# Verify all changes took effect
echo "Memlock: $(ulimit -l)"
echo "Groups: $(groups)"
echo "XRT: $(ls /opt/xilinx/xrt/setup.sh 2>/dev/null || echo 'NOT FOUND')"
```

## Step 6: Verify NPU readiness

Run the repository's verification script:

```bash
bash scripts/verify-npu.sh
```

Expected output:
```
=== GGNPU NPU Verification ===

Hardware:
  [PASS] AMD NPU present
  [PASS] PCI ID 1022:17f0

Kernel:
  [PASS] amdxdna module
  [PASS] IOMMU enabled
  [PASS] accel0 device

Groups:
  [PASS] User in render group

Memory:
  [PASS] Memlock unlimited

Firmware:
  [PASS] NPU firmware

XRT:
  [PASS] XRT library
  [PASS] XRT include

=== Results ===
Passed: 10
Failed: 0
```

## Step 7: Build with the NPU backend

```bash
# Create build directory
mkdir -p build-npu
cd build-npu

# Configure with NPU backend enabled
cmake .. -DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=OFF -DGGNPU_BUILD_TESTS=ON

# Verify configuration output shows:
#   NPU backend:        ON
#   CPU ref backend:    OFF

# Build
cmake --build . -j2
```

## Step 8: Build NPU kernels (mlir-aie + Peano)

`bench-matmul` and NPU inference need `.xclbin` kernel artifacts in `~/.cache/ggnpu/xclbin/`.
No NPU hardware is needed to build kernels — the compilation runs entirely on CPU.

### Option A: Build on an NPU machine (same machine)

Install mlir-aie and Peano, then run the kernel build script:

```bash
# 1. Clone mlir-aie
git clone https://github.com/Xilinx/mlir-aie.git ~/mlir-aie
cd ~/mlir-aie
git submodule update --init --recursive

# 2. Create and activate venv (critical — pip will fail without it)
python3 -m venv ironenv
source ironenv/bin/activate

# 3. Install Python deps
pip install --upgrade pip
pip install -r python/requirements.txt
pip install -r python/requirements_dev.txt
pip install nanobind

# 4. Build mlir-aie (uses pre-built LLVM/MLIR wheels — much lighter than building LLVM from source)
bash ./utils/build-mlir-aie-from-wheels.sh
source utils/env_setup.sh install

# 5. Clone and build Peano (AIE tile code compiler)
git clone https://github.com/Xilinx/llvm-aie.git ~/llvm-aie
cd ~/llvm-aie
mkdir build && cd build
cmake -GNinja ..
ninja
cd ../..

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

### Option B: Build on a separate machine (no NPU needed)

The kernel build machine only needs Ubuntu 24.04+, ~16 GB RAM (32 GB recommended), and ~50 GB disk.
No NPU hardware, no `amdxdna` driver, no firmware required.

1. Follow the same steps as Option A on the build machine
2. After building, copy the xclbin files to the NPU machine:
   ```bash
   # On build machine
   scp -r ~/.cache/ggnpu/xclbin/* user@npu-machine:~/.cache/ggnpu/xclbin/
   ```

### Option C: Use a cloud VM

Spin up a 32 GB RAM VM (GCP, AWS, Lambda Labs), clone the repo, build kernels, and copy the `.xclbin` files back.

## Step 9: Provide kernels (quick reference)

After building, verify the xclbin files are in place:

```bash
ls -la ~/.cache/ggnpu/xclbin/
# Expected: matmul_npu6.xclbin, rmsnorm_npu6.xclbin, etc.
```

If you built on a different machine, copy them there.

## Step 10: Run benchmark

```bash
# Run matmul benchmark (Phase 2 smoke test)
./ggnpu bench-matmul
```

Expected output:
```
GGNPU Matmul Benchmark
======================

Backend: amd_xdna
NPU device opened: profile=npu6
Loaded xclbin: /path/to/matmul_npu6.xclbin UUID=...

  256x256 x 256x256: X.XX TFLOPS (X.XX ms)
  512x512 x 512x512: X.XX TFLOPS (X.XX ms)
  ...
```

## Troubleshooting

### XRT not found

```bash
# Check if XRT is installed
dpkg -l | grep xrt

# Check if setup.sh exists
ls -la /opt/xilinx/xrt/setup.sh

# Source XRT environment manually if needed
source /opt/xilinx/xrt/setup.sh
```

### Permission denied on /dev/accel/accel0

```bash
# Verify group membership
groups | grep render

# If missing, add user and re-login
sudo usermod -aG render $USER
# Then log out and back in
```

### Memlock still limited

```bash
# Check limits file
cat /etc/security/limits.d/99-amdxdna.conf

# Verify current limit
ulimit -l

# If still not unlimited, check PAM configuration
grep -r pam_limits /etc/pam.d/
```

### IOMMU not enabled

```bash
# Check kernel parameters
cat /proc/cmdline | grep iommu

# If missing, update GRUB as shown in Step 4
```

### No xclbin kernels found

```bash
# Check cache directory
ls -la ~/.cache/ggnpu/xclbin/

# If empty, you need to provide kernels:
# Option A: Use prebuilt xclbins
# Option B: Build with mlir-aie
#   export AIE_HOME=/path/to/mlir-aie
#   export PEANO_HOME=/path/to/peano
#   ./scripts/build-kernels.sh npu6 matmul
```

## Next Steps After Setup

1. Run `./ggnpu bench-matmul` to validate NPU execution
2. Run `./ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "Hello" -c 2048` for inference
3. Check `docs/amd-krackan.md` for hardware-specific notes
