# Host Setup Guide for GGNPU NPU Development

This guide walks through the complete host setup required to build and run GGNPU with the AMD XDNA NPU backend.

## Prerequisites

- Ubuntu 24.04 or 26.04 (tested on 26.04)
- AMD Ryzen AI processor (Strix Point / Krackan)
- `lspci -vd 1022:17f0` shows the NPU device
- `lsmod | grep amdxdna` shows the kernel driver loaded

## Step 1: Install XRT

```bash
# Add AMD XRT PPA (if available)
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update

# Install XRT packages
sudo apt install libxrt2 libxrt-npu2 amdxdna-dkms

# If PPA is not available, download from AMD's XRT releases:
# https://github.com/Xilinx/XRT/releases
# Then:
# sudo dpkg -i libxrt2*.deb libxrt-npu2*.deb
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

## Step 6: Verify NPU Readiness

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

## Step 7: Build with NPU Backend

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
make -j2
```

## Step 8: Run Benchmark

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

# Source XRT environment manually
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

# If empty, you need to build kernels:
# Option A: Use prebuilt xclbins (recommended for 16GB RAM machines)
# Option B: Build with mlir-aie (requires >32GB RAM)
#   export AIE_HOME=/path/to/mlir-aie
#   cmake .. -DGGNPU_BUILD_KERNELS=ON
#   make build_npu_kernels
```

## Next Steps After Setup

1. Run `./ggnpu bench-matmul` to validate NPU execution
2. Run `./ggnpu -m models/llama-3.2-1b-q4_k_m.gguf -p "Hello" -c 2048` for inference
3. Check `docs/amd-krackan.md` for hardware-specific notes
