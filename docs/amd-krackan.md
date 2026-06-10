# AMD Krackan NPU Guide

## Hardware

- **CPU**: AMD Ryzen AI 7 350 (Krackan Point)
- **NPU**: XDNA2 (AIE2P), 4×8 tile array, ~50 TOPS
- **PCI ID**: 1022:17f0 rev 0x20
- **Profile**: npu6
- **Device**: /dev/accel/accel0

## Kernel Requirements

```
CONFIG_DRM_ACCEL_AMDXDMA=y
CONFIG_AMD_IOMMU=y
```

Boot parameter: `amd_iommu=on`

## Setup

```bash
# Verify NPU
lspci -vd 1022:17f0
lsmod | grep amdxdna
ls -la /dev/accel/accel0

# Install XRT
sudo add-apt-repository ppa:amd-team/xrt
sudo apt install libxrt2 libxrt-npu2 amdxdna-dkms

# Memlock
echo '* soft memlock unlimited' | sudo tee /etc/security/limits.d/99-amdxdna.conf
echo '* hard memlock unlimited' | sudo tee -a /etc/security/limits.d/99-amdxdna.conf

# Group
sudo usermod -aG render $USER
```

## Native run

```bash
cmake -S . -B build-npu \
  -DGGNPU_NPU_BACKEND=ON \
  -DGGNPU_TEST_CPU=OFF \
  -DGGNPU_BUILD_TESTS=ON
cmake --build build-npu -j2

./build-npu/ggnpu bench-matmul
./build-npu/ggnpu -m models/model.gguf -p "Hello"
```
