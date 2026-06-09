#!/bin/bash
# Setup host for GGNPU development
# Run on Ubuntu 26.04 with AMD Ryzen AI 7 350

set -e

echo "=== GGNPU Host Setup ==="

# Check NPU hardware
echo ""
echo "Checking NPU hardware..."
if lspci -vd 1022:17f0 >/dev/null 2>&1; then
    echo "  NPU detected: $(lspci -vd 1022:17f0 | head -1)"
else
    echo "  WARNING: NPU not detected"
fi

# Check kernel driver
echo ""
echo "Checking kernel driver..."
if lsmod | grep -q amdxdna; then
    echo "  amdxdna driver loaded"
else
    echo "  WARNING: amdxdna driver not loaded"
fi

# Check device node
echo ""
echo "Checking device node..."
if [ -e /dev/accel/accel0 ]; then
    echo "  /dev/accel/accel0 exists"
else
    echo "  WARNING: /dev/accel/accel0 not found"
fi

# Check IOMMU
echo ""
echo "Checking IOMMU..."
if grep -q amd_iommu=on /proc/cmdline 2>/dev/null; then
    echo "  IOMMU enabled (kernel cmdline)"
elif dmesg 2>/dev/null | grep -q "amd_iommu=on"; then
    echo "  IOMMU enabled"
else
    echo "  WARNING: IOMMU may not be enabled (add amd_iommu=on to kernel cmdline)"
fi

# Check memlock
echo ""
echo "Checking memlock limits..."
ulimit -l
if [ "$(ulimit -l)" = "unlimited" ]; then
    echo "  memlock: unlimited (OK)"
else
    echo "  WARNING: memlock should be unlimited for NPU buffer pinning"
fi

# Check XRT
echo ""
echo "Checking XRT..."
if ldconfig -p 2>/dev/null | grep -q libxrt_coreutil; then
    echo "  XRT runtime installed"
else
    echo "  WARNING: XRT runtime not found"
fi
if test -f /usr/include/xrt/xrt_bo.h -o -f /opt/xilinx/xrt/include/xrt/xrt_bo.h; then
    echo "  XRT development headers installed"
elif test -f third_party/xrt-dev/usr/include/xrt/xrt_bo.h; then
    echo "  XRT headers in third_party/xrt-dev (local extract)"
else
    echo "  WARNING: libxrt-dev not installed (needed to build NPU backend)"
    echo "    sudo apt install libxrt-dev"
    echo "    Or: apt-get download libxrt-dev uuid-dev && extract to third_party/"
fi

# Check xclbin cache
echo ""
echo "Checking NPU kernels..."
if ls "$HOME/.cache/ggnpu/xclbin/"*.xclbin >/dev/null 2>&1; then
    echo "  xclbin cache: $(ls "$HOME/.cache/ggnpu/xclbin/"*.xclbin | wc -l) file(s)"
else
    echo "  WARNING: no xclbins in ~/.cache/ggnpu/xclbin/"
    echo "    Build with mlir-aie: AIE_HOME=... ./scripts/build-kernels.sh npu6"
    echo "    Or copy prebuilt xclbins into that directory"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Install XRT: sudo apt install libxrt2 libxrt-npu2 libxrt-dev"
echo "  2. Set memlock: echo '* soft memlock unlimited' | sudo tee /etc/security/limits.d/99-amdxdna.conf"
echo "  3. Add to render group: sudo usermod -aG render \$USER  (re-login)"
echo "  4. Verify: bash scripts/verify-npu.sh"
echo "  5. Build NPU binary:"
echo "       mkdir -p build && cd build"
echo "       cmake .. -DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=OFF"
echo "       make -j2"
echo "  6. Smoke test: ./ggnpu bench-matmul"
