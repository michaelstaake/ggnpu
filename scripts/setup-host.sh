#!/bin/bash
# Host prerequisite checks for native GGNPU deployments.
# Run on Ubuntu 24.04 or 26.04 with an AMD Ryzen AI NPU.

set -e

echo "=== GGNPU Host Setup (native host) ==="

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
    echo "    Build with Triton-XDNA: bash scripts/setup-triton-env.sh && ./scripts/build-kernels.sh npu6"
    echo "    Or copy prebuilt xclbins into that directory"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps (native host):"
echo "  1. Install build deps: sudo apt install build-essential cmake git libxrt2 libxrt-npu2 libxrt-dev"
echo "  2. Add to render group: sudo usermod -aG render \$USER  (re-login)"
echo "  3. Build: cmake -S . -B build-npu -DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=OFF -DGGNPU_BUILD_TESTS=ON"
echo "  4. Compile: cmake --build build-npu -j2"
echo "  5. If needed, build kernels: bash scripts/setup-triton-env.sh && ./scripts/build-kernels.sh npu6 matmul"
echo "  6. Run: ./build-npu/ggnpu bench-matmul"
echo ""
echo "See README.md and docs/host-setup-guide.md"
