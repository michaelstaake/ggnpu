#!/bin/bash
# Host prerequisite checks for Docker-based GGNPU deployments.
# ggnpu runs ONLY inside Docker — this script does NOT install XRT or mlir-aie.
# Run on Ubuntu 26.04 with AMD Ryzen AI 7 350

set -e

echo "=== GGNPU Host Setup (Docker deployments) ==="

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
echo "Next steps (Docker — do not install XRT/mlir-aie on host):"
echo "  1. Install Docker: sudo apt install docker.io docker-compose-v2"
echo "  2. Add to groups: sudo usermod -aG docker,render \$USER  (re-login)"
echo "  3. cp docker/.env.example docker/.env  # set RENDER_GID"
echo "  4. docker compose -f docker/docker-compose.yml build ggnpu"
echo "  5. docker compose -f docker/docker-compose.yml --profile build run --rm builder"
echo "  6. docker compose -f docker/docker-compose.yml run --rm ggnpu bench-matmul"
echo ""
echo "See README.md and docs/docker.md"
