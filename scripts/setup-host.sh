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
if dmesg | grep -q "amd_iommu=on"; then
    echo "  IOMMU enabled"
else
    echo "  WARNING: IOMMU may not be enabled"
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

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Install XRT: sudo apt install libxrt2 libxrt-npu2"
echo "  2. Set memlock: echo '* soft memlock unlimited' | sudo tee /etc/security/limits.d/99-amdxdna.conf"
echo "  3. Add to render group: sudo usermod -aG render \$USER"
echo "  4. Build: mkdir build && cd build && cmake .. && make"
