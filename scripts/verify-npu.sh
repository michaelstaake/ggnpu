#!/bin/bash
# Verify NPU setup for GGNPU
# Checks all prerequisites for running on AMD Ryzen AI 7 350

set -e

PASS=0
FAIL=0

check() {
    local desc="$1"
    local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  [PASS] $desc"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== GGNPU NPU Verification ==="
echo ""

# Hardware
echo "Hardware:"
check "AMD NPU present" "lspci | grep -i 'neural processing'"
check "PCI ID 1022:17f0" "lspci -d 1022:17f0 | grep -q ."

# Kernel
echo ""
echo "Kernel:"
check "amdxdna module" "lsmod | grep -q amdxdna"
check "IOMMU enabled" "dmesg | grep -q 'amd_iommu=on'"
check "accel0 device" "test -e /dev/accel/accel0"

# Groups
echo ""
echo "Groups:"
check "User in render group" "id -nG | grep -qw render"

# Memory
echo ""
echo "Memory:"
check "Memlock unlimited" "[ \"\$(ulimit -l)\" = \"unlimited\" ]"

# Firmware
echo ""
echo "Firmware:"
check "NPU firmware" "test -d /usr/lib/firmware/amdnpu"

# XRT
echo ""
echo "XRT:"
check "XRT library" "ls /opt/xilinx/xrt/lib/libxrt_coreutil.so 2>/dev/null || ls /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so 2>/dev/null"
check "XRT include" "ls /opt/xilinx/include/xrt.h 2>/dev/null || ls /usr/include/xrt.h 2>/dev/null"

echo ""
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "Some checks failed. See scripts/setup-host.sh for fix instructions."
    exit 1
fi

echo ""
echo "All checks passed! Ready to build and run GGNPU."
