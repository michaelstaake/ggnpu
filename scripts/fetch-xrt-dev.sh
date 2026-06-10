#!/bin/bash
# Extract libxrt-dev + uuid-dev headers/libs into third_party/ without a full apt install.
# Kernel compilation (Triton-XDNA compile-only launcher) needs XRT C++ headers.
#
# Usage:
#   bash scripts/fetch-xrt-dev.sh          # download matching debs and extract
#   bash scripts/fetch-xrt-dev.sh --check  # exit 0 if headers already present

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
XRT_ROOT="$SCRIPT_DIR/third_party/xrt-dev/usr"
UUID_ROOT="$SCRIPT_DIR/third_party/uuid-dev/usr"
DL_DIR="$SCRIPT_DIR/build/xrt-deb-cache"

have_headers() {
    test -f "$XRT_ROOT/include/xrt/xrt_bo.h" && test -d "$XRT_ROOT/lib" \
        && test -f "$UUID_ROOT/include/uuid/uuid.h"
}

if [ "${1:-}" = "--tools" ]; then
    # Fetch xrt-tools (xclbinutil etc.) without sudo.
    if command -v xclbinutil >/dev/null 2>&1 || test -x "$SCRIPT_DIR/third_party/xrt-tools/usr/bin/xclbinutil"; then
        echo "xclbinutil already available"
        exit 0
    fi
    mkdir -p "$DL_DIR" "$SCRIPT_DIR/third_party/xrt-tools"
    cd "$DL_DIR"
    echo "Downloading xrt-tools..."
    if ! apt-get download xrt-tools; then
        echo "ERROR: apt-get download xrt-tools failed. Try: sudo apt install xrt-tools"
        exit 1
    fi
    TOOLS_DEB="$(ls -1 xrt-tools_*.deb 2>/dev/null | head -1)"
    dpkg-deb -x "$TOOLS_DEB" "$SCRIPT_DIR/third_party/xrt-tools"
    test -x "$SCRIPT_DIR/third_party/xrt-tools/usr/bin/xclbinutil" || { echo "ERROR: xclbinutil missing after extract"; exit 1; }
    echo "xclbinutil ready at third_party/xrt-tools/usr/bin/xclbinutil"
    exit 0
fi

if [ "${1:-}" = "--check" ]; then
    if have_headers; then
        exit 0
    fi
    if { test -f /usr/include/xrt/xrt_bo.h || test -f /opt/xilinx/xrt/include/xrt/xrt_bo.h; } \
        && test -f /usr/include/uuid/uuid.h; then
        exit 0
    fi
    exit 1
fi

echo "=== Fetch XRT development headers for kernel builds ==="

if have_headers; then
    echo "Already present: $XRT_ROOT"
    exit 0
fi

if test -f /usr/include/xrt/xrt_bo.h && test -f /usr/include/uuid/uuid.h; then
    echo "System libxrt-dev + uuid-dev already installed"
    exit 0
fi

if test -f /opt/xilinx/xrt/include/xrt/xrt_bo.h; then
    echo "System XRT SDK already installed (/opt/xilinx/xrt)"
    exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: apt-get not found"
    echo "  Install manually: sudo apt install libxrt-dev uuid-dev"
    exit 1
fi

if ! dpkg -l libxrt2 >/dev/null 2>&1; then
    echo "ERROR: libxrt2 runtime not installed"
    echo "  sudo apt install libxrt2 libxrt-npu2"
    exit 1
fi

XRT_VERSION="$(dpkg-query -W -f='${Version}' libxrt2 2>/dev/null || true)"
if [ -z "$XRT_VERSION" ]; then
    echo "ERROR: could not determine libxrt2 package version"
    exit 1
fi

echo "libxrt2 version: $XRT_VERSION"
mkdir -p "$DL_DIR" "$SCRIPT_DIR/third_party/xrt-dev" "$SCRIPT_DIR/third_party/uuid-dev"

cd "$DL_DIR"
echo "Downloading libxrt-dev and uuid-dev..."
if ! apt-get download "libxrt-dev=${XRT_VERSION}" uuid-dev; then
    echo ""
    echo "ERROR: apt-get download failed."
    echo "  Try: sudo apt install libxrt-dev uuid-dev"
    echo "  Or enable the AMD XRT apt repo from docs/host-setup-guide.md"
    exit 1
fi

XRT_DEB="$(ls -1 libxrt-dev_*.deb 2>/dev/null | head -1)"
UUID_DEB="$(ls -1 uuid-dev_*.deb 2>/dev/null | head -1)"
if [ -z "$XRT_DEB" ] || [ -z "$UUID_DEB" ]; then
    echo "ERROR: expected .deb files not found in $DL_DIR"
    exit 1
fi

echo "Extracting $XRT_DEB -> third_party/xrt-dev/"
dpkg-deb -x "$XRT_DEB" "$SCRIPT_DIR/third_party/xrt-dev"
echo "Extracting $UUID_DEB -> third_party/uuid-dev/"
dpkg-deb -x "$UUID_DEB" "$SCRIPT_DIR/third_party/uuid-dev"

if ! have_headers; then
    echo "ERROR: extract finished but $XRT_ROOT/include/xrt/xrt_bo.h is missing"
    exit 1
fi

echo ""
echo "XRT dev headers ready at:"
echo "  export XILINX_XRT=$XRT_ROOT"
echo ""
echo "Build kernels:"
echo "  source ~/triton-env/bin/activate"
echo "  ./scripts/build-kernels.sh npu6"
