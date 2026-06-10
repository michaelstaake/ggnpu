#!/bin/bash
# Preflight checks for Triton-XDNA kernel builds (including WSL build machines).
#
# Usage:
#   bash scripts/verify-kernel-build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0
WARN=0

ok() {
    echo "  [PASS] $1"
    PASS=$((PASS + 1))
}

bad() {
    echo "  [FAIL] $1"
    FAIL=$((FAIL + 1))
}

warn() {
    echo "  [WARN] $1"
    WARN=$((WARN + 1))
}

echo "=== GGNPU kernel build preflight ==="
echo "Repo: $SCRIPT_DIR"
echo ""

# WSL / filesystem
if grep -qi microsoft /proc/version 2>/dev/null; then
    ok "WSL detected"
    case "$SCRIPT_DIR" in
        /mnt/*)
            bad "Repo is on Windows mount ($SCRIPT_DIR)"
            echo "         Move or re-clone to native Linux FS, e.g.:"
            echo "           git clone <url> ~/ggnpu && cd ~/ggnpu"
            ;;
        *)
            ok "Repo on native Linux filesystem"
            ;;
    esac
else
    ok "Native Linux (not WSL)"
fi
echo ""

# Toolchain
if command -v g++ >/dev/null 2>&1; then
    ok "g++ present ($(g++ --version | head -1))"
else
    bad "g++ not found (sudo apt install build-essential)"
fi

if command -v python3 >/dev/null 2>&1; then
    PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    ok "python3 present ($PY_VER)"
    if [ -f "/usr/include/python${PY_VER}/Python.h" ]; then
        ok "python${PY_VER}-dev headers"
    else
        bad "python${PY_VER}-dev headers missing"
        echo "         sudo apt install python${PY_VER}-dev"
    fi
else
    bad "python3 not found"
fi
echo ""

# Triton venv
TRITON_PY=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    TRITON_PY="${VIRTUAL_ENV}/bin/python"
elif [ -x "${HOME}/triton-env/bin/python" ]; then
    TRITON_PY="${HOME}/triton-env/bin/python"
elif [ -x "${SCRIPT_DIR}/.venv-triton/bin/python" ]; then
    TRITON_PY="${SCRIPT_DIR}/.venv-triton/bin/python"
fi

if [ -n "$TRITON_PY" ]; then
    ok "Triton venv python ($TRITON_PY)"
    if "$TRITON_PY" -c "from triton.backends.amd_triton_npu.driver import NPUDriver" 2>/dev/null; then
        ok "triton-xdna importable"
    else
        bad "triton-xdna not importable in venv"
        echo "         bash scripts/setup-triton-env.sh"
    fi
    if "$TRITON_PY" -c "import pathlib, sysconfig; assert (pathlib.Path(sysconfig.get_paths()['purelib']) / 'llvm-aie').is_dir()" 2>/dev/null; then
        ok "llvm-aie / PEANO wheel present"
    else
        bad "llvm-aie (PEANO) wheel missing from venv"
    fi
    if command -v xclbinutil >/dev/null 2>&1 \
        || test -x /opt/xilinx/xrt/bin/xclbinutil \
        || test -x "$SCRIPT_DIR/third_party/xrt-tools/usr/bin/xclbinutil"; then
        ok "xclbinutil present (xclbin packaging)"
    else
        bad "xclbinutil missing — install libxrt-utils libxrt-utils-npu (or scripts/fetch-xrt-dev.sh --tools)"
    fi
else
    bad "No triton venv found (~/triton-env or .venv-triton)"
    echo "         bash scripts/setup-triton-env.sh"
fi
echo ""

# XRT packages
if dpkg -s libxrt2 >/dev/null 2>&1; then
    ok "libxrt2 runtime ($(dpkg-query -W -f='${Version}' libxrt2))"
else
    bad "libxrt2 not installed"
    echo "         sudo apt install libxrt2"
fi

if dpkg -s libxrt-dev >/dev/null 2>&1; then
    ok "libxrt-dev installed"
else
    bad "libxrt-dev not installed"
    echo "         sudo apt install libxrt-dev"
    echo "         or: bash scripts/fetch-xrt-dev.sh"
fi

if dpkg -s uuid-dev >/dev/null 2>&1; then
    ok "uuid-dev installed"
else
    bad "uuid-dev not installed"
    echo "         sudo apt install uuid-dev"
fi

if [ -f /usr/include/xrt/xrt_bo.h ] || [ -f "${SCRIPT_DIR}/third_party/xrt-dev/usr/include/xrt/xrt_bo.h" ]; then
    ok "XRT headers present"
else
    bad "XRT headers missing"
fi

if [ -f /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so ] || \
   [ -f /usr/lib/x86_64-linux-gnu/libxrt_coreutil.so.2 ] || \
   [ -f "${SCRIPT_DIR}/third_party/xrt-dev/usr/lib/x86_64-linux-gnu/libxrt_coreutil.so" ]; then
    ok "libxrt_coreutil available for linking"
else
    bad "libxrt_coreutil not found"
fi

if [ -f /usr/lib/x86_64-linux-gnu/libuuid.so ] || [ -f /usr/lib/x86_64-linux-gnu/libuuid.so.1 ]; then
    ok "libuuid available for linking"
else
    bad "libuuid not found"
fi
echo ""

# Python-driven link smoke test (same checks compile_kernels.py uses)
if [ -n "$TRITON_PY" ]; then
    echo "Running launcher link smoke test..."
    if "$TRITON_PY" - "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "kernels" / "triton"))
import compile_kernels as ck

try:
    env = ck.setup_compile_env(repo)
except RuntimeError as e:
    print(f"SMOKE_FAIL: {e}")
    sys.exit(1)

err = ck.run_launcher_link_smoke_test(env, repo)
if err:
    print("SMOKE_FAIL:")
    print(err)
    sys.exit(1)
print("SMOKE_OK")
PY
    then
        ok "g++ xrt_dispatch link smoke test"
    else
        bad "g++ xrt_dispatch link smoke test"
    fi
fi

echo ""
echo "=== Results ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Warnings: $WARN"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "Fix the [FAIL] items above, then retry:"
    echo "  ./scripts/build-kernels.sh npu6 matmul"
    exit 1
fi

echo "Preflight OK — kernel builds should work."
echo "  ./scripts/build-kernels.sh npu6"
