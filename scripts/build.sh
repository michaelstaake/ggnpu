#!/usr/bin/env bash
# Configure and build ggnpu with CMake.
#
# Usage:
#   bash scripts/build.sh            # NPU backend build -> build-npu/ggnpu
#   bash scripts/build.sh --cpu      # CPU reference build -> build-cpu/ggnpu
#   bash scripts/build.sh --jobs 2   # override parallelism (default: nproc, capped on low RAM)
#
# Environment:
#   GGNPU_BUILD_JOBS   override parallel job count

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_GREEN=$'\033[32m'; C_RED=$'\033[31m'; C_BLUE=$'\033[36m'
else
    C_RESET=""; C_GREEN=""; C_RED=""; C_BLUE=""
fi
info() { printf '%s==>%s %s\n' "$C_BLUE" "$C_RESET" "$*"; }
ok()   { printf '  %s[ok]%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
err()  { printf '  %s[fail]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }

MODE="npu"
JOBS="${GGNPU_BUILD_JOBS:-}"

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) MODE="cpu" ;;
        --npu) MODE="npu" ;;
        -j|--jobs) shift; JOBS="${1:?--jobs needs a number}" ;;
        -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next}{exit}' "$0"; exit 0 ;;
        *) err "unknown option: $1"; exit 2 ;;
    esac
    shift
done

# Choose parallelism: default to nproc, but cap at 2 when RAM is tight (<16 GB),
# matching the README guidance (kernel/C++ builds OOM on small machines).
if [ -z "$JOBS" ]; then
    JOBS="$(nproc 2>/dev/null || echo 2)"
    mem_kb="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    if [ "$mem_kb" -gt 0 ] && [ "$mem_kb" -lt 16000000 ] && [ "$JOBS" -gt 2 ]; then
        JOBS=2
    fi
fi

if ! command -v cmake >/dev/null 2>&1; then
    err "cmake not found. Install it first: bash scripts/install-host.sh"
    exit 1
fi

if [ "$MODE" = "cpu" ]; then
    BUILD_DIR="$SCRIPT_DIR/build-cpu"
    CMAKE_FLAGS=(-DGGNPU_NPU_BACKEND=OFF -DGGNPU_TEST_CPU=ON -DGGNPU_BUILD_TESTS=ON)
    info "Configuring CPU reference build ($BUILD_DIR)"
else
    BUILD_DIR="$SCRIPT_DIR/build-npu"
    CMAKE_FLAGS=(-DGGNPU_NPU_BACKEND=ON -DGGNPU_TEST_CPU=OFF -DGGNPU_BUILD_TESTS=ON)
    info "Configuring NPU backend build ($BUILD_DIR)"
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    "${CMAKE_FLAGS[@]}"

info "Building with -j$JOBS (this can take a few minutes)"
cmake --build "$BUILD_DIR" -j"$JOBS"

BIN="$BUILD_DIR/ggnpu"
if [ -x "$BIN" ]; then
    ok "Build complete: $BIN"
    if [ "$MODE" = "npu" ]; then
        printf '       Smoke test: %s bench-matmul\n' "$BIN"
    fi
else
    err "build finished but $BIN is missing — check the cmake output above."
    exit 1
fi
