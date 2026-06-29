#!/usr/bin/env bash
# ggnpu one-command setup: install host prerequisites, build, optionally build
# the NPU kernels, and verify. Drives the modular scripts in scripts/.
#
# Usage:
#   ./setup.sh                       # interactive (recommended)
#   ./setup.sh --yes                 # non-interactive; accept all defaults
#   ./setup.sh --cpu                 # CPU reference build (no NPU/driver needed)
#   ./setup.sh --skip-install        # skip the privileged host install
#   ./setup.sh --skip-build          # skip the C++ build
#   ./setup.sh --kernels-here        # build NPU kernels on this machine
#   ./setup.sh --kernels-separate    # kernels will be built elsewhere (just print docs)
#   ./setup.sh --no-kernels          # don't touch kernels
#
# Environment:
#   GGNPU_NPU_PROFILE   NPU profile for kernel builds (default: npu6)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="$SCRIPT_DIR/scripts"
PROFILE="${GGNPU_NPU_PROFILE:-npu6}"

#====// UX helpers //====
if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BLUE=$'\033[36m'
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi
banner() { printf '\n%s%s== %s ==%s\n' "$C_BOLD" "$C_BLUE" "$*" "$C_RESET"; }
say()  { printf '%s\n' "$*"; }
ok()   { printf '  %s[ok]%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '  %s[warn]%s %s\n' "$C_YELLOW" "$C_RESET" "$*"; }
err()  { printf '  %s[fail]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }

#====// Args //====
ASSUME_YES=""
DO_INSTALL=1
DO_BUILD=1
BUILD_MODE="npu"          # npu | cpu
KERNELS=""                # ""(ask) | here | separate | no

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes) ASSUME_YES=1 ;;
        --cpu) BUILD_MODE="cpu" ;;
        --skip-install) DO_INSTALL="" ;;
        --skip-build) DO_BUILD="" ;;
        --kernels-here) KERNELS="here" ;;
        --kernels-separate) KERNELS="separate" ;;
        --no-kernels) KERNELS="no" ;;
        -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next}{exit}' "$0"; exit 0 ;;
        *) err "unknown option: $1"; exit 2 ;;
    esac
    shift
done

# CPU-only builds never need the NPU driver or kernels.
if [ "$BUILD_MODE" = "cpu" ]; then
    [ -z "$KERNELS" ] && KERNELS="no"
fi

# Track state for the closing summary.
INSTALL_RESULT="skipped"
BUILD_RESULT="skipped"
KERNEL_RESULT="skipped"
VERIFY_RESULT="not run"
RELOGIN_MARKER="$(mktemp)"
trap 'rm -f "$RELOGIN_MARKER"' EXIT

# ask_kernels -> sets KERNELS to here|separate|no
ask_kernels() {
    if [ -n "$KERNELS" ]; then return; fi
    if [ -n "$ASSUME_YES" ] || ! [ -t 0 ]; then
        # Non-interactive default: don't attempt the long kernel compile.
        KERNELS="no"
        return
    fi
    say ""
    say "NPU kernels (.xclbin) are needed to actually run on the NPU."
    say "They compile on CPU and take a while (no NPU hardware required to build)."
    say "  ${C_BOLD}t${C_RESET}) build them here on this machine now"
    say "  ${C_BOLD}s${C_RESET}) I'll build them on a separate machine (show me how)"
    say "  ${C_BOLD}k${C_RESET}) skip for now"
    local reply
    printf 'Choose %s[t/s/K]%s ' "$C_BOLD" "$C_RESET"
    read -r reply || true
    case "${reply,,}" in
        t|this|here) KERNELS="here" ;;
        s|sep|separate) KERNELS="separate" ;;
        *) KERNELS="no" ;;
    esac
}

#====// Intro //====
say "${C_BOLD}ggnpu setup${C_RESET}"
say "Steps: host install -> build -> NPU kernels -> verify."
say "Re-running is safe; finished work is skipped."

#====// 1. Host install //====
banner "[1/4] Host prerequisites"
if [ -n "$DO_INSTALL" ]; then
    install_args=()
    [ -n "$ASSUME_YES" ] && install_args+=(--yes)
    # CPU-only build doesn't need the XRT/NPU driver stack.
    [ "$BUILD_MODE" = "cpu" ] && install_args+=(--skip-driver)
    if GGNPU_RELOGIN_MARKER="$RELOGIN_MARKER" bash "$SCRIPTS/install-host.sh" "${install_args[@]}"; then
        INSTALL_RESULT="done"
    else
        rc=$?
        if [ "$rc" = "130" ]; then
            err "Host install aborted. Re-run ./setup.sh when ready."
            exit 130
        fi
        err "Host install failed (see above). Fix the reported issue and re-run ./setup.sh"
        exit 1
    fi
else
    warn "skipped (--skip-install)"
fi

#====// 2. Build //====
banner "[2/4] Build ggnpu"
if [ -n "$DO_BUILD" ]; then
    build_args=()
    [ "$BUILD_MODE" = "cpu" ] && build_args+=(--cpu)
    if bash "$SCRIPTS/build.sh" "${build_args[@]}"; then
        BUILD_RESULT="done"
    else
        err "Build failed (see cmake output above)."
        if [ "$BUILD_MODE" = "npu" ]; then
            say "      If XRT headers are missing: bash scripts/install-host.sh"
        fi
        exit 1
    fi
else
    warn "skipped (--skip-build)"
fi

#====// 3. Kernels //====
banner "[3/4] NPU kernels"
ask_kernels
case "$KERNELS" in
    here)
        say "Setting up the Triton-XDNA build environment (one-time, downloads wheels)..."
        if ! bash "$SCRIPTS/setup-triton-env.sh"; then
            err "Triton-XDNA env setup failed. See README 'Common failures'."
            KERNEL_RESULT="failed (env setup)"
        else
            say "Preflight check..."
            if ! bash "$SCRIPTS/verify-kernel-build.sh"; then
                err "Kernel build preflight failed — fix the [FAIL] items above and re-run ./setup.sh"
                KERNEL_RESULT="failed (preflight)"
            else
                say "Compiling kernels for $PROFILE (this takes a while)..."
                if bash "$SCRIPTS/build-kernels.sh" "$PROFILE"; then
                    KERNEL_RESULT="built ($PROFILE)"
                else
                    err "Kernel compile failed (see above). You can re-run ./setup.sh to retry."
                    KERNEL_RESULT="failed (compile)"
                fi
            fi
        fi
        ;;
    separate)
        warn "You chose to build kernels on a separate machine."
        say "      Follow the guide:"
        say "        README.md  -> 'Build NPU kernels (Triton-XDNA)'"
        say "        docs/host-setup-guide.md"
        say "      On that machine, build for $PROFILE, then copy the artifacts here:"
        say "        scp -r ~/.cache/ggnpu/xclbin/* $USER@$(hostname):~/.cache/ggnpu/xclbin/"
        say "      Both the .xclbin AND *_sequence.bin files are needed per op."
        say "      Then re-run ./setup.sh (or just ./build-npu/ggnpu bench-matmul) here."
        KERNEL_RESULT="deferred (separate machine)"
        ;;
    no)
        warn "Skipping kernel build."
        if ls "$HOME/.cache/ggnpu/xclbin/"*.xclbin >/dev/null 2>&1; then
            ok "existing xclbins found in ~/.cache/ggnpu/xclbin/"
            KERNEL_RESULT="using existing xclbins"
        else
            warn "no xclbins present — NPU inference won't run until they're built/copied in."
            KERNEL_RESULT="skipped (no xclbins)"
        fi
        ;;
esac

#====// 4. Verify //====
banner "[4/4] Verify"
if [ "$BUILD_MODE" = "cpu" ]; then
    say "CPU build — skipping NPU hardware verification."
    VERIFY_RESULT="skipped (cpu build)"
elif bash "$SCRIPTS/verify-npu.sh"; then
    VERIFY_RESULT="all checks passed"
else
    VERIFY_RESULT="some checks failed (see above)"
fi

#====// Summary //====
banner "Summary"
say "  host install : $INSTALL_RESULT"
say "  build        : $BUILD_RESULT"
say "  kernels      : $KERNEL_RESULT"
say "  verify       : $VERIFY_RESULT"
say ""

if [ -f "$RELOGIN_MARKER" ] && [ -s "$RELOGIN_MARKER" ]; then
    warn "${C_BOLD}Action needed:${C_RESET} log out and back in (or reboot) so the render group"
    warn "and memlock limits take effect, then continue below."
    say ""
fi

if [ "$BUILD_MODE" = "cpu" ]; then
    BIN="./build-cpu/ggnpu"
else
    BIN="./build-npu/ggnpu"
fi
say "Next:"
say "  $BIN bench-matmul"
say "  $BIN -m models/<model>.gguf -p \"The capital of France is\" -c 2048 -n 32"

case "$KERNEL_RESULT" in
    skipped*|deferred*|failed*)
        say ""
        warn "NPU inference needs xclbins in ~/.cache/ggnpu/xclbin/ — it will not run until"
        warn "those are built (./setup.sh --kernels-here) or copied from a build machine."
        ;;
esac
