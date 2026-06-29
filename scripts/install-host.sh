#!/usr/bin/env bash
# Install the privileged host prerequisites for running ggnpu on an AMD NPU.
# Mirrors README "Full installation" steps 1-4, but actually performs them.
# Idempotent: re-running skips work that is already done.
#
# Usage:
#   bash scripts/install-host.sh            # interactive, asks before sudo
#   bash scripts/install-host.sh --yes      # non-interactive (assume yes)
#   bash scripts/install-host.sh --skip-driver  # build tools only (kernel-build box)
#
# Environment:
#   GGNPU_ASSUME_YES=1   same as --yes

set -euo pipefail

#====// UX helpers //====
if [ -t 1 ]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BLUE=$'\033[36m'
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_BLUE=""
fi
say()  { printf '%s\n' "$*"; }
info() { printf '%s==>%s %s\n' "$C_BLUE" "$C_RESET" "$*"; }
ok()   { printf '  %s[ok]%s %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '  %s[warn]%s %s\n' "$C_YELLOW" "$C_RESET" "$*"; }
err()  { printf '  %s[fail]%s %s\n' "$C_RED" "$C_RESET" "$*" >&2; }

ASSUME_YES="${GGNPU_ASSUME_YES:-}"
SKIP_DRIVER=""

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes) ASSUME_YES=1 ;;
        --skip-driver) SKIP_DRIVER=1 ;;
        -h|--help)
            awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next}{exit}' "$0"
            exit 0 ;;
        *) err "unknown option: $1"; exit 2 ;;
    esac
    shift
done

# Re-login / reboot reminder flag (consumed by the caller via this marker file).
NEEDS_RELOGIN=""

confirm() {
    # confirm "question" -> returns 0 for yes
    local prompt="$1"
    if [ -n "$ASSUME_YES" ]; then
        return 0
    fi
    if ! [ -t 0 ]; then
        err "no terminal for prompt and --yes not given: $prompt"
        exit 1
    fi
    local reply
    printf '%s %s[Y/n]%s ' "$prompt" "$C_BOLD" "$C_RESET"
    read -r reply || true
    case "${reply,,}" in
        n|no) return 1 ;;
        *) return 0 ;;
    esac
}

#====// Preconditions //====
info "ggnpu host install"

if ! command -v apt-get >/dev/null 2>&1; then
    err "apt-get not found — this installer targets Debian/Ubuntu hosts."
    say "  Install the equivalents of: build-essential cmake git clang lld ninja-build"
    say "  python3-{pip,venv,dev} uuid-dev, plus AMD XRT (libxrt2 libxrt-npu2 libxrt-dev"
    say "  libxrt-utils libxrt-utils-npu) and the amdxdna driver. See docs/host-setup-guide.md"
    exit 1
fi

if [ "$(id -u)" = "0" ]; then
    SUDO=""
else
    SUDO="sudo"
    if ! command -v sudo >/dev/null 2>&1; then
        err "sudo not found and not running as root."
        exit 1
    fi
fi

CORE_PKGS=(build-essential cmake git clang lld ninja-build python3-pip python3-venv python3-dev uuid-dev)
DRIVER_PKGS=(libxrt2 libxrt-npu2 libxrt-dev libxrt-utils libxrt-utils-npu amdxdna-dkms)

# Show the plan up front, then ask once.
say ""
say "${C_BOLD}This will run the following privileged steps:${C_RESET}"
say "  1. apt install: ${CORE_PKGS[*]}"
if [ -z "$SKIP_DRIVER" ]; then
    say "  2. add the AMD XRT PPA (ppa:amd-team/xrt) and apt install:"
    say "       ${DRIVER_PKGS[*]}"
    say "  3. write /etc/security/limits.d/99-amdxdna.conf (memlock unlimited)"
    say "  4. add '$USER' to the 'render' group"
else
    say "  (--skip-driver: only build tools above; no XRT/driver, memlock, or group changes)"
fi
say ""
if ! confirm "Proceed?"; then
    warn "Aborted by user. Nothing was changed."
    exit 130
fi

pkg_installed() { dpkg -s "$1" >/dev/null 2>&1; }

# missing_pkgs <array-name>  -> populates the global `MISSING` array
missing_pkgs() {
    local -n _list="$1"
    MISSING=()
    local p
    for p in "${_list[@]}"; do
        pkg_installed "$p" || MISSING+=("$p")
    done
}

#====// 1. Core build tools //====
info "[1/4] Core build tools"
missing_pkgs CORE_PKGS
need_core=("${MISSING[@]}")
if [ "${#need_core[@]}" -eq 0 ]; then
    ok "all core packages already installed"
else
    say "  installing: ${need_core[*]}"
    $SUDO apt-get update
    $SUDO apt-get install -y "${need_core[@]}"
    ok "core build tools installed"
fi

if [ -n "$SKIP_DRIVER" ]; then
    info "Skipping XRT/driver/memlock/group steps (--skip-driver)"
    say ""
    ok "Host build tools ready (kernel-build box)."
    exit 0
fi

#====// 2. XRT runtime + NPU driver //====
info "[2/4] AMD XRT runtime + NPU driver"
if grep -rqs 'amd-team/xrt' /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then
    ok "AMD XRT PPA already present"
else
    say "  adding ppa:amd-team/xrt"
    # software-properties-common provides add-apt-repository
    if ! command -v add-apt-repository >/dev/null 2>&1; then
        $SUDO apt-get install -y software-properties-common
    fi
    $SUDO add-apt-repository -y ppa:amd-team/xrt
fi

missing_pkgs DRIVER_PKGS
need_drv=("${MISSING[@]}")
if [ "${#need_drv[@]}" -eq 0 ]; then
    ok "XRT + driver packages already installed"
else
    say "  installing: ${need_drv[*]}"
    $SUDO apt-get update
    $SUDO apt-get install -y "${need_drv[@]}"
    ok "XRT runtime + amdxdna driver installed"
fi

#====// 3. Memlock limits //====
info "[3/4] Memlock limits (NPU pinned DMA buffers)"
LIMITS_FILE=/etc/security/limits.d/99-amdxdna.conf
if [ -f "$LIMITS_FILE" ] && grep -q 'memlock unlimited' "$LIMITS_FILE" 2>/dev/null; then
    ok "$LIMITS_FILE already configured"
else
    printf '* soft memlock unlimited\n* hard memlock unlimited\n' | $SUDO tee "$LIMITS_FILE" >/dev/null
    ok "wrote $LIMITS_FILE"
    NEEDS_RELOGIN=1
fi

#====// 4. Render group //====
info "[4/4] Device access (render group)"
if id -nG "$USER" | grep -qw render; then
    ok "$USER already in 'render' group"
else
    $SUDO usermod -aG render "$USER"
    ok "added $USER to 'render' group"
    NEEDS_RELOGIN=1
fi

#====// IOMMU (detect only) //====
info "IOMMU check"
if grep -q amd_iommu=on /proc/cmdline 2>/dev/null; then
    ok "amd_iommu=on present on kernel cmdline"
else
    warn "amd_iommu=on not found on the kernel cmdline."
    say "      The NPU needs IOMMU for DMA buffer pinning. To enable it:"
    say "        sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT=\"/&amd_iommu=on /' /etc/default/grub"
    say "        sudo update-grub && sudo reboot"
    say "      (also confirm the NPU/IPU is enabled in BIOS). Left unchanged — edit grub manually."
fi

#====// Summary //====
say ""
ok "Host install complete."
if [ -n "$NEEDS_RELOGIN" ]; then
    say ""
    warn "${C_BOLD}Log out and back in (or reboot)${C_RESET} for the render group and"
    warn "memlock limits to take effect before running on the NPU."
    # Surface the reminder to setup.sh, which looks for this file.
    : "${GGNPU_RELOGIN_MARKER:=}"
    if [ -n "${GGNPU_RELOGIN_MARKER}" ]; then
        : > "${GGNPU_RELOGIN_MARKER}" 2>/dev/null || true
    fi
fi
