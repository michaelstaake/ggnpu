#!/bin/bash
# Install XRT runtime libraries inside Docker (no kernel driver — host provides /dev/accel).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update

# Base image on Ubuntu 26.04 may need a refresh before security pocket syncs
apt-get install -y --no-install-recommends ca-certificates || true
apt-get update

install_xrt_packages() {
    apt-get install -y --no-install-recommends \
        libxrt2 \
        libxrt-npu2 \
        libxrt-dev \
        uuid-dev \
        libuuid1
}

if apt-cache show libxrt2 >/dev/null 2>&1; then
    echo "Installing XRT from distribution packages..."
    install_xrt_packages
else
    echo "XRT not in distro repos; trying AMD PPA..."
    apt-get install -y --no-install-recommends gnupg software-properties-common curl
    add-apt-repository -y ppa:amd-team/xrt
    apt-get update
    install_xrt_packages
fi

rm -rf /var/lib/apt/lists/*
