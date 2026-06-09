#!/bin/bash
set -euo pipefail

# Default xclbin search paths inside the image
export GGNPU_CACHE_DIR="${GGNPU_CACHE_DIR:-/root/.cache/ggnpu}"

if [ -d /usr/share/ggnpu/xclbin ] && [ -z "$(ls -A "${GGNPU_CACHE_DIR}/xclbin" 2>/dev/null || true)" ]; then
    mkdir -p "${GGNPU_CACHE_DIR}/xclbin"
    cp -n /usr/share/ggnpu/xclbin/*.xclbin "${GGNPU_CACHE_DIR}/xclbin/" 2>/dev/null || true
fi

exec ggnpu "$@"
