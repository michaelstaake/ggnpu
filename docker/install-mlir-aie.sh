#!/bin/bash
# Install mlir-aie toolchain inside Docker using upstream wheel build script.
# Produces AIE_HOME with bin/aiecc.py (and bin/aiecc) under the install prefix.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
MLIR_AIE_SRC="${MLIR_AIE_SRC:-/opt/mlir-aie}"
MLIR_AIE_INSTALL="${MLIR_AIE_INSTALL:-/opt/mlir-aie-install}"

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    git \
    clang \
    lld \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates

if [ ! -d "$MLIR_AIE_SRC/.git" ]; then
    git clone --depth 1 https://github.com/Xilinx/mlir-aie.git "$MLIR_AIE_SRC"
    git -C "$MLIR_AIE_SRC" submodule update --init --recursive
fi

python3 -m venv /opt/ironenv
# shellcheck disable=SC1091
source /opt/ironenv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r "$MLIR_AIE_SRC/python/requirements.txt"

cd "$MLIR_AIE_SRC"
bash ./utils/build-mlir-aie-from-wheels.sh

# shellcheck disable=SC1091
source utils/env_setup.sh install

# env_setup sets mlir-aie install; discover prefix containing bin/aiecc
AIE_HOME=""
for candidate in \
    "$MLIR_AIE_SRC/build/install" \
    "$MLIR_AIE_SRC/my_install/mlir_aie" \
    "$MLIR_AIE_INSTALL"; do
    if [ -x "$candidate/bin/aiecc" ] || [ -f "$candidate/bin/aiecc.py" ]; then
        AIE_HOME="$candidate"
        break
    fi
done

if [ -z "$AIE_HOME" ]; then
    echo "ERROR: mlir-aie install directory not found after build" >&2
    find "$MLIR_AIE_SRC" -name 'aiecc' -o -name 'aiecc.py' 2>/dev/null | head -5 >&2 || true
    exit 1
fi

echo "$AIE_HOME" > /etc/ggnpu-aie-home

AIE_HOME="$(cat /etc/ggnpu-aie-home)"
if [ ! -f "$AIE_HOME/bin/aiecc.py" ] && [ ! -x "$AIE_HOME/bin/aiecc" ]; then
    echo "ERROR: aiecc not found under $AIE_HOME/bin" >&2
    ls -la "$AIE_HOME/bin" 2>/dev/null || true
    exit 1
fi

echo "mlir-aie installed: AIE_HOME=$AIE_HOME"
rm -rf /var/lib/apt/lists/*
