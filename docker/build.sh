#!/bin/bash
# Convenience wrapper for Docker builds
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE="docker compose -f ${ROOT}/docker/docker-compose.yml"
UBUNTU_VERSION="${UBUNTU_VERSION:-26.04}"

usage() {
    cat <<EOF
Usage: $0 <command>

Commands:
  runtime     Build ggnpu:latest runtime image
  builder     Build ggnpu-builder:latest (mlir-aie + kernels)
  kernels     Run builder container to populate ggnpu-cache volume
  all         runtime + builder + kernels
  bench       Run bench-matmul in runtime container
  help        Show this message

Environment:
  UBUNTU_VERSION=26.04   Base image (default: 26.04)

Examples:
  $0 runtime
  $0 kernels
  $0 bench
EOF
}

cmd="${1:-help}"

case "$cmd" in
    runtime)
        UBUNTU_VERSION="$UBUNTU_VERSION" $COMPOSE build ggnpu
        ;;
    builder)
        UBUNTU_VERSION="$UBUNTU_VERSION" $COMPOSE build builder
        ;;
    kernels)
        UBUNTU_VERSION="$UBUNTU_VERSION" $COMPOSE --profile build run --rm builder
        ;;
    all)
        UBUNTU_VERSION="$UBUNTU_VERSION" $COMPOSE build ggnpu
        UBUNTU_VERSION="$UBUNTU_VERSION" $COMPOSE build builder
        UBUNTU_VERSION="$UBUNTU_VERSION" $COMPOSE --profile build run --rm builder
        ;;
    bench)
        $COMPOSE run --rm ggnpu bench-matmul
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown command: $cmd" >&2
        usage >&2
        exit 1
        ;;
esac
