# Docker

Run **ggnpu** and build **NPU kernels** inside containers so the host only needs:

- Native Docker (not Docker Desktop’s VM)
- `amdxdna` driver + `/dev/accel/accel0`
- NPU firmware at `/usr/lib/firmware/amdnpu`
- Your user in the `render` group

Everything else (XRT libs, `libxrt-dev`, mlir-aie, Peano toolchain, ggnpu binary, xclbin builds) lives in Docker images based on **Ubuntu 26.04**.

## Images

| Image | Dockerfile | Purpose |
|-------|------------|---------|
| `ggnpu:latest` | `docker/Dockerfile` | Runtime: XRT + NPU-enabled `ggnpu` |
| `ggnpu-builder:latest` | `docker/Dockerfile.builder` | One-shot: mlir-aie + xclbin compile |

Shared Docker volume `ggnpu-cache` stores compiled kernels at `/root/.cache/ggnpu/xclbin` so the runtime container can use them without rebaking the image.

## Quick start

From the repo root:

```bash
# 1. Build runtime image (~5 min)
docker compose -f docker/docker-compose.yml build ggnpu

# 2. Build kernels inside builder (30–90 min first time; needs network)
docker compose -f docker/docker-compose.yml --profile build run --rm builder

# 3. Smoke test on host NPU
docker compose -f docker/docker-compose.yml run --rm ggnpu bench-matmul

# 4. Inference
docker compose -f docker/docker-compose.yml run --rm ggnpu \
  -m /models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 32
```

Put GGUF models in `models/` (repo-local, gitignored).

## What stays on the host

| Component | Where |
|-----------|--------|
| Kernel driver `amdxdna` | Host kernel module |
| `/dev/accel/accel0` | Passed into container |
| NPU firmware | Bind-mount ` /usr/lib/firmware/amdnpu` |
| IOMMU (`amd_iommu=on`) | Host boot parameter |
| `render` group | Host user / compose `group_add` |

Do **not** install mlir-aie, Peano, or `libxrt-dev` on the host unless you want native builds.

## Manual `docker run`

Runtime:

```bash
docker build -t ggnpu:latest -f docker/Dockerfile .

RENDER_GID="$(getent group render | cut -d: -f3)"

docker run --rm \
  --device=/dev/accel/accel0 \
  --group-add "${RENDER_GID}" \
  --ulimit memlock=-1:-1 \
  -v /usr/lib/firmware/amdnpu:/usr/lib/firmware/amdnpu:ro \
  -v ggnpu-cache:/root/.cache/ggnpu \
  -v "$(pwd)/models:/models:ro" \
  ggnpu:latest bench-matmul
```

Builder (populate cache volume):

```bash
docker build -t ggnpu-builder:latest -f docker/Dockerfile.builder .

docker volume create ggnpu-cache

docker run --rm \
  -v ggnpu-cache:/root/.cache/ggnpu \
  ggnpu-builder:latest
```

## Ubuntu version

Compose uses `UBUNTU_VERSION=26.04` by default. Override if the tag is unavailable locally:

```bash
UBUNTU_VERSION=24.04 docker compose -f docker/docker-compose.yml build
```

## Troubleshooting

### `bench-matmul`: no xclbin

Run the builder profile first, or copy prebuilt `matmul_npu6.xclbin` into the cache volume:

```bash
docker run --rm -v ggnpu-cache:/cache alpine \
  ls -la /cache/xclbin/
```

### Builder: mlir-aie compile fails

- First build needs **8–16 GB RAM** and stable network (downloads LLVM/mlir wheels).
- Start with matmul only: builder already runs `./scripts/build-kernels.sh npu6 matmul`.
- MLIR sources in `kernels/amd/` are still maturing; failures may be template issues, not Docker.

### NPU not visible in container

```bash
ls -la /dev/accel/accel0    # on host
groups | grep render
docker compose -f docker/docker-compose.yml run --rm ggnpu bench-matmul
```

Use **native Docker on Linux**, not Docker Desktop’s hidden VM.

### XRT version mismatch

Runtime image installs XRT from Ubuntu/AMD packages. If device open fails after a host driver upgrade, rebuild the image:

```bash
docker compose -f docker/docker-compose.yml build --no-cache ggnpu
```
