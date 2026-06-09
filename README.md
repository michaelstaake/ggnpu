# ggnpu

Run GGUF models on AMD NPUs (Krackan / XDNA2).

## Docker required

**ggnpu runs only inside Docker.** Do not install XRT, mlir-aie, Peano, or build `ggnpu` on the host.

The container includes the `ggnpu` binary, XRT libraries, and (after a one-time builder step) NPU kernel artifacts. The host only provides the NPU driver and passes the device into the container.

### Host prerequisites (minimal)

| Required on host | Not required on host |
|------------------|----------------------|
| Linux with `amdxdna` loaded | XRT (`libxrt2`, `libxrt-dev`) |
| `/dev/accel/accel0` | mlir-aie / Peano |
| `/usr/lib/firmware/amdnpu` | `cmake`, native `ggnpu` build |
| Native Docker (not Docker Desktop VM) | `render` group name inside container (use GID; see below) |
| User in host `render` group | |

BIOS: NPU/IPU enabled. Boot: `amd_iommu=on`.

### Quick start

```bash
git clone https://github.com/michaelstaake/ggnpu.git
cd ggnpu

# Copy and set your render group GID (usually 990)
cp docker/.env.example docker/.env

# Build runtime image
docker compose -f docker/docker-compose.yml build ggnpu

# One-time: build NPU kernels inside Docker (30–90 min first run)
docker compose -f docker/docker-compose.yml build builder
docker compose -f docker/docker-compose.yml --profile build run --rm builder

# Smoke test
docker compose -f docker/docker-compose.yml run --rm ggnpu bench-matmul

# Inference (put GGUF files in models/)
docker compose -f docker/docker-compose.yml run --rm ggnpu \
  -m /models/llama-3.2-1b-q4_k_m.gguf -p "The capital of France is" -c 2048 -n 32
```

Helper script:

```bash
./docker/build.sh all    # build images + populate kernel cache
./docker/build.sh bench  # bench-matmul in container
```

Full details: [docs/docker.md](docs/docker.md)

### Verify host (optional)

```bash
bash scripts/verify-npu.sh
```

Checks hardware, driver, and Docker-related host items. It does **not** require XRT or mlir-aie on the host.

---

**Implementation spec:** [IMPLEMENTATION.md](IMPLEMENTATION.md) — complete handoff for contributors and AI agents.

**Native builds:** For development and unit tests only (`ctest` on CPU). Production NPU inference is Docker-only.
