# GGNPU runtime image — Ubuntu 26.04, XRT runtime, NPU-enabled ggnpu binary.
# Host still provides: amdxdna driver, /dev/accel/accel0, NPU firmware (mounted).
#
# Build:
#   docker build -t ggnpu:latest -f docker/Dockerfile .
#
# Run (see docs/docker.md):
#   docker compose -f docker/docker-compose.yml run --rm ggnpu bench-matmul

ARG UBUNTU_VERSION=26.04

FROM ubuntu:${UBUNTU_VERSION} AS ggnpu-build

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

COPY docker/install-xrt-runtime.sh /tmp/install-xrt-runtime.sh
RUN chmod +x /tmp/install-xrt-runtime.sh && /tmp/install-xrt-runtime.sh

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake make git \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN cmake -S . -B build \
        -DGGNPU_NPU_BACKEND=ON \
        -DGGNPU_TEST_CPU=OFF \
        -DGGNPU_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)"

FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive
LABEL org.opencontainers.image.title="ggnpu"

COPY docker/install-xrt-runtime.sh /tmp/install-xrt-runtime.sh
RUN chmod +x /tmp/install-xrt-runtime.sh && /tmp/install-xrt-runtime.sh \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ggnpu-build /workspace/build/ggnpu /usr/local/bin/ggnpu
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /usr/share/ggnpu/xclbin /root/.cache/ggnpu/xclbin

ENV GGNPU_CACHE_DIR=/root/.cache/ggnpu
VOLUME ["/root/.cache/ggnpu", "/models"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
