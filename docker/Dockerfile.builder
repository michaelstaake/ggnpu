# GGNPU builder image: mlir-aie + NPU kernel compilation + ggnpu binary.
# Heavy first build (30–90 min, 8–16 GB RAM). Host only needs Docker + NPU driver.
#
# Usage:
#   docker build -t ggnpu-builder -f docker/Dockerfile.builder .
#   docker run --rm -v ggnpu-cache:/root/.cache/ggnpu ggnpu-builder

ARG UBUNTU_VERSION=26.04
FROM ubuntu:${UBUNTU_VERSION}

LABEL org.opencontainers.image.title="ggnpu-builder"
LABEL org.opencontainers.image.description="Build ggnpu and AMD NPU xclbin kernels"

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Stage 1 deps: XRT headers/libs for linking ggnpu
COPY docker/install-xrt-runtime.sh /tmp/install-xrt-runtime.sh
RUN chmod +x /tmp/install-xrt-runtime.sh && /tmp/install-xrt-runtime.sh

# Stage 2: mlir-aie (wheel-based build from upstream script)
COPY docker/install-mlir-aie.sh /tmp/install-mlir-aie.sh
RUN chmod +x /tmp/install-mlir-aie.sh && /tmp/install-mlir-aie.sh

ENV AIE_HOME=/opt/mlir-aie/install
ENV PATH="${AIE_HOME}/bin:/opt/ironenv/bin:${PATH}"

# Build tools for ggnpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    cmake \
    make \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Build ggnpu with NPU backend
RUN cmake -S . -B build \
        -DGGNPU_NPU_BACKEND=ON \
        -DGGNPU_TEST_CPU=OFF \
        -DGGNPU_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)"

# Build NPU kernels for Krackan (npu6). Matmul first; extend as MLIR matures.
# Kernels are cached under /root/.cache/ggnpu/xclbin — mount a volume here.
RUN mkdir -p /root/.cache/ggnpu/xclbin \
    && ./scripts/build-kernels.sh npu6 matmul || { \
        echo "WARNING: matmul xclbin build failed (MLIR templates may need work)"; \
        echo "         Runtime image can still be built; mount prebuilt xclbins."; \
        exit 0; \
    }

# Collect artifacts for multi-stage copy or volume export
RUN mkdir -p /artifacts/bin /artifacts/xclbin \
    && cp build/ggnpu /artifacts/bin/ggnpu \
    && cp -a /root/.cache/ggnpu/xclbin/. /artifacts/xclbin/ 2>/dev/null || true

ENV PATH="/artifacts/bin:${PATH}"
VOLUME ["/root/.cache/ggnpu"]

CMD ["bash", "-lc", "echo 'Artifacts in /artifacts'; ls -la /artifacts/bin /artifacts/xclbin 2>/dev/null; echo 'Cache:'; ls -la /root/.cache/ggnpu/xclbin 2>/dev/null || true"]
