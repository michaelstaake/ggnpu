FROM ubuntu:24.04

# Build dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    make \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . .

RUN mkdir -p build && cd build && \
    cmake .. -DGGNPU_BUILD_TESTS=ON && \
    make -j2 && \
    ctest --output-on-failure

RUN cp ggnpu /usr/local/bin/

CMD ["ggnpu", "--help"]
