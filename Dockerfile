# Multi-stage Dockerfile for arc-llama
#
# Produces an image with:
#   - Pre-built llama-server (SYCL backend, FP16 math, optional AOT)
#   - arc-llama installed from the local source tree
#   - Intel oneAPI runtime libraries for SYCL
#
# Build:
#   docker build -t arc-llama:latest .
#
# Build with AOT (ahead-of-time) device code for your GPU generation —
# eliminates the ~20s SYCL JIT recompile on every cold start, which matters
# doubly on Battlemage where the JIT cache is disabled to dodge a SIGSEGV:
#   docker build --build-arg GGML_SYCL_DEVICE_ARCH=bmg-g21 -t arc-llama:bmg .   # B-series
#   docker build --build-arg GGML_SYCL_DEVICE_ARCH=acm-g10 -t arc-llama:acm .   # A770/750/580
#   (A380/A310: acm-g11. Multiple: GGML_SYCL_DEVICE_ARCH="acm-g10,bmg-g21".)
#
# Run (requires GPU access):
#   docker run --rm -it \
#     --device /dev/dri:/dev/dri \
#     --group-add video \
#     --group-add render \
#     -p 11437:11437 \
#     -v $HOME/models:/models:ro \
#     arc-llama:latest \
#     arc-llama serve
#
# Or with a custom config:
#   docker run --rm -it ... \
#     -v $PWD/config.toml:/root/.config/arc-llama/config.toml:ro \
#     arc-llama:latest \
#     arc-llama serve

ARG ONEAPI_IMAGE=intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04

# ------------------------------------------------------------------------------
# Stage 1: Build llama-server with SYCL
# ------------------------------------------------------------------------------
FROM ${ONEAPI_IMAGE} AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin to a known-good llama.cpp release. The SYCL backend improves fast
# (flash attention, MMVQ/reorder, graph support all post-date old pins) —
# bump this deliberately and retest, don't let it rot.
ARG LLAMA_CPP_REF=b9946
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
    https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp

WORKDIR /tmp/llama.cpp

# FP16 math path — measurably faster than FP32 on Arc, recommended by the
# upstream SYCL docs. Set --build-arg GGML_SYCL_F16=OFF to compare.
ARG GGML_SYCL_F16=ON
# Optional AOT: ocloc -device string (bmg-g21, acm-g10, ...). Empty = JIT.
ARG GGML_SYCL_DEVICE_ARCH=

# Build with SYCL backend using Intel compilers
RUN cmake -B build \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_F16=${GGML_SYCL_F16} \
    ${GGML_SYCL_DEVICE_ARCH:+-DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH}} \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j$(nproc)

# ------------------------------------------------------------------------------
# Stage 2: Runtime image
# ------------------------------------------------------------------------------
FROM ${ONEAPI_IMAGE}

# Install Python and arc-llama dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the built llama-server binary
COPY --from=llama-builder /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
RUN chmod +x /usr/local/bin/llama-server

# Install arc-llama from the local source tree
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir --break-system-packages -e ".[tui]"

# Runtime environment
ENV SYCL_CACHE_PERSISTENT=0
ENV ARC_LLAMA_SERVER=/usr/local/bin/llama-server

# Default port
EXPOSE 11437

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:11437/health || exit 1

# Default entrypoint runs init + serve if no config exists, else just serve
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]
