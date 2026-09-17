# syntax=docker/dockerfile:1.7
#
# The Batcher runtime image: Python, the compiled engine, and the extras named by EXTRAS.
#
# Build from the repository root, so the whole source tree is the build context:
#
#   docker build -f packaging/docker/Dockerfile -t batcher .                      # core + cloud
#   docker build -f packaging/docker/Dockerfile --build-arg EXTRAS=cloud,ray -t batcher:ray .
#
# The engine comes from one of two places, chosen by WHEELS:
#
#   source    (default) compile the engine inside the build. Needs nothing on the host but
#             Docker, and is what a local or air-gapped build uses.
#   prebuilt  install the wheels in ./wheelhouse. The release workflow uses this: it has
#             already built and smoke-tested a wheel per architecture, and compiling Rust again
#             under QEMU for linux/arm64 would take the better part of an hour.
#
# Both paths install the engine the same way, so the two images differ only in where the
# wheel was compiled.

ARG PYTHON_VERSION=3.12
ARG WHEELS=source

# --- the engine, compiled here -------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS wheels-source
# A C compiler for the `cc`-built dependencies (mimalloc, zstd, ring). No libclang, cmake, or
# protoc: nothing in the engine's dependency tree runs bindgen or builds protobufs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
    && pip install --no-cache-dir "maturin>=1.7,<2"
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    maturin build --release --out /wheels

# --- the engine, compiled by CI ------------------------------------------------------------
FROM scratch AS wheels-prebuilt
COPY wheelhouse/ /wheels/

FROM wheels-${WHEELS} AS wheels

# --- the runtime ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm
ARG EXTRAS=cloud

LABEL org.opencontainers.image.title="batcher" \
      org.opencontainers.image.description="Batcher: a native, adaptive data engine on Apache Arrow." \
      org.opencontainers.image.source="https://github.com/stephenoffer/batcher" \
      org.opencontainers.image.licenses="Apache-2.0"

# Two steps, because the wheelhouse holds a wheel per platform and pip must not reach PyPI for
# the engine itself. The first installs the engine from the wheelhouse and nothing else, picking
# the wheel that matches this image's architecture; the second resolves its dependencies and
# the extras from PyPI, and finds the engine already satisfied.
RUN --mount=from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --no-deps --find-links /wheels batcher-engine \
    && pip install --no-cache-dir --find-links /wheels "batcher-engine[${EXTRAS}]"

# A non-root user with UID 1000, the UID the official rayproject/ray images run as, so a pod
# spec written for those images needs no change. `/scratch` is the documented spill mount (`BATCHER_SCRATCH_DIR=/scratch`):
# Docker seeds a new named volume from the image's directory, ownership included, so creating it
# here is what makes `-v spill:/scratch` writable by this user instead of root-owned.
RUN useradd --create-home --uid 1000 batcher \
    && mkdir /scratch && chown batcher:batcher /scratch
USER batcher
WORKDIR /home/batcher

CMD ["python"]
